import argparse
import os
import time

import tqdm
import numpy as np
import skimage.measure as measure
import trimesh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import Adam

from utility import same_seed, RunningAverage, save_dict_to_json, normalize_mesh_export
from network import IDRNet
from dataset import ValDataset, SequentialPointCloudRandomPatchSampler, PointGenerateDataset

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--bd', type=float, default=0.6, help='shape bounds')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--resolution', help='mc grid points resolution', type=int, default=256)
parser.add_argument('--thresholds', help='mc iso values', type=list,
                    default=[0, 0.005]
                    )
parser.add_argument('--sub_batch_size', type=int, default=100, help='real_batch_size')
parser.add_argument('--sigma_r', type=float, default=0.3)
parser.add_argument('--multi_support_radius', type=float, default=1)
parser.add_argument('--multires', type=int, default=0)
parser.add_argument('--first', type=bool, default=False)
parser.add_argument('--gt_pts_num', type=int, default=-1, help='the number of sampled points, -1 means all points')
parser.add_argument('--query_num', type=int, default=25)
parser.add_argument('--k', type=int, default=-100, help='knn, k')
parser.add_argument('--patch_radius', type=float, default=0.01, help='ball_query (rnn), radius')
parser.add_argument('--points_per_patch_max', type=int, default=50, help='ball_query (frnn), max_num')
parser.add_argument('--pts_dir', type=str, default='data/ori_bunny.xyz.npy')
parser.add_argument('--beta', type=int, default=1000,
                    help='beta for the softplus (sometimes need to be 0 avoiding over-smooth)')
parser.add_argument('--skip_in', type=int, default=4)
parser.add_argument('--loss', type=str, default='implicit', choices=['projection', 'implicit'],
                    help='MLS Projection or implicit MLS')

parser.add_argument('--name', type=str, default='base')
parser.add_argument('--summary_freq', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--seed', type=int, default=40938661)

if __name__ == '__main__':
    # init
    args = parser.parse_args()
    args.name = args.name + '_' + str(time.time())
    print(args)
    same_seed(args.seed)
    # dataset
    print('Load dataset')
    if args.k > 0:
        print('Nearest Type: KNN')
        train_ds = PointGenerateDataset(args.pts_dir, gt_pts_num=args.gt_pts_num, batch_size=args.sub_batch_size,
                                        query_num=args.query_num, k=args.k, device=args.device)
    else:
        print('Nearest Type: RNN')
        train_ds = PointGenerateDataset(args.pts_dir, gt_pts_num=args.gt_pts_num, batch_size=args.sub_batch_size,
                                        query_num=args.query_num, patch_radius=args.patch_radius,
                                        points_per_patch=args.points_per_patch_max, device=args.device)
    train_sampler = SequentialPointCloudRandomPatchSampler(data_source=train_ds, shape_num=train_ds.shape_num)
    train_dl = data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                               sampler=train_sampler, pin_memory=False, persistent_workers=True
                               )
    val_ds = ValDataset(bd=args.bd, resolution=args.resolution)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
    args.shape_num = train_ds.shape_num
    # network
    print('Network Configure')
    net = IDRNet(skip_in=(args.skip_in,), multires=args.multires, beta=args.beta, first=args.first).to(args.device)
    criterion = nn.MSELoss().to(args.device)
    optimizer = Adam(net.parameters(), lr=args.lr)

    # save experiment json to folder experiment/args.name
    if not os.path.exists(os.path.join('experiment', args.name)):
        os.makedirs(os.path.join('experiment', args.name))
    save_dict_to_json(vars(args), os.path.join('experiment', args.name, 'args.json'))
    for epoch in tqdm.trange(1, args.epochs):
        net.train()
        loss_sum = RunningAverage()
        IGR_loss_sum = RunningAverage()
        iter_num = 0
        with tqdm.tqdm(total=len(train_dl), desc='train loop') as tq:
            for i, (t, q, lambda_p, proxy) in enumerate(train_dl):
                t = t.to(args.device, non_blocking=True)  # (1, 5000, k, 3)
                q = q.to(args.device, non_blocking=True)  # (1, 5000, 3)
                proxy = proxy.to(args.device, non_blocking=True)  # (1, 5000, 3)
                lambda_p = lambda_p.to(args.device, non_blocking=True)  # (1, 5000, 1)
                lambda_p *= args.multi_support_radius
                t.requires_grad = True
                q.requires_grad = True

                sdf = net(q)
                sdf.sum().backward(retain_graph=True)
                q_grad = q.grad.detach()  # (1, 500, 3)
                q_grad = F.normalize(q_grad, dim=2)

                # compute neighbor sdf, normal
                neigh_sdf = net(t.reshape(t.shape[0], -1, 3))
                neigh_sdf.sum().backward(retain_graph=True)
                neigh_grad = t.grad.detach()
                neigh_grad = F.normalize(neigh_grad, dim=-1)

                # IMLS
                x = q  # (1, 5000, 3)
                dist = torch.linalg.norm(x.unsqueeze(2) - t, dim=-1) + 1e-8
                dist_sq = dist ** 2

                # Gaussian kernel
                weight_theta = torch.exp(-dist_sq / lambda_p ** 2)  # (1, 5000, k)
                ## RIMLS 's Style Gaussian Kernel for Normal
                normal_proj_dist = torch.norm(q_grad.unsqueeze(2) - neigh_grad, dim=-1) ** 2
                weight_phi = torch.exp(-normal_proj_dist / args.sigma_r ** 2)

                # bilateral normal smooth
                weight = weight_theta * weight_phi + 1e-12
                weight = weight / weight.sum(2, keepdim=True)  # (1, 5000, k)

                loss = None
                if args.loss == 'projection':
                    new_grad = (weight.unsqueeze(-1) * q_grad.unsqueeze(2)).sum(2)
                    pro_p = q - new_grad * sdf
                    loss = criterion(pro_p, proxy)
                elif args.loss == 'implicit':
                    project_dist = ((x.unsqueeze(2) - t) * neigh_grad).sum(3)
                    imls_dist = (project_dist * weight).sum(2, keepdim=True)
                    loss = criterion(imls_dist, sdf)

                # q_moved = q - q_grad * sdf
                # q_moved_sdf = net(q_moved)
                # q_moved_sdf.sum().backward(retain_graph=True)
                # q_moved_grad = q_moved.grad.detach()  # (1, 500, 3)
                # q_moved_grad = F.normalize(q_moved_grad, dim=2)
                # consis_constraint = 1 - F.cosine_similarity(q_moved_grad, q_grad, dim=-1)
                # weight_moved = torch.exp(-10.0 * torch.abs(sdf)).reshape(-1, consis_constraint.shape[-1])
                # consis_constraint = weight_moved * consis_constraint
                # loss = loss + 0.01 * consis_constraint.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update loss
                loss_sum.update(loss.detach().cpu().item())
                del loss, q_grad, neigh_sdf, neigh_grad
                tq.set_postfix(loss='{:05.4f}'.format(loss_sum()))
                tq.update()
        print('epoch: {}, loss: {}'.format(epoch, loss_sum()))

        # save temp results
        if epoch % args.summary_freq == 0:
            # save model
            torch.save(net.state_dict(), os.path.join('experiment', args.name, 'epoch_{}.pth'.format(epoch)))
            net.eval()
            with torch.no_grad():
                net.eval()
                for shape_ind in range(args.shape_num):
                    vox = list()
                    for i, q in enumerate(val_dl):
                        q = q.to(args.device, non_blocking=True)
                        sdf = net(q)
                        vox.append(sdf.detach().cpu().numpy())
                    del q, sdf
                    # export each mesh
                    vox = np.asarray(vox).reshape((args.resolution, args.resolution, args.resolution))
                    vox_max = np.max(vox.reshape((-1)))
                    vox_min = np.min(vox.reshape((-1)))
                    for thresh_ind, thresh in enumerate(args.thresholds):
                        # sdf in data range?
                        if thresh < vox.min() or thresh > vox.max():
                            continue
                        if np.sum(vox > 0.0) < np.sum(vox < 0.0):
                            thresh = -thresh
                        vertices, faces, _, _ = measure.marching_cubes(vox, thresh)
                        mc_flag = True
                        # save query result
                        mesh = normalize_mesh_export(trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                                                     get_scale=False)
                        # recover scale based on input (may not align with gt well)
                        mesh.apply_transform(train_ds.input_scale_inv)
                        mesh.apply_transform(train_ds.input_trans_inv)
                        mesh.export(os.path.join('experiment', args.name, str(epoch) + '_' + str(thresh) + '.obj'))
