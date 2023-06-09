import argparse
import os
import time

import numpy as np

import skimage.measure as measure
import trimesh

import tqdm

import torch
from torch.utils import data

from utility import same_seed, eval_reconstruct_gt_mesh_p2s
from network import IDRNet
from dataset import ValDataset

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--bd', help='mc grid points range', type=float, default=0.6)
parser.add_argument('--resolution', help='mc grid points resolution', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--thresholds', help='mc iso values', type=list,
                    default=[0, 0.005]
                    )
parser.add_argument('--mesh_path', help='gt mesh for evaluation', type=str,
                    default='mesh/Armadillo.obj')
parser.add_argument('--model_path', help='model path', type=str,
                    default='experiment/famous_noisefree_Armadillo/epoch_35.pth')
parser.add_argument('--shape_num', type=int, default=1)
parser.add_argument('--name', help='experiment name(saving each experiment to "experiment/name" directory)', type=str,
                    default='famous_noisefree_Armadillo')
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--seed', type=int, default=40938661)

if __name__ == '__main__':
    # init
    args = parser.parse_args()
    print(args)
    same_seed(args.seed)
    val_ds = ValDataset(bd=args.bd, resolution=args.resolution)
    val_dl = data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    # network
    print('Network Configure and Loading parameters from {}...'.format(args.model_path))
    net = IDRNet(beta=1000).to(args.device)
    net.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # testing loop, export model and reconstruct result
    print('Inference...')
    csv_lines = ['thresh, cd_p2s']
    st = time.time()
    with torch.no_grad():
        net.eval()
        for shape_ind in range(args.shape_num):
            vox = list()
            for i, q in enumerate(tqdm.tqdm(val_dl)):
                q = q.to(args.device, non_blocking=args.device == 'cuda')
                sdf = net(q)
                vox.append(sdf.detach().cpu().numpy())
            ed = time.time()
            print('inference: {}s'.format(str(ed - st)))
            vox = np.asarray(vox).reshape(-1, 1)
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
                ed = time.time()
                print('Marching: {}s'.format(str(ed - st)))
                rec_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                rec_mesh.export(os.path.join('experiment', args.name, 'thresh_' + str(thresh) + '_rec.obj'))
                gt_mesh = trimesh.load(args.mesh_path, process=False)
                cd = eval_reconstruct_gt_mesh_p2s(rec_mesh, gt_mesh)
                print('thresh', thresh,
                      'cd', cd,
                      )
                csv_lines.append('\n')
                csv_lines.append(str(thresh))
                csv_lines.append(',' + str(cd))
    # save metric
    csv_lines = ''.join(csv_lines)
    print('Saving Metric to CSV File...')
    with open(os.path.join('experiment', args.name, 'rec_compare.csv'), 'w') as text_file:
        text_file.write(csv_lines)
