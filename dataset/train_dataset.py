import os

import numpy as np
import scipy.spatial as sp
import torch
import torch.utils.data as data
import pytorch3d.ops as ops
import tqdm
import trimesh
from utility import normalize_mesh_export


class SequentialPointCloudRandomPatchSampler(data.sampler.Sampler):
    def __init__(self, data_source, shape_num=1):
        super().__init__(data_source)
        self.data_source = data_source
        self.shape_num = shape_num

    def __iter__(self):
        rt = torch.randint(0, self.data_source.near_pts.shape[1] - 1, (self.data_source.near_pts.shape[1],))
        iter_order = [(i, rt[j]) for i in range(self.shape_num) for j in range(self.data_source.near_pts.shape[1])]
        return iter(iter_order)

    def __len__(self):
        return self.data_source.near_pts.shape[0] * self.data_source.near_pts.shape[1]
        # return self.shape_num


class PointGenerateDataset(data.Dataset):
    def __init__(self, pts_npy_path, query_num=25, gt_pts_num=20000, batch_size=500, k=1,
                 patch_radius=None, points_per_patch=None, debug=False, device='cuda'):
        super(PointGenerateDataset, self).__init__()
        if not os.path.exists(pts_npy_path):
            print("Path Error")

        # KNN, patch num is k
        if points_per_patch is None:
            points_per_patch = k

        self.query_num = query_num
        self.gt_pts_num = gt_pts_num
        self.batch_size = batch_size
        self.k = k
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch

        shape_num = 1
        self.file = list()

        print(pts_npy_path)
        self.file.append(pts_npy_path.split('/')[-1].split('.')[0])
        print(self.file[0])
        # load pt
        if os.path.splitext(pts_npy_path)[1] == '.npy':
            pnts = np.load(pts_npy_path)[:, :3]
        elif os.path.splitext(pts_npy_path)[1] == '.npz':
            pnts = np.load(pts_npy_path)['points']
        else:
            pnts = trimesh.load(pts_npy_path, process=False).vertices
        # normalize pt
        pnts, scale_inv, trans_inv = normalize_mesh_export(trimesh.PointCloud(vertices=pnts))
        pnts = np.array(pnts.vertices)
        self.input_scale_inv = scale_inv
        self.input_trans_inv = trans_inv
        # sample ground truth
        print(pnts.shape)
        if self.gt_pts_num == -1 and pnts.shape[0] > 20000:
            self.gt_pts_num = pnts.reshape(-1, 3).shape[0] // 100 * 100
        elif self.gt_pts_num == -1 and pnts.shape[0] < 20000:
            self.gt_pts_num = 20000
        print("gt_pts_num:", self.gt_pts_num)
        if query_num is None:
            self.query_num = 1e6 // self.gt_pts_num
        else:
            self.query_num = query_num
        print('query_num per input point:', self.query_num)
        pnts = pnts[self._patch_sampling(pnts, self.gt_pts_num)]
        pnts_pt = torch.from_numpy(pnts).float()
        self.pnts = pnts
        self.pnts_pt = pnts_pt
        kd_tree = sp.KDTree(pnts_pt.reshape(pnts_pt.shape[0], pnts_pt.shape[1]).numpy())
        # query each point for sigma^2
        dist, _ = kd_tree.query(pnts_pt.reshape(1, pnts_pt.shape[0], pnts_pt.shape[1]).numpy(), k=50, workers=-1)
        dist = torch.from_numpy(dist.squeeze())
        sigmas = dist[:, -1].unsqueeze(1)
        # sample query point
        query_point = pnts_pt + sigmas * torch.normal(mean=0.0, std=1.0,
                                                      size=(self.query_num, pnts_pt.shape[0], pnts_pt.shape[1])
                                                      )
        query_point = query_point.reshape(1, -1, 3).to(device).float()
        rand_idx = torch.randperm(query_point.shape[1])
        query_point = query_point[:, rand_idx, :]
        pnts_pt = pnts_pt.reshape(1, -1, 3).to(device).float()
        near_pts = None
        idx = None

        _, idx, proxy = ops.knn_points(query_point, pnts_pt, K=1, return_sorted=False, return_nn=True)
        proxy = proxy.squeeze(2)
        if patch_radius is None:
            _, idx, nn = ops.knn_points(query_point, pnts_pt, K=k, return_sorted=False, return_nn=True)
            # _, idx, nn = ops.knn_points(proxy, pnts_pt, K=k, return_sorted=False, return_nn=True)
            near_pts = nn.reshape(shape_num, -1, batch_size, k, 3).cpu().numpy()
            idx = idx.reshape(shape_num, -1, batch_size, k).cpu().numpy()
            querys = query_point.reshape(shape_num, -1, batch_size, 3).cpu().numpy()
            proxy = proxy.reshape(shape_num, -1, batch_size, 3).cpu().numpy()
        else:
            self.bbdiag_shape = np.linalg.norm(pnts.max(0) - pnts.min(0))
            self.patch_radius_absolute = self.bbdiag_shape * self.patch_radius
            # _, idx, nn = ops.ball_query(proxy, pnts_pt, radius=self.patch_radius_absolute, K=points_per_patch,
            #                             return_nn=False)
            _, idx, nn = ops.ball_query(query_point, pnts_pt, radius=self.patch_radius_absolute, K=points_per_patch,
                                        return_nn=False)
            # delete the query point does not have nn
            query_idx = torch.sum(idx, dim=-1) != (idx.shape[-1] * -1)
            idx = idx[query_idx]
            query_point = query_point[query_idx]

            # find the idx has -1, and randomly padding that
            padding_idx = torch.argwhere(torch.sum(idx == -1, dim=-1) != 0).squeeze()
            if padding_idx.shape[0] != 0:
                non_padding_idx = torch.argwhere(torch.sum(idx == -1, dim=-1) == 0).squeeze()
                new_idx = [None] * padding_idx.shape[0]

                for i, v in enumerate(tqdm.tqdm(padding_idx)):
                    vv = idx[v][idx[v] != -1]
                    new_vv = vv[torch.randint(vv.shape[0], (points_per_patch,))]
                    new_idx[i] = new_vv.long().cpu().numpy()
                # new_idx.append(new_vv.long().cpu().numpy())
                # def func(i, v):
                #     vv = idx[v][idx[v] != -1]
                #     new_vv = vv[torch.randint(vv.shape[0], (points_per_patch,))]
                #     new_idx[i] = new_vv.long().cpu().numpy()
                # Parallel(n_jobs=4)(delayed(func)(i, v) for i, v in enumerate(tqdm.tqdm(padding_idx)))
                new_idx = np.concatenate(new_idx, axis=0).reshape(-1, points_per_patch)
                if len(idx[non_padding_idx].shape) == 1:
                    target = idx[non_padding_idx].reshape(1, -1).cpu().numpy()
                else:
                    target = idx[non_padding_idx].cpu().numpy()
                new_idx = np.concatenate([target, new_idx], axis=0)
                if len(query_point[non_padding_idx].shape) == 1:
                    target = query_point[non_padding_idx].reshape(-1, 3).cpu().numpy()
                else:
                    target = query_point[non_padding_idx].cpu().numpy()
                query_point = np.concatenate(
                    (target, query_point[padding_idx].cpu().numpy()), axis=0)
                final_padding_id = self._patch_sampling(new_idx, self.gt_pts_num * self.query_num)
                new_idx = new_idx[final_padding_id]
                querys = query_point[final_padding_id]
                near_pts = pnts[new_idx].reshape(shape_num, -1, batch_size, points_per_patch, 3)
                querys = querys.reshape(shape_num, -1, batch_size, 3)
                proxy = proxy.reshape(shape_num, -1, batch_size, 3).cpu().numpy()
            else:
                # pass
                # print(query_point.shape, idx.shape)
                align_idx = np.random.choice(range(idx.shape[0]), idx.shape[0] // 100 * 100, replace=False)
                idx = idx[align_idx]
                query_point = query_point[align_idx]
                idx = idx.reshape(shape_num, -1, batch_size, points_per_patch).cpu().numpy()
                near_pts = pnts[idx].reshape(shape_num, -1, batch_size, points_per_patch, 3)
                querys = query_point.reshape(shape_num, -1, batch_size, 3).cpu().numpy()
                proxy = proxy.reshape(shape_num, -1, batch_size, 3).cpu().numpy()

        # scipy kdtree
        # query_point = query_point.reshape(-1, batch_size, 3)
        # near_pts = list()
        # querys = list()
        # for i in tqdm.trange(query_point.shape[0]):
        #     # get the most nearest point from gt_pts for query point
        #     if patch_radius is None:
        #         _, indx = kd_tree.query(
        #             query_point[i].reshape(1, query_point[i].shape[0], query_point[i].shape[1]).numpy(), k=k,
        #             workers=-1)
        #     else:
        #         # compute absolute patch radius
        #         self.bbdiag_shape = np.linalg.norm(pnts.max(0) - pnts.min(0))
        #         self.patch_radius_absolute = self.bbdiag_shape * self.patch_radius
        #         temp_indx = kd_tree.query_ball_point(
        #             x=query_point[i].reshape(query_point[i].shape[0], query_point[i].shape[1]).numpy(),
        #             r=self.patch_radius_absolute, workers=-1, return_sorted=True)
        #         q_ind = np.full(shape=(query_point[i].shape[0],), fill_value=True, dtype=np.bool)
        #         indx = []
        #         for k, l in enumerate(temp_indx):
        #             if len(l) != 0:  # has neighbor, sample max_num
        #                 pad_indx = np.asarray(l)[
        #                     self._patch_sampling(np.asarray(l), self.points_per_patch)]  # padding
        #                 indx.append(pad_indx)
        #             else:  # no neighbor,
        #                 q_ind[k] = False
        #         indx = np.array(indx, dtype=np.long)
        #     indx = torch.from_numpy(indx.squeeze())
        #     # add for each batch
        #     if self.patch_radius is None:
        #         near_pts.append(pnts_pt[indx].cpu().numpy())
        #         querys.append(query_point[i].cpu().numpy())
        #     else:
        #         near_pts.append(pnts_pt[indx].cpu().numpy())
        #         querys.append(query_point[i][q_ind].cpu().numpy())
        # if self.patch_radius is None:
        # q = trimesh.PointCloud(vertices=querys)
        # q.export(os.path.join('experiment', 'query.ply'))
        # n = trimesh.PointCloud(vertices=near_pts[0, :, :].reshape(-1, 3))
        # n.export(os.path.join('experiment', 'near.ply'))
        # near_pts = np.asarray(near_pts).reshape(shape_num, -1, batch_size, k, 3)  # SHAPE_NUM, 100, 5000, k, 3
        # querys = np.asarray(querys).reshape(shape_num, -1, batch_size, 3)

        # print(near_pts.shape, querys.shape)
        # np.save(os.path.join('experiment', 'knn_near_pts.npy'), near_pts)
        # np.save(os.path.join('experiment', 'knn_querys.npy'), querys)
        # sys.exit()
        # else:
        #     near_pts = np.concatenate(near_pts)
        #     querys = np.concatenate(querys)
        # q = trimesh.PointCloud(vertices=querys)
        # q.export(os.path.join('experiment', 'query.ply'))
        # n = trimesh.PointCloud(vertices=near_pts[0, :, :].reshape(-1, 3))
        # n.export(os.path.join('experiment', 'near.ply'))
        # ind = self._patch_sampling(querys, querys.shape[0] // batch_size * batch_size)  # get even query point
        # near_pts = near_pts[ind].reshape(shape_num, -1, batch_size, self.points_per_patch, 3)
        # querys = querys[ind].reshape(shape_num, -1, batch_size, 3)

        self.shape_num = shape_num
        self.idx = idx
        self.near_pts = near_pts
        self.query = querys
        self.proxy = proxy

    def _patch_sampling(self, patch_pts, target_num):
        if patch_pts.shape[0] > target_num:
            sample_index = np.random.choice(range(patch_pts.shape[0]), target_num, replace=False)
        elif patch_pts.shape[0] < target_num:
            sample_index = np.random.choice(range(patch_pts.shape[0]), target_num - patch_pts.shape[0], replace=True)
            sample_index = np.concatenate((sample_index, range(patch_pts.shape[0])))
        else:
            sample_index = np.arange(target_num)
        return sample_index

    def __len__(self):
        return self.near_pts.shape[0] * self.near_pts.shape[1]

    def __getitem__(self, index):
        # print(index)
        patch = self.near_pts[index]  # (batch, k, 3)
        query = self.query[index]  # (batch, 3)
        proxy = self.proxy[index]  # (batch, 3)
        # sample_ind = self._patch_sampling(patch, self.points_per_patch)
        # patch = patch[sample_ind]
        # normalize patch
        bd_diag = np.linalg.norm(patch.max(1) - patch.min(1), axis=1, keepdims=True)
        # patch = (patch - query.reshape(query.shape[0], 1, query.shape[1])) / (np.expand_dims(bd_diag, axis=2) + 1e-12)
        # compute support_radius for gauss
        support_radius = np.sqrt(bd_diag / self.points_per_patch)

        return torch.from_numpy(patch).float(), \
            torch.from_numpy(query).float(), \
            torch.from_numpy(support_radius).float(), \
            torch.from_numpy(proxy).float()
