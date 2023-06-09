import numpy as np
import trimesh

import torch
import torch.nn.functional as F


def get_query_point(bd=0.55, resolution=128):
    shape = (resolution, resolution, resolution)
    vxs = torch.arange(-bd, bd, bd * 2 / resolution)
    vys = torch.arange(-bd, bd, bd * 2 / resolution)
    vzs = torch.arange(-bd, bd, bd * 2 / resolution)
    pxs = vxs.view(-1, 1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pys = vys.view(1, -1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pzs = vzs.view(1, 1, -1).expand(*shape).contiguous().view(resolution ** 3)
    p = torch.stack([pxs, pys, pzs], dim=1).reshape(resolution, resolution ** 2, 3)
    return p


def gradient(points, net, normalize=True):
    points.requires_grad_(True)
    sdf_value = net(points)
    grad = torch.autograd.grad(sdf_value, [points], [
        torch.ones_like(sdf_value)], create_graph=True)[0]
    if normalize:
        grad = F.normalize(grad, dim=2)
    return grad


def normalize_mesh_export(mesh, file_out=None, get_scale=True):
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation_matrix = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation_matrix)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_tra = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_tra)
    if file_out is not None:
        mesh.export(file_out)
    if get_scale:
        scale_trafo_inv = np.linalg.inv(scale_tra)
        translation_inv = np.linalg.inv(translation_matrix)
        return mesh, scale_trafo_inv, translation_inv
    return mesh


def revert_normalization(mesh_rec, mesh_gt):
    '''
    recover the scale of mesh_rec based on mesh_gt
    Args:
        mesh_rec:
        mesh_gt:

    Returns:

    '''
    bounds = mesh_gt.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh_gt.bounds[0] + mesh_gt.bounds[1]) * 0.5
    translation_inv = trimesh.transformations.translation_matrix(direction=translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo_inv = trimesh.transformations.scale_matrix(factor=1.0 / scale)

    mesh_rec.apply_transform(scale_trafo_inv)
    mesh_rec.apply_transform(translation_inv)
    return mesh_rec


def eval_reconstruct_gt_mesh_p2s(rec_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, sample_num=10000):
    def _chamfer_distance_single_file(rec, gt, samples_per_model, num_processes=-1):
        # http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf

        import trimesh
        import trimesh.sample
        import sys
        import scipy.spatial as spatial

        def sample_mesh(mesh, num_samples):
            samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
            return samples

        new_mesh_samples = sample_mesh(rec, samples_per_model)
        ref_mesh_samples = sample_mesh(gt, samples_per_model)

        leaf_size = 100
        sys.setrecursionlimit(int(max(1000, round(new_mesh_samples.shape[0] / leaf_size))))
        kdtree_new_mesh_samples = spatial.KDTree(new_mesh_samples, leaf_size)
        kdtree_ref_mesh_samples = spatial.KDTree(ref_mesh_samples, leaf_size)

        ref_new_dist, corr_new_ids = kdtree_new_mesh_samples.query(ref_mesh_samples, 1, workers=num_processes)
        new_ref_dist, corr_ref_ids = kdtree_ref_mesh_samples.query(new_mesh_samples, 1, workers=num_processes)

        ref_new_dist_sum = np.sum(ref_new_dist)
        new_ref_dist_sum = np.sum(new_ref_dist)
        chamfer_dist = ref_new_dist_sum + new_ref_dist_sum

        return chamfer_dist

    rec_mesh = normalize_mesh_export(rec_mesh, get_scale=False)
    gt_mesh = normalize_mesh_export(gt_mesh, get_scale=False)
    chamfer_dist = _chamfer_distance_single_file(rec_mesh, gt_mesh, sample_num)
    return chamfer_dist
