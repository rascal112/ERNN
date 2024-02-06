from functools import partial

import numpy as np
import torch

from geotransformer.modules.ops import grid_subsample, radius_search, radius_search_graph
from geotransformer.utils.torch import build_dataloader
from IPython import embed
import open3d as o3d
# Stack mode utilities

def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search_graph(cur_points, cur_lengths, radius, neighbor_limits[i])

        # neighbors = radius_search(
        #     cur_points,
        #     cur_points,
        #     cur_lengths,
        #     cur_lengths,
        #     radius,
        #     neighbor_limits[i],
        # )
        neighbors_list.append(neighbors)
        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                1,
            )
            # neighbor_limits[i]
            subsampling_list.append(subsampling.squeeze())

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                2 * radius,
                1,
            )
            # neighbor_limits[i + 1]
            upsampling_list.append(upsampling.squeeze())
        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }

def FPFH_r_t(xyz, length, radius_normal, radius_feature, neighbor=40):
    inv_r, equ_r = FPFH(xyz[:length], radius_normal, radius_feature, neighbor)
    inv_t, equ_t = FPFH(xyz[length:], radius_normal, radius_feature, neighbor)
    inv = torch.cat([inv_r, inv_t], dim=0)
    equ = torch.cat([equ_r, equ_t], dim=0)
    return inv, equ

def FPFH(xyz, radius_normal, radius_feature, neighbor=40):
    # xyz = xyz.transpose(1, 2).cpu().numpy()
    # res = np.zeros((xyz.shape[1], 33))
    xyz = xyz.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=neighbor))
    res = pcd_fpfh.data
    res = torch.from_numpy(res).float()
    res = res.transpose(0, 1)
    normals = np.asarray(pcd.normals)
    normals = torch.from_numpy(normals).float()
    return res, normals

def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    # feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    # equ = torch.cat(collated_dict.pop('ref_equ') + collated_dict.pop('src_equ'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]
    
    # FPFH_feats, normals = FPFH_r_t(points, lengths[0], search_radius, search_radius)
    # collated_dict['features'] = FPFH_feats
    # collated_dict['normals'] = normals
    # collated_dict['features'] = feats
    # collated_dict['equ'] = equ
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        # counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        counts = [torch.bincount(neighbors[1, :]).numpy() for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
