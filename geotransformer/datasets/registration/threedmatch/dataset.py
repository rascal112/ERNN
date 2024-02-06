import os.path as osp
import pickle
import random
from typing import Dict
import os

import numpy as np
import torch
import torch.utils.data
import open3d as o3d
from IPython import embed

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences

def FPFH_r_t(xyz, length, radius_normal, radius_feature, neighbor=40):
    inv_r, equ_r = FPFH(xyz[:length], radius_normal, radius_feature, neighbor)
    inv_t, equ_t = FPFH(xyz[length:], radius_normal, radius_feature, neighbor)
    inv = torch.cat([inv_r, inv_t], dim=0)
    equ = torch.cat([equ_r, equ_t], dim=0)
    return inv, equ

def FPFH(xyz, radius_normal, radius_feature, neighbor=40):
    # xyz = xyz.transpose(1, 2).cpu().numpy()
    # res = np.zeros((xyz.shape[1], 33))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=neighbor))
    res = pcd_fpfh.data
    # res = torch.from_numpy(res).float()
    # res = res.transpose(0, 1)
    normals = np.asarray(pcd.normals)
    # normals = torch.from_numpy(normals).float()
    return res.T, normals


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
        radius=0.1,
        processed_file="/home/ajy/GeoTransforme_update/data/3DMatch/processed/",
        neighbor=50,
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
        
        # ---debug with the partial data
        # self.metadata_list = self.metadata_list[:20]

        # /home/ajy/GeoTransforme_update/data/3DMatch/processed
        # file_path = osp.join(processed_file, subset + str(radius) + '.pth')
        # if os.path.exists(file_path):
        #     # self.feats = np.load(file_path, allow_pickle=True).item()
        #     feats = torch.load(file_path)
        # else:
        #     feats = self.preprocess(file_path, radius, neighbor)
        # self.feats = feats["feats"]
        # self.equ = feats["equ"]
        

    def preprocess(self, file_name, radius, neighbor):
        res = {}
        feats = []
        equ = []
        # k = 0
        for metadata in self.metadata_list:
            # if k == 200:
            #     break
            # k += 1
            ref_points = self._load_point_cloud(metadata['pcd0'])
            src_points = self._load_point_cloud(metadata['pcd1'])
            ref_feats, ref_equ = FPFH(ref_points, radius, radius, neighbor)
            src_feats, src_equ = FPFH(src_points, radius, radius, neighbor)
            feats.append([ref_feats, src_feats])
            equ.append([ref_equ, src_equ])
        # torch.save(feats, file_name)
        res["feats"] = feats
        res["equ"] = equ
        torch.save(res, file_name)
        return res



    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']


        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])
        
        # feats = self.feats[index]
        # equ = self.equ[index]

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        # data_dict['ref_feats'] = feats[0].astype(np.float32)
        # data_dict['src_feats'] = feats[1].astype(np.float32)
        # data_dict['ref_equ'] = equ[0].astype(np.float32)
        # data_dict['src_equ'] = equ[1].astype(np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
