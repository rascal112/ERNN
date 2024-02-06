# from networkx import sigma
import torch
import torch.nn as nn
from IPython import embed
import e3nn
import e3nn.o3 as o3
from torch_geometric.nn import radius_graph
from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample, invencoder

import numpy as np
import open3d #as o3d

# version 1
class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm, e_c):
        super(KPConvFPN, self).__init__()
        # self.FPFH = FPFH(init_radius, init_radius / 2)
        # ajy do
        self.init_sigma_list = [init_sigma, init_sigma * 2, init_sigma * 4, init_sigma * 8] 
        self.init_radius_list = [init_radius, init_radius * 2, init_radius * 4, init_radius * 8] 
        # e_c = 4 #[2, 4]
        self.e_c = e_c
        self.padding = nn.Parameter(data=torch.randn(1, 1, e_c))
        self.encoder1_1 = invencoder(1, init_dim, e_c, e_c, 1, 1, radius=self.init_radius_list[0])
        self.encoder1_2 = invencoder(init_dim, init_dim * 2, e_c, 2 * e_c, 1, 1, radius=self.init_radius_list[0])
        
        self.encoder2_1 = invencoder(init_dim * 2, init_dim * 2, 2 * e_c, 2 * e_c, 1, 1, radius=self.init_radius_list[1])
        self.encoder2_2 = invencoder(init_dim * 2, init_dim * 4, 2 * e_c, 4 * e_c, 1, 1, shalldow=False, radius=self.init_radius_list[1])
        
        self.encoder3_1 = invencoder(init_dim * 4, init_dim * 4, 4 * e_c, 4 * e_c, 1, 1, radius=self.init_radius_list[2])
        self.encoder3_2 = invencoder(init_dim * 4, init_dim * 8, 4 * e_c, 8 * e_c, 1, 1, shalldow=False, radius=self.init_radius_list[2])
        
        self.encoder4_1 = invencoder(init_dim * 8, init_dim * 8, 8 * e_c, 8 * e_c, 1, 1, radius=self.init_radius_list[3])
        self.encoder4_2 = invencoder(init_dim * 8, init_dim * 16, 8 * e_c, 16 * e_c, 1, 2, shalldow=False, radius=self.init_radius_list[3])

        self.decoder3 = decoder(init_dim * 16, init_dim * 8, 16 * e_c, 8 * e_c, init_dim * 8, 1, 1, group_norm)
        self.decoder2 = decoder(init_dim * 8, init_dim * 4, 8 * e_c, 4 * e_c, output_dim, 1, 1)

        # self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        # self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)
        # self.encoder4_2.debug = True

        init_sigma = init_sigma * 0.5
        # self.normal = normal(init_radius * 0.75)
        # self.com1 = combination(e_c, e_c, init_sigma, preprocess=True)
        # self.com2 = combination(2 * e_c, 2 * e_c, init_sigma * 2)
        # self.com3 = combination(4 * e_c, 4 * e_c, init_sigma * 4)
        # self.com4 = combination(8 * e_c, 8 * e_c, init_sigma * 8)
        self.com2 = combination_normal(2 * e_c)
        self.com3 = combination_normal(4 * e_c)
        self.com4 = combination_normal(8 * e_c)

    def normal_function(self, point_list, length, radius):
        equ_r = self.normal(point_list[:length], radius)
        equ_s = self.normal(point_list[length:], radius)
        return torch.cat([equ_r, equ_s], dim=0)
    
    def test(self, data_dict, points_list, neighbors_list, subsampling_list):
        test_feat = []
        test_equ = []

        length = data_dict['lengths'][0][0].item()
        equ_r = self.normal(points_list[0][:length], self.init_sigma_list[0])
        equ_s = self.normal(points_list[0][length:], self.init_sigma_list[0])
        equ = torch.cat([equ_r, equ_s], dim=0)
        # --------------------------------------------------------------------------
        
        # original
        equ = equ.unsqueeze(-1)
        equ = equ.expand(-1, -1, self.e_c)
        # padding = torch.zeros(equ.size(0), 1, self.e_c, device=equ.device)
        padding = self.padding.expand(equ.size(0), -1, -1)
        equ = torch.cat([padding, equ], dim=1)
        #----------------------------------------------------------------

        feats1 = torch.zeros(len(points_list[0]), 1).to(device=equ.device)
        # equ1, weights1, dist1, vec1 = self.com1(equ, points_list[0], neighbors_list[0])
        equ1 = equ
        cutoff_list1 = self.cutoff_function(points_list[0], feats1, neighbors_list[0], init=True)
        feats1, equ1 = self.encoder1_1(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1)
        feats1, equ1 = self.encoder1_2(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1)

        feats2, equ2 = self.encoder2_1(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1, subsampling_list[0])
        cutoff_list2 = self.cutoff_function(points_list[1], feats2, neighbors_list[1])
        equ2 = self.com2(equ2, self.normal_function(points_list[1], data_dict['lengths'][1][0].item(), self.init_sigma_list[1]))
        feats2, equ2 = self.encoder2_2(feats2, equ2, points_list[1], neighbors_list[1], cutoff_list2)

        feats3, equ3 = self.encoder3_1(feats2, equ2, points_list[1], neighbors_list[1], cutoff_list2, subsampling_list[1])
        cutoff_list3 = self.cutoff_function(points_list[2], feats3, neighbors_list[2])
        equ3 = self.com3(equ3, self.normal_function(points_list[2], data_dict['lengths'][2][0].item(), self.init_sigma_list[2]))
        feats3, equ3 = self.encoder3_2(feats3, equ3, points_list[2], neighbors_list[2],cutoff_list3)
        test_feat.append(feats3)
        test_equ.append(equ3)

        return feats3, equ3, test_feat, test_equ
    
    def normal(self, xyz, color=None):
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        # if color is not None:
        #     if len(color) == 3:
        #         color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        #     pcd.colors = open3d.utility.Vector3dVector(color)
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location()
        src_noms = np.array(pcd.normals)
        # src_kpt = np.concatenate([src_kpt, src_noms], axis=-1)
        return torch.from_numpy(src_noms).float().cuda()
    
    def cutoff_function(self, points, feats, edge_index, num_resolutions=2, max_num_neighbors=40, init=False):
        cutoff_id = []
        x1 = points[edge_index[0, :]]
        x2 = points[edge_index[1, :]]
        edge_distance = self.distance(x1, x2)
        # if not init:
        #     maxn = feats[edge_index[0, :]].max(1)[0]
        #     edge_distance += maxn
        edge_rank = self.rank_edge_distances(edge_distance, edge_index, max_num_neighbors)
        edge_index, edge_distance, cutoff_index = self.get_cutoff_index(edge_rank, edge_index, edge_distance, max_num_neighbors, num_resolutions)
        for j in range(num_resolutions):
            cutoff_id.append(edge_index[:, cutoff_index[j]:cutoff_index[j+1]])
        return cutoff_id

    def forward(self, feats, equ, data_dict):
        feats_list = []
        equ_list = []

        # timer = Timer(torch.device('cuda:0'))
        # timer.start('a', history=True)

        points_list = data_dict['points'] # a list, each of component: [N, 3], N is not specified
        neighbors_list = data_dict['neighbors'] # a list, each of component: [N, 38 or 36] 
        subsampling_list = data_dict['subsampling'] # a list, each of component: [N, 38 or 36], the length = len(points_list), each of component: [N, 38 or 36]. N in i-th = N in (i+1)-th in points_list or neighbors_list
        upsampling_list = data_dict['upsampling']#  a list, each of component: [N, 38 or 36], the length = len(points_list), each of component: [N, 38 or 36]. N in i-th = N in i-th in points_list or neighbors_list

        # debug 
        # a= []
        # # R = torch.Tensor(data=[[-0.7887, 0.5774, 0.2113], [0.5774,0.5774, 0.5774], [0.2113, 0.5774, -0.7887]]).cuda()
        # R = torch.from_numpy(self.random_rotation_matrix()).cuda()
        # R = R.to(dtype=points_list[0].dtype)
        # a.append((R.unsqueeze(0).expand(points_list[0].shape[0], -1, -1) @ points_list[0].unsqueeze(-1)).squeeze(-1))
        # a.append((R.unsqueeze(0).expand(points_list[1].shape[0], -1, -1) @ points_list[1].unsqueeze(-1)).squeeze(-1))
        # a.append((R.unsqueeze(0).expand(points_list[2].shape[0], -1, -1) @ points_list[2].unsqueeze(-1)).squeeze(-1))
        # a.append((R.unsqueeze(0).expand(points_list[3].shape[0], -1, -1) @ points_list[3].unsqueeze(-1)).squeeze(-1))
        # feats3, equ3, test_feat, test_equ = self.test(data_dict, points_list, neighbors_list, subsampling_list)
        # feats31, equ31, test_feat1, test_equ1 = self.test(data_dict, a, neighbors_list, subsampling_list)

        # embed()
        
        length = data_dict['lengths'][0][0].item()
        equ_r = self.normal(points_list[0][:length], self.init_sigma_list[0])
        equ_s = self.normal(points_list[0][length:], self.init_sigma_list[0])
        equ = torch.cat([equ_r, equ_s], dim=0)
        # --------------------------------------------------------------------------
        
        # original
        equ = equ.unsqueeze(-1)
        equ = equ.expand(-1, -1, self.e_c)
        # padding = torch.zeros(equ.size(0), 1, self.e_c, device=equ.device)
        padding = self.padding.expand(equ.size(0), -1, -1)
        equ = torch.cat([padding, equ], dim=1)
        #-------------------------------------------------------------------

        #-------------------------------------------------------------------

        feats1 = torch.zeros(len(points_list[0]), 1).to(device=equ.device)
        # equ1, weights1, dist1, vec1 = self.com1(equ, points_list[0], neighbors_list[0])
        equ1 = equ
        cutoff_list1 = self.cutoff_function(points_list[0], feats1, neighbors_list[0], init=True)
        feats1, equ1 = self.encoder1_1(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1)
        feats1, equ1 = self.encoder1_2(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1)

        feats2, equ2 = self.encoder2_1(feats1, equ1, points_list[0], neighbors_list[0], cutoff_list1, subsampling_list[0])
        cutoff_list2 = self.cutoff_function(points_list[1], feats2, neighbors_list[1])
        equ2 = self.com2(equ2, self.normal_function(points_list[1], data_dict['lengths'][1][0].item(), self.init_sigma_list[1]))
        feats2, equ2 = self.encoder2_2(feats2, equ2, points_list[1], neighbors_list[1], cutoff_list2)

        feats3, equ3 = self.encoder3_1(feats2, equ2, points_list[1], neighbors_list[1], cutoff_list2, subsampling_list[1])
        cutoff_list3 = self.cutoff_function(points_list[2], feats3, neighbors_list[2])
        equ3 = self.com3(equ3, self.normal_function(points_list[2], data_dict['lengths'][2][0].item(), self.init_sigma_list[2]))
        feats3, equ3 = self.encoder3_2(feats3, equ3, points_list[2], neighbors_list[2],cutoff_list3)

        feats4, equ4 = self.encoder4_1(feats3, equ3, points_list[2], neighbors_list[2], cutoff_list3, subsampling_list[2])
        cutoff_list4 = self.cutoff_function(points_list[3], feats4, neighbors_list[3])
        equ4 = self.com4(equ4, self.normal_function(points_list[3], data_dict['lengths'][3][0].item(), self.init_sigma_list[3]))
        feats4, equ4 = self.encoder4_2(feats4, equ4, points_list[3], neighbors_list[3], cutoff_list4)

        latent_s4 = feats4
        feats_list.append(feats4)

        latent_s3 = self.decoder3(latent_s4, feats3, equ4, equ3, upsampling_list[2])
        feats_list.append(latent_s3)

        latent_s2 = self.decoder2(latent_s3, feats2, equ3, equ2, upsampling_list[1])
        feats_list.append(latent_s2)

        equ_list.append(equ4)

        # latent_s4 = feats4
        # feats_list.append(feats4)

        # latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        # latent_s3 = torch.cat([latent_s3, feats3], dim=1)
        # latent_s3 = self.decoder3(latent_s3)
        # feats_list.append(latent_s3)

        # latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        # latent_s2 = torch.cat([latent_s2, feats2], dim=1)
        # latent_s2 = self.decoder2(latent_s2)
        # feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list, equ_list
    
    def random_rotation_matrix(self):
        """
        Generate a random 3x3 rotation matrix.
        """
        # Generate a random rotation axis (unit vector)
        axis = np.random.rand(3)
        axis /= np.linalg.norm(axis)

        # Generate a random rotation angle
        angle = np.random.uniform(0, 2 * np.pi)

        # Create the rotation matrix using the axis-angle representation
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos_theta = 1 - cos_theta

        rotation_matrix = np.array([
            [cos_theta + axis[0]**2 * one_minus_cos_theta, axis[0] * axis[1] * one_minus_cos_theta - axis[2] * sin_theta, axis[0] * axis[2] * one_minus_cos_theta + axis[1] * sin_theta],
            [axis[1] * axis[0] * one_minus_cos_theta + axis[2] * sin_theta, cos_theta + axis[1]**2 * one_minus_cos_theta, axis[1] * axis[2] * one_minus_cos_theta - axis[0] * sin_theta],
            [axis[2] * axis[0] * one_minus_cos_theta - axis[1] * sin_theta, axis[2] * axis[1] * one_minus_cos_theta + axis[0] * sin_theta, cos_theta + axis[2]**2 * one_minus_cos_theta]
        ])

        return rotation_matrix

    def radius_search_graph(self, q_points, q_lengths, radius, neighbor_limit):
        r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

        This function is implemented on CPU.

        Args:
            q_points (Tensor): the query points (N, 3)
            s_points (Tensor): the support points (M, 3)
            q_lengths (Tensor): the list of lengths of batch elements in q_points
            s_lengths (Tensor): the list of lengths of batch elements in s_points
            radius (float): maximum distance of neighbors
            neighbor_limit (int): maximum number of neighbors

        Returns:
            neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
                Filled with M if there are less than k neighbors.
        """
        # neighbor_indices = ext_module.radius_neighbors(q_points, q_points, q_lengths, q_lengths, radius)
        # if neighbor_limit > 0:
        #     neighbor_indices = neighbor_indices[:, :neighbor_limit]
        source = q_points[:q_lengths[0]]
        target = q_points[q_lengths[0]:]
        edge_index_source = radius_graph(source, r=radius, max_num_neighbors = neighbor_limit)
        edge_index_target = radius_graph(target, r=radius, max_num_neighbors = neighbor_limit) + q_lengths[0]
        
        edge_index = torch.cat([edge_index_source, edge_index_target], dim=1)
        # edge_index = edge_index.transpose(0, 1)
        return edge_index

    def distance(self, a, b):
        return torch.sqrt(torch.sum((a-b)**2, axis = 1))

    def rank_edge_distances(
            self, edge_distance, edge_index, max_num_neighbors: int
        ) -> torch.Tensor:
        device = edge_distance.device
        # Create an index map to map distances from atom_distance to distance_sort
        # index_sort_map assumes index to be sorted
        output, num_neighbors = torch.unique(edge_index[1], return_counts=True)
        index_neighbor_offset = (
            torch.cumsum(num_neighbors, dim=0) - num_neighbors
        )
        index_neighbor_offset_expand = torch.repeat_interleave(
            index_neighbor_offset, num_neighbors
        )
        index_sort_map = (
            edge_index[1] * max_num_neighbors
            + torch.arange(len(edge_distance), device=device)
            - index_neighbor_offset_expand
        )
        num_atoms = int(torch.max(edge_index)) + 1
        distance_sort = torch.full(
            [num_atoms * max_num_neighbors], np.inf, device=device
        )
        distance_sort.index_copy_(0, index_sort_map, edge_distance)
        distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
        no_op, index_sort = torch.sort(distance_sort, dim=1)

        index_map = (
            torch.arange(max_num_neighbors, device=device)
            .view(1, -1)
            .repeat(num_atoms, 1)
            .view(-1)
        )
        index_sort = index_sort + (
            torch.arange(num_atoms, device=device) * max_num_neighbors
        ).view(-1, 1).repeat(1, max_num_neighbors)
        edge_rank = torch.zeros_like(index_map)
        edge_rank.index_copy_(0, index_sort.view(-1), index_map)
        edge_rank = edge_rank.view(num_atoms, max_num_neighbors)

        index_sort_mask = distance_sort.lt(1000.0)
        edge_rank = torch.masked_select(edge_rank, index_sort_mask)

        return edge_rank

    def get_cutoff_index(self, edge_rank, edge_index, edge_distance, max_num_neighbors, num_resolutions = 3):
        # Reorder edges so that they are grouped by distance rank (lowest to highest)
        device = edge_distance.device
        last_cutoff = -0.1
        message_block_idx = torch.zeros(len(edge_distance), device=device)
        edge_distance_reorder = torch.tensor([], device=device)
        edge_index_reorder = torch.tensor([], device=device)
        edge_distance_vec_reorder = torch.tensor([], device=device)
        cutoff_index = torch.tensor([0], device=device)

        # 要在这里设置切点位置下标
        # if num_resolutions==3:
        #     cutoff_list = torch.tensor([10 - 0.01, 20 - 0.01, max_num_neighbors - 0.01])
        # if num_resolutions==2:
        cutoff_list = torch.tensor([20 - 0.01, max_num_neighbors - 0.01])
        # if num_resolutions==1:
            # cutoff_list = torch.tensor([max_num_neighbors - 0.01])

        for i in range(num_resolutions):
            mask = torch.logical_and(
                edge_rank.gt(last_cutoff), edge_rank.le(cutoff_list[i])
            )
            last_cutoff = cutoff_list[i]
            message_block_idx.masked_fill_(mask, i)
            edge_distance_reorder = torch.cat(
                [
                    edge_distance_reorder,
                    torch.masked_select(edge_distance, mask),
                ],
                dim=0,
            )
            edge_index_reorder = torch.cat(
                [
                    edge_index_reorder,
                    torch.masked_select(
                        edge_index, mask.view(1, -1).repeat(2, 1)
                    ).view(2, -1),
                ],
                dim=1,
            )
            cutoff_index = torch.cat(
                [
                    cutoff_index,
                    torch.tensor(
                        [len(edge_distance_reorder)], device=device
                    ),
                ],
                dim=0,
            )

        edge_index = edge_index_reorder.long()
        edge_distance = edge_distance_reorder
        # edge_distance_vec = edge_distance_vec_reorder
        return edge_index, edge_distance, cutoff_index
    
class decoder(nn.Module):
    def __init__(self, inv_channel_l, inv_channel_c, equ_channel_l, equ_channel_c, out_channel, L_in, L_out, group_norm=None):
        super(decoder, self).__init__()
        self.in_basis = (L_in + 1) ** 2
        self.basis = (L_out + 1) ** 2
        self.min_basis = (min(L_in, L_out) + 1) ** 2
        self.fc_last_equ = nn.Linear(equ_channel_l, equ_channel_c, bias=False)
        self.fc_cur_equ = nn.Linear(equ_channel_c, equ_channel_c, bias=False)

        if group_norm:
            self.mlp = UnaryBlock(inv_channel_l + inv_channel_c + equ_channel_c, out_channel, group_norm)
        else:
            self.mlp = LastUnaryBlock(inv_channel_l + inv_channel_c + equ_channel_c, out_channel)
    
    def forward(self, last_inv, cur_inv, last_equ, cur_equ, upsampling):
        last_inv = nearest_upsample(last_inv, upsampling)
        last_equ = nearest_upsample(last_equ, upsampling)[:, :self.min_basis, :]
        last_equ = self.fc_last_equ(last_equ)
        cur_equ = cur_equ[:, :self.min_basis, :]
        cur_equ = self.fc_cur_equ(cur_equ)
        equ = (cur_equ * last_equ).mean(1)
        com = torch.cat([last_inv, cur_inv, equ], dim=-1)
        com = self.mlp(com)
        return com

class combination_normal(nn.Module):
    def __init__(self, channels):
        super(combination_normal, self).__init__()
        self.fc = nn.Linear(1, channels, bias=False)

    def forward(self, equ, normal):
        equ_new = equ
        normal = normal.unsqueeze(-1)
        equ_new[:, 1:4, :] += self.fc(normal)
        return equ_new

# class combination(nn.Module):
#     def __init__(self, in_channels, out_channels, sigma, preprocess=False):
#         super(combination, self).__init__()
#         self.preprocess = preprocess
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # self.fc = nn.Linear(in_channels, out_channels, bias=False)
#         if self.preprocess:
#             if in_channels == out_channels:
#                 self.shortcut = nn.Identity()
#             else:
#                 self.shortcut = nn.Linear(in_channels, out_channels, bias=False)

#         self.sigma = sigma
#         # self.initialize_parameters()

#     def initialize_parameters(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_normal_(module.weight)

#     def forward(self, equ, points, neighbors):
#         # equ: [N, L, C]
#         N, L, C = equ.shape
#         equ_ori = equ
#         # equ = self.fc(equ)
#         equ_r = equ[neighbors[0, :]]
#         # equ_t = equ[neighbors[1, :]]
#         vec = points[neighbors[0, :]] - points[neighbors[1, :]]
#         distance = torch.norm(vec, dim=1)
#         # weights = torch.clamp(1 - distance / mid, min=0.0)
#         weights = torch.clamp(1 - distance / self.sigma, min=0.0)
#         # weights = None/
#         # print(weights)
#         if self.preprocess:
#             equ_r = equ_r * weights.view(-1, 1, 1)

#             # neighbor_num = torch.max(inv_message, dim=-1, keepdim=True)[0]
#             # neighbor_num = torch.ones(neighbors.size(-1), device=neighbors.device)
#             # neighbor_num = torch.gt(neighbor_num, 0.0).to(dtype=torch.int)
#             neighbor_num = torch.ones(neighbors.shape[-1], 1, dtype=torch.int, device=neighbors.device)
#             neighbors_new = torch.ones(
#                 N,
#                 1,
#                 dtype=neighbor_num.dtype,
#                 device=neighbor_num.device,
#             )
#             neighbors_new.index_add_(0, neighbors[1, :], neighbor_num)

#             equ_new = torch.zeros(
#                 N,
#                 L,
#                 self.out_channels,
#                 dtype=equ.dtype,
#                 device=equ.device,
#             )
#             equ_new.index_add_(0, neighbors[1, :], equ_r.to(equ_new.dtype))
#             equ_new = (equ_new + self.shortcut(equ_ori)) / neighbors_new.unsqueeze(-1)
#             return equ_new, weights, distance, vec
#         else:
#             return equ, weights, distance, vec

class FPFH(nn.Module):
    def __init__(self, radius_normal=0.1, radius_feature=0.2, neighbor=30):
        super(FPFH, self).__init__()
        self.radius_normal = radius_normal
        self.radius_feature = radius_feature
        self.neighbor = neighbor

    def forward(self, xyz):
        # xyz = xyz.transpose(1, 2).cpu().numpy()
        # res = np.zeros((xyz.shape[1], 33))
        xyz = xyz.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=self.neighbor))
        res = pcd_fpfh.data
        res = torch.from_numpy(res).float().cuda()
        res = res.transpose(0, 1).contiguous()
        normals = np.asarray(pcd.normals)
        normals = torch.from_numpy(normals).float().cuda()
        return res.detach(), normals.detach()

class normal(nn.Module):
    def __init__(self, radius_normal=0.1, radius_feature=0.2, neighbor=30):
        super(normal, self).__init__()
        self.radius_normal = radius_normal
        self.radius_feature = radius_feature
        self.neighbor = neighbor

    def forward(self, xyz, radius_normal):
        xyz = xyz.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=self.neighbor))
        normals = np.asarray(pcd.normals)
        normals = torch.from_numpy(normals).float().cuda()
        return normals.detach()


if __name__=="__main__":
    # L = 1
    # encoder = ResidualBlock_cg(64, 64, L)
    # x = torch.randn(100, 3)
    # # feats = torch.randn(100, 9, 64)
    # feats = o3.spherical_harmonics([0, 1], x, False, normalization='component').view(100, (L+1)**2, 1).expand(-1,-1,64)
    # neighbor = torch.randint(100, (100, 38))
    # output = encoder(feats, x, neighbor)

    # # 
    # R = torch.Tensor(data=[[-0.7887,0.5774,0.2113], [0.5774,0.5774,0.5774],[0.2113,0.5774,-0.7887]])

    # print(R @ R.transpose(0, 1))

    # R = R.unsqueeze(0).expand(100, -1, -1)

    # x1 = R @ x.view(100, 3, 1)
    # x1 = x1.squeeze()
    # feats1 = o3.spherical_harmonics([0, 1], x1, False, normalization='component').view(100, (L+1) ** 2, 1).expand(-1, -1, 64)
    # # neighbor = torch.randint(100, (100, 38))
    # output1 = encoder(feats1, x1, neighbor)
    # output1[:, 1:, :] = R @ output1[:, 1:, :] 
    # print(output1-output)

    def generate_cube_surface_points(size=1.0, num_points_per_edge=10):
        # Generate points on the surfaces of the cube
        edge_points = torch.linspace(-size/2, size/2, num_points_per_edge)
        x, y, z = torch.meshgrid(edge_points, edge_points, edge_points)
        surface_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

        # Remove points inside the cube
        mask = ((torch.abs(surface_points[:, 0]) == size/2) |
                (torch.abs(surface_points[:, 1]) == size/2) |
                (torch.abs(surface_points[:, 2]) == size/2))
        surface_points = surface_points[mask]

        return surface_points

    # Example usage
    points = generate_cube_surface_points(size=1.0, num_points_per_edge=10)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # points = torch.ones(100, 3)
    points = points.to(device="cuda")
    model = FPFH()
    feats = model(points)

    embed()