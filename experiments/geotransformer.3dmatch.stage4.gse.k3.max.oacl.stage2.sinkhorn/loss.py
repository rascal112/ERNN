import torch
import torch.nn as nn
from IPython import embed

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.modules.ops.pairwise_distance import pairwise_distance
from e3nn.o3 import FromS2Grid, ToS2Grid


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

        # self.lmax = lmax = 1
        # self.num_bands = 1
        # self.grid_res = 4
        # self.grid_num = self.grid_res * (self.grid_res + 1)
        # self.sphere_basis = (lmax + 1) ** 2
        # self.device = "cuda"

        # self.basis = (lmax + 1) ** 2

        # self.to_grid_shb = torch.tensor([], device=self.device)
        # self.to_grid_sha = torch.tensor([], device=self.device)
        # for b in range(self.num_bands):
        #     l = self.lmax - b  # noqa: E741
        #     togrid = ToS2Grid(
        #         l,
        #         (self.grid_res, self.grid_res + 1),
        #         normalization="integral",
        #         device=self.device,
        #     )
        #     shb = togrid.shb
        #     sha = togrid.sha

        #     padding = torch.zeros(
        #         shb.size()[0],
        #         shb.size()[1],
        #         self.sphere_basis - shb.size()[2],
        #         device=self.device,
        #     )
        #     shb = torch.cat([shb, padding], dim=2)
        #     self.to_grid_shb = torch.cat([self.to_grid_shb, shb], dim=0)
        #     if b == 0:
        #         self.to_grid_sha = sha
        #     else:
        #         self.to_grid_sha = torch.block_diag(self.to_grid_sha, sha)
            
        # self.to_grid_sha = self.to_grid_sha.view(
        #     self.num_bands, self.grid_res + 1, -1
        # )
        # self.to_grid_sha = torch.transpose(self.to_grid_sha, 0, 1).contiguous()
        # self.to_grid_sha = self.to_grid_sha.view(
        #     (self.grid_res + 1) * self.num_bands, -1
        # )

        # self.to_grid_shb = self.to_grid_shb.detach()
        # self.to_grid_sha = self.to_grid_sha.detach()

        # self.from_grid = FromS2Grid(
        #     (self.grid_res, self.grid_res + 1),
        #     self.lmax,
        #     normalization="integral",
        #     device=self.device,
        # )

        self.gamma = 0.5
    
    def ToGrid(self, x):
        # x = x.view(-1, self.basis, channels)
        channels = x.shape[-1]
        x_grid = torch.einsum("mbi,zic->zbmc", self.to_grid_shb, x)
        x_grid = torch.einsum(
            "am,zbmc->zbac", self.to_grid_sha, x_grid
        ).contiguous()
        x_grid = x_grid.view(-1, self.grid_num, self.num_bands * channels)
        return x_grid
    
    def match_equ(self, equ1, equ2):
        # equ1 N, S, C
        # equ2 M, S, C
        ref = equ1.permute(2, 0, 1)
        src = equ2.permute(2, 0, 1)
        distance = torch.cdist(ref, src)
        distance = distance.mean(0)
        ref_distance = distance / distance.sum(dim=1, keepdim=True)
        src_distance = distance / distance.sum(dim=0, keepdim=True)
        distance = ref_distance * src_distance
        return distance

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']

        ref_equ = output_dict['ref_equ_c']
        src_equ = output_dict['src_equ_c']

        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists1 = pairwise_distance(ref_feats, src_feats, normalized=True)
        feat_dists2 = pairwise_distance(ref_equ, src_equ, normalized=False) * 0.1
        # ref_equ = self.ToGrid(ref_equ)
        # src_equ = self.ToGrid(src_equ)
        # feat_dists2 = self.match_equ(ref_equ, src_equ)

        feat_dists = (1 - self.gamma) * feat_dists1 + self.gamma * feat_dists2

        feat_dists = torch.sqrt(feat_dists)

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
