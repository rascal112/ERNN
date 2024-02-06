import torch
import torch.nn as nn
from IPython import embed
from geotransformer.modules.ops import pairwise_distance
from e3nn.o3 import FromS2Grid, ToS2Grid

class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
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
        if self.dual_normalization:
            ref_distance = distance / distance.sum(dim=1, keepdim=True)
            src_distance = distance / distance.sum(dim=0, keepdim=True)
            distance = ref_distance * src_distance
        return distance


    def forward(self, ref_feats, src_feats, ref_equ, src_equ, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        ref_equ = ref_equ[ref_indices]
        src_equ = src_equ[src_indices]
        # select top-k proposals

        # ref_equ = self.ToGrid(ref_equ) # N, S, C
        # src_equ = self.ToGrid(src_equ)
        # ref_equ = ref_equ.squeeze(-1)
        # src_equ = src_equ.squeeze(-1)
        matching_scores1 = pairwise_distance(ref_feats, src_feats, normalized=True)
        matching_scores2 = pairwise_distance(ref_equ, src_equ, normalized=False) * 0.1
        matching_scores = (1 - self.gamma) * matching_scores1 + self.gamma * matching_scores2
        matching_scores = torch.exp(-matching_scores)

        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores

        # matching_scores = (1 - self.gamma) * matching_scores1 + self.gamma * matching_scores2
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores
    
class SuperPointMatching_equ(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching_equ, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores
    
# class togrid(nn.Module):
#     def __init__(self, lmax):

