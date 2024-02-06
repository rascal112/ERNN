import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer
from geotransformer.modules.cg import EquivariantLayerNormV2, EquivariantLayerNormV2_channel

from einops import rearrange
from e3nn.o3 import FromS2Grid, ToS2Grid

from IPython import embed

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        in_equ,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, in_equ, num_heads, dropout=dropout, activation_fn=activation_fn)

        self.lmax = lmax = 2
        self.num_bands = 1
        self.grid_res = 6
        self.grid_num = self.grid_res * (self.grid_res + 1)
        self.sphere_basis = (lmax + 1) ** 2
        self.device = "cuda"

        self.basis = (lmax + 1) ** 2

        self.to_grid_shb = torch.tensor([], device=self.device)
        self.to_grid_sha = torch.tensor([], device=self.device)
        for b in range(self.num_bands):
            l = self.lmax - b  # noqa: E741
            togrid = ToS2Grid(
                l,
                (self.grid_res, self.grid_res + 1),
                normalization="integral",
                device=self.device,
            )
            shb = togrid.shb
            sha = togrid.sha

            padding = torch.zeros(
                shb.size()[0],
                shb.size()[1],
                self.sphere_basis - shb.size()[2],
                device=self.device,
            )
            shb = torch.cat([shb, padding], dim=2)
            self.to_grid_shb = torch.cat([self.to_grid_shb, shb], dim=0)
            if b == 0:
                self.to_grid_sha = sha
            else:
                self.to_grid_sha = torch.block_diag(self.to_grid_sha, sha)
            
        self.to_grid_sha = self.to_grid_sha.view(
            self.num_bands, self.grid_res + 1, -1
        )
        self.to_grid_sha = torch.transpose(self.to_grid_sha, 0, 1).contiguous()
        self.to_grid_sha = self.to_grid_sha.view(
            (self.grid_res + 1) * self.num_bands, -1
        )

        self.to_grid_shb = self.to_grid_shb.detach()
        self.to_grid_sha = self.to_grid_sha.detach()

        self.from_grid = FromS2Grid(
            (self.grid_res, self.grid_res + 1),
            self.lmax,
            normalization="integral",
            device=self.device,
        )


        # self.equ_self = attention_equ('self', in_equ, num_heads)
        # self.equ_cross = attention_equ('cross', in_equ, num_heads)

        self.out_proj = nn.Linear(hidden_dim, output_dim)
        # self.out_equ = nn.Linear(in_equ, 1)
        self.out_equ = nn.Sequential(nn.Linear(3, 4),
                                      nn.ReLU(),
                                      nn.Linear(4, 1))

    def ToGrid(self, x):
        # x = x.view(-1, self.basis, channels)
        channels = x.shape[-1]
        x_grid = torch.einsum("mbi,zic->zbmc", self.to_grid_shb, x)
        x_grid = torch.einsum(
            "am,zbmc->zbac", self.to_grid_sha, x_grid
        ).contiguous()
        x_grid = x_grid.view(-1, self.grid_num, self.num_bands * channels)
        return x_grid

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_equ,
        src_equ,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points) # N: 216 invariant
        src_embeddings = self.embedding(src_points) # N: 208

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, equ_new1, equ_new2 = self.transformer(
            ref_feats,
            src_feats,
            ref_equ,
            src_equ,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )
        
        ref_equ = self.ToGrid(equ_new1.squeeze(0))
        src_equ = self.ToGrid(equ_new2.squeeze(0))

        ref_equ = torch.cat([torch.max(ref_equ, 1)[0].unsqueeze(-1), torch.mean(ref_equ, 1).unsqueeze(-1), torch.var(ref_equ, 1).unsqueeze(-1)], dim=-1)

        src_equ = torch.cat([torch.max(src_equ, 1)[0].unsqueeze(-1), torch.mean(src_equ, 1).unsqueeze(-1), torch.var(src_equ, 1).unsqueeze(-1)], dim=-1)

        ref_equ = self.out_equ(ref_equ)
        src_equ = self.out_equ(src_equ)

        # equ_new1, equ_new2 = self.equ_self(attention_scores[-2], ref_equ, src_equ)
        # equ_new1, equ_new2 = self.equ_cross(attention_scores[-1], equ_new1, equ_new2)

        # equ_new1 = self.out_equ(equ_new1)
        # equ_new2 = self.out_equ(equ_new2)

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        ref_equ = ref_equ.squeeze(-1)
        src_equ = src_equ.squeeze(-1)
        return ref_feats, src_feats, ref_equ, src_equ

class attention_equ(nn.Module):
    def __init__(self, block, in_channels, head):
        super(attention_equ, self).__init__()
        self.block = block
        self.num_heads = head
        # self.proj_q = nn.Linear(in_channels, in_channels, bias=False)
        # self.proj_k = nn.Linear(in_channels, in_channels, bias=False)
        self.proj_v = nn.Linear(in_channels, in_channels, bias=False)

        self.output = nn.Linear(in_channels, in_channels, bias=False)
        # self.output2 = nn.Linear(in_channels, in_channels, bias=False)
        # self.equ_norm = EquivariantLayerNormV2_channel(1, in_channels)

    def attention(self, equ, attention):
        # q = rearrange(self.proj_q(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        # k = rearrange(self.proj_k(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        l = equ.shape[2]
        v = rearrange(self.proj_v(equ), 'b n l (h c) -> b h l n c', h=self.num_heads)
        attention = attention.unsqueeze(2)
        attention = attention.expand(-1, -1, l, -1, -1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h l n c -> b n l (h c)', h=self.num_heads)
        out = self.output(out)
        return out

    def forward(self, attention, equ1, equ2):
        if self.block == 'self':
            equ_new1 = self.attention(equ1, attention[0]) + equ1
            equ_new2 = self.attention(equ2, attention[1]) + equ2
        else:
            equ_new1 = self.attention(equ2, attention[0]) + equ1
            equ_new2 = self.attention(equ1, attention[1]) + equ2
        # equ_new1 = (self.equ_norm(equ_new1.squeeze(0)).unsqueeze(0))
        # equ_new2 = (self.equ_norm(equ_new2.squeeze(0)).unsqueeze(0))
        return equ_new1, equ_new2
            

        