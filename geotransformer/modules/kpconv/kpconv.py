import math
from sklearn import neighbors

import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import index_select
from geotransformer.modules.kpconv.kernel_points import load_kernels
from IPython import embed

from geotransformer.modules.cg import Activation, Gate
from geotransformer.modules.cg import EquivariantLayerNormV2, EquivariantLayerNormV2_channel
from geotransformer.modules.cg import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)

from geotransformer.modules.kpconv.fast_tensorproduct import escn_tensor_product

import e3nn.o3 as o3
import e3nn


class KPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        bias=False,
        dimension=3,
        inf=1e6,
        eps=1e-9,
    ):
        """Initialize parameters for KPConv.

        Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

        Deformable KPConv is not supported.

        Args:
             in_channels: dimension of input features.
             out_channels: dimension of output features.
             kernel_size: Number of kernel points.
             radius: radius used for kernel point init.
             sigma: influence radius of each kernel point.
             bias: use bias or not (default: False)
             dimension: dimension of the point space.
             inf: value of infinity to generate the padding point
             eps: epsilon for gaussian influence
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension

        self.inf = inf
        self.eps = eps

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(self.kernel_size, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('bias', None)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer('kernel_points', kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self):
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed='center')
        return torch.from_numpy(kernel_points).float()

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        r"""KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)
        neighbors = index_select(s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)  # (N+1, C) -> (M, H, C)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        weighted_feats = weighted_feats.permute(1, 0, 2)  # (M, K, C) -> (K, M, C)
        kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, C_out) -> (K, M, C_out)
        output_feats = torch.sum(kernel_outputs, dim=0, keepdim=False)  # (K, M, C_out) -> (M, C_out)

        # normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'kernel_size: {}'.format(self.kernel_size)
        format_string += ', in_channels: {}'.format(self.in_channels)
        format_string += ', out_channels: {}'.format(self.out_channels)
        format_string += ', radius: {:g}'.format(self.radius)
        format_string += ', sigma: {:g}'.format(self.sigma)
        format_string += ', bias: {}'.format(self.bias is not None)
        format_string += ')'
        return format_string

    
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class TP(nn.Module):
    def __init__(
        self, 
        l1, 
        lo,
        c_in, 
        c_out,
        first_time,
    ):
        super(TP, self).__init__()
        c_mid = max(c_in // 16, 8)
        self.first_time = first_time

        self.fc_mess_in = nn.Linear(2 * c_in, c_mid, bias=False)
        self.fc_mess_out = nn.Linear(c_mid, c_out, bias=False)

        self.cg_manner = escn_tensor_product(l1, lo)
        
        # self.sigma = init_sigma
        # self.distance_expansion = GaussianSmearing(0.0, 5.0, 16)
        # self.fc_dist = nn.Linear(16, 1)


    def forward(self, s_points, neighbor_points, s_feats, neighbor_feats, values):
        N, H, L, C = neighbor_feats.shape
        vec_n = neighbor_points - s_points[:-1].unsqueeze(1)

        s_feats = s_feats.unsqueeze(1).repeat(1, H, 1, 1)
        message = torch.cat([neighbor_feats, s_feats[:-1]], dim=-1)
        message = self.fc_mess_in(message)

        # if self.first_time:
        self.cg_manner.init_wigner(vec_n)

        message, length = self.cg_manner(message, vec_n) # [N, H', L, C]
        # message = message
        # length = self.distance_expansion(length)
        # length = self.fc_dist(length).view(N, H, 1)
        # length = torch.clamp(1 - torch.sqrt(length) / self.sigma, min=0.0)
        # values = F.softmax(values + length, dim=1)
        # message = (values.view(N, H, 1, 1) * message).sum(1) # [N, L, C]
        message = message.sum(1)
        
        message = self.fc_mess_out(message)
        return message

class CGConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        L_in,
        L_out,
        first_time=True,
        neighbor_cg=8,
        inf=1e6,
    ):
        super(CGConv, self).__init__()
        mid_channels = in_channels // 8
        self.inf = inf
        self.neighbor_cg = neighbor_cg
        # self.fc_sign = nn.Sequential(nn.Linear(in_channels, mid_channels),
        #                              nn.LayerNorm(mid_channels),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(mid_channels, 1),
        #                              nn.ReLU(inplace=True)
        #                              )
        self.layer_norm = EquivariantLayerNormV2_channel(L_in, in_channels)
        self.tp = TP(l1=L_in, lo=L_out, c_in=in_channels, c_out=out_channels, first_time=first_time)
        self.sigma = 0.1

    def forward(self, s_feats, s_points, neighbor_indices):
        r"""KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in, L_irreps)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)

        N, L, C = s_feats.shape
        s_feats = self.layer_norm(s_feats)
        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :, :])), 0)  # (N, L, C) -> (N+1, H, L, C)
        neighbor_indices = neighbor_indices[:, 1:]
        neighbor_indices = neighbor_indices.contiguous()
        neighbor_points = index_select(s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)  
        neighbor_feats = index_select(s_feats, neighbor_indices, dim=0)
        neighbor_tmp = torch.zeros_like(neighbor_indices, device=s_feats.device)
        neighbor_tmp[neighbor_indices<N] = 1
        neighbor_tmp = neighbor_tmp.unsqueeze(-1)
        # significance_score = neighbor_feats[:, :, 0, :] * neighbor_tmp.view(N, -1, 1, 1)
        # significance_score = self.fc_sign(neighbor_feats[:, :, 0, :]) # (N+1, H, 1)
        # significance_score = significance_score * neighbor_tmp.unsqueeze(-1) + self.sigma * neighbor_tmp.unsqueeze(-1)
        # # significance_score *= neighbor_tmp.unsqueeze(-1)
        # values, indices = significance_score.topk(self.neighbor_cg, dim=1, largest=True) # (N+1, H', 1)
        # # indices = indices.cpu()

        # # indices = indices.view(N, -1)

        # # TODO: remove points if their signs exceed threshold
        # neighbor_feats = torch.gather(neighbor_feats, 1, indices.unsqueeze(-1).expand(-1, -1, L, C)) # (N+1, H', L, C)
        # neighbor_points = torch.gather(neighbor_points, 1, indices.expand(-1, -1, 3)) # (N+1, H', 3)
        # neighbor_feats = index_select(s_feats, indices, dim=0)
        # neighbor_points = index_select(s_points, indices, dim=0)
        neighbor_points = (neighbor_points * neighbor_tmp)[:, :self.neighbor_cg, :]
        neighbor_feats = (neighbor_feats * neighbor_tmp.unsqueeze(-1))[:, :self.neighbor_cg, :, :]
        # neighbor_feats = neighbor_feats[:, , , :]


        values = 0
        output = self.tp(s_points, neighbor_points, s_feats, neighbor_feats, values)
        # output = neighbor_feats.sum(1)

        return output