import torch
import torch.nn as nn

from geotransformer.modules.kpconv.functional import maxpool, nearest_upsample, global_avgpool, knn_interpolate
from geotransformer.modules.kpconv.kpconv import KPConv, CGConv

from geotransformer.modules.ops import index_select
from geotransformer.modules.cg import Activation, Gate
from geotransformer.modules.cg import EquivariantLayerNormV2, EquivariantLayerNormV2_channel
from geotransformer.modules.cg import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)

from geotransformer.modules.kpconv.fast_tensorproduct import escn_tensor_product

# from geotransformer.modules.kpconv.cg_tensorproduct import sh_tensor_product

import numpy as np
import e3nn.o3 as o3
import torch.nn.init as init

from IPython import embed


class KNNInterpolate(nn.Module):
    def __init__(self, k, eps=1e-8):
        super(KNNInterpolate, self).__init__()
        self.k = k
        self.eps = eps

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        if self.k == 1:
            return nearest_upsample(s_feats, neighbor_indices)
        else:
            return knn_interpolate(s_feats, q_points, s_points, neighbor_indices, self.k, eps=self.eps)


class MaxPool(nn.Module):
    @staticmethod
    def forward(s_feats, neighbor_indices):
        return maxpool(s_feats, neighbor_indices)


class GlobalAvgPool(nn.Module):
    @staticmethod
    def forward(feats, lengths):
        return global_avgpool(feats, lengths)


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        r"""Initialize a group normalization block.

        Args:
            num_groups: number of groups
            num_channels: feature dimension
        """
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x.squeeze()


class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm, has_relu=True, bias=True, layer_norm=False):
        r"""Initialize a standard unary block with GroupNorm and LeakyReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            group_norm: number of groups in group normalization
            bias: If True, use bias
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        if has_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = None

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x


class LastUnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        r"""Initialize a standard last_unary block without GN, ReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
        """
        super(LastUnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        negative_slope=0.1,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a KPConv block with ReLU and BatchNorm.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            negative_slope: leaky relu negative slope
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x)
        x = self.leaky_relu(x)
        # feats_s1 = torch.zeros(x.shape[0], 4, self.out_channels, device=x.device)
        # feats_s1[:, 0, :] += x
        return x.unsqueeze(1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        strided=False,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm_conv = nn.LayerNorm(mid_channels)
        else:
            self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(
            mid_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(
                in_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        # x = s_feats[:, 0, :]
        s_feats = s_feats.squeeze()
        x = self.unary1(s_feats)

        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)

        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        # s_feats[:, 0, :] = x
        # feats = torch.zeros(x.shape[0], 4, self.out_channels, device=x.device)
        # feats[:, 0, :] += x

        return x.unsqueeze(1)

class e3nn_tp(nn.Module):
    def __init__(self, L_in1, L_in2=None, L_out=None):
        super(e3nn_tp, self).__init__()
        irreps_in1 = o3.Irreps.spherical_harmonics(L_in1, p=1)
        if L_in2 is not None:
            irreps_in2 = o3.Irreps.spherical_harmonics(L_in2, p=1)
        else:
            irreps_in2 = o3.Irreps("1e")
        if L_out is not None:
            irreps_out = o3.Irreps.spherical_harmonics(L_out, p=1)
        else:
            irreps_out = o3.Irreps.spherical_harmonics(L_in1, p=1)

        self.tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

    def forward(self, x1, x2):
        N, L, C = x1.shape
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()
        x1 = x1.view(N * C, -1)
        x2 = x2.view(N * C, -1)
        out = self.tp(x1, x2)

        out = out.view(N, C, -1)
        out = out.transpose(1, 2).contiguous()
        return out


# class invencoder(nn.Module):
#     def __init__(self, in_channels, out_channels, equ_inputchannel, equ_channels, L_in, L_out):
#         super(invencoder, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.equ_channels = equ_channels
#         self.basis = (L_out + 1) ** 2
#         self.min_basis = (min(L_in, L_out) + 1) ** 2
#         self.debug = False

#         self.inv_norm = nn.LayerNorm(in_channels)
#         self.equ_norm = EquivariantLayerNormV2_channel(L_in, equ_inputchannel)

#         self.fc_equ = nn.Linear(equ_inputchannel, equ_channels, bias=False)
#         # self.fc_inv = nn.Linear(in_channels, equ_channels, bias=False)

#         self.mid_channels = out_channels // 2

#         self.cg_1 = e3nn_tp(L_in, L_in, L_out)
#         self.equ_fc_1 = nn.Linear(equ_channels, equ_channels, bias=False)
#         self.L_out = L_out
#         if L_out == 0:
#             self.gate = nn.SiLU()
#         else: 
#             # self.fc_vec = nn.Linear(1, equ_channels, bias=False)
#             self.gate = Gate_3j(L_out, equ_channels)
#             # self.cg_2 = e3nn_tp(L_out, L_out=L_out)
#             self.cg_2 = escn_tensor_product(L_out, L_out)
#         self.equ_fc_2 = nn.Linear(equ_channels, equ_channels, bias=False)
#         self.equ_fc_2 = nn.Linear(equ_channels, equ_channels, bias=False)
#         self.inv_fc_1 = nn.Linear(equ_channels + 2 * in_channels, self.mid_channels)
#         self.inv_fc_2 = nn.Linear(self.mid_channels, out_channels)

#         self.act = nn.SiLU()

#         if in_channels == out_channels:
#             self.shortcut1 = nn.Identity()
#         else:
#             self.shortcut1 = nn.Linear(in_channels, out_channels)

#         if equ_inputchannel == equ_channels:
#             self.shortcut2 = nn.Identity()
#         else:
#             self.shortcut2 = nn.Linear(equ_inputchannel, equ_channels, bias=False)

#     def forward(self, feats, equ, points, neighbors, weights):
#         # neighbors: [2, edge_num]
#         # inv = self.fc_inv(feats).unsqueeze(1)
#         # equ = torch.cat([inv, equ], dim=1)

#         feats1 = self.inv_norm(feats)
#         equ1 = self.equ_norm(equ)
        
#         equ1 = self.fc_equ(equ1)
#         N = feats.shape[0]

#         # vec = points[neighbors[0, :]] - points[neighbors[1, :]]
#         # vec = self.fc_vec(vec.unsqueeze(-1))
#         s_equ = equ1[neighbors[0, :]]
#         t_equ = equ1[neighbors[1, :]]
#         s_feats = feats1[neighbors[0, :]]
#         t_feats = feats1[neighbors[1, :]]

#         # if self.debug:
#         #     embed()
#         equ_message = self.cg_1(s_equ, t_equ)
#         equ_message = self.equ_fc_1(equ_message)
#         equ_message = self.gate(equ_message)
#         if self.L_out != 0: 
#             vec = points[neighbors[0, :]] - points[neighbors[1, :]]
#             self.cg_2.init_wigner(vec)
#             vec = vec.unsqueeze(-1)
#             # vec = self.fc_vec(vec.unsqueeze(-1))
#             equ_message = self.cg_2(equ_message, vec) + equ_message
#         equ_message = self.equ_fc_2(equ_message) * weights.unsqueeze(-1)
#         inv_message = self.inv_fc_1(torch.cat([s_feats, t_feats, equ_message[:, 0, :]], dim=-1))
#         inv_message = self.act(inv_message)
#         inv_message = self.inv_fc_2(inv_message) * weights

#         inv_new = torch.zeros(
#             N,
#             self.out_channels,
#             dtype=feats.dtype,
#             device=feats.device,
#         )
#         inv_new.index_add_(0, neighbors[1, :], inv_message.to(inv_new.dtype))

#         equ_new = torch.zeros(
#             N,
#             self.basis,
#             self.equ_channels,
#             dtype=feats.dtype,
#             device=feats.device,
#         )
#         equ_new.index_add_(0, neighbors[1, :], equ_message.to(inv_new.dtype))
#         inv_new = inv_new + self.shortcut1(feats)
#         equ_new[:, :self.min_basis, :] = equ_new[:, :self.min_basis, :] + self.shortcut2(equ)[:, :self.min_basis, :]
#         return inv_new, equ_new
    
class invencoder(nn.Module):
    '''
    This class ignores self._cg_1 and employ e3nn to implement self.cg_2
    '''
    def __init__(self, in_channels, out_channels, equ_inputchannel, equ_channels, L_in, L_out, shalldow=True, init_wigner=False):
        super(invencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.equ_channels = equ_channels
        self.basis = (L_out + 1) ** 2
        self.min_basis = (min(L_in, L_out) + 1) ** 2
        self.L_in = L_in
        self.L_out = L_out
        self.debug = False
        self.shalldow = shalldow
        self.init_wigner = init_wigner

        self.inv_norm1 = nn.LayerNorm(in_channels)
        # self.inv_norm1 = GroupNorm(32, in_channels)
        self.equ_norm1 = EquivariantLayerNormV2_channel(L_in, equ_inputchannel)

        self.mid_channels = out_channels // 4

        self.inv_fc = nn.Linear(in_channels, self.mid_channels)
        self.equ_fc = nn.Linear(equ_inputchannel, self.mid_channels, bias=False)

        if L_out != 0:
            if equ_inputchannel == equ_channels:
                self.shortcut2 = nn.Identity()
            else:
                self.shortcut2 = nn.Linear(equ_inputchannel, equ_channels, bias=False)
            self.ffn_equ = nn.Sequential(EquivariantLayerNormV2_channel(L_out, equ_channels),
                                     nn.Linear(equ_channels, self.mid_channels, bias=False), 
                                     Gate_3j(L_out, self.mid_channels),
                                     nn.Linear(equ_channels, self.mid_channels, bias=False))
        else:
            self.gate = nn.SiLU()

        if self.shalldow:
            self.fc_l0 = nn.Linear(2 * self.mid_channels, self.mid_channels)
        else:
            self.cg1 = cross_function(self.mid_channels)
            self.gate1 = Gate_3j(L_out, equ_channels)
            self.equ_fc_1 = nn.Linear(self.mid_channels, self.mid_channels, bias=False)
            self.cg2 = cross_function(self.mid_channels)

        self.equ_fc_2 = nn.Linear(self.mid_channels, equ_channels, bias=False)

        self.inv_fc_1 = nn.Linear(3 * self.mid_channels, out_channels)
        # self.inv_fc_2 = nn.Linear(self.mid_channels, out_channels)
        
        self.act = nn.SiLU()

        if in_channels == out_channels:
            self.shortcut1 = nn.Identity()
        else:
            self.shortcut1 = nn.Linear(in_channels, out_channels)

        self.ffn_inv = nn.Sequential(nn.LayerNorm(out_channels),
                                    # GroupNorm(32, out_channels), 
                                     nn.Linear(out_channels, out_channels), 
                                     nn.ReLU(),
                                     nn.Linear(out_channels, out_channels))
        
        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 初始化线性层的权重
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # 初始化 LayerNorm 的权重和偏置
                init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)


    def forward(self, feats, equ, points, neighbors, subsampling=None, print_time=False):
        
        feats = self.inv_norm1(feats)
        equ = self.equ_norm1(equ)

        feats1 = self.inv_fc(feats)
        equ1 = self.equ_fc(equ)

        N = feats.shape[0]

        s_equ = equ1[neighbors[0, :]]
        t_equ = equ1[neighbors[1, :]]
        s_feats = feats1[neighbors[0, :]]
        t_feats = feats1[neighbors[1, :]]
        
        vec = points[neighbors[0, :]] - points[neighbors[1, :]]
        vec = o3.spherical_harmonics(np.arange(self.L_in + 1).tolist(), vec, False, normalization='component').unsqueeze(-1)
        vec = vec.expand(-1, -1, self.mid_channels)
        if self.shalldow:
            # equ_message = s_equ
            dot1 = (s_equ * t_equ).sum(1)
            dot2 = (s_equ * vec).sum(1)
            equ_l0 = self.act(self.fc_l0(torch.cat([dot1, dot2], dim=1)))
            equ_message = torch.cat([equ_l0.unsqueeze(1), s_equ[:, 1:, :]], dim=1)
        else:
            shortcut1 = s_equ + t_equ
            equ_message = self.cg1(s_equ, t_equ)
            equ_message = self.gate1(equ_message) + shortcut1
            equ_message = self.equ_fc_1(equ_message)
            shortcut2 = equ_message

            equ_message = self.cg2(equ_message, vec) + shortcut2

            equ_l0 = equ_message[:, 0, :]

        equ_message = self.equ_fc_2(equ_message)

        inv_message = self.inv_fc_1(torch.cat([s_feats, t_feats, equ_l0], dim=-1))
        inv_message = self.act(inv_message)

        neighbor_num = torch.max(inv_message, dim=-1, keepdim=True)[0]
        neighbor_num = torch.gt(neighbor_num, 0.0).to(dtype=torch.int)
        neighbors_new = torch.ones(
            N,
            1,
            dtype=neighbor_num.dtype,
            device=neighbor_num.device,
        )
        neighbors_new.index_add_(0, neighbors[1, :], neighbor_num)
        inv_new = torch.zeros(
            N,
            self.out_channels,
            dtype=feats.dtype,
            device=feats.device,
        )
        inv_new.index_add_(0, neighbors[1, :], inv_message.to(inv_new.dtype))
        inv_new = inv_new + self.shortcut1(feats)
        inv_new = inv_new / neighbors_new

        equ_new = torch.zeros(
            N,
            self.basis,
            self.equ_channels,
            dtype=feats.dtype,
            device=feats.device,
        )
        equ_new.index_add_(0, neighbors[1, :], equ_message.to(inv_new.dtype))
        equ_new = equ_new + self.shortcut2(equ)
        equ_new = equ_new / neighbors_new.unsqueeze(-1)

        if subsampling is not None:
            inv_new = inv_new[subsampling]
            equ_new = equ_new[subsampling]
        inv_new = self.ffn_inv(inv_new) + inv_new
        equ_new = self.ffn_equ(equ_new) + equ_new

        return inv_new, equ_new

class cross_function(nn.Module):
    def __init__(self, channel):
        super(cross_function, self).__init__()
        self.fc_t0 = nn.Linear(3 * channel, channel)
        self.fc_t1 = nn.Linear(channel, channel, bias=False)
        
    def forward(self, x1, x2):
        N, L, C = x1.shape
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()
        x1 = x1.view(N * C, -1)
        x2 = x2.view(N * C, -1)
        t1 = torch.cross(x1[:, 1:], x2[:, 1:])
        t1 = t1.view(N, C, 3)
        t0 = (x1[:, 1:] * x2[:, 1:]).sum(1, keepdim=True)
        t0 = torch.cat([t0, x1[:, 0:1], x2[:, 0:1]], dim=1)
        t0 = t0.view(N, 1, 3 * C)

        t1 = t1.transpose(1, 2).contiguous()
        
        t1 = self.fc_t1(t1)
        t0 = self.fc_t0(t0)
        out = torch.cat([t0, t1], dim=1)
        return out

class ResidualBlock_cg(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        L_in,
        L_out, 
        first_time=True,
        neighbor_cg=6,
        strided=False,
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ResidualBlock_cg, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_basis = (L_in + 1) ** 2
        self.l_out_basis = (L_out + 1) ** 2
        self.out_basis = min(self.l_basis, self.l_out_basis)
        
        mid_channels = out_channels // 4

        if in_channels != out_channels:
            self.linear3 = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.linear3 = nn.Identity()
        # cg_manner = sh_tensor_product(L_max)
        self.cg = CGConv(in_channels, out_channels, L_in, L_out, first_time, neighbor_cg)
        self.layer_norm2 = EquivariantLayerNormV2_channel(L_out, out_channels)
        self.ff = nn.Sequential(nn.Linear(out_channels, mid_channels, bias=False),
                                Gate_3j(L_out, mid_channels),
                                nn.Linear(mid_channels, out_channels, bias=False)
                                )

    def get_R_and_wigner(self):
        return self.cg.tp.cg_manner.R, self.cg.tp.cg_manner.wigner_D
    
    def update_R_and_wigner(self, x):
        self.cg.tp.cg_manner.R = x[0]
        self.cg.tp.cg_manner.wigner_D = x[1]

    def forward(self, s_feats, s_points, neighbor_indices):
        # x = self.layer_norm1(s_feats)
        # x = self.linear1(x)

        # N, K = s_feats.shape
        # feats = nn.zeros(N, self.l_basis, K, device=s_feats.device)
        # feats[:, 0, :] += s_feats
        x = self.cg(s_feats, s_points, neighbor_indices)
        x[:, :self.out_basis, :] = x[:, :self.out_basis, :] + self.linear3(s_feats[:, :self.out_basis, :])

        shortcut2 = x
        x = self.layer_norm2(x)
        x = self.ff(x) + shortcut2

        return x
    
class Gate_3j(nn.Module):
    def __init__(self, lmax, channel):
        super(Gate_3j, self).__init__()
        self.lmax = lmax
        self.channel = channel
        self.line = nn.Linear(channel, channel*(lmax+1), bias = False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): # x[L, C]
        # L = (lmax+1)**2 and C = channel
        assert(x.shape[1] == (self.lmax+1)**2)
        after_line = self.line(x[:, 0, :])
        type0 = after_line[:, :self.channel]
        mult = self.sigmoid(after_line[:, self.channel:]).unsqueeze(1)
        y = torch.zeros_like(x)
        y[:, 0, :] += self.silu(type0)
        for i in range(1, self.lmax+1):
            y[:, i**2:(i+1)**2, :] += x[:, i**2:(i+1)**2, :] * mult[:, :, (i-1) * self.channel: i * self.channel]
        return y

def equivariant_maxpool(x, neighbor_indices):
    """Max pooling from neighbors.

    Args:
        x: [n1, d] features matrix
        neighbor_indices: [n2, max_num] pooling indices

    Returns:
        pooled_feats: [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = torch.norm(neighbor_feats, dim=-1)
    pooled_index = pooled_feats.max(1)[1]
    pooled_feats = neighbor_feats[:, pooled_index]
    return pooled_feats


     