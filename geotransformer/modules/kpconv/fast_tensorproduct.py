from e3nn import o3
import numpy as np
import torch
import torch.nn as nn
import math
from IPython import embed
import os

_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))

class escn_tensor_product(nn.Module):
    def __init__(self, l1, lo):
        super(escn_tensor_product, self).__init__()
        self.lmax = l1
        self.y_l = 1
        self.lo = lo

        self.scale = 0.5
        self.x_basis = (l1 + 1) ** 2
        self.y_basis = 3
        self.out_basis = (lo + 1) ** 2
        self.CG = torch.zeros(self.x_basis, self.y_basis, self.out_basis, device="cuda")
        num_weights = 0
        # l2 = max(l1, lo)
        for l in range(self.lmax + 1):
            for out_l in range(abs(l - self.y_l), min(lo, l + self.y_l) + 1):
                self.CG[l ** 2 : (l + 1) ** 2, :, out_l ** 2 : (out_l + 1) ** 2] = self.clebsch_gordan(l, self.y_l, out_l)
                num_weights += 1
        self.weights = nn.Parameter(data=torch.ones(num_weights), requires_grad=True)

    def forward(self, t1, t2):
        N, L, C = t1.shape
        t1 = t1.view(-1, L, C)
        t2 = t2.view(-1, 3, 1)
        t2 = self.R @ t2
        t2_mod = t2[:, 1, :]

        # res = torch.zeros(N * H, L, C, device=t1.device)

        # compute tp
        res = torch.zeros(N, self.out_basis, C, device=t1.device)
        k = 0
        for l in range(self.lmax + 1):
            D = self.wigner_D[:, l ** 2 : (l + 1) ** 2, l ** 2 : (l + 1) ** 2]

            x1 = D @ t1[:, l ** 2 : (l + 1) ** 2, :]
            for out_l in range(max(abs(l - self.y_l), 0), min(self.lo, l + self.y_l) + 1):
                D_inv = self.wigner_D[:, out_l ** 2 : (out_l + 1) ** 2, out_l ** 2 : (out_l + 1) ** 2].transpose(1, 2).contiguous()
                right = self.CG[l ** 2 : (l + 1) ** 2, 1, out_l ** 2 : (out_l + 1) ** 2]
                tmp_res = torch.einsum('nkc,ko->noc', x1, right)
                tmp_res = (D_inv @ tmp_res)
                weights = t2_mod * self.weights[k] * self.scale
                res[:, out_l**2:(out_l+1)**2, :] = res[:, out_l**2:(out_l+1)**2, :] + tmp_res * weights.unsqueeze(-1)
                k += 1
        # N, H, L, C = t1.shape
        # t1 = t1.view(-1, L, C)
        # t2 = t2.view(-1, 3)
        # t2 = self.R @ t2.unsqueeze(-1)
        # t2_mod = t2[:, 1, :]

        # # res = torch.zeros(N * H, L, C, device=t1.device)

        # # compute tp
        # res = torch.zeros(N * H, self.out_basis, C, device=t1.device)
        # k = 0
        # for l in range(self.lmax + 1):
        #     D = self.wigner_D[:, l ** 2 : (l + 1) ** 2, l ** 2 : (l + 1) ** 2]

        #     x1 = D @ t1[:, l ** 2 : (l + 1) ** 2, :]
        #     for out_l in range(max(abs(l - self.y_l), 0), min(self.lo, l + self.y_l) + 1):
        #         D_inv = self.wigner_D[:, out_l ** 2 : (out_l + 1) ** 2, out_l ** 2 : (out_l + 1) ** 2].transpose(1, 2).contiguous()
        #         right = self.CG[l ** 2 : (l + 1) ** 2, 1, out_l ** 2 : (out_l + 1) ** 2]
        #         tmp_res = torch.einsum('nkc,ko->noc', x1, right)
        #         tmp_res = (D_inv @ tmp_res)
        #         res[:, out_l**2:(out_l+1)**2, :] = res[:, out_l**2:(out_l+1)**2, :] + tmp_res * self.weights[k]
        #         k += 1
        return res.view(N, self.out_basis, C)


    def clebsch_gordan(self, l1, l2, l3):
        wigner3j = o3.wigner_3j(l1, l2, l3)
        cg = wigner3j * math.sqrt(2*l3+1)
        return cg
    
    def init_wigner(self, tensor):
        self.device = tensor.device
        self.R = self._init_edge_rot_mat(tensor.view(-1, 3)).detach()
        self.wigner_D = self.RotationToWignerDMatrix(self.R, 0, self.lmax+1).detach()


    def _init_edge_rot_mat(self, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        # if torch.min(edge_vec_0_distance) < 0.0001:


        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
            -1, 1
        )
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
            -1, 1
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
        )
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(
            torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
        )

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
        )
        norm_z = norm_z / (
            torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
        )
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (
            torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
        )

        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

        return edge_rot_mat.detach()
        

    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(
                alpha, beta, torch.zeros_like(alpha)
            ).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(len(alpha), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end
        return wigner.detach()

def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc
    
def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
    
if __name__=="__main__":
    lmax = 2
    l_out = 1
    test = escn_tensor_product(lmax, l_out).cuda()
    x = torch.randn(2000 * 64, 3).cuda()
    x_feat = o3.spherical_harmonics(np.arange(lmax + 1).tolist(), x, False, normalization='component').view(100, 20, 64, (lmax+1)**2)
    x_feat = x_feat.transpose(-1, -2).contiguous()
    y = torch.randn(100, 20, 3).cuda()
    test.init_wigner(y)
    out, _ = test.tensor_product(x_feat, y, weights=None)
    # 
    # R = torch.Tensor(data=[[-0.7887, 0.5774, 0.2113], [0.5774,0.5774, 0.5774], [0.2113, 0.5774, -0.7887]])
    rand_vec = torch.Tensor(data=[1, 4, 6]).cuda()
    R = test._init_edge_rot_mat(rand_vec.view(1, 3))
    wigner_random = test.RotationToWignerDMatrix(R, 0, lmax)
    # print(R @ R.transpose(0, 1))
    R = R.view(1, 1, 3, 3).expand(100, 20, -1, -1)
    wigner_random = wigner_random.view(1, 1, (lmax+1)**2, (lmax+1)**2).expand(100, 20, -1, -1)
    x1 = wigner_random @ x_feat 
    y1 = R @ y.unsqueeze(-1)
    out1, _ = test.tensor_product(x1, y1, weights=None)

    test.init_wigner(y1)
    out2, _ = test.tensor_product(x1, y1, weights=None)

    embed()


