from e3nn import o3
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange #, einsum
import time
from IPython import embed

class sh_tensor_product:
	def __init__(self, max_degree):
		self.lmax = max_degree
		self.y_l = 1 # edge_vec
		self.irreps_sh = o3.Irreps.spherical_harmonics(lmax = self.lmax, p = 1)

		self.CG = dict()
		self.D = dict()
		for l in range(self.lmax+1):
			for out_l in range(abs(l - self.y_l), l + self.y_l + 1):
				self.CG[(l, out_l)] = self.clebsch_gordan(l, self.y_l, out_l)

	def generate_sh(self, x):
		assert(x.shape[-1]==3)
		sh = o3.spherical_harmonics(self.irreps_sh, x, False)
		return sh

	def clebsch_gordan(self, l1, l2, l3):
		wigner3j = o3.wigner_3j(l1, l2, l3)
		cg = wigner3j * np.sqrt(2*l3+1)
		return cg

	def l2norm(self, t):
		return F.normalize(t, dim = -1)

	def rot_x_to_y_direction(self, y, z = torch.tensor([0., 1., 0.])):
		z = z.cuda()
		dtype = y.dtype
		# n = x.shape[-1]
		I = torch.eye(3, dtype = dtype).cuda()
		if torch.allclose(y, z, atol = 1e-8):
			return I
		y, z = y.double(), z.double()
		y, z = map(self.l2norm, (y, z))
		yz = rearrange(y + z, '... n -> ... n 1')
		yz_t = rearrange(yz, '... n 1 -> ... 1 n')
		R = 2 * (yz @ yz_t) / (yz_t @ yz).clamp(min = 1e-8) - I
		return R.type(dtype)
	
	def init_wigner(self, tensor):
		self.R = self.rot_x_to_y_direction(tensor)
		self.angles = o3.matrix_to_angles(self.R.to(device='cpu'))
		for l in range(self.lmax + 1):
			self.D[l] = o3.wigner_D(l, self.angles[0], self.angles[1], self.angles[2]).to(device=tensor.device)
	
	def escn_tensor_product(self, t1, t2, weights):
		assert(t2.shape[-1]==3)
		N, H, L, C = t1.shape
		t1 = t1.view(-1, L, C)
		t2 = t2.view(-1, 3)
		# t1 = self.generate_sh(tensor1)
		t2 = self.R @ t2.unsqueeze(-1) # rotate
		t2_mod = t2[:, 1, :]	# tensor2模长
		# res = []
		res = torch.zeros(N * H, L, C, device=t1.device)
		k = 0
		# for l in range(self.lmax + self.y_l + 1):
		# 	D[l] = o3.wigner_D(l, angles[0], angles[1], angles[2]).cuda()
		for l in range(self.lmax+1):
			# D1 = o3.wigner_D(l, angles[0], angles[1], angles[2])
			D1 = D[l]
			x_1 = D1 @ t1[:, l**2:(l+1)**2]
			for out_l in range(abs(l - self.y_l), l + 1):
				# print(l,out_l)
				# D2 = o3.wigner_D(out_l, angles[0], angles[1], angles[2])
				D2 = D[out_l]
				# right = einsum(self.CG[(l,out_l)], t2, 'i j k,j -> ... i k')
				# tmp_res = (D2 @ (x_1 @ right))
				right = self.CG[(l, out_l)][:,1,:].cuda()
				# tmp_res = einsum(D2, x_1, right, 'i k,j,j k -> ... i') * t2_mod
				tmp_res = torch.einsum('nkc,ko->noc', x_1, right)
				tmp_res = (D2 @ tmp_res) * weights[k]# * t2_mod
				res[:, out_l**2:(out_l+1)**2, :] += tmp_res
				k += 1
				# res.append(tmp_res)
		# res.sort(key=lambda x:len(x))
		# sort_res = torch.concat(res, dim = 1)
		# embed()
		return res.view(N, H, L, C), t2_mod

	def tensor_product(self, tensor1, tensor2):
		assert(tensor2.shape[-1]==3)
		t1 = self.generate_sh(tensor1)
		t2 = tensor2
		res = []
		for l in range(self.lmax+1):
			for out_l in range(abs(l - self.y_l), l + self.y_l + 1):
				tmp_res = einsum(t1[l**2:(l+1)**2], self.CG[(l,out_l)], t2, 'i,i j k,j -> ... k')
				res.append(tmp_res)
		res.sort(key = lambda x: len(x))
		sort_res = torch.concat(res, dim = 0)
		return sort_res

	def e3nn_tensor_product(self, tensor1, tensor2):
		irr1 = []
		for l in range(self.lmax+1):
			irr1.append((1,(l,1)))
		irreps1 = o3.Irreps(irr1)
		irreps2 = o3.Irreps([(1,(self.y_l,1))])
		tp = o3.FullTensorProduct(irreps1, irreps2)
		tensor1 = self.generate_sh(tensor1)
		res = tp(tensor1, tensor2)
		return res

	def is_result_right(self, tensor1, tensor2):
		return torch.allclose(self.escn_tensor_product(tensor1, tensor2), self.e3nn_tensor_product(tensor1, tensor2))
