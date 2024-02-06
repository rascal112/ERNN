import torch
from e3nn import o3
import numpy as np
import torch.nn.functional as F
from einops import einsum, rearrange
import time

# irr1 * irr2 -> irr3
# 单个的CG_tensor_product，简单的乘法
class single_tensor_product:
	def __init__(self, irreps_l1, irreps_l2, irreps_out):
		self.l1 = irreps_l1
		self.l2 = irreps_l2
		self.l3 = irreps_out

		self.CG = self.clebsch_gordan(self.l1, self.l2, self.l3)

	def clebsch_gordan(self, l1, l2, l3):
		wigner3j = o3.wigner_3j(l1, l2, l3)
		cg = wigner3j * np.sqrt(2*l3+1)
		return cg

	def tensor_product(self, tensor1, tensor2):
		res = einsum(tensor1, self.CG, tensor2, 'i,i j k,j -> ... k')
		return res

	def e3nn_tensor_product(self, tensor1, tensor2):
		irreps1 = o3.Irreps([(1,(self.l1,1))])
		irreps2 = o3.Irreps([(1,(self.l2,1))])
		irr3 = [o3.Irrep(self.l3,1)]
		tp = o3.FullTensorProduct(irreps1, irreps2, irr3)
		res = tp(tensor1, tensor2)
		return res

	def is_result_right(self, tensor1, tensor2):
		return torch.allclose(self.tensor_product(tensor1, tensor2), self.e3nn_tensor_product(tensor1, tensor2))

# all path的tensor product
# 和e3nn.o3.FullTensorProduct完全一致
# 可用维度矩阵指定输出的path，如full_tensor_product(2, 1, [1, 2, 3])
# 默认为全path
# 依赖于上面的single_tensor_product
class full_tensor_product:
	def __init__(self, irreps_l1, irreps_l2, irreps_out = None):
		self.l1 = irreps_l1
		self.l2 = irreps_l2
		if irreps_out:
			self.l3_list = irreps_out
		else:
			self.l3_list = [l for l in range(abs(self.l1-self.l2), self.l1+self.l2+1)]
		for l in self.l3_list:
			assert((l>=abs(self.l1-self.l2)) and (l<=self.l1+self.l2))

		self.tp_list = []
		for l3 in self.l3_list:
			self.tp_list.append(single_tensor_product(self.l1, self.l2, l3))

	def tensor_product(self, tensor1, tensor2):
		res = torch.tensor([])
		for tp in self.tp_list:
			tmp_res = tp.tensor_product(tensor1, tensor2)
			res = torch.concat((res, tmp_res), dim = 0)
		return res

	def e3nn_tensor_product(self, tensor1, tensor2):
		irreps1 = o3.Irreps([(1,(self.l1,1))])
		irreps2 = o3.Irreps([(1,(self.l2,1))])
		irr3 = []
		for l3 in self.l3_list:
			irr3.append(o3.Irrep(l3, 1))
		tp = o3.FullTensorProduct(irreps1, irreps2, irr3)
		res = tp(tensor1, tensor2)
		return res

	def is_result_right(self, tensor1, tensor2):
		return torch.allclose(self.tensor_product(tensor1, tensor2), self.e3nn_tensor_product(tensor1, tensor2))


# ---------------------------------------------------------------------------------------

# 不依赖于上面的两个类
# 用于spherical_harmonics的tensor product，输入max_l，会将tensor1变为(max_l)^2的维度向量
class sh_tensor_product:
	def __init__(self, max_degree):
		self.lmax = max_degree
		self.y_l = 1 # edge_vec
		self.irreps_sh = o3.Irreps.spherical_harmonics(lmax = self.lmax, p = 1)

		self.CG = dict()
		for l in range(self.lmax+1):
			for out_l in range(abs(l - self.y_l), l + self.y_l + 1):
				self.CG[(l,out_l)] = self.clebsch_gordan(l, self.y_l, out_l)

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
		dtype = y.dtype
		n = x.shape[-1]
		I = torch.eye(n,dtype = dtype)
		if torch.allclose(y, z, atol = 1e-8):
			return I
		y, z = y.double(), z.double()
		y, z = map(self.l2norm, (y, z))
		yz = rearrange(y + z, '... n -> ... n 1')
		yz_t = rearrange(yz, '... n 1 -> ... 1 n')
		R = 2 * (yz @ yz_t) / (yz_t @ yz).clamp(min = 1e-8) - I
		return R.type(dtype)

	def escn_tensor_product(self, tensor1, tensor2):
		assert(tensor2.shape[-1]==3)
		R = self.rot_x_to_y_direction(tensor2)
		angles = o3.matrix_to_angles(R)
		t1 = self.generate_sh(tensor1)
		t2 = R @ tensor2 # rotate
		t2_mod = t2[1]	# tensor2模长
		res = []
		D = dict()
		for l in range(self.lmax + self.y_l + 1):
			D[l] = o3.wigner_D(l, angles[0], angles[1], angles[2])
		for l in range(self.lmax+1):
			# D1 = o3.wigner_D(l, angles[0], angles[1], angles[2])
			D1 = D[l]
			x_1 = D1 @ t1[l**2:(l+1)**2]
			for out_l in range(abs(l - self.y_l), l + self.y_l + 1):
				# print(l,out_l)
				# D2 = o3.wigner_D(out_l, angles[0], angles[1], angles[2])
				D2 = D[out_l]
				# right = einsum(self.CG[(l,out_l)], t2, 'i j k,j -> ... i k')
				# tmp_res = (D2 @ (x_1 @ right))

				right = self.CG[(l,out_l)][:,1,:]
				# tmp_res = einsum(D2, x_1, right, 'i k,j,j k -> ... i') * t2_mod
				tmp_res = (D2 @ (x_1 @ right)) * t2_mod
				res.append(tmp_res)
		res.sort(key=lambda x:len(x))
		sort_res = torch.concat(res, dim = 0)
		return sort_res

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

tp = sh_tensor_product(3)
x = torch.tensor([3.0, 1.0, 4.0])
print(x.shape)
y = torch.tensor([1.0, 5.0, 2.0])


# tp  = full_tensor_product(3,1,[2,3,4])
# x = torch.tensor([1.0, 5.0, 2.0, 4.0, 1.0, 2.0, 4.0])
# y = torch.tensor([1.0, 2.0, 3.0])

# tp = single_tensor_product(3,2,2)
# x = torch.tensor([1.0, 5.0, 1.0, 0.0, 3.0, 7.0, 4.0])
# y = torch.tensor([2.0, 3.0, 1.0, 1.0, 4.0])

time1 = time.time()
for i in range(10):
	res1 = tp.tensor_product(x,y)
time2 = time.time()
for i in range(10):
	res2 = tp.e3nn_tensor_product(x,y)
time3 = time.time()
for i in range(10):
	res3 = tp.escn_tensor_product(x,y)
time4 = time.time()
print(res1)
print(res2)
print(res3)

# print(time2-time1)
# print(time3-time2)
# print(time4-time3)
# print(tp.is_result_right(x,y))

