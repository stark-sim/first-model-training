import torch
import numpy as np

# # 1D Tensor
# t1 = torch.tensor([1, 2, 3], dtype=torch.float64)
# print(t1)
#
# # 2D Tensor
# t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(t2)
#
# # 3D Tensor
# t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(t3)
#
# # 从 NumPy 创建 Tensor
# arr = np.array([1, 2, 3])
# t_np = torch.tensor(arr)
# print(t_np)

# shape = (2, 3) # row col 2行 3列
# rand_tensor = torch.rand(shape) # 生成一个从 [0, 1] 均匀抽样的 tensor
# print(rand_tensor)
# randn_tensor = torch.randn(shape) # 生成一个从标准正态分布抽样的 tensor
# print(randn_tensor)
# ones_tensor = torch.ones(shape)
# print(ones_tensor)
# zeros_tensor = torch.zeros(shape)
# print(zeros_tensor)
# twos_tensor = torch.full(shape, 2)
# print(twos_tensor)

tensor = torch.rand(3, 4)
print(tensor)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(f"RequiresGrad of tensor: {tensor.requires_grad}")

x = tensor.reshape(4, 3) # 保持顺序
print(x)
y = x.permute(1, 0) # 转置，不保持顺序。0 和 1 维互换
print(y)

