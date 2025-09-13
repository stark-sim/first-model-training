import torch

a = torch.ones((2,3))
b = torch.ones((2,3))

print(a + b)  # 加法
print(a - b)  # 减法
print(a * b)  # 逐元素乘法
print(a / b)  # 逐元素除法
print(a @ b.t())  # 矩阵乘法 b.t() 是转置

import torch

t = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

mean = t.mean()
print("mean:", mean)

mean = t.mean(dim=0)
print(f"Shape of dim 0: {mean.shape}, mean on dim 0:", mean)

mean = t.mean(dim=0, keepdim=True)
print("keepdim:", mean)

mean = t.mean(dim=1)
print(f"Shape of dim0: {mean.shape}, mean on dim 1:", mean)