import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, 1])  # 访问第一行第二个元素
print(x[:, 1])  # 访问第二列
print(x[1, :])  # 访问第二行
print(x[:, :2])  # 访问前两列