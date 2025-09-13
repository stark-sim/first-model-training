import torch

x = torch.tensor([[1,2,3],[4,5,6]])
#扩展第0维
x_0 = x.unsqueeze(0)
print(x_0.shape,x_0)
#扩展第1维
x_1 = x.unsqueeze(1)
print(x_1.shape,x_1)
#扩展第2维
x_2 = x.unsqueeze(2)
print(x_2.shape,x_2)

# 你可以使用tensor的squeeze方法来缩减tensor的大小为1的维度。
# 你可以指定需要缩减的维度索引，如果不指定，则会缩减所有大小为1的维度。
x = torch.ones((1,1,3))
print(x.shape, x)
y = x.squeeze(dim=0)
print(y.shape, y)
z = x.squeeze()
print(z.shape, z)