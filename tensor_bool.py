import time

import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
mask = x > 2  # 生成一个布尔掩码 是不是大于 2
print(mask)   # tensor([False, False,  True,  True,  True])

# 用布尔掩码选出大于 2 的值
filtered_x = x[mask]
print(filtered_x)  # tensor([3, 4, 5])


# 用布尔掩码选出大于 2 的值,并赋值为0
x[mask]=1.2
print(x) # tensor([1, 2, 0, 0, 0])

time.sleep(10)