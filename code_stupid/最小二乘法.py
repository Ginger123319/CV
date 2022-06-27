import torch

x = torch.randn(3, 2)
y = 4 * x

print(x)
print(y)
# 最小二乘法，如果计算正确，输出值就是未知数（参数）的值；
# 注意都是矩阵运算
print("=======result========")
print(torch.linalg.inv(x.T @ x) @ x.T @ y)
