import torch

x = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
y = 4 * x

print(x)
print(y)
# 最小二乘法，如果计算正确，输出值就是未知数（参数）的值；
# 注意都是矩阵运算
print("=======inv求逆========")

print(torch.linalg.inv(x.T @ x) @ x.T @ y)
print("=======det求行列式========")
print(torch.linalg.det(x))

# 计算梯度
print("=======对y求backward得到变量x的梯度========")
x = torch.tensor([4.0], requires_grad=True)
y = x * 3 + 2
y.backward()
print(x.grad)

