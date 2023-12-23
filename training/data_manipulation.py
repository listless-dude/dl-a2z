import torch

# Tensors creation

# 1. arange(n): tensor([0,1,....n-1])
# Default: dtype: int64

x = torch.arange(5)
print(x, x[0].dtype)

x = torch.arange(start=5, end=10, step=0.5, dtype=torch.float32)
print(x, x[0].dtype)

# numel(): Total number of elements in a tensor
# shape: gets the tensor shape
print(x.numel(), x.shape)

# reshape()
# the following prints the same as n = 10, if one dim is 5 then other is 10/5=2
x1 = x.reshape(5,2)
print(x1)
x2 = x.reshape(5,-1)
print(x2)

# flatten x2
print(x.reshape(-1))

# zeros(), ones(), rand(), randn(), tensor()

# rand is uniform distribution [0,1)
# randn is standard normal distribution, u=0, var=1

print(torch.ones(2,3,3))
print(torch.zeros(2,3))
print(torch.rand(2,2))
print(torch.randn(2,2))

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x)
print(x.shape)

# broadcasting

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)
# The following happens while broadcasting
#  a = [[0, 0],  b = [[0, 1]]
#       [1, 1],
#       [2, 2]]

# Conversion
# Numpy <-> Tensor
A = a.numpy()
print(type(A))
B = torch.from_numpy(A)
print(type(B))

# Convert size 1 to python scalar
a = torch.tensor([5])
b = a.item()
print(b, type(b))

# Convert tensor to python list
# If only one item returns a python scalar
a = torch.randn(2,3,3)
b = a.tolist()
print(b, type(a), type(b))

