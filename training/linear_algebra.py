import torch
import time

# Tensor Properties
A = torch.arange(4).reshape(2,-1)
print(A)
B = A.clone()

# Addition
print(A.shape, (A+B).shape)

# Element wise product
print(A*B) # Multiplying by a tensor
print(A*5) # Multiplying by a scalar

# Dot products: Inputs: 1D vectors of same size
x = torch.arange(3)
y = torch.arange(3)

# both yeilds the dot product
print(torch.dot(x, y))
print(torch.sum(x*y))

# Matrix-Vector products: input: (m, n) and (n, 1): Output: (m, 1) 
z = torch.arange(2)
print(A.shape, z.shape)
print(torch.mv(A, z))
print(A@z) # shorthand operator

# Matrix-Matrix multiplication: input: (m, n) and (n, k): Output: (m, k)
print(torch.mm(A, B))
print(torch.matmul(A, B))
print(A@B)

# L2 Norm
p = torch.tensor([3.5, 4.0])
print(torch.sqrt(torch.sum(torch.square(p)))) # Using formula
print(torch.linalg.norm(p, 2)) # Using torch.linalg 

# L1 Norm
print(torch.abs(p).sum())
print(torch.linalg.norm(p, 1))

# Frobenius norm: Matrix norm
X = torch.randn((3,3))
print(torch.linalg.norm(X, ord='fro'))

# Notes
# len() will give you size of 0th axis
print(len(torch.randn(2,3,4)))
