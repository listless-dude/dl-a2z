import torch

def f(x):
    return 5 * x ** 2 - 2 * x + 1

"""
calculate derivatives
using defination of derivative formula
taking h very small, and x = 1, f'(1) is:
"""
h = 0.000001
print((f(1+h) - f(1))/h)

x = torch.arange(4.0, requires_grad=True)
print(x.grad)

# Using f(x) = 2x^2
y = 2 * torch.dot(x, x)
y.backward() # gradient calculation
print(x.grad) # updated gradients

# Verify if f'(x) = 4*x
print(x.grad == 4*x)

# x.grad doesn't reset automatically
x.grad.zero_() # Reset the gradients to zero
print(x.grad)
y = x.sum() # A scalar
y.backward()
print(x.grad) # Should be ones

# backward() computes: J.v' 
x.grad.zero_()
y = x * x

"""
If we don't give gradient of vector of shape of y, we will get error
Or remove gradient param and change y = torch.(x, x)
Either of two yields same gradients
"""
y.backward(gradient=torch.ones(len(y))) # or use y.sum().backward()
print(x.grad)

"""
Graph detach compute
Suppose z = x * y, and y = x * x
For independent computations:
let y = u
then dz/dx = u
and dy/dx = 2x

Else, it would have been,
dz/dx = 3x^2
dy/dx = 2x
"""
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)