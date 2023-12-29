import torch
from torch import nn
import matplotlib.pyplot as plt

# Cross-Correlation Operation
# Input: (Nh x Nw), Kernel: (Kh x Kw)
# Output: (Nh - Kh + 1) x (Nw - Kw + 1)

def corr2d(input, kernel):
    """2D cross-correlation"""
    Nh, Nw = input.shape
    Kh, Kw = kernel.shape
    # Initialize the shape of Output
    output = torch.zeros((Nh - Kh + 1, Nw - Kw + 1))
    # Iterate over the output to fill with values
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Take slices of input of size == kernel size and add all
            output[i, j] = (input[i:i+Kh, j:j+Kw] * kernel).sum()
    return output

# Testing the above function
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

# Convolutional Layers using corr2d
# Here kernel represents the weights

class Conv2D(nn.Module):
    def __init__(self, kernel_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return corr2d(x, self.weight)

# Object Detection using convolution operation
input = torch.ones((6, 8))
input[:, 2:6] = 0
plt.subplot(2,2,1)
plt.title('input')
plt.axis('off')
plt.imshow(input, cmap='Grays')

# Let's take a kernel of 1x2, which return 0 if output is same
kernel = torch.tensor([[1.0, -1.0]])
output = corr2d(input, kernel)
# For 0 to 1: Edge is black
# For 1 to 0: Edge is white
# For no change: Gray
plt.subplot(2,2,2)
plt.title('output')
plt.axis('off')
plt.imshow(output, cmap='Grays')

# For no change in pixels, we get white image
plt.subplot(2,2,3)
plt.title('No edges')
plt.imshow(corr2d(input.T, kernel), cmap='Grays')
plt.axis('off')
plt.show()

# Train a Kernel, no manual selection like [1, -1]
# Given input and output, we want to estimate kernel

conv2d = Conv2D(kernel_size=(1, 2), bias=False)
lr = 3e-2 # Learning rate
epochs = 10

for i in range(epochs):
    y = conv2d(input)
    loss = (y - output) ** 2 # MSE 
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'Epoch {i}, Loss: {loss.sum():.2f}')

print(conv2d.weight.data) # Very close to original [1, -1]