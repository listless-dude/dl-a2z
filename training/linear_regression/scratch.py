"""
Linear Regression from scratch.
Implemented:
1. Synthetic data for regression
2. Stochastic Gradient Descent
3. Trainer Class for training and evaluation
4. Linear Regression model with MSE
"""
import torch
from torch import nn
from utils.trainer import Trainer
from regression_data import RegressionData

class SGD:
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegression(nn.Module):
    """
    Linear Regression model from scratch
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        super(LinearRegression, self).__init__()
        """
        w: Weights initialized with mean=0, std deviation=sigma from a normal distribution
        b: Bias initialized with zeros
        """
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.lr = lr

    def forward(self, X):
        # Vectorized implementation of y = w'X + b
        return X@self.w + self.b
    
    def loss(self, y_pred, y_true):
        # MSE loss
        l = (y_pred - y_true) ** 2 / 2
        return l.mean()
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def configure_optimizers(self):
        return SGD(params=[self.w, self.b], lr=self.lr)
    
model = LinearRegression(2, lr=0.05)
data = RegressionData(w=torch.tensor([2,-3.4]), b=4.2)
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)

with torch.no_grad():
    print(f'Error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'Error in estimating b: {data.b - model.b}')