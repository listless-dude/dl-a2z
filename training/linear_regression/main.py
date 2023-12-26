import math
import time
import numpy as np  
import random
import torch
from torch import nn

class HyperParameters:
    """
    Utility class for saving Hyperparameters
    """
    def __init__(self):
        self.hparams = {}
    
    def add_param(self, param_key, param_value):
        self.hparams[param_key] = param_value
        setattr(self, param_key, param_value)

class Trainer:
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model
    
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    
    def prepare_batch(self, batch):
        return batch
    
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        
        if self.val_dataloader is None:
            return 

        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
    
class SGD:
    """Minibatch stochastic gradient descent."""
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad_zero_()

class RegressionData:
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape(-1, 1)) + b + noise
    
    def get_dataloader(self, train):
        if train:
            indices = list(range(0, self.num_train))
            random.shuffle(indices)
        else:
            indices = list(range(self.num_train, self.num_train+self.num_val))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = torch.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        

class LinearRegression(nn.Module):
    """
    Linear Regression model from scratch
    """
    def __init__(self, num_inputs, lr, sigma=0.01):
        """
        w: Weights initialized with mean=0, std deviation=sigma from a normal distribution
        b: Bias initialized with zeros
        """
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        # Vectorized implementation of y = w'X + b
        return X@self.w + self.b
    
    def loss(self, y_pred, y_true):
        # MSE loss
        l = (y_pred - y_true) ** 2 / 2
        return l.mean()
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
model = LinearRegression(2, lr=0.03)
data = 
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)

with torch.no_grad():
    print(f'Error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'Error in estimating b: {data.b - model.b}')