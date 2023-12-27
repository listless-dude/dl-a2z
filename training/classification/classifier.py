"""
Classification model
"""
import sys
sys.path.append("D:\ML-DL-A2Z")
import torch
from torch import nn
from data import FashionMNIST
from utils.trainer import Trainer

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition

def cross_entropy(y_pred, y_true):
    return -torch.log(y_pred[list(range(len(y_pred))), y_true]).mean()

class Classifier(nn.Module):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.lr = lr
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
        
    def parameters(self):
        return [self.W, self.b]
        
    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0])) # Flatten Image
        res = X@self.W + self.b
        return softmax(res)
    
    def loss(self, y_pred, y_true):
        return cross_entropy(y_pred, y_true)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        acc = self.accuracy(self(*batch[:-1]), batch[-1])
        return l, acc
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def accuracy(self, y_pred, y_true, averaged=True):
        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        preds = y_pred.argmax(axis=1).type(y_true.dtype)
        compare = (preds == y_true.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    
traindata = FashionMNIST(batch_size=256)
model = Classifier(num_inputs=784, num_outputs=10, lr=0.01) # inputs=1*28*28, outputs=10 (classes)
trainer = Trainer(max_epochs=10)
trainer.fit(model, traindata)