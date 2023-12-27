"""
Classification model
"""
import torch
from data import FashionMNIST

data = FashionMNIST(resize=(32, 32))
X, y = next(iter(data.train_dataloader()))
print(X.shape, y.shape) # Batch Inputs
print(len(data.train), len(data.val)) # Total train/val sizes
print(data.train.classes) # Class names
print(data.train.class_to_idx) # Image-Label Mapping
batch = next(iter(data.val_dataloader()))
data.visualize(batch)

