"""
Regression with L2 Regularization(Weight Decay)
"""
import torch
import random
from concise import LinearRegression
from trainer import Trainer

class Data:
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        n = num_train + num_val
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = self.X@w + b + noise
    
    def get_dataloader(self, train):
            if train:
                indices = list(range(0, self.num_train))
                # The examples are read in random order
                random.shuffle(indices)
            else:
                indices = list(range(self.num_train, self.num_train+self.num_val))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = torch.tensor(indices[i: i+self.batch_size])
                yield self.X[batch_indices], self.y[batch_indices]
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)

def l2_reg(w):
     return (w ** 2).sum() / 2

class WeightDecayRegression(LinearRegression):
    def __init__(self, lambd, lr):
        super().__init__(lr)
        self.lambd = lambd
     
    def configure_optimizers(self):
        return torch.optim.SGD(params=[self.net.weight, self.net.bias], lr=self.lr, weight_decay=self.lambd)  
   
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = Trainer(max_epochs=10)

def train(lambd):
    model = WeightDecayRegression(lambd=lambd, lr=0.01)
    trainer.fit(model, data)
    print(f'L2 norm of w {l2_reg(model.get_w_b()[0])}')

print("Without Reglarization:")
train(0)
print("With Regularization:")
train(4)