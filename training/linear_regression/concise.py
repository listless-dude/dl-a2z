"""
Pytorch Implementation of Linear Regression.
"""
import torch
from torch import nn
from trainer import Trainer
from regression_data import RegressionData

class LinearRegression(nn.Module):
    """Linear Regression model with pytorch high-level APIs"""
    def __init__(self, lr):
        super().__init__()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        self.lr = lr

    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_pred, y_true):
        loss_fn = nn.MSELoss()
        return loss_fn(y_pred, y_true)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)

model = LinearRegression(lr=0.03)
data = RegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)

w, b = model.get_w_b()

print(f'Error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'Error in estimating b: {data.b - b}')
