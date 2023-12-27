"""
Loading FashionMNIST dataset.
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

class FashionMNIST():
    """Load FashionMNIST dataset."""
    def __init__(self, batch_size=64, resize=(28,28)):
        self.batch_size = batch_size
        transform = transforms.Compose([transforms.Resize(resize),
                                         transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root='datasets', train=True, transform=transform, download=True
        )
        self.val = torchvision.datasets.FashionMNIST(
            root='datasets', train=False, transform=transform, download=True
        )

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def visualize(self, batch, rows=1, columns=8, labels=[]):
        X, y = batch
        class_names = self.train.classes
        labels = [class_names[int(i)] for i in y]
        figsize = (columns * 1.5, rows * 1.5)
        _, axes = plt.subplots(rows, columns, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, X.squeeze(1))):
            img = img.detach().numpy()
            ax.imshow(img)
            ax.set_title(labels[i])
        plt.show()
