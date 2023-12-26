"""
Training Module
"""
import torch

class Trainer:
    """The base class for training models with data."""
    def __init__(self, max_epochs, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val

    def prepare_data(self, data):
        self.train_dataloader = list(data.train_dataloader())
        self.val_dataloader = list(data.val_dataloader())
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
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
            train_loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                train_loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        
        if self.val_dataloader is None:
            return 

        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                val_loss = self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

        print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")

