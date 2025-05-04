import os
import torch
import json
import numpy as np

class Trainer:
    """Class for training the model"""

    def __init__(self, model, epochs, train_dataloader, 
                 val_dataloader, criterion, optimizer, 
                 lr_scheduler, device, model_dir, model_name):
        
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.model.to(self.device)
        self.loss = {"train": [], "val": []}

    def train(self):
        
        for epoch in range(1, self.epochs+1):

            self._train_epoch()
            self._validate_epoch()

            print(f"Epoch: {epoch}/{self.epochs}, \
                    Train Loss: {self.loss['train'][-1]}, \
                    Val Loss: {self.loss['val'][-1]}")
            
            self.lr_scheduler.step() # updating learning rate after each epoch

    
    def _train_epoch(self):
        
        self.model.train()
        running_loss = []


        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # backward pass
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss['train'].append(epoch_loss)
            
        
    def _validate_epoch(self):
        
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)
            self.loss['val'].append(epoch_loss)

    
    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pth")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)