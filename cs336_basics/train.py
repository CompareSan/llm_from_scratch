import torch
import torch.nn as nn
import numpy as np
from cs336_basics.utils.data_loading import get_shuffled_batches
from cs336_basics.utils.save_model import save_checkpoint, load_checkpoint
from cs336_basics.losses.cross_entropy_loss import cross_entropy_loss

class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
    
    def train(self, dataset: np.ndarray, batch_size: int, context_length: int, num_epochs: int) -> list[float]:
        train_losses = []
        self.model.train()
        iteration = 0
        for _ in range(num_epochs):
            batch_generator = get_shuffled_batches(dataset, batch_size, context_length, self.device)
            epoch_loss = 0.0
            
            for inputs, targets in batch_generator:
                loss = self._train_step(inputs, targets)
                epoch_loss += loss
                iteration += 1

                if iteration % 100 == 0:
                    self.save(f'checkpoint_{iteration}.pt', iteration)
            
            epoch_loss /= len(dataset) // batch_size
            train_losses.append(epoch_loss)
        
        return train_losses
        

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = cross_entropy_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval(self, dataset: np.ndarray, batch_size: int, context_length: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            batch_generator = get_shuffled_batches(dataset, batch_size, context_length, self.device)
            for inputs, targets in batch_generator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = cross_entropy_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def save(self, out: str, iteration: int) -> None:
        save_checkpoint(self.model, self.optimizer, iteration, out)
