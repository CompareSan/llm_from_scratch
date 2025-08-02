import torch
import torch.nn as nn
import numpy as np
from cs336_basics.utils.data_loading import get_shuffled_batches
from cs336_basics.utils.save_model import save_checkpoint
from cs336_basics.losses.cross_entropy_loss import cross_entropy_loss
import tqdm

class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cpu', iteration: int = 0):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.iteration = iteration
    
    def train(self, dataset: np.ndarray, batch_size: int, context_len: int, num_epochs: int) -> list[float]:
        train_losses = []
        self.model.train()
        for _ in range(num_epochs):
            batch_generator = get_shuffled_batches(dataset, batch_size, context_len, self.device)
            epoch_loss = 0.0
            pbar = tqdm.tqdm(batch_generator, desc=f"Epoch {_ + 1}/{num_epochs}")
            for inputs, targets in pbar:
                loss = self._train_step(inputs, targets)
                epoch_loss += loss
                self.iteration += 1
                pbar.set_postfix(loss=loss)

                if self.iteration % 1000 == 0:
                    self.save(f'./checkpoints/checkpoint_{self.iteration}.pt')

            epoch_loss /= len(dataset) // batch_size
            train_losses.append(epoch_loss)

        self.save('./checkpoints/final_checkpoint.pt')
        return train_losses
        

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = cross_entropy_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval(self, dataset: np.ndarray, batch_size: int, context_len: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            batch_generator = get_shuffled_batches(dataset, batch_size, context_len, self.device)
            for inputs, targets in batch_generator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = cross_entropy_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def save(self, out: str) -> None:
        save_checkpoint(self.model, self.optimizer, self.iteration, out)
