import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):

    max_start = len(x) - context_length
    
    starts = np.random.randint(0, max_start, size=batch_size)
    
    inputs = np.stack([x[i : i + context_length] for i in starts])
    targets = np.stack([x[i + 1 : i + 1 + context_length] for i in starts])
    
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_t, targets_t

