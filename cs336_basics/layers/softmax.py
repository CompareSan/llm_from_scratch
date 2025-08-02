import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: # x (batch, seq_len, d_model)
    x = x - torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)