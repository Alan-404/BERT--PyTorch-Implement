import torch
import torch.nn as nn

from typing import Union, Callable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClassificationLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.activation = activation
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)

        return x