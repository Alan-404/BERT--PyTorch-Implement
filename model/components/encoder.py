import torch
import torch.nn as nn

from model.utils.layer import EncoderLayer
from model.utils.postion import PositionalEncoding

from typing import Union, Callable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = PositionalEncoding()
        self.encoder_layers = [EncoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

        self.embedding_dim = embedding_dim
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        x = self.embedding_layer(x)
        x = x + self.positional_encoding.generate_positon_code(length=x.size(1), embedding_dim=self.embedding_dim)

        for layer in self.encoder_layers:
            x = layer(x, mask, training)

        return x