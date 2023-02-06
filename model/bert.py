import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Callable

from model.components.encoder import Encoder
from model.components.classification import ClassificationLayer

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class BERTModel(nn.Module):
    def __init__(self, 
                vocab_size: int, 
                n: int, 
                embedding_dim: int, 
                heads: int, 
                d_ff: int, 
                dropout_rate: float, 
                eps: float, 
                activation_ff: Union[str, Callable[[torch.Tensor], torch.Tensor]],
                activation_cls: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation_ff)
        self.classification_layer = ClassificationLayer(vocab_size=vocab_size, embedding_dim=embedding_dim, eps=eps, activation=activation_cls)
        
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        x = self.encoder(x, mask, training)
        x = self.classification_layer(x)

        return x


class BERT:
    def __init__(self, 
                vocab_size: int, 
                n: int = 12, 
                embedding_dim: int = 768, 
                heads: int = 12, 
                d_ff: int = 2048, 
                dropout_rate: float = 0.1, 
                eps: float = 0.1, 
                activation_ff: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                activation_cls: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu):
        self.model = BERTModel(vocab_size=vocab_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff,dropout_rate=dropout_rate, eps=eps, activation_ff=activation_ff, activation_cls=activation_cls)
