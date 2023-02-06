import torch
import torch.nn as nn

from model.utils.attention import MultiHeadAttention
from model.utils.ffn import PositionWiseFeedForward
from model.utils.res import ResidualConnection

from typing import Union, Callable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.ffn = PositionWiseFeedForward(d_ff=d_ff ,embedding_dim=embedding_dim, activation=activation)

        self.residual_connection_1 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(embedding_dim=embedding_dim, dropout_rate=dropout_rate, eps=eps)

        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        # sub layer 1
        q = k = v = x
        attention_output = self.multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_connection_1(attention_output, x, training)

        # sub layer 2
        ffn_output = self.ffn(sub_layer_1)
        sub_layer_2 = self.residual_connection_2(ffn_output, sub_layer_1, training)

        return sub_layer_2
        