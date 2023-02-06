import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ResidualConnection(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float, eps: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.to(device)

    def forward(self, tensor: torch.Tensor, pre_tensor: torch.Tensor, training: bool):
        tensor = F.dropout(input=tensor, p=self.dropout_rate, training=training)
        tensor = tensor + pre_tensor
        tensor = self.layer_norm(tensor)

        return tensor

