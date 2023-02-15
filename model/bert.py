import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.utils.mask import generate_padding_mask

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
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.0006)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 0
        self.loss = 0.0
    
    def loss_function(self, outputs: torch.Tensor, targets: torch.Tensor):
        batch_size = targets.size(1)
        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], targets[batch])
        loss = loss/batch_size
        return loss

    def pretrain(self, inputs: torch.Tensor, labels: torch.Tensor, epochs: int = 1, batch_size: int = 1, shuffle: bool = True):
        dataloader = self.build_dataset(inputs=inputs, labels=labels, batch_size=batch_size, shuffle=shuffle)

        for _ in range(epochs):
            self.epoch += 1

            for index, data in enumerate(dataloader):
                self.train_step(inputs=data[0], labels=data[1])

                if index%batch_size == 0:
                    print(f"Epoch: {self.epoch} Batch: {index + 1} Loss: {self.loss/batch_size}")

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs = inputs.to(device)
        labels = labels.to(device)

        padding_mask = generate_padding_mask(inputs)

        # Feed forward Propagation
        outputs = self.model(inputs, padding_mask, True)

        # Back Propagation and apply gradient
        self.loss = self.loss_function(outputs=outputs, targets=labels)
        self.loss.backward()
        self.optimizer.step()

    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool):
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
    

        
