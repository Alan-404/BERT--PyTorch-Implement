import torch
import numpy as np


class PositionalEncoding:
    def generate_length_code(self, length: int):
        pos = np.arange(length)
        pos = np.expand_dims(pos, axis=1)

        return pos

    def generate_embedding_code(self, embedding_dim: int):
        angles = np.arange(embedding_dim)
        angles[0::2] = angles[1::2]

        angles = 1/(np.power(10000, (angles/embedding_dim)))

        angles = np.expand_dims(angles, axis=0)

        return angles

    def generate_positon_code(self, length: int, embedding_dim: int):
        pos = self.generate_length_code(length)
        angles = self.generate_embedding_code(embedding_dim)

        angles_pos = np.dot(pos, angles)

        angles_pos[0::2] = np.sin(angles_pos[0::2])
        angles_pos[1::2] = np.cos(angles_pos[1::2])

        angles_pos = np.expand_dims(angles_pos, axis=0)

        return torch.tensor(angles_pos, dtype=torch.float32)