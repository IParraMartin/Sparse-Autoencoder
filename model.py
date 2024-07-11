import torch
import torch.nn as nn


class SparseAutoEncoder(nn.Module):

    def __init__(self, in_dims):
        super().__init__()
        self.in_dims = in_dims
        self.features = in_dims * 2

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.features),
            nn.ReLU()
        )



    def forward(self, x):
        pass


if __name__ == "__main__":

    I_DIM = 1_000_000
    FEATURAL_H_DIMS = I_DIM * 2
    



