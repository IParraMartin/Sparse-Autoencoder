import torch
import torch.nn as nn


class SparseAutoEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.encoder = nn.Sequential(
            
        )

    def forward(self, x):
        pass


if __name__ == "__main__":

    I_DIM = 1_000_000
    FEATURAL_H_DIMS = I_DIM * 2
    



