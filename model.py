import torch
import torch.nn as nn


class SparseAutoEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def encoder(self):
        pass

    def h_layer(self):
        pass

    def decoder(self):
        pass


if __name__ == "__main__":
    pass
