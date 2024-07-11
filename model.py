import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SparseAutoEncoder(nn.Module):

    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-4, sparsity_target=0.05):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        """
        Map the original dimensions to a higher dimensional layer of features.
        Apply relu non-linearity to the linear transformation.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.h_dims),
            nn.ReLU()
        )

        """
        Map back the features to the original input dimensions.
        Apply relu non-linearity to the linear transformation.
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dims, self.in_dims),
            nn.ReLU()
        )

    """
    We pass the original signal through the encoder. Then we pass
    that transformation to the decoder and return both results.
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    """
    This is the sparsity penalty we are going to use (KL)
    """
    def sparsity_penalty(self, encoded):
        # Compute the average activation of each hidden neuron across all the samples in a batch
        # how often do hidden units activate?
        rho_hat = torch.mean(encoded, dim=0)
        # Create a tensor of dimensions equal to the original tensor filled with the target sparsity (0.05 = 5%)
        # This means: we want 5% of the hidden units to be active on average
        rho = torch.ones_like(rho_hat) * self.sparsity_target
        # KL: quantify how one p-distribution diverges from a expected p-distribution
        # Our case: measure how average rho_hat diverges from our target rho
        kl_divergence = F.kl_div(rho_hat.log(), rho, reduction='batchmean')
        return self.sparsity_lambda * kl_divergence
    
    """
    Create a custom loss that combine mean squared error (MSE) loss 
    for reconstruction with the sparsity penalty.
    """
    def loss_function(self, x_hat, x, encoded):
        # x_hat is the reconstructed version of x
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss + sparsity_loss


if __name__ == "__main__":

    in_dims = 500
    h_dims = int(in_dims * 2)
    sparsity_lambda = 1e-4
    sparsity_target = 0.05

    model = SparseAutoEncoder(
        in_dims=in_dims, 
        h_dims=h_dims, 
        sparsity_lambda=sparsity_lambda, 
        sparsity_target=sparsity_target
    )

    summary(model, (in_dims,))

    x = torch.randn(10, in_dims)
    encoded, decoded = model(x)
    loss = model.loss_function(decoded, x, encoded)
    print(loss.item())
    