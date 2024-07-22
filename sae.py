import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchsummary import summary
import os
import certifi


# ensure that Python uses up-to-date CA certificates when making secure connections and prevent errors
os.environ['SSL_CERT_FILE'] = certifi.where()

class SparseAutoencoder(nn.Module):

    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-4, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.xavier_norm_init = xavier_norm_init

        """
        Map the original dimensions to a higher dimensional layer of features.
        Apply relu non-linearity to the linear transformation.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.h_dims),
            nn.ReLU()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight)

        """
        Map back the features to the original input dimensions.
        Apply relu non-linearity to the linear transformation.
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dims, self.in_dims),
            nn.ReLU()
        )

        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.decoder[0].weight)

    """
    We pass the original signal through the encoder. Then we pass
    that transformation to the decoder and return both results.
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    """
    This is the sparsity penalty we are going to use KL divergence
    """
    def sparsity_penalty(self, encoded):
        # Compute the average activation of each hidden neuron across all the samples in a batch
        # how often do hidden units activate?
        rho_hat = torch.mean(encoded, dim=0)
        # Create a tensor of dimensions equal to the original tensor filled with the target sparsity (0.05 = 5%)
        # This means: we want 5% of the hidden units to be active on average
        rho = torch.ones_like(rho_hat) * self.sparsity_target
        # Add a small epsilon value to avoid log(0)
        epsilon = 1e-8
        # KL: quantify how one p-distribution diverges from a expected p-distribution
        # Our case: measure how average rho_hat diverges from our target rho
        kl_divergence = F.kl_div((rho_hat + epsilon).log(), rho + epsilon, reduction='batchmean')
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


def train_model(model, dataloader, n_epochs, optimizer, device):
    model.to(device)
    for epoch in range(n_epochs):
        total_loss = 0
        for data, _ in dataloader:
            # Flatten the img
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            encoded, decoded = model(data)
            loss = model.loss_function(decoded, data, encoded)
            loss.backward()
            # Implement gradient cliping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}/{n_epochs} - Train L: {float(total_loss/len(dataloader)):.4f}')
        print('-'*64)
    
    # save activations of the hidden layer
    sample_data, _ = next(iter(dataloader))
    sample_data = sample_data.view(sample_data.size(0), -1).to(device)
    with torch.no_grad():
        activations, _ = model(sample_data)
    return activations.cpu().numpy(), sample_data.cpu().numpy()


def plot_activations(activations, num_neurons=50, neurons_per_row=10, save_path=None):
    # Calculate the number of rows needed
    num_rows = (num_neurons + neurons_per_row - 1) // neurons_per_row  
    fig, axes = plt.subplots(num_rows, neurons_per_row, figsize=(neurons_per_row * 2, num_rows * 2))
    # Flatten the axes array to make iteration easier
    axes = axes.flatten()  

    for i in range(num_neurons):
        if i >= activations.shape[1]:
            break
        ax = axes[i]
        ax.imshow(activations[:, i].reshape(-1, 1), aspect='auto', cmap='hot')
        ax.set_title(f'Neuron {i+1}', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--in_dims', type=int, default=784)
    parser.add_argument('--h_dims', type=int, default=3136)
    parser.add_argument('--sparsity_lambda', type=float, default=1e-4)
    parser.add_argument('--sparsity_target', type=float, default=0.05)
    parser.add_argument('--xavier_norm_init', type=bool, default=True)
    parser.add_argument('--show_summary', type=bool, default=True)
    parser.add_argument('--download_mnist', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--visualize_activations', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--save_plot', type=bool, default=False)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=args.download_mnist
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    sae_model = SparseAutoencoder(
        in_dims=args.in_dims, 
        h_dims=args.h_dims, 
        sparsity_lambda=args.sparsity_lambda, 
        sparsity_target=args.sparsity_target,
        xavier_norm_init=args.xavier_norm_init
    )

    optimizer = torch.optim.Adam(sae_model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('<>'*64)
    print(f'Using [{str(device).upper()}] for training.\nTo change the device manually, use the argument in the command line.')
    print('<>'*64 + '\n')

    if args.show_summary:
        print('MODEL SUMMARY:')
        summary(sae_model, (args.in_dims,))

    if args.train:
        print('\nTraining...\n')
        activations, sample_data = train_model(
            model=sae_model,
            dataloader=train_dataloader,
            n_epochs=args.n_epochs,
            optimizer=optimizer,
            device=device
        )
        print('-'*64)
        print('Trained!')

        if args.visualize_activations:
            print(f'There are {len(activations[0])} neurons in the hidden layer.')
            plot_activations(activations, num_neurons=40, neurons_per_row=10, save_path=None)

            if args.save_plot:
                plot_save_dir = './files'
                os.makedirs(plot_save_dir, exist_ok=True)
                plot_save_path = os.path.join(plot_save_dir, 'activations.png')
                plot_activations(activations, num_neurons=40, neurons_per_row=10, save_path=plot_save_path)

    # python3 sae.py --train True --n_epochs 100 --visualize_activations True --save_plot True

    if args.save_model:
        model_save_dir = './files'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, 'sae_model.pth')
        torch.save(sae_model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}.')
