import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 获取当前脚本的目录和父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 确保父目录在 sys.path 中
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Seed everything for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Define the GeneVAE model
class GeneVAE(nn.Module):
    def __init__(self, input_dim=978, latent_dim=128, hidden_dim1=512, hidden_dim2=256):
        super(GeneVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU()  # Ensure non-negative gene expression
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return mu, logvar, reconstructed


# Dataset class for loading gene expression data
class GeneExpressionDataset(Dataset):
    def __init__(self, adata_path):
        self.adata = sc.read_h5ad(adata_path)
        self.unpert_expr = self.adata.layers['unpert_expr']

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        return self.unpert_expr[idx].squeeze()


# VAE loss function
def vae_loss_function(reconstructed, target, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + kl_loss, recon_loss, kl_loss


# Train function
def train_vae(model, dataloader, epochs, learning_rate, device, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    best_loss = float("inf")  # Track the best loss
    for epoch in range(epochs):
        epoch_loss, recon_loss_epoch, kl_loss_epoch = 0, 0, 0
        for batch in dataloader:
            #batch = torch.tensor(batch).float().to(device)
            batch = batch.clone().detach().float().to(device)
            optimizer.zero_grad()
            mu, logvar, reconstructed = model(batch)
            loss, recon_loss, kl_loss = vae_loss_function(reconstructed, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Recon Loss: {recon_loss_epoch:.4f} | KL Loss: {kl_loss_epoch:.4f}")

    print(f"Best model saved with loss: {best_loss:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the GeneVAE model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input AnnData file")
    parser.add_argument("--save_path", type=str, default="../models/genevae.pth", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed and device
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset and model
    dataset = GeneExpressionDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = GeneVAE().to(device)

    # Train the model
    train_vae(model, dataloader, args.epochs, args.learning_rate, device, args.save_path)


if __name__ == "__main__":
    main()

