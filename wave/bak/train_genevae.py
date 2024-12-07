import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
sys.path.append(".")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 定义 VAE 模型
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
            nn.ReLU()  # 确保降噪后的基因表达为非负
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


class GeneExpressionDataset(Dataset):
    def __init__(self, adata_path):
        self.adata = sc.read_h5ad(adata_path)
        self.unpert_expr = self.adata.layers['unpert_expr']

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        unpert_expr_vec = self.unpert_expr[idx].squeeze()
        return unpert_expr_vec


def vae_loss_function(reconstructed, target, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + kl_loss, recon_loss, kl_loss


def train_vae(model, dataloader, epochs=100, learning_rate=0.001, save_path="../models/genevae.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    best_loss = float("inf")  # Initialize best loss to infinity
    best_model_state = None  # To store the state dict of the best model
    best_epoch = -1  # To store the epoch number of the best model

    for epoch in range(epochs):
        epoch_loss = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            mu, logvar, reconstructed = model(batch)
            loss, recon_loss, kl_loss = vae_loss_function(reconstructed, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            kl_loss_epoch += kl_loss.item()

        avg_loss = epoch_loss / len(dataloader)  # Average loss for the epoch

        # Check if the current model is the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1  # Record the best epoch (1-indexed)

        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.4f} | Recon Loss: {recon_loss_epoch:.4f} | KL Loss: {kl_loss_epoch:.4f}")

    # Save the best model after training
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Best model saved with total loss: {best_loss:.4f} at epoch {best_epoch} in {save_path}")

    return model


if __name__ == "__main__":
    seed_everything(42)
    adata_path = "/slurm/home/yrd/liaolab/lvtianhang/Work/Single_drug_perturbation_v3/00.dataset/level3_MODZ_raw/level3_cp_ctrl.h5ad"
    
    dataset = GeneExpressionDataset(adata_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneVAE().to(device)

    trained_model = train_vae(model, dataloader, epochs=300, learning_rate=0.001)

    print("VAE training completed.")
