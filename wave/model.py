import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from wave.utils import morgan_fp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(".")

class GeneVAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for extracting inherent expression patterns 
    of specific cell lines or cell types.

    Parameters:
    - input_dim (int): Dimension of the input gene expression (default: 978).
    - latent_dim (int): Dimension of the latent space (default: 128).
    - hidden_dim1 (int): Dimension of the first hidden layer (default: 512).
    - hidden_dim2 (int): Dimension of the second hidden layer (default: 256).

    Returns:
    - reconstructed (torch.Tensor): Gene expression reconstructed by the decoder, 
                                     with noise and artifacts removed.
    """
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
            nn.ReLU()  # Ensure that the output value is not negative
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
        return reconstructed


def load_model(model_path):
    """
    Load the pretrained Gene VAE model.

    Parameters:
    - model_path (str): Path to the pretrained VAE model.
    - device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
    - model (GeneVAE): Loaded Gene VAE model in evaluation mode.
    """
    model = GeneVAE()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model


def denoise_gene_expression(model, raw_expression):
    """
    Denoise raw gene expression values using a trained VAE model.

    Parameters:
    - model (GeneVAE): Pretrained Gene VAE model.
    - raw_expression (torch.Tensor): Raw gene expression tensor.
    - device (str): Device to perform computation on.

    Returns:
    - torch.Tensor: Denoised gene expression tensor.
    """
    raw_tensor = raw_expression
    with torch.no_grad():  # Disable gradient computation
        denoised_expression = model(raw_tensor)
    return denoised_expression


class FingerprintNN(nn.Module):
    """
    Drug fingerprint feature extracting module.
    
    Parameters:
    - input_dim (int): Dimension of the input drug fingerprint. Default is 2048.
    - hidden_dim (int): Dimension of the hidden layer. Default is 256.
    - output_dim (int): Dimension of the output drug embedding. Default is 2048.

    Returns:
    - drug_embed (Tensor): The extracted drug embedding with shape `(batch_size, output_dim)`.
    """
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=2048):
        super(FingerprintNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        drug_embed = self.fc2(x)
        return drug_embed


class GeneDrugFusionPerturbation(nn.Module):
    """
    Gene and drug fusion module using Transformer-based architecture.

    Parameters:
    - drug_input_dim (int): Dimension of the drug embedding input.
    - gene_input_dim (int): Dimension of the gene embedding input.
    - fusion_dim (int): Dimension of the fusion space after projecting gene and drug embeddings. Default is 1024.
    - output_dim (int): Dimension of the output gene expression. Default is 978.
    - num_heads (int): Number of attention heads in the Transformer Decoder. Default is 8.
    - num_layers (int): Number of Transformer Decoder layers. Default is 2.
    """
    def __init__(self, drug_input_dim, gene_input_dim, fusion_dim=1024, output_dim=978, num_heads=8, num_layers=2):
        super(GeneDrugFusionPerturbation, self).__init__()
        self.gene_proj = nn.Linear(gene_input_dim, fusion_dim)
        self.drug_proj = nn.Linear(drug_input_dim, fusion_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=fusion_dim // num_heads, nhead=num_heads, dim_feedforward=2048)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(fusion_dim, output_dim)

    def forward(self, gene_embed, drug_embed):
        gene_embed = self.gene_proj(gene_embed)
        drug_embed = self.drug_proj(drug_embed)

        gene_embed_seq = gene_embed.view(gene_embed.size(0), 8, -1).permute(1, 0, 2)
        drug_embed_seq = drug_embed.view(drug_embed.size(0), 8, -1).permute(1, 0, 2)

        fusion_embed = self.transformer_decoder(tgt=gene_embed_seq, memory=drug_embed_seq)
        fusion_embed = fusion_embed.permute(1, 0, 2).contiguous().view(fusion_embed.size(1), -1)

        return self.output_proj(fusion_embed)


class WAVE(nn.Module):
    """
    A complete model, including a gene VAE module, a drug processing module, and a gene-drug fusion module.

    Parameters:
    - genevae_model_path (str): Path to the pretrained Gene VAE model file.
    - device (str): Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        genevae_model_path,
        gene_input_dim=978,
        drug_input_dim=2048,
        fusion_dim=512,
        gene_output_dim=978,
        drug_hidden_dim=256,
        drug_output_dim=2048,
        num_heads=8,
        num_layers=2,
    ):
        super(WAVE, self).__init__()
        
        self.fpnn = FingerprintNN(
            input_dim=drug_input_dim, hidden_dim=drug_hidden_dim, output_dim=drug_output_dim
        )
        self.gene_drug_fusion = GeneDrugFusionPerturbation(
            drug_input_dim=drug_output_dim,
            gene_input_dim=gene_input_dim,
            fusion_dim=fusion_dim,
            output_dim=gene_output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.genevae_model = load_model(genevae_model_path)
        self.relu = nn.ReLU()

    def forward(self, batch):
        """
        Forward pass for WAVE model.

        Parameters:
        - batch (dict): Batch of data containing:
            - 'unpert_expr': Unperturbed gene expression tensor.
            - 'smiles': List of SMILES strings for drugs.

        Returns:
        - torch.Tensor: Predicted perturbed gene expression.
        """
        
        batch['unpert_expr_raw'] = batch['unpert_expr']
        batch['unpert_expr'] = denoise_gene_expression(self.genevae_model, batch['unpert_expr'])

        drug_embs = self.fpnn(batch["drug_fp"])
        

        # Fuse gene and drug embeddings
        expression_shift = self.gene_drug_fusion(batch["unpert_expr"], drug_embs)

        # Compute final perturbed gene expression
        perturbated_gene_expression = expression_shift + batch["unpert_expr_raw"]
        return perturbated_gene_expression
