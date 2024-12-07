import torch
import pandas as pd
from wave.model import WAVE
from wave.utils import morgan_fp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_wave_model(wave_model_path, genevae_model_path='../models/gene_vae.pth', device="cpu", **kwargs):
    """
    Load the pretrained WAVE model.

    Parameters:
    - model_path (str): Path to the pretrained model file (.pth).
    - device (str): Device for computation ('cpu' or 'cuda').
    - kwargs: Additional parameters for WAVE initialization.

    Returns:
    - model (WAVE): Loaded WAVE model.
    """
    model = WAVE().to(device)
    model.load_state_dict(torch.load(wave_model_path, map_location=device))
    model.eval()
    return model


def predict_expression(unpert_expr, smiles_list, model, device="cpu"):
    """
    Predict perturbed gene expression.

    Parameters:
    - unpert_expr (numpy.ndarray): Baseline gene expression (shape: n_samples x 978).
    - smiles_list (list): List of SMILES strings for drugs (length: n_samples).
    - model (WAVE): Loaded WAVE model.
    - device (str): Device for computation ('cpu' or 'cuda').

    Returns:
    - pd.DataFrame: Predicted gene expression for each sample.
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Prepare input batch
        unpert_expr_tensor = torch.tensor(unpert_expr, dtype=torch.float32).to(device)
        drug_fps = [torch.tensor(morgan_fp(s), dtype=torch.float32) for s in smiles_list]
        drug_fps_tensor = torch.stack(drug_fps).to(device)

        # Prepare batch dictionary
        batch = {"unpert_expr": unpert_expr_tensor, "drug_fp": drug_fps_tensor}

        # Predict perturbed expression
        predicted_expression = model(batch).cpu().numpy()

    return pd.DataFrame(predicted_expression, columns=[f"Gene_{i+1}" for i in range(predicted_expression.shape[1])])
