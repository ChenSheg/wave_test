"""
WAVE Package
A package for predicting perturbed gene expression based on baseline expression and drug SMILES.
"""

# Import common functions
from .predict import load_wave_model, predict_expression
from .model import WAVE, GeneVAE
from .train import train_wave
from .utils import morgan_fp
from .utils import time_postfix, seed_everything, loss_fct, log_args
from .load_dataset import PerturbationDataset
from .inference import evaluate, compute_mean_metrics, compute_metrics


# Define version number
__version__ = "0.1.0"
__author__ = "Tianhang Lyu"
__email__ = "lyu202302@126.com"
__license__ = "MIT"
__description__ = "A package for predicting perturbed gene expression."
__url__ = "https://github.com/your-repo/wave"


# Defining a public interface
__all__ = [
    "load_wave_model",
    "predict_expression",
    "WAVE",
    "GeneVAE",
    "train_wave",
]
