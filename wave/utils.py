import torch
import numpy as np
import os
import random
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def morgan_fp(smiles, radius=2,n_bits=2048):
    """
    Generate Morgan fingerprints from a given SMILES string.

    Parameters:
    - smiles (str): The SMILES representation of the molecule.
    - radius (int): The radius for the Morgan fingerprint, default is 2.
    - n_bits (int): The number of bits for the fingerprint, default is 2048.

    Returns:
    - np.ndarray: A binary array representing the Morgan fingerprint of the molecule.
    """

    molecule = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
    fingerprint_array = np.array(fingerprint)
    return fingerprint_array

def loss_fct(pred_gene_expression, target_gene_expression):
    loss = F.mse_loss(pred_gene_expression, target_gene_expression, reduction='mean')
    
    return loss




def time_postfix():
    current_time = datetime.now()
    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    TIME_POSTFIX = f"_{year}-{month:02d}-{day:02d}_{hour}-{minute:02d}-{second:02d}"
    return TIME_POSTFIX


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def log_args(args, logger):
    logger.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

