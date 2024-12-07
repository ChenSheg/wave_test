import scanpy as sc
from torch.utils.data import Dataset
from wave.utils import morgan_fp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class PerturbationDataset(Dataset):
    """
    Dataset class for loading and processing bulk and single-cell gene expression datasets,
    including unperturbed and perturbed gene expression data.

    This class reads data from a given `.h5ad` file, providing structured access
    to unperturbed gene expression, perturbed gene expression, and associated metadata.

    Parameters:
    ----------
    - adata_path (str): 
        Path to the `.h5ad` file containing:
        - `unpert_expr` in the `.layers` attribute (baseline gene expression).
        - `X` in the main matrix (perturbed gene expression).
        - Metadata in `.obs` (e.g., SMILES structure of drugs).

    Attributes:
    ----------
    - adata (AnnData): 
        The AnnData object loaded from the `.h5ad` file.
    - unpert_expr (numpy.ndarray): 
        Baseline (unperturbed) gene expression matrix from `adata.layers['unpert_expr']`.
    - pert_expr (numpy.ndarray): 
        Perturbed gene expression matrix from `adata.X`.
    - obs (pd.DataFrame): 
        Metadata (e.g., drug SMILES structure) from `adata.obs`.

    Returns:
    ----------
    This class provides data samples containing:
        - `unpert_expr`: Baseline gene expression vector.
        - `pert_expr`: Perturbed gene expression vector.
        - `smiles`: SMILES structure for the drug perturbation.
        - `drug_fp`: Molecular fingerprint vector of the drug (computed from SMILES).
        - `index`: Index of the sample.
    """

    def __init__(self, adata_path):
        """
        Initialize the dataset by loading the `.h5ad` file.

        Parameters:
        ----------
        - adata_path (str): 
            Path to the `.h5ad` file containing gene expression data 
            and relevant metadata.
        """
        self.adata = sc.read_h5ad(adata_path)
        self.unpert_expr = self.adata.layers['unpert_expr']
        self.pert_expr = self.adata.X
        self.obs = self.adata.obs

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        ----------
        - int: The number of samples (rows) in the dataset.
        """
        return len(self.adata)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset based on its index.

        Parameters:
        ----------
        - idx (int): 
            The index of the sample to retrieve.

        Returns:
        ----------
        - dict: A dictionary containing:
            - `unpert_expr` (numpy.ndarray): 
                Baseline gene expression vector for the sample.
            - `pert_expr` (numpy.ndarray): 
                Perturbed gene expression vector for the sample.
            - `smiles` (str): 
                SMILES structure of the drug associated with the sample.
            - `drug_fp` (numpy.ndarray): 
                Morgan fingerprint of the drug, computed from its SMILES string.
            - `index` (int): 
                Index of the sample in the dataset.
        """
        index = idx
        unpert_expr_vec = self.unpert_expr[idx].squeeze()
        pert_expr_vec = self.pert_expr[idx].squeeze()
        smiles = self.obs['smiles'].iloc[idx]
        drug_fp = morgan_fp(smiles)


        #print('drug_fp',type(drug_fp))
        #print('unpert_expr_vec',type(unpert_expr_vec))

        return {
            'unpert_expr': unpert_expr_vec,
            'pert_expr': pert_expr_vec,
            'smiles': smiles,
            'drug_fp': drug_fp,
            'index': index
        }
