import pandas as pd
import scanpy as sc
import anndata as ad
from cmapPy.pandasGEXpress.parse import parse
import numpy as np
from scipy.stats import spearmanr

"""
LINCS L1000 Data Preprocessing Script

This script processes LINCS L1000 treatment data to generate a clean and
filtered dataset for downstream analysis. Key steps include:

1. Parse the GCTX file to extract treatment expression data.
2. Subset the dataset to include only landmark genes.
3. Align instance metadata with expression data.
4. Filter out samples without matching control data.
5. Remove low-quality data based on variance thresholds.
6. Map compound SMILES strings to the dataset using metadata.
7. Exclude invalid entries such as DMSO samples or entries with missing SMILES.
8. Filter data to retain only samples with specific treatment times and doses.
9. Add a "condition" column for grouping by cell line and compound treatment.
10. Remove rows (observations) with negative values in the expression matrix.
11. Aggregate replicates by conditions using MODZ scores.
12. Save the final filtered dataset in H5AD format.
"""

# Paths to input files
treatment_gctx_path = "/slurm/home/yrd/liaolab/public/datasets/l1000/level3_beta_trt_cp_n1805898x12328.gctx"
gene_metadata_path = "/slurm/home/yrd/liaolab/public/datasets/l1000/geneinfo_beta.txt"
instance_metadata_path = "/slurm/home/yrd/liaolab/public/datasets/l1000/instinfo_beta.txt"
compound_metadata_path = "/slurm/home/yrd/liaolab/public/datasets/l1000/compoundinfo_beta.txt"
control_data_path = "level3_ctrl.h5ad"

# Filter criteria
treatment_time = "24 h"
treatment_dose = "10 uM"

# Function to calculate MODZ scores
def calc_modz(matrix):
    """
    Compute MODZ scores for the input matrix.
    Args:
        matrix (np.ndarray): Expression matrix with replicates as rows and genes as columns.
    Returns:
        np.ndarray: Aggregated expression vector.
    """
    matrix = matrix.astype(float)
    if len(matrix) <= 2:
        return np.mean(matrix, axis=0)

    corr_matrix = spearmanr(matrix.T)[0]
    weights = np.sum(corr_matrix, axis=1) - 1
    weights = weights / np.sum(weights)
    weights = weights.reshape((-1, 1))
    modz_result = np.dot(matrix.T, weights).reshape(-1)

    return modz_result


# Step 1: Parse the GCTX file and extract the expression matrix
gctx_data = parse(treatment_gctx_path)
expression_matrix = gctx_data.data_df.T

# Step 2: Load gene metadata and subset to landmark genes
gene_metadata = pd.read_csv(gene_metadata_path, sep="\t").astype(str)
gene_metadata.index = gene_metadata["gene_id"]
landmark_genes_metadata = gene_metadata[gene_metadata["feature_space"] == "landmark"]
landmark_genes = landmark_genes_metadata["gene_id"].tolist()

# Subset expression matrix to include only landmark genes
expression_matrix = expression_matrix[landmark_genes]

# Create AnnData object for the expression matrix
adata_treatment = sc.AnnData(expression_matrix)

# Step 3: Load instance metadata and align with expression data
instance_metadata = pd.read_csv(instance_metadata_path, sep="\t")
instance_metadata.index = instance_metadata["sample_id"]

shared_sample_ids = list(set(adata_treatment.obs.index.tolist()) & set(instance_metadata.index.tolist()))
filtered_instance_metadata = instance_metadata.loc[shared_sample_ids].copy()

# Align instance metadata to AnnData object
adata_treatment.obs = filtered_instance_metadata.reindex(index=adata_treatment.obs.index.tolist()).copy()

# Align gene metadata to AnnData object
adata_treatment.var = landmark_genes_metadata.reindex(index=adata_treatment.var_names.tolist()).copy()
adata_treatment.var.index = adata_treatment.var["gene_symbol"]

# Step 4: Filter out samples without matching control data
control_data = sc.read_h5ad(control_data_path)
valid_control_plates = set(control_data.obs["det_plate"])
adata_treatment = adata_treatment[adata_treatment.obs["det_plate"].isin(valid_control_plates)].copy()

# Step 5: Remove low-variance samples
mean_expression = np.mean(adata_treatment.X, axis=1)
ss_tot = np.sum((adata_treatment.X - mean_expression[:, None]) ** 2, axis=1)
high_variance_mask = ss_tot > 1
adata_treatment = adata_treatment[high_variance_mask].copy()

# Step 6: Map compound SMILES strings to metadata
compound_metadata = pd.read_csv(compound_metadata_path, sep="\t")
smiles_mapping = dict(zip(compound_metadata["pert_id"], compound_metadata["canonical_smiles"]))
adata_treatment.obs["smiles"] = adata_treatment.obs["pert_id"].map(smiles_mapping)

# Step 7: Exclude invalid samples (e.g., DMSO) and entries with missing SMILES
adata_treatment = adata_treatment[adata_treatment.obs["cmap_name"] != "DMSO"].copy()
adata_treatment = adata_treatment[adata_treatment.obs["smiles"].notna()].copy()
adata_treatment = adata_treatment[adata_treatment.obs["smiles"] != ""].copy()
adata_treatment = adata_treatment[adata_treatment.obs["smiles"] != "restricted"].copy()

# Step 8: Subset to specific treatment time and dose
adata_treatment = adata_treatment[
    (adata_treatment.obs["pert_itime"] == treatment_time) &
    (adata_treatment.obs["pert_idose"] == treatment_dose)
].copy()

# Step 9: Add a "condition" column for grouping
adata_treatment.obs["condition"] = (
    adata_treatment.obs["cell_iname"] + "_" + adata_treatment.obs["smiles"]
)

# Step 10: Remove rows with negative values in the expression matrix
negative_rows = (adata_treatment.X < 0).any(axis=1)
if negative_rows.sum() > 0:
    print(f"Removing {negative_rows.sum()} rows with negative values.")
    adata_treatment = adata_treatment[~negative_rows, :]

# Step 11: Aggregate replicates by conditions using MODZ scores
grouped_modz = adata_treatment.to_df().groupby(adata_treatment.obs['condition']).apply(lambda x: calc_modz(x.values))
grouped_modz_index = grouped_modz.index.tolist()

# Aggregate results into new DataFrame
aggregated_matrix = np.vstack(grouped_modz.values)
grouped_modz_df = pd.DataFrame(aggregated_matrix, index=grouped_modz_index)

# Aggregate plate information
plate_groups = adata_treatment.obs.groupby('condition')['det_plate'].apply(list)
grouped_modz_df = grouped_modz_df.loc[plate_groups.index]

# Prepare new metadata
new_obs = pd.DataFrame({
    'det_plate': plate_groups.values
}, index=plate_groups.index)

# Step 12: Create new AnnData object
adata_aggregated = ad.AnnData(X=grouped_modz_df, obs=new_obs)

adata_aggregated.var_names = adata_treatment.var['gene_symbol']
adata_aggregated.obs_names = new_obs.index
adata_aggregated.obs['det_plate'] = new_obs['det_plate'].apply(lambda x: '|'.join(map(str, x)))

# Extract SMILES and cell line information
conditions = adata_aggregated.obs.index.to_series()
adata_aggregated.obs['smiles'] = conditions.str.split("_").str[1]
adata_aggregated.obs['cell'] = conditions.str.split("_").str[0]

# Map MOA information from compound metadata
moa_mapping = dict(zip(compound_metadata['canonical_smiles'], compound_metadata['moa']))
adata_aggregated.obs['moa'] = adata_aggregated.obs['smiles'].map(moa_mapping)
cmap_name_mapping = dict(zip(compound_metadata['canonical_smiles'], compound_metadata['cmap_name']))
adata_aggregated.obs['cmap_name'] = adata_aggregated.obs['smiles'].map(cmap_name_mapping)


# Save the final aggregated dataset
adata_aggregated.write_h5ad("level3_cp.h5ad")

