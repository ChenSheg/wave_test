import pandas as pd
import scanpy as sc
import anndata as ad
from cmapPy.pandasGEXpress.parse import parse

"""
This script processes LINCS L1000 level 3 control data and converts it into a structured AnnData format 
for downstream analysis. 

Key functionalities of the script:
1. Parse the GCTX file to extract the control expression matrix.
2. Identify landmark genes using the `geneinfo_beta.txt` file and subset the expression matrix to include only landmark genes.
3. Match the sample IDs from the expression matrix with the metadata in `instinfo_beta.txt`, ensuring proper alignment of experimental information.
4. Reorder metadata and gene information to match the structure of the expression matrix.
5. Save the processed data as an AnnData `.h5ad` file for easy loading and analysis in Python.

Output:
The script saves the processed control data in an AnnData object named `level3_ctrl.h5ad`.
"""


# Path to the LINCS L1000 GCTX file containing level 3 control data
input_file = "/slurm/home/yrd/liaolab/public/datasets/l1000/level3_beta_ctl_n188708x12328.gctx"

# Parse the GCTX file to extract expression matrix
gctx_data = parse(input_file)
expression_matrix = gctx_data.data_df.T

# Read gene metadata to identify landmark genes
gene_metadata = pd.read_csv("/slurm/home/yrd/liaolab/public/datasets/l1000/geneinfo_beta.txt", sep="\t").astype(str)
gene_metadata.index = gene_metadata['gene_id']
landmark_genes_metadata = gene_metadata[gene_metadata['feature_space'] == 'landmark']

# Extract landmark genes and subset the expression matrix
landmark_genes = landmark_genes_metadata['gene_id'].tolist()
expression_matrix = expression_matrix[landmark_genes]

# Create AnnData object from the expression matrix
adata = sc.AnnData(expression_matrix)

# Read instance information metadata
# The `instinfo_beta.txt` file contains experimental metadata such as sample ID, experimental conditions, and cell lines.
instance_metadata = pd.read_csv("/slurm/home/yrd/liaolab/public/datasets/l1000/instinfo_beta.txt", sep="\t")
instance_metadata.index = instance_metadata['sample_id']

# Subset instance metadata using sample IDs present in the GCTX file
# Ensure that control samples have corresponding metadata
matching_sample_ids = list(set(adata.obs.index.tolist()) & set(instance_metadata.index.tolist()))
filtered_instance_metadata = instance_metadata.loc[matching_sample_ids]

# Reorder instance metadata to match the order of sample IDs in the AnnData object
sorted_instance_metadata = filtered_instance_metadata.reindex(index=adata.obs.index.tolist())
adata.obs = sorted_instance_metadata
#adata.obs.index = sorted_instance_metadata.index

# Reorder landmark gene metadata to match the order of gene IDs in the AnnData object
sorted_landmark_gene_metadata = landmark_genes_metadata.reindex(index=adata.var_names.tolist())
adata.var = sorted_landmark_gene_metadata
adata.var.index = sorted_landmark_gene_metadata['ensembl_id']


# Remove rows (observations) with negative values in the expression matrix
# Check for rows with any negative values
negative_rows = (adata.X < 0).any(axis=1)

# Filter out rows with negative values
if negative_rows.sum() > 0:
    print(f"Removing {negative_rows.sum()} rows with negative values.")
    adata = adata[~negative_rows, :]


# Save the processed AnnData object as an h5ad file
adata.write_h5ad("level3_ctrl.h5ad")

