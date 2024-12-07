import os
import sys
import re
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr

"""
Script to compute MODZ baseline expressions for LINCS L1000 experiments.

Purpose:
This script calculates the MODZ (Median of the Weighted Z-scores) baseline expression data 
for perturbation experiments by matching their corresponding control data. The MODZ baseline 
expressions are then integrated into the perturbation AnnData object as a new layer 
(layers['unpert_expr']).

Steps:
1. Load the perturbation and control AnnData datasets.
2. Iterate over each perturbation experiment, identify its corresponding plates (control data).
3. Compute the MODZ scores for the gene expression matrix of the control data.
4. Add the computed MODZ results as a new layer in the perturbation AnnData object.
5. Save the updated AnnData object to a new H5AD file for further analysis.

Output:
The updated AnnData file ("level3_cp_ctrl.MODZ.h5ad") contains the perturbation data with the 
MODZ baseline expressions stored in layers['unpert_expr'].
"""

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
    if len(matrix) == 2:
        # If only two replicates, return the mean directly
        return np.mean(matrix, axis=0)

    # Compute Spearman correlation matrix
    corr_matrix = spearmanr(matrix.T)[0]

    # Calculate weights based on Spearman correlation
    weights = np.sum(corr_matrix, axis=1) - 1
    weights /= np.sum(weights)
    weights = weights.reshape((-1, 1))

    # Compute weighted average using the weights
    modz_result = np.dot(matrix.T, weights).reshape(-1)
    return modz_result


# Load perturbation and control AnnData objects
perturbation_data = sc.read_h5ad("level3_cp.h5ad")
control_data = sc.read_h5ad("level3_ctrl.h5ad")

# List to store computed MODZ baseline expressions
all_baseline_expressions = []

# Iterate over each perturbation experiment
for idx, det_plate_entry in enumerate(perturbation_data.obs['det_plate']):
    # Split plate information (in case of multiple plates) and filter control data
    related_plates = det_plate_entry.split("|")
    sub_control_data = control_data[control_data.obs['det_plate'].isin(related_plates)]

    # Calculate MODZ baseline expression for the selected control data
    modz_baseline = calc_modz(sub_control_data.X)
    all_baseline_expressions.append(modz_baseline)

    # Print progress every 1000 iterations
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1} perturbation samples...")

# Combine all baseline expressions into a single matrix
baseline_expression_matrix = np.vstack(all_baseline_expressions)

# Add the computed MODZ baseline expression to the perturbation data as a new layer
perturbation_data.layers['unpert_expr'] = baseline_expression_matrix

# Save the updated perturbation data with MODZ baseline expressions
perturbation_data.write_h5ad("level3_cp_ctrl.MODZ.24h.10uM.h5ad")

print("MODZ baseline expressions computed and saved successfully.")
