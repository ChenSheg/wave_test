import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.set_printoptions(threshold=np.inf)

def evaluate(data_loader, model):
    """
    Evaluate the model in inference mode with the specified data loader (val or test).

    Parameters:
    - data_loader (DataLoader): DataLoader for evaluation.
    - model (torch.nn.Module): Model to evaluate.

    Returns:
    - dict: A dictionary containing predictions, ground truths, 
            unperturbed expressions, and expression changes.
    """
    model.eval()  # Set the model to evaluation mode

    # Get the model's device dynamically
    model_device = next(model.parameters()).device

    results = {"prediction": [], "ground_truth": [], "unpert": []}

    # Iterate through the data loader
    with torch.no_grad():
        for batch in data_loader:
            # Ensure all batch fields are moved to the correct device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(model_device).float()

            # Predict perturbed expression
            predictions = model(batch)
            # Store results
            results["prediction"].append(predictions)
            results["ground_truth"].append(batch['pert_expr'])
            results["unpert"].append(batch['unpert_expr'])

    # Concatenate all results and move to CPU for further processing
    for key in results:
        results[key] = torch.cat(results[key]).cpu().numpy()

    # Calculate expression changes
    results["pred_expr_change"] = results["prediction"] - results["unpert"]
    results["true_expr_change"] = results["ground_truth"] - results["unpert"]

    # Add pseudocount of 1 for log2(FC) calculation to avoid log(0)
    epsilon = 1e-10
    results["prediction"] = np.maximum(results["prediction"], epsilon)
    results["unpert"] = np.maximum(results["unpert"], epsilon)
    results["ground_truth"] = np.maximum(results["ground_truth"], epsilon)

    results["pred_log2fc"] = np.log2((results["prediction"] + epsilon) / (results["unpert"] + epsilon))
    results["true_log2fc"] = np.log2((results["ground_truth"] + epsilon) / (results["unpert"] + epsilon))

    return results
# 移除 NaN 和 Inf
def remove_invalid_values(preds, targets):
    valid_mask = np.isfinite(preds) & np.isfinite(targets)
    return preds[valid_mask], targets[valid_mask]




def mse(preds, targets):
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)    
    squared_diff = (preds - targets) ** 2
    mse_values = torch.mean(squared_diff, dim=1)
    return mse_values.numpy()

def r2(preds, targets):
    ss_res = np.sum((targets - preds) ** 2, axis=1)
    ss_tot = np.sum((targets - np.mean(targets, axis=1, keepdims=True)) ** 2, axis=1)
    r2_scores = 1 - ss_res / ss_tot
    return r2_scores
def pearson(preds, targets):
    pearson_values = []
    for pred, target in zip(preds, targets):
        pred, target = remove_invalid_values(pred, target)
        if len(pred) > 1 and len(target) > 1:  # Pearson 相关系数至少需要两个点
            pearson_values.append(pearsonr(pred, target)[0])
        else:
            pearson_values.append(np.nan)  # 如果数据不足，返回 NaN
    return np.array(pearson_values)


def compute_mean_metrics(results):
    """
    Compute evaluation metrics for model predictions, including MSE, R², Pearson,
    deg_pearson, and fold change correlation.
    """
    pred, truth = results['prediction'], results['ground_truth']
    pred_expr_change, true_expr_change = results['pred_expr_change'], results['true_expr_change']
    pred_log2fc, true_log2fc = results['pred_log2fc'], results['true_log2fc']
    
    # Primary metrics
    metrics = {
        'mean_mse': np.nanmean(mse(pred, truth)),
        'mean_r2': np.nanmean(r2(pred, truth)),
        'mean_pearson': np.nanmean(pearson(pred, truth)),
        'mean_expr_change_pearson': np.nanmean(pearson(pred_expr_change, true_expr_change)),
        'mean_log2fc_pearson': np.nanmean(pearson(pred_log2fc, true_log2fc))
    }

    return metrics

def compute_metrics(results):
    """
    Compute evaluation metrics for model predictions, including MSE, R², Pearson,
    deg_pearson, and fold change correlation.
    """
    pred, truth = results['prediction'], results['ground_truth']
    pred_expr_change, true_expr_change = results['pred_expr_change'], results['true_expr_change']
    pred_log2fc, true_log2fc = results['pred_log2fc'], results['true_log2fc']
    
    # Primary metrics
    metrics = {
        'mse': mse(pred, truth),
        'r2': r2(pred, truth),
        'pearson': pearson(pred, truth),
        'expr_change_pearson': pearson(pred_expr_change, true_expr_change),
        'log2fc_pearson': pearson(pred_log2fc, true_log2fc)
    }
    
    return metrics

