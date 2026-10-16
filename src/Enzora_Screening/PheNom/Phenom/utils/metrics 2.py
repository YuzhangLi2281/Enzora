from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
import numpy as np
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from utils.vs_metrics import cal_vs_metrics
import torch
import copy
import math
from sklearn.metrics import r2_score


def torch_pearson_correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + eps) * torch.sqrt(torch.sum(vy ** 2) + eps))
    return torch.clamp(corr, -1.0, 1.0)

def torch_spearman_correlation(x, y, eps=1e-8):
    x = x.float()
    y = y.float()
    
    # Add small noise to break ties
    x = x + torch.randn_like(x) * eps
    y = y + torch.randn_like(y) * eps
    
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    
    return torch_pearson_correlation(x_rank, y_rank)

def get_cindex(Y, P):
    return concordance_index(Y, P)


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


# Prepare for rm2
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def cos_formula(a, b, c):
    ''' formula to calculate the angle between two edges
        a and b are the edge lengths, c is the angle length.
    '''
    res = (a**2 + b**2 - c**2) / (2 * a * b)
    # sanity check
    res = -1. if res < -1. else res
    res = 1. if res > 1. else res
    return np.arccos(res)

def get_rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def get_mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def get_sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def get_pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def get_spearman(y,f):
    sp = stats.spearmanr(y,f)[0]

    return sp

def categorize_vectorized(vals):
    categories = np.full(vals.shape, 'strong_binder', dtype=object)
    categories[vals < 8] = 'moderate_binder'
    categories[vals < 6] = 'weak_binder'
    categories[vals < 4] = 'inactive'

    
    return categories


def evaluate_r2(y_true, y_pred, y_train=None):
    """
    Calculate both traditional R² and R²-out-of-sample
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training set values (optional)
    """
    # Traditional R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1]
    if math.isnan(r2) or r2 < 0.:
        r2 = 0.
    else:
        r2 = r2 ** 2
        
    # R² out-of-sample
    if y_train is not None:
        y_train_mean = np.mean(y_train)
        numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((y_true - y_train_mean) ** 2).sum(axis=0, dtype=np.float64)
        R2os = 1.0 - (numerator / denominator) if denominator != 0 else 0.0
        return r2, R2os
    
    return r2

def evaluate_reg(input_Y, input_F):

    Y = copy.deepcopy(input_Y)
    F = copy.deepcopy(input_F)

    not_nan_indices = ~np.isnan(Y)
    Y = Y[not_nan_indices]
    F = F[not_nan_indices]

    reg_result = { 
        'mse': float(get_mse(Y,F)),
        'rmse': float(get_rmse(Y,F)),
        'mae': float(get_mae(Y,F)),
        'sd': float(get_sd(Y,F)),
        'pearson': float(get_pearson(Y,F)),
        'spearman': float(get_spearman(Y,F)),
        'rm2': float(get_rm2(Y,F)),
        'r2': float(evaluate_r2(Y,F)),
        'ci': float(get_cindex(Y,F))
    }

    # Categorize true values and predictions using vectorized function
    Y_cat = categorize_vectorized(Y)
    F_cat = categorize_vectorized(F)
    # Convert string categories to integer indices
    reg_result['reg_accuracy'] = float(accuracy_score(Y_cat, F_cat))
    reg_result['reg_macro_f1'] =  float(f1_score(Y_cat, F_cat, average='macro'))
    
    return reg_result

def evaluate_cls(Y,P,threshold=0.5):
    predicted_label = P > threshold
    cls_metric = {
        'roc': float(roc_auc_score(Y,P)),
        'prc': float(average_precision_score(Y,P)),
        'f1': float(f1_score(Y,predicted_label)),
        'recall':float(recall_score(Y, predicted_label)),
        'precision': float(precision_score(Y, predicted_label))
    }

    auc, bedroc, ef = cal_vs_metrics(Y, P)
    cls_metric['vs_roc'] = auc
    cls_metric['vs_bedroc'] = bedroc
    cls_metric['vs_ef_0.005'] = ef['0.005']
    cls_metric['vs_ef_0.01'] = ef['0.01']
    cls_metric['vs_ef_0.02'] = ef['0.02']
    cls_metric['vs_ef_0.05'] = ef['0.05']

    return cls_metric

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score

def multiclass_ap(Y_test, y_score, n_classes):
    # For each class
    # precision = dict()
    # recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        num_labels = Y_test[:, i].sum()
        if num_labels == 0: continue
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    return sum(average_precision.values()) / len(average_precision)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import warnings

# Keep the helper if needed, but it's not used in the multi-label version below
# def indices_to_one_hot(data, nb_classes):
#     """Convert an iterable of indices to one-hot encoded labels."""
#     targets = np.array(data).reshape(-1)
#     return np.eye(nb_classes)[targets]

# # Keep if needed elsewhere, but not directly used in the new evaluate_mcls
# def multiclass_ap(y_true_onehot, y_pred_scores, n_classes):
#     """Calculate mean average precision for multi-class classification."""
#     APs = []
#     for i in range(n_classes):
#         APs.append(average_precision_score(y_true_onehot[:, i], y_pred_scores[:, i]))
#     return np.mean(APs)


def evaluate_mcls(Y_true: np.ndarray, P_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluates multi-label classification performance for each class independently.

    Args:
        Y_true (np.ndarray): Ground truth labels. Shape: (num_samples, num_classes).
                             Values should be 0 (negative), 1 (positive), or -1 (ignore).
        P_pred_probs (np.ndarray): Predicted probabilities. Shape: (num_samples, num_classes).
                                   Values should be between 0 and 1.
        threshold (float): Threshold to convert probabilities to binary predictions 
                           for accuracy and F1-score. Defaults to 0.5.

    Returns:
        dict: A dictionary containing ROC AUC, PRC AUC, Accuracy, and F1-score
              for each individual class. Returns np.nan for metrics that cannot
              be computed (e.g., AUC with only one class present).
    """
    if Y_true.shape != P_pred_probs.shape:
        raise ValueError(f"Shape mismatch: Y_true {Y_true.shape} vs P_pred_probs {P_pred_probs.shape}")
    if Y_true.ndim != 2 or P_pred_probs.ndim != 2:
         raise ValueError("Inputs Y_true and P_pred_probs must be 2D arrays.")
        
    num_samples, num_classes = Y_true.shape
    results = {}
    
    # Define class names matching the typical order if known, otherwise use indices
    # Adjust this list based on the actual column order in your Y_true/P_pred_probs
    class_names = ['OrthoInhibitor', 'OrthoActivator', 'AlloInhibitor', 'AlloActivator']
    if num_classes != len(class_names):
        warnings.warn(f"Number of classes ({num_classes}) doesn't match expected names ({len(class_names)}). Using generic names.")
        class_names = [f'Class_{i}' for i in range(num_classes)]

    for i in range(num_classes):
        class_name = class_names[i]
        
        # --- 1. Filter out ignored samples (-1) for this class ---
        valid_mask = Y_true[:, i] != -1
        y_true_class = Y_true[valid_mask, i]
        y_pred_prob_class = P_pred_probs[valid_mask, i]

        # --- 2. Check if there are enough valid samples and classes ---
        if len(y_true_class) == 0:
            warnings.warn(f"No valid samples found for class '{class_name}'. Skipping metrics.")
            roc, prc, acc, f1 = [np.nan] * 4
        elif len(np.unique(y_true_class)) < 2:
            # AUC metrics are undefined if only one class is present
            warnings.warn(f"Only one class present for class '{class_name}' after filtering. ROC/PRC will be NaN.")
            roc, prc = np.nan, np.nan
            # Accuracy and F1 can still be calculated but might be trivial
            y_pred_binary_class = (y_pred_prob_class >= threshold).astype(int)
            acc = accuracy_score(y_true_class, y_pred_binary_class)
            # Use zero_division=0 to handle cases where F1 is ill-defined (no true positives AND no predicted positives)
            f1 = f1_score(y_true_class, y_pred_binary_class, zero_division=0)
        else:
            # --- 3. Calculate metrics for this class ---
            try:
                roc = roc_auc_score(y_true_class, y_pred_prob_class)
            except ValueError as e:
                 warnings.warn(f"Could not calculate ROC AUC for class '{class_name}': {e}")
                 roc = np.nan # Should already be covered by unique check, but for safety

            try:
                # Average precision score is equivalent to PRC AUC for binary tasks
                prc = average_precision_score(y_true_class, y_pred_prob_class)
            except ValueError as e:
                warnings.warn(f"Could not calculate PRC AUC for class '{class_name}': {e}")
                prc = np.nan # Should already be covered by unique check, but for safety

            # Convert probabilities to binary predictions based on threshold
            y_pred_binary_class = (y_pred_prob_class >= threshold).astype(int)
            
            acc = accuracy_score(y_true_class, y_pred_binary_class)
            f1 = f1_score(y_true_class, y_pred_binary_class, zero_division=0)

        # --- 4. Store results for this class ---
        results[f'{class_name}_roc_auc'] = float(roc) if not np.isnan(roc) else roc
        results[f'{class_name}_prc_auc'] = float(prc) if not np.isnan(prc) else prc
        results[f'{class_name}_accuracy'] = float(acc) if not np.isnan(acc) else acc
        results[f'{class_name}_f1'] = float(f1) if not np.isnan(f1) else f1

    return results

# --- Example Usage ---
# Y_true_example = np.array([
#     [1, 0, -1, 1],  # Sample 1
#     [0, -1, 0, 0],  # Sample 2
#     [-1, 1, 1, -1], # Sample 3
#     [1, 1, 0, 1],   # Sample 4
#     [0, 0, 0, 0]    # Sample 5
# ])
# 
# # Example probabilities (replace with your model's output)
# P_pred_probs_example = np.array([
#     [0.9, 0.2, 0.5, 0.8], 
#     [0.1, 0.6, 0.3, 0.4],
#     [0.4, 0.7, 0.9, 0.1],
#     [0.7, 0.8, 0.2, 0.9],
#     [0.2, 0.1, 0.4, 0.3]
# ])
# 
# metrics_results = evaluate_mcls(Y_true_example, P_pred_probs_example)
# print(metrics_results) 
# 
# # Example with a class having only one label type after filtering
# Y_true_single = np.array([
#     [1, 0, -1, 1],  
#     [1, -1, -1, 0],  
#     [-1, 0, -1, -1], 
#     [1, 0, -1, 1]   
# ])
# P_pred_single = np.array([
#     [0.9, 0.2, 0.5, 0.8], 
#     [0.8, 0.6, 0.3, 0.4],
#     [0.4, 0.7, 0.9, 0.1],
#     [0.7, 0.8, 0.2, 0.9]
# ])
# metrics_single = evaluate_mcls(Y_true_single, P_pred_single)
# print("\nMetrics with single class present (Class 2 all -1):")
# print(metrics_single)