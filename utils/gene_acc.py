import torch
import torch.nn as nn
import numpy as np
import os
import glob
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix
)
from torch_geometric.data import Data
from typing import Dict, Any, Optional

def find_best_model_path(model_dir: str) -> Optional[str]:
    """
    Finds the model file in a directory that corresponds to the best performance.
    """
    search_pattern = os.path.join(model_dir, '*best_epoch*.pth')
    best_model_files = glob.glob(search_pattern)
    if not best_model_files:
        print(f"Warning: No 'best model' file found in '{model_dir}'. Looking for any model file.")
        search_pattern = os.path.join(model_dir, '*.pth')
        best_model_files = glob.glob(search_pattern)
        if not best_model_files:
            print(f"Error: No model files found in '{model_dir}'.")
            return None
    latest_file = max(best_model_files, key=os.path.getctime)
    print(f"Found model to evaluate: {os.path.basename(latest_file)}")
    return latest_file


def evaluate_homogeneous_gene_disease_links(
    model: nn.Module,
    model_path: str,
    test_data: Data,
    device: torch.device,
    decision_threshold: float = 0.5,
    k_for_precision_at_k: int = 10
) -> Optional[Dict[str, float]]:
    """
    Loads a trained GNN model and evaluates its performance on predicting
    only the links between gene and disease nodes in a homogeneous graph.

    Args:
        model (nn.Module): An instantiated model compatible with homogeneous data.
        model_path (str): The file path to the saved model's state_dict (.pth file).
        test_data (Data): The test `Data` object. It MUST have
                          a 'num_gene_nodes' attribute.
        device (torch.device): The device to run evaluation on ('cpu' or 'cuda').
        decision_threshold (float): Threshold for converting probabilities to binary predictions.
        k_for_precision_at_k (int): The 'K' value for Precision@K.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing the calculated performance metrics.
    """
    # === 1. Load Model Weights ===
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    model.to(device)
    model.eval()
    print(f"Model loaded from {os.path.basename(model_path)} and set to evaluation mode.")

    # === 2. Validate Input Data ===
    if not hasattr(test_data, 'num_gene_nodes'):
        raise ValueError("The 'test_data' object must have the 'num_gene_nodes' attribute.")

    test_data = test_data.to(device)
    num_gene_nodes = test_data.num_gene_nodes
    edge_label_index = test_data.edge_label_index
    edge_label = test_data.edge_label

    # === 3. Perform Inference ===
    with torch.no_grad():
        # A homogeneous model expects Tensors, not dictionaries
        logits = model(test_data.x, test_data.edge_index, edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = edge_label.cpu().numpy().flatten()

    # === 4. Filter for Gene-Disease Links ONLY ===
    source_nodes, target_nodes = edge_label_index[0].cpu(), edge_label_index[1].cpu()
    is_source_gene = source_nodes < num_gene_nodes
    is_target_disease = target_nodes >= num_gene_nodes
    
    # Gene -> Disease links
    mask1 = torch.logical_and(is_source_gene, is_target_disease)
    # Disease -> Gene links
    mask2 = torch.logical_and(torch.logical_not(is_source_gene), torch.logical_not(is_target_disease))
    
    gene_disease_mask = torch.logical_or(mask1, mask2).numpy()

    filtered_probs = probs[gene_disease_mask]
    filtered_labels = labels[gene_disease_mask]

    print(f"Evaluation: Found {len(filtered_labels)} gene-disease links to evaluate.")
    if len(filtered_labels) == 0:
        return None

    # === 5. Calculate Metrics on Filtered Links ===
    binary_preds = (filtered_probs >= decision_threshold).astype(int)

    if len(np.unique(filtered_labels)) < 2:
        auc_roc = 0.5
        auc_pr = average_precision_score(filtered_labels, filtered_probs)
    else:
        auc_roc = roc_auc_score(filtered_labels, filtered_probs)
        auc_pr = average_precision_score(filtered_labels, filtered_probs)

    f1 = f1_score(filtered_labels, binary_preds, zero_division=0)
    acc = accuracy_score(filtered_labels, binary_preds)
    
    # ... (rest of metric calculations are the same)
    metrics = { 'accuracy': acc, 'f1_score': f1, 'auc_roc': auc_roc, 'auc_pr': auc_pr, }
    return metrics
