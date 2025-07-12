import torch
import torch.nn.functional as F
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
    Assumes the best model contains 'best_epoch' in its filename.

    Args:
        model_dir (str): The directory where models are saved.

    Returns:
        Optional[str]: The full path to the best model file, or None if not found.
    """
    search_pattern = os.path.join(model_dir, '*best_epoch*.pth')
    best_model_files = glob.glob(search_pattern)
    if not best_model_files:
        print(f"Warning: No best model file found in '{model_dir}' matching pattern '{search_pattern}'.")
        return None
    # In case of multiple 'best' files, return the most recently modified one.
    latest_file = max(best_model_files, key=os.path.getctime)
    print(f"Found best model: {os.path.basename(latest_file)}")
    return latest_file


def evaluate_gene_disease_links(
    model_class: torch.nn.Module,
    model_hyperparams: Dict[str, Any],
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
        model_class (torch.nn.Module): The class of the model to instantiate (e.g., GNN_FeedFoward).
        model_hyperparams (Dict[str, Any]): A dictionary of hyperparameters required
                                           to initialize the model_class.
        model_path (str): The file path to the saved model's state_dict (.pth file).
        test_data (Data): The test data object from a PyG split. It must have
                          'num_gene_nodes' attribute.
        device (torch.device): The device to run evaluation on ('cpu' or 'cuda').
        decision_threshold (float): Threshold for converting probabilities to binary predictions.
        k_for_precision_at_k (int): The 'K' value for Precision@K.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing the calculated performance metrics, or None.
    """
    # 1. Instantiate the model and load the trained weights
    model = model_class(**model_hyperparams).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {os.path.basename(model_path)} and set to evaluation mode.")

    # Check for necessary attributes in test_data
    if not hasattr(test_data, 'num_gene_nodes'):
        raise ValueError("test_data object must have the 'num_gene_nodes' attribute.")

    num_gene_nodes = test_data.num_gene_nodes
    edge_label_index = test_data.edge_label_index
    edge_label = test_data.edge_label

    # 2. Perform inference in a single batch (full graph evaluation)
    with torch.no_grad():
        # The model expects the full edge_index for message passing, and edge_label_index for prediction
        logits = model(test_data.x, test_data.edge_index, edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        labels = edge_label.cpu().numpy().flatten()

    # 3. Filter for gene-disease links ONLY
    source_nodes, target_nodes = edge_label_index[0].cpu(), edge_label_index[1].cpu()

    # A link is gene-disease if one node index is < num_gene_nodes and the other is >= num_gene_nodes
    is_source_gene = source_nodes < num_gene_nodes
    is_target_gene = target_nodes < num_gene_nodes

    # Use bitwise XOR to find edges where one is a gene and the other is a disease
    gene_disease_mask = torch.bitwise_xor(is_source_gene, is_target_gene).numpy()

    filtered_probs = probs[gene_disease_mask]
    filtered_labels = labels[gene_disease_mask]

    num_total_links = len(labels)
    num_gd_links = len(filtered_labels)
    
    print(f"Evaluation: Found {num_gd_links} gene-disease links out of {num_total_links} total evaluation links.")
    if num_gd_links == 0:
        print("Warning: No gene-disease links found for evaluation. Cannot compute metrics.")
        return None

    # 4. Calculate metrics on the filtered set of gene-disease links
    binary_preds = (filtered_probs >= decision_threshold).astype(int)

    # Handle cases where only one class is present in the filtered labels
    if len(np.unique(filtered_labels)) < 2:
        print(f"Warning: Only one class ({np.unique(filtered_labels)[0]}) present in the gene-disease links.")
        auc_roc = 0.5  # Metric is undefined, 0.5 is a neutral value
        auc_pr = average_precision_score(filtered_labels, filtered_probs)
    else:
        auc_roc = roc_auc_score(filtered_labels, filtered_probs)
        auc_pr = average_precision_score(filtered_labels, filtered_probs)

    f1 = f1_score(filtered_labels, binary_preds, zero_division=0)
    acc = accuracy_score(filtered_labels, binary_preds)
    
    # Confusion Matrix
    try:
        tn, fp, fn, tp = confusion_matrix(filtered_labels, binary_preds, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0 # Could not compute

    # Precision@K
    k_val = min(k_for_precision_at_k, len(filtered_probs))
    p_at_k = 0.0
    if k_val > 0:
        # Get indices of the top k predictions
        top_k_indices = np.argsort(filtered_probs)[-k_val:][::-1]
        top_k_true_labels = filtered_labels[top_k_indices]
        p_at_k = np.sum(top_k_true_labels) / k_val

    metrics = {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'accuracy': acc,
        f'precision_at_{k_for_precision_at_k}': p_at_k,
        'true_positives': float(tp),
        'true_negatives': float(tn),
        'false_positives': float(fp),
        'false_negatives': float(fn),
        'total_gene_disease_links': num_gd_links,
        'positive_gene_disease_links': int(np.sum(filtered_labels))
    }

    return metrics
