import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score, # This is AUC-PR
    f1_score,
    confusion_matrix
)
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict
import os
import datetime
from torch_geometric.data import Data, HeteroData

def train_final_model(
    model: nn.Module,
    total_training_loader: DataLoader,
    total_test_loader: DataLoader, # Using "test" as per original, typically validation loader
    n_epochs: int,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    criterion: callable,
    device: Union[torch.device, str],
    model_save_path: Optional[str],
    save_interval: int = 10,
    is_data_homogeneous: bool = False, # Added for flexibility
    hetero_predict_edge_type: Optional[Tuple[str, str, str]] = None, # Added
    decision_threshold: float = 0.5, # Added
    k_for_precision_at_k: int = 10  # Added for P@K
) -> Dict[str, List[float]]: # Updated return type

    """
    Trains a PyTorch model for link prediction, adaptable for homogeneous or heterogeneous graphs.
    Calculates Loss, F1, Accuracy, AUC, AUC-PR, P@K, and Confusion Matrix components.

    Args:
    - model (nn.Module): the PyTorch model to be trained.
    - total_training_loader (DataLoader): DataLoader containing the training set.
    - total_test_loader (DataLoader): DataLoader containing the test/validation set.
    - n_epochs (int): number of training epochs.
    - optimizer (optim.Optimizer): the optimizer to use for training.
    - scheduler (optim.lr_scheduler): the scheduler used for training (can be None).
    - criterion (callable): the loss function to use for training (e.g., F.binary_cross_entropy_with_logits).
    - device (Union[torch.device, str]): device to run the model on (e.g. 'cpu' or 'cuda').
    - model_save_path (str, optional): path to the folder where the model will be saved. If None, model not saved.
    - save_interval (int): save the model every save_interval epochs.
    - is_data_homogeneous (bool): True if data is homogeneous, False for heterogeneous.
    - hetero_predict_edge_type (tuple, optional): Required if is_data_homogeneous is False.
                                                  Specifies the edge type for labels and prediction.
    - decision_threshold (float): Threshold to convert probabilities to binary predictions for F1/Accuracy/CM.
    - k_for_precision_at_k (int): The 'K' value for calculating Precision@K.

    Returns:
    - results (Dict[str, List[float]]): Dictionary containing lists of metrics per epoch.
    """

    if not is_data_homogeneous and hetero_predict_edge_type is None:
        raise ValueError(
            "If 'is_data_homogeneous' is False, 'hetero_predict_edge_type' must be provided."
        )

    # Initialize lists to keep track of metrics
    results = {
        'epoch': [],
        'mean_loss_train': [], 'mean_f1_train': [], 'mean_acc_train': [], 'mean_auc_train': [], 'mean_auc_pr_train': [],
        'mean_loss_test': [], 'mean_f1_test': [], 'mean_acc_test': [], 'mean_auc_test': [], 'mean_auc_pr_test': [],
        'mean_p_at_k_test': [], 'mean_tn_test': [], 'mean_fp_test': [], 'mean_fn_test': [], 'mean_tp_test': []
    }
    
    best_val_metric_for_saving = 0.0 # Example: track best Val AUC-PR for model saving

    for epoch in range(1, n_epochs + 1):
        results['epoch'].append(epoch)
        # --- Training Phase ---
        model.train()
        epoch_train_batch_losses: List[float] = []
        all_train_probs_epoch: List[torch.Tensor] = []
        all_train_labels_epoch: List[torch.Tensor] = []

        for sampled_data in tqdm(total_training_loader, desc=f"Epoch {epoch}/{n_epochs} Training", leave=False):
            sampled_data = sampled_data.to(device)
            optimizer.zero_grad()
            logits: torch.Tensor
            targets: torch.Tensor

            if is_data_homogeneous:
                logits = model(sampled_data.x, sampled_data.edge_index, sampled_data.edge_label_index)
                targets = sampled_data.edge_label
            else: # Heterogeneous
                logits = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data[hetero_predict_edge_type].edge_label_index)
                targets = sampled_data[hetero_predict_edge_type].edge_label
            
            loss = criterion(logits, targets.float())
            loss.backward()
            optimizer.step()

            epoch_train_batch_losses.append(loss.item())
            all_train_probs_epoch.append(torch.sigmoid(logits).detach().cpu())
            all_train_labels_epoch.append(targets.detach().cpu())

        # Calculate training metrics for the epoch
        if epoch_train_batch_losses:
            results['mean_loss_train'].append(np.mean(epoch_train_batch_losses))
            
            concat_train_probs = torch.cat(all_train_probs_epoch).numpy().flatten()
            concat_train_labels = torch.cat(all_train_labels_epoch).numpy().flatten()
            train_binary_preds = (concat_train_probs >= decision_threshold).astype(int)
            
            try:
                results['mean_f1_train'].append(f1_score(concat_train_labels, train_binary_preds, zero_division=0))
                results['mean_acc_train'].append(accuracy_score(concat_train_labels, train_binary_preds))
                results['mean_auc_train'].append(roc_auc_score(concat_train_labels, concat_train_probs))
                results['mean_auc_pr_train'].append(average_precision_score(concat_train_labels, concat_train_probs))
            except ValueError: # Handle cases with only one class
                results['mean_f1_train'].append(0.0); results['mean_acc_train'].append(0.0); results['mean_auc_train'].append(0.0); results['mean_auc_pr_train'].append(0.0)
        else:
            for key in ['mean_loss_train', 'mean_f1_train', 'mean_acc_train', 'mean_auc_train', 'mean_auc_pr_train']: results[key].append(float('nan'))


        # --- Evaluation Phase (on total_test_loader, typically validation set) ---
        model.eval()
        epoch_test_batch_losses: List[float] = []
        all_test_probs_epoch: List[torch.Tensor] = []
        all_test_labels_epoch: List[torch.Tensor] = []

        with torch.no_grad():
            for sampled_data in tqdm(total_test_loader, desc=f"Epoch {epoch}/{n_epochs} Validation", leave=False):
                sampled_data = sampled_data.to(device)
                logits: torch.Tensor
                targets: torch.Tensor

                if is_data_homogeneous:
                    logits = model(sampled_data.x, sampled_data.edge_index, sampled_data.edge_label_index)
                    targets = sampled_data.edge_label
                else: # Heterogeneous
                    logits = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data[hetero_predict_edge_type].edge_label_index)
                    targets = sampled_data[hetero_predict_edge_type].edge_label
                
                loss = criterion(logits, targets.float())
                epoch_test_batch_losses.append(loss.item())
                all_test_probs_epoch.append(torch.sigmoid(logits).detach().cpu())
                all_test_labels_epoch.append(targets.detach().cpu())
        
        # Calculate validation metrics for the epoch
        if epoch_test_batch_losses:
            results['mean_loss_test'].append(np.mean(epoch_test_batch_losses))

            concat_test_probs = torch.cat(all_test_probs_epoch).numpy().flatten()
            concat_test_labels = torch.cat(all_test_labels_epoch).numpy().flatten()
            test_binary_preds = (concat_test_probs >= decision_threshold).astype(int)
            
            current_val_auc, current_val_auc_pr, current_val_f1, current_val_acc = 0.0, 0.0, 0.0, 0.0
            current_tn, current_fp, current_fn, current_tp = 0, 0, 0, 0
            current_p_at_k = 0.0

            try:
                current_val_auc = roc_auc_score(concat_test_labels, concat_test_probs)
                current_val_auc_pr = average_precision_score(concat_test_labels, concat_test_probs)
                current_val_f1 = f1_score(concat_test_labels, test_binary_preds, zero_division=0)
                current_val_acc = accuracy_score(concat_test_labels, test_binary_preds)
                
                tn, fp, fn, tp = confusion_matrix(concat_test_labels, test_binary_preds, labels=[0,1]).ravel()
                current_tn, current_fp, current_fn, current_tp = int(tn), int(fp), int(fn), int(tp) # Ensure they are native Python ints for JSON if needed

                # Precision@K
                k_val = min(k_for_precision_at_k, len(concat_test_probs))
                if k_val > 0:
                    top_k_indices = np.argsort(concat_test_probs)[-k_val:][::-1]
                    top_k_true_labels = concat_test_labels[top_k_indices]
                    current_p_at_k = np.sum(top_k_true_labels) / k_val
            except ValueError:
                print(f"Warning: Could not calculate some validation metrics for epoch {epoch} due to label/prediction issues.")

            results['mean_auc_test'].append(current_val_auc)
            results['mean_auc_pr_test'].append(current_val_auc_pr)
            results['mean_f1_test'].append(current_val_f1)
            results['mean_acc_test'].append(current_val_acc)
            results['mean_tn_test'].append(current_tn); results['mean_fp_test'].append(current_fp); results['mean_fn_test'].append(current_fn); results['mean_tp_test'].append(current_tp)
            results['mean_p_at_k_test'].append(current_p_at_k)

            if scheduler:
                scheduler.step(results['mean_auc_test'][-1]) # Original behavior: step on val loss
                # Or, if scheduler mode is 'max', e.g., scheduler.step(current_val_auc_pr)
        
            lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}/{n_epochs} -> \n"
                  f"  Train | Loss: {results['mean_loss_train'][-1]:.4f}, F1: {results['mean_f1_train'][-1]:.4f}, AUC: {results['mean_auc_train'][-1]:.4f}, AUC-PR: {results['mean_auc_pr_train'][-1]:.4f}\n"
                  f"  Val   | Loss: {results['mean_loss_test'][-1]:.4f}, F1: {results['mean_f1_test'][-1]:.4f} (Th:{decision_threshold}), AUC: {results['mean_auc_test'][-1]:.4f}, AUC-PR: {results['mean_auc_pr_test'][-1]:.4f}, P@{k_for_precision_at_k}: {results['mean_p_at_k_test'][-1]:.4f}\n"
                  f"  Val CM| TN: {results['mean_tn_test'][-1]}, FP: {results['mean_fp_test'][-1]}, FN: {results['mean_fn_test'][-1]}, TP: {results['mean_tp_test'][-1]}\n"
                  f"  LR: {lr:.1e}")

            # Model saving logic
            metric_to_monitor_for_best_model = results['mean_auc_pr_test'][-1] # Example: save based on Val AUC-PR
            if metric_to_monitor_for_best_model > best_val_metric_for_saving:
                best_val_metric_for_saving = metric_to_monitor_for_best_model
                if model_save_path:
                    if not os.path.exists(model_save_path): os.makedirs(model_save_path)
                    best_model_filename = os.path.join(model_save_path, f'GNN_best_epoch_{epoch}_valAUCPR_{best_val_metric_for_saving:.4f}.pth')
                    torch.save(model.state_dict(), best_model_filename)
                    print(f"Saved new best model to {best_model_filename}")
        else: # No validation batches
            for key in ['mean_loss_test', 'mean_f1_test', 'mean_acc_test', 'mean_auc_test', 'mean_auc_pr_test', 'mean_p_at_k_test', 'mean_tn_test', 'mean_fp_test', 'mean_fn_test', 'mean_tp_test']:
                results[key].append(float('nan') if key.startswith('mean_loss') else 0.0)
            print(f"Epoch {epoch}/{n_epochs} -> Train Loss: {results['mean_loss_train'][-1]:.4f} | Val metrics not available (no val batches).")


        if model_save_path and epoch % save_interval == 0 and epoch != 0:
            if not os.path.exists(model_save_path): os.makedirs(model_save_path)
            # Use a consistent metric for checkpoint naming, e.g. val_auc_pr
            checkpoint_filename = os.path.join(model_save_path,f'GNN_epoch_{epoch}_valAUCPR_{results["mean_auc_pr_test"][-1]:.4f}_date{datetime.datetime.now().strftime("%d%m%Y-%H%M%S")}.pth')
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved model checkpoint to {checkpoint_filename}")
            
    return results