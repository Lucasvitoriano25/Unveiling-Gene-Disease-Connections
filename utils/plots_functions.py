import matplotlib.pyplot as plt
from itertools import cycle

def plot_metrics_over_time(results_dict, title, save_path=None):
    """
    Creates a 2x2 plot for Loss, AUC, F1-Score, and AUC-PR over epochs.
    """
    if not results_dict or not results_dict.get('epoch'):
        print(f"Skipping plot for '{title}' due to empty or invalid results dictionary.")
        return

    epochs = results_dict['epoch']

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Loss
    axs[0, 0].plot(epochs, results_dict.get('mean_loss_train', []), label='Train Loss', marker='.')
    axs[0, 0].plot(epochs, results_dict.get('mean_loss_test', []), label='Validation Loss', marker='.')
    axs[0, 0].set_title('Loss Over Epochs')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: AUC (ROC)
    axs[0, 1].plot(epochs, results_dict.get('mean_auc_train', []), label='Train AUC')
    axs[0, 1].plot(epochs, results_dict.get('mean_auc_test', []), label='Validation AUC')
    axs[0, 1].set_title('AUC-ROC Over Epochs')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('AUC')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: F1-Score
    axs[1, 0].plot(epochs, results_dict.get('mean_f1_train', []), label='Train F1-Score')
    axs[1, 0].plot(epochs, results_dict.get('mean_f1_test', []), label='Validation F1-Score')
    axs[1, 0].set_title('F1-Score Over Epochs')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('F1-Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: AUC-PR (Average Precision)
    axs[1, 1].plot(epochs, results_dict.get('mean_auc_pr_train', []), label='Train AUC-PR')
    axs[1, 1].plot(epochs, results_dict.get('mean_auc_pr_test', []), label='Validation AUC-PR')
    axs[1, 1].set_title('AUC-PR (Average Precision) Over Epochs')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('AUC-PR')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show() # Display the plot in the notebook

def plot_accuracy_f1_over_epoch(results_to_compare: Dict[str, Dict], title: str, save_path=None):
    """
    Creates a 1x2 comparison plot for Accuracy and F1-Score for multiple models.
    """
    if not results_to_compare:
        print(f"Skipping comparison plot '{title}' as there are no results to compare.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(title, fontsize=16)

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_to_compare)))

    for i, (model_name, data) in enumerate(results_to_compare.items()):
        if not data or not data.get('epoch'):
            continue

        epochs = data['epoch']
        color = colors[i]

        # Subplot 1: Accuracy
        val_acc = data.get('mean_acc_test', [])
        if val_acc:
            axs[0].plot(epochs, val_acc, label=f'{model_name} Val Accuracy', color=color)

        # Subplot 2: F1-Score
        val_f1 = data.get('mean_f1_test', [])
        if val_f1:
            axs[1].plot(epochs, val_f1, label=f'{model_name} Val F1-Score', color=color)

    axs[0].set_title('Validation Accuracy Comparison')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Validation F1-Score Comparison')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('F1-Score')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving comparison plot to {save_path}: {e}")

    plt.show()
    """
    Plots accuracy and F1 score over epochs for multiple GNN model results with automatically changing line styles.

    Args:
        results_dict: A dictionary where each key is a model name and each value is another
                      dictionary containing arrays of metric values (accuracy and F1) for training and testing.
        title: The overall title for the plots.
    """
    # Setup figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    # Define a cycle of line styles
    line_styles = ['-', '--', '-.', ':']
    line_cycle = cycle(line_styles)

    for model_name, metrics in results_dict.items():
        epochs = range(len(metrics['mean_acc_train']))
        current_style = next(line_cycle)

        # Accuracy plot
        axs[0].plot(epochs, metrics['mean_acc_test'], label=f'{model_name} Test', linestyle=current_style)
        axs[0].set_title('Accuracy Over Epochs')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()

        # F1 Score plot
        axs[1].plot(epochs, metrics['mean_f1_test'], label=f'{model_name} Test', linestyle=current_style)
        axs[1].set_title('F1 Score Over Epochs')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('F1 Score')
        axs[1].legend()

    plt.tight_layout()
    plt.show()