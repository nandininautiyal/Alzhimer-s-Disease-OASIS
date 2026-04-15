"""
plotting.py
───────────
Plotting utilities called by train.py:
  - plot_CM()               confusion matrix heatmap
  - plot_training_history() accuracy / loss / precision / recall curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = ["CN", "MCI", "AD"]


def plot_CM(cm, save_dir, filename="confusion_matrix.png"):
    """
    Parameters
    ----------
    cm       : np.array  (3×3) confusion matrix from sklearn
    save_dir : str       directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)

    path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Confusion matrix saved → {path}")


def plot_training_history(history: dict, save_dir: str):
    """
    Plots accuracy, loss, precision, recall curves (train + val).

    Parameters
    ----------
    history  : dict   keys: loss, acc, precision, recall,
                            val_loss, val_acc, val_precision, val_recall
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    pairs = [
        ("acc",       "val_acc",       "Accuracy",   "accuracy.png"),
        ("loss",      "val_loss",      "Loss",       "loss.png"),
        ("precision", "val_precision", "Precision",  "precision.png"),
        ("recall",    "val_recall",    "Recall",     "recall.png"),
    ]

    for train_key, val_key, title, fname in pairs:
        if train_key not in history:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history[train_key], label=f"Train {title}", linewidth=1.5)
        ax.plot(epochs, history[val_key],   label=f"Val {title}",   linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Training vs Validation {title}")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(save_dir, fname)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [PLOT] {title} curve saved → {path}")