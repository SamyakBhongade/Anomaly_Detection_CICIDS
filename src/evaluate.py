"""
evaluate.py
Complete evaluation suite: all metrics, confusion matrix, ROC curve,
convergence curves, comparison table vs base paper.
All plots saved to outputs/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    matthews_corrcoef
)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: np.ndarray = None,
                        model_name: str = 'Model') -> dict:
    """Compute all metrics needed for IEEE paper tables."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mcc  = matthews_corrcoef(y_true, y_pred)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = None

    metrics = {
        'Model':     model_name,
        'Accuracy':  round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall':    round(rec  * 100, 2),
        'F1-Score':  round(f1   * 100, 2),
        'TPR':       round(tpr  * 100, 2),
        'FPR':       round(fpr  * 100, 2),
        'MCC':       round(mcc,        4),
        'AUC-ROC':   round(auc  * 100, 2) if auc else 'N/A',
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

    print(f"\n{'='*55}")
    print(f"  Results: {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k not in ['TP','TN','FP','FN']:
            print(f"  {k:<12}: {v}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*55}\n")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(save_dir, f'cm_{model_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Confusion matrix saved -> {path}")


def plot_roc_curves(results: list, y_true: np.ndarray, save_dir: str,
                    dataset_name: str = ''):
    """
    Plot ROC curves for all models on the same figure.
    results: list of dicts with keys: name, proba
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, r in enumerate(results):
        if r.get('proba') is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, r['proba'])
        auc = roc_auc_score(y_true, r['proba'])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{r['name']} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves — {dataset_name}', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'roc_{dataset_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] ROC curve saved -> {path}")


def plot_convergence(convergence_curves: dict, save_dir: str,
                     dataset_name: str = ''):
    """
    Plot SSA convergence curves for all models.
    convergence_curves: dict {model_name: list_of_fitness_values}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (name, curve) in enumerate(convergence_curves.items()):
        ax.plot(range(1, len(curve)+1), curve,
                color=colors[i % len(colors)], lw=2, marker='o',
                markersize=3, label=name)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Fitness (Validation Loss)', fontsize=12)
    ax.set_title(f'SSA Convergence Curves — {dataset_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'convergence_{dataset_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Convergence curve saved -> {path}")


def plot_metrics_bar(metrics_list: list, save_dir: str, dataset_name: str = ''):
    """Bar chart comparing Accuracy, Precision, Recall, F1 across models."""
    models  = [m['Model']     for m in metrics_list]
    acc     = [m['Accuracy']  for m in metrics_list]
    prec    = [m['Precision'] for m in metrics_list]
    rec     = [m['Recall']    for m in metrics_list]
    f1      = [m['F1-Score']  for m in metrics_list]

    x  = np.arange(len(models))
    w  = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*w, acc,  w, label='Accuracy',  color='#1f77b4')
    ax.bar(x - 0.5*w, prec, w, label='Precision', color='#ff7f0e')
    ax.bar(x + 0.5*w, rec,  w, label='Recall',    color='#2ca02c')
    ax.bar(x + 1.5*w, f1,   w, label='F1-Score',  color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Performance Comparison — {dataset_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'metrics_bar_{dataset_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Metrics bar chart saved -> {path}")


def plot_training_history(history_dict: dict, save_dir: str, model_name: str):
    """Plot training vs validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history_dict['accuracy'],     label='Train', lw=2)
    axes[0].plot(history_dict['val_accuracy'], label='Val',   lw=2, linestyle='--')
    axes[0].set_title(f'{model_name} — Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history_dict['loss'],     label='Train', lw=2)
    axes[1].plot(history_dict['val_loss'], label='Val',   lw=2, linestyle='--')
    axes[1].set_title(f'{model_name} — Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f'history_{model_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Training history saved -> {path}")


def save_metrics_table(metrics_list: list, save_dir: str,
                       dataset_name: str = ''):
    """Save metrics as CSV and print as formatted table."""
    df = pd.DataFrame(metrics_list)
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'TPR', 'FPR', 'AUC-ROC', 'MCC']
    df_display = df[[c for c in display_cols if c in df.columns]]

    csv_path = os.path.join(save_dir,
                            f'metrics_{dataset_name.replace(" ", "_")}.csv')
    df_display.to_csv(csv_path, index=False)
    print(f"\n[Metrics] Table saved -> {csv_path}")
    print(df_display.to_string(index=False))
    return df_display


def full_evaluation(y_true, y_pred, y_proba,
                    model_name, dataset_name, save_dir):
    """One-call full evaluation: metrics + plots."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = compute_all_metrics(y_true, y_pred, y_proba, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name, save_dir)
    return metrics
