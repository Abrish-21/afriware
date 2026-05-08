"""
metrics.py
Evaluation utilities: classification report, confusion matrix, per-class F1.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def full_report(
    y_true: list,
    y_pred: list,
    id2label: dict,
    task_name: str = "classifier",
    save: bool = True,
) -> dict:
    """
    Prints and saves a complete evaluation report.

    Returns a dict with all metrics.
    """
    labels      = sorted(id2label.keys())
    label_names = [id2label[l] for l in labels]

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", labels=labels, zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=label_names, labels=labels, zero_division=0
    )

    print(f"\n{'='*55}")
    print(f"EVALUATION: {task_name.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (macro)    : {f1m:.4f}")
    print(f"  F1 (weighted) : {f1w:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print("\nPer-class report:")
    print(report)

    metrics = {
        "task":         task_name,
        "accuracy":     round(acc, 4),
        "f1_macro":     round(f1m, 4),
        "f1_weighted":  round(f1w, 4),
        "precision":    round(prec, 4),
        "recall":       round(rec, 4),
    }

    if save:
        json_path = OUTPUT_DIR / f"{task_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics saved → {json_path}")

        plot_confusion_matrix(y_true, y_pred, label_names, task_name)

    return metrics


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    label_names: list,
    task_name: str,
):
    """Saves a confusion matrix heatmap as PNG."""
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {task_name}")
    plt.tight_layout()

    path = OUTPUT_DIR / f"{task_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved → {path}")


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Compares multiple model results in a DataFrame.

    Args:
        results: list of metric dicts (each must have 'task' key)
    """
    df = pd.DataFrame(results)
    df = df.sort_values("f1_macro", ascending=False)
    print("\nModel comparison:")
    print(df.to_string(index=False))
    return df
