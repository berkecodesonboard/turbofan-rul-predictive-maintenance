from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, output_path: str | Path):
    """
    Saves actual vs predicted RUL scatter plot.
    """
    output_path = Path(output_path)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"Actual vs Predicted RUL - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_histogram(y_true, y_pred, model_name: str, output_path: str | Path):
    """
    Saves prediction error histogram.
    """
    output_path = Path(output_path)
    errors = y_pred - y_true

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title(f"Prediction Error Histogram - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, metric: str, output_path: str | Path):
    """
    Saves model comparison bar chart for a selected metric.
    """
    output_path = Path(output_path)

    plt.figure(figsize=(8, 4))
    plt.bar(metrics_df["model"], metrics_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=20)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()