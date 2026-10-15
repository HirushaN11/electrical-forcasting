"""
Visualization Module to reproduce plots and figures from the research paper:
- Figure 1: Electricity load profile of the 8760 hours.
- Figure 3(a-f): Training data and forecasted values.
- Figure 4(a-f): Test data and forecasted values across optimizer settings.
- Training Loss Curves.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import Config


# Set modern aesthetic plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_figure_1_load_profile(raw_values: np.ndarray, save_path: str = None) -> None:
    """
    Reproduces Figure 1 from the paper:
    'The electricity load profile of dataset for the 8760 h.'
    """
    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    ax.plot(raw_values, color="#1f77b4", linewidth=0.8, alpha=0.9, label="Electricity Consumption")
    ax.set_title("Figure 1: Electricity Load Profile for 8,760 Hours (Phoenix Hospital)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Hourly Steps)", fontsize=11)
    ax.set_ylabel("Power Consumption [kW]", fontsize=11)
    ax.set_xlim(0, len(raw_values))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure 1 saved to: {save_path}")
    plt.close()


def plot_training_loss(loss_history: Dict[str, List[float]], optimizer_name: str, save_path: str = None) -> None:
    """Plot Train Loss vs. Validation Loss across training epochs."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    epochs = range(1, len(loss_history["train_losses"]) + 1)
    ax.plot(epochs, loss_history["train_losses"], label="Train Loss (MSE)", color="#2ca02c", linewidth=1.5)
    ax.plot(epochs, loss_history["val_losses"], label="Validation Loss (MSE)", color="#d62728", linewidth=1.5, linestyle="--")
    ax.set_title(f"Training Dynamics - Optimizer: {optimizer_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_figure_4_test_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    optimizer_name: str,
    r2_score: float,
    save_path: str = None,
) -> None:
    """
    Reproduces Figure 4(a-f) from the paper:
    'Test data and forecasted values at step setting'
    """
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    ax.plot(y_true, label="Actual Electricity Load [kW]", color="#1f77b4", linewidth=1.5, alpha=0.85)
    ax.plot(y_pred, label=f"Forecasted ({optimizer_name}, $R^2$={r2_score*100:.1f}%)", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title(f"Figure 4: Test Data vs. Forecasted Values ({optimizer_name})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Test Horizon (Hourly Steps)", fontsize=10)
    ax.set_ylabel("Electricity [kW]", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure 4 ({optimizer_name}) saved to: {save_path}")
    plt.close()


def plot_all_optimizers_grid(
    test_actuals: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str = None,
) -> None:
    """
    Combines all 6 optimizer predictions into a 3x2 grid of subplots
    mirroring Figure 4(a-f) of the research paper.
    """
    optimizers = list(predictions_dict.keys())
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 11), dpi=300, sharex=True)
    axes = axes.flatten()

    for idx, opt in enumerate(optimizers):
        ax = axes[idx]
        y_pred = predictions_dict[opt]
        r2 = metrics_dict[opt]["R2"]
        ax.plot(test_actuals, label="Actual", color="#1f77b4", linewidth=1.1, alpha=0.8)
        ax.plot(y_pred, label=f"{opt} ($R^2$={r2*100:.1f}%)", color="#d62728", linewidth=1.1, linestyle="--")
        ax.set_title(f"({chr(97 + idx)}) Setting {idx+1}: {opt}", fontsize=11, fontweight="bold")
        ax.set_ylabel("kW", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Hours", fontsize=10)
    axes[-2].set_xlabel("Hours", fontsize=10)
    fig.suptitle("Figure 4 (a-f): Comparative Forecasts Across 6 Optimizers on Test Horizon", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"All Optimizers Comparison Grid saved to: {save_path}")
    plt.close()
