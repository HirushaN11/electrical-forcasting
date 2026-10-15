"""
Master Experiment Runner.
Executes the end-to-end pipeline to replicate the findings of:
'Deep learning-based forecasting of electricity consumption' (Scientific Reports 2024).

Usage:
  # Train and benchmark all 6 optimizers (Table 3 & Table 4 in paper):
  python run_experiments.py --all_optimizers

  # Train only the primary Adam model:
  python run_experiments.py --optimizer Adam --epochs 60
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure workspace root is in python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from configs.config import Config
from src.data_loader import prepare_data, verify_dataset_integrity
from src.models import build_model
from src.trainer import Trainer, get_optimizer
from src.evaluate import compute_metrics, predict_rolling, predict_recursive_autoregressive
from src.visualize import (
    plot_figure_1_load_profile,
    plot_training_loss,
    plot_figure_4_test_comparison,
    plot_all_optimizers_grid,
)


def verify_paper_table_2(df: pd.DataFrame, config: Config) -> None:
    """Check dataset monthly averages against Table 2 of the research paper."""
    print("\n" + "=" * 65)
    print("STEP 1: Verifying Dataset Against Research Paper Table 2")
    print("=" * 65)
    paper_table_2 = {
        "01": 1189.23, "02": 1166.03, "03": 1190.21, "04": 1182.77,
        "05": 1163.78, "06": 1137.14, "07": 1111.31, "08": 1132.36,
        "09": 1114.51, "10": 1143.15, "11": 1191.02, "12": 1155.29,
    }
    actual_means = verify_dataset_integrity(df, config.target_column, config.date_column)

    print(f"{'Month':<8}{'Paper Table 2 [kW]':<22}{'Our Dataset [kW]':<22}{'Match':<10}")
    print("-" * 65)
    all_matched = True
    for m, paper_val in paper_table_2.items():
        actual_val = actual_means.get(m, 0.0)
        is_close = abs(paper_val - round(actual_val, 2)) < 0.05
        if not is_close:
            all_matched = False
        print(f"{m:<8}{paper_val:<22.2f}{actual_val:<22.2f}{'[MATCH]' if is_close else '[DIFF]'}")
    print("-" * 65)
    if all_matched:
        print("[SUCCESS] Dataset exactly matches the paper's Phoenix Hospital dataset.\n")
    else:
        print("! Notice: Slight variation detected in monthly averages.\n")


def run_single_optimizer(
    opt_name: str,
    opt_params: dict,
    data_bundle: dict,
    config: Config,
) -> dict:
    """Train, evaluate, and plot for a single optimizer setting."""
    print(f"\n>>> Running Optimizer: {opt_name} with params: {opt_params}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. Instantiate Model
    model = build_model(config)

    # 2. Setup Optimizer and Loss
    optimizer = get_optimizer(model, opt_name, opt_params)
    criterion = nn.MSELoss()

    # 3. Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        optimizer_name=opt_name,
    )
    train_results = trainer.fit(
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        verbose=True,
    )

    # 4. Save Training Loss Curve
    loss_plot_path = config.figures_dir / f"{opt_name.lower()}_loss.png"
    plot_training_loss(train_results, opt_name, str(loss_plot_path))

    # 5. Evaluate on Test Split
    test_preds = predict_rolling(model, data_bundle["X_test"], data_bundle["scaler"], config.device)
    test_actuals = data_bundle["test_values"]

    # Compute metrics on unscaled kW values
    metrics = compute_metrics(test_actuals, test_preds)
    print(f"[{opt_name} Test Results] R2: {metrics['R2']*100:.2f}% | MAE: {metrics['MAE_kW']:.2f} kW | RMSE: {metrics['RMSE_kW']:.2f} kW | MAPE: {metrics['MAPE_pct']:.2f}%")

    # 6. Save Figure 4 for this optimizer
    fig4_path = config.figures_dir / f"figure_4_{opt_name.lower()}.png"
    plot_figure_4_test_comparison(test_actuals, test_preds, opt_name, metrics["R2"], str(fig4_path))

    return {
        "optimizer_name": opt_name,
        "metrics": metrics,
        "predictions": test_preds,
        "train_results": train_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Replicate Deep Learning Electricity Forecasting Paper")
    parser.add_argument("--all_optimizers", action="store_true", help="Benchmark all 6 optimizers (SGD, NAdam, Adadelta, Adagrad/Ftrl, RMSprop, Adam)")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Single optimizer to run (SGD, NAdam, Adadelta, Adagrad, RMSprop, Adam)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = Config()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Ensure output directories exist
    os.makedirs(config.artifacts_dir, exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    print("=" * 65)
    print("DEEP LEARNING ELECTRICITY CONSUMPTION FORECASTING")
    print(f"Device: {config.device.upper()} | Lookback Lag p: {config.sequence_length} hours | Epochs: {config.num_epochs}")
    print("=" * 65)

    # Step 1: Prepare data and verify monthly averages
    data_bundle = prepare_data(config)
    verify_paper_table_2(data_bundle["df"], config)

    # Step 2: Reproduce Figure 1 (Full Load Profile)
    fig1_path = config.figures_dir / "figure_1_load_profile.png"
    plot_figure_1_load_profile(data_bundle["raw_values"], str(fig1_path))

    # Step 3: Run Optimizer Experiment(s)
    if args.all_optimizers:
        print("\n" + "=" * 65)
        print("STEP 2: Benchmarking All 6 Optimizers (Replicating Paper Table 3 & 4)")
        print("=" * 65)
        predictions_dict = {}
        metrics_dict = {}

        for opt_name, opt_params in config.optimizers:
            res = run_single_optimizer(opt_name, opt_params, data_bundle, config)
            predictions_dict[opt_name] = res["predictions"]
            metrics_dict[opt_name] = res["metrics"]

        # Step 4: Produce Comparative Benchmark Table (Table 4)
        print("\n" + "=" * 70)
        print("FINAL RESULTS: Replicating Paper Table 4 (Objective Function Results)")
        print("=" * 70)
        paper_r2_ref = {
            "SGD": "-5%", "NAdam": "14%", "Adadelta": "46%",
            "Adagrad": "49% (Ftrl)", "RMSprop": "93%", "Adam": "95%"
        }
        print(f"{'Setting':<10}{'Optimizer':<12}{'Paper R2':<15}{'Our Model R2':<16}{'MAE [kW]':<12}{'RMSE [kW]':<12}")
        print("-" * 70)
        for idx, (opt_name, _) in enumerate(config.optimizers):
            m = metrics_dict[opt_name]
            ref = paper_r2_ref.get(opt_name, "N/A")
            print(f"Setting {idx+1:<3}{opt_name:<12}{ref:<15}{m['R2']*100:<15.2f}%{m['MAE_kW']:<12.2f}{m['RMSE_kW']:<12.2f}")
        print("-" * 70)

        # Step 5: Save Combined 6-Optimizer Comparison Grid
        grid_path = config.figures_dir / "figure_4_all_optimizers_grid.png"
        plot_all_optimizers_grid(data_bundle["test_values"], predictions_dict, metrics_dict, str(grid_path))

        # Save summary CSV
        summary_rows = []
        for idx, (opt_name, _) in enumerate(config.optimizers):
            m = metrics_dict[opt_name]
            summary_rows.append({
                "Setting": f"Setting {idx+1}",
                "Optimizer": opt_name,
                "Paper_R2": paper_r2_ref.get(opt_name, "N/A"),
                "Replicated_R2_pct": round(m["R2"] * 100, 2),
                "MAE_kW": round(m["MAE_kW"], 2),
                "RMSE_kW": round(m["RMSE_kW"], 2),
                "MAPE_pct": round(m["MAPE_pct"], 2),
            })
        summary_df = pd.DataFrame(summary_rows)
        csv_path = config.artifacts_dir / "benchmark_results_table_4.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary table saved to: {csv_path}")

    else:
        # Run single optimizer (default: Adam)
        opt_name = args.optimizer
        opt_params = dict(config.optimizers).get(opt_name, {"lr": config.learning_rate})
        res = run_single_optimizer(opt_name, opt_params, data_bundle, config)

    print("\n[COMPLETE] Pipeline execution complete!")


if __name__ == "__main__":
    main()
