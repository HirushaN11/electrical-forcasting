"""
Configuration settings for the Deep Learning Electricity Consumption Forecasting Project.
Based on: 'Deep learning-based forecasting of electricity consumption', Scientific Reports (2024) 14:6489.
"""

from dataclasses import dataclass
from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # File Paths
    data_path: Path = BASE_DIR / "Hospital Building Dataset.xlsx"
    paper_pdf_path: Path = BASE_DIR / "s41598-024-56602-4 (1).pdf"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    checkpoints_dir: Path = BASE_DIR / "artifacts" / "checkpoints"
    figures_dir: Path = BASE_DIR / "artifacts" / "figures"

    # Dataset & Preprocessing Settings
    target_column: str = "Electricity:Facility [kW](Hourly)"
    date_column: str = "Date/Time"
    total_hours: int = 8760
    sequence_length: int = 60          # p = 60 time lags as specified in paper
    train_ratio: float = 0.90          # 90% train (as specified in paper)
    test_ratio: float = 0.10           # 10% test (as specified in paper)
    val_ratio: float = 0.10            # Internal validation split from training set

    # Model Hyperparameters
    input_size: int = 1                # Univariate time series
    hidden_size: int = 64              # Level I: hidden units
    num_layers: int = 2                # Level II: hidden layers
    dropout: float = 0.2               # Dropout rate
    output_size: int = 1               # Next hourly consumption forecast

    # Training Settings
    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 0.001
    early_stopping_patience: int = 12
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Optimizer Settings evaluated in paper (Table 3)
    optimizers: tuple = (
        ("SGD", {"lr": 0.01, "momentum": 0.9}),
        ("NAdam", {"lr": 0.001}),
        ("Adadelta", {"lr": 1.0}),
        ("Adagrad", {"lr": 0.01}),       # PyTorch equivalent for Ftrl
        ("RMSprop", {"lr": 0.001}),
        ("Adam", {"lr": 0.001})          # Benchmark setting (Paper target R2 ~ 95%)
    )
