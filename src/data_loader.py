"""
Data Loading, Preprocessing, and Sequence Windowing Module.
Paper specification:
- Hospital Building hourly dataset (8760 hours).
- Sequence length p = 60 time steps.
- Normalization into [0, 1] range.
- 90% training, 10% testing, with internal validation split.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import Config


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sliding windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_raw_data(data_path: str, target_column: str) -> pd.DataFrame:
    """Load raw dataset from Excel file and validate structure."""
    df = pd.read_excel(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Columns: {df.columns.tolist()}")
    return df


def verify_dataset_integrity(df: pd.DataFrame, target_column: str, date_column: str) -> Dict[str, float]:
    """Verify monthly averages against Table 2 in the research paper."""
    # Date/Time format: '01/01  01:00:00'
    month_series = df[date_column].astype(str).str.strip().str.split("/").str[0]
    monthly_means = df.groupby(month_series)[target_column].mean().to_dict()
    return monthly_means


def create_sliding_windows(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform 1D time series into 3D LSTM input tensor [samples, seq_len, 1]
    and target vector [samples, 1].
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def prepare_data(config: Config) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    1. Load data
    2. Chronological Train (90%) and Test (10%) split
    3. Internal validation split from training set
    4. Fit MinMaxScaler on train only
    5. Generate sliding windows (seq_len = 60)
    6. Return PyTorch DataLoaders and metadata
    """
    df = load_raw_data(str(config.data_path), config.target_column)
    raw_values = df[config.target_column].values.astype(np.float32)
    n_total = len(raw_values)

    # 90% train, 10% test split as in paper
    n_train_total = int(n_total * config.train_ratio)
    train_total_values = raw_values[:n_train_total]
    test_values = raw_values[n_train_total:]

    # Internal validation split from training portion
    val_size = int(len(train_total_values) * config.val_ratio)
    train_values = train_total_values[:-val_size]
    val_values = train_total_values[-val_size:]

    # Min-Max Scaling [0, 1] (fitted on training data only)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()

    # In time series with sliding windows, we also want the sequence preceding the test split
    # so test starts right at index n_train_total
    test_with_lookback = np.concatenate([train_total_values[-config.sequence_length:], test_values])
    test_with_lookback_scaled = scaler.transform(test_with_lookback.reshape(-1, 1)).flatten()

    # Generate sliding window sequences
    X_train, y_train = create_sliding_windows(train_scaled, config.sequence_length)
    X_val, y_val = create_sliding_windows(val_scaled, config.sequence_length)
    X_test, y_test = create_sliding_windows(test_with_lookback_scaled, config.sequence_length)

    # Create PyTorch DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return {
        "df": df,
        "raw_values": raw_values,
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_values": train_values,
        "val_values": val_values,
        "test_values": test_values,
        "n_train_total": n_train_total,
    }
