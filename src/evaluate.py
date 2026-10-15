"""
Evaluation and Forecasting Module.
Computes:
- R-squared (R2) score as formulated in paper Eq. (1)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- One-step-ahead rolling forecast & Recursive multi-step forecast
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

from configs.config import Config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate R2, MAE, RMSE, MAPE on actual physical values (kW)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0

    return {
        "R2": float(r2),
        "MAE_kW": float(mae),
        "RMSE_kW": float(rmse),
        "MAPE_pct": float(mape),
    }


def predict_rolling(model: nn.Module, X_tensor: np.ndarray, scaler: Any, device: str = "cpu") -> np.ndarray:
    """
    Perform 1-step-ahead rolling prediction over the sequence dataset.
    Returns:
        y_pred_original: Unscaled predictions in kW.
    """
    model.eval()
    with torch.no_grad():
        x_in = torch.tensor(X_tensor, dtype=torch.float32).to(device)
        preds_scaled = model(x_in).cpu().numpy().reshape(-1, 1)

    # Invert MinMax scaling back to physical kW
    preds_original = scaler.inverse_transform(preds_scaled).flatten()
    return preds_original


def predict_recursive_autoregressive(
    model: nn.Module,
    initial_window_scaled: np.ndarray,
    steps_ahead: int,
    scaler: Any,
    device: str = "cpu",
) -> np.ndarray:
    """
    Autoregressive recursive forecasting as described in paper:
    'In LSTM, the prediction of next time step value occurs one at a time and update the LSTM network at each prediction.
    The first prediction is made using the last observation of training.
    Subsequently, the previous predicted values were being used as input for next prediction steps.'

    Args:
        model: Trained ElectricityLSTM
        initial_window_scaled: Scaled values of shape (60, 1) or (60,)
        steps_ahead: Number of future hours to forecast (e.g. 876 test hours)
        scaler: Fitted MinMaxScaler
        device: 'cpu' or 'cuda'

    Returns:
        forecast_original: Unscaled multi-step forecasts in kW.
    """
    model.eval()
    current_window = initial_window_scaled.copy().reshape(1, -1, 1)
    predictions_scaled = []

    with torch.no_grad():
        for _ in range(steps_ahead):
            x_in = torch.tensor(current_window, dtype=torch.float32).to(device)
            next_pred_scaled = model(x_in).cpu().numpy().item()
            predictions_scaled.append(next_pred_scaled)

            # Slide window: drop oldest, append newly predicted value
            new_window = np.append(current_window[0, 1:, 0], next_pred_scaled)
            current_window = new_window.reshape(1, -1, 1)

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    forecast_original = scaler.inverse_transform(predictions_scaled).flatten()
    return forecast_original
