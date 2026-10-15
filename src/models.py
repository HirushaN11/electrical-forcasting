"""
LSTM Neural Network Architecture Module.
Matches the architecture specified in the research paper:
- Recurrent LSTM layers
- Dropout regularization to prevent overfitting
- Linear dense projection head for regression
"""

import torch
import torch.nn as nn
from configs.config import Config


class ElectricityLSTM(nn.Module):
    """
    Stacked LSTM Regression Network for Hourly Electricity Consumption Forecasting.
    Input shape: (batch_size, sequence_length=60, input_features=1)
    Output shape: (batch_size, output_size=1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        # Note: PyTorch applies dropout between LSTM layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output Dropout & Fully Connected Layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            out: Predicted electricity consumption of shape (batch_size, output_size)
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take output of the last time step
        last_step_out = lstm_out[:, -1, :]

        # Apply dropout and linear projection
        dropped = self.dropout(last_step_out)
        out = self.fc(dropped)
        return out


def build_model(config: Config) -> ElectricityLSTM:
    """Factory helper to instantiate ElectricityLSTM from config."""
    model = ElectricityLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_size=config.output_size,
    )
    return model.to(config.device)
