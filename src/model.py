"""LSTM architecture for multi-step closing-price forecasting."""

import torch
import torch.nn as nn

from src import config


class LSTMForecaster(nn.Module):
    """Two-layer LSTM followed by a small MLP head that outputs `output_size`
    consecutive daily closing prices in one forward pass (direct multi-step
    forecasting, not recursive one-step-at-a-time prediction).
    """

    def __init__(self, input_size: int = 1, hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS, output_size: int = config.FORECAST_HORIZON,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # (batch, hidden_size) — final time step's hidden state
        return self.head(last_step)
