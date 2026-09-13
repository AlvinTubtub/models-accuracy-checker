"""PyTorch univariate LSTM forecasting model."""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnivariateLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, 1]
        out, (hn, _) = self.lstm(x)
        # Take last time step
        last_out = out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction


def _create_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - lookback):
        x = data[i : i + lookback]
        y = data[i + lookback]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_and_predict_lstm(
    dev_closes: np.ndarray,
    eval_closes: np.ndarray,
    *,
    lookback: int = 10,
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 40,
    seed: int = 42,
    device: str | None = None,
) -> tuple[list[float], dict]:
    """Train univariate LSTM on development series and walk-forward evaluate."""
    _set_deterministic(seed)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    dev_array = np.array(dev_closes, dtype=np.float32)
    min_val = float(dev_array.min())
    max_val = float(dev_array.max())
    denom = max_val - min_val if max_val > min_val else 1.0

    # MinMax scaling in [0, 1]
    dev_scaled = (dev_array - min_val) / denom

    X_tr, y_tr = _create_sequences(dev_scaled, lookback)
    X_tr = X_tr[..., np.newaxis]  # [N, lookback, 1]

    dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UnivariateLSTM(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Walk-forward one-step-ahead evaluation
    model.eval()
    predictions: list[float] = []

    # Combine dev and eval to feed genuine lagged windows for each evaluation step
    full_closes = np.concatenate([dev_closes, eval_closes])
    full_scaled = (full_closes - min_val) / denom

    eval_len = len(eval_closes)
    dev_len = len(dev_closes)

    with torch.no_grad():
        for i in range(eval_len):
            # Sequence ending at dev_len + i - 1
            idx_end = dev_len + i
            seq = full_scaled[idx_end - lookback : idx_end]
            inp = torch.from_numpy(seq).view(1, lookback, 1).float().to(device)
            scaled_pred = model(inp).item()

            # Inverse scale
            orig_pred = scaled_pred * denom + min_val
            predictions.append(round(float(orig_pred), 4))

    metadata = {
        "lookback": lookback,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
    }
    return predictions, metadata
