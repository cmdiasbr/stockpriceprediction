"""Train the LSTM forecaster and save the model + scaler + loss curve.

    python -m src.train
    python -m src.train --ticker PETR4.SA --epochs 80 --batch-size 16
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config, data_loader, preprocessing
from src.model import LSTMForecaster


def train(ticker: str = config.TICKER, epochs: int = config.EPOCHS,
          batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
          patience: int = config.EARLY_STOPPING_PATIENCE) -> LSTMForecaster:
    torch.manual_seed(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = data_loader.load(ticker)
    train_df, test_df = preprocessing.time_based_split(df)
    # carve a validation tail out of the training split for early stopping —
    # the test split stays untouched until evaluate.py
    train_df, val_df = preprocessing.time_based_split(train_df, test_size=0.15)

    scaler = preprocessing.fit_scaler(train_df["Close"].values)
    preprocessing.save_scaler(scaler, ticker)

    X_train, y_train, X_val, y_val = preprocessing.prepare_windows(train_df, val_df, scaler)
    print(f"train windows: {X_train.shape[0]}  |  val windows: {X_val.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement in {patience} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), config.model_path(ticker))
    print(f"best val_loss={best_val_loss:.5f} — saved to {config.model_path(ticker)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.set_title(f"Training curve — {ticker}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"training_curve_{ticker.replace('.', '-')}.png")
    plt.close(fig)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM stock price forecaster")
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    train(args.ticker, args.epochs, args.batch_size, args.lr)
