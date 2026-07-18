"""Evaluate a trained model on the held-out test split.

    python -m src.evaluate
    python -m src.evaluate --ticker PETR4.SA
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import config, data_loader, preprocessing
from src.model import LSTMForecaster


def load_model(ticker: str = config.TICKER) -> LSTMForecaster:
    model = LSTMForecaster()
    model.load_state_dict(torch.load(config.model_path(ticker), map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate(ticker: str = config.TICKER):
    df = data_loader.load(ticker)
    train_df, test_df = preprocessing.time_based_split(df)

    scaler = preprocessing.load_scaler(ticker)
    _, _, X_test, y_test = preprocessing.prepare_windows(train_df, test_df, scaler)

    model = load_model(ticker)
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    horizon = actual.shape[1]
    per_day = []
    for h in range(horizon):
        rmse = np.sqrt(mean_squared_error(actual[:, h], preds[:, h]))
        mae = mean_absolute_error(actual[:, h], preds[:, h])
        per_day.append({"day_ahead": h + 1, "rmse": round(rmse, 3), "mae": round(mae, 3)})

    metrics_df = pd.DataFrame(per_day)
    print(f"\n{ticker} — test set: {len(actual)} windows, {len(test_df)} trading days")
    print(metrics_df.to_string(index=False))

    overall_rmse = np.sqrt(mean_squared_error(actual.flatten(), preds.flatten()))
    print(f"\noverall RMSE (all {horizon} horizon days pooled): {overall_rmse:.3f}")

    _plot_day_ahead(test_df, actual, preds, ticker)
    _plot_rmse_by_horizon(metrics_df, ticker)

    return metrics_df


def _plot_day_ahead(test_df: pd.DataFrame, actual: np.ndarray, preds: np.ndarray, ticker: str, day: int = 1):
    """Plot actual vs. predicted closing price, `day` trading days ahead."""
    idx = day - 1
    dates = test_df["Date"].values[idx:idx + len(actual)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.asarray(dates), actual[:, idx], label="actual", color="#2a9d8f")
    ax.plot(np.asarray(dates), preds[:, idx], label="predicted", color="#e76f51", linestyle="--")
    ax.set_title(f"{ticker} — actual vs. predicted close, {day} day ahead")
    ax.set_xlabel("date")
    ax.set_ylabel("close price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"eval_day{day}_{ticker.replace('.', '-')}.png")
    plt.close(fig)


def _plot_rmse_by_horizon(metrics_df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics_df["day_ahead"], metrics_df["rmse"], color="#6d5ce7")
    ax.set_xlabel("days ahead")
    ax.set_ylabel("RMSE (price units)")
    ax.set_title(f"{ticker} — forecast error by horizon")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"eval_rmse_by_horizon_{ticker.replace('.', '-')}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM forecaster on the test split")
    parser.add_argument("--ticker", default=config.TICKER)
    args = parser.parse_args()

    evaluate(args.ticker)
