"""Forecast the next FORECAST_HORIZON trading days from the most recent data.

    python -m src.predict
    python -m src.predict --ticker PETR4.SA --history-days 90
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config, data_loader, preprocessing
from src.evaluate import load_model


def predict_next(ticker: str = config.TICKER, history_days: int = 90):
    df = data_loader.load(ticker)
    scaler = preprocessing.load_scaler(ticker)
    model = load_model(ticker)

    last_window = df["Close"].values[-config.SEQ_LEN:]
    if len(last_window) < config.SEQ_LEN:
        raise ValueError(
            f"need at least {config.SEQ_LEN} days of history, got {len(last_window)}"
        )

    scaled_window = scaler.transform(last_window.reshape(-1, 1)).reshape(1, config.SEQ_LEN, 1)

    with torch.no_grad():
        pred_scaled = model(torch.tensor(scaled_window, dtype=torch.float32)).numpy()

    forecast = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast))

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": forecast.round(2)})
    print(f"\n{ticker} — last close on {last_date.date()}: {df['Close'].iloc[-1]:.2f}")
    print(forecast_df.to_string(index=False))

    _plot_forecast(df, forecast_df, ticker, history_days)
    forecast_df.to_csv(config.FIGURES_DIR / f"forecast_{ticker.replace('.', '-')}.csv", index=False)

    return forecast_df


def _plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str, history_days: int):
    recent = df.tail(history_days)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent["Date"].to_numpy(), recent["Close"].to_numpy(), label="history", color="#2a9d8f")
    ax.plot(forecast_df["Date"].to_numpy(), forecast_df["Predicted_Close"].to_numpy(), label="forecast",
            color="#e76f51", linestyle="--", marker="o")
    ax.axvline(recent["Date"].iloc[-1], color="grey", linewidth=0.8, linestyle=":")
    ax.set_title(f"{ticker} — {len(forecast_df)}-day forecast")
    ax.set_xlabel("date")
    ax.set_ylabel("close price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"forecast_{ticker.replace('.', '-')}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast the next N trading days")
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument("--history-days", type=int, default=90, help="days of history to show in the plot")
    args = parser.parse_args()

    predict_next(args.ticker, args.history_days)
