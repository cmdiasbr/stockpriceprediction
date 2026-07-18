"""Scaling, sequence windowing and train/test splitting for the LSTM forecaster."""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config


def time_based_split(df: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Split a time-ordered dataframe by position, not randomly.

    Shuffling before splitting a time series leaks future information into
    training and makes the reported test error meaningless.
    """
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def fit_scaler(train_close: np.ndarray) -> MinMaxScaler:
    """Fit the scaler on the training split only."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_close.reshape(-1, 1))
    return scaler


def create_sequences(scaled_close: np.ndarray, seq_len: int = config.SEQ_LEN,
                      horizon: int = config.FORECAST_HORIZON):
    """Build (X, y) windows: X = past `seq_len` days, y = next `horizon` days.

    scaled_close: 1D array of scaled closing prices.
    Returns X with shape (n_samples, seq_len, 1) and y with shape (n_samples, horizon).
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_close) - horizon + 1):
        X.append(scaled_close[i - seq_len:i])
        y.append(scaled_close[i:i + horizon])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, horizon)
    return X, y


def prepare_windows(train_df: pd.DataFrame, test_df: pd.DataFrame, scaler: MinMaxScaler,
                     seq_len: int = config.SEQ_LEN, horizon: int = config.FORECAST_HORIZON):
    """Scale both splits and build (X, y) windows for train and test.

    The last `seq_len` days of the training split are prepended to the test
    split before windowing, so the first test window has real trailing
    history instead of starting from a cold start.
    """
    train_scaled = scaler.transform(train_df["Close"].values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_df["Close"].values.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, seq_len, horizon)

    test_with_context = np.concatenate([train_scaled[-seq_len:], test_scaled])
    X_test, y_test = create_sequences(test_with_context, seq_len, horizon)

    return X_train, y_train, X_test, y_test


def save_scaler(scaler: MinMaxScaler, ticker: str = config.TICKER) -> None:
    with open(config.scaler_path(ticker), "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(ticker: str = config.TICKER) -> MinMaxScaler:
    with open(config.scaler_path(ticker), "rb") as f:
        return pickle.load(f)
