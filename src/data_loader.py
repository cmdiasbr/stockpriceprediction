"""Load historical OHLCV data, either from the bundled CSV or fresh from Yahoo Finance.

Run directly to (re)download the default ticker and overwrite the local CSV:
    python -m src.data_loader
    python -m src.data_loader --ticker VALE3.SA --start 2015-01-01 --end 2024-12-31
"""

import argparse

import pandas as pd

from src import config


def download(ticker: str = config.TICKER, start: str = config.DATA_START, end: str = config.DATA_END) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and save it to data/<ticker>.csv."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} between {start} and {end}")

    df = df.reset_index()
    df.to_csv(config.csv_path(ticker), index=False)
    return df


def load(ticker: str = config.TICKER) -> pd.DataFrame:
    """Load OHLCV data from the local CSV, downloading it first if it doesn't exist yet."""
    path = config.csv_path(ticker)
    if not path.exists():
        df = download(ticker)
    else:
        df = pd.read_csv(path, parse_dates=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    missing = df["Close"].isna().sum()
    if missing:
        # a handful of exchange-holiday rows come through from yfinance as
        # fully empty OHLCV rows rather than being absent — drop them rather
        # than let NaNs propagate into the scaler and blow up training
        print(f"dropping {missing} row(s) with missing Close price (likely exchange holidays)")
        df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock data from Yahoo Finance")
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument("--start", default=config.DATA_START)
    parser.add_argument("--end", default=config.DATA_END)
    args = parser.parse_args()

    df = download(args.ticker, args.start, args.end)
    print(f"saved {len(df)} rows to {config.csv_path(args.ticker)}")
