# Stock Price Prediction — LSTM

Multi-step stock price forecasting with an LSTM in PyTorch: given the last 60 days of closing price, predict the next 5 trading days. Originally a take-home exercise ("analyze one asset's historical time series with Yahoo Finance, extract a pattern, forecast the next 5 days"); rebuilt here as a proper modular pipeline — data, preprocessing, model, training, evaluation and inference all as separate, independently runnable scripts instead of a single notebook.

Default asset: **PETR4.SA** (Petrobras, B3) — swappable for any Yahoo Finance ticker via `--ticker`.

## Results

| days ahead | RMSE (R$) | MAE (R$) |
|---|---|---|
| 1 | 1.21 | 0.81 |
| 2 | 1.37 | 0.91 |
| 3 | 1.47 | 1.04 |
| 4 | 1.59 | 1.09 |
| 5 | 1.68 | 1.16 |

Error grows with forecast distance, as expected. On the held-out test period (Mar–Dec 2020, which includes the COVID crash and recovery), the model tracks the overall trend well but — like any close-price-only LSTM — lags sharp reversals by a day or two rather than anticipating them; it's a trend follower, not a crystal ball. See `outputs/figures/eval_day1_PETR4-SA.png` for the actual-vs-predicted plot and `outputs/figures/training_curve_PETR4-SA.png` for the training/validation loss curve.

## Project structure

```
stockpriceprediction/
├── data/
│   └── PETR4-SA.csv              # bundled historical OHLCV data (2015–2020), used if present
│
├── src/
│   ├── config.py                 # every path and hyperparameter — the single place to change ticker,
│   │                              #   sequence length, forecast horizon, model size, etc.
│   ├── data_loader.py             # download(): pulls OHLCV data from Yahoo Finance
│   │                              # load(): reads the local CSV (downloading it first if missing),
│   │                              #   drops rows with missing Close (exchange holidays)
│   ├── preprocessing.py           # time_based_split(): chronological train/test split (never shuffled)
│   │                              # fit_scaler() / save_scaler() / load_scaler(): MinMax scaling,
│   │                              #   always fit on the training split only
│   │                              # create_sequences(): builds (60-day window → 5-day target) pairs
│   │                              # prepare_windows(): scales + windows train and test together,
│   │                              #   carrying the last 60 training days into the test windows as context
│   ├── model.py                   # LSTMForecaster — 2-layer LSTM + small MLP head, outputs 5 values
│   │                              #   in a single forward pass (direct multi-step, not recursive)
│   ├── train.py                   # trains the model with early stopping, saves the model weights,
│   │                              #   the fitted scaler, and a loss-curve plot
│   ├── evaluate.py                 # loads the trained model, scores it on the untouched test split,
│   │                              #   reports RMSE/MAE per forecast day, saves evaluation plots
│   └── predict.py                 # loads the trained model and the most recent 60 days of data,
│                                  #   forecasts the next 5 trading days, saves a plot + CSV
│
├── outputs/
│   ├── models/
│   │   ├── lstm_PETR4-SA.pt      # trained model weights (produced by train.py)
│   │   └── scaler_PETR4-SA.pkl   # fitted MinMaxScaler (produced by train.py, reused by evaluate/predict)
│   └── figures/
│       ├── training_curve_PETR4-SA.png
│       ├── eval_day1_PETR4-SA.png
│       ├── eval_rmse_by_horizon_PETR4-SA.png
│       ├── forecast_PETR4-SA.png
│       └── forecast_PETR4-SA.csv
│
├── requirements.txt
└── README.md
```

## Running it

All commands run from the repo root, as modules (`python -m src.<script>`) so the `src/` package imports resolve correctly.

```bash
git clone https://github.com/cmdiasbr/stockpriceprediction.git
cd stockpriceprediction
pip install -r requirements.txt
```

**1. Get the data** — optional, a CSV for PETR4.SA (2015–2020) is already bundled in `data/`. To refresh it or fetch a different asset/date range from Yahoo Finance:

```bash
python -m src.data_loader --ticker PETR4.SA --start 2015-01-01 --end 2024-12-31
```

**2. Train the model** — reads `data/<ticker>.csv`, trains with early stopping, writes the model, scaler and loss curve to `outputs/`:

```bash
python -m src.train --ticker PETR4.SA --epochs 60 --batch-size 32
```

**3. Evaluate on the test split** — RMSE/MAE per forecast day, printed and plotted:

```bash
python -m src.evaluate --ticker PETR4.SA
```

**4. Forecast the next 5 trading days** from the most recent data:

```bash
python -m src.predict --ticker PETR4.SA
```

Each script also runs with just `python -m src.<script>` — every argument has a sensible default pulled from `src/config.py`.

## Methodology notes

- **Chronological split, not random.** `preprocessing.time_based_split` slices by position; shuffling a time series before splitting leaks future prices into training and makes the reported error meaningless. The last 60 training days are carried into the test windows as context, so the first test prediction isn't a cold start.
- **Scaler fit on training data only.** Fitting `MinMaxScaler` on the full dataset (including test) is a common leakage bug in stock-prediction tutorials — the scaler here only ever sees the training split.
- **Direct multi-step forecasting.** The model outputs all 5 future days in one forward pass, rather than predicting one day and recursively feeding it back in — the latter compounds errors quickly over a 5-day horizon.
- **Data quality check that actually mattered:** the bundled CSV has 3 rows with entirely missing OHLCV values (exchange holidays that came through as empty rows rather than being absent). `data_loader.load()` detects and drops them — leaving them in silently produced `NaN` losses during training.

## Limitations

- Univariate on closing price only — no volume, no macro/index correlation (IBOV), no sentiment. The original assignment's stretch goal of relating the asset to IBOV is a natural next step.
- No hyperparameter search; architecture and sequence length are reasonable defaults, not tuned.
- Backtested on a single asset and a single (turbulent) time window. A serious evaluation would walk this forward across multiple periods and assets before trusting it with anything real.

## Author

**Camila Alves Dias** — Machine Learning Engineer & AI R&D Specialist
[Portfolio](https://cmdiasbr.github.io) · [LinkedIn](https://www.linkedin.com/in/camilalvesdias/) · [GitHub](https://github.com/cmdiasbr)
