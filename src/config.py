"""Central configuration — every script imports paths and hyperparameters from here."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

for d in (DATA_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Default asset. yfinance ticker format ("PETR4.SA" for B3); the bundled CSV
# uses the same naming convention.
TICKER = "PETR4.SA"
DATA_START = "2015-12-08"
DATA_END = "2020-12-08"

# Sequence-to-sequence framing: look back SEQ_LEN days to forecast the next
# FORECAST_HORIZON days of closing price.
SEQ_LEN = 60
FORECAST_HORIZON = 5

# Time-based split — the last TEST_SIZE fraction of the series is held out.
# Never shuffle a time series before splitting; it leaks the future into training.
TEST_SIZE = 0.15

# Model / training hyperparameters
HIDDEN_SIZE = 50
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8

RANDOM_STATE = 42


def csv_path(ticker: str = TICKER) -> Path:
    return DATA_DIR / f"{ticker.replace('.', '-')}.csv"


def model_path(ticker: str = TICKER) -> Path:
    return MODELS_DIR / f"lstm_{ticker.replace('.', '-')}.pt"


def scaler_path(ticker: str = TICKER) -> Path:
    return MODELS_DIR / f"scaler_{ticker.replace('.', '-')}.pkl"
