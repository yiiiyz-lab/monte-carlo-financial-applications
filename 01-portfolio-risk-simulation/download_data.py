
from pathlib import Path
import pandas as pd
import yfinance as yf

# Portfolio assets
TICKERS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "GLD"]

# Historical data window
START_DATE = "2020-01-01"

# Output directory
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download adjusted daily prices
prices = yf.download(
    TICKERS,
    start=START_DATE,
    auto_adjust=True,
    progress=False,
)["Close"]

# Ensure consistent column ordering
prices = prices[TICKERS]

# Check for missing observations
if prices.empty:
    raise ValueError("No price data downloaded.")

if prices.isna().any().any():
    raise ValueError(
        "Missing prices detected. Inspect the data before proceeding."
    )

# Calculate daily log returns
log_returns = (
    pd.DataFrame(
        __import__("numpy").log(prices / prices.shift(1)),
        index=prices.index,
        columns=prices.columns,
    )
    .dropna()
)

# Save results
prices.to_csv(DATA_DIR / "adjusted_prices.csv")
log_returns.to_csv(DATA_DIR / "daily_log_returns.csv")

print("Data download successful!")
print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Observations: {len(prices)}")
print("\nLatest adjusted prices:")
print(prices.tail())
