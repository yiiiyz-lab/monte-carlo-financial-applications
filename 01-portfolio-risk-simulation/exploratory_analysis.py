
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

TICKERS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "GLD"]

COLORS = {
    "SPY": "#4776E6",
    "QQQ": "#8E54E9",
    "IEF": "#26A69A",
    "TLT": "#00ACC1",
    "LQD": "#F5A623",
    "GLD": "#E76F51",
}

# --------------------------------------------------
# 2. Load historical market data
# --------------------------------------------------

prices = pd.read_csv(
    DATA_DIR / "adjusted_prices.csv",
    index_col=0,
    parse_dates=True,
)[TICKERS]

returns = pd.read_csv(
    DATA_DIR / "daily_log_returns.csv",
    index_col=0,
    parse_dates=True,
)[TICKERS]

assert prices.index.is_monotonic_increasing
assert not prices.isna().any().any()
assert not returns.isna().any().any()

# --------------------------------------------------
# 3. Calculate historical risk statistics
# --------------------------------------------------

TRADING_DAYS = 252

mean_daily = returns.mean()
annualized_vol = returns.std() * np.sqrt(TRADING_DAYS)
annualized_log_return = mean_daily * TRADING_DAYS

cov_matrix = returns.cov()
corr_matrix = returns.corr()

summary = pd.DataFrame({
    "Annualized Log Return": annualized_log_return,
    "Annualized Volatility": annualized_vol,
})

summary.to_csv(DATA_DIR / "historical_risk_summary.csv")
cov_matrix.to_csv(DATA_DIR / "covariance_matrix.csv")
corr_matrix.to_csv(DATA_DIR / "correlation_matrix.csv")

print("\nHistorical Risk Summary:")
print(summary.round(4))

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# --------------------------------------------------
# 4. Professional visualization configuration
# --------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

BG = "#0B1220"
PANEL = "#111C2E"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GRID = "#334155"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "savefig.facecolor": BG,
})

start_label = prices.index.min().strftime("%b %Y")
end_label = prices.index.max().strftime("%b %Y")
period_label = f"{start_label} – {end_label}"

# --------------------------------------------------
# 5. Figure 01: Normalized ETF performance
# --------------------------------------------------

normalized = prices.div(prices.iloc[0]).mul(100)

fig, ax = plt.subplots(figsize=(14, 7.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for ticker in TICKERS:
    ax.plot(
        normalized.index,
        normalized[ticker],
        color=COLORS[ticker],
        linewidth=2.2,
        label=ticker,
        alpha=0.95,
    )

ax.axhline(
    100,
    color=MUTED,
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

ax.set_title(
    "Historical ETF Performance",
    fontsize=20,
    fontweight="bold",
    loc="left",
    pad=35,
)

ax.text(
    0, 1.035,
    f"{period_label}  |  Adjusted prices indexed to 100",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=11,
)

ax.set_ylabel("Normalized Price Index", labelpad=12)
ax.set_xlabel("")

ax.grid(color=GRID, alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)

# Add endpoint labels in a dedicated right-side margin
last_date = normalized.index[-1]

for ticker in TICKERS:
    last_value = normalized[ticker].iloc[-1]
    ax.annotate(
        f"{ticker}  {last_value:.0f}",
        xy=(last_date, last_value),
        xytext=(12, 0),
        textcoords="offset points",
        color=COLORS[ticker],
        fontsize=10,
        fontweight="bold",
        va="center",
        annotation_clip=False,
    )

ax.margins(x=0.02)
ax.set_xlim(
    normalized.index[0],
    normalized.index[-1] + pd.Timedelta(days=180),
)

ax.legend(
    ncol=6,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    frameon=False,
    fontsize=10,
)

fig.text(
    0.08, 0.025,
    "Source: Yahoo Finance via yfinance | "
    "Adjusted closing prices | Historical performance is not predictive",
    color=MUTED,
    fontsize=9,
)

fig.subplots_adjust(
    left=0.08,
    right=0.94,
    top=0.85,
    bottom=0.18,
)

fig.savefig(
    FIGURES_DIR / "01_normalized_etf_performance.png",
    dpi=220,
    facecolor=BG,
    bbox_inches="tight",
)

plt.close(fig)

# --------------------------------------------------
# 6. Figure 02: Correlation heatmap
# --------------------------------------------------

cmap = LinearSegmentedColormap.from_list(
    "risk_correlation",
    ["#3B82F6", "#E2E8F0", "#EF6C5B"],
    N=256,
)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=1.5,
    linecolor=BG,
    annot_kws={
        "fontsize": 11,
        "fontweight": "bold",
    },
    cbar_kws={
        "label": "Pearson Correlation",
        "shrink": 0.82,
        "pad": 0.035,
    },
    ax=ax,
)

# Adjust annotation contrast for readability
for text_obj in heatmap.texts:
    value = float(text_obj.get_text())
    text_obj.set_color(
        "#FFFFFF" if abs(value) > 0.65 else "#172033"
    )

ax.set_title(
    "Asset Return Correlation Matrix",
    fontsize=20,
    fontweight="bold",
    loc="left",
    pad=38,
)

ax.text(
    0, 1.045,
    f"{period_label}  |  Daily log returns",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=11,
)

ax.tick_params(axis="both", length=0, labelsize=11)
ax.set_xticklabels(TICKERS, rotation=0)
ax.set_yticklabels(TICKERS, rotation=0)

cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.label.set_color(MUTED)
cbar.ax.tick_params(colors=MUTED)

fig.text(
    0.08, 0.035,
    "Source: Yahoo Finance via yfinance | "
    "Full-sample Pearson correlations | "
    "Correlations may change under stress",
    color=MUTED,
    fontsize=9,
)

fig.subplots_adjust(
    left=0.08,
    right=0.94,
    top=0.85,
    bottom=0.13,
)

fig.savefig(
    FIGURES_DIR / "02_asset_correlation_heatmap.png",
    dpi=220,
    facecolor=BG,
    bbox_inches="tight",
)

plt.close(fig)

print("\nAnalysis completed.")
print("Charts saved to:", FIGURES_DIR)
