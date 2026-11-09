
"""
Monte Carlo VaR / ES Convergence Analysis

Tests:
1. Convergence with increasing simulation counts
2. Stability across independent random seeds

Model:
- Multivariate normal daily log returns
- Historical covariance
- Zero log-return drift
- 20-day buy-and-hold portfolio
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

TICKERS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "GLD"]

WEIGHTS = np.array([
    0.30,  # SPY
    0.15,  # QQQ
    0.20,  # IEF
    0.10,  # TLT
    0.15,  # LQD
    0.10,  # GLD
])

INITIAL_NAV = 100_000_000
HORIZON = 20

SIMULATION_COUNTS = [1000, 5000, 10000, 25000, 50000]
SEEDS = [42, 123, 456, 789, 2026]

CONFIDENCE = 0.99


# --------------------------------------------------
# 2. Load historical data
# --------------------------------------------------

returns = pd.read_csv(
    DATA_DIR / "daily_log_returns.csv",
    index_col=0,
    parse_dates=True,
)

returns = returns[TICKERS].dropna()

cov = returns.cov().to_numpy()
L = np.linalg.cholesky(cov)


# --------------------------------------------------
# 3. Risk metric calculation
# --------------------------------------------------

def calculate_var_es(terminal_pnl, confidence=0.99):
    """
    Positive loss-based VaR and ES.

    VaR = negative lower-tail P&L quantile.
    ES = average loss at or beyond the VaR cutoff.
    """

    alpha = 1 - confidence

    cutoff = np.quantile(terminal_pnl, alpha)

    var = -cutoff

    tail_pnl = terminal_pnl[
        terminal_pnl <= cutoff
    ]

    es = -tail_pnl.mean()

    return var, es


# --------------------------------------------------
# 4. Simulate terminal portfolio P&L
# --------------------------------------------------

def simulate_terminal_pnl(seed, n_simulations):
    """
    Simulate correlated daily log returns.

    For buy-and-hold:
    terminal asset value =
        initial asset value * exp(cumulative log return)
    """

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(
        (n_simulations, HORIZON, len(TICKERS))
    )

    simulated_returns = z @ L.T

    cumulative_log_returns = simulated_returns.sum(axis=1)

    terminal_asset_values = (
        INITIAL_NAV
        * WEIGHTS
        * np.exp(cumulative_log_returns)
    )

    terminal_nav = terminal_asset_values.sum(axis=1)

    return terminal_nav - INITIAL_NAV


# --------------------------------------------------
# 5. Run convergence experiment
# --------------------------------------------------

records = []

max_simulations = max(SIMULATION_COUNTS)

for seed in SEEDS:

    print(f"Running seed {seed}...")

    terminal_pnl = simulate_terminal_pnl(
        seed=seed,
        n_simulations=max_simulations,
    )

    for n in SIMULATION_COUNTS:

        sample_pnl = terminal_pnl[:n]

        var, es = calculate_var_es(
            sample_pnl,
            confidence=CONFIDENCE,
        )

        records.append({
            "seed": seed,
            "simulations": n,
            "var_99": var,
            "es_99": es,
        })

results = pd.DataFrame(records)

results.to_csv(
    DATA_DIR / "mc_convergence_results.csv",
    index=False,
)


# --------------------------------------------------
# 6. Summary statistics across seeds
# --------------------------------------------------

summary = (
    results
    .groupby("simulations")
    .agg(
        var_mean=("var_99", "mean"),
        var_std=("var_99", "std"),
        es_mean=("es_99", "mean"),
        es_std=("es_99", "std"),
    )
    .reset_index()
)

# Coefficient of variation measures
# relative dispersion across seeds.

summary["var_cv"] = (
    summary["var_std"] / summary["var_mean"]
)

summary["es_cv"] = (
    summary["es_std"] / summary["es_mean"]
)

summary.to_csv(
    DATA_DIR / "mc_convergence_summary.csv",
    index=False,
)


# --------------------------------------------------
# 7. Console output
# --------------------------------------------------

print("\n" + "=" * 70)
print("MONTE CARLO CONVERGENCE ANALYSIS")
print("=" * 70)

display = summary.copy()

for column in [
    "var_mean",
    "var_std",
    "es_mean",
    "es_std",
]:
    display[column] /= 1e6

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nCV = standard deviation / mean across seeds")
print("VaR and ES values are in $ millions.")


# --------------------------------------------------
# 8. Visualization
# --------------------------------------------------

BG = "#0B1220"
PANEL = "#111C2E"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GRID = "#334155"

BLUE = "#38BDF8"
RED = "#FB7185"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(12, 7))

x = summary["simulations"].to_numpy()

var_mean = summary["var_mean"].to_numpy() / 1e6
es_mean = summary["es_mean"].to_numpy() / 1e6

var_std = summary["var_std"].to_numpy() / 1e6
es_std = summary["es_std"].to_numpy() / 1e6

# Mean convergence curves
ax.plot(
    x,
    var_mean,
    color=BLUE,
    linewidth=2.3,
    marker="o",
    markersize=6,
    label="99% VaR",
)

ax.plot(
    x,
    es_mean,
    color=RED,
    linewidth=2.3,
    marker="o",
    markersize=6,
    label="99% ES",
)

# +/- one standard deviation across seeds
ax.fill_between(
    x,
    var_mean - var_std,
    var_mean + var_std,
    color=BLUE,
    alpha=0.13,
)

ax.fill_between(
    x,
    es_mean - es_std,
    es_mean + es_std,
    color=RED,
    alpha=0.13,
)

ax.set_title(
    "Monte Carlo VaR & ES Convergence",
    fontsize=19,
    fontweight="bold",
    loc="left",
    pad=36,
)

ax.text(
    0,
    1.035,
    "99% confidence | 20-day horizon | "
    "5 independent seeds | Shaded bands: +/- 1 SD",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
)

ax.set_xlabel(
    "Number of Simulations",
    fontsize=11,
    labelpad=12,
)

ax.set_ylabel(
    "Estimated Portfolio Risk ($M)",
    fontsize=11,
    labelpad=12,
)

ax.set_xticks(x)
ax.set_xticklabels(
    ["1K", "5K", "10K", "25K", "50K"]
)

ax.grid(
    color=GRID,
    alpha=0.32,
    linewidth=0.8,
)

ax.set_axisbelow(True)

ax.legend(
    loc="best",
    fontsize=10,
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
)

fig.text(
    0.08,
    0.035,
    "Bands represent variation across random seeds, "
    "not model or market uncertainty.",
    fontsize=9,
    color=MUTED,
)

fig.subplots_adjust(
    left=0.11,
    right=0.95,
    top=0.83,
    bottom=0.16,
)

fig.savefig(
    FIGURES_DIR / "06_mc_var_es_convergence.png",
    dpi=220,
    facecolor=BG,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nSaved Figure 06: "
    "06_mc_var_es_convergence.png"
)
