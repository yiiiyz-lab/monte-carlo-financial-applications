
"""
Monte Carlo Portfolio Risk Simulation
-------------------------------------
Model: Correlated multivariate normal log returns
Horizon: 20 trading days
Simulations: 50,000
Portfolio: $100 million, six ETFs
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NAV_0 = 100_000_000
HORIZON = 20
N_SIMULATIONS = 50_000
SEED = 42

TICKERS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "GLD"]

WEIGHTS = np.array([
    0.30,  # SPY
    0.15,  # QQQ
    0.20,  # IEF
    0.10,  # TLT
    0.15,  # LQD
    0.10,  # GLD
])

assert np.isclose(WEIGHTS.sum(), 1.0)


# --------------------------------------------------
# 2. Load historical returns
# --------------------------------------------------

returns = pd.read_csv(
    DATA_DIR / "daily_log_returns.csv",
    index_col=0,
    parse_dates=True,
)

returns = returns[TICKERS].dropna()

# Daily historical covariance matrix
cov_matrix = returns.cov().to_numpy()

# Zero daily log-return drift for baseline
mu = np.zeros(len(TICKERS))

print("Historical observations:", len(returns))
print("Assets:", TICKERS)
print("Covariance matrix shape:", cov_matrix.shape)


# --------------------------------------------------
# 3. Generate correlated daily returns
# --------------------------------------------------

rng = np.random.default_rng(SEED)

# Cholesky factorization: L @ L.T = covariance
L = np.linalg.cholesky(cov_matrix)

# Independent standard normal shocks
Z = rng.standard_normal(
    size=(N_SIMULATIONS, HORIZON, len(TICKERS))
)

# Introduce historical cross-asset covariance
simulated_returns = Z @ L.T + mu

print("\nSimulated return array shape:")
print(simulated_returns.shape)


# --------------------------------------------------
# 4. Simulate asset and portfolio values
# --------------------------------------------------

# Cumulative log returns along each path
cumulative_log_returns = np.cumsum(
    simulated_returns,
    axis=1,
)

# Convert to relative asset prices
relative_prices = np.exp(cumulative_log_returns)

# Portfolio NAV under buy-and-hold assumptions
portfolio_nav = NAV_0 * (
    relative_prices @ WEIGHTS
)

# Add day-zero portfolio value
initial_nav = np.full(
    (N_SIMULATIONS, 1),
    NAV_0,
)

portfolio_nav = np.concatenate(
    [initial_nav, portfolio_nav],
    axis=1,
)

# Terminal P&L and simple returns
terminal_nav = portfolio_nav[:, -1]
terminal_pnl = terminal_nav - NAV_0
terminal_returns = terminal_pnl / NAV_0


# --------------------------------------------------
# 5. Calculate risk metrics
# --------------------------------------------------

# Losses are positive when portfolio loses value
losses = -terminal_pnl


def calculate_var_es(losses, confidence):
    var = np.quantile(losses, confidence)

    # Expected loss beyond the VaR threshold
    es = losses[losses >= var].mean()

    return var, es


var_95, es_95 = calculate_var_es(losses, 0.95)
var_99, es_99 = calculate_var_es(losses, 0.99)

prob_loss_5pct = np.mean(terminal_returns < -0.05)
prob_any_loss = np.mean(terminal_returns < 0)

metrics = {
    "Initial NAV": NAV_0,
    "Horizon (days)": HORIZON,
    "Simulations": N_SIMULATIONS,
    "Mean Terminal P&L": terminal_pnl.mean(),
    "Median Terminal P&L": np.median(terminal_pnl),
    "P&L Standard Deviation": terminal_pnl.std(ddof=1),
    "VaR 95%": var_95,
    "ES 95%": es_95,
    "VaR 99%": var_99,
    "ES 99%": es_99,
    "Probability of Loss": prob_any_loss,
    "Probability of >5% Loss": prob_loss_5pct,
}


# --------------------------------------------------
# 6. Save simulation outputs
# --------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Terminal simulation outcomes
results = pd.DataFrame({
    "simulation_id": np.arange(1, N_SIMULATIONS + 1),
    "terminal_nav": terminal_nav,
    "terminal_pnl": terminal_pnl,
    "terminal_return": terminal_returns,
})

results.to_csv(
    DATA_DIR / "mc_terminal_results.csv",
    index=False,
)

# Daily NAV percentiles for future fan chart
days = np.arange(HORIZON + 1)

percentile_data = {
    "day": days,
}

for q in [1, 5, 25, 50, 75, 95, 99]:
    percentile_data[f"p{q:02d}"] = np.percentile(
        portfolio_nav,
        q,
        axis=0,
    )

fan_data = pd.DataFrame(percentile_data)

fan_data.to_csv(
    DATA_DIR / "mc_nav_percentiles.csv",
    index=False,
)

# Save a representative subset of simulated NAV paths
# for detailed visualization.

N_PLOT_PATHS = 1000

# Use a separate RNG to avoid changing the simulation
# results or their reproducibility.
plot_rng = np.random.default_rng(2026)

selected_indices = plot_rng.choice(
    N_SIMULATIONS,
    size=N_PLOT_PATHS,
    replace=False,
)

sampled_paths = portfolio_nav[selected_indices, :]

paths_df = pd.DataFrame(
    sampled_paths.T / 1e6,
    index=np.arange(HORIZON + 1),
    columns=[
        f"path_{i+1:04d}"
        for i in range(N_PLOT_PATHS)
    ],
)

paths_df.index.name = "day"

paths_df.to_csv(
    DATA_DIR / "mc_sample_paths.csv"
)

# --------------------------------------------------
# Select representative paths closest to
# their corresponding percentile trajectories
# --------------------------------------------------

target_percentiles = {
    "p05": 5,
    "p50": 50,
    "p95": 95,
}

# Terminal NAV tolerance: $0.15M
tolerance = 150_000

representative_paths = pd.DataFrame({
    "day": np.arange(HORIZON + 1)
})

print("\nRepresentative path selection:")

for label, percentile in target_percentiles.items():

    # Pointwise percentile trajectory
    target_curve = np.percentile(
        portfolio_nav,
        percentile,
        axis=0,
    )

    target_terminal_nav = target_curve[-1]

    # Select paths with terminal NAV close
    # to the target percentile
    terminal_distance = np.abs(
        terminal_nav - target_terminal_nav
    )

    candidates = np.where(
        terminal_distance <= tolerance
    )[0]

    # Fallback if too few candidates
    if len(candidates) < 20:
        candidates = np.argsort(
            terminal_distance
        )[:100]

    # Compare each candidate path with
    # the corresponding percentile curve
    candidate_paths = portfolio_nav[candidates]

    squared_deviation = (
        candidate_paths - target_curve
    ) ** 2

    # Mean squared deviation across time
    path_scores = squared_deviation[:, 1:].mean(
        axis=1
    )

    # Select the closest actual path
    best_idx = candidates[
        np.argmin(path_scores)
    ]

    representative_paths[label] = (
        portfolio_nav[best_idx] / 1e6
    )

    print(
        f"{label}: "
        f"Simulation {best_idx}, "
        f"Terminal NAV = "
        f"${terminal_nav[best_idx] / 1e6:.3f}M, "
        f"Target = "
        f"${target_terminal_nav / 1e6:.3f}M"
    )

representative_paths.to_csv(
    DATA_DIR / "mc_representative_paths.csv",
    index=False,
)

# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n" + "=" * 55)
print("MONTE CARLO PORTFOLIO RISK REPORT")
print("=" * 55)

print(f"Initial NAV:        ${NAV_0:,.0f}")
print(f"Horizon:            {HORIZON} trading days")
print(f"Simulations:        {N_SIMULATIONS:,}")

print("\nRisk Metrics")
print("-" * 55)

print(f"Mean P&L:           ${terminal_pnl.mean():,.0f}")
print(f"VaR (95%):          ${var_95:,.0f}")
print(f"ES  (95%):          ${es_95:,.0f}")
print(f"VaR (99%):          ${var_99:,.0f}")
print(f"ES  (99%):          ${es_99:,.0f}")

print(f"\nProbability of loss:      {prob_any_loss:.2%}")
print(f"Probability of >5% loss:  {prob_loss_5pct:.2%}")

print("\nOutput files saved to:")
print(DATA_DIR)

print("\nSimulation completed successfully.")
