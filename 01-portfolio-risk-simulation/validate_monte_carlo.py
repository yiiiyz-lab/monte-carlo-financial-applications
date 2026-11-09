
"""
Monte Carlo Portfolio Risk Model Validation

Validation 1: Cholesky covariance reconstruction
Validation 2: Simulated return moments

Uses the same assumptions as monte_carlo.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TICKERS = ["SPY", "QQQ", "IEF", "TLT", "LQD", "GLD"]

N_SIMULATIONS = 50_000
HORIZON = 20
SEED = 42

# Diagnostic tolerances
CHOLESKY_TOL = 1e-10
MEAN_Z_THRESHOLD = 3.0
COV_REL_TOL = 0.05
CORR_ABS_TOL = 0.03


# --------------------------------------------------
# 2. Load historical returns
# --------------------------------------------------

returns = pd.read_csv(
    DATA_DIR / "daily_log_returns.csv",
    index_col=0,
    parse_dates=True,
)

returns = returns[TICKERS].dropna()

historical_cov = returns.cov().to_numpy()
historical_corr = returns.corr().to_numpy()

daily_vol = np.sqrt(np.diag(historical_cov))

print("=" * 60)
print("MONTE CARLO MODEL VALIDATION")
print("=" * 60)

print(f"Historical observations: {len(returns)}")
print(f"Simulated daily observations: "
      f"{N_SIMULATIONS * HORIZON:,}")


# --------------------------------------------------
# 3. Validation 1: Cholesky reconstruction
# --------------------------------------------------

L = np.linalg.cholesky(historical_cov)

reconstructed_cov = L @ L.T

max_abs_error = np.max(
    np.abs(reconstructed_cov - historical_cov)
)

cholesky_pass = max_abs_error < CHOLESKY_TOL

print("\n[1] CHOLESKY COVARIANCE RECONSTRUCTION")
print("-" * 60)

print(f"Maximum absolute error: {max_abs_error:.12e}")
print(f"Tolerance:              {CHOLESKY_TOL:.1e}")
print(f"Result:                 "
      f"{'PASS' if cholesky_pass else 'FAIL'}")


# --------------------------------------------------
# 4. Reproduce simulated daily returns
# --------------------------------------------------

rng = np.random.default_rng(SEED)

Z = rng.standard_normal(
    (N_SIMULATIONS, HORIZON, len(TICKERS))
)

simulated_returns = Z @ L.T

# Flatten paths and time into daily observations
sim_flat = simulated_returns.reshape(
    -1,
    len(TICKERS),
)

sim_mean = sim_flat.mean(axis=0)
sim_cov = np.cov(sim_flat, rowvar=False)
sim_corr = np.corrcoef(sim_flat, rowvar=False)

sim_vol = np.sqrt(np.diag(sim_cov))


# --------------------------------------------------
# 5. Validation 2A: Simulated means
# --------------------------------------------------

# Under zero drift, sample mean should be near 0.
# Use a 3-standard-error diagnostic threshold.

n_obs = len(sim_flat)

mean_se = daily_vol / np.sqrt(n_obs)

mean_z = np.abs(sim_mean) / mean_se

mean_pass = np.all(
    mean_z < MEAN_Z_THRESHOLD
)

mean_df = pd.DataFrame({
    "ticker": TICKERS,
    "target_daily_mean": np.zeros(len(TICKERS)),
    "simulated_daily_mean": sim_mean,
    "mean_z_score": mean_z,
})

print("\n[2A] SIMULATED DAILY RETURN MEANS")
print("-" * 60)

print(mean_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print(f"\nResult: "
      f"{'PASS' if mean_pass else 'FAIL'}")


# --------------------------------------------------
# 6. Validation 2B: Simulated volatility
# --------------------------------------------------

vol_relative_error = np.abs(
    sim_vol - daily_vol
) / daily_vol

vol_pass = np.all(
    vol_relative_error < COV_REL_TOL
)

vol_df = pd.DataFrame({
    "ticker": TICKERS,
    "target_daily_vol": daily_vol,
    "simulated_daily_vol": sim_vol,
    "relative_error": vol_relative_error,
})

print("\n[2B] SIMULATED DAILY VOLATILITY")
print("-" * 60)

print(vol_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print(f"\nMaximum relative error: "
      f"{vol_relative_error.max():.4%}")

print(f"Result: "
      f"{'PASS' if vol_pass else 'FAIL'}")


# --------------------------------------------------
# 7. Validation 2C: Simulated correlations
# --------------------------------------------------

corr_abs_error = np.abs(
    sim_corr - historical_corr
)

# Exclude diagonal values
off_diag = ~np.eye(
    len(TICKERS),
    dtype=bool,
)

max_corr_error = corr_abs_error[off_diag].max()

corr_pass = max_corr_error < CORR_ABS_TOL

print("\n[2C] SIMULATED ASSET CORRELATIONS")
print("-" * 60)

print(f"Maximum absolute correlation error: "
      f"{max_corr_error:.6f}")

print(f"Tolerance: {CORR_ABS_TOL:.4f}")

print(f"Result: "
      f"{'PASS' if corr_pass else 'FAIL'}")


# --------------------------------------------------
# 8. Overall validation summary
# --------------------------------------------------

checks = {
    "Cholesky reconstruction": cholesky_pass,
    "Simulated mean": mean_pass,
    "Simulated volatility": vol_pass,
    "Simulated correlation": corr_pass,
}

summary_df = pd.DataFrame({
    "check": list(checks.keys()),
    "status": [
        "PASS" if passed else "FAIL"
        for passed in checks.values()
    ],
})

summary_df.to_csv(
    DATA_DIR / "mc_validation_summary.csv",
    index=False,
)

mean_df.to_csv(
    DATA_DIR / "mc_validation_means.csv",
    index=False,
)

vol_df.to_csv(
    DATA_DIR / "mc_validation_volatility.csv",
    index=False,
)

pd.DataFrame(
    sim_corr,
    index=TICKERS,
    columns=TICKERS,
).to_csv(
    DATA_DIR / "mc_simulated_correlation.csv",
)

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

overall_pass = all(checks.values())

print("\nOverall result:",
      "PASS" if overall_pass else "REVIEW REQUIRED")
