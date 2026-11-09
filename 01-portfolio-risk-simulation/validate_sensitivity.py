
"""
Monte Carlo Portfolio Risk - Sensitivity Analysis

Stage 1: Volatility sensitivity

Baseline assumptions:
- $100M initial portfolio NAV
- Six ETFs, fixed initial weights
- Multivariate normal daily log returns
- Historical covariance matrix
- Zero log-return drift
- 20-day horizon
- Buy-and-hold portfolio
- 50,000 Monte Carlo simulations

Common random numbers are used across scenarios.
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
    0.30,
    0.15,
    0.20,
    0.10,
    0.15,
    0.10,
])

INITIAL_NAV = 100_000_000
HORIZON = 20
N_SIMULATIONS = 50_000
SEED = 42

CONFIDENCE = 0.99

VOL_SCENARIOS = {
    "Low volatility": 0.75,
    "Baseline": 1.00,
    "Elevated volatility": 1.25,
    "Severe volatility": 1.50,
}


# --------------------------------------------------
# 2. Load historical returns
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
# 3. Risk metric functions
# --------------------------------------------------

def calculate_risk_metrics(terminal_pnl):
    """
    Compute loss-based VaR, ES and loss probabilities.
    """

    cutoff = np.quantile(
        terminal_pnl,
        1 - CONFIDENCE,
    )

    var = -cutoff

    es = -terminal_pnl[
        terminal_pnl <= cutoff
    ].mean()

    probability_of_loss = np.mean(
        terminal_pnl < 0
    )

    probability_of_5pct_loss = np.mean(
        terminal_pnl < -0.05 * INITIAL_NAV
    )

    return {
        "var_99": var,
        "es_99": es,
        "probability_of_loss": probability_of_loss,
        "probability_of_5pct_loss":
            probability_of_5pct_loss,
    }


# --------------------------------------------------
# 4. Portfolio simulation function
# --------------------------------------------------

def simulate_terminal_pnl(
    standard_normal_draws,
    cholesky_factor,
    drift=None,
):
    """
    Simulate terminal P&L under a buy-and-hold strategy.

    standard_normal_draws:
        shape (simulations, horizon, assets)

    cholesky_factor:
        factor of the scenario covariance matrix

    drift:
        daily expected log-return vector
    """

    simulated_returns = (
        standard_normal_draws @ cholesky_factor.T
    )

    if drift is not None:
        simulated_returns += drift

    cumulative_log_returns = (
        simulated_returns.sum(axis=1)
    )

    terminal_asset_values = (
        INITIAL_NAV
        * WEIGHTS
        * np.exp(cumulative_log_returns)
    )

    terminal_nav = terminal_asset_values.sum(axis=1)

    return terminal_nav - INITIAL_NAV


# --------------------------------------------------
# 5. Generate common random numbers
# --------------------------------------------------

rng = np.random.default_rng(SEED)

Z = rng.standard_normal(
    (N_SIMULATIONS, HORIZON, len(TICKERS))
)


# --------------------------------------------------
# 6. Volatility sensitivity analysis
# --------------------------------------------------

records = []

for scenario_name, multiplier in VOL_SCENARIOS.items():

    print(f"Running: {scenario_name} ({multiplier:.2f}x)")

    scenario_L = multiplier * L

    terminal_pnl = simulate_terminal_pnl(
        standard_normal_draws=Z,
        cholesky_factor=scenario_L,
    )

    metrics = calculate_risk_metrics(terminal_pnl)

    records.append({
        "scenario": scenario_name,
        "vol_multiplier": multiplier,
        **metrics,
    })

results = pd.DataFrame(records)

baseline = results.loc[
    results["scenario"] == "Baseline"
].iloc[0]

results["var_change_pct"] = (
    results["var_99"] / baseline["var_99"] - 1
) * 100

results["es_change_pct"] = (
    results["es_99"] / baseline["es_99"] - 1
) * 100

results.to_csv(
    DATA_DIR / "mc_volatility_sensitivity.csv",
    index=False,
)


# --------------------------------------------------
# 7. Console summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("VOLATILITY SENSITIVITY ANALYSIS")
print("=" * 75)

display = results.copy()

display["var_99"] /= 1e6
display["es_99"] /= 1e6

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nVaR and ES reported in $ millions.")
print("Changes are relative to baseline.")


# --------------------------------------------------
# 8. Visualization: Figure 07
# --------------------------------------------------

BG = "#0B1220"
PANEL = "#111C2E"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GRID = "#334155"

BLUE = "#38BDF8"
RED = "#FB7185"
AMBER = "#FBBF24"

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

x = results["vol_multiplier"].to_numpy()

var_values = results["var_99"].to_numpy() / 1e6
es_values = results["es_99"].to_numpy() / 1e6

ax.plot(
    x,
    var_values,
    color=BLUE,
    marker="o",
    markersize=7,
    linewidth=2.3,
    label="99% VaR",
)

ax.plot(
    x,
    es_values,
    color=RED,
    marker="o",
    markersize=7,
    linewidth=2.3,
    label="99% ES",
)

# Baseline reference
ax.axvline(
    1.0,
    color=AMBER,
    linestyle="--",
    linewidth=1.1,
    alpha=0.7,
)

ax.set_title(
    "Portfolio Risk Sensitivity to Volatility",
    fontsize=19,
    fontweight="bold",
    loc="left",
    pad=36,
)

ax.text(
    0,
    1.035,
    "99% confidence | 20-day horizon | "
    "50,000 simulations | Common random numbers",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
)

ax.set_xlabel(
    "Volatility Multiplier",
    fontsize=11,
    labelpad=12,
)

ax.set_ylabel(
    "Estimated Portfolio Risk ($M)",
    fontsize=11,
    labelpad=12,
)

ax.set_xticks(x)
ax.set_xticklabels([
    "0.75x",
    "1.00x",
    "1.25x",
    "1.50x",
])

ax.grid(
    color=GRID,
    alpha=0.32,
    linewidth=0.8,
)

ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    fontsize=10,
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
)

fig.text(
    0.08,
    0.035,
    "Volatility is scaled uniformly across assets; "
    "correlations and portfolio weights remain unchanged.",
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
    FIGURES_DIR / "07_mc_volatility_sensitivity.png",
    dpi=220,
    facecolor=BG,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nSaved Figure 07: "
    "07_mc_volatility_sensitivity.png"
)

# --------------------------------------------------
# 9. Correlation sensitivity analysis
# --------------------------------------------------

# Extract historical asset volatilities
daily_vol = np.sqrt(np.diag(cov))

# Historical correlation matrix
baseline_corr = (
    cov / np.outer(daily_vol, daily_vol)
)

n_assets = len(TICKERS)

identity_corr = np.eye(n_assets)
perfect_corr = np.ones((n_assets, n_assets))

CORR_SCENARIOS = {
    "Lower dependence": (
        0.5 * baseline_corr
        + 0.5 * identity_corr
    ),
    "Baseline": baseline_corr,
    "Higher co-movement": (
        0.5 * baseline_corr
        + 0.5 * perfect_corr
    ),
}

corr_records = []

for scenario_name, scenario_corr in CORR_SCENARIOS.items():

    print(f"Running correlation scenario: {scenario_name}")

    # Reconstruct covariance while keeping
    # individual asset volatilities unchanged.
    scenario_cov = (
        np.outer(daily_vol, daily_vol)
        * scenario_corr
    )

    # Validate the correlation matrix
    min_eigenvalue = np.linalg.eigvalsh(
        scenario_corr
    ).min()

    if min_eigenvalue < -1e-10:
        raise ValueError(
            f"Invalid correlation matrix: {scenario_name}"
        )

    scenario_L = np.linalg.cholesky(
        scenario_cov
    )

    terminal_pnl = simulate_terminal_pnl(
        standard_normal_draws=Z,
        cholesky_factor=scenario_L,
    )

    metrics = calculate_risk_metrics(terminal_pnl)

    corr_records.append({
        "scenario": scenario_name,
        "min_eigenvalue": min_eigenvalue,
        **metrics,
    })

corr_results = pd.DataFrame(corr_records)

corr_baseline = corr_results.loc[
    corr_results["scenario"] == "Baseline"
].iloc[0]

corr_results["var_change_pct"] = (
    corr_results["var_99"]
    / corr_baseline["var_99"] - 1
) * 100

corr_results["es_change_pct"] = (
    corr_results["es_99"]
    / corr_baseline["es_99"] - 1
) * 100

corr_results.to_csv(
    DATA_DIR / "mc_correlation_sensitivity.csv",
    index=False,
)

# --------------------------------------------------
# 10. Correlation sensitivity summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("CORRELATION SENSITIVITY ANALYSIS")
print("=" * 75)

corr_display = corr_results.copy()

corr_display["var_99"] /= 1e6
corr_display["es_99"] /= 1e6

print(
    corr_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nVaR and ES reported in $ millions.")
print("Individual asset volatilities remain unchanged.")

# --------------------------------------------------
# 11. Baseline consistency check
# --------------------------------------------------

assert np.isclose(
    corr_baseline["var_99"],
    baseline["var_99"],
    rtol=1e-8,
), "Correlation baseline VaR does not match."

assert np.isclose(
    corr_baseline["es_99"],
    baseline["es_99"],
    rtol=1e-8,
), "Correlation baseline ES does not match."

print("\nBaseline consistency check: PASS")

# --------------------------------------------------
# 12. Drift sensitivity analysis
# --------------------------------------------------

# Historical mean daily log returns
historical_drift = returns.mean().to_numpy()

# Annualized volatility
annual_vol = daily_vol * np.sqrt(252)

# Illustrative negative annualized drift:
# -25% of each asset's annualized volatility
negative_annual_drift = -0.25 * annual_vol

# Convert annualized log-return drift to daily
negative_daily_drift = negative_annual_drift / 252

DRIFT_SCENARIOS = {
    "Zero drift": np.zeros(len(TICKERS)),
    "Historical mean": historical_drift,
    "Negative drift": negative_daily_drift,
}

drift_records = []

for scenario_name, drift_vector in DRIFT_SCENARIOS.items():

    print(f"Running drift scenario: {scenario_name}")

    terminal_pnl = simulate_terminal_pnl(
        standard_normal_draws=Z,
        cholesky_factor=L,
        drift=drift_vector,
    )

    metrics = calculate_risk_metrics(terminal_pnl)

    drift_records.append({
        "scenario": scenario_name,
        **metrics,
    })

drift_results = pd.DataFrame(drift_records)

drift_baseline = drift_results.loc[
    drift_results["scenario"] == "Zero drift"
].iloc[0]

drift_results["var_change_pct"] = (
    drift_results["var_99"]
    / drift_baseline["var_99"] - 1
) * 100

drift_results["es_change_pct"] = (
    drift_results["es_99"]
    / drift_baseline["es_99"] - 1
) * 100

drift_results.to_csv(
    DATA_DIR / "mc_drift_sensitivity.csv",
    index=False,
)

# --------------------------------------------------
# 13. Drift sensitivity summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("DRIFT SENSITIVITY ANALYSIS")
print("=" * 75)

drift_display = drift_results.copy()

drift_display["var_99"] /= 1e6
drift_display["es_99"] /= 1e6

print(
    drift_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nVaR and ES reported in $ millions.")

# --------------------------------------------------
# 14. Drift assumption diagnostics
# --------------------------------------------------

drift_assumptions = pd.DataFrame({
    "ticker": TICKERS,
    "historical_annual_log_return":
        historical_drift * 252,
    "negative_annual_log_return":
        negative_annual_drift,
})

drift_assumptions.to_csv(
    DATA_DIR / "mc_drift_assumptions.csv",
    index=False,
)

print("\nANNUALIZED DRIFT ASSUMPTIONS")
print("-" * 75)

print(
    drift_assumptions.to_string(
        index=False,
        float_format=lambda x: f"{x:.4%}",
    )
)

# --------------------------------------------------
# 15. Baseline consistency check
# --------------------------------------------------

assert np.isclose(
    drift_baseline["var_99"],
    baseline["var_99"],
    rtol=1e-8,
), "Drift baseline VaR does not match."

assert np.isclose(
    drift_baseline["es_99"],
    baseline["es_99"],
    rtol=1e-8,
), "Drift baseline ES does not match."

print("\nDrift baseline consistency check: PASS")

# --------------------------------------------------
# 16. Distribution sensitivity analysis
# --------------------------------------------------

DISTRIBUTION_SCENARIOS = [
    ("Normal", None),
    ("Student's t (df=10)", 10),
    ("Student's t (df=5)", 5),
]

distribution_records = []

# Independent random stream for chi-square scaling.
# Reuse the baseline standard normal draws Z.
t_rng = np.random.default_rng(SEED + 1000)

for scenario_name, df in DISTRIBUTION_SCENARIOS:

    print(f"Running distribution scenario: {scenario_name}")

    if df is None:
        scenario_draws = Z
    else:
        # One chi-square scaling variable per
        # simulation path and trading day.
        #
        # The same scaling variable applies
        # to all six assets on that day.
        chi_square = t_rng.chisquare(
            df=df,
            size=(N_SIMULATIONS, HORIZON, 1),
        )

        scale = np.sqrt(
            (df - 2) / chi_square
        )

        scenario_draws = Z * scale

    terminal_pnl = simulate_terminal_pnl(
        standard_normal_draws=scenario_draws,
        cholesky_factor=L,
    )

    metrics = calculate_risk_metrics(terminal_pnl)

    distribution_records.append({
        "scenario": scenario_name,
        "degrees_of_freedom": (
            np.nan if df is None else df
        ),
        **metrics,
    })

distribution_results = pd.DataFrame(
    distribution_records
)

# --------------------------------------------------
# 17. Compare against normal baseline
# --------------------------------------------------

distribution_baseline = distribution_results.loc[
    distribution_results["scenario"] == "Normal"
].iloc[0]

distribution_results["var_change_pct"] = (
    distribution_results["var_99"]
    / distribution_baseline["var_99"] - 1
) * 100

distribution_results["es_change_pct"] = (
    distribution_results["es_99"]
    / distribution_baseline["es_99"] - 1
) * 100

distribution_results.to_csv(
    DATA_DIR / "mc_distribution_sensitivity.csv",
    index=False,
)

# --------------------------------------------------
# 18. Console summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("DISTRIBUTION SENSITIVITY ANALYSIS")
print("=" * 75)

distribution_display = distribution_results.copy()

distribution_display["var_99"] /= 1e6
distribution_display["es_99"] /= 1e6

print(
    distribution_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nVaR and ES reported in $ millions.")
print("All scenarios use the same theoretical covariance.")

# --------------------------------------------------
# 19. Baseline consistency check
# --------------------------------------------------

assert np.isclose(
    distribution_baseline["var_99"],
    baseline["var_99"],
    rtol=1e-8,
), "Distribution baseline VaR does not match."

assert np.isclose(
    distribution_baseline["es_99"],
    baseline["es_99"],
    rtol=1e-8,
), "Distribution baseline ES does not match."

print("\nDistribution baseline consistency check: PASS")

# --------------------------------------------------
# 20. Figure 08: Consolidated model sensitivity
# --------------------------------------------------

from matplotlib.ticker import FuncFormatter

sensitivity_groups = [
    ("Volatility", results),
    ("Correlation", corr_results),
    ("Drift", drift_results),
    ("Distribution", distribution_results),
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
)

fig.patch.set_facecolor(BG)

for ax, (group_name, df) in zip(
    axes.flat, sensitivity_groups
):
    ax.set_facecolor(BG)

    # Remove baseline observations
    scenario_df = df.loc[
        df["var_change_pct"].abs().gt(1e-9)
        | df["es_change_pct"].abs().gt(1e-9)
    ].copy()

    labels = scenario_df["scenario"].tolist()

    var_changes = scenario_df[
        "var_change_pct"
    ].to_numpy()

    es_changes = scenario_df[
        "es_change_pct"
    ].to_numpy()

    y = np.arange(len(labels))
    bar_height = 0.32

    ax.barh(
        y - bar_height / 2,
        var_changes,
        height=bar_height,
        color=BLUE,
        alpha=0.90,
        label="99% VaR",
    )

    ax.barh(
        y + bar_height / 2,
        es_changes,
        height=bar_height,
        color=RED,
        alpha=0.90,
        label="99% ES",
    )

    ax.axvline(
        0,
        color=MUTED,
        linewidth=1,
        alpha=0.75,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    ax.invert_yaxis()

    ax.set_title(
        group_name,
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: f"{value:+.0f}%")
    )

    ax.grid(
        axis="x",
        color=GRID,
        alpha=0.28,
    )

    ax.set_axisbelow(True)

    # Symmetric limits with room for labels
    max_change = max(
        np.max(np.abs(var_changes)),
        np.max(np.abs(es_changes)),
        1.0,
    )

    limit = max_change * 1.45

    ax.set_xlim(-limit, limit)

    # Annotate each bar
    for positions, values in [
        (y - bar_height / 2, var_changes),
        (y + bar_height / 2, es_changes),
    ]:
        for ypos, value in zip(positions, values):
            offset = max_change * 0.04

            ax.text(
                value + (
                    offset if value >= 0 else -offset
                ),
                ypos,
                f"{value:+.2f}%",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.5,
                color=TEXT,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

# Figure title
fig.suptitle(
    "Monte Carlo Model Sensitivity Analysis",
    fontsize=21,
    fontweight="bold",
    x=0.08,
    y=0.98,
    ha="left",
)

fig.text(
    0.08,
    0.935,
    "Change in 99% VaR and Expected Shortfall "
    "relative to baseline | 20-day horizon",
    fontsize=11,
    color=MUTED,
)

# Shared legend
handles, legend_labels = axes.flat[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="upper right",
    bbox_to_anchor=(0.93, 0.945),
    ncol=2,
    fontsize=10,
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
)

fig.text(
    0.08,
    0.035,
    "50,000 simulations | One-factor-at-a-time "
    "sensitivity | Common Gaussian random draws",
    fontsize=9,
    color=MUTED,
)

fig.subplots_adjust(
    left=0.19,
    right=0.95,
    top=0.85,
    bottom=0.11,
    hspace=0.48,
    wspace=0.32,
)

fig.savefig(
    FIGURES_DIR / "08_mc_model_sensitivity.png",
    dpi=220,
    facecolor=BG,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nSaved Figure 08: "
    "08_mc_model_sensitivity.png"
)
