
"""
Monte Carlo Portfolio Risk Visualizations

Figure 03: Terminal P&L distribution
Figure 04: Representative Monte Carlo simulation paths
Figure 05: VaR and Expected Shortfall comparison

All figures use actual simulation outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# --------------------------------------------------
# 1. Configuration and data
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    DATA_DIR / "mc_terminal_results.csv"
)

fan = pd.read_csv(
    DATA_DIR / "mc_nav_percentiles.csv"
)

paths = pd.read_csv(
    DATA_DIR / "mc_sample_paths.csv",
    index_col="day",
)

representative = pd.read_csv(
    DATA_DIR / "mc_representative_paths.csv"
)

metrics = (
    pd.read_csv(DATA_DIR / "mc_risk_metrics.csv")
    .set_index("metric")["value"]
)

NAV_0 = metrics["Initial NAV"]
HORIZON = int(metrics["Horizon (days)"])
N_SIMULATIONS = int(metrics["Simulations"])

var95 = metrics["VaR 95%"] / 1e6
var99 = metrics["VaR 99%"] / 1e6
es95 = metrics["ES 95%"] / 1e6
es99 = metrics["ES 99%"] / 1e6

prob5 = metrics["Probability of >5% Loss"]

pnl_m = results["terminal_pnl"].to_numpy() / 1e6
days = fan["day"].to_numpy()


# --------------------------------------------------
# 2. Consistent research chart style
# --------------------------------------------------

BG = "#0B1220"
PANEL = "#111C2E"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GRID = "#334155"

BLUE = "#38BDF8"
TEAL = "#2DD4BF"
AMBER = "#FBBF24"
RED = "#FB7185"
PURPLE = "#A78BFA"
GREEN = "#4ADE80"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def style_axis(ax):
    ax.grid(
        color=GRID,
        alpha=0.32,
        linewidth=0.8,
    )
    ax.set_axisbelow(True)


def add_header(ax, title, subtitle):
    ax.set_title(
        title,
        fontsize=19,
        fontweight="bold",
        loc="left",
        pad=37,
    )

    ax.text(
        0,
        1.035,
        subtitle,
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )


def add_footer(fig, text):
    fig.text(
        0.08,
        0.025,
        text,
        color=MUTED,
        fontsize=9,
    )


def save_figure(fig, filename):
    path = FIGURES_DIR / filename

    fig.savefig(
        path,
        dpi=220,
        facecolor=BG,
        bbox_inches="tight",
    )

    plt.close(fig)
    print("Saved:", path.name)


# --------------------------------------------------
# 3. Figure 03: Terminal P&L distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 7.5))

bins = np.linspace(
    np.percentile(pnl_m, 0.05),
    np.percentile(pnl_m, 99.95),
    90,
)

counts, edges = np.histogram(pnl_m, bins=bins)
widths = np.diff(edges)

for count, left, width in zip(
    counts, edges[:-1], widths
):
    midpoint = left + width / 2

    if midpoint <= -var99:
        color = RED
    elif midpoint <= -var95:
        color = AMBER
    else:
        color = BLUE

    ax.bar(
        midpoint,
        count,
        width=width * 0.94,
        color=color,
        alpha=0.88,
        align="center",
    )

ax.axvline(
    -var95,
    color=AMBER,
    linewidth=2,
    linestyle="--",
    label=f"95% VaR: ${var95:.2f}M",
)

ax.axvline(
    -var99,
    color=RED,
    linewidth=2,
    linestyle="--",
    label=f"99% VaR: ${var99:.2f}M",
)

ax.axvline(
    0,
    color=TEXT,
    linewidth=1,
    alpha=0.55,
)

ax.axvline(
    -5,
    color=PURPLE,
    linewidth=1.7,
    linestyle=":",
    label="5% portfolio loss",
)

add_header(
    ax,
    "Monte Carlo Portfolio P&L Distribution",
    f"{N_SIMULATIONS:,} simulations | "
    f"{HORIZON}-day horizon | "
    "USD millions",
)

ax.set_xlabel("Terminal Portfolio P&L ($M)", labelpad=12)
ax.set_ylabel("Simulation Frequency", labelpad=12)

style_axis(ax)

ax.legend(
    loc="upper right",
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
    fontsize=10,
)

ax.text(
    0.98,
    0.68,
    f"P(Loss > 5%) = {prob5:.2%}",
    transform=ax.transAxes,
    ha="right",
    fontsize=12,
    color=TEXT,
    fontweight="bold",
    bbox={
        "facecolor": PANEL,
        "edgecolor": GRID,
        "boxstyle": "round,pad=0.65",
    },
)

add_footer(
    fig,
    "Model: Multivariate normal log returns | "
    "Historical covariance | Zero log-return drift | "
    "Buy-and-hold",
)

fig.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.84,
    bottom=0.17,
)

save_figure(
    fig,
    "03_monte_carlo_pnl_distribution.png",
)


# --------------------------------------------------
# 4. Figure 04: Representative simulation paths
# --------------------------------------------------

# More balanced, near-square figure
fig, ax = plt.subplots(figsize=(12, 10))

p05 = fan["p05"].to_numpy() / 1e6
p95 = fan["p95"].to_numpy() / 1e6

# 1. Subtle 90% pointwise percentile envelope
ax.fill_between(
    days,
    p05,
    p95,
    color=BLUE,
    alpha=0.07,
    label="90% simulation envelope",
    zorder=1,
)

# 2. Draw 1,000 background simulation paths
path_values = paths.to_numpy()

for i in range(path_values.shape[1]):
    path = path_values[:, i]

    if path[-1] < 95:
        color = RED
    elif path[-1] > 105:
        color = TEAL
    else:
        color = MUTED

    ax.plot(
        days,
        path,
        color=color,
        linewidth=0.65,
        alpha=0.11,
        zorder=2,
    )

# 3. Highlight actual paths selected by terminal percentiles
highlighted_paths = [
    ("p95", "95th-percentile ending path", GREEN),
    ("p50", "Median-ending path", BLUE),
    ("p05", "5th-percentile ending path", RED),
]

for column, label, color in highlighted_paths:
    ax.plot(
        representative["day"],
        representative[column],
        color=color,
        linewidth=1.7,
        alpha=0.85,
        label=label,
        zorder=6,
    )

# 4. Initial NAV reference
ax.axhline(
    NAV_0 / 1e6,
    color=AMBER,
    linestyle="--",
    linewidth=1.2,
    alpha=0.65,
    label="Initial NAV",
    zorder=3,
)

# 5. Endpoint labels
endpoint_labels = [
    (representative["p95"].iloc[-1], "95th", GREEN),
    (representative["p50"].iloc[-1], "Median", BLUE),
    (representative["p05"].iloc[-1], "5th", RED),
]

for value, name, color in endpoint_labels:
    ax.annotate(
        f"{name}  ${value:.2f}M",
        xy=(HORIZON, value),
        xytext=(12, 0),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color=color,
        va="center",
        annotation_clip=False,
        zorder=8,
    )

# 6. Title and subtitle
add_header(
    ax,
    "Monte Carlo Portfolio Simulation Paths",
    f"{N_SIMULATIONS:,} simulations | "
    f"{path_values.shape[1]:,} background paths | "
    f"{HORIZON}-day horizon",
)

# 7. Risk metrics annotation
risk_text = (
    f"99% VaR       ${var99:.2f}M\n"
    f"99% ES        ${es99:.2f}M\n"
    f"P(Loss > 5%)  {prob5:.2%}"
)

ax.text(
    0.025,
    0.065,
    risk_text,
    transform=ax.transAxes,
    fontsize=10,
    color=TEXT,
    linespacing=1.7,
    family="monospace",
    bbox={
        "facecolor": PANEL,
        "edgecolor": GRID,
        "boxstyle": "round,pad=0.8",
        "alpha": 0.94,
    },
    zorder=10,
)

# 8. Axis formatting
ax.set_xlabel(
    "Trading Day",
    fontsize=11,
    labelpad=12,
)

ax.set_ylabel(
    "Portfolio NAV ($M)",
    fontsize=11,
    labelpad=12,
)

ax.set_xlim(0, HORIZON + 2.5)

# Automatic y-axis limits preserve extreme outcomes
ax.autoscale(axis="y")

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.tick_params(axis="both", labelsize=10)

style_axis(ax)

# 9. Legend
ax.legend(
    loc="upper left",
    fontsize=9.5,
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
    framealpha=0.92,
    borderpad=0.8,
    labelspacing=0.55,
)

# 10. Footer
add_footer(
    fig,
    "Actual paths selected for proximity to pointwise "
    "percentile curves | "
    "Terminal NAV near 5th, 50th and 95th percentiles",
)

# 11. Adjust figure layout
fig.subplots_adjust(
    left=0.11,
    right=0.82,
    top=0.86,
    bottom=0.13,
)

# 12. Save high-resolution figure
save_figure(
    fig,
    "04_monte_carlo_nav_fan_chart.png",
)

# --------------------------------------------------
# 5. Figure 05: VaR vs Expected Shortfall
# --------------------------------------------------

# Refined institutional color palette
VAR_COLOR = "#648FB5"   # Muted steel blue
ES_COLOR = "#C2A36B"    # Muted gold

fig, ax = plt.subplots(figsize=(12, 6.5))

confidence = ["95%", "99%"]

# Wider spacing between confidence groups
x = np.array([0, 1.4])
bar_width = 0.18

# Use existing Monte Carlo risk estimates (USD millions)
var_values = [var95, var99]
es_values = [es95, es99]

# VaR bars
bars_var = ax.bar(
    x - bar_width / 2,
    var_values,
    width=bar_width,
    color=VAR_COLOR,
    label="Value at Risk",
    alpha=0.95,
    zorder=3,
)

# ES bars
bars_es = ax.bar(
    x + bar_width / 2,
    es_values,
    width=bar_width,
    color=ES_COLOR,
    label="Expected Shortfall",
    alpha=0.95,
    zorder=3,
)

# Value labels
for bars in [bars_var, bars_es]:
    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"${value:.2f}M",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=TEXT,
        )

# Header
add_header(
    ax,
    "Value at Risk vs. Expected Shortfall",
    f"{HORIZON}-day horizon | "
    f"${NAV_0 / 1e6:.0f}M portfolio | "
    "USD millions",
)

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(
    confidence,
    fontsize=12,
    color=TEXT,
)

ax.set_xlabel(
    "Confidence Level",
    labelpad=12,
)

ax.set_ylabel(
    "Estimated Loss ($M)",
    labelpad=12,
)

ax.set_xlim(-0.55, 1.95)
ax.set_ylim(0, max(es_values) * 1.20)

# Apply existing dark theme
style_axis(ax)

# Subtle horizontal gridlines
ax.grid(
    axis="y",
    color=GRID,
    alpha=0.30,
    linewidth=0.7,
)

ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

# Legend
ax.legend(
    loc="upper left",
    facecolor=PANEL,
    edgecolor=GRID,
    labelcolor=TEXT,
    fontsize=10,
    framealpha=0.85,
)

# Footer
add_footer(
    fig,
    "VaR: loss quantile | "
    "ES: average loss beyond VaR | "
    "Monte Carlo estimates under model assumptions",
)

# Layout
fig.subplots_adjust(
    left=0.09,
    right=0.96,
    top=0.83,
    bottom=0.17,
)

# Save Figure 05
save_figure(
    fig,
    "05_var_es_comparison.png",
)

print("\nFigure 05 updated successfully.")
