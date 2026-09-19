"""
Visualisation layer.

Every figure in this project is produced here, from one style definition, so
that the whole report reads as a single system rather than a dozen defaults.

Design rules applied throughout
-------------------------------
* **Colour follows the entity, not its rank.** Each portfolio keeps the same
  hue in every chart; the benchmark is always the neutral dashed reference
  line, because it is a reference rather than a competing series.
* **Categorical hues are assigned in a fixed, colour-vision-deficiency-safe
  order** (validated for adjacent-pair and all-pair separation), never cycled.
* **One axis.** No chart in this module has two y-scales.
* **Sequential means one hue; diverging means two hues around a neutral grey.**
  The correlation heatmap is diverging because -1 and +1 are opposites and zero
  is genuinely "nothing".
* Identity is never carried by colour alone: every multi-series chart has a
  legend, key charts carry direct labels, and the underlying table of every
  figure is written to ``outputs/tables`` as CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical order.
CATEGORICAL: List[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Portfolios keep these hues in every figure. The benchmark is deliberately a
# neutral dashed line rather than a fourth hue.
PORTFOLIO_COLORS: Dict[str, str] = {
    "equal_weight": CATEGORICAL[0],
    "minimum_variance": CATEGORICAL[1],
    "volatility_target": CATEGORICAL[2],
    "benchmark": INK_SECONDARY,
}
PORTFOLIO_STYLES: Dict[str, str] = {
    "equal_weight": "-",
    "minimum_variance": "-",
    "volatility_target": "-",
    "benchmark": "--",
}

SLEEVE_COLORS: Dict[str, str] = {
    "equity": CATEGORICAL[0],
    "equity_beta": CATEGORICAL[1],
    "fixed_income": CATEGORICAL[2],
    "commodity": CATEGORICAL[3],
    "real_estate": CATEGORICAL[4],
    "cash": CATEGORICAL[5],
}

# Diverging ramp for correlations: blue <-> neutral grey <-> red.
DIVERGING = LinearSegmentedColormap.from_list(
    "prisk_diverging",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a6a6", "#d03b3b", "#7d1f1f"],
)
# Single-hue sequential ramp for magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "prisk_sequential", ["#cde2fb", "#3987e5", "#184f95", "#0d366b"]
)


def use_style() -> None:
    """Apply the project's matplotlib style. Call once per notebook or script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "figure.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def _pct(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")


def _title(ax: plt.Axes, title: str, subtitle: Optional[str] = None) -> None:
    """Left-aligned title with an optional deck line beneath it."""
    if subtitle:
        # Reserve room for the deck so the two never collide.
        ax.set_title(title, color=INK_PRIMARY, pad=30)
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=INK_SECONDARY, va="bottom",
        )
    else:
        ax.set_title(title, color=INK_PRIMARY)


def save(fig: plt.Figure, path: Path | str, dpi: int = 160) -> Path:
    """Write a figure to disk and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _label(name: str, labels: Optional[Dict[str, str]]) -> str:
    return (labels or {}).get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def plot_cumulative_growth(
    returns_by_portfolio: Dict[str, pd.Series],
    labels: Optional[Dict[str, str]] = None,
    log_scale: bool = True,
    initial: float = 1.0,
) -> plt.Figure:
    """
    Growth of one unit of capital.

    Plotted on a log scale by default: on a linear scale a nineteen-year chart
    makes the last five years look like all the volatility there ever was,
    because equal vertical distances no longer mean equal percentage moves.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, series in returns_by_portfolio.items():
        curve = initial * (1.0 + series).cumprod()
        ax.plot(
            curve.index, curve.values,
            color=PORTFOLIO_COLORS.get(name, CATEGORICAL[0]),
            linestyle=PORTFOLIO_STYLES.get(name, "-"),
            label=_label(name, labels),
            zorder=3 if name != "benchmark" else 2,
        )
        # Direct end-of-line label: identity is not left to colour alone.
        ax.annotate(
            f"{curve.iloc[-1]:.2f}x",
            xy=(curve.index[-1], curve.iloc[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=PORTFOLIO_COLORS.get(name, CATEGORICAL[0]),
            fontsize=9, va="center", fontweight="semibold",
        )
    if log_scale:
        ax.set_yscale("log")
        top = max(
            float((initial * (1.0 + s).cumprod()).max())
            for s in returns_by_portfolio.values()
        )
        ticks = [t for t in [0.5, 1, 2, 4, 8, 16, 32] if t <= top * 1.6]
        ax.set_yticks(ticks)
        ax.set_yticks([], minor=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))

    _title(
        ax,
        "Growth of $1",
        "Net of 5bp round-trip transaction costs; log scale" if log_scale else
        "Net of 5bp round-trip transaction costs",
    )
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncols=2)
    ax.margins(x=0.06)
    fig.tight_layout()
    return fig


def plot_drawdowns(
    returns_by_portfolio: Dict[str, pd.Series],
    labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Drawdown from the running peak, filled to the zero baseline."""
    from prisk.metrics import drawdown_series

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, series in returns_by_portfolio.items():
        dd = drawdown_series(series)
        color = PORTFOLIO_COLORS.get(name, CATEGORICAL[0])
        ax.plot(dd.index, dd.values, color=color,
                linestyle=PORTFOLIO_STYLES.get(name, "-"),
                label=_label(name, labels), linewidth=1.6)
        if name != "benchmark":
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.10, linewidth=0)

    ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.yaxis.set_major_formatter(_pct(0))
    _title(ax, "Drawdown from prior peak",
           "Depth and duration of losses, not just their size")
    ax.legend(loc="lower left", ncols=2)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns_by_portfolio: Dict[str, pd.Series],
    window: int = 63,
    target: Optional[float] = None,
    labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Rolling annualised volatility, with the volatility target marked."""
    from prisk.metrics import rolling_volatility

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, series in returns_by_portfolio.items():
        vol = rolling_volatility(series, window)
        ax.plot(vol.index, vol.values,
                color=PORTFOLIO_COLORS.get(name, CATEGORICAL[0]),
                linestyle=PORTFOLIO_STYLES.get(name, "-"),
                label=_label(name, labels), linewidth=1.6)
    if target is not None:
        ax.axhline(target, color=INK_MUTED, linewidth=1.2, linestyle=":")
        ax.annotate(f"{target:.0%} target", xy=(0.01, target), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9)

    ax.yaxis.set_major_formatter(_pct(0))
    _title(ax, f"Rolling {window}-day annualised volatility",
           "Risk is not a constant — every portfolio's own volatility moves by a factor of three or more")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return fig


def plot_risk_return_scatter(
    summary: pd.DataFrame,
    labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Annualised return against annualised volatility, one point per portfolio."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, row in summary.iterrows():
        color = PORTFOLIO_COLORS.get(name, CATEGORICAL[0])
        ax.scatter(row["annualised_volatility"], row["annualised_return"],
                   s=170, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            f"{_label(name, labels)}\nSharpe {row['sharpe_ratio']:.2f}",
            xy=(row["annualised_volatility"], row["annualised_return"]),
            xytext=(10, -4), textcoords="offset points",
            fontsize=9, color=INK_SECONDARY, va="top",
        )
    ax.xaxis.set_major_formatter(_pct(0))
    ax.yaxis.set_major_formatter(_pct(0))
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return")
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    _title(ax, "Risk and return", "Whole sample, net of costs")
    ax.margins(0.22)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Daily return correlation",
    subtitle: Optional[str] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Correlation matrix on a diverging ramp.

    Diverging, not sequential, because zero correlation is a meaningful neutral
    midpoint and +0.8 versus -0.8 are opposites rather than "more" and "less".
    """
    size = max(7.0, 0.52 * len(correlation) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    image = ax.imshow(correlation.values, cmap=DIVERGING, vmin=-1, vmax=1)

    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation)), correlation.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for i in range(len(correlation)):
            for j in range(len(correlation)):
                value = correlation.values[i, j]
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if abs(value) > 0.55 else INK_SECONDARY,
                )

    bar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(color=INK_MUTED, labelcolor=INK_MUTED)
    _title(ax, title, subtitle)
    fig.tight_layout()
    return fig


def plot_risk_contribution(
    contributions: pd.DataFrame,
    title: str = "Capital weight versus risk contribution",
    subtitle: str = "Where the money sits is not where the risk sits",
) -> plt.Figure:
    """
    Paired horizontal bars: capital weight against percentage risk
    contribution.

    The gap between the two bars is the whole point of the chart — an asset
    whose risk bar runs well past its weight bar is consuming risk budget out
    of proportion to the capital committed to it.
    """
    frame = contributions.sort_values("pct_risk_contribution")
    y = np.arange(len(frame))
    height = 0.38
    gap = 0.02  # 2px-equivalent surface gap between the paired fills

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(frame) + 2)))
    ax.barh(y + (height + gap) / 2, frame["weight"], height=height,
            color=CATEGORICAL[0], label="Capital weight")
    ax.barh(y - (height + gap) / 2, frame["pct_risk_contribution"], height=height,
            color=CATEGORICAL[1], label="Risk contribution")

    ax.set_yticks(y, frame.index)
    ax.xaxis.set_major_formatter(_pct(0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASELINE, linewidth=1.0)
    _title(ax, title, subtitle)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_allocation_over_time(
    weights_daily: pd.DataFrame,
    sleeve_map: Dict[str, str],
    title: str = "Allocation through time",
    subtitle: Optional[str] = None,
) -> plt.Figure:
    """Stacked area of sleeve weights, aggregated from asset-level weights."""
    # Transpose-group-transpose rather than groupby(axis=1), which pandas 3
    # removed.
    groups = [sleeve_map.get(c, "other") for c in weights_daily.columns]
    grouped = weights_daily.T.groupby(groups).sum().T
    order = [s for s in SLEEVE_COLORS if s in grouped.columns]
    order += [s for s in grouped.columns if s not in order]
    grouped = grouped[order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.stackplot(
        grouped.index,
        [grouped[c].values for c in grouped.columns],
        labels=[c.replace("_", " ").title() for c in grouped.columns],
        colors=[SLEEVE_COLORS.get(c, CATEGORICAL[6]) for c in grouped.columns],
        edgecolor=SURFACE, linewidth=0.8,  # surface gap between segments
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(_pct(0))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.5)
    _title(ax, title, subtitle)
    ax.legend(loc="upper center", ncols=6, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------
def plot_return_distribution(
    returns: pd.Series,
    var: float,
    expected_shortfall: float,
    confidence: float = 0.99,
    label: str = "Portfolio",
) -> plt.Figure:
    """
    Daily return histogram with the VaR and ES thresholds marked, against a
    fitted normal density.

    The visible gap between the histogram's left tail and the normal curve is
    the reason parametric-normal VaR understates risk.
    """
    from scipy import stats

    clean = returns.dropna()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    counts, bins, patches = ax.hist(
        clean.values, bins=120, color="#9ec5f4", edgecolor=SURFACE, linewidth=0.4,
        density=True,
    )
    for patch, left in zip(patches, bins[:-1]):
        if left <= -var:
            patch.set_facecolor(STATUS["critical"])

    grid = np.linspace(clean.min(), clean.max(), 600)
    ax.plot(grid, stats.norm.pdf(grid, clean.mean(), clean.std(ddof=1)),
            color=INK_SECONDARY, linewidth=1.8, linestyle="--",
            label="Fitted normal density")

    ax.axvline(-var, color=STATUS["critical"], linewidth=2)
    ax.axvline(-expected_shortfall, color="#7d1f1f", linewidth=2, linestyle=":")
    top = ax.get_ylim()[1]
    ax.annotate(f"VaR {confidence:.0%}\n{var:.2%}", xy=(-var, top * 0.78),
                xytext=(-46, 0), textcoords="offset points",
                color=STATUS["critical"], fontsize=9, fontweight="semibold", ha="right")
    ax.annotate(f"ES {confidence:.0%}\n{expected_shortfall:.2%}",
                xy=(-expected_shortfall, top * 0.45),
                xytext=(-46, 0), textcoords="offset points",
                color="#7d1f1f", fontsize=9, fontweight="semibold", ha="right")

    ax.xaxis.set_major_formatter(_pct(0))
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    _title(ax, f"{label}: distribution of daily returns",
           "Shaded bars are days beyond VaR; ES is their average")
    fig.tight_layout()
    return fig


def plot_var_comparison(
    comparison: pd.DataFrame,
    horizon: int = 1,
    value_column_pairs: Optional[Sequence[tuple]] = None,
) -> plt.Figure:
    """Grouped bars comparing VaR estimates across methods and confidence levels."""
    pairs = value_column_pairs or [
        ("historical_var", "Historical"),
        ("parametric_normal_var", "Parametric normal"),
        ("parametric_t_var", "Parametric Student-t"),
        ("cornish_fisher_var", "Cornish-Fisher"),
        ("monte_carlo_var", "Monte Carlo"),
    ]
    subset = comparison[comparison["horizon_days"] == horizon]
    confidences = sorted(subset["confidence"].unique())
    pairs = [(c, label) for c, label in pairs if c in subset.columns]

    x = np.arange(len(confidences))
    width = 0.8 / max(len(pairs), 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, (column, label) in enumerate(pairs):
        values = [
            float(subset.loc[subset["confidence"] == c, column].iloc[0])
            for c in confidences
        ]
        offset = (i - (len(pairs) - 1) / 2) * (width + 0.004)
        bars = ax.bar(x + offset, values, width=width,
                      color=CATEGORICAL[i % len(CATEGORICAL)], label=label)
        ax.bar_label(bars, fmt="%.2f%%",
                     labels=[f"{v:.2%}" for v in values],
                     padding=3, fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x, [f"{c:.0%} confidence" for c in confidences])
    ax.yaxis.set_major_formatter(_pct(1))
    ax.set_ylabel(f"{horizon}-day VaR (loss)")
    _title(ax, f"{horizon}-day Value at Risk by estimation method",
           "Same portfolio, same data — the method itself moves the answer")
    ax.legend(loc="upper left", ncols=2)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig


def plot_var_backtest(
    realised: pd.Series,
    var_forecast: pd.Series,
    exceptions: pd.Series,
    confidence: float = 0.99,
    method: str = "historical",
) -> plt.Figure:
    """
    Realised daily returns against the VaR forecast, with breaches highlighted.

    A well-calibrated model shows breaches scattered across the sample. Breaches
    bunched into a few months are the visual signature of a model that ignores
    volatility clustering.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.scatter(realised.index, realised.values, s=3.2, color=BASELINE,
               label="Daily return", zorder=2)
    ax.plot(var_forecast.index, -var_forecast.values, color=CATEGORICAL[0],
            linewidth=1.6, label=f"{confidence:.0%} VaR forecast", zorder=3)

    breaches = realised[exceptions == 1]
    ax.scatter(breaches.index, breaches.values, s=26,
               color=STATUS["critical"], edgecolor=SURFACE, linewidth=0.8,
               label=f"Exceptions ({len(breaches)})", zorder=4)

    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.yaxis.set_major_formatter(_pct(0))
    _title(ax, f"VaR backtest — {method.replace('_', ' ')} method at {confidence:.0%}",
           "Every red dot is a day the model said should have been rarer than it was")
    ax.legend(loc="lower left", ncols=3)
    fig.tight_layout()
    return fig


def plot_exceptions_by_year(
    by_year: pd.DataFrame, confidence: float = 0.99
) -> plt.Figure:
    """Exception counts by calendar year against the expected count."""
    expected = by_year["observations"] * (1 - confidence)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = [
        STATUS["critical"] if actual > 2 * exp else CATEGORICAL[0]
        for actual, exp in zip(by_year["exceptions"], expected)
    ]
    bars = ax.bar(by_year.index.astype(str), by_year["exceptions"],
                  color=colors, width=0.68)
    ax.bar_label(bars, fontsize=8, color=INK_SECONDARY, padding=2)
    ax.plot(by_year.index.astype(str), expected.values, color=INK_SECONDARY,
            linestyle="--", linewidth=1.6, label="Expected count")
    ax.set_ylabel("Exceptions")
    _title(ax, "VaR exceptions by calendar year",
           "Clustering, not the total count, is what invalidates a model")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------
def plot_stress_scenarios(
    table: pd.DataFrame,
    value_column: str = "portfolio_impact",
    title: str = "Hypothetical stress scenarios",
    subtitle: str = "Hypothetical assumptions, not forecasts",
    labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Grouped horizontal bars: scenario impact for each portfolio."""
    pivot = table.pivot(index="scenario", columns="portfolio", values=value_column)
    order = [c for c in PORTFOLIO_COLORS if c in pivot.columns]
    order += [c for c in pivot.columns if c not in order]
    pivot = pivot[order].sort_values(order[0])

    y = np.arange(len(pivot))
    height = 0.8 / len(pivot.columns)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.85 * len(pivot) + 2)))
    for i, column in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * (height + 0.012)
        color = PORTFOLIO_COLORS.get(column, CATEGORICAL[i % len(CATEGORICAL)])
        bars = ax.barh(y + offset, pivot[column].values, height=height,
                       color=color, label=_label(column, labels))
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in pivot[column].values],
                     padding=3, fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y, pivot.index)
    ax.axvline(0, color=BASELINE, linewidth=1.0)
    ax.xaxis.set_major_formatter(_pct(0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _title(ax, title, subtitle)
    # Below the axes: the bars run to the left edge, so any in-axes corner
    # would sit on top of a value label.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=4)
    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


def plot_correlation_stress(frame: pd.DataFrame) -> plt.Figure:
    """Portfolio volatility as pairwise correlations are forced upwards."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [INK_SECONDARY] + [
        SEQUENTIAL(0.35 + 0.2 * i) for i in range(len(frame) - 1)
    ]
    bars = ax.bar(frame["scenario"], frame["annualised_volatility"],
                  color=colors, width=0.62)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in frame["annualised_volatility"]],
                 padding=3, fontsize=9, color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(_pct(0))
    ax.set_ylabel("Annualised volatility")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _title(ax, "Correlation stress",
           "Individual volatilities held fixed; only the correlations move")
    ax.margins(y=0.16)
    fig.tight_layout()
    return fig


def plot_monte_carlo_distribution(
    simulated: np.ndarray,
    var: float,
    expected_shortfall: float,
    confidence: float = 0.99,
    horizon: int = 10,
    n_paths_label: Optional[str] = None,
) -> plt.Figure:
    """Simulated horizon-return distribution with VaR and ES marked."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    counts, bins, patches = ax.hist(simulated, bins=140, color="#9ec5f4",
                                    edgecolor=SURFACE, linewidth=0.3, density=True)
    for patch, left in zip(patches, bins[:-1]):
        if left <= -var:
            patch.set_facecolor(STATUS["critical"])

    ax.axvline(-var, color=STATUS["critical"], linewidth=2)
    ax.axvline(-expected_shortfall, color="#7d1f1f", linewidth=2, linestyle=":")
    top = ax.get_ylim()[1]
    ax.annotate(f"VaR {confidence:.0%}: {var:.2%}", xy=(-var, top * 0.8),
                xytext=(8, 0), textcoords="offset points",
                color=STATUS["critical"], fontsize=9, fontweight="semibold")
    ax.annotate(f"ES {confidence:.0%}: {expected_shortfall:.2%}",
                xy=(-expected_shortfall, top * 0.55),
                xytext=(-8, 0), textcoords="offset points", ha="right",
                color="#7d1f1f", fontsize=9, fontweight="semibold")

    ax.xaxis.set_major_formatter(_pct(0))
    ax.set_xlabel(f"Simulated {horizon}-day return")
    ax.set_ylabel("Density")
    _title(ax, f"Monte Carlo: simulated {horizon}-day return distribution",
           n_paths_label or "")
    fig.tight_layout()
    return fig


def plot_episode_comparison(
    table: pd.DataFrame,
    labels: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """Total return through each historical episode, by portfolio."""
    return plot_stress_scenarios(
        table,
        value_column="total_return",
        title="Historical episode replay",
        subtitle="Current weights applied to the actual returns of each crisis",
        labels=labels,
    )
