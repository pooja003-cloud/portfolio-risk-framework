"""
Performance and risk-adjusted performance measurement.

All functions take daily **simple** returns. Annualisation uses the configured
252-day convention. Return statistics are geometric (CAGR) rather than
arithmetic, because a compounded figure is what an investor actually earns.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Decimal places used when sorting risk tables. Contributions that are
# conceptually zero come out of the linear algebra as +/-1e-18 and differ in
# the last bits between BLAS implementations, so sorting on the raw value
# orders them differently on different machines. Rounding the sort key (not the
# reported value) to this precision makes those rows genuinely tie, and an
# explicit ticker tiebreak then fixes the order everywhere.
_SORT_PRECISION = 12


def _stable_sort(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Sort descending by ``column``, breaking ties deterministically by index."""
    frame = frame.copy()
    frame.index.name = frame.index.name or "ticker"
    key = frame.index.name
    ordered = (
        frame.assign(_sort_key=frame[column].round(_SORT_PRECISION))
        .sort_values(["_sort_key", key], ascending=[False, True])
        .drop(columns="_sort_key")
    )
    return ordered


# ---------------------------------------------------------------------------
# Return and volatility
# ---------------------------------------------------------------------------
def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the sample."""
    return float((1.0 + returns.dropna()).prod() - 1.0)


def annualised_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Geometric mean return, annualised (CAGR)."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    years = len(clean) / trading_days
    growth = float((1.0 + clean).prod())
    if growth <= 0:  # total loss
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(trading_days))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, trading_days: int = TRADING_DAYS
) -> float:
    """
    Annualised downside deviation below a minimum acceptable return.

    Shortfalls are squared and averaged over **all** observations, not only the
    negative ones, which is the definition consistent with the Sortino ratio.
    """
    clean = returns.dropna()
    daily_mar = mar / trading_days
    shortfall = np.minimum(clean - daily_mar, 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(trading_days))


def rolling_volatility(
    returns: pd.Series, window: int = 63, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series,
    risk_free: Optional[pd.Series] = None,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe ratio on excess returns.

    Excess returns are formed daily against the cash proxy, then annualised, so
    the numerator and denominator refer to the same series.
    """
    excess = _excess_returns(returns, risk_free)
    if excess.std(ddof=1) == 0:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    risk_free: Optional[pd.Series] = None,
    mar: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio: excess return over downside deviation."""
    excess = _excess_returns(returns, risk_free)
    dd = downside_deviation(excess, mar=mar, trading_days=trading_days)
    if dd == 0:
        return np.nan
    return float(excess.mean() * trading_days / dd)


def calmar_ratio(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)["max_drawdown"]
    if mdd == 0:
        return np.nan
    return float(annualised_return(returns, trading_days) / abs(mdd))


def _excess_returns(
    returns: pd.Series, risk_free: Optional[pd.Series]
) -> pd.Series:
    clean = returns.dropna()
    if risk_free is None:
        return clean
    aligned = risk_free.reindex(clean.index).fillna(0.0)
    return clean - aligned


# ---------------------------------------------------------------------------
# Market sensitivity
# ---------------------------------------------------------------------------
def beta(
    returns: pd.Series,
    market: pd.Series,
    risk_free: Optional[pd.Series] = None,
) -> float:
    """
    CAPM beta of the portfolio against the market proxy.

    Estimated on excess returns: ``cov(r_p - r_f, r_m - r_f) / var(r_m - r_f)``.
    """
    frame = pd.concat([returns.rename("p"), market.rename("m")], axis=1).dropna()
    if risk_free is not None:
        rf = risk_free.reindex(frame.index).fillna(0.0)
        frame["p"] = frame["p"] - rf
        frame["m"] = frame["m"] - rf
    variance = frame["m"].var(ddof=1)
    if variance == 0:
        return np.nan
    return float(frame["p"].cov(frame["m"]) / variance)


def alpha(
    returns: pd.Series,
    market: pd.Series,
    risk_free: Optional[pd.Series] = None,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Jensen's alpha from the CAPM regression."""
    frame = pd.concat([returns.rename("p"), market.rename("m")], axis=1).dropna()
    rf = (
        risk_free.reindex(frame.index).fillna(0.0)
        if risk_free is not None
        else pd.Series(0.0, index=frame.index)
    )
    b = beta(returns, market, risk_free)
    daily_alpha = (frame["p"] - rf).mean() - b * (frame["m"] - rf).mean()
    return float(daily_alpha * trading_days)


def tracking_error(
    returns: pd.Series, benchmark: pd.Series, trading_days: int = TRADING_DAYS
) -> float:
    """Annualised standard deviation of the active return."""
    active = (returns - benchmark.reindex(returns.index)).dropna()
    return float(active.std(ddof=1) * np.sqrt(trading_days))


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, trading_days: int = TRADING_DAYS
) -> float:
    """Annualised active return divided by tracking error."""
    te = tracking_error(returns, benchmark, trading_days)
    if te == 0:
        return np.nan
    active = (returns - benchmark.reindex(returns.index)).dropna()
    return float(active.mean() * trading_days / te)


def capture_ratios(
    returns: pd.Series, benchmark: pd.Series
) -> Tuple[float, float]:
    """
    Upside and downside capture against the benchmark.

    Computed on the days when the benchmark rose (respectively fell), as the
    ratio of the portfolio's **mean** return to the benchmark's mean return on
    those days.

    The mean, not the compounded total, is the right aggregator here.
    Compounding a sub-sample of several thousand non-consecutive up-days
    produces an astronomically large number for both legs, and their ratio is
    then dominated by floating-point scale rather than by the portfolio's
    actual participation — it collapses towards zero for any low-beta
    portfolio. Capture on daily data is conventionally a mean ratio for
    exactly this reason.

    An upside capture of 0.60 with a downside capture of 0.40 means the
    portfolio gives up 40% of the market's gains to avoid 60% of its losses —
    an asymmetry that is the whole point of a defensive allocation.
    """
    frame = pd.concat([returns.rename("p"), benchmark.rename("b")], axis=1).dropna()

    def _capture(mask: pd.Series) -> float:
        subset = frame[mask]
        if subset.empty:
            return np.nan
        b = float(subset["b"].mean())
        return float(subset["p"].mean() / b) if b != 0 else np.nan

    return _capture(frame["b"] > 0), _capture(frame["b"] < 0)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Drawdown from the running peak of the compounded equity curve.

    The running peak is floored at the starting value of 1.0. Without that
    floor, a drawdown that begins on the very first day of the sample is
    invisible: the equity curve starts *below* 1.0, the running maximum starts
    with it, and the loss is silently rebased away. Flooring at the initial
    capital treats the starting point as a peak, which is what an investor who
    funded the account on day one experiences.
    """
    equity = (1.0 + returns.dropna()).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> Dict[str, object]:
    """
    Maximum drawdown and its anatomy.

    Returns the depth, the peak and trough dates, the recovery date (``None``
    if the portfolio has not yet recovered), and the duration in trading days.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return {"max_drawdown": np.nan}
    trough = dd.idxmin()
    depth = float(dd.loc[trough])
    equity = (1.0 + returns.dropna()).cumprod()
    peak_date = equity.loc[:trough].idxmax()
    # Mirror the floor applied in drawdown_series: initial capital is a peak.
    peak_value = max(float(equity.loc[peak_date]), 1.0)
    after = equity.loc[trough:]
    recovered = after[after >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "max_drawdown": depth,
        "peak_date": peak_date,
        "trough_date": trough,
        "recovery_date": recovery_date,
        "drawdown_days": int(len(equity.loc[peak_date:trough])),
        "recovery_days": (
            int(len(equity.loc[trough:recovery_date])) if recovery_date is not None else None
        ),
    }


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown — a depth-and-duration measure of pain."""
    dd = drawdown_series(returns)
    return float(np.sqrt((dd ** 2).mean()))


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------
def distribution_stats(returns: pd.Series) -> Dict[str, float]:
    """Skewness, excess kurtosis and a Jarque-Bera normality test."""
    from scipy import stats

    clean = returns.dropna()
    jb_stat, jb_p = stats.jarque_bera(clean)
    return {
        "skewness": float(stats.skew(clean)),
        "excess_kurtosis": float(stats.kurtosis(clean)),  # Fisher: normal = 0
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "worst_day": float(clean.min()),
        "best_day": float(clean.max()),
        "pct_positive_days": float((clean > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------
def risk_contributions(
    weights: pd.Series, cov: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """
    Decompose portfolio volatility into asset contributions.

    For portfolio volatility ``σ_p = sqrt(w'Σw)``:

    * marginal contribution   ``MCTR_i = (Σw)_i / σ_p``
    * component contribution  ``CTR_i  = w_i · MCTR_i``   (these sum to σ_p)
    * percentage contribution ``CTR_i / σ_p``

    This is the key diagnostic for the difference between *capital* allocation
    and *risk* allocation: a 5% weight in a volatile asset can carry 20% of the
    portfolio's risk.
    """
    assets = [a for a in cov.columns if a in weights.index]
    w = weights.reindex(assets).fillna(0.0).values
    sigma = cov.loc[assets, assets].values

    portfolio_var = float(w @ sigma @ w)
    portfolio_vol = float(np.sqrt(portfolio_var))
    if portfolio_vol == 0:  # pragma: no cover - degenerate guard
        return pd.DataFrame(index=assets)

    mctr = (sigma @ w) / portfolio_vol
    ctr = w * mctr

    table = pd.DataFrame(
        {
            "weight": w,
            "annualised_volatility": np.sqrt(np.diag(sigma) * trading_days),
            "marginal_contribution": mctr * np.sqrt(trading_days),
            "component_contribution": ctr * np.sqrt(trading_days),
            "pct_risk_contribution": ctr / portfolio_vol,
        },
        index=assets,
    )
    # A portfolio that holds only two of the fifteen instruments leaves
    # thirteen rows at zero contribution, and without a deterministic tiebreak
    # their order varies by platform — which makes the committed output
    # irreproducible for no reason.
    table.index.name = "ticker"
    return _stable_sort(table, "pct_risk_contribution")


def group_risk_contributions(
    contributions: pd.DataFrame, mapping: Dict[str, str], group_name: str = "sleeve"
) -> pd.DataFrame:
    """Aggregate asset-level risk contributions to sleeve or sector level."""
    frame = contributions.copy()
    frame[group_name] = [mapping.get(idx, "unclassified") for idx in frame.index]
    grouped = frame.groupby(group_name)[
        ["weight", "component_contribution", "pct_risk_contribution"]
    ].sum()
    grouped["risk_to_capital_ratio"] = (
        grouped["pct_risk_contribution"] / grouped["weight"].replace(0, np.nan)
    )
    return _stable_sort(grouped, "pct_risk_contribution")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def performance_summary(
    returns: pd.Series,
    market: Optional[pd.Series] = None,
    risk_free: Optional[pd.Series] = None,
    benchmark: Optional[pd.Series] = None,
    trading_days: int = TRADING_DAYS,
    mar: float = 0.0,
) -> Dict[str, float]:
    """One row of the headline performance and risk table."""
    mdd = max_drawdown(returns)
    summary: Dict[str, float] = {
        "cumulative_return": cumulative_return(returns),
        "annualised_return": annualised_return(returns, trading_days),
        "annualised_volatility": annualised_volatility(returns, trading_days),
        "sharpe_ratio": sharpe_ratio(returns, risk_free, trading_days),
        "sortino_ratio": sortino_ratio(returns, risk_free, mar, trading_days),
        "calmar_ratio": calmar_ratio(returns, trading_days),
        "max_drawdown": mdd["max_drawdown"],
        "ulcer_index": ulcer_index(returns),
    }
    summary.update(distribution_stats(returns))

    if market is not None:
        summary["beta"] = beta(returns, market, risk_free)
        summary["alpha"] = alpha(returns, market, risk_free, trading_days)
    if benchmark is not None:
        summary["tracking_error"] = tracking_error(returns, benchmark, trading_days)
        summary["information_ratio"] = information_ratio(returns, benchmark, trading_days)
        up, down = capture_ratios(returns, benchmark)
        summary["upside_capture"] = up
        summary["downside_capture"] = down
    return summary
