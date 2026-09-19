"""
Portfolio construction and the rebalancing backtest engine.

Three research portfolios plus a benchmark are built on exactly the same
mechanics, so that differences in their results come from the allocation rule
alone:

1. **Equal weight** — 1/N across every holding.
2. **Minimum variance** — long-only global minimum-variance portfolio with a
   box constraint, solved on a shrunk covariance matrix.
3. **Volatility target** — an equal-risk-contribution growth sleeve scaled so
   that its forecast volatility equals the target, with the residual in cash.
4. **Benchmark** — static 60% SPY / 40% AGG.

No-look-ahead discipline
------------------------
Weights applied from rebalance date *t* onwards are estimated **only** from
returns up to and including *t*. Between rebalance dates weights drift with
realised performance; they are reset at the next rebalance. Turnover is
measured against the drifted weights, and transaction costs are charged on the
day of the trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from prisk.config import Config, load_config


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
def estimate_covariance(
    returns: pd.DataFrame, method: str = "ledoit_wolf", annualise: bool = False,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Estimate the return covariance matrix.

    ``ledoit_wolf`` shrinks towards a scaled identity. With 15 assets and a
    756-day window the sample estimator is usable but noisy in its smallest
    eigenvalues, which is precisely where a minimum-variance optimiser
    concentrates weight; shrinkage is the standard remedy and materially
    stabilises the optimised weights.

    **Shrinkage is applied to the correlation matrix, not to the covariance
    matrix.** This matters a great deal for a multi-asset universe. Shrinking
    the covariance matrix directly pulls every variance towards the *average*
    variance, and the variances here span four orders of magnitude — a T-bill
    ETF at 0.5% annualised volatility sits alongside an industrial equity at
    32%. Applied naively, shrinkage reported the cash sleeve at roughly 4.8%
    volatility, nearly ten times its true value, and handed that inflated
    number to the optimiser and to every risk-contribution table.

    Standardising first, shrinking the correlation matrix, and rescaling by the
    sample standard deviations preserves each asset's own volatility exactly
    while still regularising the correlation structure — which is where the
    estimation noise actually lives.
    """
    clean = returns.dropna(how="any")
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf

            std = clean.std(ddof=1)
            usable = std > 0
            standardised = (clean.loc[:, usable] - clean.loc[:, usable].mean()) / std[usable]

            estimator = LedoitWolf(assume_centered=True).fit(standardised.values)
            shrunk = estimator.covariance_
            # Renormalise to an exact unit diagonal, then rescale by the sample
            # standard deviations.
            diagonal = np.sqrt(np.diag(shrunk))
            correlation = shrunk / np.outer(diagonal, diagonal)

            scale = std[usable].values
            rebuilt = correlation * np.outer(scale, scale)

            cov = clean.cov()
            cov.loc[usable, usable] = rebuilt
        except ImportError:  # pragma: no cover - environment dependent
            cov = clean.cov()
    elif method == "sample":
        cov = clean.cov()
    else:  # pragma: no cover - guard
        raise ValueError(f"Unknown covariance method: {method}")

    if annualise:
        cov = cov * trading_days
    return cov


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def minimum_variance_weights(
    cov: pd.DataFrame,
    long_only: bool = True,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    prefer_cvxpy: bool = True,
) -> pd.Series:
    """
    Long-only global minimum-variance weights.

    Solves ``min w'Σw`` subject to ``Σw = 1`` and ``min_weight ≤ w ≤
    max_weight``. Uses CVXPY when it is installed (the problem is a convex QP,
    so the solution is global) and falls back to SciPy SLSQP otherwise. The
    unit test ``test_portfolios.py`` checks that the two agree.
    """
    assets = list(cov.columns)
    sigma = cov.values
    n = len(assets)
    lower = min_weight if long_only else -abs(max_weight)

    if prefer_cvxpy:
        try:
            import cvxpy as cp

            w = cp.Variable(n)
            # PSD projection guards against tiny negative eigenvalues from
            # floating-point noise in the shrinkage estimator.
            objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
            constraints = [cp.sum(w) == 1, w >= lower, w <= max_weight]
            cp.Problem(objective, constraints).solve()
            if w.value is not None:
                weights = pd.Series(np.asarray(w.value).ravel(), index=assets)
                return _clean_weights(weights)
        except Exception:  # pragma: no cover - fall through to SciPy
            pass

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ w

    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=gradient,
        method="SLSQP",
        bounds=[(lower, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _clean_weights(pd.Series(result.x, index=assets))


def risk_parity_weights(
    cov: pd.DataFrame, max_weight: float = 1.0
) -> pd.Series:
    """
    Equal-risk-contribution (ERC) weights.

    Each asset contributes the same share of portfolio variance:
    ``w_i (Σw)_i`` is equalised across ``i``. Solved numerically by minimising
    the dispersion of percentage risk contributions. The inverse-volatility
    portfolio is used as the starting point.
    """
    assets = list(cov.columns)
    sigma = cov.values
    n = len(assets)

    vols = np.sqrt(np.diag(sigma))
    x0 = (1.0 / vols) / (1.0 / vols).sum()

    def objective(w: np.ndarray) -> float:
        portfolio_var = w @ sigma @ w
        if portfolio_var <= 0:
            return 1e6
        contributions = w * (sigma @ w) / portfolio_var
        return float(((contributions - 1.0 / n) ** 2).sum())

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(1e-6, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    weights = result.x if result.success else x0
    return _clean_weights(pd.Series(weights, index=assets))


def _clean_weights(weights: pd.Series, tol: float = 1e-8) -> pd.Series:
    """Zero out numerical dust and renormalise to sum to one."""
    cleaned = weights.copy()
    cleaned[cleaned.abs() < tol] = 0.0
    total = cleaned.sum()
    if abs(total) > tol:
        cleaned = cleaned / total
    return cleaned


# ---------------------------------------------------------------------------
# Weight rules
# ---------------------------------------------------------------------------
WeightRule = Callable[[pd.DataFrame, Config], pd.Series]


def equal_weight_rule(window: pd.DataFrame, cfg: Config) -> pd.Series:
    """1/N across all holdings. Uses no estimation at all."""
    holdings = [c for c in window.columns]
    return pd.Series(1.0 / len(holdings), index=holdings)


def minimum_variance_rule(window: pd.DataFrame, cfg: Config) -> pd.Series:
    """Long-only minimum variance on the shrunk covariance of the window."""
    spec = cfg.portfolios["minimum_variance"]
    cov = estimate_covariance(
        window, method=cfg.portfolios.get("covariance_estimator", "ledoit_wolf")
    )
    return minimum_variance_weights(
        cov,
        long_only=bool(spec.get("long_only", True)),
        min_weight=float(spec.get("min_weight", 0.0)),
        max_weight=float(spec.get("max_weight", 1.0)),
    )


def volatility_target_rule(window: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Scale a growth sleeve to hit a forecast volatility target.

    The growth sleeve is every holding except the cash proxy, allocated by
    equal risk contribution. Let ``σ_g`` be the sleeve's forecast annualised
    volatility and ``σ*`` the target. The sleeve is held at
    ``k = clip(σ* / σ_g, k_min, k_max)`` and ``1 - k`` sits in cash. With
    ``k_max = 1`` no leverage is used, so in calm regimes the realised
    volatility can sit below target — a deliberate, documented asymmetry.
    """
    spec = cfg.portfolios["volatility_target"]
    cash = cfg.riskfree_ticker
    growth_assets = [c for c in window.columns if c != cash]

    cov = estimate_covariance(
        window[growth_assets],
        method=cfg.portfolios.get("covariance_estimator", "ledoit_wolf"),
    )
    if spec.get("base_allocation", "risk_parity") == "risk_parity":
        base = risk_parity_weights(cov)
    else:
        base = pd.Series(1.0 / len(growth_assets), index=growth_assets)

    trading_days = cfg.trading_days
    forecast_vol = float(np.sqrt(base.values @ cov.values @ base.values) * np.sqrt(trading_days))
    target_vol = float(spec["target_volatility"])

    if forecast_vol <= 0:  # pragma: no cover - degenerate guard
        scale = float(spec.get("min_growth_allocation", 0.1))
    else:
        scale = target_vol / forecast_vol
    scale = float(
        np.clip(
            scale,
            float(spec.get("min_growth_allocation", 0.0)),
            float(spec.get("max_growth_allocation", 1.0)),
        )
    )

    weights = pd.Series(0.0, index=window.columns)
    weights[growth_assets] = base * scale
    weights[cash] = weights.get(cash, 0.0) + (1.0 - scale)
    return weights


def benchmark_rule(window: pd.DataFrame, cfg: Config) -> pd.Series:
    """Static 60/40 benchmark weights."""
    target = cfg.portfolios["benchmark"]["weights"]
    weights = pd.Series(0.0, index=window.columns)
    for ticker, weight in target.items():
        if ticker in weights.index:
            weights[ticker] = float(weight)
    return _clean_weights(weights)


WEIGHT_RULES: Dict[str, WeightRule] = {
    "equal_weight": equal_weight_rule,
    "minimum_variance": minimum_variance_rule,
    "volatility_target": volatility_target_rule,
    "benchmark": benchmark_rule,
}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Everything produced by one portfolio run."""

    name: str
    label: str
    returns_gross: pd.Series
    returns_net: pd.Series
    weights_daily: pd.DataFrame
    weights_target: pd.DataFrame
    turnover: pd.Series
    cost_drag: pd.Series
    meta: Dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        """Net-of-cost returns — the series every risk statistic is built on."""
        return self.returns_net

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns_net).cumprod()

    def latest_weights(self) -> pd.Series:
        return self.weights_daily.iloc[-1]


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "Q") -> List[pd.Timestamp]:
    """Last trading day of each period in ``index``."""
    if frequency.upper().startswith("Q"):
        rule = "QE"
    elif frequency.upper().startswith("M"):
        rule = "ME"
    elif frequency.upper().startswith("A") or frequency.upper().startswith("Y"):
        rule = "YE"
    else:  # pragma: no cover - guard
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    series = pd.Series(index, index=index)
    try:
        dates = series.resample(rule).last().dropna()
    except ValueError:  # pragma: no cover - older pandas aliases
        dates = series.resample(rule[0]).last().dropna()

    schedule = [pd.Timestamp(d) for d in dates.values]
    # Drop a rebalance falling on the final observation. Resampling an
    # incomplete trailing period returns its last available date, which for a
    # sample ending mid-quarter is simply "today"; trading on it would book
    # turnover and costs for a position held for zero days.
    return [d for d in schedule if d != index[-1]]


def run_backtest(
    returns: pd.DataFrame,
    rule: WeightRule,
    cfg: Config,
    name: str = "portfolio",
    label: Optional[str] = None,
) -> BacktestResult:
    """
    Run one portfolio through the rebalancing engine.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns of the holdings.
    rule : callable
        ``(window, cfg) -> weights``. ``window`` contains only returns strictly
        available at the rebalance date.

    Returns
    -------
    BacktestResult
    """
    spec = cfg.portfolios
    window_len = int(spec["estimation_window"])
    min_window = int(spec["min_estimation_window"])
    cost_rate = float(spec.get("transaction_cost_bps", 0.0)) / 10_000.0

    index = returns.index
    schedule = rebalance_dates(index, spec.get("rebalance_frequency", "Q"))
    # The portfolio starts on the first rebalance date with enough history.
    schedule = [d for d in schedule if index.get_loc(d) + 1 >= min_window]
    if not schedule:
        raise ValueError(
            "Not enough history to form a portfolio: the sample is shorter than "
            f"min_estimation_window ({min_window} days)."
        )

    start_pos = index.get_loc(schedule[0])
    active_index = index[start_pos:]
    assets = list(returns.columns)

    target_rows: Dict[pd.Timestamp, pd.Series] = {}
    daily_weights = pd.DataFrame(0.0, index=active_index, columns=assets)
    gross = pd.Series(0.0, index=active_index, dtype=float)
    costs = pd.Series(0.0, index=active_index, dtype=float)
    turnover = pd.Series(0.0, index=active_index, dtype=float)

    schedule_set = set(schedule)
    current = pd.Series(0.0, index=assets)

    for i, date in enumerate(active_index):
        if i == 0 or date in schedule_set:
            pos = index.get_loc(date)
            window = returns.iloc[max(0, pos + 1 - window_len): pos + 1]
            if len(window) >= min_window:
                target = rule(window, cfg).reindex(assets).fillna(0.0)
                traded = float((target - current).abs().sum())
                # One-way turnover; the cost rate is quoted round-trip.
                turnover.loc[date] = traded / 2.0
                costs.loc[date] = traded * cost_rate
                current = target
                target_rows[date] = target.copy()

        daily_weights.loc[date] = current.values
        day_return = float((current * returns.loc[date]).sum())
        gross.loc[date] = day_return

        # Drift the weights with the day's realised returns.
        grown = current * (1.0 + returns.loc[date])
        total = grown.sum()
        if total > 0:
            current = grown / total

    net = gross - costs
    weights_target = pd.DataFrame(target_rows).T.sort_index()
    weights_target.index.name = "rebalance_date"

    return BacktestResult(
        name=name,
        label=label or name,
        returns_gross=gross,
        returns_net=net,
        weights_daily=daily_weights,
        weights_target=weights_target,
        turnover=turnover,
        cost_drag=costs,
        meta={
            "n_rebalances": len(weights_target),
            "cost_bps": spec.get("transaction_cost_bps", 0.0),
            "rebalance_frequency": spec.get("rebalance_frequency", "Q"),
            "estimation_window": window_len,
        },
    )


def build_all_portfolios(
    returns: pd.DataFrame, cfg: Optional[Config] = None
) -> Dict[str, BacktestResult]:
    """Run every configured portfolio on the same return panel."""
    cfg = cfg or load_config()
    holdings = [t for t in cfg.holdings if t in returns.columns]
    panel = returns[holdings]

    results: Dict[str, BacktestResult] = {}
    for key in ["equal_weight", "minimum_variance", "volatility_target", "benchmark"]:
        label = cfg.portfolios[key].get("label", key)
        results[key] = run_backtest(panel, WEIGHT_RULES[key], cfg, name=key, label=label)
    return results


def annual_turnover(result: BacktestResult, trading_days: int = 252) -> float:
    """Average annualised one-way turnover."""
    years = len(result.turnover) / trading_days
    return float(result.turnover.sum() / years) if years > 0 else np.nan
