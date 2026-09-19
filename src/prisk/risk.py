"""
Value at Risk, Expected Shortfall, and Monte Carlo simulation.

Sign convention
---------------
VaR and ES are reported as **positive numbers representing losses**. A 1-day
99% VaR of 0.0231 means: on 99 days out of 100 the portfolio is not expected to
lose more than 2.31% of its value in one day.

Three estimation routes are implemented, and they disagree for reasons worth
stating explicitly:

* **Historical** makes no distributional assumption. It reproduces the actual
  fat tails, skew, and volatility clustering of the sample, but it can only
  produce losses that have already happened, and it weights a 2008 observation
  the same as yesterday's.
* **Parametric** assumes a distribution and needs only a mean and a covariance.
  Under a normal assumption it systematically understates tail risk for daily
  equity-like returns, whose excess kurtosis is large and positive. A Student-t
  fit and a Cornish-Fisher expansion are provided as tail-aware alternatives.
* **Monte Carlo** re-samples from an assumed multivariate process. It is the
  most flexible — it can price horizons and portfolios with no history — but
  its output is only as good as the covariance matrix and the distributional
  family fed into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------
def historical_var(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Historical (empirical) VaR.

    The ``1 - confidence`` empirical quantile of the realised return
    distribution. Multi-day horizons are scaled by ``sqrt(horizon)``; see
    :func:`historical_var_overlapping` for the assumption-free alternative.
    """
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    quantile = float(np.quantile(clean.values, 1.0 - confidence))
    return float(-quantile * np.sqrt(horizon))


def historical_var_overlapping(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 10
) -> float:
    """
    Historical VaR computed on overlapping ``horizon``-day compounded returns.

    Makes no square-root-of-time assumption, so it captures the serial
    dependence (mean reversion or momentum) present in the data, at the cost of
    a much smaller effective sample of independent observations.
    """
    clean = returns.dropna()
    compounded = (1.0 + clean).rolling(horizon).apply(np.prod, raw=True) - 1.0
    compounded = compounded.dropna()
    if compounded.empty:
        return np.nan
    return float(-np.quantile(compounded.values, 1.0 - confidence))


def historical_expected_shortfall(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Historical Expected Shortfall (Conditional VaR).

    The average loss on the days that breached VaR — the answer to "how bad is
    bad?", which VaR alone never gives. Unlike VaR, ES is a coherent risk
    measure: it is sub-additive, so diversification can never increase it.
    """
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    threshold = np.quantile(clean.values, 1.0 - confidence)
    tail = clean[clean <= threshold]
    if tail.empty:
        return np.nan
    return float(-tail.mean() * np.sqrt(horizon))


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    distribution: str = "normal",
    dof: Optional[float] = None,
) -> float:
    """
    Parametric (variance-covariance) VaR.

    ``normal``
        ``VaR = -(μ·h + z_(1-c)·σ·sqrt(h))`` with ``z`` the standard normal
        quantile.
    ``student_t``
        A Student-t is fitted by maximum likelihood and the VaR is read
        directly off the fitted location and scale:
        ``VaR = -(μ̂·h + ν-quantile · ŝ · sqrt(h))``.

        Using the *fitted* scale rather than the sample standard deviation
        matters. Maximum likelihood down-weights outliers when it estimates ŝ
        and pushes the fat tail into a low ν instead; re-imposing the sample
        σ on top of a low-ν quantile double-counts the scale and produces a
        VaR that is far too small. The backtest in ``backtest.py`` makes the
        difference visible — the naive version breaches several times more
        often than its confidence level allows.

    Note that a Student-t does not always give a *larger* VaR than a normal.
    Standardised for scale, the t has thinner shoulders and a fatter extreme
    tail, so it typically sits below the normal at 95% and above it at 99%.
    """
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))

    if distribution == "normal":
        quantile = stats.norm.ppf(1.0 - confidence)
        return float(-(mu * horizon + quantile * sigma * np.sqrt(horizon)))

    if distribution == "student_t":
        if dof is None:
            dof, loc, scale = stats.t.fit(clean.values)
        else:
            _, loc, scale = stats.t.fit(clean.values, f0=float(dof))
        dof = max(float(dof), 2.05)  # variance is undefined at or below 2
        quantile = stats.t.ppf(1.0 - confidence, dof)
        return float(-(loc * horizon + quantile * scale * np.sqrt(horizon)))

    raise ValueError(f"Unknown distribution: {distribution}")


def cornish_fisher_var(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Cornish-Fisher (modified) VaR.

    Adjusts the normal quantile for the sample's skewness and excess kurtosis:

    ``z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36``

    A cheap way to keep the analytical tractability of the parametric approach
    while acknowledging that daily returns are neither symmetric nor mesokurtic.
    """
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    skew = float(stats.skew(clean.values))
    kurt = float(stats.kurtosis(clean.values))  # excess

    z = stats.norm.ppf(1.0 - confidence)
    z_cf = (
        z
        + (z ** 2 - 1) * skew / 6.0
        + (z ** 3 - 3 * z) * kurt / 24.0
        - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36.0
    )
    return float(-(mu * horizon + z_cf * sigma * np.sqrt(horizon)))


def parametric_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    distribution: str = "normal",
    dof: Optional[float] = None,
) -> float:
    """
    Closed-form Expected Shortfall.

    Normal
        ``ES = -(μ·h + σ·sqrt(h)·(-φ(z)/(1-c)))``.
    Student-t
        ``ES = -(μ̂·h + ŝ·sqrt(h)·(-(ν + t²)/(ν - 1)·f(t)/(1-c)))`` evaluated on
        the maximum-likelihood location and scale, consistent with
        :func:`parametric_var`.
    """
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    tail_prob = 1.0 - confidence

    if distribution == "normal":
        mu = float(clean.mean())
        sigma = float(clean.std(ddof=1))
        z = stats.norm.ppf(tail_prob)
        es_standard = -stats.norm.pdf(z) / tail_prob
        return float(-(mu * horizon + es_standard * sigma * np.sqrt(horizon)))

    if distribution == "student_t":
        if dof is None:
            dof, loc, scale = stats.t.fit(clean.values)
        else:
            _, loc, scale = stats.t.fit(clean.values, f0=float(dof))
        dof = max(float(dof), 2.05)
        t_q = stats.t.ppf(tail_prob, dof)
        es_standard = (
            -(dof + t_q ** 2) / (dof - 1.0) * stats.t.pdf(t_q, dof) / tail_prob
        )
        return float(-(loc * horizon + es_standard * scale * np.sqrt(horizon)))

    raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    """Simulated portfolio return distribution and the risk read off it."""

    simulated_returns: np.ndarray
    var: Dict[float, float]
    expected_shortfall: Dict[float, float]
    horizon: int
    n_simulations: int
    distribution: str
    dof: Optional[float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "confidence": list(self.var.keys()),
                "monte_carlo_var": list(self.var.values()),
                "monte_carlo_es": [self.expected_shortfall[c] for c in self.var],
            }
        )


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    confidences: Iterable[float] = (0.95, 0.99),
    horizon: int = 1,
    n_simulations: int = 100_000,
    distribution: str = "student_t",
    dof: Optional[float] = None,
    seed: int = 42,
    cov: Optional[pd.DataFrame] = None,
) -> MonteCarloResult:
    """
    Monte Carlo VaR and ES from a simulated multivariate return process.

    Method
    ------
    1. Estimate the mean vector ``μ`` and covariance matrix ``Σ`` of the asset
       returns (a shrunk ``Σ`` may be supplied).
    2. Draw ``n_simulations`` paths of ``horizon`` daily joint return vectors
       from a multivariate normal or a multivariate Student-t built from the
       Cholesky factor of ``Σ``. The Student-t is generated as a normal-variance
       mixture, ``x = μ + L·z·sqrt(ν/χ²_ν)``, which preserves the correlation
       structure while producing joint tail events far more often than a normal
       — the empirically relevant behaviour, since assets crash together.
    3. Compound each path at fixed weights and read the quantiles off the
       resulting distribution of horizon returns.

    Fixed weights are the correct convention for a risk measure: they answer
    "what could this portfolio lose", not "what could a rebalancing strategy
    lose".
    """
    assets = [a for a in asset_returns.columns if a in weights.index]
    data = asset_returns[assets].dropna(how="any")
    w = weights.reindex(assets).fillna(0.0).values

    mu = data.mean().values
    if cov is None:
        sigma = data.cov().values
    else:
        sigma = cov.loc[assets, assets].values

    # Cholesky with a jitter fallback for numerically non-PSD matrices.
    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:  # pragma: no cover - rare
        jitter = 1e-10 * np.eye(len(assets))
        chol = np.linalg.cholesky(sigma + jitter)

    rng = np.random.default_rng(seed)
    n_assets = len(assets)

    if distribution == "student_t":
        if dof is None:
            portfolio_hist = data.values @ w
            fitted_dof, _, _ = stats.t.fit(portfolio_hist)
            # Floor of 3. Maximum likelihood routinely returns nu below 3 for a
            # long daily series, but that is the fit telling you the data is a
            # *mixture* of volatility regimes rather than a single t: below 3
            # the fourth moment is infinite and the simulated sample kurtosis
            # runs into the hundreds, which is a property of the estimator, not
            # of the portfolio. Ceiling of 30, above which the t is normal for
            # practical purposes.
            dof = float(np.clip(fitted_dof, 3.0, 30.0))
        scale = np.sqrt((dof - 2.0) / dof)  # unit-variance standardisation
    elif distribution == "normal":
        dof = None
        scale = 1.0
    else:  # pragma: no cover - guard
        raise ValueError(f"Unknown distribution: {distribution}")

    path_returns = np.empty((n_simulations, horizon))
    for step in range(horizon):
        z = rng.standard_normal((n_simulations, n_assets))
        if distribution == "student_t":
            chi = rng.chisquare(dof, size=(n_simulations, 1))
            z = z * np.sqrt(dof / chi) * scale
        shocks = mu + z @ chol.T
        path_returns[:, step] = shocks @ w

    horizon_returns = np.prod(1.0 + path_returns, axis=1) - 1.0

    var: Dict[float, float] = {}
    es: Dict[float, float] = {}
    for confidence in confidences:
        threshold = np.quantile(horizon_returns, 1.0 - confidence)
        var[confidence] = float(-threshold)
        tail = horizon_returns[horizon_returns <= threshold]
        es[confidence] = float(-tail.mean()) if tail.size else np.nan

    return MonteCarloResult(
        simulated_returns=horizon_returns,
        var=var,
        expected_shortfall=es,
        horizon=horizon,
        n_simulations=n_simulations,
        distribution=distribution,
        dof=dof,
    )


# ---------------------------------------------------------------------------
# Component VaR
# ---------------------------------------------------------------------------
def component_var(
    weights: pd.Series,
    cov: pd.DataFrame,
    confidence: float = 0.95,
    horizon: int = 1,
    mean_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Decompose parametric VaR into per-asset contributions.

    Because VaR is homogeneous of degree one in the weights, Euler's theorem
    gives an exact additive decomposition:

    ``VaR = Σ_i w_i · ∂VaR/∂w_i``

    ``component_var`` columns therefore sum to the portfolio VaR. This is the
    number that answers "where is my loss potential concentrated?" — and it can
    look very different from the capital allocation.
    """
    assets = [a for a in cov.columns if a in weights.index]
    w = weights.reindex(assets).fillna(0.0).values
    sigma = cov.loc[assets, assets].values
    mu = (
        mean_returns.reindex(assets).fillna(0.0).values
        if mean_returns is not None
        else np.zeros(len(assets))
    )

    portfolio_vol = float(np.sqrt(w @ sigma @ w))
    if portfolio_vol == 0:  # pragma: no cover
        return pd.DataFrame(index=assets)

    z = stats.norm.ppf(1.0 - confidence)
    scale = np.sqrt(horizon)
    marginal = -(mu * horizon) - z * scale * (sigma @ w) / portfolio_vol
    component = w * marginal
    total = component.sum()

    table = pd.DataFrame(
        {
            "weight": w,
            "marginal_var": marginal,
            "component_var": component,
            "pct_var_contribution": component / total if total != 0 else np.nan,
        },
        index=assets,
    )
    # Zero-weight holdings all contribute nothing, and without a deterministic
    # tiebreak their order varies between platforms. See
    # ``prisk.metrics._stable_sort``.
    from prisk.metrics import _stable_sort

    table.index.name = "ticker"
    return _stable_sort(table, "component_var")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def var_comparison(
    portfolio_returns: pd.Series,
    asset_returns: Optional[pd.DataFrame] = None,
    weights: Optional[pd.Series] = None,
    confidences: Iterable[float] = (0.95, 0.99),
    horizons: Iterable[int] = (1, 10),
    mc_kwargs: Optional[Dict] = None,
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Build the side-by-side VaR / ES comparison table across all methods.

    Returns one row per (horizon, confidence) with historical, normal,
    Student-t, Cornish-Fisher and Monte Carlo figures, both as a percentage of
    portfolio value and as a currency loss on ``portfolio_value``.
    """
    mc_kwargs = dict(mc_kwargs or {})
    rows = []

    for horizon in horizons:
        for confidence in confidences:
            row: Dict[str, object] = {
                "horizon_days": horizon,
                "confidence": confidence,
                "historical_var": historical_var(portfolio_returns, confidence, horizon),
                "historical_es": historical_expected_shortfall(
                    portfolio_returns, confidence, horizon
                ),
                "parametric_normal_var": parametric_var(
                    portfolio_returns, confidence, horizon, "normal"
                ),
                "parametric_normal_es": parametric_expected_shortfall(
                    portfolio_returns, confidence, horizon, "normal"
                ),
                "parametric_t_var": parametric_var(
                    portfolio_returns, confidence, horizon, "student_t"
                ),
                "parametric_t_es": parametric_expected_shortfall(
                    portfolio_returns, confidence, horizon, "student_t"
                ),
                "cornish_fisher_var": cornish_fisher_var(
                    portfolio_returns, confidence, horizon
                ),
            }
            if asset_returns is not None and weights is not None:
                mc = monte_carlo_var(
                    asset_returns,
                    weights,
                    confidences=[confidence],
                    horizon=horizon,
                    **mc_kwargs,
                )
                row["monte_carlo_var"] = mc.var[confidence]
                row["monte_carlo_es"] = mc.expected_shortfall[confidence]
            rows.append(row)

    table = pd.DataFrame(rows)
    for column in [c for c in table.columns if c.endswith(("_var", "_es"))]:
        table[f"{column}_usd"] = table[column] * portfolio_value
    return table
