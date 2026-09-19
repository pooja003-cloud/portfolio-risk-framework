"""
Stress testing.

Two complementary families of test are implemented, because they answer
different questions.

**Historical episode replay** takes the portfolio's *current* weights and runs
them through the actual asset returns of a named crisis. It asks: "if the
Global Financial Crisis happened again to the book I hold today, what would
it cost me?" Its strength is internal consistency — the cross-asset
correlations, the volatility path and the sequencing are all real, not assumed.
Its weakness is that it can only replay crises that have already occurred.

**Hypothetical factor shocks** specify a move in a small set of risk factors
(equity, rates, commodities, real estate) and propagate it through each asset's
estimated factor sensitivities. It asks: "what if rates rose 200bp tomorrow?"
Its strength is that it can express a scenario with no historical precedent;
its weakness is that the transmission relies on betas estimated in normal
times, which is exactly when they are least reliable. Both are reported
side by side, and neither is a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prisk.config import Config, load_config
from prisk.metrics import drawdown_series

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Historical episode replay
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """One portfolio's behaviour through one historical episode."""

    episode: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    total_return: float
    max_drawdown: float
    worst_day: float
    annualised_volatility: float
    note: str = ""


def replay_episode(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    start: str,
    end: str,
    name: str = "episode",
    note: str = "",
    rebalance: bool = False,
) -> Optional[EpisodeResult]:
    """
    Apply a fixed weight vector to the asset returns of a historical window.

    Parameters
    ----------
    rebalance : bool
        ``False`` (default) lets the weights drift with realised performance,
        which is what actually happens to a portfolio nobody trades during a
        crash. ``True`` rebalances daily back to target, which flatters the
        result by mechanically buying the fallers.
    """
    window = asset_returns.loc[
        (asset_returns.index >= pd.Timestamp(start))
        & (asset_returns.index <= pd.Timestamp(end))
    ].dropna(how="any")
    if window.empty:
        return None

    assets = [a for a in window.columns if a in weights.index]
    w = weights.reindex(assets).fillna(0.0)
    window = window[assets]

    if rebalance:
        path = window.mul(w, axis=1).sum(axis=1)
    else:
        current = w.copy()
        daily = []
        for _, row in window.iterrows():
            daily.append(float((current * row).sum()))
            grown = current * (1.0 + row)
            total = grown.sum()
            current = grown / total if total > 0 else current
        path = pd.Series(daily, index=window.index)

    return EpisodeResult(
        episode=name,
        start=window.index[0],
        end=window.index[-1],
        n_days=len(window),
        total_return=float((1.0 + path).prod() - 1.0),
        max_drawdown=float(drawdown_series(path).min()),
        worst_day=float(path.min()),
        annualised_volatility=float(path.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        note=note,
    )


def historical_stress_table(
    asset_returns: pd.DataFrame,
    portfolio_weights: Dict[str, pd.Series],
    cfg: Optional[Config] = None,
    rebalance: bool = False,
) -> pd.DataFrame:
    """Replay every configured episode for every portfolio."""
    cfg = cfg or load_config()
    rows: List[Dict[str, object]] = []
    for episode in cfg.stress["historical_episodes"]:
        for portfolio_name, weights in portfolio_weights.items():
            result = replay_episode(
                asset_returns,
                weights,
                episode["start"],
                episode["end"],
                name=episode["name"],
                note=episode.get("note", ""),
                rebalance=rebalance,
            )
            if result is None:
                continue
            rows.append(
                {
                    "scenario": result.episode,
                    "portfolio": portfolio_name,
                    "start": result.start.date(),
                    "end": result.end.date(),
                    "trading_days": result.n_days,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "worst_day": result.worst_day,
                    "annualised_volatility": result.annualised_volatility,
                    "note": result.note,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Factor sensitivities
# ---------------------------------------------------------------------------
def estimate_factor_betas(
    asset_returns: pd.DataFrame,
    cfg: Optional[Config] = None,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimate each asset's sensitivity to the stress factors.

    A multivariate OLS regression of each asset's daily return on the daily
    returns of the factor proxies:

    ``r_i,t = α_i + Σ_k β_i,k · f_k,t + ε_i,t``

    A multivariate rather than univariate regression matters here: SPY and VNQ
    are highly correlated, so univariate betas would double-count the equity
    shock. The factor proxies themselves regress to the identity by
    construction, which is the desired behaviour — a -20% equity shock means
    SPY falls 20%.
    """
    cfg = cfg or load_config()
    proxies = cfg.stress["factor_proxies"]
    factor_names = list(proxies.keys())
    factor_tickers = [proxies[f] for f in factor_names]

    available = [t for t in factor_tickers if t in asset_returns.columns]
    if len(available) < len(factor_tickers):  # pragma: no cover - config guard
        missing = set(factor_tickers) - set(available)
        raise ValueError(f"Factor proxies missing from the panel: {missing}")

    data = asset_returns.dropna(how="any")
    if window:
        data = data.iloc[-window:]

    factors = data[factor_tickers].copy()
    factors.columns = factor_names
    design = np.column_stack([np.ones(len(factors)), factors.values])

    rows = {}
    for asset in data.columns:
        y = data[asset].values
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        rows[asset] = dict(zip(["alpha"] + factor_names, coefficients))

    betas = pd.DataFrame(rows).T[factor_names]
    betas.index.name = "ticker"
    return betas


def apply_factor_scenario(
    weights: pd.Series,
    betas: pd.DataFrame,
    shocks: Dict[str, float],
) -> pd.Series:
    """
    Translate factor shocks into per-asset return shocks.

    ``shock_i = Σ_k β_i,k · shock_k``. Cash-like assets end up with betas near
    zero and are therefore left approximately unchanged, which is the point of
    holding them.
    """
    factor_names = [f for f in betas.columns if f in shocks]
    shock_vector = np.array([shocks[f] for f in factor_names])
    asset_shocks = betas[factor_names].values @ shock_vector
    return pd.Series(asset_shocks, index=betas.index).reindex(weights.index).fillna(0.0)


def hypothetical_stress_table(
    asset_returns: pd.DataFrame,
    portfolio_weights: Dict[str, pd.Series],
    cfg: Optional[Config] = None,
    cov: Optional[pd.DataFrame] = None,
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Run every configured hypothetical scenario against every portfolio.

    Two numbers are reported per scenario. ``portfolio_impact`` is the
    deterministic mark-to-market loss implied by propagating the factor shocks
    through the estimated betas. ``stressed_annualised_volatility`` answers a
    different question — what the portfolio's *ongoing* risk would look like in
    that regime — and is computed by forcing every pairwise correlation to the
    scenario's ``correlation_override`` while holding individual volatilities
    fixed. Scenarios without an override report the observed-correlation
    volatility, so the column is comparable across rows.
    """
    cfg = cfg or load_config()
    betas = estimate_factor_betas(asset_returns, cfg)
    if cov is None:
        cov = asset_returns.dropna(how="any").cov()

    rows: List[Dict[str, object]] = []
    for scenario in cfg.stress["hypothetical_scenarios"]:
        shocks = scenario["shocks"]
        override = scenario.get("correlation_override")
        for portfolio_name, weights in portfolio_weights.items():
            asset_shocks = apply_factor_scenario(weights, betas, shocks)
            aligned = weights.reindex(asset_shocks.index).fillna(0.0)
            contributions = aligned * asset_shocks

            assets = [a for a in cov.columns if a in aligned.index]
            w = aligned.reindex(assets).fillna(0.0).values
            scenario_cov = (
                stressed_covariance(cov.loc[assets, assets], override)
                if override is not None
                else cov.loc[assets, assets]
            )
            stressed_vol = float(
                np.sqrt(w @ scenario_cov.values @ w) * np.sqrt(TRADING_DAYS)
            )

            impact = float(contributions.sum())
            rows.append(
                {
                    "scenario": scenario["name"],
                    "portfolio": portfolio_name,
                    "portfolio_impact": impact,
                    "portfolio_impact_usd": impact * portfolio_value,
                    "stressed_annualised_volatility": stressed_vol,
                    "correlation_override": override,
                    "worst_asset": str(asset_shocks.idxmin()),
                    "worst_asset_shock": float(asset_shocks.min()),
                    "largest_loss_contributor": str(contributions.idxmin()),
                    "largest_loss_contribution": float(contributions.min()),
                    "description": scenario.get("description", ""),
                }
            )
    return pd.DataFrame(rows)


def scenario_asset_detail(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    scenario_name: str,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """Per-asset shock and loss contribution for one named scenario."""
    cfg = cfg or load_config()
    scenario = next(
        s for s in cfg.stress["hypothetical_scenarios"] if s["name"] == scenario_name
    )
    betas = estimate_factor_betas(asset_returns, cfg)
    shocks = apply_factor_scenario(weights, betas, scenario["shocks"])
    aligned = weights.reindex(shocks.index).fillna(0.0)
    detail = pd.DataFrame(
        {
            "weight": aligned,
            "asset_shock": shocks,
            "loss_contribution": aligned * shocks,
        }
    )
    detail["pct_of_total_impact"] = detail["loss_contribution"] / detail[
        "loss_contribution"
    ].sum()
    return detail.sort_values("loss_contribution")


# ---------------------------------------------------------------------------
# Correlation stress
# ---------------------------------------------------------------------------
def stressed_covariance(cov: pd.DataFrame, target_correlation: float) -> pd.DataFrame:
    """
    Rebuild a covariance matrix with every pairwise correlation forced to
    ``target_correlation``, holding individual volatilities fixed.

    This isolates the single most damaging feature of a crisis: diversification
    disappearing exactly when it is needed. The 2020 and 2022 episodes are both
    cases where correlations moved sharply towards one.
    """
    vols = np.sqrt(np.diag(cov.values))
    n = len(vols)
    correlation = np.full((n, n), float(target_correlation))
    np.fill_diagonal(correlation, 1.0)
    stressed = correlation * np.outer(vols, vols)
    return pd.DataFrame(stressed, index=cov.index, columns=cov.columns)


def correlation_stress_impact(
    weights: pd.Series,
    cov: pd.DataFrame,
    target_correlations: List[float] = (0.5, 0.75, 0.9, 0.95),
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Portfolio volatility as pairwise correlations are forced upwards."""
    assets = [a for a in cov.columns if a in weights.index]
    w = weights.reindex(assets).fillna(0.0).values
    base = cov.loc[assets, assets]
    base_vol = float(np.sqrt(w @ base.values @ w) * np.sqrt(trading_days))

    rows = [{"scenario": "Observed correlations", "annualised_volatility": base_vol,
             "vol_multiple": 1.0}]
    for rho in target_correlations:
        stressed = stressed_covariance(base, rho)
        vol = float(np.sqrt(w @ stressed.values @ w) * np.sqrt(trading_days))
        rows.append(
            {
                "scenario": f"All correlations = {rho:.2f}",
                "annualised_volatility": vol,
                "vol_multiple": vol / base_vol if base_vol else np.nan,
            }
        )
    return pd.DataFrame(rows)
