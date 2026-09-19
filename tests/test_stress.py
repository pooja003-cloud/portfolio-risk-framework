"""Tests for the stress-testing layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import stress as st


# ---------------------------------------------------------------------------
# Factor betas
# ---------------------------------------------------------------------------
def test_factor_proxies_have_unit_beta_to_themselves(synthetic_returns, cfg):
    """
    Each factor proxy regressed on the factor set must load 1.0 on its own
    factor and 0.0 on the others. This is what makes a "-20% equity" shock
    mean exactly what it says.
    """
    betas = st.estimate_factor_betas(synthetic_returns, cfg)
    for factor, ticker in cfg.stress["factor_proxies"].items():
        assert betas.loc[ticker, factor] == pytest.approx(1.0, abs=1e-8)
        others = [f for f in betas.columns if f != factor]
        assert np.allclose(betas.loc[ticker, others].values, 0.0, atol=1e-8)


def test_cash_has_near_zero_factor_betas(synthetic_returns, cfg):
    betas = st.estimate_factor_betas(synthetic_returns, cfg)
    assert betas.loc[cfg.riskfree_ticker].abs().max() < 0.10


def test_equities_load_positively_on_the_equity_factor(synthetic_returns, cfg):
    betas = st.estimate_factor_betas(synthetic_returns, cfg)
    equities = [
        t for t, sleeve in cfg.sleeve_map().items() if sleeve == "equity"
    ]
    assert (betas.loc[equities, "equity"] > 0.3).all()


def test_factor_scenario_is_linear(synthetic_returns, cfg):
    """Doubling every shock must double every asset's implied move."""
    betas = st.estimate_factor_betas(synthetic_returns, cfg)
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    base = {"equity": -0.10, "duration": -0.05, "commodity": -0.08,
            "real_estate": -0.12}
    doubled = {k: 2 * v for k, v in base.items()}

    single = st.apply_factor_scenario(weights, betas, base)
    double = st.apply_factor_scenario(weights, betas, doubled)
    assert np.allclose(double.values, 2 * single.values, atol=1e-12)


def test_zero_shock_produces_zero_impact(synthetic_returns, cfg):
    betas = st.estimate_factor_betas(synthetic_returns, cfg)
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    shocks = {factor: 0.0 for factor in betas.columns}
    assert np.allclose(
        st.apply_factor_scenario(weights, betas, shocks).values, 0.0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Episode replay
# ---------------------------------------------------------------------------
def test_replay_episode_matches_manual_compounding(synthetic_returns, cfg):
    """With daily rebalancing the replay is a plain weighted compounding."""
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    start, end = "2020-01-02", "2020-06-30"
    result = st.replay_episode(
        synthetic_returns[cfg.holdings], weights, start, end, rebalance=True
    )
    window = synthetic_returns[cfg.holdings].loc[start:end]
    expected = float((1.0 + (window @ weights)).prod() - 1.0)
    assert result.total_return == pytest.approx(expected, rel=1e-9)


def test_replay_episode_drift_differs_from_rebalanced(synthetic_returns, cfg):
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    drifted = st.replay_episode(
        synthetic_returns[cfg.holdings], weights, "2018-01-01", "2021-12-31",
        rebalance=False,
    )
    rebalanced = st.replay_episode(
        synthetic_returns[cfg.holdings], weights, "2018-01-01", "2021-12-31",
        rebalance=True,
    )
    assert drifted.total_return != pytest.approx(rebalanced.total_return, rel=1e-6)


def test_replay_episode_returns_none_outside_the_sample(synthetic_returns, cfg):
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    assert st.replay_episode(
        synthetic_returns[cfg.holdings], weights, "1990-01-01", "1990-12-31"
    ) is None


def test_episode_drawdown_is_at_least_as_deep_as_the_total_loss(
    synthetic_returns, cfg
):
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    result = st.replay_episode(
        synthetic_returns[cfg.holdings], weights, "2015-01-01", "2016-12-31"
    )
    assert result.max_drawdown <= min(result.total_return, 0.0) + 1e-12


# ---------------------------------------------------------------------------
# Correlation stress
# ---------------------------------------------------------------------------
def test_stressed_covariance_hits_the_target_correlation(synthetic_returns, cfg):
    cov = synthetic_returns[cfg.holdings].cov()
    stressed = st.stressed_covariance(cov, 0.9)
    correlation = stressed / np.outer(
        np.sqrt(np.diag(stressed)), np.sqrt(np.diag(stressed))
    )
    off_diagonal = correlation.values[~np.eye(len(cov), dtype=bool)]
    assert np.allclose(off_diagonal, 0.9, atol=1e-10)


def test_stressed_covariance_preserves_individual_volatilities(
    synthetic_returns, cfg
):
    cov = synthetic_returns[cfg.holdings].cov()
    stressed = st.stressed_covariance(cov, 0.75)
    assert np.allclose(np.diag(cov.values), np.diag(stressed.values), atol=1e-14)


def test_correlation_stress_increases_volatility_monotonically(
    synthetic_returns, cfg
):
    """
    Portfolio volatility must rise monotonically in the forced correlation.

    Note that the *first* forced level is not required to exceed the observed
    one: if the sample's average pairwise correlation already sits above 0.5,
    forcing 0.5 is a de-stress, not a stress. Only the forced levels are
    compared with each other, and the extreme level is checked against the
    observed baseline.
    """
    cov = synthetic_returns[cfg.holdings].cov()
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    frame = st.correlation_stress_impact(weights, cov, [0.5, 0.75, 0.9, 0.99])

    forced = frame["annualised_volatility"].values[1:]
    assert np.all(np.diff(forced) > 0)
    assert frame["vol_multiple"].iloc[0] == pytest.approx(1.0)
    assert forced[-1] >= frame["annualised_volatility"].iloc[0]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def test_hypothetical_table_covers_every_scenario(synthetic_returns, cfg):
    weights = {"test": pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)}
    table = st.hypothetical_stress_table(synthetic_returns[cfg.holdings], weights, cfg)
    assert len(table) == len(cfg.stress["hypothetical_scenarios"])
    assert table["portfolio_impact"].notna().all()
    # Every configured scenario is a loss scenario for a long-only portfolio.
    assert (table["portfolio_impact"] < 0).all()


def test_severe_scenario_is_the_worst(synthetic_returns, cfg):
    weights = {"test": pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)}
    table = st.hypothetical_stress_table(synthetic_returns[cfg.holdings], weights, cfg)
    worst = table.loc[table["portfolio_impact"].idxmin(), "scenario"]
    assert worst == "Severe Combined Stress"


def test_scenario_detail_contributions_sum_to_the_total(synthetic_returns, cfg):
    weights = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)
    detail = st.scenario_asset_detail(
        weights, synthetic_returns[cfg.holdings], "Equity Bear Market", cfg
    )
    table = st.hypothetical_stress_table(
        synthetic_returns[cfg.holdings], {"p": weights}, cfg
    )
    total = float(
        table.loc[table["scenario"] == "Equity Bear Market", "portfolio_impact"].iloc[0]
    )
    assert detail["loss_contribution"].sum() == pytest.approx(total, rel=1e-10)
    assert detail["pct_of_total_impact"].sum() == pytest.approx(1.0, rel=1e-10)


def test_historical_table_uses_configured_episodes(synthetic_returns, cfg):
    weights = {"test": pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)}
    table = st.historical_stress_table(synthetic_returns[cfg.holdings], weights, cfg)
    # The synthetic sample starts in 2014, so pre-2014 episodes drop out.
    assert len(table) > 0
    assert set(table["scenario"]).issubset(
        {episode["name"] for episode in cfg.stress["historical_episodes"]}
    )
