"""Tests for configuration and the data layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import data as dataio
from prisk.config import load_config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_config_loads_and_exposes_the_universe(cfg):
    assert len(cfg.tickers) == 15
    assert len(set(cfg.tickers)) == len(cfg.tickers), "duplicate ticker in universe"
    assert cfg.market_ticker == "SPY"
    assert cfg.riskfree_ticker == "BIL"


def test_universe_covers_every_required_asset_class(cfg):
    sleeves = set(cfg.sleeve_map().values())
    for required in ["equity", "fixed_income", "commodity", "real_estate", "cash"]:
        assert required in sleeves


def test_every_holding_has_a_sleeve_and_sector(cfg):
    sleeves, sectors = cfg.sleeve_map(), cfg.sector_map()
    for ticker in cfg.holdings:
        assert sleeves.get(ticker)
        assert sectors.get(ticker)


def test_factor_proxies_are_in_the_universe(cfg):
    for ticker in cfg.stress["factor_proxies"].values():
        assert ticker in cfg.tickers


def test_benchmark_weights_sum_to_one(cfg):
    weights = cfg.portfolios["benchmark"]["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    for ticker in weights:
        assert ticker in cfg.tickers


def test_stress_scenarios_are_well_formed(cfg):
    factors = set(cfg.stress["factor_proxies"])
    for scenario in cfg.stress["hypothetical_scenarios"]:
        assert scenario["name"] and scenario["description"]
        assert set(scenario["shocks"]).issubset(factors)


def test_historical_episodes_have_valid_date_ranges(cfg):
    for episode in cfg.stress["historical_episodes"]:
        start, end = pd.Timestamp(episode["start"]), pd.Timestamp(episode["end"])
        assert start < end
        assert start >= pd.Timestamp(cfg.data["start_date"])


def test_minimum_variance_box_constraint_is_feasible(cfg):
    """max_weight * n_assets must be at least 1 or the problem is infeasible."""
    max_weight = cfg.portfolios["minimum_variance"]["max_weight"]
    assert max_weight * len(cfg.holdings) >= 1.0


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def test_simple_returns_match_manual_calculation():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    returns = dataio.compute_returns(prices, "simple")
    assert returns["A"].iloc[0] == pytest.approx(0.10)
    assert returns["A"].iloc[1] == pytest.approx(-0.10)


def test_log_returns_are_additive():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    log_returns = dataio.compute_returns(prices, "log")
    total = float(np.exp(log_returns["A"].sum()) - 1.0)
    assert total == pytest.approx(0.21, rel=1e-9)


def test_compute_returns_rejects_unknown_kind():
    with pytest.raises(ValueError):
        dataio.compute_returns(pd.DataFrame({"A": [1.0, 2.0]}), "geometric")


def test_risk_free_falls_back_to_zero_when_absent(synthetic_returns):
    stripped = synthetic_returns.drop(columns=["BIL"])
    rf = dataio.risk_free_daily(stripped)
    assert (rf == 0).all()


def test_risk_free_uses_the_cash_proxy(synthetic_returns, cfg):
    rf = dataio.risk_free_daily(synthetic_returns, cfg)
    pd.testing.assert_series_equal(
        rf, synthetic_returns[cfg.riskfree_ticker].rename("rf")
    )


# ---------------------------------------------------------------------------
# The shipped panel
# ---------------------------------------------------------------------------
def test_shipped_panel_is_clean(cfg):
    """
    Guards the committed price snapshot: complete, sorted, in-window, and
    covering the whole universe.
    """
    panel = dataio.load_panel(cfg)
    assert list(panel.columns) == cfg.tickers
    assert panel.isna().sum().sum() == 0
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique
    assert (panel > 0).all().all()
    assert panel.index[0] >= pd.Timestamp(cfg.data["start_date"])
    assert panel.index[-1] <= pd.Timestamp(cfg.data["end_date"])
    assert len(panel) > 4_000, "expected roughly 19 years of daily data"


def test_shipped_panel_has_no_implausible_daily_moves(cfg):
    """
    A 60% one-day move in a mega-cap or a broad ETF is almost always an
    unadjusted split rather than a real event. This is the tripwire for a
    corrupted refresh of the price data.
    """
    returns = dataio.compute_returns(dataio.load_panel(cfg))
    assert returns.abs().max().max() < 0.60


def test_cash_proxy_behaves_like_cash(cfg):
    returns = dataio.compute_returns(dataio.load_panel(cfg))
    cash = returns[cfg.riskfree_ticker]
    assert cash.std() * np.sqrt(252) < 0.02, "cash proxy is too volatile"
    assert cash.mean() > 0
