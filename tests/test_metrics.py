"""Tests for performance and risk metrics.

Each test pins a property that must hold by definition, so a refactor that
silently changes a convention (arithmetic versus geometric, ddof, the
annualisation factor) fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import metrics as mx


def test_annualised_return_is_geometric():
    """A series that doubles over exactly one year must return 100%."""
    daily = (2.0 ** (1 / 252)) - 1.0
    series = pd.Series([daily] * 252)
    assert mx.annualised_return(series) == pytest.approx(1.0, rel=1e-9)


def test_annualised_return_handles_multi_year():
    daily = (2.0 ** (1 / 252)) - 1.0
    series = pd.Series([daily] * 504)  # doubles twice over two years
    assert mx.annualised_return(series) == pytest.approx(1.0, rel=1e-9)
    assert mx.cumulative_return(series) == pytest.approx(3.0, rel=1e-9)


def test_volatility_annualisation():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 0.01, 100_000))
    assert mx.annualised_volatility(series) == pytest.approx(
        0.01 * np.sqrt(252), rel=0.02
    )


def test_max_drawdown_known_path():
    """100 -> 120 -> 60 -> 90: peak 120, trough 60, drawdown -50%."""
    prices = pd.Series([100.0, 120.0, 60.0, 90.0])
    returns = prices.pct_change().dropna()
    result = mx.max_drawdown(returns)
    assert result["max_drawdown"] == pytest.approx(-0.5)
    # Never recovers to the prior peak within the sample.
    assert result["recovery_date"] is None


def test_max_drawdown_records_recovery():
    prices = pd.Series([100.0, 80.0, 100.0, 110.0])
    returns = prices.pct_change().dropna()
    result = mx.max_drawdown(returns)
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["recovery_date"] is not None


def test_drawdown_is_never_positive():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.0005, 0.01, 5000))
    assert mx.drawdown_series(series).max() <= 1e-12


def test_beta_of_market_against_itself_is_one(synthetic_returns, cfg):
    market = synthetic_returns[cfg.market_ticker]
    assert mx.beta(market, market) == pytest.approx(1.0, rel=1e-9)


def test_beta_scales_linearly(synthetic_returns, cfg):
    market = synthetic_returns[cfg.market_ticker]
    assert mx.beta(2.0 * market, market) == pytest.approx(2.0, rel=1e-9)


def test_sharpe_zero_when_excess_return_is_zero():
    series = pd.Series([0.001] * 1000)
    risk_free = pd.Series([0.001] * 1000)
    assert mx.sharpe_ratio(series, risk_free) == pytest.approx(0.0) or np.isnan(
        mx.sharpe_ratio(series, risk_free)
    )


def test_sortino_exceeds_sharpe_for_right_skewed_series():
    """
    Downside deviation ignores upside dispersion, so a series whose large
    moves are mostly positive must score better on Sortino than on Sharpe.
    The series still contains losses — with none at all the downside deviation
    is zero and Sortino is undefined.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0.0004, 0.004, 20_000)
    jumps = (rng.random(20_000) < 0.02) * rng.exponential(0.03, 20_000)
    series = pd.Series(base + jumps)  # symmetric core, one-sided upside tail
    assert (series < 0).any(), "fixture must contain losses"
    assert mx.sortino_ratio(series) > mx.sharpe_ratio(series)


def test_drawdown_captures_a_loss_on_the_first_day():
    """
    Regression test. A sample that opens with a loss must record it: the
    running peak starts at the initial capital, not at the first day's closing
    value.
    """
    prices = pd.Series([100.0, 80.0, 100.0, 110.0])
    returns = prices.pct_change().dropna()
    assert mx.drawdown_series(returns).min() == pytest.approx(-0.2)


def test_capture_ratios_are_one_against_self(synthetic_returns, cfg):
    market = synthetic_returns[cfg.market_ticker]
    up, down = mx.capture_ratios(market, market)
    assert up == pytest.approx(1.0, rel=1e-9)
    assert down == pytest.approx(1.0, rel=1e-9)


def test_capture_ratios_scale(synthetic_returns, cfg):
    """A half-weight version of the benchmark captures half of both legs."""
    market = synthetic_returns[cfg.market_ticker]
    up, down = mx.capture_ratios(0.5 * market, market)
    assert up == pytest.approx(0.5, rel=1e-9)
    assert down == pytest.approx(0.5, rel=1e-9)


def test_tracking_error_is_zero_against_self(synthetic_returns, cfg):
    market = synthetic_returns[cfg.market_ticker]
    assert mx.tracking_error(market, market) == pytest.approx(0.0, abs=1e-12)


def test_risk_contributions_sum_to_portfolio_volatility(synthetic_returns, cfg):
    """
    Euler decomposition: component contributions must sum exactly to the
    portfolio's volatility, and the percentage column must sum to one.
    """
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pd.Series(1.0 / len(holdings), index=holdings)

    table = mx.risk_contributions(weights, cov)
    portfolio_vol = float(
        np.sqrt(weights.values @ cov.values @ weights.values) * np.sqrt(252)
    )

    assert table["component_contribution"].sum() == pytest.approx(
        portfolio_vol, rel=1e-10
    )
    assert table["pct_risk_contribution"].sum() == pytest.approx(1.0, rel=1e-10)


def test_risk_contribution_of_single_asset_portfolio(synthetic_returns, cfg):
    """A portfolio holding one asset gets 100% of its risk from that asset."""
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pd.Series(0.0, index=holdings)
    weights[holdings[0]] = 1.0

    table = mx.risk_contributions(weights, cov)
    assert table.loc[holdings[0], "pct_risk_contribution"] == pytest.approx(1.0)


def test_group_risk_contributions_preserve_total(synthetic_returns, cfg):
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    table = mx.risk_contributions(weights, cov)

    grouped = mx.group_risk_contributions(table, cfg.sleeve_map())
    assert grouped["pct_risk_contribution"].sum() == pytest.approx(1.0, rel=1e-10)
    assert grouped["weight"].sum() == pytest.approx(1.0, rel=1e-10)


def test_performance_summary_contains_required_fields(synthetic_returns, cfg):
    market = synthetic_returns[cfg.market_ticker]
    summary = mx.performance_summary(
        synthetic_returns[cfg.holdings].mean(axis=1), market=market, benchmark=market
    )
    for field in [
        "annualised_return", "annualised_volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "beta", "skewness", "excess_kurtosis",
    ]:
        assert field in summary
        assert not pd.isna(summary[field])


def test_risk_contribution_order_is_deterministic(synthetic_returns, cfg):
    """
    Regression test. A portfolio holding only two of the fifteen instruments
    leaves thirteen rows tied at exactly zero risk contribution. Without an
    explicit tiebreak their order depends on the platform's sort
    implementation, which made the committed outputs differ between macOS and
    Linux for no substantive reason.
    """
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pd.Series(0.0, index=holdings)
    weights[["SPY", "AGG"]] = [0.6, 0.4]

    table = mx.risk_contributions(weights, cov)
    # "Zero" contributions come out of the linear algebra as +/-1e-18, so the
    # tolerance here is deliberately loose: the point is that they are
    # conceptually tied and must therefore order by ticker.
    ties = table[table["pct_risk_contribution"].abs() < 1e-10]
    assert len(ties) > 5, "fixture should produce tied rows"
    assert list(ties.index) == sorted(ties.index), "tied rows must sort by ticker"

    # And the whole table is stable across repeated calls on shuffled input.
    shuffled = weights.sample(frac=1.0, random_state=1)
    again = mx.risk_contributions(shuffled, cov)
    assert list(again.index) == list(table.index)
