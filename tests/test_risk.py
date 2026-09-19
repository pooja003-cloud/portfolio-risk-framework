"""Tests for VaR, expected shortfall and Monte Carlo simulation.

Where a closed form exists, the test checks against it. Where one does not,
the test checks an invariant that must hold for any correct implementation:
ES is never below VaR, VaR rises with confidence, component contributions sum
to the total, and ES is sub-additive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from prisk import risk as rk


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------
def test_historical_var_is_the_empirical_quantile(normal_returns):
    expected = -np.quantile(normal_returns.values, 0.05)
    assert rk.historical_var(normal_returns, 0.95) == pytest.approx(expected)


def test_var_increases_with_confidence(normal_returns):
    assert rk.historical_var(normal_returns, 0.99) > rk.historical_var(
        normal_returns, 0.95
    )


def test_expected_shortfall_never_below_var(normal_returns):
    for confidence in [0.90, 0.95, 0.99]:
        var = rk.historical_var(normal_returns, confidence)
        es = rk.historical_expected_shortfall(normal_returns, confidence)
        assert es >= var


def test_overlapping_horizon_var_exceeds_one_day(normal_returns):
    one_day = rk.historical_var(normal_returns, 0.99, 1)
    ten_day = rk.historical_var_overlapping(normal_returns.iloc[:50_000], 0.99, 10)
    assert ten_day > one_day


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def test_parametric_normal_matches_closed_form(normal_returns):
    mu, sigma = normal_returns.mean(), normal_returns.std(ddof=1)
    expected = -(mu + stats.norm.ppf(0.01) * sigma)
    assert rk.parametric_var(normal_returns, 0.99, 1, "normal") == pytest.approx(
        expected, rel=1e-12
    )


def test_parametric_normal_agrees_with_historical_on_normal_data(normal_returns):
    """
    On genuinely normal data the two methods must agree. Any disagreement on
    real data is therefore evidence about the data, not about the code.
    """
    historical = rk.historical_var(normal_returns, 0.99)
    parametric = rk.parametric_var(normal_returns, 0.99, 1, "normal")
    assert historical == pytest.approx(parametric, rel=0.02)


def test_parametric_var_scales_with_square_root_of_time(normal_returns):
    """
    The volatility term scales with sqrt(h) but the drift term scales with h.
    For a positive-drift series the ratio must therefore sit just *below*
    sqrt(10): the expected gain over ten days offsets part of the loss.
    """
    one_day = rk.parametric_var(normal_returns, 0.99, 1, "normal")
    ten_day = rk.parametric_var(normal_returns, 0.99, 10, "normal")
    ratio = ten_day / one_day
    assert ratio == pytest.approx(np.sqrt(10), rel=0.05)
    assert ratio < np.sqrt(10)

    # With the drift removed, the scaling is exact.
    centred = normal_returns - normal_returns.mean()
    exact = rk.parametric_var(centred, 0.99, 10, "normal") / rk.parametric_var(
        centred, 0.99, 1, "normal"
    )
    assert exact == pytest.approx(np.sqrt(10), rel=1e-9)


def test_student_t_exceeds_normal_in_the_far_tail():
    """
    A fat-tailed sample must produce a larger 99% VaR under a Student-t fit
    than under a normal fit. (At 95% the ordering can legitimately reverse:
    the standardised t has thinner shoulders than the normal.)
    """
    rng = np.random.default_rng(5)
    fat = pd.Series(rng.standard_t(4, size=100_000) * 0.005)
    assert rk.parametric_var(fat, 0.99, 1, "student_t") > rk.parametric_var(
        fat, 0.99, 1, "normal"
    )


def test_parametric_es_matches_closed_form(normal_returns):
    mu, sigma = normal_returns.mean(), normal_returns.std(ddof=1)
    tail = 0.01
    expected = -(mu - sigma * stats.norm.pdf(stats.norm.ppf(tail)) / tail)
    assert rk.parametric_expected_shortfall(
        normal_returns, 0.99, 1, "normal"
    ) == pytest.approx(expected, rel=1e-12)


def test_cornish_fisher_equals_normal_for_gaussian_data(normal_returns):
    """With zero skew and zero excess kurtosis the correction must vanish."""
    normal = rk.parametric_var(normal_returns, 0.99, 1, "normal")
    modified = rk.cornish_fisher_var(normal_returns, 0.99, 1)
    assert modified == pytest.approx(normal, rel=0.02)


def test_cornish_fisher_widens_for_negatively_skewed_data():
    rng = np.random.default_rng(9)
    skewed = pd.Series(-np.abs(rng.normal(0, 0.01, 100_000)) ** 1.5 * 10 + 0.001)
    assert rk.cornish_fisher_var(skewed, 0.99) > rk.parametric_var(
        skewed, 0.99, 1, "normal"
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_monte_carlo_normal_converges_to_closed_form(synthetic_returns, cfg):
    """
    Simulating from a multivariate normal with the sample covariance must
    reproduce the analytical normal VaR of the portfolio.
    """
    holdings = cfg.holdings
    panel = synthetic_returns[holdings]
    weights = pd.Series(1.0 / len(holdings), index=holdings)

    result = rk.monte_carlo_var(
        panel, weights, confidences=[0.99], horizon=1,
        n_simulations=400_000, distribution="normal", seed=1,
    )
    portfolio = panel @ weights
    analytical = rk.parametric_var(portfolio, 0.99, 1, "normal")
    assert result.var[0.99] == pytest.approx(analytical, rel=0.03)


def test_monte_carlo_is_reproducible(synthetic_returns, cfg):
    holdings = cfg.holdings
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    kwargs = dict(
        confidences=[0.99], horizon=1, n_simulations=20_000,
        distribution="student_t", seed=42,
    )
    first = rk.monte_carlo_var(synthetic_returns[holdings], weights, **kwargs)
    second = rk.monte_carlo_var(synthetic_returns[holdings], weights, **kwargs)
    assert first.var[0.99] == pytest.approx(second.var[0.99], rel=1e-12)


def test_monte_carlo_student_t_is_heavier_than_normal(synthetic_returns, cfg):
    holdings = cfg.holdings
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    shared = dict(confidences=[0.995], horizon=1, n_simulations=200_000, seed=3)

    normal = rk.monte_carlo_var(
        synthetic_returns[holdings], weights, distribution="normal", **shared
    )
    student = rk.monte_carlo_var(
        synthetic_returns[holdings], weights, distribution="student_t", dof=4,
        **shared,
    )
    assert student.var[0.995] > normal.var[0.995]


def test_monte_carlo_preserves_portfolio_volatility(synthetic_returns, cfg):
    """The simulated distribution must have the covariance it was given."""
    holdings = cfg.holdings
    panel = synthetic_returns[holdings]
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    cov = panel.cov()

    result = rk.monte_carlo_var(
        panel, weights, confidences=[0.99], horizon=1,
        n_simulations=300_000, distribution="normal", seed=2, cov=cov,
    )
    target = float(np.sqrt(weights.values @ cov.values @ weights.values))
    assert result.simulated_returns.std() == pytest.approx(target, rel=0.03)


def test_monte_carlo_horizon_scaling(synthetic_returns, cfg):
    holdings = cfg.holdings
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    shared = dict(
        confidences=[0.99], n_simulations=200_000, distribution="normal", seed=4
    )
    one = rk.monte_carlo_var(synthetic_returns[holdings], weights, horizon=1, **shared)
    ten = rk.monte_carlo_var(synthetic_returns[holdings], weights, horizon=10, **shared)
    assert ten.var[0.99] / one.var[0.99] == pytest.approx(np.sqrt(10), rel=0.08)


# ---------------------------------------------------------------------------
# Component VaR and coherence
# ---------------------------------------------------------------------------
def test_component_var_sums_to_total(synthetic_returns, cfg):
    """Euler's theorem: the decomposition must be exact, not approximate."""
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pd.Series(1.0 / len(holdings), index=holdings)

    table = rk.component_var(weights, cov, 0.99, 1)
    analytical = -stats.norm.ppf(0.01) * float(
        np.sqrt(weights.values @ cov.values @ weights.values)
    )
    assert table["component_var"].sum() == pytest.approx(analytical, rel=1e-10)
    assert table["pct_var_contribution"].sum() == pytest.approx(1.0, rel=1e-10)


def test_expected_shortfall_is_subadditive(synthetic_returns, cfg):
    """
    ES is a coherent risk measure, so diversification can never increase it:
    ES(a + b) <= ES(a) + ES(b). VaR carries no such guarantee, which is the
    main theoretical argument for reporting ES alongside it.
    """
    a = synthetic_returns[cfg.holdings[0]]
    b = synthetic_returns[cfg.holdings[9]]
    combined = 0.5 * a + 0.5 * b

    es_combined = rk.historical_expected_shortfall(combined, 0.95)
    es_parts = 0.5 * rk.historical_expected_shortfall(
        a, 0.95
    ) + 0.5 * rk.historical_expected_shortfall(b, 0.95)
    assert es_combined <= es_parts + 1e-12


def test_var_comparison_table_shape(synthetic_returns, cfg):
    holdings = cfg.holdings
    weights = pd.Series(1.0 / len(holdings), index=holdings)
    portfolio = synthetic_returns[holdings] @ weights

    table = rk.var_comparison(
        portfolio, synthetic_returns[holdings], weights,
        confidences=[0.95, 0.99], horizons=[1, 10],
        mc_kwargs={"n_simulations": 20_000, "seed": 1},
        portfolio_value=1_000_000,
    )
    assert len(table) == 4
    for column in [
        "historical_var", "parametric_normal_var", "parametric_t_var",
        "cornish_fisher_var", "monte_carlo_var", "historical_es",
    ]:
        assert column in table.columns
        assert table[column].notna().all()
    assert (table["historical_var_usd"] == table["historical_var"] * 1e6).all()
