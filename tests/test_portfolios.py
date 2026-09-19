"""Tests for portfolio construction and the rebalancing engine.

The most important test in this file is ``test_no_look_ahead``: a backtest that
peeks at future data produces flattering results that cannot be achieved in
practice, and it is the single easiest mistake to make in this kind of project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import portfolios as pf


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def test_minimum_variance_weights_sum_to_one(synthetic_returns, cfg):
    cov = synthetic_returns[cfg.holdings].cov()
    weights = pf.minimum_variance_weights(cov, max_weight=0.25)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_minimum_variance_respects_box_constraints(synthetic_returns, cfg):
    cov = synthetic_returns[cfg.holdings].cov()
    weights = pf.minimum_variance_weights(cov, min_weight=0.0, max_weight=0.20)
    assert weights.min() >= -1e-8
    assert weights.max() <= 0.20 + 1e-6


def test_minimum_variance_beats_equal_weight_variance(synthetic_returns, cfg):
    """
    By construction the minimum-variance portfolio cannot have a higher
    in-sample variance than any other portfolio in the feasible set.
    """
    cov = synthetic_returns[cfg.holdings].cov()
    mv = pf.minimum_variance_weights(cov, max_weight=1.0)
    ew = pd.Series(1.0 / len(cfg.holdings), index=cfg.holdings)

    mv_var = float(mv.values @ cov.values @ mv.values)
    ew_var = float(ew.values @ cov.values @ ew.values)
    assert mv_var <= ew_var + 1e-12


def test_cvxpy_and_scipy_solvers_agree(synthetic_returns, cfg):
    """The QP is convex, so both solvers must reach the same global optimum."""
    pytest.importorskip("cvxpy")
    cov = synthetic_returns[cfg.holdings].cov()
    with_cvxpy = pf.minimum_variance_weights(cov, max_weight=0.25, prefer_cvxpy=True)
    with_scipy = pf.minimum_variance_weights(cov, max_weight=0.25, prefer_cvxpy=False)

    var_cvxpy = float(with_cvxpy.values @ cov.values @ with_cvxpy.values)
    var_scipy = float(with_scipy.values @ cov.values @ with_scipy.values)
    assert var_cvxpy == pytest.approx(var_scipy, rel=1e-4)


def test_risk_parity_equalises_risk_contributions(synthetic_returns, cfg):
    """Every asset should carry 1/N of the portfolio variance."""
    holdings = cfg.holdings
    cov = synthetic_returns[holdings].cov()
    weights = pf.risk_parity_weights(cov)

    portfolio_var = float(weights.values @ cov.values @ weights.values)
    contributions = weights.values * (cov.values @ weights.values) / portfolio_var
    assert contributions.max() - contributions.min() < 5e-3
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_risk_parity_overweights_the_calm_asset(synthetic_returns, cfg):
    """Cash is the least volatile holding, so it must get the largest weight."""
    cov = synthetic_returns[cfg.holdings].cov()
    weights = pf.risk_parity_weights(cov)
    assert weights.idxmax() == cfg.riskfree_ticker


# ---------------------------------------------------------------------------
# Weight rules
# ---------------------------------------------------------------------------
def test_equal_weight_rule(synthetic_returns, cfg):
    weights = pf.equal_weight_rule(synthetic_returns[cfg.holdings], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.nunique() == 1


def test_volatility_target_rule_hits_the_target(synthetic_returns, cfg):
    """
    When the scaling constraint does not bind, forecast volatility must equal
    the target exactly.
    """
    holdings = cfg.holdings
    window = synthetic_returns[holdings]
    weights = pf.volatility_target_rule(window, cfg)

    cash = cfg.riskfree_ticker
    growth = [c for c in holdings if c != cash]
    cov = pf.estimate_covariance(window[growth])
    base = pf.risk_parity_weights(cov)
    forecast = float(np.sqrt(base.values @ cov.values @ base.values) * np.sqrt(252))
    target = float(cfg.portfolios["volatility_target"]["target_volatility"])

    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    if forecast > target:  # the scale is interior, so the target is achievable
        achieved = float(
            np.sqrt(
                weights[growth].values @ cov.values @ weights[growth].values
            ) * np.sqrt(252)
        )
        assert achieved == pytest.approx(target, rel=1e-6)


def test_volatility_target_never_levers(synthetic_returns, cfg):
    weights = pf.volatility_target_rule(synthetic_returns[cfg.holdings], cfg)
    assert weights.min() >= -1e-9
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_benchmark_rule_matches_configuration(synthetic_returns, cfg):
    weights = pf.benchmark_rule(synthetic_returns[cfg.holdings], cfg)
    for ticker, target in cfg.portfolios["benchmark"]["weights"].items():
        assert weights[ticker] == pytest.approx(target)
    assert weights.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def test_rebalance_dates_are_quarter_ends(synthetic_returns):
    dates = pf.rebalance_dates(synthetic_returns.index, "Q")
    assert len(dates) > 8
    assert all(d in synthetic_returns.index for d in dates)
    months = {d.month for d in dates}
    assert months.issubset({3, 6, 9, 12})


def test_backtest_weights_always_sum_to_one(synthetic_returns, cfg):
    result = pf.run_backtest(
        synthetic_returns[cfg.holdings], pf.equal_weight_rule, cfg, "ew"
    )
    row_sums = result.weights_daily.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-8)


def test_backtest_costs_reduce_returns(synthetic_returns, cfg):
    result = pf.run_backtest(
        synthetic_returns[cfg.holdings], pf.minimum_variance_rule, cfg, "mv"
    )
    assert (result.returns_net <= result.returns_gross + 1e-15).all()
    assert result.cost_drag.sum() > 0
    assert result.turnover.min() >= 0


def test_equal_weight_return_matches_manual_calculation(synthetic_returns, cfg):
    """
    On the rebalance day itself the weights are exactly 1/N, so the portfolio
    return must equal the cross-sectional mean of the asset returns.
    """
    panel = synthetic_returns[cfg.holdings]
    result = pf.run_backtest(panel, pf.equal_weight_rule, cfg, "ew")
    rebalance_day = result.weights_target.index[1]
    expected = float(panel.loc[rebalance_day].mean())
    assert result.returns_gross.loc[rebalance_day] == pytest.approx(expected, rel=1e-9)


def test_no_look_ahead(synthetic_returns, cfg):
    """
    Corrupting the *future* must not change *past* weights.

    The tail of the return panel is replaced with extreme values and the
    backtest re-run. Every target weight set before the corruption begins must
    be bit-for-bit identical; if the optimiser were fitted on the full sample,
    or the estimation window were centred rather than trailing, this fails.
    """
    panel = synthetic_returns[cfg.holdings].copy()
    baseline = pf.run_backtest(panel, pf.minimum_variance_rule, cfg, "mv")

    corrupted = panel.copy()
    cutoff = corrupted.index[-300]
    corrupted.loc[cutoff:] = corrupted.loc[cutoff:] * 10.0
    perturbed = pf.run_backtest(corrupted, pf.minimum_variance_rule, cfg, "mv")

    unaffected = baseline.weights_target.index[baseline.weights_target.index < cutoff]
    pd.testing.assert_frame_equal(
        baseline.weights_target.loc[unaffected],
        perturbed.weights_target.loc[unaffected],
    )


def test_build_all_portfolios_returns_every_strategy(synthetic_returns, cfg):
    results = pf.build_all_portfolios(synthetic_returns, cfg)
    assert set(results) == {
        "equal_weight", "minimum_variance", "volatility_target", "benchmark"
    }
    for result in results.values():
        assert len(result.returns) > 1000
        assert result.equity_curve.iloc[-1] > 0


def test_minimum_variance_is_least_volatile_realised(synthetic_returns, cfg):
    """
    Out of sample the ranking is not guaranteed, but over a twelve-year sample
    the minimum-variance portfolio should still be the least volatile of the
    three research portfolios. If it is not, the estimation window or the
    shrinkage is broken.
    """
    results = pf.build_all_portfolios(synthetic_returns, cfg)
    vols = {
        name: results[name].returns.std()
        for name in ["equal_weight", "minimum_variance", "volatility_target"]
    }
    assert min(vols, key=vols.get) == "minimum_variance"


def test_estimate_covariance_is_positive_semidefinite(synthetic_returns, cfg):
    for method in ["sample", "ledoit_wolf"]:
        cov = pf.estimate_covariance(synthetic_returns[cfg.holdings], method=method)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() > -1e-12


def _correlation(cov: pd.DataFrame) -> np.ndarray:
    diagonal = np.sqrt(np.diag(cov.values))
    return cov.values / np.outer(diagonal, diagonal)


def test_ledoit_wolf_shrinks_the_correlation_spectrum(synthetic_returns, cfg):
    """
    Shrinkage must compress the eigenvalue spread of the *correlation* matrix —
    that is where the estimation noise lives, and it is what the estimator
    targets. The covariance matrix's own conditioning is dominated by the
    four-orders-of-magnitude spread in variances between cash and equities,
    which shrinkage deliberately leaves alone.
    """
    panel = synthetic_returns[cfg.holdings].iloc[-300:]  # short window: noisy
    sample = np.linalg.eigvalsh(_correlation(pf.estimate_covariance(panel, "sample")))
    shrunk = np.linalg.eigvalsh(
        _correlation(pf.estimate_covariance(panel, "ledoit_wolf"))
    )
    assert (shrunk.max() / shrunk.min()) < (sample.max() / sample.min())
    assert shrunk.min() > sample.min()


def test_shrinkage_preserves_individual_volatilities(synthetic_returns, cfg):
    """
    Regression test. Shrinking the covariance matrix directly pulls every
    variance towards the average variance, which inflated the cash proxy's
    annualised volatility from 0.2% to roughly 4.8% and fed that number to the
    optimiser and to every risk table. Shrinking the correlation matrix instead
    must leave each asset's own variance untouched.
    """
    panel = synthetic_returns[cfg.holdings].iloc[-300:]
    shrunk = pf.estimate_covariance(panel, "ledoit_wolf")
    realised = panel.std(ddof=1)

    implied = pd.Series(np.sqrt(np.diag(shrunk.values)), index=shrunk.columns)
    pd.testing.assert_series_equal(
        implied, realised[implied.index], check_names=False, rtol=1e-10
    )


def test_shrinkage_pulls_correlations_towards_zero(synthetic_returns, cfg):
    panel = synthetic_returns[cfg.holdings].iloc[-300:]
    mask = ~np.eye(len(cfg.holdings), dtype=bool)
    sample = np.abs(_correlation(pf.estimate_covariance(panel, "sample"))[mask]).mean()
    shrunk = np.abs(
        _correlation(pf.estimate_covariance(panel, "ledoit_wolf"))[mask]
    ).mean()
    assert shrunk < sample
