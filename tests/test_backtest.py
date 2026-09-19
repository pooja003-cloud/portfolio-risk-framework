"""Tests for VaR backtesting and the coverage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import backtest as bt


# ---------------------------------------------------------------------------
# Kupiec
# ---------------------------------------------------------------------------
def test_kupiec_accepts_a_correctly_calibrated_model():
    """Exactly the expected number of exceptions gives LR = 0 and p = 1."""
    result = bt.kupiec_pof_test(n_exceptions=10, n_observations=1000, confidence=0.99)
    assert result["lr_statistic"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert result["reject_at_5pct"] is False


def test_kupiec_rejects_a_model_that_breaches_too_often():
    result = bt.kupiec_pof_test(n_exceptions=40, n_observations=1000, confidence=0.99)
    assert result["observed_rate"] == pytest.approx(0.04)
    assert result["p_value"] < 0.01
    assert result["reject_at_5pct"] is True


def test_kupiec_rejects_a_model_that_never_breaches():
    """
    Too few exceptions is also a failure: the model is overstating risk, which
    wastes capital. Zero breaches in 2000 days at 99% is not conservatism, it
    is a miscalibration.
    """
    result = bt.kupiec_pof_test(n_exceptions=0, n_observations=2000, confidence=0.99)
    assert result["reject_at_5pct"] is True


def test_kupiec_handles_empty_sample():
    result = bt.kupiec_pof_test(0, 0, 0.99)
    assert np.isnan(result["lr_statistic"])


# ---------------------------------------------------------------------------
# Christoffersen
# ---------------------------------------------------------------------------
def test_independence_accepts_scattered_exceptions():
    rng = np.random.default_rng(1)
    exceptions = pd.Series(rng.binomial(1, 0.01, 5000))
    result = bt.christoffersen_independence_test(exceptions)
    assert result["p_value"] > 0.05
    assert result["reject_at_5pct"] is False


def test_independence_rejects_clustered_exceptions():
    """
    Every breach arriving in one consecutive block is the pathological case
    the test exists to catch.
    """
    values = np.zeros(2000, dtype=int)
    values[500:540] = 1
    result = bt.christoffersen_independence_test(pd.Series(values))
    assert result["p_value"] < 0.01
    assert result["reject_at_5pct"] is True


def test_conditional_coverage_combines_both_tests():
    values = np.zeros(2000, dtype=int)
    values[100:160] = 1
    exceptions = pd.Series(values)
    uc = bt.kupiec_pof_test(int(exceptions.sum()), len(exceptions), 0.99)
    ind = bt.christoffersen_independence_test(exceptions)
    cc = bt.conditional_coverage_test(exceptions, 0.99)
    assert cc["lr_statistic"] == pytest.approx(
        uc["lr_statistic"] + ind["lr_statistic"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# Basel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exceptions,zone",
    [(0, "Green"), (4, "Green"), (5, "Yellow"), (9, "Yellow"), (10, "Red"), (20, "Red")],
)
def test_basel_zones(exceptions, zone):
    assert bt.basel_traffic_light(exceptions, 250, 0.99)["zone"] == zone


def test_basel_scales_to_longer_samples():
    """12 exceptions over 1000 days is 3 per 250 days: still green."""
    assert bt.basel_traffic_light(12, 1000, 0.99)["zone"] == "Green"
    assert bt.basel_traffic_light(60, 1000, 0.99)["zone"] == "Red"


def test_basel_is_not_defined_at_95_percent():
    result = bt.basel_traffic_light(12, 250, 0.95)
    assert "n/a" in result["zone"]


# ---------------------------------------------------------------------------
# Rolling forecasts
# ---------------------------------------------------------------------------
def test_rolling_forecast_has_no_look_ahead():
    """
    The forecast for day t must be unchanged when day t's own return, and
    everything after it, is corrupted.
    """
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 1500))
    baseline = bt.rolling_var_forecast(returns, window=250, confidence=0.99)

    corrupted = returns.copy()
    corrupted.iloc[1000:] = -0.5
    perturbed = bt.rolling_var_forecast(corrupted, window=250, confidence=0.99)

    pd.testing.assert_series_equal(baseline.iloc[:1000], perturbed.iloc[:1000])


def test_rolling_forecast_starts_after_the_window():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 800))
    forecast = bt.rolling_var_forecast(returns, window=250, confidence=0.99)
    assert forecast.iloc[:250].isna().all()
    assert forecast.iloc[250:].notna().all()


def test_rolling_forecast_too_short_sample_returns_all_nan():
    returns = pd.Series(np.random.default_rng(4).normal(0, 0.01, 100))
    assert bt.rolling_var_forecast(returns, window=250).isna().all()


@pytest.mark.parametrize(
    "method", ["historical", "normal", "student_t", "cornish_fisher"]
)
def test_all_methods_produce_positive_var(method):
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.standard_t(5, size=1500) * 0.008)
    forecast = bt.rolling_var_forecast(returns, 250, 0.99, method).dropna()
    assert (forecast > 0).all()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_backtest_on_normal_data_is_well_calibrated():
    """
    Feeding i.i.d. normal returns to the normal VaR model is the one case where
    the model is exactly right, so Kupiec must not reject it.
    """
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0, 0.01, 8000))
    result = bt.backtest_var(returns, window=500, confidence=0.99, method="normal")
    assert result.kupiec["p_value"] > 0.05
    assert result.independence["p_value"] > 0.05
    assert result.basel["zone"] == "Green"


def test_backtest_detects_an_understated_model():
    """
    Volatility-clustered data breaks a model that assumes constant variance:
    both the count and the independence test should fail.
    """
    rng = np.random.default_rng(7)
    n = 6000
    regime = np.where(np.arange(n) % 1000 < 120, 4.0, 1.0)  # bursts of high vol
    returns = pd.Series(rng.normal(0, 0.008, n) * regime)
    result = bt.backtest_var(returns, window=500, confidence=0.99, method="normal")
    assert result.independence["reject_at_5pct"] is True


def test_exception_count_matches_the_flagged_days():
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.standard_t(4, size=3000) * 0.01)
    result = bt.backtest_var(returns, window=500, confidence=0.99)
    manual = int((result.realised < -result.forecasts).sum())
    assert int(result.exceptions.sum()) == manual
    assert result.kupiec["n_exceptions"] == manual


def test_breach_severity_exceeds_one():
    """By definition a breach loses more than the forecast, so the ratio > 1."""
    rng = np.random.default_rng(9)
    returns = pd.Series(rng.standard_t(4, size=3000) * 0.01)
    result = bt.backtest_var(returns, window=500, confidence=0.99)
    assert result.average_breach_severity > 1.0


def test_exceptions_by_year_totals_reconcile():
    rng = np.random.default_rng(10)
    index = pd.bdate_range("2010-01-01", periods=3000)
    returns = pd.Series(rng.standard_t(5, size=3000) * 0.01, index=index)
    result = bt.backtest_var(returns, window=500, confidence=0.99)
    by_year = bt.exceptions_by_year(result)
    assert by_year["exceptions"].sum() == int(result.exceptions.sum())
    assert by_year["observations"].sum() == len(result.exceptions)


def test_backtest_summary_covers_every_combination():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.standard_t(5, size=2000) * 0.01)
    summary = bt.backtest_summary(
        returns, methods=["historical", "normal"], confidences=[0.95, 0.99], window=500
    )
    assert len(summary) == 4
    assert summary["n_exceptions"].notna().all()
