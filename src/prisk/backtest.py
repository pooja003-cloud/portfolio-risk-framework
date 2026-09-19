"""
VaR backtesting.

A VaR model is a falsifiable forecast: at 99% confidence, losses should exceed
the one-day VaR on about 1% of days. Backtesting checks two things, and both
matter:

* **Unconditional coverage** — is the *number* of exceptions right? Tested with
  Kupiec's proportion-of-failures likelihood-ratio test.
* **Independence** — are the exceptions *spread out*, or do they cluster? A
  model can have exactly the right exception count and still be useless if all
  the breaches arrive in the same fortnight, because that is precisely when the
  capital is needed. Tested with Christoffersen's Markov independence test, and
  the two are combined into the conditional-coverage test.

The Basel traffic-light zones are also reported: they are the supervisory
standard for judging a 99% one-day model over a 250-day window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from prisk.risk import (
    cornish_fisher_var,
    historical_var,
    parametric_var,
)


# ---------------------------------------------------------------------------
# Rolling VaR forecasts
# ---------------------------------------------------------------------------
def rolling_var_forecast(
    returns: pd.Series,
    window: int = 500,
    confidence: float = 0.99,
    method: str = "historical",
    refit_every: int = 63,
) -> pd.Series:
    """
    One-day-ahead VaR forecast for each date, estimated on the trailing window.

    Critically, the forecast for day *t* uses only returns up to *t-1*. The
    series is therefore directly comparable with the realised return on day
    *t*, with no look-ahead.

    Parameters
    ----------
    method : {"historical", "normal", "student_t", "cornish_fisher"}
    refit_every : int
        The Student-t shape is re-fitted by maximum likelihood every
        ``refit_every`` days rather than daily; the *scale* still updates every
        day. Concretely, a refit produces both the degrees of freedom ν and the
        ratio of the fitted scale to the window standard deviation, and those
        two shape parameters are held until the next refit while the rolling
        standard deviation moves daily. A one-day change in a 500-day window
        moves the MLE by a negligible amount, and daily refitting makes a
        twenty-year backtest an order of magnitude slower for no gain.
    """
    clean = returns.dropna()
    values = clean.values.astype(float)
    index = clean.index
    n = len(clean)
    forecasts = np.full(n, np.nan)
    if n <= window:
        return pd.Series(forecasts, index=index, name=f"var_{method}_{confidence:.2f}")

    # Trailing windows ending at t-1, for t = window ... n-1.
    windows = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
    rows = np.arange(window, n)

    if method == "historical":
        quantiles = np.quantile(windows, 1.0 - confidence, axis=1)
        forecasts[rows] = -quantiles
        return pd.Series(forecasts, index=index, name=f"var_{method}_{confidence:.2f}")

    means = windows.mean(axis=1)
    stds = windows.std(axis=1, ddof=1)

    if method == "normal":
        z = stats.norm.ppf(1.0 - confidence)
        forecasts[rows] = -(means + z * stds)

    elif method == "cornish_fisher":
        skews = stats.skew(windows, axis=1)
        kurts = stats.kurtosis(windows, axis=1)
        z = stats.norm.ppf(1.0 - confidence)
        z_cf = (
            z
            + (z ** 2 - 1) * skews / 6.0
            + (z ** 3 - 3 * z) * kurts / 24.0
            - (2 * z ** 3 - 5 * z) * (skews ** 2) / 36.0
        )
        forecasts[rows] = -(means + z_cf * stds)

    elif method == "student_t":
        dof_values = np.empty(len(rows))
        scale_ratio = np.empty(len(rows))
        current_dof, current_ratio = 6.0, 1.0
        for i in range(len(rows)):
            if i % max(int(refit_every), 1) == 0:
                fitted_dof, _, fitted_scale = stats.t.fit(windows[i])
                current_dof = float(np.clip(fitted_dof, 2.05, 100.0))
                window_std = stds[i]
                current_ratio = (
                    float(fitted_scale / window_std) if window_std > 0 else 1.0
                )
            dof_values[i] = current_dof
            scale_ratio[i] = current_ratio
        q = stats.t.ppf(1.0 - confidence, dof_values)
        forecasts[rows] = -(means + q * scale_ratio * stds)

    else:  # pragma: no cover - guard
        raise ValueError(f"Unknown method: {method}")

    return pd.Series(forecasts, index=index, name=f"var_{method}_{confidence:.2f}")


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------
def kupiec_pof_test(n_exceptions: int, n_observations: int, confidence: float) -> Dict:
    """
    Kupiec proportion-of-failures test.

    ``H0``: the true exception rate equals ``1 - confidence``.

    ``LR_uc = -2 ln[ (1-p)^(n-x) p^x / (1-π̂)^(n-x) π̂^x ]``  ~ χ²(1)

    A p-value below 0.05 rejects the model: it breaches either too often
    (understating risk) or too rarely (overstating it, which wastes capital).
    """
    p = 1.0 - confidence
    n, x = int(n_observations), int(n_exceptions)
    if n == 0:
        return {"lr_statistic": np.nan, "p_value": np.nan, "reject_at_5pct": None}

    pi_hat = x / n
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    elif x == n:  # pragma: no cover - degenerate
        lr = -2.0 * (n * np.log(p))
    else:
        log_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
        lr = -2.0 * (log_null - log_alt)

    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    return {
        "n_observations": n,
        "n_exceptions": x,
        "expected_exceptions": n * p,
        "observed_rate": pi_hat,
        "expected_rate": p,
        "lr_statistic": float(lr),
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
    }


def christoffersen_independence_test(exceptions: pd.Series) -> Dict:
    """
    Christoffersen Markov test for independence of exceptions.

    Counts the transitions ``n_ij`` between the no-exception (0) and exception
    (1) states and tests ``H0: π_01 = π_11`` — that an exception today tells
    you nothing about tomorrow. Rejection means the breaches cluster, the
    classic symptom of a model that ignores volatility clustering.
    """
    e = exceptions.dropna().astype(int).values
    if len(e) < 2:
        return {"lr_statistic": np.nan, "p_value": np.nan, "reject_at_5pct": None}

    n00 = int(np.sum((e[:-1] == 0) & (e[1:] == 0)))
    n01 = int(np.sum((e[:-1] == 0) & (e[1:] == 1)))
    n10 = int(np.sum((e[:-1] == 1) & (e[1:] == 0)))
    n11 = int(np.sum((e[:-1] == 1) & (e[1:] == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log(value: float) -> float:
        return np.log(value) if value > 0 else 0.0

    log_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_alt = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11)
    )
    lr = -2.0 * (log_null - log_alt)
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": pi01, "pi_11": pi11,
        "lr_statistic": float(lr),
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
    }


def conditional_coverage_test(
    exceptions: pd.Series, confidence: float
) -> Dict:
    """
    Christoffersen conditional-coverage test.

    ``LR_cc = LR_uc + LR_ind`` ~ χ²(2). Jointly tests that the exception rate
    is correct *and* that exceptions are independent.
    """
    uc = kupiec_pof_test(int(exceptions.sum()), int(exceptions.notna().sum()), confidence)
    ind = christoffersen_independence_test(exceptions)
    lr = float(np.nansum([uc["lr_statistic"], ind["lr_statistic"]]))
    p_value = float(1.0 - stats.chi2.cdf(lr, df=2))
    return {
        "lr_statistic": lr,
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
    }


def basel_traffic_light(
    n_exceptions: int, n_observations: int = 250, confidence: float = 0.99
) -> Dict:
    """
    Basel Committee traffic-light zone for a 99% one-day VaR model.

    Over 250 observations: 0-4 exceptions is green (the model is accepted),
    5-9 is yellow (a capital multiplier applies and the bank must explain
    itself), 10 or more is red (the model is presumed flawed).

    The zones are defined **only** for a 99% model; applying them to a 95%
    model, whose expected exception count over 250 days is 12.5, would label
    every correctly calibrated model red. Other confidence levels therefore
    return ``"n/a"``. Counts are scaled proportionally when the sample is
    longer than 250 days, so a twenty-year backtest is judged on its average
    250-day behaviour.
    """
    if abs(confidence - 0.99) > 1e-9:
        return {
            "zone": "n/a (defined for 99% only)",
            "capital_multiplier": np.nan,
            "n_exceptions": int(n_exceptions),
            "n_observations": int(n_observations),
        }

    scale = max(n_observations / 250.0, 1e-9)
    equivalent = n_exceptions / scale  # exceptions per 250 observations
    if equivalent < 5:
        zone, multiplier = "Green", 3.00
    elif equivalent < 10:
        increments = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
        zone = "Yellow"
        multiplier = increments.get(int(np.floor(equivalent)), 3.65)
    else:
        zone, multiplier = "Red", 4.00
    return {
        "zone": zone,
        "capital_multiplier": multiplier,
        "n_exceptions": int(n_exceptions),
        "n_observations": int(n_observations),
        "exceptions_per_250_days": round(float(equivalent), 2),
    }


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------
@dataclass
class VarBacktestResult:
    """Everything produced by one VaR backtest."""

    method: str
    confidence: float
    window: int
    forecasts: pd.Series
    realised: pd.Series
    exceptions: pd.Series
    kupiec: Dict = field(default_factory=dict)
    independence: Dict = field(default_factory=dict)
    conditional_coverage: Dict = field(default_factory=dict)
    basel: Dict = field(default_factory=dict)
    average_breach_severity: float = np.nan

    def summary_row(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "window": self.window,
            "n_observations": self.kupiec.get("n_observations"),
            "n_exceptions": self.kupiec.get("n_exceptions"),
            "expected_exceptions": self.kupiec.get("expected_exceptions"),
            "observed_rate": self.kupiec.get("observed_rate"),
            "kupiec_p_value": self.kupiec.get("p_value"),
            "kupiec_reject": self.kupiec.get("reject_at_5pct"),
            "independence_p_value": self.independence.get("p_value"),
            "independence_reject": self.independence.get("reject_at_5pct"),
            "conditional_coverage_p_value": self.conditional_coverage.get("p_value"),
            "basel_zone": self.basel.get("zone"),
            "avg_breach_severity": self.average_breach_severity,
        }


def backtest_var(
    returns: pd.Series,
    window: int = 500,
    confidence: float = 0.99,
    method: str = "historical",
) -> VarBacktestResult:
    """
    Run a full one-day VaR backtest.

    An exception is a day on which the realised loss exceeded the VaR forecast
    made the previous day. ``average_breach_severity`` reports the mean ratio of
    realised loss to forecast VaR on exception days: a value well above one
    says that when the model is wrong, it is badly wrong.
    """
    forecasts = rolling_var_forecast(returns, window, confidence, method)
    aligned = pd.concat(
        [returns.rename("realised"), forecasts.rename("var")], axis=1
    ).dropna()

    exceptions = (aligned["realised"] < -aligned["var"]).astype(int)
    breaches = aligned[exceptions == 1]
    severity = (
        float((-breaches["realised"] / breaches["var"]).mean())
        if not breaches.empty
        else np.nan
    )

    return VarBacktestResult(
        method=method,
        confidence=confidence,
        window=window,
        forecasts=aligned["var"],
        realised=aligned["realised"],
        exceptions=exceptions,
        kupiec=kupiec_pof_test(int(exceptions.sum()), len(exceptions), confidence),
        independence=christoffersen_independence_test(exceptions),
        conditional_coverage=conditional_coverage_test(exceptions, confidence),
        basel=basel_traffic_light(
            int(exceptions.sum()), len(exceptions), confidence
        ),
        average_breach_severity=severity,
    )


def backtest_summary(
    returns: pd.Series,
    methods: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    window: int = 500,
) -> pd.DataFrame:
    """Backtest every method at every confidence level and tabulate."""
    methods = methods or ["historical", "normal", "student_t", "cornish_fisher"]
    confidences = confidences or [0.95, 0.99]
    rows = []
    for method in methods:
        for confidence in confidences:
            rows.append(
                backtest_var(returns, window, confidence, method).summary_row()
            )
    return pd.DataFrame(rows)


def exceptions_by_year(result: VarBacktestResult) -> pd.DataFrame:
    """
    Exception counts by calendar year.

    The clearest way to see clustering: a model whose breaches are evenly
    spread is doing its job, one that delivers all of them in 2008 and 2020 is
    not.
    """
    frame = pd.DataFrame(
        {
            "exception": result.exceptions,
            "realised": result.realised,
            "var": result.forecasts,
        }
    )
    grouped = frame.groupby(frame.index.year).agg(
        observations=("exception", "size"),
        exceptions=("exception", "sum"),
        worst_loss=("realised", "min"),
        average_var=("var", "mean"),
    )
    grouped["exception_rate"] = grouped["exceptions"] / grouped["observations"]
    grouped.index.name = "year"
    return grouped
