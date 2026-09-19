"""Shared fixtures.

The test suite runs on a *synthetic* panel rather than the shipped price data.
Synthetic data has a known generating process, so the tests can assert exact
mathematical properties (a Monte Carlo VaR must converge to the closed-form
normal VaR; component contributions must sum to the total) instead of merely
asserting that today's numbers equal yesterday's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk.config import load_config

TRADING_DAYS = 252


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synthetic_returns(cfg) -> pd.DataFrame:
    """
    A 12-year daily panel with a realistic factor structure.

    Three latent factors (market, duration, commodity) drive the assets, with
    sleeve-appropriate volatilities and a fat-tailed common shock, so that
    correlations, betas and tail behaviour are all non-trivial.
    """
    rng = np.random.default_rng(20260919)
    tickers = cfg.tickers
    sleeves = cfg.sleeve_map()
    dates = pd.bdate_range("2014-01-01", "2026-01-01")
    n, k = len(dates), len(tickers)

    vol_map = {"equity": 0.26, "equity_beta": 0.17, "fixed_income": 0.08,
               "commodity": 0.19, "real_estate": 0.23, "cash": 0.004}
    mu_map = {"equity": 0.09, "equity_beta": 0.08, "fixed_income": 0.03,
              "commodity": 0.02, "real_estate": 0.06, "cash": 0.015}
    load_map = {
        "equity": [0.80, -0.10, 0.15], "equity_beta": [0.95, -0.05, 0.10],
        "fixed_income": [-0.15, 0.88, 0.00], "commodity": [0.20, -0.08, 0.85],
        "real_estate": [0.70, 0.20, 0.05], "cash": [0.0, 0.05, 0.0],
    }

    vols = np.array([vol_map[sleeves[t]] for t in tickers]) / np.sqrt(TRADING_DAYS)
    mus = np.array([mu_map[sleeves[t]] for t in tickers]) / TRADING_DAYS
    loadings = np.array([load_map[sleeves[t]] for t in tickers])

    factors = rng.standard_normal((n, 3))
    idiosyncratic = rng.standard_normal((n, k))
    residual_scale = np.sqrt(np.clip(1 - (loadings ** 2).sum(axis=1), 0.01, None))
    z = factors @ loadings.T + idiosyncratic * residual_scale
    z = z / z.std(axis=0)

    shock = rng.standard_t(4, size=(n, 1)) * 0.25
    values = mus + vols * (z + shock * (loadings[:, 0] > 0.5))

    return pd.DataFrame(values, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def synthetic_prices(synthetic_returns) -> pd.DataFrame:
    return 100.0 * (1.0 + synthetic_returns).cumprod()


@pytest.fixture(scope="session")
def normal_returns() -> pd.Series:
    """A long, exactly-normal return series for closed-form checks."""
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(0.0004, 0.01, 200_000))
