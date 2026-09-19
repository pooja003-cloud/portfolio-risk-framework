"""
Data acquisition and cleaning.

Responsibilities
----------------
1. Download daily split- and dividend-adjusted closing prices for the
   configured universe from Yahoo Finance (via ``yfinance``) and cache one CSV
   per ticker under ``data/raw/``.
2. Align those series onto a single trading calendar, apply the documented
   missing-data policy, and persist a consolidated price panel and return
   panel under ``data/processed/``.
3. Emit a data-quality report so that every cleaning decision is auditable.

Reproducibility
---------------
Vendors revise history. To keep every published number rebuildable, the
repository ships a **snapshot** of the consolidated adjusted-close panel at
``data/raw/prices_snapshot.csv``. If no per-ticker cache is present, the loader
reads the snapshot rather than hitting the network, so a fresh clone reproduces
the results in the README exactly. Passing ``force=True`` (or running
``python -m prisk.pipeline fetch --force``) re-downloads from Yahoo Finance and
rewrites the caches, which is the right thing to do when extending the sample.

Missing-data policy (also documented in ``config/config.yaml``)
--------------------------------------------------------------
* The trading calendar is the set of dates on which the market proxy (SPY)
  traded, restricted to the configured sample window. This avoids creating
  phantom zero-return days from foreign or bond-market holidays.
* Interior gaps are forward-filled for at most ``max_ffill_days`` sessions. A
  stale price implies a zero return, which is the economically correct
  treatment for a non-trading day.
* Dates still missing any ticker after forward-filling are dropped.
* Back-filling is never performed: it would inject look-ahead bias.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from prisk.config import Config, load_config

SNAPSHOT_FILENAME = "prices_snapshot.csv"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def load_snapshot(cfg: Optional[Config] = None) -> Optional[pd.DataFrame]:
    """
    Read the versioned price snapshot, if one is present.

    Returns a wide DataFrame of adjusted closes (dates x tickers), or ``None``
    when no snapshot has been committed.
    """
    cfg = cfg or load_config()
    path = Path(cfg.path("data.cache_dir")) / SNAPSHOT_FILENAME
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame.sort_index()


def download_prices(
    cfg: Optional[Config] = None,
    force: bool = False,
    tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download (or load from cache) the raw OHLCV history for each ticker.

    Parameters
    ----------
    cfg : Config, optional
        Loaded configuration; defaults to ``load_config()``.
    force : bool
        Re-download even when a cache file already exists.
    tickers : list of str, optional
        Restrict the download to these tickers.

    Returns
    -------
    dict
        ``ticker -> DataFrame`` indexed by date with an ``adj_close`` column.
    """
    cfg = cfg or load_config()
    cache_dir = cfg.path("data.cache_dir")
    tickers = tickers or cfg.tickers
    start, end = cfg.data["start_date"], cfg.data["end_date"]

    out: Dict[str, pd.DataFrame] = {}
    to_download = []
    for ticker in tickers:
        cache_file = cache_dir / f"{ticker}.csv"
        if cache_file.exists() and not force:
            frame = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            out[ticker] = frame
        else:
            to_download.append(ticker)

    # Fall back to the committed snapshot before reaching for the network, so a
    # fresh clone reproduces the published results offline.
    if to_download and not force:
        snapshot = load_snapshot(cfg)
        if snapshot is not None:
            still_missing = []
            for ticker in to_download:
                if ticker in snapshot.columns:
                    frame = pd.DataFrame({"adj_close": snapshot[ticker].dropna()})
                    frame["volume"] = np.nan
                    frame.index.name = "date"
                    out[ticker] = frame
                else:
                    still_missing.append(ticker)
            to_download = still_missing

    if to_download:
        frames = _yahoo_download(to_download, start, end)
        for ticker, frame in frames.items():
            frame.to_csv(cache_dir / f"{ticker}.csv")
            out[ticker] = frame

    return {t: out[t] for t in tickers if t in out}


def _yahoo_download(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch adjusted closes from Yahoo Finance one ticker at a time."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "yfinance is required to download prices. Install it with "
            "`pip install yfinance`, or place pre-downloaded per-ticker CSV "
            "files (columns: date, adj_close) in data/raw/."
        ) from exc

    frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # 'Close' is then split- and dividend-adjusted
                progress=False,
                actions=False,
            )
        if raw is None or raw.empty:
            raise RuntimeError(
                f"No data returned for {ticker}. Check the ticker and the "
                f"sample window ({start} to {end})."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        frame = pd.DataFrame(
            {
                "adj_close": raw["Close"].astype(float),
                "volume": raw["Volume"].astype(float) if "Volume" in raw else np.nan,
            }
        )
        frame.index.name = "date"
        frames[ticker] = frame.sort_index()
    return frames


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def build_price_panel(
    cfg: Optional[Config] = None,
    raw: Optional[Dict[str, pd.DataFrame]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Align raw per-ticker series into one clean price panel.

    Returns
    -------
    DataFrame
        Index: trading date. Columns: one adjusted close per ticker.
    """
    cfg = cfg or load_config()
    raw = raw if raw is not None else download_prices(cfg)

    wide = pd.DataFrame(
        {ticker: frame["adj_close"] for ticker, frame in raw.items()}
    ).sort_index()
    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    wide.index.name = "date"

    # 1. Restrict to the sample window.
    wide = wide.loc[
        (wide.index >= pd.Timestamp(cfg.data["start_date"]))
        & (wide.index <= pd.Timestamp(cfg.data["end_date"]))
    ]

    # 2. Use the market proxy's calendar as the reference trading calendar.
    market = cfg.market_ticker
    if market in wide.columns:
        calendar = wide.index[wide[market].notna()]
        wide = wide.reindex(calendar)

    # 3. Forward-fill interior gaps up to the configured limit. Never back-fill.
    limit = int(cfg.data.get("max_ffill_days", 3))
    filled = wide.ffill(limit=limit)

    # 4. Drop any remaining incomplete rows.
    if cfg.data.get("drop_incomplete_rows", True):
        filled = filled.dropna(how="any")

    filled = filled[cfg.tickers] if set(cfg.tickers).issubset(filled.columns) else filled

    if save:
        out_dir = cfg.path("data.processed_dir")
        filled.to_csv(out_dir / "prices.csv", float_format="%.6f")
        quality_report(cfg, wide, filled).to_csv(out_dir / "data_quality_report.csv")

    return filled


def compute_returns(
    prices: pd.DataFrame, kind: str = "simple"
) -> pd.DataFrame:
    """
    Daily returns from a price panel.

    Parameters
    ----------
    kind : {"simple", "log"}
        Simple returns aggregate correctly across assets (a portfolio return is
        the weighted sum of simple asset returns), so they are the default for
        everything portfolio-level. Log returns aggregate correctly across
        time and are used for multi-period simulation.
    """
    if kind == "simple":
        returns = prices.pct_change()
    elif kind == "log":
        returns = np.log(prices / prices.shift(1))
    else:  # pragma: no cover - guard
        raise ValueError("kind must be 'simple' or 'log'")
    return returns.dropna(how="all")


def quality_report(
    cfg: Config, before: pd.DataFrame, after: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-ticker audit of the cleaning step.

    Columns
    -------
    first_obs, last_obs : coverage of the raw series.
    n_raw               : non-missing observations on the reference calendar.
    n_filled            : observations created by forward-filling.
    pct_filled          : those fills as a share of the final panel.
    n_final             : observations surviving into the clean panel.
    n_zero_returns      : zero-return days (a stale-price diagnostic).
    """
    rows = []
    for ticker in before.columns:
        raw_series = before[ticker]
        clean_series = after[ticker] if ticker in after.columns else pd.Series(dtype=float)
        n_filled = int(raw_series.reindex(clean_series.index).isna().sum())
        zero_returns = int(
            (clean_series.pct_change().abs() < 1e-12).sum()
        ) if not clean_series.empty else 0
        rows.append(
            {
                "ticker": ticker,
                "first_obs": raw_series.first_valid_index(),
                "last_obs": raw_series.last_valid_index(),
                "n_raw": int(raw_series.notna().sum()),
                "n_filled": n_filled,
                "pct_filled": round(100 * n_filled / max(len(clean_series), 1), 3),
                "n_final": int(clean_series.notna().sum()),
                "n_zero_returns": zero_returns,
            }
        )
    report = pd.DataFrame(rows).set_index("ticker")
    return report


def load_panel(cfg: Optional[Config] = None) -> pd.DataFrame:
    """Load the persisted clean price panel, building it if it is absent."""
    cfg = cfg or load_config()
    path = Path(cfg.path("data.processed_dir")) / "prices.csv"
    if path.exists():
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        panel.index.name = "date"
        return panel
    return build_price_panel(cfg)


def risk_free_daily(returns: pd.DataFrame, cfg: Optional[Config] = None) -> pd.Series:
    """
    Daily risk-free return series.

    The total return of the 1-3 month T-bill ETF is used as the risk-free
    proxy. It is observable, matches the daily frequency of the portfolio
    returns exactly, and keeps the Sharpe ratio internally consistent with the
    cash sleeve actually held in the volatility-targeted portfolio.
    """
    cfg = cfg or load_config()
    ticker = cfg.risk.get("risk_free_source", cfg.riskfree_ticker)
    if ticker in returns.columns:
        return returns[ticker].rename("rf")
    return pd.Series(0.0, index=returns.index, name="rf")
