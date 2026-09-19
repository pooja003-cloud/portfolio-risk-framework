"""
End-to-end pipeline.

Rebuilds every table and figure in the repository from the price panel, so
that no published number is the product of an un-rerunnable notebook cell.

Usage
-----
    python -m prisk.pipeline              # build everything
    python -m prisk.pipeline fetch        # refresh the price data only
    python -m prisk.pipeline fetch --force  # re-download from Yahoo Finance
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from prisk import backtest as bt
from prisk import data as dataio
from prisk import metrics as mx
from prisk import plots
from prisk import portfolios as pf
from prisk import risk as rk
from prisk import stress as st
from prisk.config import Config, load_config

PORTFOLIO_ORDER = ["equal_weight", "minimum_variance", "volatility_target", "benchmark"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(message: str) -> None:
    print(f"  {message}", flush=True)


def _write(frame: pd.DataFrame, cfg: Config, name: str, index: bool = True) -> None:
    path = Path(cfg.path("output.tables_dir")) / f"{name}.csv"
    frame.to_csv(path, index=index)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_data(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Download or load prices and build the clean panel."""
    raw = dataio.download_prices(cfg, force=force)
    panel = dataio.build_price_panel(cfg, raw=raw, save=True)
    _log(f"price panel: {panel.shape[0]} days x {panel.shape[1]} tickers "
         f"({panel.index[0].date()} to {panel.index[-1].date()})")
    return panel


def stage_portfolios(
    cfg: Config, returns: pd.DataFrame
) -> Dict[str, pf.BacktestResult]:
    """Run all four portfolios through the rebalancing engine."""
    results = pf.build_all_portfolios(returns, cfg)
    for name in PORTFOLIO_ORDER:
        result = results[name]
        _log(f"{result.label:<24} {result.meta['n_rebalances']:>3} rebalances, "
             f"turnover {pf.annual_turnover(result, cfg.trading_days):.1%}/yr")
        _write(result.weights_target, cfg, f"weights_rebalance_{name}")
    return results


def stage_performance(
    cfg: Config, returns: pd.DataFrame, results: Dict[str, pf.BacktestResult]
) -> pd.DataFrame:
    """Headline performance and risk table, plus the core performance charts."""
    market = returns[cfg.market_ticker]
    risk_free = dataio.risk_free_daily(returns, cfg)
    benchmark = results["benchmark"].returns

    summary = pd.DataFrame(
        {
            name: mx.performance_summary(
                results[name].returns,
                market=market,
                risk_free=risk_free,
                benchmark=benchmark if name != "benchmark" else None,
                trading_days=cfg.trading_days,
                mar=cfg.risk.get("sortino_mar", 0.0),
            )
            for name in PORTFOLIO_ORDER
        }
    ).T
    summary["annual_turnover"] = [
        pf.annual_turnover(results[n], cfg.trading_days) for n in PORTFOLIO_ORDER
    ]
    summary["cost_drag_annual"] = [
        results[n].cost_drag.sum() / (len(results[n].cost_drag) / cfg.trading_days)
        for n in PORTFOLIO_ORDER
    ]
    _write(summary, cfg, "performance_summary")

    labels = {n: results[n].label for n in PORTFOLIO_ORDER}
    series = {n: results[n].returns for n in PORTFOLIO_ORDER}
    figures = Path(cfg.path("output.figures_dir"))
    dpi = int(cfg.output.get("dpi", 160))

    plots.save(plots.plot_cumulative_growth(series, labels), figures / "01_growth_of_1.png", dpi)
    plots.save(plots.plot_drawdowns(series, labels), figures / "02_drawdowns.png", dpi)
    plots.save(
        plots.plot_rolling_volatility(
            series,
            window=int(cfg.risk["rolling_vol_window"]),
            target=float(cfg.portfolios["volatility_target"]["target_volatility"]),
            labels=labels,
        ),
        figures / "03_rolling_volatility.png", dpi,
    )
    plots.save(plots.plot_risk_return_scatter(summary, labels),
               figures / "04_risk_return.png", dpi)

    # Drawdown anatomy is a table in its own right.
    anatomy = pd.DataFrame(
        {n: mx.max_drawdown(results[n].returns) for n in PORTFOLIO_ORDER}
    ).T
    _write(anatomy, cfg, "drawdown_anatomy")
    return summary


def stage_structure(
    cfg: Config, returns: pd.DataFrame, results: Dict[str, pf.BacktestResult]
) -> Dict[str, pd.DataFrame]:
    """Correlations, allocations, and asset-level risk contribution."""
    figures = Path(cfg.path("output.figures_dir"))
    dpi = int(cfg.output.get("dpi", 160))
    holdings = [t for t in cfg.holdings if t in returns.columns]
    window = int(cfg.portfolios["estimation_window"])

    correlation = returns[holdings].corr()
    _write(correlation, cfg, "correlation_matrix")
    plots.save(
        plots.plot_correlation_heatmap(
            correlation,
            subtitle=f"Daily returns, {returns.index[0].date()} to {returns.index[-1].date()}",
        ),
        figures / "05_correlation_matrix.png", dpi,
    )

    # Recent-window correlation, for the "diversification decays" comparison.
    recent = returns[holdings].iloc[-window:].corr()
    _write(recent, cfg, "correlation_matrix_recent")

    cov = pf.estimate_covariance(
        returns[holdings].iloc[-window:],
        method=cfg.portfolios.get("covariance_estimator", "ledoit_wolf"),
    )

    contributions: Dict[str, pd.DataFrame] = {}
    sleeve_map = cfg.sleeve_map()
    for name in PORTFOLIO_ORDER:
        weights = results[name].latest_weights()
        table = mx.risk_contributions(weights, cov, cfg.trading_days)
        contributions[name] = table
        _write(table, cfg, f"risk_contribution_{name}")
        _write(
            mx.group_risk_contributions(table, sleeve_map),
            cfg, f"risk_contribution_by_sleeve_{name}",
        )
        plots.save(
            plots.plot_risk_contribution(
                table,
                title=f"{results[name].label}: weight versus risk contribution",
            ),
            figures / f"06_risk_contribution_{name}.png", dpi,
        )
        plots.save(
            plots.plot_allocation_over_time(
                results[name].weights_daily, sleeve_map,
                title=f"{results[name].label}: allocation through time",
                subtitle="Quarterly rebalancing; weights drift between dates",
            ),
            figures / f"07_allocation_{name}.png", dpi,
        )

    # The headline allocation-and-risk deliverable.
    allocation = pd.DataFrame(
        {results[n].label: results[n].latest_weights() for n in PORTFOLIO_ORDER}
    )
    allocation.insert(0, "sleeve", [sleeve_map.get(t, "") for t in allocation.index])
    allocation.insert(1, "sector", [cfg.sector_map().get(t, "") for t in allocation.index])
    allocation["annualised_volatility"] = np.sqrt(
        np.diag(cov.loc[allocation.index, allocation.index]) * cfg.trading_days
    )
    for name in PORTFOLIO_ORDER:
        allocation[f"{results[name].label} risk %"] = contributions[name][
            "pct_risk_contribution"
        ]
    _write(allocation, cfg, "allocation_and_risk_table")
    _write_allocation_markdown(cfg, allocation, results, contributions)
    return contributions


def _write_allocation_markdown(
    cfg: Config,
    allocation: pd.DataFrame,
    results: Dict[str, pf.BacktestResult],
    contributions: Dict[str, pd.DataFrame],
) -> None:
    """Render the allocation-and-risk deliverable as a readable Markdown table."""
    path = cfg.root / "reports" / "portfolio_allocation_and_risk_table.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    names = cfg.name_map()
    lines = [
        "# Portfolio Allocation and Risk Table",
        "",
        f"Current target weights as at **{results['equal_weight'].weights_daily.index[-1].date()}**, "
        "with each holding's standalone annualised volatility and its percentage "
        "contribution to each portfolio's total volatility.",
        "",
        "> Hypothetical research portfolios built on publicly available price data. "
        "Not investment advice and not a forecast.",
        "",
        "## Weights",
        "",
        "| Ticker | Name | Sleeve | Sector | Ann. vol | "
        + " | ".join(results[n].label for n in PORTFOLIO_ORDER)
        + " |",
        "|---|---|---|---|---:|" + "---:|" * len(PORTFOLIO_ORDER),
    ]
    for ticker in allocation.index:
        row = allocation.loc[ticker]
        weights = " | ".join(
            f"{row[results[n].label]:.2%}" for n in PORTFOLIO_ORDER
        )
        lines.append(
            f"| {ticker} | {names.get(ticker, '')} | {row['sleeve']} | "
            f"{row['sector']} | {row['annualised_volatility']:.1%} | {weights} |"
        )

    lines += [
        "",
        "## Percentage contribution to portfolio volatility",
        "",
        "| Ticker | " + " | ".join(results[n].label for n in PORTFOLIO_ORDER) + " |",
        "|---|" + "---:|" * len(PORTFOLIO_ORDER),
    ]
    for ticker in allocation.index:
        row = allocation.loc[ticker]
        risks = " | ".join(
            f"{row[f'{results[n].label} risk %']:.1%}" for n in PORTFOLIO_ORDER
        )
        lines.append(f"| {ticker} | {risks} |")

    lines += ["", "## By sleeve", ""]
    sleeve_map = cfg.sleeve_map()
    for name in PORTFOLIO_ORDER:
        grouped = mx.group_risk_contributions(contributions[name], sleeve_map)
        lines += [
            f"### {results[name].label}",
            "",
            "| Sleeve | Weight | Risk contribution | Risk / capital |",
            "|---|---:|---:|---:|",
        ]
        for sleeve, row in grouped.iterrows():
            ratio = row["risk_to_capital_ratio"]
            ratio_text = "—" if pd.isna(ratio) else f"{ratio:.2f}"
            lines.append(
                f"| {sleeve.replace('_', ' ').title()} | {row['weight']:.1%} | "
                f"{row['pct_risk_contribution']:.1%} | {ratio_text} |"
            )
        lines.append("")

    lines += [
        "A risk-to-capital ratio above 1.0 means the sleeve consumes more risk budget "
        "than capital. It is the number that turns *\"we hold 20% in commodities\"* into "
        "*\"that 20% is doing 35% of the work\"*.",
        "",
        "*Generated by `python -m prisk.pipeline`.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def stage_risk(
    cfg: Config, returns: pd.DataFrame, results: Dict[str, pf.BacktestResult]
) -> Dict[str, pd.DataFrame]:
    """VaR, expected shortfall, Monte Carlo, and component VaR."""
    figures = Path(cfg.path("output.figures_dir"))
    dpi = int(cfg.output.get("dpi", 160))
    holdings = [t for t in cfg.holdings if t in returns.columns]
    window = int(cfg.risk["monte_carlo"]["estimation_window"])
    value = float(cfg.risk.get("portfolio_value", 1_000_000))
    confidences = list(cfg.risk["confidence_levels"])
    horizons = list(cfg.risk["var_horizons_days"])

    mc_config = cfg.risk["monte_carlo"]
    estimator = cfg.portfolios.get("covariance_estimator", "ledoit_wolf")

    # Every method must see the same data, or the comparison measures the
    # sample rather than the method. Two samples are reported side by side:
    # the full history (unconditional risk) and the trailing estimation window
    # (risk in the current regime).
    samples = {
        "full_sample": returns[holdings],
        f"trailing_{window}d": returns[holdings].iloc[-window:],
    }
    covariances = {
        key: pf.estimate_covariance(frame, method=estimator)
        for key, frame in samples.items()
    }
    cov = covariances[f"trailing_{window}d"]

    comparisons: Dict[str, pd.DataFrame] = {}
    for name in PORTFOLIO_ORDER:
        weights = results[name].latest_weights()
        blocks = []
        for sample_key, asset_frame in samples.items():
            portfolio_returns = (
                results[name].returns
                if sample_key == "full_sample"
                else results[name].returns.iloc[-window:]
            )
            block = rk.var_comparison(
                portfolio_returns,
                asset_returns=asset_frame,
                weights=weights,
                confidences=confidences,
                horizons=horizons,
                portfolio_value=value,
                mc_kwargs={
                    "n_simulations": int(mc_config["n_simulations"]),
                    "distribution": mc_config["distribution"],
                    "seed": int(mc_config["seed"]),
                    "cov": covariances[sample_key],
                },
            )
            block.insert(0, "sample", sample_key)
            blocks.append(block)

        table = pd.concat(blocks, ignore_index=True)
        table.insert(0, "portfolio", results[name].label)
        comparisons[name] = table
        _write(table, cfg, f"var_comparison_{name}", index=False)

        for confidence in confidences:
            _write(
                rk.component_var(weights, cov, confidence, 1),
                cfg, f"component_var_{name}_{int(confidence * 100)}",
            )

        full = table[table["sample"] == "full_sample"]
        plots.save(plots.plot_var_comparison(full, horizon=1),
                   figures / f"08_var_methods_{name}.png", dpi)

        one_day_99 = full[(full["horizon_days"] == 1) & (full["confidence"] == 0.99)]
        plots.save(
            plots.plot_return_distribution(
                results[name].returns,
                var=float(one_day_99["historical_var"].iloc[0]),
                expected_shortfall=float(one_day_99["historical_es"].iloc[0]),
                confidence=0.99,
                label=results[name].label,
            ),
            figures / f"09_return_distribution_{name}.png", dpi,
        )

    _write(pd.concat(comparisons.values(), ignore_index=True), cfg,
           "var_comparison_all", index=False)

    # Monte Carlo deep dive on the primary portfolio, on the full sample.
    primary = cfg.output.get("primary_portfolio", "volatility_target")
    mc = rk.monte_carlo_var(
        returns[holdings],
        results[primary].latest_weights(),
        confidences=confidences,
        horizon=10,
        n_simulations=int(mc_config["n_simulations"]),
        distribution=mc_config["distribution"],
        seed=int(mc_config["seed"]),
        cov=covariances["full_sample"],
    )
    plots.save(
        plots.plot_monte_carlo_distribution(
            mc.simulated_returns, mc.var[0.99], mc.expected_shortfall[0.99],
            confidence=0.99, horizon=10,
            n_paths_label=(
                f"{mc.n_simulations:,} paths, multivariate Student-t "
                f"(v={mc.dof:.1f}) on the Ledoit-Wolf covariance"
            ),
        ),
        figures / f"10_monte_carlo_{primary}.png", dpi,
    )
    return comparisons


def stage_stress(
    cfg: Config, returns: pd.DataFrame, results: Dict[str, pf.BacktestResult]
) -> None:
    """Historical replay, factor shocks, and correlation stress."""
    figures = Path(cfg.path("output.figures_dir"))
    dpi = int(cfg.output.get("dpi", 160))
    holdings = [t for t in cfg.holdings if t in returns.columns]
    value = float(cfg.risk.get("portfolio_value", 1_000_000))
    window = int(cfg.portfolios["estimation_window"])
    labels = {n: results[n].label for n in PORTFOLIO_ORDER}

    weights_map = {n: results[n].latest_weights() for n in PORTFOLIO_ORDER}

    historical = st.historical_stress_table(returns[holdings], weights_map, cfg)
    historical["loss_usd"] = historical["total_return"] * value
    _write(historical, cfg, "stress_historical_episodes", index=False)
    plots.save(plots.plot_episode_comparison(historical, labels),
               figures / "11_stress_historical.png", dpi)

    cov = pf.estimate_covariance(
        returns[holdings].iloc[-window:],
        method=cfg.portfolios.get("covariance_estimator", "ledoit_wolf"),
    )
    hypothetical = st.hypothetical_stress_table(
        returns[holdings], weights_map, cfg, cov=cov, portfolio_value=value
    )
    _write(hypothetical, cfg, "stress_hypothetical_scenarios", index=False)
    plots.save(plots.plot_stress_scenarios(hypothetical, labels=labels),
               figures / "12_stress_hypothetical.png", dpi)

    betas = st.estimate_factor_betas(returns[holdings], cfg)
    _write(betas, cfg, "factor_betas")

    primary = cfg.output.get("primary_portfolio", "volatility_target")
    _write(
        st.scenario_asset_detail(
            weights_map[primary], returns[holdings], "Severe Combined Stress", cfg
        ),
        cfg, f"stress_severe_detail_{primary}",
    )

    correlation_stress = st.correlation_stress_impact(
        weights_map[primary], cov, trading_days=cfg.trading_days
    )
    _write(correlation_stress, cfg, "stress_correlation", index=False)
    plots.save(plots.plot_correlation_stress(correlation_stress),
               figures / "13_stress_correlation.png", dpi)


def stage_backtest(
    cfg: Config, results: Dict[str, pf.BacktestResult]
) -> pd.DataFrame:
    """Rolling VaR backtests and coverage tests."""
    figures = Path(cfg.path("output.figures_dir"))
    dpi = int(cfg.output.get("dpi", 160))
    spec = cfg.risk["backtest"]
    window = int(spec["rolling_window"])
    confidence = float(spec["confidence_level"])

    rows = []
    for name in PORTFOLIO_ORDER:
        table = bt.backtest_summary(
            results[name].returns,
            confidences=list(cfg.risk["confidence_levels"]),
            window=window,
        )
        table.insert(0, "portfolio", results[name].label)
        rows.append(table)

        result = bt.backtest_var(
            results[name].returns, window, confidence, spec["var_method"]
        )
        plots.save(
            plots.plot_var_backtest(
                result.realised, result.forecasts, result.exceptions,
                confidence=confidence, method=spec["var_method"],
            ),
            figures / f"14_var_backtest_{name}.png", dpi,
        )
        _write(bt.exceptions_by_year(result), cfg, f"var_exceptions_by_year_{name}")

    summary = pd.concat(rows, ignore_index=True)
    _write(summary, cfg, "var_backtest_summary", index=False)

    primary = cfg.output.get("primary_portfolio", "volatility_target")
    primary_result = bt.backtest_var(
        results[primary].returns, window, confidence, spec["var_method"]
    )
    plots.save(
        plots.plot_exceptions_by_year(
            bt.exceptions_by_year(primary_result), confidence
        ),
        figures / f"15_exceptions_by_year_{primary}.png", dpi,
    )
    return summary


def stage_headlines(
    cfg: Config,
    summary: pd.DataFrame,
    comparisons: Dict[str, pd.DataFrame],
    backtests: pd.DataFrame,
    panel: pd.DataFrame,
) -> Dict:
    """Write the headline numbers the README and memo quote, as JSON."""
    headline = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": {
            "start": str(panel.index[0].date()),
            "end": str(panel.index[-1].date()),
            "trading_days": int(len(panel)),
            "tickers": int(panel.shape[1]),
        },
        "portfolio_value": float(cfg.risk.get("portfolio_value", 1_000_000)),
        "performance": {
            name: {
                key: (None if pd.isna(value) else float(value))
                for key, value in summary.loc[name].items()
                if isinstance(value, (int, float, np.floating))
            }
            for name in summary.index
        },
        "var_1d_99_full_sample": {
            name: {
                column: float(
                    table[
                        (table["sample"] == "full_sample")
                        & (table["horizon_days"] == 1)
                        & (table["confidence"] == 0.99)
                    ][column].iloc[0]
                )
                for column in [
                    "historical_var", "parametric_normal_var", "parametric_t_var",
                    "cornish_fisher_var", "monte_carlo_var", "historical_es",
                ]
                if column in table.columns
            }
            for name, table in comparisons.items()
        },
        "backtest_99_historical": backtests[
            (backtests["confidence"] == 0.99) & (backtests["method"] == "historical")
        ].to_dict("records"),
    }
    path = Path(cfg.path("output.tables_dir")) / "headline_results.json"
    path.write_text(json.dumps(headline, indent=2), encoding="utf-8")
    return headline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(cfg: Config | None = None, force_download: bool = False) -> Dict:
    """Run every stage in order."""
    cfg = cfg or load_config()
    plots.use_style()
    started = time.time()

    print("[1/7] data")
    panel = stage_data(cfg, force=force_download)
    returns = dataio.compute_returns(panel)

    print("[2/7] portfolio construction")
    results = stage_portfolios(cfg, returns)

    print("[3/7] performance")
    summary = stage_performance(cfg, returns, results)

    print("[4/7] structure and risk contribution")
    stage_structure(cfg, returns, results)

    print("[5/7] value at risk")
    comparisons = stage_risk(cfg, returns, results)

    print("[6/7] stress testing")
    stage_stress(cfg, returns, results)

    print("[7/7] var backtesting")
    backtests = stage_backtest(cfg, results)

    headline = stage_headlines(cfg, summary, comparisons, backtests, panel)
    print(f"\nDone in {time.time() - started:.1f}s.")
    print(f"Tables:  {cfg.path('output.tables_dir')}")
    print(f"Figures: {cfg.path('output.figures_dir')}")
    return headline


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    force = "--force" in argv
    argv = [a for a in argv if not a.startswith("--")]
    command = argv[0] if argv else "all"

    cfg = load_config()
    if command == "fetch":
        panel = stage_data(cfg, force=force)
        print(f"Wrote {len(panel)} rows to {cfg.path('data.processed_dir')}/prices.csv")
        return 0
    if command in {"all", "run"}:
        run(cfg, force_download=force)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
