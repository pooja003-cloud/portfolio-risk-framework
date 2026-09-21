"""
Interactive portfolio risk dashboard.

Run with::

    streamlit run dashboard/app.py

The dashboard is a front end onto the same ``prisk`` package that produces the
static report, so nothing shown here is computed twice or differently. Every
panel recomputes live from the price panel, which lets a reader change the
confidence level, the horizon, or the stress assumptions and watch the numbers
move — the point being that a risk number is the output of a set of choices,
not a fact.

DISCLAIMER: hypothetical portfolios built on public data for research purposes.
Not investment advice, not a forecast, and not a real client account.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prisk import backtest as bt          # noqa: E402
from prisk import data as dataio          # noqa: E402
from prisk import metrics as mx           # noqa: E402
from prisk import portfolios as pf        # noqa: E402
from prisk import risk as rk              # noqa: E402
from prisk import stress as stx           # noqa: E402
from prisk.config import load_config      # noqa: E402

# ---------------------------------------------------------------------------
# Palette (mirrors prisk.plots so the app and the report look like one system)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CRITICAL = "#d03b3b"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PORTFOLIO_COLORS = {
    "equal_weight": CATEGORICAL[0],
    "minimum_variance": CATEGORICAL[1],
    "volatility_target": CATEGORICAL[2],
    "benchmark": INK_SECONDARY,
}

st.set_page_config(
    page_title="Portfolio Risk & Stress Testing",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def style(
    fig: go.Figure, height: int = 420, hovermode: str = "x unified", **kwargs
) -> go.Figure:
    """
    Apply the shared chart styling.

    Every text colour is set explicitly rather than left to inherit. Plotly's
    global ``font.color`` does *not* cascade into titles, legends and axis
    titles once another template has an opinion about them, so those are
    assigned individually — otherwise a dark host theme renders them in a pale
    ink that vanishes against the chart's light surface.

    ``hovermode`` is a named parameter so callers can override it without
    colliding with the default passed through ``**kwargs``.
    """
    title = kwargs.pop("title", None)

    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=INK),
        # Room at the top for a title line and a legend line that do not
        # overlap each other. The left and bottom values are floors only —
        # `automargin` on both axes grows them to fit the tick labels and axis
        # titles, which a fixed margin clips.
        margin=dict(l=10, r=10, t=86, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK_SECONDARY, size=11.5)),
        hovermode=hovermode,
        **kwargs,
    )

    if title is not None:
        # Bold via markup rather than font(weight=...), which older Plotly
        # releases reject — Streamlit Cloud does not always ship the newest.
        fig.update_layout(title=dict(
            text=f"<b>{title}</b>", x=0, xanchor="left", y=0.97, yanchor="top",
            font=dict(color=INK, size=15),
        ))

    fig.update_xaxes(
        showgrid=False, linecolor=GRID, tickcolor=MUTED,
        tickfont=dict(color=MUTED),
        title_font=dict(color=INK_SECONDARY, size=11.5),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=SURFACE,
        tickfont=dict(color=MUTED),
        title_font=dict(color=INK_SECONDARY, size=11.5),
        automargin=True,
    )
    # Annotations (the vline labels on the VaR/ES markers) inherit nothing
    # useful either — but only fill in the ones that have not set their own
    # colour, so a caller can keep a label matched to the line it belongs to.
    for annotation in fig.layout.annotations:
        if annotation.font is None or annotation.font.color is None:
            annotation.font = dict(color=INK_SECONDARY, size=11)
    return fig


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading prices…")
def load_everything():
    cfg = load_config()
    panel = dataio.load_panel(cfg)
    returns = dataio.compute_returns(panel)
    results = pf.build_all_portfolios(returns, cfg)
    payload = {
        name: {
            "label": result.label,
            "returns": result.returns,
            "weights_daily": result.weights_daily,
            "weights_target": result.weights_target,
            "turnover": pf.annual_turnover(result, cfg.trading_days),
            "n_rebalances": result.meta["n_rebalances"],
        }
        for name, result in results.items()
    }
    return cfg, panel, returns, payload


@st.cache_data(show_spinner=False)
def rolling_backtest(_returns: pd.Series, window: int, confidence: float, method: str):
    result = bt.backtest_var(_returns, window, confidence, method)
    return (result.realised, result.forecasts, result.exceptions,
            result.summary_row(), bt.exceptions_by_year(result))


cfg, panel, returns, portfolios = load_everything()
holdings = [t for t in cfg.holdings if t in returns.columns]
names = ["equal_weight", "minimum_variance", "volatility_target", "benchmark"]
labels = {n: portfolios[n]["label"] for n in names}
value_default = float(cfg.risk.get("portfolio_value", 1_000_000))

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
selected = st.sidebar.selectbox(
    "Portfolio", names, index=2, format_func=lambda n: labels[n]
)
confidence = st.sidebar.select_slider(
    "Confidence level", options=[0.90, 0.95, 0.975, 0.99, 0.995], value=0.99,
    format_func=lambda v: f"{v:.1%}",
)
horizon = st.sidebar.select_slider(
    "VaR horizon (trading days)", options=[1, 5, 10, 21], value=1
)
portfolio_value = st.sidebar.number_input(
    "Portfolio value (USD)", min_value=10_000, max_value=1_000_000_000,
    value=int(value_default), step=100_000,
)
window = st.sidebar.slider(
    "Estimation window (days)", min_value=252, max_value=1512,
    value=int(cfg.portfolios["estimation_window"]), step=63,
)
st.sidebar.caption(
    f"Sample: {panel.index[0].date()} to {panel.index[-1].date()} · "
    f"{len(panel):,} trading days · {len(cfg.tickers)} instruments"
)
st.sidebar.warning(
    "Hypothetical research portfolios built on public price data. "
    "Not investment advice and not a forecast."
)

selected_returns = portfolios[selected]["returns"]
selected_weights = portfolios[selected]["weights_daily"].iloc[-1]
cov = pf.estimate_covariance(
    returns[holdings].iloc[-window:],
    method=cfg.portfolios.get("covariance_estimator", "ledoit_wolf"),
)

st.title("Multi-Asset Portfolio Risk & Stress Testing")
st.caption(
    "Three construction methodologies and a 60/40 benchmark, measured on the "
    "same data with the same engine."
)

tabs = st.tabs(
    ["Performance", "Risk decomposition", "Value at Risk", "Stress testing",
     "VaR backtest", "Allocation"]
)

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
with tabs[0]:
    risk_free = dataio.risk_free_daily(returns, cfg)
    market = returns[cfg.market_ticker]
    summary = pd.DataFrame(
        {
            labels[n]: mx.performance_summary(
                portfolios[n]["returns"], market=market, risk_free=risk_free,
                benchmark=portfolios["benchmark"]["returns"] if n != "benchmark" else None,
                trading_days=cfg.trading_days,
            )
            for n in names
        }
    ).T

    row = summary.loc[labels[selected]]
    cols = st.columns(5)
    cols[0].metric("Annualised return", f"{row['annualised_return']:.2%}")
    cols[1].metric("Annualised volatility", f"{row['annualised_volatility']:.2%}")
    cols[2].metric("Sharpe ratio", f"{row['sharpe_ratio']:.2f}")
    cols[3].metric("Max drawdown", f"{row['max_drawdown']:.2%}")
    cols[4].metric("Beta vs S&P 500", f"{row['beta']:.2f}")

    growth = go.Figure()
    for n in names:
        curve = (1 + portfolios[n]["returns"]).cumprod()
        growth.add_trace(go.Scatter(
            x=curve.index, y=curve.values, name=labels[n],
            line=dict(color=PORTFOLIO_COLORS[n], width=2,
                      dash="dash" if n == "benchmark" else "solid"),
        ))
    growth.update_yaxes(type="log", title="Growth of $1 (log scale)")
    st.plotly_chart(style(growth, 440, title="Growth of $1, net of costs"),
                    use_container_width=True, theme=None)

    left, right = st.columns(2)
    drawdown = go.Figure()
    for n in names:
        dd = mx.drawdown_series(portfolios[n]["returns"])
        drawdown.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=labels[n],
            line=dict(color=PORTFOLIO_COLORS[n], width=1.5,
                      dash="dash" if n == "benchmark" else "solid"),
        ))
    drawdown.update_yaxes(tickformat=".0%", title="Drawdown")
    left.plotly_chart(style(drawdown, 380, title="Drawdown from prior peak"),
                      use_container_width=True, theme=None)

    vol = go.Figure()
    for n in names:
        rolling = mx.rolling_volatility(
            portfolios[n]["returns"], int(cfg.risk["rolling_vol_window"])
        )
        vol.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values, name=labels[n],
            line=dict(color=PORTFOLIO_COLORS[n], width=1.5,
                      dash="dash" if n == "benchmark" else "solid"),
        ))
    vol.add_hline(
        y=float(cfg.portfolios["volatility_target"]["target_volatility"]),
        line=dict(color=MUTED, dash="dot"),
    )
    vol.update_yaxes(tickformat=".0%", title="Annualised volatility")
    right.plotly_chart(
        style(vol, 380, title=f"Rolling {cfg.risk['rolling_vol_window']}-day volatility"),
        use_container_width=True, theme=None
    )

    st.subheader("Performance and risk summary")
    display = summary[[
        "annualised_return", "annualised_volatility", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "max_drawdown", "beta",
        "skewness", "excess_kurtosis", "upside_capture", "downside_capture",
    ]]
    st.dataframe(
        display.style.format({
            "annualised_return": "{:.2%}", "annualised_volatility": "{:.2%}",
            "max_drawdown": "{:.2%}", "sharpe_ratio": "{:.2f}",
            "sortino_ratio": "{:.2f}", "calmar_ratio": "{:.2f}", "beta": "{:.2f}",
            "skewness": "{:.2f}", "excess_kurtosis": "{:.1f}",
            "upside_capture": "{:.2f}", "downside_capture": "{:.2f}",
        }),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"{labels[selected]}: where the risk actually sits")
    st.caption(
        "Capital weight answers 'how much did I buy'. Risk contribution answers "
        "'how much of my volatility does it cause'. The gap between them is the "
        "part a weights table cannot show you."
    )
    contributions = mx.risk_contributions(selected_weights, cov, cfg.trading_days)

    bars = go.Figure()
    bars.add_trace(go.Bar(
        y=contributions.index, x=contributions["weight"], name="Capital weight",
        orientation="h", marker_color=CATEGORICAL[0],
    ))
    bars.add_trace(go.Bar(
        y=contributions.index, x=contributions["pct_risk_contribution"],
        name="Risk contribution", orientation="h", marker_color=CATEGORICAL[1],
    ))
    bars.update_xaxes(tickformat=".0%")
    bars.update_layout(barmode="group", hovermode="y unified")
    st.plotly_chart(style(bars, 560, title="Capital weight versus risk contribution"),
                    use_container_width=True, theme=None)

    left, right = st.columns([1, 1])
    grouped = mx.group_risk_contributions(contributions, cfg.sleeve_map())
    left.markdown("**By sleeve**")
    left.dataframe(
        grouped.style.format({
            "weight": "{:.1%}", "component_contribution": "{:.2%}",
            "pct_risk_contribution": "{:.1%}", "risk_to_capital_ratio": "{:.2f}",
        }),
        use_container_width=True,
    )
    right.markdown("**Correlation matrix (estimation window)**")
    correlation = returns[holdings].iloc[-window:].corr()
    heat = go.Figure(go.Heatmap(
        z=correlation.values, x=correlation.columns, y=correlation.index,
        colorscale=[[0, "#0d366b"], [0.25, "#2a78d6"], [0.5, "#f0efec"],
                    [0.75, "#d03b3b"], [1, "#7d1f1f"]],
        zmid=0, zmin=-1, zmax=1, colorbar=dict(outlinewidth=0),
    ))
    right.plotly_chart(style(heat, 480, hovermode="closest"),
                       use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"{labels[selected]}: {horizon}-day VaR at {confidence:.1%}")
    methods = {
        "Historical": rk.historical_var(selected_returns, confidence, horizon),
        "Parametric normal": rk.parametric_var(
            selected_returns, confidence, horizon, "normal"),
        "Parametric Student-t": rk.parametric_var(
            selected_returns, confidence, horizon, "student_t"),
        "Cornish-Fisher": rk.cornish_fisher_var(selected_returns, confidence, horizon),
    }
    mc = rk.monte_carlo_var(
        returns[holdings], selected_weights, confidences=[confidence],
        horizon=horizon, n_simulations=50_000,
        distribution=cfg.risk["monte_carlo"]["distribution"],
        seed=int(cfg.risk["monte_carlo"]["seed"]),
    )
    methods["Monte Carlo"] = mc.var[confidence]

    cols = st.columns(len(methods))
    for col, (name, var) in zip(cols, methods.items()):
        # The currency figure goes in a caption rather than st.metric's delta
        # slot: a delta renders with an arrow, and an upward arrow beside a
        # loss reads as "risk improved" when it means the opposite.
        col.metric(name, f"{var:.2%}")
        col.caption(f"${var * portfolio_value:,.0f} loss")

    bar = go.Figure(go.Bar(
        x=list(methods), y=list(methods.values()),
        marker_color=CATEGORICAL[: len(methods)],
        text=[f"{v:.2%}" for v in methods.values()], textposition="outside",
    ))
    bar.update_yaxes(tickformat=".1%", title=f"{horizon}-day VaR")
    st.plotly_chart(
        style(bar, 380, hovermode="closest",
              title="Same portfolio, same data — the method moves the answer"),
        use_container_width=True, theme=None
    )

    left, right = st.columns(2)
    es_hist = rk.historical_expected_shortfall(selected_returns, confidence, horizon)
    var_hist = methods["Historical"]

    hist = go.Figure()
    hist.add_trace(go.Histogram(
        x=selected_returns.values, nbinsx=140, marker_color="#9ec5f4",
        name="Daily returns",
    ))
    # ES always sits to the left of VaR and the two lines are close together,
    # so their labels are stacked on different rows rather than left to
    # collide on the same one.
    hist.add_vline(x=-var_hist, line=dict(color=CRITICAL, width=2),
                   annotation_text=f"VaR {var_hist:.2%}",
                   annotation_position="top right",
                   annotation_font=dict(color=CRITICAL, size=11))
    hist.add_vline(x=-es_hist, line=dict(color="#7d1f1f", width=2, dash="dot"),
                   annotation_text=f"ES {es_hist:.2%}",
                   annotation_position="top left",
                   annotation_yshift=-18,
                   annotation_font=dict(color="#7d1f1f", size=11))
    hist.update_xaxes(tickformat=".1%", title="Daily return")
    left.plotly_chart(
        style(hist, 400, hovermode="closest", title="Historical return distribution"),
        use_container_width=True, theme=None
    )

    sim = go.Figure()
    sim.add_trace(go.Histogram(
        x=mc.simulated_returns, nbinsx=140, marker_color="#9ec5f4",
        name="Simulated",
    ))
    sim.add_vline(x=-mc.var[confidence], line=dict(color=CRITICAL, width=2),
                  annotation_text=f"VaR {mc.var[confidence]:.2%}",
                  annotation_position="top right",
                  annotation_font=dict(color=CRITICAL, size=11))
    sim.update_xaxes(tickformat=".1%", title=f"Simulated {horizon}-day return")
    right.plotly_chart(
        style(sim, 400, hovermode="closest",
              title=f"Monte Carlo — {mc.n_simulations:,} paths, "
                    f"Student-t (v={mc.dof:.1f})"),
        use_container_width=True, theme=None
    )

    st.subheader("Component VaR")
    st.caption(
        "VaR is homogeneous of degree one in the weights, so Euler's theorem "
        "splits it exactly across assets. These contributions sum to the total."
    )
    st.dataframe(
        rk.component_var(selected_weights, cov, confidence, horizon).style.format({
            "weight": "{:.2%}", "marginal_var": "{:.4f}",
            "component_var": "{:.4%}", "pct_var_contribution": "{:.1%}",
        }),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Historical episode replay")
    st.caption(
        "Today's weights, run through the actual returns of each crisis. The "
        "correlations and the sequencing are real, not assumed."
    )
    weights_map = {n: portfolios[n]["weights_daily"].iloc[-1] for n in names}
    historical = stx.historical_stress_table(returns[holdings], weights_map, cfg)

    pivot = historical.pivot(index="scenario", columns="portfolio",
                             values="total_return")
    episodes = go.Figure()
    for n in names:
        if n in pivot.columns:
            episodes.add_trace(go.Bar(
                y=pivot.index, x=pivot[n], name=labels[n], orientation="h",
                marker_color=PORTFOLIO_COLORS[n],
                text=[f"{v:.1%}" for v in pivot[n]], textposition="outside",
            ))
    # Extend the axis so the outside value labels on the longest bars are not
    # clipped at the plot edge.
    worst = float(np.nanmin(pivot.values))
    episodes.update_xaxes(tickformat=".0%", range=[worst * 1.22, 0.015])
    episodes.update_layout(barmode="group", hovermode="y unified")
    st.plotly_chart(style(episodes, 520, title="Total return through each episode"),
                    use_container_width=True, theme=None)

    st.subheader("Build your own scenario")
    st.caption(
        "Shocks are total returns on the factor proxy, not yield moves. A rise "
        "in interest rates is a **negative** duration shock: +200bp against "
        "TLT's ~17-year duration is roughly -30%."
    )
    betas = stx.estimate_factor_betas(returns[holdings], cfg)
    controls = st.columns(4)
    shocks = {
        "equity": controls[0].slider("Equity (SPY)", -0.60, 0.30, -0.30, 0.01),
        "duration": controls[1].slider("Duration (TLT)", -0.50, 0.40, -0.25, 0.01),
        "commodity": controls[2].slider("Commodity (DBC)", -0.60, 0.60, -0.20, 0.01),
        "real_estate": controls[3].slider("Real estate (VNQ)", -0.70, 0.30, -0.35, 0.01),
    }

    impacts = {}
    for n in names:
        asset_shocks = stx.apply_factor_scenario(weights_map[n], betas, shocks)
        aligned = weights_map[n].reindex(asset_shocks.index).fillna(0.0)
        impacts[n] = float((aligned * asset_shocks).sum())

    cols = st.columns(len(names))
    for col, n in zip(cols, names):
        col.metric(labels[n], f"{impacts[n]:.2%}")
        col.caption(f"${abs(impacts[n]) * portfolio_value:,.0f} loss")

    detail_shocks = stx.apply_factor_scenario(weights_map[selected], betas, shocks)
    aligned = weights_map[selected].reindex(detail_shocks.index).fillna(0.0)
    detail = pd.DataFrame({
        "weight": aligned,
        "asset_shock": detail_shocks,
        "loss_contribution": aligned * detail_shocks,
    }).sort_values("loss_contribution")

    waterfall = go.Figure(go.Bar(
        x=detail.index, y=detail["loss_contribution"],
        marker_color=[CRITICAL if v < 0 else CATEGORICAL[2]
                      for v in detail["loss_contribution"]],
        text=[f"{v:.2%}" for v in detail["loss_contribution"]],
        textposition="outside",
    ))
    waterfall.update_yaxes(tickformat=".1%", title="Contribution to scenario loss")
    st.plotly_chart(
        style(waterfall, 400, hovermode="closest",
              title=f"{labels[selected]}: who causes the loss"),
        use_container_width=True, theme=None
    )

    st.subheader("Correlation stress")
    st.caption(
        "Individual volatilities held fixed; only the correlations move. This "
        "isolates the failure mode where diversification disappears exactly "
        "when it is needed."
    )
    correlation_stress = stx.correlation_stress_impact(
        weights_map[selected], cov, [0.5, 0.75, 0.9, 0.95], cfg.trading_days
    )
    st.dataframe(
        correlation_stress.style.format({
            "annualised_volatility": "{:.2%}", "vol_multiple": "{:.2f}x",
        }),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# VaR backtest
# ---------------------------------------------------------------------------
with tabs[4]:
    method = st.radio(
        "VaR method", ["historical", "normal", "student_t", "cornish_fisher"],
        horizontal=True,
        format_func=lambda m: m.replace("_", " ").title(),
    )
    realised, forecast, exceptions, row, by_year = rolling_backtest(
        selected_returns, int(cfg.risk["backtest"]["rolling_window"]),
        confidence, method,
    )

    cols = st.columns(5)
    cols[0].metric("Observations", f"{row['n_observations']:,}")
    cols[1].metric("Exceptions", f"{row['n_exceptions']}",
                   f"expected {row['expected_exceptions']:.0f}", delta_color="off")
    cols[2].metric("Kupiec p-value", f"{row['kupiec_p_value']:.3f}",
                   "reject" if row["kupiec_reject"] else "accept", delta_color="off")
    cols[3].metric("Independence p-value", f"{row['independence_p_value']:.3f}",
                   "clustered" if row["independence_reject"] else "independent",
                   delta_color="off")
    cols[4].metric("Basel zone", str(row["basel_zone"]).split(" ")[0])

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=realised.index, y=realised.values, mode="markers", name="Daily return",
        marker=dict(size=2.5, color="#c3c2b7"),
    ))
    chart.add_trace(go.Scatter(
        x=forecast.index, y=-forecast.values, name=f"{confidence:.0%} VaR forecast",
        line=dict(color=CATEGORICAL[0], width=1.6),
    ))
    breaches = realised[exceptions == 1]
    chart.add_trace(go.Scatter(
        x=breaches.index, y=breaches.values, mode="markers",
        name=f"Exceptions ({len(breaches)})",
        marker=dict(size=7, color=CRITICAL, line=dict(width=1, color=SURFACE)),
    ))
    chart.update_yaxes(tickformat=".1%")
    st.plotly_chart(
        style(chart, 460, title="Realised returns against the VaR forecast"),
        use_container_width=True, theme=None
    )

    st.subheader("Exceptions by calendar year")
    st.caption(
        "The total count is only half the test. A model whose breaches all "
        "arrive in one year is failing even if the count looks right."
    )
    year_chart = go.Figure()
    year_chart.add_trace(go.Bar(
        x=by_year.index.astype(str), y=by_year["exceptions"], name="Exceptions",
        marker_color=[CRITICAL if a > 2 * o * (1 - confidence) else CATEGORICAL[0]
                      for a, o in zip(by_year["exceptions"], by_year["observations"])],
        text=by_year["exceptions"], textposition="outside",
    ))
    year_chart.add_trace(go.Scatter(
        x=by_year.index.astype(str), y=by_year["observations"] * (1 - confidence),
        name="Expected", line=dict(color=INK_SECONDARY, dash="dash", width=1.6),
    ))
    st.plotly_chart(style(year_chart, 380), use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"{labels[selected]}: allocation through time")
    daily = portfolios[selected]["weights_daily"]
    sleeve_map = cfg.sleeve_map()
    grouped = daily.T.groupby([sleeve_map.get(c, "other") for c in daily.columns]).sum().T

    area = go.Figure()
    for i, column in enumerate(grouped.columns):
        area.add_trace(go.Scatter(
            x=grouped.index, y=grouped[column], name=column.replace("_", " ").title(),
            stackgroup="one", mode="none",
            fillcolor=CATEGORICAL[i % len(CATEGORICAL)],
        ))
    area.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(style(area, 420), use_container_width=True, theme=None)

    cols = st.columns(3)
    cols[0].metric("Rebalances", portfolios[selected]["n_rebalances"])
    cols[1].metric("Annual turnover", f"{portfolios[selected]['turnover']:.1%}")
    cols[2].metric("Transaction cost",
                   f"{cfg.portfolios['transaction_cost_bps']:.0f} bps round trip")

    st.subheader("Current target weights")
    allocation = pd.DataFrame({
        "sleeve": [sleeve_map.get(t, "") for t in selected_weights.index],
        "sector": [cfg.sector_map().get(t, "") for t in selected_weights.index],
        "weight": selected_weights.values,
        "annualised_volatility": np.sqrt(
            np.diag(cov.loc[selected_weights.index, selected_weights.index])
            * cfg.trading_days
        ),
    }, index=selected_weights.index)
    allocation["pct_risk"] = mx.risk_contributions(
        selected_weights, cov, cfg.trading_days
    )["pct_risk_contribution"]
    st.dataframe(
        allocation.sort_values("weight", ascending=False).style.format({
            "weight": "{:.2%}", "annualised_volatility": "{:.1%}",
            "pct_risk": "{:.1%}",
        }),
        use_container_width=True,
    )

    st.subheader("Rebalance history")
    st.dataframe(
        portfolios[selected]["weights_target"].style.format("{:.2%}"),
        use_container_width=True,
    )
