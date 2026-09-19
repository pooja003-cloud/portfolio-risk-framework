# Methodology

Every modelling choice that affects a published number is recorded here and, where it is a
parameter, lives in `config/config.yaml` rather than in code. The aim is that a reader can
disagree with a specific decision rather than having to guess what was decided.

---

## 1. Data

| Item | Value |
|---|---|
| Source | Yahoo Finance daily bars |
| Price type | Split- and dividend-adjusted close |
| Frequency | Daily |
| Start | 2007-06-01 |
| End | 2026-09-17 |
| Observations | 4,855 trading days |
| Instruments | 15 |

**Why this start date.** The sample window is constrained by the youngest instrument in the
universe: BIL, the 1–3 month T-bill ETF, first traded on 2007-05-30. Beginning in June 2007
keeps the entire Global Financial Crisis inside the sample, which matters more than it might
appear — a risk framework whose worst observed episode is COVID has never been shown a
slow-moving credit crisis, and the two look nothing alike.

**Universe.** Fifteen instruments across five sleeves:

| Sleeve | Instruments |
|---|---|
| Equity (single names) | AAPL, MSFT, JNJ, JPM, XOM, PG, CAT, NEE — one per sector bucket |
| Broad equity | SPY (also the market proxy for beta) |
| Fixed income | AGG (core), TLT (long duration) |
| Commodity | DBC (broad), GLD (precious metals) |
| Real assets | VNQ (US REITs) |
| Cash | BIL (also the risk-free proxy) |

Eight individual equities rather than an all-ETF universe, so that single-name idiosyncratic
risk is present and visible in the risk-contribution analysis. Two instruments do double
duty: SPY is both a holding and the market proxy, and BIL is both the cash sleeve and the
risk-free rate.

**Risk-free rate.** The daily total return of BIL, rather than a quoted T-bill yield. It is
observable, it matches the daily frequency of the portfolio returns exactly, and it keeps
the Sharpe ratio internally consistent with the cash sleeve actually held in the
volatility-targeted portfolio.

### Missing-data treatment

1. The **trading calendar** is the set of dates on which SPY traded, restricted to the sample
   window. Using the union of all tickers' dates would create phantom zero-return days from
   foreign and bond-market holidays.
2. Interior gaps are **forward-filled** for at most 3 sessions. A stale price implies a zero
   return, which is the economically correct treatment of a non-trading day.
3. Any date still missing a price for any ticker is **dropped**.
4. **No back-filling, ever.** Back-filling uses tomorrow's price to fill today's gap, which
   leaks future information into every downstream calculation.

On the shipped snapshot, steps 2 and 3 are no-ops: the panel is complete. The data-quality
report (`outputs/tables/data_quality_report.csv`) records this per ticker, including
`n_zero_returns` — BIL shows ~1,800 zero-return days out of 4,855, which is not an error but
a reminder that the cash sleeve's measured volatility is biased downward by price
discreteness.

### Reproducibility

Vendors revise history. The repository therefore ships a snapshot of the consolidated
adjusted-close panel at `data/raw/prices_snapshot.csv`, and the loader reads it in preference
to hitting the network. A fresh clone reproduces every published figure offline and exactly.
`python -m prisk.pipeline fetch --force` re-downloads from Yahoo Finance and rewrites the
caches, which is the right thing to do when extending the sample.

---

## 2. Portfolio construction

All four portfolios run through the same engine, so differences between them come from the
allocation rule alone.

### Common settings

| Parameter | Value | Rationale |
|---|---|---|
| Rebalance frequency | Quarterly | Balances estimation freshness against turnover |
| Estimation window | 756 days (~3y) | Long enough to estimate 15×15 covariance; short enough to reflect regime |
| Minimum window | 252 days | No optimised weight is formed before this |
| Transaction cost | 5bp round trip | Charged on turnover, on the day of the trade |
| Covariance estimator | Ledoit-Wolf shrinkage | See below |

### Covariance estimation

The minimum-variance optimiser concentrates weight in the directions with the smallest
estimated variance — precisely where a sample covariance matrix is least reliable. Ledoit-Wolf
shrinkage pulls the estimate towards a scaled identity, lifting the smallest eigenvalues and
lowering the largest. Those small eigenvalues are exactly the near-arbitrage directions that
do not survive out of sample.

**Shrinkage is applied to the correlation matrix, not to the covariance matrix.** This
distinction is easy to miss and it matters a great deal in a multi-asset universe. Shrinking
the covariance matrix directly pulls every *variance* towards the average variance — and the
variances here span four orders of magnitude, from a T-bill ETF at 0.2% annualised volatility
to an industrial equity at 32%. Applied naively, shrinkage reported the cash sleeve at
roughly 4.8% volatility, nearly twenty times its true value, and handed that number to the
optimiser and to every risk-contribution table.

Standardising the returns, shrinking the correlation matrix, and rescaling by the sample
standard deviations preserves each asset's own volatility *exactly* while regularising the
correlation structure, which is where the estimation noise actually lives. On a 756-day
window this cuts the correlation matrix's condition number from about 62 to about 29 and
roughly doubles its smallest eigenvalue, while the reported volatilities match the realised
ones to machine precision. Two unit tests guard both halves of that claim.

### The four rules

**Equal weight (1/N).** Every holding gets the same capital, reset quarterly. It estimates
nothing, so it cannot be wrong about anything, and it is the control against which the
optimised portfolios must justify themselves. Note that 1/N is *not* a neutral allocation —
it is a bet on whatever the universe happens to contain, and with eight of fifteen slots in
single equities it is structurally equity-heavy.

**Minimum variance.** Solves

> minimise `w'Σw` subject to `Σw = 1` and `0 ≤ wᵢ ≤ 0.25`

on the shrunk covariance matrix. Solved with CVXPY when available (the problem is a convex
QP, so the solution is global) and SciPy SLSQP otherwise; a unit test asserts the two agree.

The 25% box constraint is doing real work. Unconstrained, the solution puts almost everything
in the cash proxy — technically the minimum-variance answer and practically useless. The cap
forces the optimiser to diversify rather than simply de-risk.

**Volatility target (8%).** The growth sleeve is every holding except the cash proxy,
allocated by **equal risk contribution** (solved numerically by minimising the dispersion of
percentage risk contributions, starting from inverse-volatility weights). Let `σ_g` be the
sleeve's forecast annualised volatility and `σ* = 8%` the target. The sleeve is held at
`k = clip(σ*/σ_g, 0.10, 1.00)` with `1 − k` in cash.

The `k ≤ 1` ceiling means **no leverage**, so in calm regimes realised volatility sits below
target — a deliberate, documented asymmetry. As of the latest rebalance the constraint binds:
forecast sleeve volatility is below 8%, so the portfolio is fully invested with a zero cash
weight.

**Benchmark.** Static 60% SPY / 40% AGG, rebalanced quarterly.

### No-look-ahead discipline

Weights applied from rebalance date *t* onwards are estimated **only** from returns up to and
including *t*. Between rebalance dates weights drift with realised performance; they are
reset at the next rebalance. Turnover is measured against the drifted weights.

This is enforced, not merely intended. `tests/test_portfolios.py::test_no_look_ahead`
multiplies the last 300 days of the return panel by ten, re-runs the backtest, and asserts
that every target weight formed before that point is bit-for-bit identical.

One subtlety: resampling an incomplete trailing quarter returns its last available date,
which for a sample ending mid-quarter is simply "today". Trading on it would book turnover
and costs for a position held for zero days, so a rebalance falling on the final observation
is dropped.

---

## 3. Performance and risk metrics

All metrics take daily **simple** returns (which aggregate correctly across assets) and
annualise on a 252-day convention. Return statistics are geometric (CAGR), because a
compounded figure is what an investor actually earns.

| Metric | Definition / note |
|---|---|
| Annualised return | Geometric: `(Π(1+r))^(252/n) − 1` |
| Annualised volatility | `std(r, ddof=1) × √252` |
| Sharpe ratio | Excess returns formed **daily** against BIL, then annualised |
| Sortino ratio | Excess return over downside deviation; shortfalls averaged over *all* observations, not only negative ones |
| Calmar ratio | Annualised return ÷ \|max drawdown\| |
| Beta | `cov(rₚ−r_f, r_m−r_f) / var(r_m−r_f)` |
| Jensen's alpha | Annualised CAPM intercept |
| Tracking error / IR | Against the 60/40 benchmark |
| Capture ratios | **Mean** ratio on up/down benchmark days — see below |
| Ulcer index | RMS drawdown: depth *and* duration |

**Drawdown.** The running peak is floored at the starting value of 1.0. Without that floor a
drawdown beginning on the very first day of the sample is invisible: the equity curve starts
below 1.0, the running maximum starts with it, and the loss is silently rebased away.
Flooring at initial capital treats the starting point as a peak, which is what an investor
who funded the account on day one experiences. (This was a real bug, caught by
`test_drawdown_captures_a_loss_on_the_first_day`.)

**Capture ratios** use the *mean* return on up/down benchmark days, not the compounded total.
Compounding a sub-sample of several thousand non-consecutive up-days produces an
astronomically large number on both legs, and the ratio is then dominated by floating-point
scale rather than by actual participation — it collapses towards zero for any low-beta
portfolio. (Also a real bug, caught before publication.)

**Risk contribution.** For portfolio volatility `σₚ = √(w'Σw)`, Euler's theorem gives an
exact additive decomposition:

- marginal contribution `MCTRᵢ = (Σw)ᵢ / σₚ`
- component contribution `CTRᵢ = wᵢ · MCTRᵢ` — these sum to `σₚ` exactly
- percentage contribution `CTRᵢ / σₚ`

Aggregated to sleeve level, `risk_to_capital_ratio = pct_risk / weight` is the diagnostic:
above 1.0 means the sleeve consumes more risk budget than capital.

---

## 4. Value at Risk and Expected Shortfall

**Sign convention.** VaR and ES are reported as **positive numbers representing losses**. A
1-day 99% VaR of 2.45% means: on 99 days out of 100, the portfolio is not expected to lose
more than 2.45% in a day.

**Estimation samples.** Every method is computed on the *same* data, otherwise the comparison
measures the sample rather than the method. Two samples are reported side by side: the full
history (unconditional risk) and the trailing 756-day window (risk in the current regime).

### Historical

The `1 − c` empirical quantile of realised returns. Multi-day horizons are scaled by `√h`;
`historical_var_overlapping` provides the assumption-free alternative on overlapping
compounded returns.

*Strengths:* no distributional assumption; inherits the real fat tails, skew and clustering.
*Weaknesses:* can only produce losses that have already happened; weights a 2008 observation
exactly like yesterday's.

### Parametric

**Normal:** `VaR = −(μh + z₍₁₋c₎ σ√h)`.

**Student-t:** a t is fitted by maximum likelihood and the VaR read off the **fitted location
and scale**, not off the sample standard deviation. This distinction matters: MLE down-weights
outliers when estimating the scale and pushes the fat tail into a low ν instead. Re-imposing
the sample σ on top of a low-ν quantile double-counts the scale and produces a VaR far too
small — the naive version breached 11% of the time at a 95% confidence level in testing.

Note that a Student-t does **not** always give a larger VaR than a normal. Standardised for
scale, the t has thinner shoulders and a fatter extreme tail; the crossover sits around the
97th–99th percentile. At 95% the t typically sits *below* the normal, at 99% above it.

**Cornish-Fisher:** adjusts the normal quantile for sample skewness `S` and excess kurtosis
`K`:

> `z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36`

**This method is reported and then rejected for this data.** It is a third-order correction
valid for *mild* departures from normality. With excess kurtosis near 12 the correction term
dominates and the implied quantile function stops being monotonic; it returns 4.16% where
the empirical estimate is 2.45%. That is breakdown, not conservatism. It is kept in the
report because knowing *which* methods fail on your data is part of the result.

### Monte Carlo

1. Estimate the mean vector and (shrunk) covariance matrix of asset returns.
2. Draw `n` paths of `h` daily joint return vectors from a multivariate normal or
   multivariate Student-t built from the Cholesky factor of Σ. The Student-t is generated as
   a **normal-variance mixture**, `x = μ + L·z·√(ν/χ²_ν)`, which preserves the correlation
   structure while producing *joint* tail events far more often than a Gaussian copula would.
   Assets crash together; a Gaussian copula does not know that.
3. Compound each path at **fixed weights** and read quantiles off the resulting distribution.

Fixed weights are the correct convention for a risk measure: the question is "what could this
portfolio lose", not "what could a rebalancing strategy lose".

Default: 100,000 paths, Student-t, seed 20260919 (fixed for reproducibility).

**On the degrees of freedom.** Maximum likelihood on a long daily series routinely returns
ν < 3, which the implementation floors at 3.0. That is the estimator saying something real:
a single Student-t cannot describe nineteen years of daily returns, because the data is a
*mixture* of volatility regimes rather than draws from one fat-tailed distribution. Below
ν = 3 the fourth moment is infinite and simulated kurtosis runs into the hundreds — a
property of the estimator, not of the portfolio. The floor is a documented modelling choice;
a GARCH-based simulation is the principled alternative.

### Expected Shortfall

The average loss conditional on breaching VaR, computed empirically and in closed form for
both the normal and Student-t cases. ES is **coherent** — in particular sub-additive, so
diversification can never increase it. VaR carries no such guarantee.

### Component VaR

VaR is homogeneous of degree one in the weights, so Euler's theorem splits it exactly:
`VaR = Σᵢ wᵢ · ∂VaR/∂wᵢ`. The `component_var` column sums to the portfolio VaR, making it a
genuine risk budget rather than a ranking.

---

## 5. Stress testing

Two complementary families, because they answer different questions. **Neither is a
forecast.**

### Historical episode replay

Current weights, run through the actual asset returns of a named crisis. *If the GFC happened
again to the book I hold today, what would it cost?*

*Strength:* internal consistency — correlations, the volatility path and the sequencing are
all real, not assumed. *Weakness:* can only replay crises that have already happened.

Weights are allowed to **drift** during each episode rather than being rebalanced daily.
Daily rebalancing through a crash mechanically buys the fallers and flatters the result, and
it is not what happens to a portfolio nobody trades.

Episodes: GFC (2007-10-09 → 2009-03-09), Euro crisis / US downgrade (2011), China
devaluation / oil collapse (2015–16), COVID-19 (2020-02-19 → 2020-03-23), 2022 inflation and
rate shock, 2018 volatility spike.

### Hypothetical factor shocks

Asset sensitivities come from a **multivariate** OLS regression of each asset's daily return
on four factor proxies jointly:

> `rᵢ,ₜ = αᵢ + Σₖ βᵢ,ₖ · fₖ,ₜ + εᵢ,ₜ`

Multivariate rather than univariate matters: SPY and VNQ are highly correlated, so univariate
betas would double-count the equity shock. By construction each proxy loads 1.0 on its own
factor and 0.0 on the others, which is what makes "a −20% equity shock" mean exactly that.

| Factor | Proxy |
|---|---|
| equity | SPY |
| duration | TLT |
| commodity | DBC |
| real_estate | VNQ |

**Sign convention — the thing most likely to be misread.** Every shock is a **total return on
the factor proxy**, not a move in a yield. The `duration` factor is the return on 20y+
Treasuries, so **a rise in interest rates is a negative duration shock**. A +200bp parallel
shift against TLT's ~17-year effective duration implies roughly −30%; the regression beta
(~0.35 for AGG) then delivers about −11% on core bonds without that being asserted anywhere.

*Strength:* can express a scenario with no historical precedent. *Weakness:* the transmission
relies on betas estimated in normal times, which is exactly when they are least reliable.

### Correlation stress

Rebuilds the covariance matrix with every pairwise correlation forced to a target level,
holding individual volatilities **fixed**. This isolates the single most damaging feature of
a real crisis: diversification disappearing exactly when it is needed. Scenarios carrying a
`correlation_override` in the config are also reported with their stressed volatility.

---

## 6. VaR backtesting

A VaR model is a falsifiable forecast: at 99% confidence, losses should exceed the one-day
VaR on about 1% of days.

**Forecasts.** Rolling 500-day window (Basel-style), one-day-ahead. The forecast for day *t*
uses **only** returns up to *t−1*, enforced by
`tests/test_backtest.py::test_rolling_forecast_has_no_look_ahead`. For the Student-t method
the shape (ν and the fitted-scale-to-sample-σ ratio) is refitted every 63 days while the
rolling standard deviation updates daily — a one-day change in a 500-day window moves the MLE
negligibly, and daily refitting makes a twenty-year backtest an order of magnitude slower for
no gain.

**Two things are tested, and both matter.**

*Unconditional coverage* — is the number of exceptions right? Kupiec's proportion-of-failures
LR test, χ²(1). Note that **too few** exceptions is also a failure: it means the model
overstates risk and wastes capital.

*Independence* — are exceptions spread out? Christoffersen's Markov test on the transition
counts between the no-exception and exception states, χ²(1). A model can have exactly the
right exception count and still be useless if every breach arrives in the same fortnight,
because that is precisely when the capital is needed.

The two combine into the conditional-coverage test, χ²(2).

**Basel traffic light.** Over 250 observations: 0–4 exceptions green, 5–9 yellow (capital
multiplier applies), 10+ red. Counts are scaled proportionally for longer samples. The zones
are defined for **99% models only** — applying them at 95%, where the expected count over 250
days is 12.5, would label every correctly calibrated model red — so other confidence levels
return `n/a`.

**Breach severity.** The mean ratio of realised loss to forecast VaR on exception days. A
value well above 1 says that when the model is wrong, it is badly wrong.

---

## 7. Known limitations

- **No conditional volatility model (GARCH/EWMA).** The largest single gap, and what every
  failed independence test is pointing at.
- **Survivorship bias.** The universe is a fixed set chosen with hindsight in 2026.
- **One rate-shock regime** in the sample; conclusions about duration risk rest on it.
- **Stress betas are estimated in calm conditions** and applied to stressed ones.
- **Flat 5bp costs**, no market impact or liquidity modelling.
- **Monte Carlo ν is floored at 3**, papering over a regime mixture.
- **In-sample optimisation of the *rules*.** The rebalance frequency, estimation window and
  box constraint were not themselves tuned, but they were chosen by a person who knows what
  happened in 2008, 2020 and 2022.
