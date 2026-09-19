# Investment Risk Memorandum

**Subject:** Multi-Asset Portfolio Risk and Stress-Testing Framework — findings
**Author:** pooja003-cloud
**Date:** 19 September 2026
**Sample:** 1 June 2007 – 17 September 2026 · 4,855 trading days · 15 instruments
**Notional for currency figures:** $1,000,000

> **Disclaimer.** The portfolios described here are *hypothetical research constructs* built
> from publicly available price data. Nothing in this memo represents a real client account,
> real assets under management, investment advice, or a forecast. Stress scenarios are
> analytical assumptions chosen to probe the portfolios, not predictions about markets.

---

## 1. Question

How does portfolio construction affect return, volatility, drawdown and downside risk under
normal and stressed market conditions — and do the standard risk models actually hold up
when they are tested?

Three portfolios were built from one universe of 15 instruments (eight equities across
sectors, a broad equity ETF, two bond ETFs, two commodity ETFs, a REIT ETF and a T-bill
proxy), rebalanced quarterly, net of 5bp round-trip costs, against a static 60/40 benchmark.
The construction rules are ordered by how much estimation they require, so that the
assumption-free equal-weight portfolio acts as a control: **equal weight** (estimates
nothing), **minimum variance** (estimates a covariance matrix, long-only, 25% position cap)
and **volatility target** (an 8% target on an equal-risk-contribution growth sleeve, no
leverage, remainder in cash).

## 2. Headline results

| Portfolio | Ann. return | Ann. vol | Sharpe | Sortino | Max drawdown | Beta | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equal Weight | 11.87% | 13.08% | **0.82** | 1.18 | −33.4% | 0.62 | 16.6% |
| Minimum Variance | 5.55% | 5.10% | **0.83** | 1.18 | −12.9% | 0.11 | 18.0% |
| Volatility Target 8% | 7.55% | 7.41% | **0.84** | 1.19 | −15.7% | 0.30 | 29.0% |
| 60/40 Benchmark | 8.82% | 11.83% | 0.66 | 0.94 | −29.8% | 0.59 | 9.5% |

**Construction relocated risk; it did not create return.** All three research portfolios
produced Sharpe ratios within 0.02 of each other (0.82 to 0.84), across a volatility range of
5.1% to 13.1% — a 2.6× spread. Each beat the 60/40 benchmark's 0.66 by a quarter or more. The benchmark is
the clear outlier: it accepted near-equal-weight drawdown (−29.8%) for materially worse
risk-adjusted performance.

The capture ratios show the mechanism. Minimum variance participated in 25% of the
benchmark's up days and only 19% of its down days. Giving up less on the downside than you
surrender on the upside is the entire economic case for a defensive allocation, and it is
what the Sharpe ratio is summarising.

**Capital weights are not risk weights.** In the equal-weighted portfolio every holding is
rebalanced to 6.7% of capital, but risk contributions range from 13.4% (the largest-cap
technology name) to effectively zero (the cash proxy). At sleeve level the equity sleeve
holds 54% of capital and produces 69% of the volatility — a risk-to-capital ratio of 1.28 —
while fixed income holds 12% and produces 3.7%.

## 3. Downside risk

One-day 99% figures, full sample, as a percentage of portfolio value and in dollars:

| Portfolio | Historical VaR | Parametric normal | Student-t | Monte Carlo | **Expected shortfall** | ES on $1m |
|---|---:|---:|---:|---:|---:|---:|
| Equal Weight | 2.45% | 1.87% | 2.32% | 2.15% | **3.51%** | $35,100 |
| Minimum Variance | 0.85% | 0.73% | 0.81% | 1.05% | **1.22%** | $12,200 |
| Volatility Target 8% | 1.22% | 1.06% | 1.19% | 1.79% | **1.97%** | $19,700 |
| 60/40 Benchmark | 2.23% | 1.70% | 2.08% | 1.97% | **3.21%** | $32,100 |

**The normal model understates the tail by roughly a quarter.** For the equal-weight
portfolio it reports 1.87% where the empirical distribution says 2.45%. This is a direct
consequence of the data: excess kurtosis runs between 10.3 and 19.5 across the four
portfolios, and Jarque-Bera rejects normality with p-values indistinguishable from zero.
Under a normal distribution, the equal-weight portfolio's worst observed day is an event
that should occur roughly once in the age of the universe. It occurred in March 2020.

**Expected shortfall is the number to manage against.** The ES/VaR ratio runs 1.43 to 1.62
against a normal-distribution benchmark of about 1.15 — when these portfolios breach, the
average breach is far worse than the threshold. ES is also sub-additive, so unlike VaR it
can never penalise diversification.

**Cornish-Fisher was computed and rejected.** It returned 4.16% for the equal-weight
portfolio against a historical 2.45%. That is not conservatism, it is breakdown: the
expansion is a third-order correction valid for *mild* non-normality, and at this level of
kurtosis the correction term dominates. The method is reported with that caveat rather than
quietly dropped.

## 4. Stress testing

**Historical episode replay** — current weights run through the actual returns of each crisis:

| Episode | Equal Weight | Min Variance | Vol Target | 60/40 |
|---|---:|---:|---:|---:|
| Global Financial Crisis (2007–09) | −29.5% | **−11.2%** | −22.5% | −30.6% |
| COVID-19 crash (Feb–Mar 2020) | −23.9% | **−10.9%** | −20.1% | −20.9% |
| 2022 inflation and rate shock | −10.1% | **−7.0%** | −9.9% | **−20.5%** |
| Euro crisis / US downgrade (2011) | −6.5% | −1.3% | −3.8% | −9.2% |
| 2018 volatility spike | −10.8% | −4.5% | −8.5% | −10.7% |

**Hypothetical factor shocks** (multivariate betas; all figures hypothetical assumptions):

| Scenario | Equal Weight | Min Variance | Vol Target | 60/40 |
|---|---:|---:|---:|---:|
| Equity bear market (−20% equities) | −12.1% | −4.5% | −9.6% | −12.2% |
| Rate shock (+200bp parallel) | −7.3% | −5.7% | −8.1% | −8.4% |
| Stagflation (equities −25%, duration −20%) | −14.2% | −5.3% | −12.1% | −18.0% |
| Severe combined stress | −22.8% | −13.3% | −21.1% | −21.9% |

Three findings carry the section.

**2022 is the episode that breaks the usual ranking.** In every equity-driven crisis the
60/40 benchmark performs in line with the equal-weighted portfolio. In 2022 it lost twice as
much (−20.5% against −10.1%), because bonds and equities fell together. The rolling
one-year SPY/AGG correlation — the assumption the entire 60/40 construction rests on —
spends most of the sample negative and turned decisively positive during the inflation
shock. A balanced portfolio is a bet on that correlation, and the bet does not always pay.

**Reducing one risk substitutes another.** The minimum-variance portfolio's worst drawdown
was not the Global Financial Crisis. It was **2022**, at −12.9%, taking 395 trading days to
recover. By retreating into bonds and cash it largely sidestepped equity crises and
concentrated itself into duration — precisely what the rate shock attacked. It is the best
performer in an equity bear market and, relative to its own volatility, among the worst in a
rate shock.

**Diversification is a borrowed defence.** Holding individual volatilities fixed and forcing
every pairwise correlation to 0.95 raises the volatility-targeted portfolio's annualised
volatility from 7.6% to 17.0% — a **2.23× increase from correlation alone**. The portfolios
that look safest under observed correlations have the most to lose when correlations
converge, because more of their apparent safety comes from diversification than from holding
genuinely low-volatility assets.

## 5. Model validation

Rolling 500-day VaR forecasts, one-day-ahead, no look-ahead, tested over ~4,080 observations
per portfolio at 99% confidence (expected exceptions: 40.8).

| Method | Exceptions (Equal Weight) | Kupiec p | Independence p | Verdict |
|---|---:|---:|---:|---|
| Historical | 48 | 0.27 | **0.000** | Passes count, fails clustering |
| Parametric normal | **78** | **0.000** | **0.000** | **Rejected** |
| Student-t | 43 | 0.74 | **0.000** | Passes count, fails clustering |
| Cornish-Fisher | 29 | 0.05 | **0.000** | Over-conservative |

**Parametric-normal VaR is rejected at 99% for every portfolio**, breaching roughly twice as
often as it promises (78 against 40.8 for equal weight; 91 for the 60/40 benchmark, which
lands it in the Basel yellow zone). This turns "fat tails matter" from an assertion into a
rejected null hypothesis.

**Every method fails the independence test, without exception.** Exceptions cluster: 2020
and 2022 absorb a disproportionate share of all breaches. None of the four methods models
volatility clustering — all assume the distribution is stable across the estimation window.
A model that is right on average and wrong when it matters cannot be used to size positions.
Average breach severity runs 1.35–1.74 across all methods: when they are wrong, the realised
loss is 50–70% larger than the forecast.

The failure mode is visible in the forecast path itself. Historical VaR is a step function
that stays flat for long stretches and then jumps *after* a crisis has already happened. In
February 2020 it sat at its calm-regime level while the portfolio lost multiples of it on
consecutive days, then reset upward — by which point the danger had passed, leaving the
model too conservative for the following two years.

## 6. Limitations

- **No conditional volatility model.** This is the largest gap and it is what every failed
  independence test is pointing at. GARCH or EWMA-weighted historical simulation is the
  clearest next step.
- **Survivorship bias.** The universe is a fixed set chosen with hindsight in 2026.
- **One rate-shock regime.** Conclusions about duration risk rest on a single episode.
- **Stress betas are estimated in normal conditions** and applied to abnormal ones — exactly
  when they are least reliable.
- **Costs are a flat 5bp** with no market-impact or liquidity modelling. At 25bp the ranking
  of the two optimised portfolios would begin to tighten.
- **The Monte Carlo Student-t fit floors ν at 3.** The unfloored fit indicates a mixture of
  volatility regimes rather than a single fat-tailed distribution.

## 7. Conclusions

1. Over this sample, construction methodology moved risk far more than it moved
   risk-adjusted return. All three research portfolios delivered 0.82–0.84 Sharpe at volatilities
   from 5.1% to 13.1%; the choice between them is a choice of risk level, not of skill.
2. All three beat a static 60/40 on both Sharpe and drawdown, and the benchmark's weakness
   was concentrated in the one regime where its core diversification assumption failed.
3. Normal-distribution VaR should not be used on this data. The empirical and Student-t
   estimates are defensible on count; none of them is defensible on clustering.
4. Expected shortfall, not VaR, is the appropriate headline tail measure here: it is coherent
   and it reflects a breach distribution that is 40%+ worse than the breach threshold.
5. Any risk statement from this framework should carry the regime it was estimated in.
   Current-regime VaR sits well below the full-sample figure — which is how risk systems
   quietly understate exposure in calm periods, immediately before it matters.

---

*Full methodology: `docs/methodology.md`. All figures reproducible via `python -m prisk.pipeline`.
Source data: Yahoo Finance adjusted daily closes. 119 unit tests cover the calculations,
including explicit no-look-ahead checks on both the rebalancing engine and the VaR backtest.*
