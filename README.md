# Systematic Fund Explorer

**FINS5545 FinTech Project — Part B**
**Student ID:** z5727282

[Live Streamlit app](https://z5727282-projectb.streamlit.app/) · [GitHub repository](https://github.com/EdwarrdsLi/z5727282_projectB)

## Project overview

Systematic Fund Explorer is a prototype investment research application covering
50 US equities and 10 cryptocurrencies. It provides nine systematic funds across
three asset families and three portfolio methods:

| Asset family | Equal Weight | Risk Parity | Minimum Variance |
|---|---:|---:|---:|
| Equity only | Yes | Yes | Yes |
| Cryptocurrency only | Yes | Yes | Yes |
| Combined equity and cryptocurrency | Yes | Yes | Yes |

The project also builds a VADER headline-sentiment index for ten equity sectors
and tests an equity-only sentiment-tilt extension. The deployed app lets a user
compare funds, open fund fact sheets, inspect current target holdings, test a
hypothetical allocation, explore risk and drawdown, and review the sentiment
analysis and robustness checks.

All reported performance is historical walk-forward out-of-sample evidence. It
is not a forecast, trading signal, investment recommendation, or promise of
future returns.

## Main design

- Prices and headlines are loaded through `src/data_access.py` from the supplied
  hosted course dataset. Raw data is never committed to the repository.
- Returns use adjusted-close prices and are calculated separately on each native
  market calendar before any cross-asset alignment.
- Equity and combined funds use a rolling 252-observation estimation window and
  the equity trading calendar. Cryptocurrency-only funds use a rolling
  365-observation window and the native seven-day calendar.
- Portfolio targets are long-only, fully invested and unlevered. Optimised funds
  rebalance on the first observed trading date of each month.
- Risk Parity and Minimum Variance targets use only information available before
  the return period to which the targets apply.
- Equity headlines are aligned to the same or next observed equity trading date.
  The sector index equal-weights observed ticker-day VADER compound scores within
  each sector. A missing sector-day remains missing rather than being assigned a
  neutral score.
- The approved sentiment extension applies a `0.5` sector tilt to the Equity
  Equal Weight fund using the preceding observed equity day's sector index.
- Turnover, 0 and 10 basis-point illustrative one-way transaction-cost checks,
  and sentiment-tilt sensitivity results are saved under `results/tables/`.

## Historical results

The equity and combined evaluation period is 4 January 2021 to 29 December 2023
(753 observed equity trading days). The cryptocurrency-only evaluation period is
1 January 2021 to 31 December 2023 (1,095 calendar days). Metrics use a zero
annual risk-free rate and are gross of management fees and taxes.

| Fund | Annualised return | Annualised volatility | Sharpe ratio | Maximum drawdown | Growth of $1 |
|---|---:|---:|---:|---:|---:|
| Equity Equal Weight | 12.64% | 16.12% | 0.819 | -20.25% | $1.427 |
| Equity Risk Parity | 9.91% | 14.53% | 0.723 | -18.49% | $1.326 |
| Equity Minimum Variance | 5.50% | 12.67% | 0.486 | -15.28% | $1.174 |
| Cryptocurrency Equal Weight | 40.50% | 81.89% | 0.829 | -81.57% | $2.774 |
| Cryptocurrency Risk Parity | 44.30% | 79.90% | 0.862 | -79.88% | $3.005 |
| Cryptocurrency Minimum Variance | 37.39% | 65.24% | 0.814 | -74.56% | $2.593 |
| Combined Equal Weight | 15.14% | 21.60% | 0.761 | -27.87% | $1.524 |
| Combined Risk Parity | 13.97% | 16.20% | 0.888 | -19.47% | $1.478 |
| Combined Minimum Variance | 5.47% | 12.70% | 0.483 | -15.42% | $1.173 |

Within this historical sample, the approved sentiment fusion reduced the Equity
Equal Weight fund's annualised return from 12.64% to 12.28% and its volatility
from 16.12% to 16.08%. Its Sharpe ratio decreased from 0.819 to 0.801. This is a
descriptive negative result rather than evidence that sentiment has no value in
other samples or designs.

The complete machine-readable results are in
`results/tables/performance_metrics.csv` and
`results/tables/sentiment_fusion_comparison.csv`.

## App sections

The Streamlit app contains six sections:

1. **Compare funds** — compares all nine completed funds.
2. **Fact sheets** — displays verified metrics, growth and current target holdings.
3. **Allocation lab** — applies a hypothetical allocation to saved historical
   out-of-sample growth paths.
4. **Fund analytics** — presents growth, drawdown, risk-return and target-weight
   figures.
5. **Sentiment & fusion** — explores the ten-sector index, approved fusion result,
   transaction-cost check and tilt-strength sensitivity.
6. **Method notes** — states the evidence period, implementation choices and
   limitations.

The deployed app reads committed, precomputed files from `results/`. It does not
download raw data, fit portfolios, or run VADER at page load.

## Repository structure

```text
streamlit_app.py       Streamlit application entry point
.streamlit/            deployment and theme configuration
src/                   data preparation, portfolios, sentiment and fusion logic
scripts/               reproducible build and hand-in checks
tests/                 deterministic unit and integration tests
results/data/          precomputed app-readable data
results/tables/        metrics, holdings, turnover and robustness tables
results/figures/       report and app figures
report/                editable report source and submitted PDF
ai/                    AI-use records submitted for assessment
context/               supplied course context and data guide
```

## Environment setup

Streamlit Community Cloud is configured to use Python 3.13. A local Python 3.13
environment is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install both the application and reproduction dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` contains the lightweight deployment dependencies.
`requirements-dev.txt` adds NLTK for rebuilding the VADER sentiment artifacts.
The first complete rebuild may download the VADER lexicon and the supplied course
data bundle.

## Reproduce the results

From the repository root, run:

```bash
python scripts/run_part_b.py
```

This rebuilds the derived CSV files, tables and figures under `results/`. To use a
local copy of the supplied data ZIP instead of the hosted source, set
`FINS_DATA_ZIP` to that ZIP before running the command.

## Run the app locally

```bash
streamlit run streamlit_app.py
```

## Tests and hand-in checks

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run the mechanical submission checks:

```bash
python scripts/check_handin.py
```

At the latest verified project audit, all 42 tests and all 22 mechanical checks
passed. Re-run both commands after any change before submission.

## Important limitations

- The evidence ends on 31 December 2023 at the latest and covers one historical
  market period; no later performance is claimed.
- Backtests are sensitive to the selected universe, estimation window,
  rebalance rule, constraints and transaction-cost assumptions.
- Saved headline data contains titles rather than full articles. VADER is a
  general-purpose lexicon model and can miss finance-specific meaning, irony and
  context.
- Sector-days with no observed headlines remain missing, so sentiment coverage is
  uneven across sectors and dates.
- The allocation lab replays saved gross historical growth paths. It does not
  model fees, taxes, cash flows, investor-level rebalancing or live execution.
- Cryptocurrency results show high historical returns alongside very high
  volatility and drawdowns. Historical results should not be interpreted as
  expected returns.

## Deployment and submission

- Live app: <https://z5727282-projectb.streamlit.app/>
- Public repository at hand-in:
  <https://github.com/EdwarrdsLi/z5727282_projectB>
- Streamlit entry point: `streamlit_app.py`
- Cloud Python version: 3.13

The final submission also requires the complete project ZIP and the PDF report as
specified in `SUBMISSION_CHECKLIST.md` and `PROJECT_BRIEF.md`.
