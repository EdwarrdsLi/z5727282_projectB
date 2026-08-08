# AI workflow log v2.04 - R4 sentiment fusion

## Phase and date

- Phase: R4 look-ahead-safe equity sentiment fusion
- Date: 2026-08-08 (Australia/Sydney)
- Tool: Codex

## Student request and boundaries

The student asked Codex to complete R4 sentiment fusion after reading the full
project brief, constitution, README, every supplied context file, and the R0-R3
AI logs, and after inspecting the existing portfolio and sentiment outputs. The
student required Codex to:

- combine the equity sector sentiment index with an existing equity strategy;
- lag sentiment by at least one observed equity trading day and prevent all
  look-ahead;
- never apply equity-news sentiment to cryptocurrencies;
- preserve the R2 outputs and save fusion evidence separately;
- create a CSV table and figure comparing the strategy before and after fusion,
  with suitable metrics and an explicit sample;
- avoid unsupported improvement claims, forecasts, recommendations, trading
  advice, report interpretation, and final report wording;
- add focused timing, scope, schema, and reproducibility tests;
- integrate R4 into `scripts/run_part_b.py`, create this log, and run the tests,
  pipeline, hand-in check, whitespace check, and final Git status;
- avoid changes to context, raw data, the Streamlit app, and earlier outputs
  unless technically necessary, and create no commit.

This is a faithful compact transcription of the controlling prompt. No report,
app, deployment, raw-data, context, or Git-history work was added.

## Student approval and R4 finalisation request

After reviewing the initial R4 evidence, the student approved the main design:

- equity-only Equal Weight is the comparison base;
- the fusion rule is `1 + 0.5 * lagged sentiment`;
- missing prior-day sentiment means no active tilt;
- the rolling estimation window remains 252 observations with monthly
  rebalancing; and
- R4 must not claim sentiment improved performance because the observed fusion
  annualised return and Sharpe ratio are slightly below the base.

The student then asked Codex to add portfolio turnover, a transparent
transaction-cost check, a small sentiment-multiplier sensitivity, and clear
before-and-after results; update the R4 log and tests; run the relevant checks;
and show the final changed files and Git status without creating a commit. The
student explicitly limited the work to descriptive historical analysis and
prohibited Codex from writing economic interpretation or report wording.

## Material reviewed before implementation

Codex fully read:

- `PROJECT_BRIEF.md`, `AGENTS.md`, `README.md`, and
  `SUBMISSION_CHECKLIST.md`;
- every file under `context/`;
- `ai/log_v2_00_constitution.md` through
  `ai/log_v2_03_sentiment.md`;
- the current portfolio, sentiment, fusion, feature, ETL, runner, checker, and
  test code relevant to R4.

Before any edits, Codex inspected the schemas and sample records of the four
saved R2/R3 artifacts. It also recorded these SHA-256 hashes:

```text
fund_returns.csv           EDB2B47AFD742F8E9B2850629EC430087F024E272647E49F1CD062927608A14D
fund_weights.csv           75D6708E1C1A6C5BEB2B5B6B3D4CCB2FC4F0CB4EF27170BA4A7612F5791A61DF
performance_metrics.csv    27FD9D1102B8373DDF70CE4E4DBE3EC96E782BADC87FE77E578C5322B5CF2978
sector_sentiment_index.csv 5BA6CAE86156A2B0DCB9B91DE9B54D5B5270978F65BE63795CBFC84D6A546F01
```

No file under `context/`, no raw data, `src/data_access.py`, the Streamlit app,
or an earlier AI log was edited.

## Approved main implementation choices

R2 contained the Equal Weight and Risk Parity methods in combined funds, but no
separate equity-only fund. To keep news sentiment within its approved scope, R4
reuses the existing Equal Weight method on the 50-equity universe as the base
and creates a separate equity-only sentiment variant. The student explicitly
approved this comparison.

At every first-observed-equity-day monthly rebalance, the fusion target is:

```text
unnormalised ticker target
    = equal ticker weight * (1 + 0.5 * preceding-day sector sentiment)
```

The targets are then renormalised to one. Since the R3 VADER compound index lies
in `[-1, 1]`, the approved strength of 0.5 gives a strictly positive raw
multiplier in `[0.5, 1.5]`. The production sample's realised multipliers were
0.8347486111 to 1.2553. Tickers within a sector receive the same signal.

The decision on date `t` uses the index dated exactly one preceding observed
equity trading day, never the index from `t`. Sector missingness remains visible.
When that preceding sector value is missing, its multiplier is one (no active
tilt); the R3 index itself is not filled or changed. This occurred for Materials
at three decisions: 2021-07-01, 2022-01-03, and 2023-08-01.

Both variants use:

- 50 equities only; no cryptocurrency input, weight, signal, or output;
- adjusted-close simple returns on the observed equity calendar;
- a rolling 252-observation prior estimation period;
- first-observed-trading-day monthly targets and self-financing drift between
  targets;
- long-only, fully invested, no-leverage constraints;
- main comparison paths gross of transaction costs, zero management fee, and a
  zero annual risk-free rate;
- 252 observations per year and the identical 2021-01-04 to 2023-12-29 OOS
  evaluation sample.

The student approved the base, tilt strength, missing-signal treatment,
estimation window, and monthly schedule. The long-only/full-investment
constraints, zero management fee, and zero risk-free-rate convention remain
documented technical assumptions for student verification before the report.

## Turnover, cost-check, and sensitivity conventions

R4 now defines realised one-way turnover at each rebalance after launch as:

```text
0.5 * sum(abs(new target weight - pre-trade drifted weight))
```

The pre-trade weights are the prior target drifted by the realised asset
returns through the trading day before the next target reset. Initial portfolio
establishment is marked explicitly and excluded from turnover totals and costs.

Because the student did not choose a single cost estimate, Codex did not replace
the main gross evidence with one arbitrary net series. Instead, a separate
illustrative schedule checks 0, 5, 10, and 25 basis points per unit of one-way
turnover. On a rebalance date:

```text
net return = (1 - one-way turnover * cost rate) * (1 + gross return) - 1
```

The sensitivity check uses the pre-specified multiplier grid 0, 0.25, 0.5,
0.75, and 1.0. The approved 0.5 row is labelled. The zero row exactly recovers
the base. This is a descriptive robustness grid, not an optimisation exercise;
no multiplier was selected after observing which performed best.

## Code, tests, and outputs added

### `src/fusion.py`

Codex replaced the starter stub with reusable R4 logic that:

- constructs equity-only adjusted-close returns and rejects crypto-like input;
- validates the sector universe, complete return history, sample cap, and
  long-only full-investment targets;
- calls the tested R3 lag helper and additionally requires the observation date
  to equal the immediately preceding observed equity date;
- builds identical-sample base and fusion paths with monthly target resets and
  daily holdings drift;
- saves per-target signal dates, values, missingness, multipliers, sectors,
  estimation bounds, and weights for audit;
- calculates the same geometric annual return, volatility, Sharpe, drawdown,
  and growth measures used by R2;
- creates a two-row comparison with metric differences versus the base and a
  factual Boolean showing whether fusion had higher annualised return;
- calculates one-way turnover from pre-trade drifted holdings, excludes and
  labels initial establishment, and saves every monthly observation;
- applies the disclosed transaction-cost formula across the illustrative
  0/5/10/25-basis-point schedule and keeps gross and net fields side by side;
- evaluates the pre-specified multiplier grid while identifying 0.5 as the
  approved design and 0 as the exact base recovery;
- writes a labelled two-panel growth and drawdown comparison figure with the
  sample period and historical/descriptive disclaimer.

### `scripts/run_part_b.py`

The documented Part B command now builds R4 after reproducing R2 and R3. It
writes the four separate fusion artifacts and prints their shapes and computed
comparison. It does not merge fusion into the required R2 files.

### `tests/test_fusion.py`

Nine deterministic tests cover:

- exact one-observed-trading-day sentiment timing and pre-rebalance return
  estimation;
- invariance of the first target to same-day and future sentiment changes;
- equity-only holdings and rejection of cryptocurrency prices;
- explicit missing-signal/no-tilt treatment;
- exact schemas, identical base/fusion samples, and deterministic repeated
  in-memory builds;
- exact separate filenames, CSV structure, and non-empty figure creation;
- turnover calculated against independently reconstructed pre-trade drifted
  weights, with initial establishment excluded;
- the exact multiplicative transaction-cost growth arithmetic, zero-cost
  equality, and declining net growth as the assumed rate rises; and
- the zero-multiplier/base and approved-0.5/fusion sensitivity identities.

### Saved R4 artifacts

The live pipeline wrote:

```text
results/data/sentiment_fusion_returns.csv       1,506 rows x 10 columns
results/data/sentiment_fusion_weights.csv       3,600 rows x 18 columns
results/tables/sentiment_fusion_comparison.csv  2 rows x 37 columns
results/tables/sentiment_fusion_turnover.csv    72 rows x 9 columns
results/tables/sentiment_fusion_transaction_cost_check.csv
                                                 8 rows x 23 columns
results/tables/sentiment_multiplier_sensitivity.csv
                                                 5 rows x 19 columns
results/figures/sentiment_fusion_comparison.png 10 x 7.2 inch figure at 180 dpi
```

The daily data contain 753 rows per variant. The target audit contains 50
tickers x 36 monthly targets x two variants. These are additional artifacts;
the required R2/R3 files remain separate.

## Actual historical comparison

The saved table contains this computed evidence. It is descriptive evidence,
not economic interpretation, a forecast, or advice.

| Variant | Annualised return | Annualised volatility | Sharpe | Maximum drawdown | Final growth of $1 |
|---|---:|---:|---:|---:|---:|
| Equity Equal Weight base | 0.1264237978 | 0.1611798067 | 0.8192923666 | -0.2024921015 | 1.4272168148 |
| Equity Equal Weight + Sector Sentiment | 0.1228043878 | 0.1607981057 | 0.8008304890 | -0.2036098416 | 1.4135573890 |

Relative to the base, fusion's saved differences are:

```text
annualised return       -0.0036194100
annualised volatility   -0.0003817009
Sharpe ratio             -0.0184618776
maximum drawdown         -0.0011177401
final growth of $1       -0.0136594258
```

The saved evidence therefore does not support a claim that this fusion baseline
improved returns or risk-adjusted performance. Maximum drawdown is also slightly
more negative. The volatility estimate is slightly lower, but this alone is not
evidence that the overall fusion added value.

### Realised turnover

Excluding the initial 2021-01-04 establishment, each variant has 35 measured
monthly rebalances:

| Variant | Total one-way turnover | Mean per rebalance | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Equity Equal Weight base | 0.9533748669 | 0.0272392819 | 0.0185641438 | 0.0480809096 |
| Approved sentiment fusion | 1.3788471125 | 0.0393956318 | 0.0258326127 | 0.0580617050 |

These are realised historical quantities under the stated monthly target-reset
convention. They are not forecasts of future trading volume.

### Illustrative transaction-cost check

The separate cost table keeps gross and net evidence side by side. Selected
net results are:

| One-way cost rate | Base net annualised return | Fusion net annualised return | Base net Sharpe | Fusion net Sharpe |
|---:|---:|---:|---:|---:|
| 0 bps | 0.1264237978 | 0.1228043878 | 0.8192923666 | 0.8008304890 |
| 5 bps | 0.1262441137 | 0.1225453578 | 0.8183193867 | 0.7994159376 |
| 10 bps | 0.1260644557 | 0.1222863823 | 0.8173462863 | 0.7980011544 |
| 25 bps | 0.1255256381 | 0.1215097824 | 0.8144262638 | 0.7937554211 |

Under every displayed illustrative rate, the fusion annualised return and
Sharpe remain below the same-cost base. This is a mechanical historical cost
check, not a claim that any displayed basis-point rate is the correct real-world
implementation cost.

### Sentiment-multiplier sensitivity

All rows use the same sample, lag, missing-score policy, and gross-of-cost basis:

| Multiplier | Annualised return | Sharpe | Total one-way turnover |
|---:|---:|---:|---:|
| 0.00 | 0.1264237978 | 0.8192923666 | 0.9533748669 |
| 0.25 | 0.1245728137 | 0.8099063736 | 1.0857675960 |
| 0.50 (approved) | 0.1228043878 | 0.8008304890 | 1.3788471125 |
| 0.75 | 0.1211113486 | 0.7920462590 | 1.7208655945 |
| 1.00 | 0.1194875043 | 0.7835370193 | 2.0794571045 |

No tested positive multiplier has a higher annualised return or Sharpe than the
zero-multiplier base in this historical sample. The grid is not evidence for
choosing or tuning a future multiplier.

## Verification performed

The restricted sandbox again could not import the project virtual environment's
standard `encodings` module. With user-approved execution outside that
restriction, the following checks were run and inspected:

- Final focused `tests/test_fusion.py`: 9 tests passed in 3.242 seconds.
- Full `unittest` discovery: 31 tests passed (7 foundation, 9 fusion, 7
  portfolio, and 8 sentiment tests) in 3.382 seconds.
- `tests/test_smoke.py`: imports OK and live course data load OK.
- `scripts/run_part_b.py`: completed twice; each run reproduced R2, R3, and R4
  artifacts from the cleaned course inputs.

Independent PowerShell and deterministic-test checks found:

- 753 complete, unique daily rows per variant on the same 2021-01-04 to
  2023-12-29 dates;
- zero missing daily returns and no date after 2023-12-31;
- zero duplicate date/variant or rebalance/variant/ticker keys;
- zero crypto holdings and only `equity` holding classifications;
- zero sentiment-date/rebalance-date or estimation-end/rebalance-date timing
  violations;
- all 72 targets summed to exactly one at displayed precision;
- one excluded initial-establishment row and 35 finite non-negative measured
  turnover rows per variant;
- the independently reconstructed second base turnover matched the saved value;
- zero-cost net growth exactly matched gross growth, and the saved 10-basis-point
  terminal growth matched gross growth multiplied by all disclosed cost factors;
- sensitivity multiplier zero exactly matched the base and multiplier 0.5
  exactly matched the approved fusion metrics;
- 36 monthly decisions, with 15 ticker rows across three Materials decisions
  carrying explicit missing sector sentiment and multiplier one;
- fusion target weights from 0.0163562435 to 0.0226822369 versus a 0.02 equal
  base, with maximum absolute deviation 0.0036437565.

The comparison PNG was rendered and visually inspected. It has labelled axes,
growth and drawdown panels, a legend, the exact sample, and a historical-only
disclaimer.

The second full build produced the same hashes captured after the first build:

```text
sentiment_fusion_returns.csv    C67CB68CD83B30CABBD2066003B54772EB8C9388055EA022DE221853575E5C31
sentiment_fusion_weights.csv    9D0F89A25DDAC92977B2517D69CE563E80238C9DE294517547C670DE19A4C162
sentiment_fusion_comparison.csv 0ADCD3856EF600FF8F3BD8D460D4DB43EA65DEBB22F2208FAE2168B0B4599A00
sentiment_fusion_turnover.csv   D1CCA93E92D34E56B3FB051821AE7F1B78EA55ECB330655BA146AB1AF3474459
sentiment_fusion_transaction_cost_check.csv
                                634A150F07563108DD6231DCAD3C8CD4933386D8550AFF1C0C65770FC343D048
sentiment_multiplier_sensitivity.csv
                                55EE7982C11E8FC04445EE7EC94A68CDFAC2675583D8CF08A935E3D19B46C385
sentiment_fusion_comparison.png 7978E1678C8F78EFF653E8F9015F172C78C1A0575C0C8A19E271EF1749421EC3
```

The earlier four hashes also remained identical to their pre-R4 values, proving
that R2 portfolio results and the saved R3 index were unchanged byte-for-byte.

`scripts/check_handin.py` reported 20 passes, two reminders, and one failure.
The reminders concern compiled Python clutter in the ignored local environment
and the report not yet existing. The failure again names three SciPy/PyArrow
CSV/Parquet test fixtures inside `.venv/`. `git check-ignore -v` confirms all
three are ignored by `.gitignore`'s `.venv/` rule, and `git ls-files -- .venv`
returns no paths. They are not submitted or tracked; this is the same course
checker false positive recorded in R1-R3.

`git diff --check` exited successfully with no whitespace-error output. Git
printed only its Windows line-ending notices. The final short status contained
the two intended modified implementation files and nine intended new R4 files;
it contained no context, raw-data, app, R2 output, or R3 output change.

## Limitations and pending student work

- The design is a simple sector-score tilt with a mandatory one-day operational
  lag. The student approved 0.5 after reviewing the first R4 results, so it must
  not be described as an independently pre-registered or optimally tuned
  parameter. The added grid is descriptive sensitivity, not cross-validation;
  any future tuning claim would require a nested or pre-specified validation
  design.
- The all-equity Equal Weight base reuses an existing R2 method but is a new
  separate R4 strategy because R2 saved only combined funds. The student has
  now approved this comparison.
- Plain VADER headline sentiment inherits the R3 limitations: headlines are not
  full articles, many scores are zero, financial context can be misread, and
  sparse coverage can affect sector values.
- Missing prior news is treated as no active tilt. Other defensible choices may
  produce different results, but carry-forward could introduce stale signals
  and zero imputation would conflate missingness with observed neutrality.
- The main comparison remains gross of costs and assumes zero fees and taxes.
  The 0/5/10/25-basis-point check is illustrative, excludes market impact and
  taxes, and does not establish a correct implementable cost rate.
- Turnover and multiplier sensitivity are now measured, but there is still no
  statistical-significance, regime, or broader robustness analysis. The small
  observed differences must not be described as reliably different from zero.
- Evidence ends on 2023-12-29 and is hypothetical, historical, and descriptive.
  It is not a forecast, causal result, recommendation, or trading advice.

**Pending student review:** the student must independently verify the artifacts
and personally write the economic interpretation, critical reflection, report
wording, recommendations, and final conclusions. Codex did not draft those
items. The student should also decide whether any illustrative cost rate belongs
in the report; Codex did not select one as the definitive assumption.

No Git commit was created.
