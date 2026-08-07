# AI workflow log v2.02 - R2 portfolios and walk-forward backtests

## Phase and date

- Phase: R2 portfolio construction and walk-forward out-of-sample backtesting
- Date: 2026-08-08 (Australia/Sydney)
- Tool: Codex

## Student prompt

The student gave Codex this request:

> Please complete R2: portfolio construction and walk-forward out-of-sample
> backtesting. First, fully read PROJECT_BRIEF.md, AGENTS.md, README.md,
> SUBMISSION_CHECKLIST.md, context/, the existing R0-R1 AI logs, and all current
> source and test files. Preserve R0-R1 and work only on portfolio construction
> and backtesting. Implement mainly in src/portfolios.py using the R1 clean-return
> foundation; include a combined equity-and-cryptocurrency fund with Equal Weight
> and Risk Parity; use a genuine prior estimation window, monthly-or-less-frequent
> walk-forward rebalancing, and only pre-rebalance information. Calculate crypto
> returns on its native calendar before equity-date alignment, do not forward-fill,
> use sentiment, forecast, recommend, make causal claims, or report after
> 2023-12-31. Generate fund_returns.csv, fund_weights.csv, and
> performance_metrics.csv with the required fund facts and holdings link. Update
> scripts/run_part_b.py, create tests/test_portfolios.py for look-ahead, monthly
> rebalancing, weight sums, schemas, first OOS date, and metrics, and create this
> log with prompt, choices, checks, limitations, and actual results. Do not write
> the report, implement later phases, edit context/raw data, commit, or deploy.
> Finally run all relevant tests and scripts/check_handin.py, then report changed
> files, output dimensions/date ranges, methods/rule, test results, warnings,
> git diff --check, and git status --short.

This is a faithful compact transcription of the full prompt. The controlling
requirements and exclusions have been retained; no requested capability was
silently broadened.

## Material read before implementation

Codex fully read:

- `PROJECT_BRIEF.md`, `AGENTS.md`, `README.md`, and
  `SUBMISSION_CHECKLIST.md`;
- every file under `context/`;
- the R0 and R1 logs plus the AI README and prompt template;
- every current file under `src/` and `tests/`;
- `scripts/run_part_b.py` and `scripts/check_handin.py`.

No `context/`, raw-data, `src/data_access.py`, report, sentiment, fusion,
Streamlit, deployment, or Git-history file was changed.

## Provisional implementation choices for student review

The student specified the two methods and the combined family but did not fix
the remaining backtest parameters. Codex used these explicit baseline
assumptions so R2 could be reproduced:

- Universe: all 50 cleaned equities and all 10 cleaned cryptocurrencies.
- Return input: daily simple returns from `adjClose`, calculated separately on
  native equity and crypto calendars by the R1 feature code. Already-calculated
  crypto returns are then left-aligned to observed equity dates. Nothing is
  forward-filled.
- Evaluation calendar and annualisation: observed equity dates and 252 periods
  per year for both combined funds.
- Estimation: rolling 252 complete equity-calendar return observations.
- First target and live return: 2021-01-04, after 252 valid prior observations.
- Rebalance rule: first observed equity trading day of every calendar month.
  The estimation window ends on the preceding observed date. The new target is
  applied to the rebalance-date return, then holdings drift self-financingly
  between target resets; this does not assume daily rebalancing.
- Constraints: long-only, fully invested, and no leverage.
- Equal Weight: `1 / 60` target for each asset at each monthly reset.
- Risk Parity: equal-risk-contribution weights from the rolling sample
  covariance. A 5% diagonal covariance shrinkage is applied for positive
  definiteness and numerical stability. The convex risk-budgeting problem is
  solved by deterministic coordinate descent, then convergence, finite values,
  positivity, weight sum, and achieved equal risk contributions are checked.
- Risk-free rate: zero annually. Transaction-cost rate: zero. No management fee.
- Annualised return: geometric return from the OOS growth path. Volatility and
  Sharpe use sample daily volatility and 252-day scaling. Maximum drawdown uses
  an initial wealth peak of $1.
- Current holdings: the latest target weights dated 2023-12-01 in
  `results/data/fund_weights.csv`, linked from each performance row.

These are technical baseline choices, not student economic conclusions. The
student must approve or revise the universe, window, shrinkage, rebalance timing,
constraints, risk-free rate, costs, and fees before the final report.

## Code and output changes

### `src/portfolios.py`

Codex implemented:

- a wrapper around the R1 native-calendar/combined-return constructor;
- equal-weight and equal-risk-contribution target estimators;
- strict input, missing-value, cutoff, covariance, optimisation, and weight
  validation;
- monthly rolling walk-forward backtesting with pre-return weights and drifting
  holdings between target resets;
- daily fund return, growth, drawdown, rebalance linkage, target-weight, and
  performance-metric schemas;
- in-memory R2 artifact construction and exact required-file writing.

### `scripts/run_part_b.py`

The script now loads cleaned equity and crypto prices through `src.etl`, runs the
R1 return construction and R2 backtests, writes the three requested CSVs, and
prints dimensions plus computed metrics. Sentiment and subsequent phases remain
out of scope.

### `tests/test_portfolios.py`

Seven deterministic tests cover first-OOS separation, strict weight timing,
first-observed-day monthly rebalancing, long-only weight sums, future-data
invariance of the first Risk Parity target, hand-calculated metrics, required
schemas and both combined methods, the 2023 cutoff, and rejection rather than
filling of a missing return after complete history begins.

## AI error and correction record

The first implementation used SciPy L-BFGS-B on the convex Risk Parity objective.
The first focused run produced two errors because the solver returned an
`ABNORMAL` status on deterministic rolling windows. Codex did not accept or hide
that result. It replaced the fragile line search with coordinate descent for the
same convex objective and added a convergence check. A second run still exposed
non-convergence on a near-singular synthetic covariance. Codex added the stated
5% diagonal shrinkage, which guarantees a positive-definite estimate when every
asset variance is positive. One look-ahead test then failed for a test-design
reason: its future perturbation made one asset exactly constant in later windows,
so the estimator correctly rejected zero variance. Codex changed that future
perturbation to a non-constant sign-and-scale transformation; it still changes
all future information without invalidating later covariance estimates. The
next run passed all seven tests.

This history matters: solver status alone was not treated as proof of a valid
portfolio, and the synthetic tests caught both numerical and fixture risks.

## Actual reproduced outputs

`python scripts/run_part_b.py` completed successfully using cleaned inputs of
50,300 equity rows and 14,610 crypto rows. It wrote:

- `results/data/fund_returns.csv`: 1,506 rows x 8 columns; 753 daily OOS rows
  per fund, 2021-01-04 through 2023-12-29.
- `results/data/fund_weights.csv`: 4,320 rows x 10 columns; 60 assets x 36
  monthly targets x 2 methods, 2021-01-04 through 2023-12-01.
- `results/tables/performance_metrics.csv`: 2 rows x 21 columns.

Computed evidence (not interpretation):

| Fund | Annualised return | Annualised volatility | Sharpe | Maximum drawdown | Final growth of $1 |
|---|---:|---:|---:|---:|---:|
| Combined Equal Weight | 0.1513526196 | 0.2160038524 | 0.7608055516 | -0.2787245867 | 1.5236892003 |
| Combined Risk Parity | 0.1396633066 | 0.1620335199 | 0.8880221069 | -0.1947488460 | 1.4779297300 |

Independent PowerShell recomputation of growth, geometric annual return,
annualised volatility, arithmetic Sharpe, and drawdown matched every saved
metric to displayed floating-point precision. It also found zero missing cells,
zero return rows after 2023-12-31, zero estimation-end/rebalance-date violations,
and target-weight sums from 0.999999999999999 to 1.0 across all 72 targets. On
2023-12-01 the maximum absolute asset-weight difference between methods was
0.0212375622. Latest aggregate crypto target weights were 0.1666666667 for Equal
Weight and 0.0908601488 for Risk Parity. These figures are diagnostics, not an
economic recommendation or interpretation.

## Tests and hand-in check

Because the restricted sandbox could not import Python's standard `encodings`
module, the student approved Python execution outside that restriction. The
final relevant test command ran:

- `tests/test_foundation.py`: 7 tests passed in 0.080 seconds.
- `tests/test_portfolios.py`: 7 tests passed in 0.210 seconds.
- `tests/test_smoke.py`: imports OK and live course data load OK.

The Streamlit cache helper printed its documented harmless `No runtime found,
using MemoryCacheStorageManager` warnings outside a Streamlit process.

`python scripts/check_handin.py` ran and returned 19 passes, 3 reminders, and 1
failure. The reminders were compiled-Python clutter inside the local environment,
the report not yet being authored, and `sector_sentiment_index.csv` not yet
existing because sentiment is explicitly outside R2. The failure listed three
SciPy/PyArrow test-data CSV/Parquet files under `.venv/`; the checker scans the
ignored local environment rather than only submission-tracked files. R1 had
already identified the same false positive. `git check-ignore -v` mapped all
three files to the repository's `.venv/` ignore rule, and `git ls-files --
.venv` returned no paths. They are local environment dependencies, not committed
project data.

`git diff --check` exited successfully with no whitespace-error output. Git did
print Windows line-ending notices that LF would become CRLF if it later rewrites
the two modified tracked Python files; these are notices, not diff-check errors.
`git status --short` listed only the two intended tracked modifications and five
intended new R2 files. No Git commit was created.

## Limitations and unresolved student decisions

- This R2 contains only the required combined family, not optional equity-only
  or crypto-only funds. It intentionally contains only the two requested methods.
- Complete cross-asset history is required after the initial all-missing return
  row. A later missing observation stops the build for investigation instead of
  being filled or silently dropping a trading day.
- Crypto weekend returns remain excluded after native-calendar calculation and
  equity-date alignment, matching the project rules. Weekend moves are not
  compounded into Monday.
- The sample covariance and 5% diagonal shrinkage are modelling choices, not
  universally correct parameters. Risk Parity results depend on them and on the
  rolling-window length.
- Zero costs, zero fees, and a zero risk-free rate are simplifying assumptions.
  Turnover and implementation frictions are not measured in R2.
- The latest saved weights are target holdings, not post-return drifted holdings.
- Backtested evidence is hypothetical and ends on 2023-12-29. It is not a
  forecast, causal claim, trading recommendation, or evidence after 2023.
- No report analysis or economic interpretation was drafted. The student must
  verify all outputs, make the final methodological choices, and write any
  interpretation in their own words.
