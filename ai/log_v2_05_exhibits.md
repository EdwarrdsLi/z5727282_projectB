# AI workflow log v2.05 - R5 report exhibits and fund fact sheets

## Phase and date

- Phase: R5 report-ready exhibits and factual fund fact sheets
- Date: 2026-08-08 (Australia/Sydney)
- Tool: Codex

## Student request and boundaries

The student asked Codex to complete R5 after reading `PROJECT_BRIEF.md`,
`AGENTS.md`, every supplied context file, and the R0-R4 AI logs. The student
required Codex to use only the verified R2-R4 results and create:

- a performance comparison across funds and methods;
- growth-of-$1, drawdown, portfolio-weight, risk-return or Sharpe, and sector
  sentiment-index exhibits;
- concise fact sheets containing growth of $1, annualised return, annualised
  volatility, Sharpe ratio, maximum drawdown, and current target holdings;
- self-contained titles, axes, units, sample periods, and technical captions or
  sources;
- deterministic tests and this exact R5 log;
- the final full relevant tests, `python scripts/run_part_b.py`,
  `python scripts/check_handin.py`, generated-file audit, and Git status.

The student explicitly prohibited changes to approved portfolio, sentiment, or
fusion rules; data downloads; report analysis or economic conclusions; Git
commits; deployment; and ZIP creation. The student required the existing fusion
figure and table to remain in place and asked Codex to stop for visual and
factual review before any commit.

This is a faithful compact transcription of the controlling request. Codex did
not broaden the work into report prose, the app, deployment, or Git history.

## Material read before implementation

Codex fully read:

- `PROJECT_BRIEF.md`, `AGENTS.md`, `README.md`, and
  `SUBMISSION_CHECKLIST.md`;
- every file under `context/`;
- `ai/log_v2_00_constitution.md` through
  `ai/log_v2_04_fusion.md`;
- the R2 portfolio, R3 sentiment, R4 fusion, runner, checker, and test code
  relevant to downstream exhibit generation;
- the schemas, boundary records, dimensions, and recorded metrics of the four
  required precomputed R2/R3 CSV files.

Before editing, the worktree was clean. Codex recorded the following verified
input hashes, which matched the R4 log:

```text
fund_returns.csv           EDB2B47AFD742F8E9B2850629EC430087F024E272647E49F1CD062927608A14D
fund_weights.csv           75D6708E1C1A6C5BEB2B5B6B3D4CCB2FC4F0CB4EF27170BA4A7612F5791A61DF
performance_metrics.csv    27FD9D1102B8373DDF70CE4E4DBE3EC96E782BADC87FE77E578C5322B5CF2978
sector_sentiment_index.csv 5BA6CAE86156A2B0DCB9B91DE9B54D5B5270978F65BE63795CBFC84D6A546F01
```

Codex also recorded all R4 hashes before implementation so the separate fusion
evidence could be checked after reproduction.

## Implementation performed by Codex

### `src/exhibits.py`

Codex added a downstream-only R5 module. It does not import or call raw-data
loaders and does not calculate a portfolio, score a headline, or alter fusion.
It reads only these precomputed files:

```text
results/data/fund_returns.csv
results/data/fund_weights.csv
results/tables/performance_metrics.csv
results/data/sector_sentiment_index.csv
```

Before rendering, the module validates:

- exact required schemas and matching fund names;
- unique fund-date, fund-rebalance-ticker, and date-sector keys;
- the 2023-12-31 sample cap;
- finite daily paths, long-only weights, target sums, and pre-rebalance
  estimation dates;
- growth-of-$1 and drawdown arithmetic from saved daily returns;
- all five saved performance metrics against an independent recomputation using
  each row's recorded annualisation and risk-free-rate conventions;
- evaluation dates, observations, and latest-rebalance links;
- R3 sentiment bounds and the retention of missing values on unobserved
  sector-days.

The module writes a concise performance table, exact current-holdings table,
fact-sheet summary, and exhibit manifest. The manifest marks student
interpretation and final figure acceptance as pending for every row and includes
the unchanged R4 fusion table and figure.

The figure design uses deterministic Matplotlib settings and embeds factual
source notes. Literal dollar signs are rendered as text rather than math markup.
The portfolio-weight figure shows the ten tickers with the highest mean Risk
Parity target weights and aggregates the remaining exact holdings as `Other
holdings`; its caption states this rule. Each fact sheet separately displays all
60 latest target holdings, split into three readable table blocks, while the
exact unrounded holdings remain in CSV.

### `scripts/run_part_b.py`

Codex extended the documented runner after the unchanged R2-R4 build. R5 then
reads the just-written precomputed R2/R3 CSVs, validates them, writes its
artifacts, and prints the number of exhibits and fact sheets plus the two pending
student-review states.

### `tests/test_exhibits.py`

Codex added four deterministic tests using small synthetic precomputed frames.
They cover:

- exact metric reuse, complete latest holdings, schemas, and target sums;
- rejection of inconsistent growth, inconsistent saved metrics, and post-2023
  data;
- exact output filenames, non-empty valid PNGs, required exhibit coverage,
  factual metadata, and pending-review flags; and
- byte-for-byte deterministic CSV and PNG output across two independent output
  directories.

No earlier source module, earlier AI log, context file, raw data, app file,
portfolio rule, sentiment rule, fusion rule, or deployment file was edited.

## Generated R5 artifacts

The final pipeline wrote these new files:

```text
results/tables/performance_comparison.csv                 2 rows x 15 columns
results/tables/fund_fact_sheet_summary.csv                2 rows x 19 columns
results/tables/fund_fact_sheet_holdings.csv             120 rows x 7 columns
results/tables/report_exhibit_manifest.csv               10 rows x 9 columns
results/figures/growth_of_1_comparison.png
results/figures/drawdown_comparison.png
results/figures/portfolio_weights_over_time_risk_parity.png
results/figures/risk_return_comparison.png
results/figures/sector_sentiment_index.png
results/figures/fund_fact_sheet_combined_equal_weight.png
results/figures/fund_fact_sheet_combined_risk_parity.png
```

The ten-row manifest consists of the six required R5 exhibit types, two fund
fact sheets, and the unchanged R4 fusion table and figure. All ten rows say that
student interpretation and final visual acceptance are pending.

Final SHA-256 hashes of the new artifacts are:

```text
performance_comparison.csv                       7D40886EF92D324BD6E47322861FB7C8B955DCDF61A9DCC606F6B10FF44ACB75
fund_fact_sheet_summary.csv                      3CB84988884DB284B886784717690DAD98B5268154988446D914D15AB1C33311
fund_fact_sheet_holdings.csv                     CBCC5D99173C4F1F0452BF2F92F2CE979585B466438C120FBD368E0A7D7E9A64
report_exhibit_manifest.csv                      13D3C22B6BB7A24FCF30283518E853E8AC52854800CE966454E6A05F3EF7BC3B
growth_of_1_comparison.png                       61566C772721BE93EF68A0742984B6915543D075ABA6DEAD15EA1E10211FD84B
drawdown_comparison.png                          B389DEDCAEF00892A38DB31F0D5572B3E75B89C9BBED079FAF544E36D2963AE0
portfolio_weights_over_time_risk_parity.png      0A9B712BA1CFD8374843C7C8F7E45B9F922405855B24D867B19D244B5C54AC32
risk_return_comparison.png                       EABC2E63E48D37916C68DFD026CE161A3C4C406155CD1601F2E768FF25989C61
sector_sentiment_index.png                       9D925A12FACAE1B06D1BDBCFD5077DBF6BB94BD01BB4ECA31C94377860DD9961
fund_fact_sheet_combined_equal_weight.png        3A27C1B92365357150EDD4BE834818FBAFE7F1A150035BA3156E54A9757777CB
fund_fact_sheet_combined_risk_parity.png         039F060ED14E552307E3EA81A4B6F22C3FC84D5B4AFCF0F4086F8CA749BC8602
```

## Factual fund-sheet values

The fact sheets reproduce the verified R2 evidence on the common historical OOS
sample from 2021-01-04 to 2023-12-29:

| Fund | Growth of $1 | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Combined Equal Weight | 1.5236892003 | 0.1513526196 | 0.2160038524 | 0.7608055516 | -0.2787245867 |
| Combined Risk Parity | 1.4779297300 | 0.1396633066 | 0.1620335199 | 0.8880221069 | -0.1947488460 |

Each fact sheet uses the latest target dated 2023-12-01 and includes 60 holdings.
The exact saved holding sums are 0.999999999999996 for Equal Weight and
0.999999999999997 for Risk Parity due only to CSV floating-point
representation. The current Equal Weight target aggregates to 0.8333333333
equity and 0.1666666667 cryptocurrency. The current Risk Parity target
aggregates to 0.9091398512 equity and 0.0908601488 cryptocurrency. These are
factual diagnostics, not an allocation recommendation or interpretation.

The sentiment exhibit covers all ten equity sectors on 1,006 observed equity
dates from 2020-01-02 to 2023-12-29. It retains the verified R3 missing-sector-day
gaps and does not fill, lag, or transform the saved descriptive index.

## Error and correction record

The first sandboxed Python command did not reach the tests because the project
interpreter could not import the standard `encodings` module under the
restricted sandbox. As in R1-R4, Codex reran Python outside that restriction
after approval.

The first focused R5 run passed the two table-validation tests but produced two
renderer errors. `numpy.array_split` converted the holdings DataFrame blocks to
arrays, so `.iloc` was unavailable. Codex replaced that operation with explicit
DataFrame slices and reran the suite. All four tests then passed.

The passing run emitted a Matplotlib `tight_layout` warning for fact-sheet table
axes. Codex replaced the automatic layout call with explicit deterministic
subplot margins; the next focused run passed without that warning.

During visual QA, Codex found that paired literal dollar signs were being parsed
as math text, which italicised parts of an axis label and caption. Codex disabled
math parsing in the R5 visual style, reran the focused tests and full pipeline,
and visually inspected the corrected output. No result value changed.

## Final verification performed

The final checks were run and inspected:

- Focused `tests/test_exhibits.py`: 4 tests passed in 6.931 seconds after the
  final visual-style correction.
- Full `unittest` discovery: 35 tests passed in 8.691 seconds. This consists of
  4 exhibit, 7 foundation, 9 fusion, 7 portfolio, and 8 sentiment tests.
- Explicit `tests/test_smoke.py`: `imports OK` and `data load OK`.
- Final `python scripts/run_part_b.py`: completed in 20.4 seconds, loading
  50,300 equity rows, 14,610 cryptocurrency rows, and 146,836 cleaned news
  rows; it reproduced R2-R4 and wrote all R5 artifacts.
- Visual QA: Codex opened and inspected growth, drawdown, portfolio weights,
  risk-return, sector sentiment, both fact sheets, and the retained fusion
  figure. Titles, axes, units, sample periods, captions, legends, and holdings
  were readable after the correction.
- Independent PowerShell audit: both fact sheets contain 60 latest target
  holdings, their sums reconcile to one at floating-point precision, all ten
  manifest rows carry both pending-review flags, and every generated file is
  non-empty.
- `git diff --check`: exited successfully with no whitespace-error output. Git
  printed only its Windows line-ending notice for `scripts/run_part_b.py`.

The final R2/R3 hashes remained byte-for-byte identical to the recorded input
hashes. All seven recorded R4 hashes also remained identical, including:

```text
sentiment_fusion_comparison.csv 0ADCD3856EF600FF8F3BD8D460D4DB43EA65DEBB22F2208FAE2168B0B4599A00
sentiment_fusion_comparison.png 7978E1678C8F78EFF653E8F9015F172C78C1A0575C0C8A19E271EF1749421EC3
```

Therefore R5 did not change the approved R2 portfolio, R3 sentiment, or R4
fusion evidence.

## Hand-in checker and remaining warnings

`python scripts/check_handin.py` reported:

```text
20 checks passed.
2 reminder(s).
1 problem(s) to fix.
```

The two reminders are local compiled Python clutter and the absence of
`report/report.pdf`, which the student has not yet authored. The single failure
again lists three SciPy/PyArrow CSV/Parquet test fixtures inside `.venv/`.
`git check-ignore -v` maps every path to `.gitignore`'s `.venv/` rule, and
`git ls-files -- .venv` returns no paths. These files are ignored package test
data, not project data and not submission-tracked files. This is the same course
checker false positive documented in R1-R4.

The Streamlit cache helper printed its documented `No runtime found, using
MemoryCacheStorageManager` warnings during Python commands outside Streamlit.
The commands completed and the warnings do not change result data.

## Pending student work and acceptance boundary

**Student interpretation is pending.** The student must independently verify
the factual results and personally write all report analysis, economic
interpretation, critical reflection, recommendations, and conclusions. Codex
did not draft any of that prose.

**Final figure acceptance is pending.** The student must visually review and
approve every figure, both fact sheets, the performance table, and the factual
values before using them in the report or making any Git commit.

The report is still absent. No Git commit, deployment, data download, ZIP,
report analysis, or external change was created during R5.

## Post-R5 technical correction - date-axis bounds

On 2026-08-08, the student identified that six R5 figures displayed an automatic
`2024-01` x-axis tick even though their approved data ended in December 2023.
The student asked Codex to correct only those figure axes, regenerate only the
six affected PNGs, leave all underlying data, calculations, metrics, tables,
and interpretation unchanged, run the relevant exhibit tests and whitespace
check, and create no commit.

Codex added a shared date-axis helper in `src/exhibits.py` that sets each affected
axis to its exact first and last observed date with zero x-margin. The helper is
used by the growth, drawdown, Risk Parity target-weight, sector-sentiment, and
fund-fact-sheet growth plots. Codex also added a dedicated writer that targets
only these six filenames and two deterministic tests covering the exact upper
axis bound and the restricted output set.

The final focused exhibit suite ran six tests in 9.024 seconds and passed. Codex
then invoked only the dedicated six-figure writer against the unchanged
precomputed R2/R3 CSVs. It regenerated:

```text
results/figures/drawdown_comparison.png
results/figures/fund_fact_sheet_combined_equal_weight.png
results/figures/fund_fact_sheet_combined_risk_parity.png
results/figures/growth_of_1_comparison.png
results/figures/portfolio_weights_over_time_risk_parity.png
results/figures/sector_sentiment_index.png
```

Codex opened and visually inspected all six corrected PNGs. None displays a
2024 tick. Their titles still report the approved samples, including 2023-12-29
for daily fund and sentiment evidence and 2023-12-01 for the final target-weight
date.

Hash checks confirmed that `risk_return_comparison.png`, the retained fusion
figure, and all four R5 CSV tables remained byte-for-byte unchanged. No pipeline,
portfolio, sentiment, fusion, metric, holding, sample, or interpretation value
was changed. Student review and final figure acceptance remain pending. No Git
commit was created.

## Student visual review and final figure acceptance

On 2026-08-08, the student reported that they had reviewed all eight R5 figures,
including the six figures with corrected date axes. The student confirmed that
the figures were clear, the sample periods were displayed correctly, the
misleading `2024-01` ticks had been removed, and the underlying results and
approved sample periods were unchanged. The student accepted all eight figures
for use in the final report.

Codex recorded that acceptance in the eight figure or fact-sheet rows of
`results/tables/report_exhibit_manifest.csv`. The two table rows were not part of
the stated figure review and retain their existing pending acceptance status.
Every `student_interpretation_status` remains pending because the student did not
approve or supply report interpretation in this request. No calculation, metric,
table value other than the manifest review status, figure, report text, source
code, test code, earlier log, or other artifact was changed. No Git commit was
created.
