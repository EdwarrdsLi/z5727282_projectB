# AI workflow log v2.06 - R6 Streamlit investor app

## Phase and date

- Phase: R6 results-only Streamlit investor app
- Date: 2026-08-09 (Australia/Sydney)
- Tool: Codex

## Student request and boundaries

The student asked Codex to begin R6 by fully reading `PROJECT_BRIEF.md`,
`AGENTS.md`, `README.md`, every file under `context/`, the R2-R5 AI logs, and the
existing generated files under `results/`. The student requested a complete root
`streamlit_app.py` that lets an investor:

- compare the completed fund methods;
- read each fund fact sheet and its performance metrics;
- inspect growth, drawdown, risk-return, and portfolio-weight figures;
- enter a hypothetical investment amount and compare allocations across the
  completed funds; and
- explore the sector sentiment index and the approved sentiment-fusion
  comparison.

The student required the deployed app to read only committed precomputed files
under `results/`, perform no raw-data load, model fitting, backtest, VADER, NLTK,
or pipeline work, present no forecast, signal, or investment advice, describe the
evidence as historical out-of-sample and descriptive, and show no evidence after
2023-12-31. The student also asked for focused loading, section, and deployment
restriction tests, this log, the relevant tests, `scripts/check_handin.py`,
changed files, remaining warnings, and Git status. The student prohibited report
analysis, a Git commit, and deployment.

This is a faithful compact transcription of the controlling prompt. Codex did
not broaden the work into report writing, portfolio/sentiment changes, Git
history, deployment, or external actions.

## Material reviewed before implementation

Codex fully read and inspected:

- `PROJECT_BRIEF.md`, `AGENTS.md`, `README.md`, and
  `SUBMISSION_CHECKLIST.md`;
- all three supplied files under `context/`;
- `ai/log_v2_02_portfolios.md` through `ai/log_v2_05_exhibits.md`;
- the existing app starter, deployment requirements, hand-in checker, relevant
  source and test code;
- the schemas, dimensions, boundary records, and date coverage of every CSV
  under `results/`;
- all eight existing PNGs under `results/figures/`, by visual inspection; and
- the R4 approved fusion design and its recorded negative historical comparison.

No `context/`, raw-data, `src/data_access.py`, R1-R5 model source, R2-R5 result,
report, requirement, deployment configuration, or existing AI-log file was
edited.

## Implementation performed by Codex

### `src/app_data.py`

Codex added a deployment-safe data boundary that:

- declares the exact precomputed CSV and PNG inputs used by the app, all under
  `results/`;
- rejects missing, empty, out-of-schema, duplicate-key, or post-2023 artifacts;
- cross-checks fund names between returns, performance metrics, and fact sheets;
- resolves artifact paths only within the repository's `results/` folder; and
- calculates hypothetical fund-level allocation paths solely from the saved
  `growth_of_1` series.

The allocation calculation fixes the user's chosen fund split immediately
before the first saved OOS return and does not rebalance between funds. Each
underlying fund still follows its saved monthly target-reset convention. The
calculation is historical scenario arithmetic, not a backtest rerun or forecast.

### `streamlit_app.py`

Codex replaced the raw-data-loading starter with a results-only investor app. It
contains six main sections:

1. **Compare funds** - same-sample metrics and the accepted growth figure for
   Combined Equal Weight and Combined Risk Parity.
2. **Fact sheets** - a fund selector, five fact-sheet metrics, the accepted
   one-page fact sheet, the equity/cryptocurrency target mix, and all 60 latest
   target holdings.
3. **Allocation lab** - an investment amount, two-fund percentage split, 100%
   single-fund comparisons, custom historical ending values, and a saved-path
   value chart.
4. **Fund analytics** - the accepted growth, drawdown, risk-return, and Risk
   Parity target-weight figures.
5. **Sentiment & fusion** - selectable equity-sector VADER index paths, coverage
   counts, the complete accepted sentiment figure, the approved equity-only
   base-versus-fusion comparison, and the saved turnover, transaction-cost, and
   multiplier-sensitivity evidence.
6. **Method notes** - the saved sample, 252-observation rolling window, monthly
   rebalance rule, constraints, annualisation, risk-free-rate, cost basis,
   sentiment timing, and explicit exclusions.

The app says prominently that the evidence is historical, walk-forward OOS, and
descriptive and that it is not a forecast, signal, recommendation, promise, or
investment advice. It states that displayed fund returns are gross of costs,
fees, and taxes. The fusion section reports honestly that the approved variant's
historical annualised return and Sharpe ratio are below the base. It does not
provide an economic explanation or causal claim.

The page title `Systematic Fund Explorer` is a neutral working label, not a final
product-name decision. No raw dataset, raw-data loader, build module, VADER,
NLTK, portfolio fitter, or Part B runner is imported or called at app runtime.

### `tests/test_app.py`

Codex added five focused tests covering:

- successful loading of every required precomputed CSV and figure;
- the approved 2023-12-31 date cap;
- exact arithmetic for the historical allocation comparison;
- a real Streamlit test render containing all six required investor sections;
  and
- static deployment-boundary checks against raw-data/model-build imports and
  calls, plus confirmation that app dependencies exclude NLTK and every declared
  artifact path begins under `results/`.

## Risks and implementation choices for student review

- Only the two completed R2 combined funds are offered because those are the
  completed investable fund artifacts. The separate R4 equity base and fusion
  variant are shown as an approved analytic comparison, not added to the fund
  allocation product.
- The hypothetical allocation is a transparent fund-level buy-and-hold scenario.
  A different fund-level rebalancing rule would produce a different path and is
  a consequential product choice requiring student approval.
- The default amount is $10,000 and the default split is 50/50. These are neutral
  interface defaults, not recommendations.
- The app exposes the precomputed 10-basis-point cost row by default in an
  interactive illustrative schedule, but does not label any cost rate as the
  correct real-world assumption.
- The sector chart defaults to Tech, Financials, and Energy only for legibility;
  all ten sectors remain selectable and the accepted complete figure is present.
- The app uses a generic working title and a restrained navy/teal/amber visual
  system. The student must choose the final product name, approve the visual
  identity and wording, and decide whether the default controls support the
  intended customer journey.
- Historical paths end on 2023-12-29 and latest targets on 2023-12-01. No saved
  data or claim extends past 2023-12-31.
- The saved fund evidence assumes a zero annual risk-free rate and is gross of
  transaction costs, management fees, and taxes. The allocation lab inherits
  those limitations.
- Plain VADER scores supplied equity headlines, not full articles. The index is
  noisy, retains missing sector-days, and is not a trading signal.

## Verification performed and actual results

The first sandboxed Python attempt failed before test collection because the
project interpreter could not import Python's standard `encodings` module. This
is the same local sandbox issue recorded in R1-R5. After explicit approval,
Codex ran the Python checks outside that restriction.

Actual inspected results:

- Focused command `python -m unittest tests.test_app -v`: **5 tests passed** in
  1.965 seconds.
- Full command `python -m unittest discover -s tests -p 'test_*.py' -v`:
  **42 tests passed** in 16.141 seconds. This includes 5 app, 6 exhibit, 7
  foundation, 9 fusion, 7 portfolio, and 8 sentiment tests.
- The Streamlit render test completed without an app exception and found all six
  required main sections.
- Streamlit emitted expected bare-test warnings about a missing script context
  and its memory cache manager; they did not stop execution.
- Independent result-date audit found maxima of 2023-12-29 for fund returns,
  sector sentiment, and fusion returns, and 2023-12-01 for fund and fusion target
  weights.
- The app restriction scan found no raw-data loader, pipeline, model-builder,
  VADER, or NLTK reference in executable app modules.
- `git diff --check` exited successfully with no whitespace-error output. Git
  printed only its Windows LF-to-CRLF notice for `streamlit_app.py`.
- `results/` had no Git status change; the app work did not regenerate or alter
  any verified R2-R5 result.

## Hand-in checker and remaining warnings

`python scripts/check_handin.py` reported exactly:

```text
20 checks passed.
2 reminder(s).
1 problem(s) to fix.
```

The reminders are auto-generated compiled Python clutter in the ignored local
environment and the still-missing student-authored `report/report.pdf`. The one
failure again lists three SciPy/PyArrow CSV/Parquet package fixtures under
`.venv/`. `git ls-files -- .venv` returns no paths. The named files are ignored
environment dependencies, not project data and not submission-tracked files.
This is the same broad-scanner false positive documented in R1-R5; Codex did not
delete or alter the student's virtual environment.

## Student review and remaining work

The student must now:

- run and visually review the app locally, including responsive layout and all
  controls;
- approve or replace the working product name and visual identity;
- approve the allocation convention, default amount/split, sector defaults, and
  robustness-control defaults;
- verify the factual labels and decide what belongs in the final customer
  journey; and
- personally write the report analysis, economic interpretation, critical
  reflection, recommendations, and conclusions.

Codex did not write report prose or economic interpretation. No Git commit,
deployment, push, ZIP, raw-data access, data download, or external change was
created.

## Follow-up interface adjustment - 2026-08-09

The student reported that the green `HISTORICAL FUND EVIDENCE` kicker was partly
hidden behind Streamlit's top toolbar at normal browser zoom. They asked Codex to
increase only the main page container's top spacing, preserve every other aspect
of the app, run the focused app tests, show `git status --short`, and create no
commit.

Codex changed only the `.block-container` `padding-top` value in
`streamlit_app.py`, from `2.2rem` to `4rem`. No product name, content, colour,
tab, figure, table, data, calculation, or other layout declaration was changed.

The initial sandboxed test attempt failed before collection because the local
Python runtime could not import its standard `encodings` module. With explicit
approval, Codex ran `python -m unittest tests.test_app -v` outside that
restriction: **5 tests passed** in 1.504 seconds. Streamlit emitted its expected
bare-mode missing `ScriptRunContext` warning; no app exception occurred. The
final `git status --short` output was reported directly to the student. No Git
commit was created.

## Follow-up Fact sheets display fix - 2026-08-09

The student's prompt was:

> Please fix only two display issues in the Fact sheets tab.
>
> 1. The five metric cards are too narrow at normal browser zoom. Some values
> and labels are shown with ellipses. Make the cards responsive so that every
> label and full value is visible. Keep all values unchanged.
> 2. Change the asset-class chart x-axis label from `target_weight` to
> `Target weight (%)`.
>
> Do not change the product name, calculations, data, fund methods, tabs,
> colours, fact-sheet figures, holdings, or other page content.
>
> Update the R6 AI log with this small display fix. Run the focused app tests
> and show the final `git status --short`. Do not create a Git commit.

Codex changed only the requested Fact sheets presentation code in
`streamlit_app.py`. The five existing `st.metric` calls retain their exact
labels, formatting expressions, and source values. They now sit in Streamlit's
native horizontal wrapping container with a 220-pixel card width, so cards move
to another line when the available browser width cannot show them in one row.
The asset-class bar chart now explicitly sets its horizontal value-axis label to
`Target weight (%)`. Its `target_weight` data, percentage conversion, teal
colour, orientation, holdings, and figure content were not changed.

The focused command
`python -m unittest tests.test_app -v` completed successfully: **5 tests
passed** in 1.512 seconds. The Streamlit render test emitted its expected
bare-mode missing `ScriptRunContext` warning; no app exception occurred. The
test suite verifies the results-only data boundary and all required investor
sections, but it does not perform pixel-level browser screenshot comparison, so
the student should still confirm the wrapping visually at their normal browser
width. Codex did not change calculations, data, methods, tabs, colours, saved
figures, holdings, product text, or other app content, and did not create a Git
commit.

## Follow-up Fact sheets axis-label correction - 2026-08-09

The student asked Codex to change only the axis labels of the **Target mix by
asset class** chart on the Fact sheets tab: show `Target weight (%)` as the
x-axis title, remove the y-axis title while retaining the `equity` and
`cryptocurrency` category labels, update this R6 log, run the focused app tests,
show `git status --short`, and create no commit.

The x-axis title was already explicitly set to `Target weight (%)` by the prior
Fact sheets display adjustment. Codex added only an empty `y_label` argument to
the same Streamlit bar chart so that its category-axis title is suppressed. The
chart data, percentage conversion, values, category tick labels, colour, and
orientation were not changed. No other chart, page text, tab, or layout code was
changed.

The initial sandboxed test attempt failed before collection because the local
Python runtime could not import its standard `encodings` module. With explicit
approval, Codex reran `python -m unittest tests.test_app -v` outside that
restriction: **5 tests passed** in 1.495 seconds. Streamlit emitted its expected
bare-mode missing `ScriptRunContext` warning; no app exception occurred. Codex
then inspected the focused diff and final `git status --short`. No Git commit was
created.

## Follow-up horizontal-axis label correction - 2026-08-09

The student supplied screenshots of both completed fund fact sheets for visual
verification. The metric cards rendered completely, but the physical placement
of the asset-class chart title was still wrong: `Target weight (%)` appeared
vertically on the left and the horizontal value axis had no title.

Codex corrected only the Streamlit label mapping for the existing horizontal bar
chart. Because Streamlit swaps the rendered axes when `horizontal=True`, the
chart now passes an empty `x_label` and `Target weight (%)` as `y_label`. The
chart data, percentage conversion, category labels, colour, orientation,
holdings, calculations, and other page content were not changed. The correction
still requires a refreshed browser screenshot for pixel-level confirmation.

## Follow-up fusion metric-card display fix - 2026-08-09

The student supplied screenshots of the three Sentiment & fusion subpages. The
sector index, approved fusion evidence, and robustness tables rendered with the
saved values, but the four difference-card labels on the Approved fusion subpage
were truncated with ellipses at the student's normal browser width.

Codex changed only those four existing metric cards from fixed four-column
placement to Streamlit's horizontal wrapping container. Each card now has a
260-pixel width so it can wrap to another row when necessary. The exact labels,
values, calculations, comparison rows, charts, tables, colours, and explanatory
text were preserved.

The student's refreshed screenshot confirmed that all four labels and values
rendered completely. The remaining Method notes page was also visually checked
and accepted. Codex then ran the full project test suite with `python -m unittest
discover -s tests -p 'test_*.py' -v`: **42 tests passed** in 12.177 seconds.
`git diff --check` passed with only Git's informational LF-to-CRLF warning. The
four required assignment CSV files were present and tracked. No Git commit was
created.
