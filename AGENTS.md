# AGENTS.md - Project B constitution for Codex

## Project purpose and authority

This repository is the student's individual FINS5545 Project B: a prototype
FinTech product offering systematically managed equity, cryptocurrency, and
combined funds, supported by walk-forward out-of-sample evidence, equity-news
sentiment analytics, and a deployed Streamlit app.

`PROJECT_BRIEF.md` is the canonical assignment specification. Read it together
with `README.md`, `SUBMISSION_CHECKLIST.md`, and every file in `context/` before
substantial work. Files under `context/` are supplied reference material and must
not be edited. Work only with this student's own Project A and Project B files;
never inspect or copy another student's project.

The student makes all final project choices, including the product name and value
proposition, fund universe, portfolio methods, constraints, estimation window,
rebalance schedule, transaction-cost assumptions, sentiment treatment, fusion
design, innovation, visual identity, and conclusions. Codex may present options,
trade-offs, and technical evidence, but must identify assumptions and leave
consequential choices for the student to approve.

## Division of responsibility and academic integrity

The student personally writes the submitted analysis, economic interpretation,
critical reflection, recommendations, and final conclusions. Codex may assist
with code, tests, technical checks, debugging, documentation structure,
formatting, and clearly labelled drafts or outlines. AI-generated analysis or
economic interpretation must never be presented as the student's own work.

Treat every AI output as an unverified draft. Never invent a citation, source,
statistic, result, test outcome, or explanation. Trace every reported number to a
reproducible computation and every factual or methodological claim to a source
the student can open and verify. Flag uncertainty and unsupported claims instead
of filling gaps. Record meaningful AI assistance, risks, corrections, and student
decisions in `ai/`.

Keep all project-facing writing in English, including documentation, code
comments, figure labels, table labels, and app text.

## Data and repository safety

- Load course data only through `src/data_access.py`; do not create another raw
  data loader or scrape replacement data.
- Treat `src/data_access.py` as provided course infrastructure and do not edit it
  unless the student gives explicit, assignment-consistent instructions.
- Do not edit any file under `context/`.
- Do not commit raw or source data, Parquet files, secrets, credentials, API keys,
  `.env` files, Streamlit secrets, or machine-specific absolute paths. Use paths
  relative to the repository root.
- Do not use observations after 2023-12-31 or claim that the evidence covers a
  later period.
- Commit only derived, non-secret artifacts needed for reproduction, reporting,
  or the deployed app under `results/`.
- Preserve unrelated student work and respect the files explicitly allowed by
  each request. Do not create commits, push, deploy, or make external changes
  unless the student explicitly requests them.

## Return construction and calendar rules

- Use adjusted-close (`adjClose`) prices for returns.
- Calculate equity and cryptocurrency returns separately on their native
  calendars before any alignment.
- For a combined panel, align already-calculated cryptocurrency returns to the
  equity trading calendar. Never merge price levels first and calculate returns
  afterward.
- Preserve the distinction between the equity trading calendar and the
  seven-day cryptocurrency calendar. Use an annualisation factor appropriate to
  the evaluated fund/calendar, normally 252 for equity-calendar or combined funds
  and 365 for cryptocurrency-only funds, and state the convention.
- Normalise date types and time zones deliberately before alignment. Map each
  headline to the same equity trading day when possible and otherwise to the next
  equity trading day.
- Deduplicate prices by ticker and date and exact news duplicates by ticker,
  date, and title. Do not treat multiple distinct headlines on one ticker-date as
  duplicates.

## Backtest and portfolio rules

- Prevent look-ahead bias in every feature, signal, optimisation, and evaluation.
- Calculate portfolio weights using only information available before the return
  period to which those weights are applied.
- Use a walk-forward out-of-sample design with a genuine prior estimation period;
  the live backtest must not begin on the first observation.
- Rebalance monthly or less frequently. State the estimation-window type and
  length, first live date, rebalance rule, constraints, risk-free-rate convention,
  and transaction-cost assumption.
- Treat each asset-family and optimisation-method pair as a distinct fund and
  keep its returns, weights, metrics, and current holdings internally consistent.
- Check optimiser status and scaling. Do not accept a solver's success flag as
  sufficient evidence: validate constraints, weight sums, finite outputs, and
  whether methods produce meaningfully different weights.

## Sentiment and fusion rules

- Sentiment is derived from equity headlines only. Do not apply equity-news
  sentiment directly to cryptocurrencies.
- Preserve headline features needed by the chosen model, including casing,
  punctuation, negation, and other context when using VADER.
- Build sector sentiment by aggregating ticker-day scores and equal-weighting
  tickers within each equity sector, unless the student explicitly chooses and
  justifies a permitted alternative.
- Decide and document how ticker-days without headlines are handled.
- Lag sentiment by at least one equity trading day before using it in an
  investment decision. A signal used on day t must contain no information first
  available on day t or later.
- Evaluate fusion against its untilted base fund on the same out-of-sample dates
  and assumptions. Report negative or insignificant results honestly.

## Reproducibility, tests, and outputs

- Keep reusable logic in `src/`, orchestration in `scripts/`, app-readable derived
  data in `results/data/`, report tables in `results/tables/`, and figures in
  `results/figures/`.
- Maintain one documented command that reproduces Part B artifacts from the
  hosted course data. Avoid undocumented manual transformations.
- Produce the exact required files:
  `results/data/fund_returns.csv`, `results/data/fund_weights.csv`,
  `results/data/sector_sentiment_index.csv`, and
  `results/tables/performance_metrics.csv`.
- Add focused tests for return construction, native-calendar handling, temporal
  alignment, estimation/live-period separation, weight timing, sentiment lags,
  portfolio constraints, metrics, and output schemas. Use small deterministic
  fixtures where possible.
- Run relevant tests and reproducibility checks after implementation changes, but
  only when the current task permits execution. Never claim a test, pipeline, or
  app check passed unless it was actually run and its result was inspected.
- Sanity-check key results independently, including weight sums, missing values,
  date coverage, first live date, growth arithmetic, drawdowns, and annualisation.
- Use deterministic settings or seeds where randomness is unavoidable and record
  important package or environment assumptions.

## Streamlit deployment boundary

The deployed Streamlit app must read precomputed derived files from `results/`.
It must not run portfolio backtests, VADER, NLTK, model fitting, or other heavy
build steps at runtime. Keep build-only dependencies such as NLTK outside the
deployed app requirements. The app must not depend on raw data files, local paths,
secrets, or uncommitted artifacts.

## Review and reporting standard

Before handing work to the student:

1. Review the diff and confirm that only authorised files changed.
2. Run only the checks permitted by the current request and report their exact
   outcomes, including skipped checks.
3. Check outputs against the brief, required filenames, schemas, and sample dates.
4. Identify errors, limitations, assumptions, solver problems, missing coverage,
   and weak or negative results candidly; do not hide or cosmetically improve them.
5. Separate computed evidence from interpretation and clearly mark any text the
   student must verify or rewrite in their own words.
6. Update the appropriate `ai/` log with the real prompt, assistance, risks,
   verification, student corrections, and unresolved decisions.
