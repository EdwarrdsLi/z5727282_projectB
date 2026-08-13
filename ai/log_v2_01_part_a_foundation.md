# AI workflow log v2.01 - R1 Part A foundation

## Student direction and authorship clarification - 14 August 2026

The decision to reuse the student's own verified Part A foundation, and the
boundaries on what could be changed, came from the student. Codex assisted with
the technical adaptation, focused tests and documentation. The student retained
ownership of the data-treatment decisions and used the test evidence to decide
whether the foundation was acceptable for the later portfolio work.

## Phase and date

- Phase: R1 integration of the student's own Part A data foundation
- Date: 2026-08-08 (Australia/Sydney)
- Tool: Codex

## Student request and boundaries

The student asked Codex to integrate only the verified Part A data-cleaning,
return, calendar-alignment, and headline-panel foundation into Project B. The
student explicitly excluded portfolios, sentiment scoring, fusion, backtests,
the Streamlit app, report analysis, and economic interpretation. The student also
required focused tests, the Project B hand-in check, transparent logging, no raw
data or context-file changes, and no Git commit.

## Material reviewed before editing

Codex fully read the following Project B files:

- `PROJECT_BRIEF.md`
- `AGENTS.md`
- `context/DATA_GUIDE.md`
- `context/project_context.md`
- `README.md`
- `SUBMISSION_CHECKLIST.md`

Codex then inspected only the following relevant material in the student's own
Part A project at
`C:\Users\Cun\Desktop\projectA_starter\z5727282_projectA`:

- `src/etl.py`
- `src/features.py`
- `tests/test_etl.py`
- `tests/test_features.py`
- the relevant R2 orchestration in `scripts/run_part_a.py`
- `results/tables/dataset_inventory.csv`
- `results/tables/duplicate_audit.csv`
- `results/tables/descriptive_stats_returns.csv`
- `results/tables/headline_alignment_audit.csv`
- the schema and boundary rows of `results/data/combined_returns_panel.csv`
- the schema and first records of
  `results/data/headline_daily_panel_sample.csv`

Codex did not inspect another student's work, Part A report prose, Part A AI logs,
or raw data files.

## Verified Part A evidence reused

The inspected Part A outputs recorded these reproducible facts:

- Equity prices: 50,300 cleaned rows through 2023-12-29, unique on ticker/date.
- Cryptocurrency prices: 14,610 cleaned rows through 2023-12-31 after removing
  the ten 2024-01-01 rows, unique on ticker/date.
- News: 146,836 cleaned headlines after removing 2,847 exact duplicates using
  ticker/date/title; distinct headlines on the same ticker/date were retained.
- Returns were native-calendar daily simple returns from `adjClose`, with 252-day
  and 365-day annualisation conventions recorded for equities and crypto.
- The combined return panel used observed equity dates and aligned already
  calculated crypto returns; it did not merge or forward-fill price levels.
- Of 146,836 cleaned headlines, 134,279 mapped to the same observed equity date,
  12,551 mapped forward to the next observed equity date, and six after the last
  observed trading date were unmapped. Counts reconciled and exact text was
  preserved.

These figures are copied from the student's existing Part A derived audit files;
Codex did not rerun Part A or independently recompute them during R1.

## Changes made by Codex

### `src/etl.py`

Codex replaced the Project B starter with a focused adaptation of the student's
Part A cleaning rules. It now:

- imports raw loaders only from Project B's `src.data_access`;
- normalises timestamps through UTC to timezone-naive calendar dates;
- enforces `SAMPLE_END = 2023-12-31`;
- deduplicates equities and crypto on ticker/date;
- deduplicates news only on ticker/date/title;
- retains extreme observations because no return-based deletion, trimming, or
  winsorisation is applied;
- provides clean loaders and `load_part_a_foundation()` for later Project B work.

### `src/features.py`

Codex adapted the student's Part A feature code. It now:

- calculates simple returns from `adjClose` within ticker;
- rejects attempts to substitute raw `close`;
- calculates equity and crypto returns independently before alignment;
- left-aligns already calculated crypto returns to exactly the observed equity
  calendar without filling prices or returns;
- retains the separate 252-day and 365-day descriptive conventions;
- maps cleaned headlines to the same observed equity trading date or the next
  observed equity trading date, never backward;
- preserves exact headline text and produces an alignment audit;
- creates no sentiment score, lagged signal, portfolio, or backtest.

### `tests/test_foundation.py`

Codex added seven deterministic, in-memory tests covering:

- the sample cutoff, price keys, and retention of an extreme price movement;
- the exact news duplicate key and preservation of distinct same-day headlines;
- exclusive use of Project B's three `src.data_access` loaders;
- adjusted-close returns and ticker boundaries;
- crypto return calculation on its native calendar before equity alignment;
- absence of any forward-filled crypto value when an equity date has no crypto
  observation;
- same-day/next-observed-day headline alignment, no backward mapping, exact text
  preservation, reconciliation, and absence of sentiment fields.

## Test execution and correction record

The first command attempted was:

```text
python tests/test_foundation.py
```

It did not reach the tests. Inside the restricted sandbox, the project virtual
environment could not access its base Python standard library and failed while
importing `encodings`. Codex did not report this as a test failure in the code.

With user-approved execution outside that restriction, Codex reran the same test
file using `.venv\Scripts\python.exe` and disabled bytecode generation. Result:

```text
Ran 7 tests in 0.118s
OK
```

Streamlit printed five `No runtime found, using MemoryCacheStorageManager`
warnings during import. These are the documented harmless warnings from using
the provided cached data helper outside Streamlit. All raw loaders were mocked,
so the tests did not download or read course data and did not generate project
outputs.

## Hand-in checker result

Codex ran `python scripts/check_handin.py` as requested. It reported:

```text
15 checks passed.
7 reminder(s).
1 problem(s) to fix.
```

The seven reminders were the existing `.venv` bytecode warning and the expected
absence of the report, result exhibits, and four required Part B model artifacts.
Those artifacts belong to later phases and were not created in R1.

The single failure named three CSV/Parquet test fixtures inside installed SciPy
and PyArrow packages under `.venv/`. Codex checked each path with
`git check-ignore -v`; all three are ignored by the repository's `.venv/` rule.
`git ls-files -- .venv` returned no files, confirming that none is committed. This
is therefore a checker false positive caused by scanning the local ignored virtual
environment rather than the submitted Git file set. Codex did not change the
course checker or delete environment files during this phase.

## Risks and limitations retained for honest review

- Duplicate removal keeps the first row for an exact key. This matches Part A,
  but source order matters if non-key metadata differs across duplicate news rows.
- The combined panel intentionally drops weekend-only crypto return rows instead
  of compounding them into the next equity date. The value on an equity date is
  that date's already-calculated native crypto return.
- A missing crypto observation on an equity date remains missing; it is not
  forward-filled or replaced in this phase.
- The first return for each ticker remains missing because there is no prior
  observation. Any later missing return raises an error for investigation.
- Headlines after the final observed equity trading date cannot map forward and
  are excluded from the assembled panel while remaining counted in the audit.
- R1 does not choose a later missing-data policy for portfolio estimation. That
  is a consequential Project B design choice for the student to make in a future
  phase.
- No live-data integration test was run in R1; the focused tests deliberately
  mocked the loaders to avoid downloading data or generating outputs.

## Remaining student review

The student should confirm that the adapted Project B modules faithfully reflect
their intended Part A foundation, especially the keep-first duplicate policy,
the intentional treatment of weekend-only crypto returns, and the exclusion of
headlines that cannot map forward within the observed sample. The student retains
all final choices for later portfolio missing-data treatment, models, constraints,
sentiment design, fusion, and interpretation.

No analysis or economic interpretation was written for the student. No context,
raw-data, portfolio, sentiment, fusion, backtest, report, or Streamlit file was
changed, and no Git commit was created.