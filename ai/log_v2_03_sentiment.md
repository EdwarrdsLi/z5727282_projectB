# AI workflow log v2.03 - R3 descriptive headline sentiment

## Phase and date

- Phase: R3 VADER headline scoring and descriptive sector sentiment index
- Date: 2026-08-08 (Australia/Sydney)
- Tool: Codex

## Student prompt and boundaries

The student asked Codex to complete R3 only: read the project brief,
constitution, README, data guide, project context, prior R0-R1 logs, and relevant
code and tests; implement VADER headline scoring in `src/sentiment.py`; build a
daily equity-sector sentiment index; integrate it into `scripts/run_part_b.py`;
create focused tests, this log, and the exact required
`results/data/sector_sentiment_index.csv`; and run the new, foundation,
portfolio, and smoke tests, the Part B pipeline, the hand-in checker, whitespace
check, and final Git status.

The student required original headline text, supplied equity news only, no
cryptocurrency sentiment, a 2023-12-31 sample cap, ticker-fair sector
aggregation, honest treatment of missing news, and documentation and testing of
the minimum one-observed-equity-day lag for any future trading use. The student
explicitly excluded sentiment fusion, portfolio changes, the app, deployment,
and report work, and prohibited a Git commit.

This section is a faithful compact transcription of the controlling request.
No excluded later-phase work was added.

## Material reviewed before implementation

Codex fully read:

- `PROJECT_BRIEF.md`, `AGENTS.md`, `README.md`, and
  `SUBMISSION_CHECKLIST.md`;
- every supplied file under `context/`;
- `ai/log_v2_00_constitution.md`, `ai/log_v2_01_part_a_foundation.md`, and the
  relevant R2 portfolio log;
- the current data foundation, sentiment stub, portfolio module, fusion stub,
  data-access helper, Part B runner, hand-in checker, requirements, and all
  existing tests.

No file under `context/`, `src/data_access.py`, portfolio logic, fusion logic,
the report, the Streamlit app, or deployment configuration was edited.

## R3 method and provisional missing-news choice

The implementation uses unmodified NLTK VADER. Each cleaned supplied equity
headline is mapped to the same observed equity trading day or otherwise the next
observed equity trading day. A headline after the last observed trading day
cannot map forward and is excluded. The exact title string, including casing,
punctuation, and negation context, is passed once to VADER. The compound score in
the range from -1 to 1 is the index input; VADER's negative, neutral, and
positive components are retained in memory for auditability.

Aggregation is deliberately two-stage:

1. Arithmetic mean of headline compound scores within each ticker/trading-day.
2. Arithmetic mean of the observed ticker-day means within each
   sector/trading-day.

This gives each ticker with news one equal sector weight regardless of how many
headlines it has. Ticker-days without headlines are omitted from the mean. The
saved output is a complete observed-equity-date by sector grid: when a whole
sector has no headlines, `sentiment_value` is missing, headline and observed
ticker counts are factual zeros, and no neutral score is imputed. A genuine
observed VADER mean of zero remains distinguishable because its counts are
positive and `sentiment_observed` is true.

This missing-news treatment is a transparent baseline choice for student review.
It does not carry news forward or assume that no news is neutral.

## Code, test, dependency, and output changes

### `src/sentiment.py`

Codex implemented:

- validation against the cleaned equity ticker-sector universe, which rejects
  cryptocurrency or unknown tickers;
- same-day/next-observed-equity-day headline mapping and the sample cutoff;
- per-original-headline VADER scoring with component and range checks;
- two-stage equal-ticker sector aggregation on a full daily grid;
- explicit news-coverage audit counts and missing-sector-day handling;
- a one-observed-trading-day lag helper that is tested and documented for later
  work but is not called by the R3 build or any portfolio code;
- in-memory R3 artifact construction and exact-filename CSV writing.

### `scripts/run_part_b.py`

The single Part B build command now loads cleaned equity headlines through
`src.etl`, runs R3 after the existing R2 portfolios, writes the sentiment CSV,
and prints headline, output, date, sector, and coverage counts. It does not run
fusion or feed sentiment into portfolio decisions.

### `tests/test_sentiment.py`

Eight deterministic tests cover exact text preservation, weekend-to-next-trading
day mapping, VADER component retention, rejection of cryptocurrency and
post-sample inputs, ticker-fair two-stage aggregation, the distinction between
missing news and an observed zero score, the full output grid and schema, the
future minimum trading-day lag, and required-file round-trip behaviour.

### Build dependency correction

The installed NLTK 3.10.1 introduced a current-directory import guard that
misclassified dependencies inside this repository's project-local Windows
virtual environment and blocked the documented project-root command. Codex did
not bypass the guard. It constrained the build-only requirement to
`nltk>=3.8,<3.10`, installed NLTK 3.9.4 locally, and downloaded the official
`vader_lexicon` to the user's NLTK data cache. NLTK remains absent from the
deployed app requirements. The lexicon is not committed; on a clean machine it
must be installed once with:

```text
python -m nltk.downloader vader_lexicon
```

## Actual reproduced output and independent checks

`python scripts/run_part_b.py` completed successfully from the project root. It
loaded 50,300 cleaned equity-price rows, 14,610 cleaned cryptocurrency-price
rows, and 146,836 cleaned supplied equity headlines. Existing portfolio outputs
were reproduced without changing portfolio logic.

The R3 build mapped and scored 146,830 headlines. Six cleaned headlines after
the final observed 2023-12-29 equity trading date could not map forward and were
excluded. It wrote:

- `results/data/sector_sentiment_index.csv`: 10,060 rows x 8 columns;
- 1,006 observed equity dates from 2020-01-02 through 2023-12-29;
- 10 equity sectors, each with five universe tickers;
- 9,832 sector-days with at least one headline and 228 zero-news sector-days;
- 37,962 observed ticker-days and a reconciled total of 146,830 headlines.

The exact saved columns are:

```text
date, sector, sentiment_value, headline_count, ticker_with_news_count,
sector_ticker_count, zero_news_ticker_count, sentiment_observed
```

Independent checks found no duplicate date-sector rows, no dates after the
approved sample, no sentiment values outside [-1, 1], and an exact match between
the 228 missing sentiment values and 228 zero-news rows. The saved headline
counts sum to 146,830. A separate regrouping of the headline scores reproduced
all 9,832 observed saved sector-day values with maximum absolute floating-point
difference `9.975e-17`. A multiset comparison against the eligible cleaned news
confirmed exact preservation of every original headline string.

For audit context, plain VADER produced compound scores from -0.9186 to 0.9552
and exactly 71,720 zero compound scores. A zero score is a model output for an
observed headline, not evidence that no headline exists and not a factual claim
that the news is economically neutral.

## Tests and requested checks

The final requested regression run produced:

- `tests/test_sentiment.py`: 8 tests passed in 0.083 seconds;
- `tests/test_foundation.py`: 7 tests passed in 0.065 seconds;
- `tests/test_portfolios.py`: 7 tests passed in 0.210 seconds;
- `tests/test_smoke.py`: imports OK and the live hosted course data load OK;
- `scripts/run_part_b.py`: completed and wrote all four required Part B model
  artifacts;
- Streamlit's documented `No runtime found, using MemoryCacheStorageManager`
  warnings appeared during non-Streamlit data loads and did not stop execution.

`scripts/check_handin.py` reported 20 passes, two reminders, and one failure. The
reminders are local compiled-Python clutter under the ignored virtual environment
and the report not yet being authored. The failure lists three SciPy/PyArrow test
fixture CSV/Parquet files under `.venv/`. The checker scans the ignored local
environment; those paths are covered by the repository's `.venv/` ignore rule
and are not submitted or tracked. This is the same checker false positive
recorded in R1-R2, not committed project data.

`git diff --check` exited successfully with no whitespace-error output. Git only
printed its Windows line-ending notices for three modified tracked text files.
The final `git status --short` showed the intended R3 changes only:

```text
 M requirements-dev.txt
 M scripts/run_part_b.py
 M src/sentiment.py
?? ai/log_v2_03_sentiment.md
?? results/data/sector_sentiment_index.csv
?? tests/test_sentiment.py
```

`git check-ignore -v` confirmed that each package fixture named by the hand-in
checker is excluded by `.gitignore`'s `.venv/` rule, and `git ls-files -- .venv`
returned no paths.

## Limitations and student reminders

- The supplied text consists of headlines, not full articles; important details
  and context are absent.
- Plain VADER is a general sentiment model and can misread financial terminology,
  entities, ambiguity, sarcasm, and context. Its many zero scores should not be
  over-interpreted.
- The sector index is descriptive. It does not prove causality, predict returns,
  establish investability, or support a recommendation.
- Sparse news is exposed through counts and missing values. Results can be
  sensitive to which tickers happened to receive coverage on a date.
- The index is contemporaneous descriptive evidence only. It is not used by the
  R3 portfolios. Any later trading or fusion phase must use sentiment from at
  least the preceding observed equity trading day, preserve missingness, and be
  evaluated out of sample.
- The student must review and approve the missing-news policy and method before
  writing the report, independently verify the evidence, and write all analysis
  and economic interpretation in their own words.

No fusion, portfolio-method change, report text, app, deployment action, Git
commit, causal claim, forecast, or recommendation was created in R3.
