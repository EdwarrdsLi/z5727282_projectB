# AI assistance log v2.07 — nine-fund HD expansion

## Student direction and authorship clarification - 14 August 2026

The student's original workflow already required an evidence-based extension
beyond the minimum baseline. After reviewing the marking criteria, the student
chose the HD objective and approved the final form of the extension: a nine-fund
shelf, three asset families, three portfolio methods, native-calendar evidence,
turnover and transaction-cost sensitivity. Codex helped implement and test that
approved design; it did not independently make the final product decision.

## Student request and approval

The student asked for a marking-criteria review, selected the goal of pursuing an
HD result, and approved the proposed expansion from two combined funds to a
nine-fund shelf:

- Equity only, Cryptocurrency only, and Combined;
- Equal Weight, Risk Parity, and long-only Minimum Variance for each family;
- native calendars, monthly target resets, and walk-forward OOS evaluation;
- turnover and an illustrative 10 bps one-way transaction-cost check.

The student retains responsibility for the report's analysis, economic
interpretation, critical reflection, recommendations, and final conclusions.

## Technical assistance provided

Codex extended the portfolio engine, deterministic artifact build, report
exhibits, results-only Streamlit app, and focused tests. The approved technical
conventions are:

- adjusted-close returns loaded only through the provided data layer;
- 252 complete prior observations for Equity and Combined funds;
- 365 complete prior observations for Cryptocurrency-only funds;
- 252/365 annualisation matched to the evaluated calendar;
- first observed native-calendar date of each month as the target reset date;
- long-only, fully invested, no leverage, zero annual risk-free rate;
- minimum variance solved with SLSQP from a deterministic starting point and a
  5% diagonal covariance shrinkage, followed by constraint and objective checks;
- turnover calculated as one half of the absolute change from drifted pre-trade
  weights to the new target, with initial establishment excluded;
- gross saved returns plus separate 0 and 10 bps one-way cost sensitivity.

## App and exhibit changes

The app now provides family-filtered method comparison, all nine fact sheets,
any-two-fund historical allocation on an overlapping rebased sample, fund-level
turnover/cost sensitivity, the existing equity-news sentiment evidence, and the
approved fusion comparison. Growth and drawdown figures are faceted by asset
family so native-calendar and equity-calendar evidence is not visually merged
into a single undifferentiated line chart.

## Verification record

- The Part B portfolio pipeline produced exactly nine funds and wrote the two new
  files `results/tables/fund_turnover.csv` and
  `results/tables/fund_transaction_cost_check.csv`.
- Seven focused portfolio tests passed after the engine extension.
- The portfolio, exhibit, and app test run passed its first 17 tests; the final
  Streamlit test process then encountered a Windows Tcl/Tk deallocator crash
  after Matplotlib tests. This was an environment-level process cleanup failure,
  not a failed assertion.
- The five Streamlit/app tests were rerun separately and all passed.
- The unified `scripts/run_part_b.py` entrypoint completed successfully from the
  hosted course inputs. It wrote 7,803 fund-return rows, 12,960 target-weight
  rows, 9 performance rows, 324 fund-rebalance turnover rows, 18 cost-check rows,
  17 report exhibits, and 9 fact sheets.
- The final core suite passed 37/37 tests and the separately run app suite passed
  5/5 tests (42/42 total).
- The corrected hand-in checker passed 21 mechanical checks. Its only reminders
  were local compiled-Python clutter and the intentionally absent student-authored
  report PDF.
- Independent artifact checks confirmed 9 funds, non-negative target weights,
  every target sum within floating-point tolerance of 1.0, no cost scenario with
  net growth above its corresponding gross growth, and no saved return date after
  2023-12-31.
- Growth, drawdown, risk-return, and a concentrated cryptocurrency Minimum
  Variance fact sheet were visually inspected after final regeneration. The
  risk-return chart was changed to three asset-family panels to prevent label
  overlap, and fact sheets omit zero-weight eligible assets from the displayed
  current-holdings table.

## Risks and limitations for the student to discuss in their own words

- Cryptocurrency results are extremely volatile and include very large historical
  drawdowns; high annualised returns do not remove that risk.
- Minimum Variance can be concentrated and sensitive to covariance estimation,
  even with the stated shrinkage and long-only constraints.
- The 10 bps cost scenario is illustrative, not a claim about achievable future
  execution costs.
- Backtest evidence is historical and limited to the approved period ending no
  later than 2023-12-31; it is not a forecast or causal claim.
- Sentiment fusion remains an equity-only design and did not outperform its base
  strategy on annualised return or Sharpe ratio in the saved historical sample.

The student must verify these points against the generated tables and express all
submitted interpretation and conclusions independently.