# AI workflow log v2.00 - R0 project constitution

## Student direction and authorship clarification - 14 August 2026

The student supplied the initial project workflow, integrity rules, technical
red lines and acceptance checks before implementation. Codex converted those
requirements into the first structured wording of AGENTS.md; it did not choose
the project purpose, evidence standard or division of responsibility. The
student reviewed the constitution, retained the rules that matched the intended
project and remained the final authority for later design choices.

## Phase and date

- Phase: R0 project constitution
- Date: 2026-08-02 (Australia/Sydney)
- Tool: Codex

## Student request

The student asked Codex to complete only the R0 constitution phase. The permitted
changes were limited to `AGENTS.md` and this log. The student required a
project-specific Codex constitution covering responsibility for final choices and
writing, AI assistance boundaries, data and backtest safeguards, sentiment timing,
deployment constraints, reproducibility, tests, and honest reporting. The student
also prohibited implementation work, pipeline or test execution, changes to
`CLAUDE.md`, and creation of a Git commit.

## Material reviewed before editing

Codex fully read:

- `PROJECT_BRIEF.md`
- the original placeholder `AGENTS.md`
- `README.md`
- `SUBMISSION_CHECKLIST.md`
- `context/DATA_GUIDE.md`
- `context/project_context.md`
- `context/verify_ai_output.md`

No file under `context/` was edited.

## Work performed by Codex

Codex replaced the placeholder `AGENTS.md` with instructions tailored to Project
B. The constitution records:

- the student's authority over final product, methodology, innovation, design,
  and conclusion choices;
- the student's responsibility to personally write the analysis, economic
  interpretation, reflection, recommendations, and conclusions;
- the permitted Codex role in code, tests, technical checking, debugging,
  documentation structure, formatting, and clearly labelled drafting support;
- the prohibition on presenting AI-generated analysis or economic interpretation
  as student-authored work;
- immutable context files, controlled data access, repository safety, the
  2023-12-31 coverage cap, adjusted-close returns, native-calendar return
  calculation, and correct subsequent alignment;
- look-ahead-safe portfolio and sentiment rules, a prior estimation period,
  monthly-or-less-frequent rebalancing, and the equity-only scope of news
  sentiment;
- precomputed Streamlit artifacts and the prohibition on runtime backtests,
  VADER, or NLTK;
- testing, deterministic and reproducible outputs, independent sanity checks,
  source and number verification, and candid disclosure of errors, limitations,
  uncertainty, solver failures, and negative findings.

Codex also created this file to preserve a transparent record of the R0 work.

## Choices deliberately not made by Codex

R0 did not select the product name, target customer, fund lineup, optimisation
methods, constraints, estimation window, rebalance dates, costs, missing-sentiment
treatment, fusion method, innovation, or design system. These are consequential
project decisions reserved for the student.

Codex did not draft report analysis or economic interpretation. No pipeline,
portfolio, sentiment, fusion, report, or app implementation was changed.

## Risks considered and controls added

- Look-ahead risk: weights must use past data, out-of-sample evaluation must
  follow a prior estimation period, and sentiment must be lagged by at least one
  trading day.
- Calendar risk: equity and cryptocurrency returns must be calculated separately
  before alignment, using adjusted close and appropriate annualisation.
- Scope risk: equity-news sentiment cannot be transferred to cryptocurrencies.
- Reproducibility risk: required artifacts, focused tests, deterministic settings,
  schema/date checks, and a documented build command are required.
- Deployment risk: Streamlit must read committed, precomputed `results/` files and
  cannot run backtests, VADER, or NLTK.
- AI integrity risk: invented citations, numbers, results, and test outcomes are
  prohibited, and the student's own writing and final choices are explicitly
  protected.

## Verification for this phase

Codex reviewed the saved contents of both permitted files and inspected their Git
diff and `git status --short`. The first whitespace check found one extra blank
line at the end of `AGENTS.md`; Codex removed it. The final whitespace check passed,
and Git status showed only modified `AGENTS.md` and new
`ai/log_v2_00_constitution.md`. The data pipeline and tests were intentionally not
run during R0. No Git commit was created.

## Student follow-up

The student should review this constitution and change any rule that does not
match their intended workflow. In later phases, the student should record their
own final methodological choices and corrections in the relevant AI logs and
personally write the report's analysis and economic interpretation.