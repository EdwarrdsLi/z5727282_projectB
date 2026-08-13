# AI use notes

## Authorship and review status

Codex assembled the first version of these notes from the dated logs, tests and
deployment record. On 14 August 2026 I clarified the division of responsibility
using the detailed workflow I had supplied before implementation. Codex updated
the log pack at my direction without altering the original prompts, technical
results or error records.

I adopted and directed the project's substantive workflow. AI helped convert my
requirements into initial code, tests, figures, app components and a report
framework. I reviewed the outputs, made the consequential choices, checked the
evidence and substantially rewrote the report. I remain responsible for a final
read of these notes and for ensuring that every statement matches what I did.

## How I directed the AI

Before implementation I supplied a detailed staged workflow. It divided the work
into bounded roles for project rules, portfolio construction, sentiment,
fusion, innovation, exhibits, the investor app and final delivery. Each stage
used the same cycle: build an initial artifact, identify risks through a red-team
review, and leave the final verdict to me.

The workflow set the main project decisions and acceptance criteria. These
included adjusted-close returns, native equity and cryptocurrency calendars,
genuine prior estimation data, look-ahead-safe monthly targets, 252/365
annualisation, original-text VADER scoring, equal-weighted sector aggregation, a
one-observed-equity-day sentiment lag, honest reporting of negative results, a
results-only Streamlit app, reproducible artifacts and a public deployment.

I used separate prompts for each phase and restricted the files that could be
changed. I asked for focused tests, actual failure reporting, no unsupported
claims and no commit until I approved it. AGENTS.md records these project rules.
The dated logs preserve the prompts, implementation assistance, risks, checks,
corrections and later decisions.

## What the AI helped with

Codex helped translate the staged requirements into working technical artifacts.
Its assistance included:

- adapting my verified Part A cleaning and feature foundation;
- implementing initial versions of the walk-forward portfolio methods and
  focused tests;
- implementing the equity-sector VADER index and lagged fusion experiment;
- generating result tables, figures, fact sheets, turnover and illustrative
  cost checks from the saved evidence;
- building and debugging the results-only Streamlit app;
- identifying technical and marking-criteria gaps for my review;
- supporting Git, deployment troubleshooting and mechanical hand-in checks; and
- producing early report outlines, draft paragraphs and Word-formatting support.

These outputs were drafts and implementation assistance. AI did not have
authority to decide the final product, select the final interpretation or submit
its first report wording as my work.

## What I decided and did

I chose or approved the project direction and final design. This included the
fund families and methods, the walk-forward evidence standard, native-calendar
treatment, portfolio constraints, sentiment scope, lag and missing-news rules,
fusion design, innovation target, app journey and deployment boundary.

After reviewing the marking criteria, I approved the final nine-fund shelf with
Equal Weight, Risk Parity and long-only Minimum Variance across equity,
cryptocurrency and combined families. I approved the turnover and illustrative
transaction-cost checks. I also approved the 0.5 sentiment tilt and decided to
retain its weaker result rather than change the specification until it appeared
successful.

I reviewed the app in the browser and supplied screenshots that exposed display
problems not detected by code tests. I personally completed the actions that
required my accounts: creating and connecting the GitHub repository,
authorising Streamlit, selecting the cloud settings, deploying the app and
making the repository public.

For the report, AI provided an initial framework and drafting support. I reviewed
the document section by section, checked claims against the project artifacts,
changed wording that did not reflect my understanding and substantially rewrote
the analysis, limitations, recommendations, conclusions and AI-workflow
explanation. The final argument and submitted wording remain my responsibility.

## How I checked the work

I did not treat an AI answer as evidence. The checks included:

- focused unit and integration tests for calendars, return construction,
  temporal separation, portfolio constraints, metrics, sentiment aggregation
  and lagging, fusion scope, schemas and the app boundary;
- a final suite of 42 successful tests;
- all 22 mechanical hand-in checks passing, apart from the reminder to exclude
  generated cache files from the ZIP;
- a full pipeline run from supplied hosted data to regenerate the committed
  artifacts;
- checks of fund counts, target sums, non-negative weights, date caps, required
  files and transaction-cost arithmetic;
- comparison of report and app figures with the saved CSV evidence; and
- logged-out checks of the public GitHub repository and Streamlit app.

## Where AI output was wrong or risky

The first output was not always correct. Restricted-environment commands
initially failed while importing Python's standard encodings module. The
hand-in checker also scanned ignored virtual-environment fixtures until the
scope was investigated.

A horizontal Streamlit chart still had its title on the wrong physical axis
after the first attempted fix. My screenshot exposed the problem, after which
the mapping was corrected and focused tests were rerun. Other screenshots showed
truncated metric-card labels and a cloud dark-theme conflict. Those display
problems were corrected without changing calculations.

AI-generated report prose also created an authorship and verification risk. I
did not treat the first Word draft as final. I checked the figures and claims,
removed or changed wording that did not match my understanding and rewrote large
parts of the report. The failed sentiment result was retained instead of being
tuned away or described as a success.

## Remaining limitations I understand

The project is a historical prototype using evidence only through 2023. Results
depend on the selected assets, estimation windows, covariance treatment,
rebalance rule, cost assumptions and missing-headline policy. VADER uses
headlines rather than full articles and may miss finance-specific meaning.
Cryptocurrency volatility and drawdowns remain very large. The 10 basis point
cost case is illustrative, not a forecast of execution costs. None of the
results is a live recommendation or promise of future returns.
