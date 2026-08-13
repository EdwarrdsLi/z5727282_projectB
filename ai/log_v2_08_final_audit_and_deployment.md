# AI workflow log v2.08 — final audit, report support and deployment

## Student direction and authorship clarification - 14 August 2026

The project direction, acceptance criteria and final decisions remained with the
student throughout this phase. Codex supported the audit, Git operations,
deployment troubleshooting, Word structure and early report drafts. The student
performed the authenticated GitHub and Streamlit actions, approved changes,
checked the live app and later supplied a substantially rewritten report.
Codex's report contribution was an initial structure and drafting framework plus
technical and formatting checks. The current report wording must not be
described as an unchanged AI draft.

## Phase and date

- Phase: final marking-criteria review, deployment, public-access audit, README,
  and AI workflow-pack completion
- Dates: 9–13 August 2026 (Australia/Sydney)
- Tool: Codex

## Student requests

The short controlling prompts in this phase included:

> 我想让你自己检查一下这个project哪有不足

> 冲击HD

> 批准

> 提交并部署

> 我把 GitHub 仓库改为 Public，检查一下

> 下一步需要提交并推送 README。

The student also asked for Word report drafts and later asked for a complete
report with wording intended to appear less AI-generated. That request creates
an academic-authorship risk. Codex report output can only be drafting and
formatting assistance: the student must verify the evidence and personally write
the submitted interpretation, reflection, recommendations, and conclusions.

## Marking-criteria review and approved extension

Codex reviewed the implemented project against the Part B criteria and identified
that the two completed combined funds met the minimum product requirement but did
not fully use the opportunity in the 30% Innovation & Data-Driven Results band.
The student selected the goal of pursuing an HD and approved the proposed
nine-fund extension. The implementation and technical checks are recorded in
`ai/log_v2_07_hd_fund_shelf.md` and commit `c2f5a50`.

The resulting shelf contains Equity, Cryptocurrency, and Combined families,
each implemented as Equal Weight, Risk Parity, and long-only Minimum Variance.
It adds native-calendar evaluation, family-specific annualisation, fund-level
turnover and an illustrative 10 basis-point one-way transaction-cost check. The
app, figures, fact sheets, and allocation lab were updated to expose this
evidence without converting historical results into recommendations.

## Deployment assistance and student-controlled actions

Codex prepared and pushed the repository. The student performed the browser-only
steps requiring personal authentication: creating the GitHub repository,
authorising Streamlit Community Cloud to access it, selecting `main`, selecting
the root `streamlit_app.py`, using Python 3.13 as specified in the course deploy
guide, and clicking Deploy.

The initial deployment workflow exposed two access issues:

1. Streamlit initially could not resolve the private repository until the student
   linked and authorised the correct GitHub account. The pasted URL also had to
   point to the Python entry file when the URL-based picker was used.
2. The repository remained private during development. At final audit the student
   changed it to public, as required for marking.

The first live app screenshot also exposed a theme problem: Streamlit Cloud used
a dark theme while the application's custom surface remained light. Headings and
inactive tabs had insufficient contrast. Codex added `base = "light"` to
`.streamlit/config.toml`, ran the focused five-app-test suite successfully, and
pushed commit `c545136`. The refreshed screenshot showed the intended readable
light design.

Earlier visual review also found truncated metric-card labels and incorrect
physical placement of the horizontal asset-mix axis title. Codex replaced fixed
card columns with responsive wrapping and corrected the axis mapping after the
student's screenshots showed that the first attempted label fix was incomplete.
Values, calculations, fund methods, and data were unchanged by these display
fixes.

## Report assistance and boundary

Codex created Word-format report drafts and formatting support in `report/`.
During this audit those documents remained untracked and no report draft was
included in a Git commit or pushed to GitHub. The report has not been treated as
student-authored merely because a Word file exists. Before submission, the
student must independently verify every number and claim, remove unsupported
material, and express the analysis, economic interpretation, reflection,
recommendations, and conclusions in the student's own words. Avoiding an AI
detector is not a valid substitute for genuine authorship and disclosure.

## Final README work

The original README still contained starter instructions such as `FIRST: rename
this folder`. At the student's request, Codex replaced it with a final public
repository explanation covering:

- the nine-fund product and three portfolio methods;
- the walk-forward, calendar, constraint, sentiment, and fusion design;
- verified historical metrics read from committed result tables;
- app sections, repository structure, setup and reproduction commands;
- the 42-test suite and hand-in command;
- material data, backtest, sentiment, allocation, and cryptocurrency limitations;
  and
- the public GitHub and Streamlit URLs.

No AI log was changed during the README-only request because the student
explicitly excluded project-log changes. The README was later committed alone as
`4f3f6b0` (`docs: finalise project README`) and pushed to `origin/main`.

## Final verification performed

Codex performed the following checks on 13 August 2026:

- `python scripts/check_handin.py`: **22 checks passed**, with one reminder to
  delete or exclude generated `__pycache__/` directories and `*.pyc` files before
  creating the ZIP;
- `python -m unittest discover -s tests -p 'test_*.py' -v`: **42 tests passed** in
  12.197 seconds;
- anonymous GitHub API request: HTTP 200, `private=False`,
  `visibility=public`, default branch `main`;
- Streamlit request to `https://z5727282-projectb.streamlit.app/`: HTTP 200 with
  no detected sign-in interstitial;
- final local and remote `main` SHA after the README push:
  `4f3f6b0231c586a34b7dcf11679888b8b79be56c`; and
- README whitespace validation: `git diff --check` passed before commit.

Streamlit emitted its expected bare-mode cache and missing-script-context
warnings during tests. No test failed and no app exception was reported.

## Corrections and risk controls

- Public-access status was checked anonymously instead of inferred from the
  student's logged-in browser.
- Local and remote commit IDs were compared after each final push.
- The cloud app uses committed precomputed `results/` artifacts and does not fit
  portfolios or run VADER at page load.
- Python 3.13 was selected for Streamlit Cloud according to the supplied deploy
  guide; the local PyCharm interpreter version did not determine the cloud
  runtime.
- Negative sentiment-fusion results and high cryptocurrency drawdowns remained
  visible rather than being hidden or optimised away.
- Report files were kept outside the repository commits pending student review.
- The prompt-log template's worked example was removed before hand-in so it could
  not be mistaken for the student's real evidence.

## Work performed in this AI-pack completion

Codex created `ai/AI_NOTES.md`, created this final log, and removed the disposable
worked example from `ai/prompt_log_template.md`. The initial AI-notes text is
explicitly labelled as AI-assisted and requires the student's personal review.
No source code, result artifact, report document, context file, deployment
setting, Git commit, or external service was changed in this documentation step.

## Student actions still required

- Read `ai/AI_NOTES.md` against the dated logs and rewrite any reflection that is
  not an accurate statement of the student's own understanding.
- Complete and personally verify the report; submit `report/report.pdf` in the
  required form.
- Close Word and exclude temporary lock files, `.venv`, `.git`, `__pycache__`, and
  `*.pyc` when building the final ZIP.
- Rerun the test suite and hand-in checker after any later project change.
- Confirm both public URLs in a logged-out browser, then submit the ZIP, GitHub
  URL, and live Streamlit URL through the required course channels.

## Report collaboration follow-up - 13 August 2026

The student asked Codex to collaborate on completing the report. Codex compared
the three existing Word files against the brief and selected
`report_polished_ai_assisted.docx` as the most complete review base: it contained
approximately 4,400 words, five tables, fifteen embedded figures, the required
fund/sentiment/fusion exhibits, references, and nine appendix fact sheets. Codex
did not treat that file as student-authored or ready to submit.

The student explicitly approved three substantive positions after Codex checked
them against the saved project evidence:

1. the target user is a retail investor with basic financial knowledge who needs
   a clearer way to compare systematic fund methods;
2. Combined Risk Parity showed the strongest historical risk-adjusted balance
   within the combined-fund family, without being presented as a forecast or
   universal recommendation; and
3. the report's three future recommendations should address portfolio stability
   and concentration, finance-specific sentiment, and realistic production and
   trading controls.

Codex then generated a single `report/report.docx` working copy from the complete
review base. This edit was mechanical and structural: it repaired title-page
character corruption, corrected section 3.1, 3.2 and 4.2 to Heading 2, applied
the Caption style to numbered table and figure captions, and replaced the prior
AI-review banner with a visible working-draft notice. No result number, table,
figure, recommendation, or approved conclusion was changed. The working-draft
notice remains because paragraph-level student review and the final authorship
check have not yet been completed.

When the student opened this version in Word, References began on page 18 and the
fact-sheet appendix on page 19. This showed that the narrative presentation was
well above the brief's ten-page maximum. Codex used the brief's explicit allowance
for exhibits in an appendix rather than shrinking the font. It moved the verified
performance table, six required figures, and fusion table to a new
`Appendix A. Required exhibits`; renamed the nine fact sheets as Appendix B; and
updated sections 3 and 4 to reference the relocated exhibits. All five tables,
fifteen embedded images, captions, sources, result numbers, analytical paragraphs,
and nine fact sheets remained present. Word held the open report during the first
save attempt, so Codex saved and structurally audited a temporary review copy,
waited until Word was closed, and then wrote the verified structure back to the
single `report/report.docx` working file. The student still needs to open the new
version and report its updated page boundaries for the final ten-page check.