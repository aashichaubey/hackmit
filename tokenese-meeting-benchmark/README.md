# Tokenese meeting benchmark

## Key findings

- **The best reliable optimization was batching, not a new symbolic language.** On the original synthetic benchmark, answering four known questions together reduced actual input tokens by **65.7%** (20,370 to 6,990) while preserving the exact 119/120 answer pattern. The same test on the stress set saved **66.6%** with no change in correctness.
- **Compression gains shrink once the whole request is counted.** In the short-note benchmark, meeting content was only 14.5% of input tokens; instructions, schemas, questions, and API framing dominated. A compact representation can therefore look impressive in isolation without materially reducing end-to-end usage.
- **The evolved fact language did not satisfy its quality gate.** It cut answer-input tokens by **85.6%** versus raw excerpts, but achieved 105/120 exact field-preservation answers against a required 119/120. It also did not beat the initial concise-English seed, so the experiment remains unqualified.
- **Source-preserving transcript formatting produced a small real-world saving with mixed quality.** On a fresh 64-question MeeQA test, the selected separator format used **2.95% fewer input tokens** and **1.97% fewer total tokens**. Workspace-policy F1 rose from 41.37% to 43.55%, but answerable-only F1 fell 5.02 points, rejected quotations increased, and dollar cost rose 0.59% because outputs were longer.
- **Token savings are not automatically cost or quality savings.** The strongest product direction is to batch known questions, preserve source text, measure complete request usage, and expose the quality tradeoff instead of claiming general-purpose lossless compression.

These findings are exploratory. The synthetic corpus is small, the original evolution labels were model-drafted, and the published-corpus confidence interval does not establish superiority. Full measurements and limitations are in the [evolution report](reports/evolution/README.md), [published meeting-QA report](reports/publicqa/README.md), and [input-token research note](reports/input_token_research.md).

## Evolution redesign

The app now opens in a **Language lab** with recorded generations, an accuracy/token frontier, inspectable grammar rules, real failure/repair traces, and frozen held-out results. **Meeting workspace** keeps extraction and compilation in session memory for follow-up questions. The V1 benchmark and comparison remain available in their own tabs.

The new classifier preserves literal owners, date modifiers, negation and proposal/decision status, and reports unsupported or ambiguous content. In the original fact-memory mode, automatic encoded answering requires a compatible qualified frozen manifest; otherwise that mode uses raw notes. An explicit research-preview checkbox can inspect an unqualified frozen grammar.

See [measured evolution results](reports/evolution/README.md), [machine-readable metrics](reports/evolution/metrics.json), [evaluation protocol](docs/evaluation-protocol-evolution.md), and [development learnings](docs/evolution-learnings.md). The evaluation separates bounded-language comprehension from broader source-question usefulness, with classifier precision/recall, category errors, paired regressions, and extraction-inclusive workflow costs.

```sh
.venv/bin/python -m tokenese.evo_search --local --seed 42
.venv/bin/python -m pytest -q
.venv/bin/python -m tokenese.evo_benchmark --audit
.venv/bin/streamlit run app.py
```

The completed experiment is frozen: its validation and test cannot be used to resume mutation or adaptively rerun test. Paid commands for a fresh, separately recorded experiment require `--run-api`; the persistent SQLite ledger enforces 3,500 actual calls / $10 globally, plus a 1,500-call discovery ceiling. Exact public prompts, usage, errors and replay attribution are saved. Private pasted notes are never sent to the research cache or reports.

```sh
# Before selection is frozen, on a fresh experiment:
.venv/bin/python -m tokenese.evo_search --run-api --seed 42 --max-calls 3500 --max-usd 10
.venv/bin/python -m tokenese.evo_benchmark --ablations --run-api
.venv/bin/python -m tokenese.evo_benchmark --select-validation --run-api
.venv/bin/python -m tokenese.evo_benchmark --frozen reports/evolution/frozen.json --split test --run-api
.venv/bin/python -m tokenese.evo_report
```

The original evolution corpus annotations are provisional model drafts, not independently reviewed ground truth. The report distinguishes measured token reduction from quality qualification and does not claim general-purpose lossless compression.

A separate [classifier follow-up](docs/classifier-next-study.md) tested two development-only fixes without changing the frozen experiment. The cheaper source-span selector lost answer quality and remains unqualified. Its paired errors, fallback coverage, and complete workflow costs are also visible in the Language lab. Reproduce its four-source pilot with `.venv/bin/python -m tokenese.evo_classifier_next --run-api --cases 4`; it shares the original budget and replays identical requests.

The [complete-source grammar study](docs/source-grammar-study.md) preserves all source information with reversible structural edits and no extraction call. It saved only 0.0506% of answer input on development and failed the quality check. A repeated-call diagnostic identified a missing accepted name variant and unstable discussion/decision answers. Both results are available in the Language lab; neither activates a new product route. Commands: `.venv/bin/python -m tokenese.evo_source_study --run-api` and `.venv/bin/python -m tokenese.evo_stability --run-api`. Omit `--run-api` from the first for a free local screen.

The offline annotation audit flags development labels for source review without changing them: `.venv/bin/python -m tokenese.evo_label_audit`. The [concise review packet](reports/evolution/label-review.md) includes three examples and full excerpts; the complete queue is `reports/evolution/label-review.json`. No model answers are included, and a review candidate is not automatically treated as an annotation error.

## Published meeting-QA successor

The user-supplied MeetingQA, MeeQA and MISeD corpora now have pinned downloads, native annotation adapters, source-family checks and separate extractive/dialogue evaluations. Published annotations remove the old review dependency for this successor; the original model-drafted experiment remains historical.

See [results and limitations](reports/publicqa/README.md), [protocol](docs/publicqa-protocol.md), and [reproducibility audit](reports/publicqa/audit.json). The fresh 64-question, 31-family test of the selected separator format saved **2.95% actual input tokens** and **1.97% total tokens**. Workspace-policy answer F1 was **43.55% versus 41.37% raw**, with mixed submetrics: balanced answerability fell 1.56 points, answerable-only F1 fell 5.02 points, and source-quotation rejections increased from two to five. Dollar cost rose 0.59% because outputs were longer. These are exploratory tradeoffs, not an equal-accuracy guarantee.

The Language lab now plots savings against quality loss and lets you filter by a chosen F1 tolerance. The default workspace offers source-preserving transcript formats with one private answer call and no extraction; unsupported text stays raw. Historical strict-gate results remain unchanged, alongside the original fact-memory mode and V1 views. MISeD has a separate eight-turn raw response/attribution baseline.

```sh
.venv/bin/python -m tokenese.evo_public_fetch
.venv/bin/python -m tokenese.evo_public_data
.venv/bin/python -m tokenese.evo_public_report
.venv/bin/python -m tokenese.evo_public_inheritance --audit
.venv/bin/python -m tokenese.evo_public_holdout_audit
```

## Preserved V1 demo

A Streamlit report for comparing raw meeting notes, compact English facts, Tokenese facts, LLMLingua-2, and The Token Company's Bear-2 on meeting questions. Extraction and answering use pinned `gpt-4.1-mini-2025-04-14`.

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test,bear]'
export OPENAI_API_KEY=...
export TOKEN_COMPANY_API_KEY=...
python -m tokenese.benchmark
python -m tokenese.benchmark --run-api --smoke
python -m tokenese.benchmark --run-api --include-bear --include-deletion
python -m tokenese.stress
python -m tokenese.diagnostics
python -m tokenese.grade_report
streamlit run app.py
```

The keys may instead be placed in a local `.env` file. It is ignored by Git. The first benchmark command is a free local token screen. The smoke command uses ten development meetings and 30 questions and writes `reports/smoke.json`. The full API benchmark makes paid calls and needs OpenAI credentials. `--include-bear` also needs Token Company credentials. `--include-deletion` loads the local LLMLingua-2 checkpoint, which can take time and disk space. The browser reads `reports/latest.json` without rerunning the benchmark. Pasted notes stay in Streamlit session state.

The Tokenese profile is used interactively only after it qualifies; otherwise compact English is selected. Full prompt token counts include the legend. Actual API usage includes schema overhead. See `docs/evaluation-protocol.md` for scoring rules.
