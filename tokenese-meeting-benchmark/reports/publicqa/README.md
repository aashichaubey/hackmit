# Published meeting-QA successor results

The three user-supplied datasets removed the need to wait for independent review of the old model-drafted labels. This successor integrates published human annotations and evaluates native extractive QA separately from dialogue response/attribution. The original frozen experiment is unchanged.

**The user changed the objective to exploratory savings with modest quality loss allowed.** The app now offers source-preserving transcript experiments, with raw fallback for unsupported formatting. Historical zero-loss gate failures remain below; they are not relabeled as passing.

## Fresh test: selected separator format

The selected compiler was frozen before opening 64 MeeQA test questions from **31 meeting families excluded from every training corpus we used**. It keeps every source character recoverable and makes no extraction call. No candidate edits or selections were based on the test answers.

| Metric | Raw | Compressed | Change |
|---|---:|---:|---:|
| Actual API input tokens | 115,639 | 112,222 | **2.95% fewer** |
| Actual output tokens | 4,408 | 5,458 | 23.82% more |
| Total input + output tokens | 120,047 | 117,680 | **1.97% fewer** |
| Actual cost | $0.0533084 | $0.0536216 | **0.59% higher** |
| Workspace-policy answer F1 | 41.37% | 43.55% | +2.18 points |
| Balanced answerability | 56.25% | 54.69% | −1.56 points |
| Answerable-question F1 | 42.12% | 37.10% | −5.02 points |
| Unanswerable accuracy | 40.63% | 50.00% | +9.38 points |
| Rejected source quotations | 2/64 | 5/64 | 3 more failures |
| Exact match | 20.31% | 25.00% | +4.69 points |

These results establish a small token-efficiency improvement on this measured workload, with mixed quality effects. Longer answers make this **a token saving, not a dollar saving**. The paired 95% meeting-family bootstrap interval for the workspace-policy F1 change is −6.32 to +10.64 points; the observed improvement is not proof of superiority. The exploratory five-point aggregate F1/answerability tolerance is met, but answerable-only performance and rejected quotations deserve attention. It is an experiment, not a general accuracy guarantee.

The original frozen native scorer reports 43.43% encoded F1 versus 41.37% raw and unchanged balanced answerability. During the raw arm, independent code review identified a source-label restoration mismatch with the UI. A timestamped amendment was recorded before encoded answers were inspected. The frozen code/results were retained, and a separate offline **workspace-policy** audit applied identical source-first grounding and rejection rules to both arms. It changes 25 score records, often only decoded label text, without any extra model calls or gold-based correction. Both reports are retained; the table above matches what the app actually displays.

[Original frozen results](heldout.json), [workspace-policy results](heldout-product-policy.json), [frozen artifact/request audit](heldout-audit.json), and [pre-result scoring amendment](heldout-scoring-amendment.json). All source hashes, 128 recorded requests, native scores, aggregates and exact reconstruction checks reproduce.

## Interactive exploration

The Language lab plots token savings against F1 loss and provides an adjustable loss-tolerance filter. The workspace offers **Minimal separator**, **Speaker blocks**, or **Raw transcript**; it shows the representation and local counts before answering, then actual provider usage. Each question uses one private answer call, with no extraction or retrieval. Unsupported/non-saving formats remain raw, and follow-up questions reuse the session representation. The original fact-memory experiment and V1 tabs remain available.

The separate block pilot saved 3.14% input tokens with a 3.86-point training F1 decrease. Its three rejected quotations remain visible. Named-speaker block support has integrity tests, not an independent quality benchmark.

## What was built

* Pinned-commit download and native training adapters for MeetingQA, MeeQA and MISeD.
* Exact answer-offset validation, multi-annotator alternatives, jointly required answer spans, overlap-union scoring, and explicit exclusions.
* Meeting-family leakage checks across repositories; test labels were opened only after the selected successor was frozen; validation and WOZ remain unused.
* Unicode token-overlap F1/IoU, exact match, answerability precision/recall/balanced accuracy, source grounding, per-dataset/answerability strata and per-meeting averages.
* Paired meeting-family bootstrap uncertainty, actual API input/output/cost, unknown-usage accounting and shared replay/budgets.
* Three speaker-label grammars plus a separately developed consecutive-speaker inheritance compiler. No extraction model, retrieval, or grammar legend is used.
* A separate MISeD raw-dialogue baseline with reference history and attributed segment scoring.
* Saved results and local compression preview in the application.

## Native extractive pilot

Each main pilot uses the same deterministic 32 training questions: eight answerable and eight unanswerable from each extractive dataset, at most two questions per meeting. These are development results, not held-out accuracy estimates.

| Experiment | Raw F1 | Encoded F1 | Raw balanced answerability | Encoded balanced answerability | Actual input reduction | Result |
|---|---:|---:|---:|---:|---:|---|
| `@N:` speaker notation | 27.51% | 25.63% | 50.00% | 46.88% | Unknown: raw parse failure | Rejected |
| Readable `Speaker N:` notation | 27.51% | 23.24% | 50.00% | 43.75% | Unknown: raw parse failure | Rejected |
| Remove decorative `& ` only | 29.86% | 26.11% | 50.00% | 46.88% | **2.69%** | Rejected |

The third experiment uses a 2048-token output cap for both arms after repeated raw truncation at 512. It has a new matched baseline; differences between experiments cannot be attributed solely to the grammar. It consumed 34,589 raw versus 33,659 encoded input tokens, and $0.0194628 versus $0.0187356 in measured workflow cost (**3.74% lower**). No extraction cost is omitted. One output in each arm contained a quotation absent from the source. Both arms correctly abstained on only 2/16 unanswerable questions.

The third experiment's paired F1 change is −3.76 percentage points; exploratory 95% meeting-family bootstrap interval: −9.47 to −0.20 points. This is not evidence of equivalent comprehension. Under the original zero-loss requirement, no candidates advanced. After the user explicitly allowed quality tradeoffs, the separator format was selected and tested under the separately declared exploratory protocol above. Historical gates and results remain unchanged.

## Consecutive-speaker inheritance

The parallel local study retains first/changed speaker labels and replaces a consecutive same-speaker label with `↳`. Across 13,061 unique retained training contexts, there were **zero reconstruction failures**. MeeQA context tokens fell from 12,344,503 to 11,448,083: **7.26% saved**. MeetingQA saw no savings. This is tokenizer-only measurement on the adapter-retained subset, not actual API usage or full-corpus qualification.

A fixed four-question MeeQA development diagnostic measured **4,894 → 4,430 input tokens (9.48% less)** and $0.0022504 → $0.0021608 (**3.98% lower cost**). Raw F1 was 20%; encoded F1 was 0%. One question regressed and three tied. That tiny diagnostic neither satisfies the 32-question advancement criterion nor establishes general-purpose quality. Quotes containing inherited labels are restored using their complete source context; ambiguous restoration fails instead of choosing an interpretation using the gold answer.

## MISeD dialogue baseline

Eight raw-transcript training turns from eight meeting families: four initial questions, four with preceding **published reference history**, not generated conversation.

| Metric | Result |
|---|---:|
| Response token-overlap F1 | 35.62% |
| Attribution-set F1 | 26.75% |
| Attribution IoU | 20.53% |
| Invalid citation outputs / API errors | 0 / 0 |
| Actual input / output tokens | 100,036 / 2,837 |
| Actual cost | $0.0445536 |

Lexical response similarity is not a faithfulness judgment. Citation overlap measures agreement with published attribution, not entailment. No compressed dialogue candidate or fully manual WOZ evaluation is claimed.

## Dataset handling and scope

The catalog contains 2,976 MeetingQA logical questions, 10,309 MeeQA logical questions, and 2,817 MISeD turns. Duplicate annotator records are grouped; answer spans within one annotation are jointly required, while annotators provide alternatives. Duplicate/overlapping gold intervals count once in scoring while original annotations remain intact.

Excluded: 23 MeetingQA rows with invalid offsets; three MeetingQA and 3,102 MeeQA logical groups with answerability disagreement; 105 MISeD turns with malformed or omitted attribution bounds. These exclusions are disclosed and are not selected by model outcomes. The resulting consensus subset must not be represented as the entire published benchmark. Missing MISeD bounds are not silently interpreted as zero.

Meeting-family overlaps in the training sources: MeetingQA/MeeQA 33; MeetingQA/MISeD 23; MeeQA/MISeD 56. Corpus split names alone are insufficient leakage protection. These metrics use our Unicode tokenizer and are not exact reproductions of paper leaderboard results.

MeetingQA includes CC BY-NC-SA 4.0. No top-level license was found in the pinned MeeQA/MISeD trees. Downloaded source data and raw response rows remain local and are ignored by version control; these reports summarize research results without republishing the corpora.

Sources: [MeetingQA repository](https://github.com/adobe-research/meetingqa), [MeetingQA paper](https://aclanthology.org/2023.acl-long.837/), [MeeQA repository](https://github.com/reutapel/MeeQA), [MISeD repository and paper citation](https://github.com/google-research-datasets/MISeD). MISeD is semi-automatically created and human-verified; only its WOZ subset is described as fully manual.

## Reproduction

From the project root:

```sh
.venv/bin/python -m tokenese.evo_public_fetch
.venv/bin/python -m tokenese.evo_public_data
.venv/bin/python -m tokenese.evo_public_report
.venv/bin/python -m tokenese.evo_public_inheritance --audit
.venv/bin/python -m tokenese.evo_transcript_search
.venv/bin/python -m pytest -q
```

The offline audit recomputes all three pilot scores and aggregate usage from recorded rows, verifies source hashes and exact reconstruction, and reproduces their reports. Inheritance audit also verifies that today's compiler recreates every recorded prompt. A quote-restoration helper was added after its dispatch manifest; that helper changes the full module hash but not the verified compiler inputs. The original manifest is retained rather than rewritten.

Paid experiment entry points require `--run-api`; successful identical requests replay the ledger. The three 32-question pilots are `tokenese.evo_public_eval` with no grammar flag, `--readable`, or `--minimal`. Dialogue uses `tokenese.evo_public_dialogue --run-api`. The four-question inheritance dispatch is intentionally protected against module/config drift; use its offline audit to reproduce this completed run.

This continuation made 142 actual API calls, adding $0.1236768 in known charges. The shared ledger totals 3,321 actual calls, $1.3931312 known charges, and $1.565938 reserved/spent including 33 historical unknown-usage calls. At that checkpoint discovery reached its original 1,500-call default. The subsequent explicit successor allocation and fresh evaluation are accounted below. Unknown charges are not reported as zero, and replayed results are not new spending.

Verification: **95 tests passed**. Parallel code review findings were corrected and regression-tested. The running Language lab was checked with the new public benchmark panel visible.

## Final successor accounting and commands

The follow-up used 13 new block-candidate calls plus 128 fresh paired-test calls. The shared ledger now records **3,462 actual calls**, **$1.5091868 known charges**, and **$1.6819936 reserved/spent**, including 33 historical unknown-usage calls. The original 3,500-call/$10 total limits remain intact. The original discovery counter remains 1,500; a separately recorded immutable allocation allowed up to 48 additional successor calls while retaining 128 final-evaluation calls. Only 13 of that allocation were used. No failed or unknown-usage calls were erased.

```sh
# Reproduce original and corrected fresh-test metrics without model calls:
.venv/bin/python -m tokenese.evo_public_holdout_audit
# Verify the original frozen evolution experiment still reproduces:
.venv/bin/python -m tokenese.evo_benchmark --audit
# Open the complete application:
.venv/bin/streamlit run app.py
```

The implemented loop is feedback-driven grammar search and repair; it does not fine-tune model weights or claim formal reinforcement learning. An independent agent review of four development regressions found two substantive answer regressions and two quote/reference-boundary effects; [the analysis](failure-analysis.md) is disclosed as agent judgment and does not replace human dataset labels.

Final verification: **125 tests passed**; all eight fresh-test audit checks pass; legacy frozen results still reproduce. The running app was restarted and its tradeoff chart and transcript workspace were visually checked. Runtime versions and source hashes are in [verification.json](verification.json).
