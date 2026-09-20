# Tokenese evolution — measured results

Language status: **incomplete_evaluation**. Product status: **incomplete_evaluation**. Accounting: **incomplete**.

On the frozen provisional test corpus, the selected grammar used **85.6% fewer answer-input tokens than raw excerpts**, with **105/120** exact field-preservation answers. Same-fact input savings were **0.4% versus compact English** and **27.9% versus explicit prose**.

The test language metric gate **failed**: 105/120 against the 119/120 threshold, with 13 new critical errors relative to at least one English baseline. Independent annotation review is an additional requirement, not the only missing gate.

**This is not a qualified general-purpose compression claim.** Public annotations were model-drafted and have not been independently reviewed. Raw-excerpt savings include fact extraction; only the same-fact comparison isolates encoding. The product retains raw notes for unsupported content, uncertainty, or a missing qualified grammar.

Context-only local token reductions (excluding question/instruction/schema overhead): raw **91.1%**, compact English **0.7%**, prose **40.0%**.

## Frozen language comparison

| Split | Representation | Exact answers | Actual input tokens | Actual output tokens | Answer cost |
|---|---|---:|---:|---:|---:|
| validation | evolved | 98/120 | 51620 | 1297 | $0.022723 |
| validation | raw | 64/120 | None | None | unknown |
| validation | prose | 87/120 | 69860 | 1306 | $0.030034 |
| validation | english | 98/120 | 51370 | 1297 | $0.022623 |
| validation | initial_seed | 98/120 | 51620 | 1297 | $0.022723 |
| test | evolved | 105/120 | 45989 | 1316 | $0.020501 |
| test | raw | 42/120 | 318339 | 1728 | $0.118926 |
| test | prose | 90/120 | 63759 | 1319 | $0.027614 |
| test | english | 107/120 | 46189 | 1315 | $0.020580 |
| test | initial_seed | 105/120 | 45989 | 1316 | $0.020501 |

These are provider input/output measurements including schema overhead. Replayed rows retain original provider usage. Logical workflow cost is separate from actual research spending; replay is not a provider cache hit.

## Broader source-question / classifier results

| Test route | Exact source-question answers | Unsupported answers on unknowns | Wrong/missing abstentions | Input tokens |
|---|---:|---:|---:|---:|
| evolved | 64/120 | 21 | 1 | 317043 |
| prose | 50/120 | 12 | 34 | 171353 |
| english | 50/120 | 13 | 34 | 168193 |
| raw | 64/120 | 21 | 1 | 317043 |

Classifier exact semantic-tuple precision **4.8%**, recall **1.7%**, F1 **2.5%**. Duplicates count. This is a literal-preservation diagnostic against provisional gold; harmless paraphrases can fail it. Source-question QA is the complementary usefulness measure. Extraction metrics were available for 12/12 excerpts; failed extractions still contribute all their gold facts as recall misses.

Frozen-route diagnostics: `{"empty_or_unmatched_source": 6, "unsupported_or_ambiguous": 6}`.

## Recurring workflow cost

Extraction is charged once per meeting, answer context once per independent question. These are the first 1/4/10 recorded source questions per meeting, not scaled averages.

| Questions / meeting | Raw input | Product input incl. extraction | Raw cost | Product cost | Cost savings |
|---:|---:|---:|---:|---:|---:|
| 1 | 31681 | 66182 | $0.010848 | $0.031295 | -188.5% |
| 4 | 126765 | 161266 | $0.040285 | $0.060732 | -50.8% |
| 10 | 317043 | 351544 | $0.106443 | $0.126890 | -19.2% |

A raw fallback after extraction can preserve answer behavior while costing more than starting with raw notes. The shipped unqualified default skips extraction and answers raw. Research previews are explicitly labeled.

## Discovery and repair

Seed 42; **6 generations**, at most six candidate slots and sixteen questions per round. Earlier development runs before the suite/probe audit are retained separately and all their paid calls remain in the ledger.

Selected grammar: `e74f92bbf304d74c`.

```json
{
  "kind_forms": [
    "action",
    "proposal",
    "decision",
    "agreement",
    "disagreement"
  ],
  "layout": "kind",
  "owner_marker": "",
  "deadline_marker": "@",
  "separator": ":",
  "grouping": false,
  "repairs": []
}
```

**Evolution did not beat the initial English seed on the selected held-out result.** Shorter development candidates existed, but traded away accuracy. A seed winning is an observed failure of the stronger search-improvement hypothesis, not evidence of discovery.

Controlled repair `a3f927ffe6f30dba` → `dd3fb1559aee615b`: `decision`; fixed 3 probe answers across 3 source IDs, +18 visible probe-input tokens, no new shared-probe regressions. Paired fixes include synthetic development contrasts; this does not qualify the child on the full suite.

Matched repair/no-repair ablation: **192 vs 192 logical probes** over the first two rounds.
```json
[
  {
    "generation": 0,
    "identical_probe": true,
    "repair_best_correct": 16,
    "no_repair_best_correct": 16,
    "probe_questions": 16
  },
  {
    "generation": 1,
    "identical_probe": true,
    "repair_best_correct": 16,
    "no_repair_best_correct": 16,
    "probe_questions": 16
  }
]
```

The ablation compares the same questions and logical budgets; cache reuse changes actual spending. Later adaptive rounds and full finalists are not a matched causal comparison.

### Four-question batching (development only)

| Method | Correct / 16 | Batch input tokens | Batch answer cost |
|---|---:|---:|---:|
| raw | 8/16 | 3570 | $0.001690 |
| english | 16/16 | 2225 | $0.001164 |
| evolved | 16/16 | 2231 | $0.001166 |

Batching is a supporting comparison, not the central language-discovery claim. See `ablations.json` for exact requests and single-question measurements.

## Spending and reproducibility

**2981 actual API calls; $1.200160 known cost** across annotation, failed drafts, all development searches, repairs, ablations, extraction, validation, and test. Unknown usage records: **30**. Global ceiling: 3,500 calls / $10. Rates: $0.40/M input, $0.10/M provider-cached input, $1.60/M output for the pinned reference model.

[Pinned-model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini). Usage is recorded from the API, not inferred from local character counts.

```sh
.venv/bin/python -m tokenese.evo_benchmark --audit
.venv/bin/python -m pytest -q
.venv/bin/streamlit run app.py
```

Artifacts: `frozen.json` contains model/tokenizer/prompt/schema/compiler/data hashes; `search.json` has generations and controlled repairs; `validation.json` and `test.json` retain every public question, response, and usage record; `verification.json` reproduces gates and arithmetic. `ledger.sqlite` is the shared persistent research budget/replay store. Private notes never enter it.

## Evaluation changes and limits

- Kept V1 and earlier results intact. Preserved original public source questions as a separate holistic track.
- Added 36 attributed public source excerpts, 360 bounded-language questions, 360 broader source questions, and 40 deterministic development contrast questions.
- Changed extraction grading from loose quote overlap to multiset exact semantic fields; added precision/recall and field diagnostics.
- Added strict Unicode-aware answer grading: names, signs, C++/C#, negation, and this/next date modifiers cannot collapse through ASCII punctuation deletion.
- Added source/category breakdowns, hallucination/abstention counts, paired critical regressions, and descriptive Wilson intervals. Correlated questions are not independent statistical evidence.
- Model-authored facts still contain annotation weaknesses (for example, meeting dates mistaken for deadlines and discussion mistaken for commitment in development). No independent reviewer is claimed.
- Cross-organization splits prevent series overlap but confound organization with distribution shift. Meeting updates can summarize multiple sessions; the unit is a source document/excerpt.
- Literal field-preservation questions are intentionally narrow. The broader track can reveal schema coverage loss and also has provisional labels.
- Missing independent review, failed 119/120 quality gates, new critical errors, or insufficient same-fact savings prevent qualification. Gates were not relaxed to obtain a win.
- Custom model weights, arbitrary literal rewriting, aliases, external compression services, and an optional model grammar proposer were not needed for this bounded local-mutation implementation.
