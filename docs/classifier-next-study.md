# Indexed-evidence classifier development study

The completed evolution experiment stays frozen. This successor candidate addresses the nonliteral-source-quote problem first observed in development annotation and extraction. It does not rewrite the selected grammar, previously reported failures, or held-out labels.

The candidate selects numbered source spans instead of generating source quotes. Evidence text is copied locally from the original character offsets. Fact values must occur literally in selected spans, and evidence order must follow the source. Missing, out-of-range, reordered or nonliteral references reject the extraction route. This is a structural grounding check, not proof of semantic correctness.

The first probe uses the first four existing **development** public excerpts, their unchanged broader source questions, the existing answer prompt/schema, and the original classifier as comparator. The CLI may expand to all twelve development excerpts. It never loads validation or test. Both model runs and replays share the original persistent budget; this does not create a new spending allowance.

Report evidence-valid case coverage, rejected literals, extraction input/output tokens and cost, and diagnostic QA on the same questions. QA deliberately exposes successfully extracted facts for diagnosis even when unsupported source content exists; it is not the deployed product route. Unsupported/ambiguous flags remain intact. None of these development measurements can qualify the candidate.

Before deployment, any successor needs a separately predeclared, untouched source-disjoint evaluation with independently checked semantic annotations and the full language/product gates. The old test is not eligible for adaptive reuse.

## Measured iterations

The first version still generated literal fields. All four extractions failed a field-membership check despite selecting valid source evidence. Its 28/40 answers came entirely from raw fallback. That score must not be attributed to compression. The original artifact is retained as `reports/evolution/classifier-next-v1-dev.json`.

The second version emits only kind and source-span IDs. Local materialization copies complete enclosing sentences into task text; owner/deadline fields remain empty rather than inventing a second interpretation. Three of four extractions were structurally valid; one returned evidence out of source order and fell back to raw. No semantic correctness claim follows from exact copying. All four model outputs incorrectly failed to flag broader unsupported content; these flags cannot establish comprehensive source coverage.

| Four-excerpt development pilot | Original classifier | Span selection | Raw notes |
| --- | ---: | ---: | ---: |
| Broad QA | 23/40 | 22/40 | 28/40 |
| Extraction cost | $0.0056568 | $0.0021408 | $0 |
| Answer input tokens | 14,882 | 16,152 | 30,672 |
| Ten questions/excerpt, total cost | $0.0123248 | $0.0093456 | $0.0130848 |
| New wrong answers where raw was correct | 9 | 11 | — |

Span selection reduces extraction cost by 62.2% and diagnostic ten-question workflow cost by 28.6% versus raw. It loses six net correct answers and introduces eleven paired errors; **reject this as a product improvement**. On the actual encoded subset it answers 15/30, and raw fallback contributes 7/10. Its one-question workflow costs 118.9% more than raw; four-question cost falls 4.1% while accuracy falls from 11/16 to 9/16. Labels remain provisional; this small adaptive pilot is not a generalization estimate.

The implementation now reports paired regressions, encoded-only results, fallback-only results, and complete diagnostic workflows. No new model calls are needed to recompute those metrics. The live workspace retains the original qualification guard and raw default.

The shared ledger after both follow-ups contains 3,059 actual calls, including the original frozen experiment. Forty-four first-pilot calls were initially recorded under `classifier_development`; they remain transparently attributed there and now count conservatively toward the existing 1,500 discovery ceiling. Subsequent diagnostic QA uses `probe`. No limits were raised or reset. Original frozen reports retain their historical spend snapshots; the successor artifact records the later total.
