# Implementation review and remaining experimental limits

Review performed in the main coding session, with executable regression checks. This is not claimed to be an independent review.

## Contracts verified

- Every seeded grammar and deterministic mutations preserve literal semantic fields, order, and duplicates through a parser that cannot read the compilation trace.
- Delimiter collisions, escaped markers, absent fields, Unicode text, repeated facts, corrupted dates, orphan inheritance, unsafe lexemes, and instance-specific repair rules have rejection/round-trip tests.
- Question grading checks both answer value and expected `found`; Unicode letters, numeric signs, significant symbols, and date modifiers remain distinct.
- Missing extraction cannot accidentally load benchmark gold; private product paths reject disk-backed/public runners.
- Four follow-ups use one extraction in the session-memory test. Unknown qualification, incompatible hashes, empty extraction, unsupported content and ambiguity retain raw notes.
- Atomic ledger reservations enforce limits across concurrent requests and restarted processes. Replays preserve provider usage attribution and do not increase actual spending. Unknown usage retains a reservation.
- A shared pacing schedule was added after validation rate limits; it changes dispatch timing, not model requests or selected language behavior.
- Development mutation is blocked after the selection lock. Read-only local seed screening remains possible. Test artifacts prevent adaptive reruns. Frozen grammar, code, schema, prompt, tokenizer, model and corpus identities are checked before test/product use.
- Saved-result auditing reproduces gates and 1/4/10-question extraction-inclusive workflow arithmetic without model calls.
- Streamlit tests cover an empty report directory, a tiny trace, and the full application without creating an API client. The live language-lab and workspace views were visually inspected.

## Important boundaries

The data-only local mutation and repair path is implemented. The optional model proposer, external compressors, custom tokenizers, learned aliases and model-weight training are intentionally absent.

The public corpus has 12 source excerpts per split, not a claim of 12 independently reviewed single-session meetings. Provisional annotations remain a material weakness. `annotation-review.csv` includes failed held-out rows, their source URLs and evidence for future review; populating it does not retroactively qualify this frozen run.

The experimental quality gate can fail even when the software works and tokens are reduced against raw excerpts. The product does not silently activate a failing or unreviewed grammar. Actual runtime and benchmark fallback can differ because research deliberately evaluates an unqualified grammar diagnostically; the normal unqualified product route is raw without extraction.

The rate-limited validation requests remain failures and retain unknown usage. Reports show known spending separately from conservative outstanding reservations. No missing record is treated as zero-cost success.

## Development-only classifier follow-up review

Sequential self-review covered the new `evo_classifier_next.py`, budget-purpose accounting and saved-result UI. This was not an independent reviewer. Both successor versions read only development excerpts; no frozen implementation module or original labels changed. V1 remains as a separate artifact. Invalid/duplicate/reordered source IDs reject extraction; complete source spans preserve literal negation and date modifiers without generated task text. Exact copying does not certify selection completeness, kind correctness or general-purpose usefulness.

The review caught two reporting/accounting risks: raw fallback could masquerade as an improved compressed answer, and the first pilot's new purpose name could escape discovery accounting. Paired diagnostics now separate encoded and fallback results, reject missing/duplicate question pairs, and expose every new error versus raw. Existing `classifier_development` ledger rows remain preserved and conservatively count against the unchanged discovery ceiling.

The second candidate is rejected on quality despite cheaper extraction. The live product stays unchanged. Full suite: 54 tests passed; original frozen verification still passes; the live Language lab visibly shows 11 paired regressions and encoded 15/30 versus fallback 7/10. Rendering and opening the report do not call the model. Follow-up cumulative spend is $1.2175152 known, $1.3828952 reserved/spent, 3,059 actual calls, including the same 30 unknown-usage validation failures.

## Complete-source and repeatability follow-up review

Sequential review found that selected multi-sentence evidence can contain line breaks or `|`, which the inherited Fact schema rejects. The successor now records a rejected extraction instead of crashing. Tests cover both cases; the original frozen classifier is unchanged.

New source-grammar modules preserve complete text and prove byte-for-byte reconstruction from emitted text plus rules, independently of a trace. This remains separate from LLM comprehension. The phrase grammar failed its development quality check; the actor-grouping screen saved only five tokens and was not model-evaluated. Neither is deployed. Tests exercise order, duplicate clauses, negation, Unicode/raw fallback, literal collisions, code regions and complete-source retention.

Repeat calls use an explicit nonnegative replicate index in the cache namespace only, retaining identical provider inputs. A fake-client regression proves default request hashes/replay remain unchanged, distinct replicates consume the same shared budget, and repeated access to a replicate replays without spending. All twenty actual diagnostic calls remain recorded. These selected-failure repeats do not change the original pilot, labels, or qualification status. The source name-variant defect is documented separately from the substantive discussion/decision failure.

The original frozen gate and workflow audit still passes. These are self-review findings, not an independent review of code or annotations.

## Annotation-audit review

The development-only offline audit produces review candidates, never accepted labels or revised scores. Its packet includes full source excerpts and omits model responses. Review status and reviewer identity remain pending; no independent review is fabricated. A first-pass heuristic overflagged an either/or question; a regression now excludes such choice questions and distinguishes extra accepted synonyms from a missing yes/no answer. The final queue contains 34 candidates, with three selected for concise review.

Tests verify immutable input annotations, held-out rejection, broader-track source offsets, expected-found handling, name-variant candidates and the choice-question distinction. The current full suite passes 68 tests; the original freeze/gate/workflow reproduction also passes. These results do not satisfy the empirical product goal. See `completion-audit.md` for requirement-by-requirement evidence and remaining prerequisites.
