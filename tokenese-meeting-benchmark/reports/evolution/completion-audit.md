# Goal completion audit

**The user objective is not achieved.** The software and research apparatus are substantially implemented, but no measured candidate satisfies the requested combination of general-purpose answer quality and improved token efficiency. Neither more tests nor a passing artifact-reproduction check establishes that empirical result.

## Requirements and current evidence

| Requirement from the goal/plan | Authoritative evidence | Assessment |
| --- | --- | --- |
| Task 1: checked corpus, provenance, splits, contrasts | `data/evolution/{dev,validation,test}.json`, manifest, `evo_data.py`, `label-review.json` | Provisional corpus exists; independently checked annotations are missing. Twelve excerpts per split do not establish twelve single-session meetings. Development audit flags 34 review candidates. |
| Task 2: immutable grammar, deterministic reversible compilation, literal preservation | `evo_grammar.py`, `tests/test_evo_grammar.py`, saved integrity checks | Implemented and tested; structural preservation is not LLM comprehension. |
| Task 3: fixed prompts/schema, grading, actual usage, replay and persistent budgets | `evo_eval.py`, `evo_usage.py`, ledger, their tests | Implemented. Thirty historical validation failures have unknown usage, so full research accounting remains incomplete. |
| Task 4: bounded seeded evolution, lineage, screening and protected splits | `evo_search.py`, `evo_mutate.py`, `search.json`, selection lock, search tests | Six recorded generations; implemented. The initial English seed won the held-out comparison. |
| Task 5: paired failure-driven structural repair | `evo_repair.py`, recorded parent/child results, repair tests | One real controlled repair fixed three probes with no shared regressions at +18 tokens. This did not qualify a full language. |
| Task 6: select, freeze and independently qualify | `frozen.json`, `validation.json`, `test.json`, locks, `verification.json` | Freeze and one-time evaluation exist. Language/product gates fail; independent annotation review is absent. Reproduction passes, qualification does not. |
| Task 7: isolate encoding, extraction, repair, seed and batching effects | `ablations.json`, `search-no-repair.json`, full results report | Required comparisons recorded. No evidence evolution beats the initial seed; matched repair ablation does not show overall improvement. |
| Task 8: private compile-once workspace, fallback and recurring cost | `evo_memory.py`, workspace, memory tests, test workflows | Implemented. Unqualified default answers raw without extraction. Diagnostic extraction-plus-fallback costs more than raw at 1/4/10 questions. |
| Task 9: inspectable UI, actual repairs, examples, lineage and cost | `app.py`, `evo_view.py`, app tests, inspected running app | Implemented artifact replay and private workspace; research failures remain visible and unqualified. |
| Task 10: tests, reproducible audit and honest release report | pytest results, `--audit`, `README.md`, root README, implementation review | Reports and software checks exist; they explicitly report failure to meet empirical qualification. |
| User: evaluate and improve the suite holistically | separate language/source-question tracks; paired errors; abstentions; extraction precision/recall; workflows; repeatability and label audits | Materially improved. A reliable independently checked qualification target is still missing; exact scoring alone can misclassify valid name variants. |
| User: iterative improvement without replacing the creative mechanism | six generations, structural repair, classifier successors, complete-source grammar and grouping screens | Iterations completed and unsuccessful candidates rejected. No globally optimal or successful RL-trained system is claimed. |
| User: better than raw text while retaining general purpose | held-out product report and later development pilots | **Not achieved.** Gold-fact token reduction does not establish broad source usefulness; cheaper classifier candidates lost answers. Full-source grammar gains were tiny and failed quality checks. |

## Commands and artifact scope

The plan's local screen, paid search, validation selection, frozen test, pytest and Streamlit entry points are implemented. Selection/test are one-way operations and are not rerun to seek better outcomes. `.venv/bin/python -m tokenese.evo_benchmark --audit` reproduces saved gates and arithmetic without model calls. The original frozen compiler/prompt/schema/product hash remains `20a1ee626ef53e3b4a0f3e424408780301c7dc289296d9a4632094dd760b4f1e`.

The optional model proposer and external-compressor follow-ups were not implemented as part of this experiment; the plan explicitly permits cutting them. Learned aliases, custom tokenizers, broad schema expansion and model-weight training were deferred rather than silently substituted for the language-discovery objective.

## Remaining prerequisites

1. Establish trustworthy source-grounded development labels. `label-review.md` is a concise first review, not completion of corpus review. The original labels and frozen scores remain immutable; any accepted corrections require a separately versioned dataset.
2. Find a representation/extraction policy that preserves broad usefulness and passes quality/cost checks. Independent review alone cannot make the current losing candidates successful.
3. Qualify any successor on fresh, source-disjoint, independently checked data. The opened validation/test sets cannot be reused for adaptive qualification.
4. Retain complete accounting and sufficient evaluation allowance. The shared ledger currently has 3,179 actual calls, 1,431 discovery calls; 321 global and 69 discovery calls remain. The existing limits have not been reset or raised. No claim is made that this remainder can cover a new full qualification experiment.

The outstanding external input requested is an independent annotation reviewer or a trusted source-grounded meeting-QA dataset. Further optimization against the currently suspect labels risks rewarding annotation artifacts instead of useful compression. This audit does not mark the goal complete.
