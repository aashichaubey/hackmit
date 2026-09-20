# Complete-source structural grammar pilot

This is a separate development-only successor hypothesis. The frozen fact-language experiment is unchanged. The preceding extractor studies showed that selecting sentences can lower cost while losing useful information. This pilot retains every source character except explicit structural verb phrases, which have a deterministic inverse. It uses no model extraction, custom tokenizer, dictionary legend, retrieval, or question-dependent selection.

The fixed API-pilot rules map ` decided to ` to ` decided `, ` agreed to ` to ` agreed `, ` proposed to ` to ` proposed `, and ` disagreed with ` to ` disagreed `. Local classification identifies those structural phrases only. Unrecognized text, background, statuses, names, dates, negation, ordering and repetitions remain in place. Quoted text and code spans are protected. Marker collisions, unclosed literals, integrity failures and non-saving output fall back to the complete original source. An initial local-only colon variant saved no tokens across the twelve excerpts and was rejected before model evaluation; isolated phrase token counts had overestimated savings at real boundaries.

The inverse reads encoded text and these rules alone. Exact reversibility certifies source preservation, not LLM comprehension. Literal-quotation questions and different syntaxes remain potential comprehension risks. The reference answerer never receives the rules or a legend. This is an extension beyond the original five-kind fact compiler, not evidence that that experiment passed.

Before API evaluation: load only the twelve existing development excerpts; use their unchanged broader source questions, all 120, with the original fixed answer prompt/schema. Compare raw versus this one fixed grammar. Keep raw baseline replays. Record all routes, actual full input/output usage, question-paired gains/regressions, source counts, and 1/4/10-question workflows with zero extraction usage. Replays and actual requests remain distinct. Count all new grammar comprehension calls against the shared discovery ceiling. Do not tune from validation/test or reinterpret development as unseen qualification. No deployment is authorized by this pilot's results alone.

A useful pilot must save aggregate actual input tokens and introduce no new errors on questions the raw baseline answers correctly. No numerical savings target is substituted for quality. A small gain does not establish the full user objective; subsequent fresh, source-disjoint qualification and checked annotations remain necessary.

## Failure attribution follow-up, declared before repeat calls

The pilot saved 60/118,463 actual answer input tokens but scored 77/120 versus raw's 79/120. The two failures are `dev-02-q04` (short owner name versus a longer accepted name/title) and `dev-02-q08` (discussion incorrectly answered as a recorded decision). The first suggests a provisional-label defect; the second is a substantive answer failure. Neither is silently relabeled.

Repeat these two questions five times for each representation with identical provider prompts, parameters and schemas. Alternate raw/encoded dispatch order across replicates. Use separately indexed, budgeted cache entries so repetition is real while interruption/replay does not repeat spend. The index is never sent to the model. All twenty calls count against discovery and global budgets. Report every output and correctness count; do not select a best answer, change the original score, or generalize this selected-failure diagnostic to overall efficacy.

## Observations

The full-source pilot's 0.0506% answer-input saving is too small to compensate for the observed quality failures. Only two of twelve source excerpts used the grammar; ten retained raw text. Its twenty encoded questions scored 11/20; raw fallback contributed 66/100. These remain development measurements, not a general-purpose qualification.

| Selected failure, five fresh repeats | Raw exact passes | Encoded exact passes |
| --- | ---: | ---: |
| `dev-02-q04`: owner name | 4/5 | 0/5 |
| `dev-02-q08`: whether discussion is a decision | 3/5 | 1/5 |

Every owner response identifies Łukasz. The source introduces him as the Developer-in-Residence and assigns the task to the DiR. The accepted-answer list omits the plain name, so this question needs annotation correction in a separately versioned suite. Its current exact score is not a reliable semantic-error count. Original labels and all scores remain preserved.

For the decision question, both inputs sometimes cause an unsupported affirmative response despite temperature zero. The encoded input failed more often in this small diagnostic, but five selected repeats do not establish statistical significance or population error rates. The representation remains rejected; the report does not substitute the most favorable repeat for the first result.

These findings improve the evaluation protocol: retain strict critical-field checks, validate accepted name variants against source evidence before freezing a suite, distinguish annotation defects from model errors, and use predeclared repeated paired calls for stability diagnostics. Do not tune a grader to make a candidate pass. Exact source reconstruction remains an independent integrity metric.

## Additional local screen

`evo_source_groups.py` factors a repeated literal actor over at least three consecutive inline bullet clauses, using `{actor:clause;clause}` and a deterministic inverse. All other source text remains present; delimiter collisions, fenced code and non-saving groups fall back. This tests consecutive inheritance without an extraction model or changing names. On these twelve excerpts it saved only five context tokens in one excerpt (10,024 → 10,019). That coverage and gain do not justify another paid comprehension run. The local-only artifact is `reports/evolution/source-groups-local.json`; it makes no model-quality claim and is not an active product route.

After the phrase pilot and twenty repeat calls, the shared research ledger contains 3,179 actual calls, $1.2694544 known cost, and $1.4348344 reserved/spent. The same thirty historical validation failures still have unknown usage. Neither budgets nor original frozen results were reset.
