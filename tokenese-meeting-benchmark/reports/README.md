# Tokenese benchmark results

Dataset v1 has 30 short synthetic meetings, four reference questions per meeting, and an 18/6/6 development/validation/test split. The answerer and extractor used `gpt-4.1-mini-2025-04-14`; the separate answer judge used `gpt-4.1-2025-04-14`. Token counts use the answerer's tokenizer. These results are a smoke benchmark for this corpus, not a claim about real transcripts.

## Held-out test: 24 questions

| Method | Gold-fact exact | Parsed-fact exact | Parsed median full input text tokens | Parsed workflow API tokens |
| --- | ---: | ---: | ---: | ---: |
| Raw notes | 24/24 | 24/24 | 125 | 4,417 |
| Compact English facts | 21/24 | 21/24 | 128 | 6,562 |
| Tokenese symbols | 23/24 | 24/24 | 164 | 7,416 |
| Tokenese mixed | 23/24 | 23/24 | 167 | 7,496 |
| LLMLingua-2 | 24/24 | 24/24 | 117 | 4,225 |
| Bear-2 | 22/24 | 22/24 | 117 | 4,211 |

Workflow API tokens include extraction once per meeting for compact English and Tokenese. Raw, LLMLingua-2, and Bear-2 use the notes directly and do not pay extraction cost. Bear-2 also uses its own compression API; its provider tokens are separate from OpenAI tokens. LLMLingua-2 uses local compute.
These workflow totals sum measured API usage for each logical question. The benchmark cache avoided rebilling identical research calls. No fact-based method reaches a token break-even point against raw notes on this corpus.

## All 120 questions

| Method | Gold-fact exact | Parsed-fact exact | Parsed workflow API tokens |
| --- | ---: | ---: | ---: |
| Raw notes | 119/120 | 119/120 | 21,639 |
| Compact English facts | 105/120 | 101/120 | 32,051 |
| Tokenese symbols | 119/120 | 118/120 | 36,331 |
| Tokenese mixed | 119/120 | 116/120 | 36,719 |
| LLMLingua-2 | 115/120 | 115/120 | 20,735 |
| Bear-2 | 112/120 | 112/120 | 20,665 |

Tokenese did **not** meet the success gate. Neither profile reduced full answering input tokens versus compact English on any of the 96 development and validation questions. Both passed literal-field preservation checks, but their grammar legends outweighed savings for these short notes. The interactive app therefore selects compact English. On this corpus, compact English also costs more than raw notes after extraction.

After human review of noncritical longer answers, full-pipeline accuracy is raw 120/120, compact English 101/120, Tokenese symbols 118/120, Tokenese mixed 116/120, LLMLingua-2 116/120, and Bear-2 113/120. Held-out test scores are unchanged. Exact date failures remain failures.

## Failure audit

- Parsed extraction missed one of 87 gold facts: on validation meeting 19, “Jules will approve the copy by tomorrow” was classified as a decision rather than an action. It also added one unsupported development proposal and the misclassified validation decision. No test extraction facts were missed. The matching rule accepts a longer text description or a shared supporting quote when kind, required person, and deadline agree.
- Bear-2 lost two test deadline modifiers: `next Friday` became `Friday`, and `end of month` became `end month`. Its 22/24 test score reflects these exact-date failures. On all 120 questions it also missed two development owners and two other deadlines.
- LLMLingua-2 scored 24/24 on test, but missed two development owners across the full set.
- Compact English answered `yes` for two test proposals that were never decisions, and `no` for one explicit test decision. The latter error also occurred with the oracle Tokenese profiles; disagreement in the same notes appears to have confused the answerer. The frozen profile was not changed using test feedback.
- The exact scorer rejected a few longer, factually correct descriptions of “run the demo.” A separate judge marked these correct. The same judge also marked four truncated deadline answers correct; human review retained the strict date failures. Judge verdicts did not override the acceptance gate.
- Embedding cosine similarity was diagnostic only. Eleven of 20 corrupted pilot pairs scored at least 0.9, confirming that high similarity cannot certify critical-field preservation.

## Usage and verification

The full parsed-fact run used 7,340 extraction input tokens and 2,775 extraction output tokens across 30 meetings. Bear-2's three-rate trials consumed 2,100 provider input tokens and 1,298 provider output tokens, tracked separately from OpenAI usage. LLMLingua-2 took 8.03 seconds of local compression time on the gold pass; cached contexts made the parsed pass nearly free. Research overhead was nine judge calls (2,008 input, 498 output tokens) and five embedding calls (2,359 tokens). The optional Jev pilot was not run because no TypeSafe key was available.

The report contains every bundled input, prompt, answer, local count, API usage record, and error. The prompt and usage consistency tests pass. See `reports/latest.json` for per-question evidence, `reports/search.jsonl` for candidate screening, and `data/judge_review.json` for the manual review decisions.

## Supplemental speaker-labeled stress set

Six new meetings with 24 questions were evaluated separately; they did not influence the frozen profile. Exact full-pipeline scores were raw 22/24, compact English 23/24, Tokenese symbols 21/24, Tokenese mixed 21/24, LLMLingua-2 21/24, and Bear-2 20/24. Bear-2 omitted `this` from `this Friday` and returned `Jo Raj` for an owner expected to be `Raj`. Extraction missed two gold facts and added two unsupported facts. The complete stress run is in `reports/stress.json`.
