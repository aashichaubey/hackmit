# No-legend Tokenese ablation

The 120-question parsed-fact benchmark was rerun with the Tokenese grammar legend removed. Both profiles kept the same shared answer instruction, question, structured output schema, and extracted facts as the earlier benchmark. The answer model was `gpt-4.1-mini-2025-04-14`. This isolates the legend's effect. The test split was already examined in the earlier benchmark, so these are exploratory results rather than a fresh held-out evaluation.

| Method | Exact answers | Total answer input-text tokens | Cheaper than compact English |
| --- | ---: | ---: | ---: |
| Raw notes | 119/120 | 14,610 | — |
| Compact English | 101/120 | 14,922 | — |
| Symbols with legend | 118/120 | 19,194 | 0/120 |
| Symbols without legend | 92/120 | 15,114 | 4/120 |
| Mixed with legend | 116/120 | 19,582 | 0/120 |
| Mixed without legend | 107/120 | 15,142 | 0/120 |

The input-text counts are tokenizer counts of the full answer prompt. The raw, compact English, and with-legend values above are summed from `latest.json`; the no-legend values are in `no_legend.json`. The no-legend profiles still use more answering tokens than raw notes or compact English, before adding fact-extraction cost.

Measured OpenAI usage for no-legend symbols was 22,160 answer tokens plus 10,115 extraction tokens, or 32,275 workflow tokens. No-legend mixed used 22,181 answer tokens plus 10,115 extraction tokens, or 32,296. Raw notes used 21,639 workflow tokens. These totals count extraction once per meeting in this benchmark; reuse over many more questions could amortize that step.

I also tested a literal no-prefix prompt: context followed only by `Question: ...`, with no shared answer instruction or legend. Across 120 questions it used 4,050 tokenizer tokens for raw notes, 4,554 for symbols, and 4,582 for mixed. Thus dropping the shared instruction for every method does not make either encoding cheaper than raw notes. All three methods scored 0/120 under the existing *exact-format* scorer because the model returned explanatory sentences; this result is about output formatting and does not measure semantic correctness. The responses and usage are in `bare.json`.

On the 24-question test split, no-legend symbols answered 19/24 and no-legend mixed answered 21/24. With their legends, they answered 24/24 and 23/24 respectively. In no-legend symbols, 13 of 28 errors were decision questions, six were agreement, and six were disagreement. For example, `P|Sarah|launch plan` and `+|Maya|launch plan` sometimes led to an answer of both Sarah and Maya when only Maya agreed. The model also sometimes returned `not found` for explicit `D|...` decisions.

The papers point toward compressing ordinary language rather than expecting a model to decode an undefined private alphabet. [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) trains a smaller model to retain important original tokens and evaluates on MeetingBank and other tasks. [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) uses question-aware selection for long contexts. [Symbol tuning](https://arxiv.org/abs/2305.08298) shows that robust handling of arbitrary labels improves with targeted training and in-context mappings; its evidence does not establish reliable zero-shot decoding of this grammar.

For this short-note dataset, removing the legend alone does not produce a usable Tokenese profile. A better next experiment is a self-explanatory, terse English representation with no legend, evaluated on real, longer meeting transcripts and compared on end-to-end token usage and literal owners, dates, and decision status.
