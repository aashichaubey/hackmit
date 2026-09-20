# How to reduce actual input tokens

## What the current benchmark sends

On the 30-meeting, 120-question synthetic set, each raw-note answer call uses an average of 169.8 OpenAI input tokens. The meeting notes average 24.7 tokens. The shared answer instruction is 78 tokens. The question and visible wrappers account for about 19 tokens, and the API reports 48 more tokens than the tokenizer count of visible text. In one identical-prompt comparison, structured output accounted for 41 of those extra tokens and request framing accounted for seven. These are measured API usage numbers from `latest.json`, not character counts.

The raw meeting content is only 14.5% of the current answer input. Even perfect deletion of the meeting content could save at most that fraction on these short examples. The present Tokenese symbols use more context tokens than raw notes, and omitting their legend lowered answer quality. The controlled ablation is in `no_legend.md`.

## Measured experiments

| Method and corpus | Correct | OpenAI input tokens | Change from corresponding raw calls |
| --- | ---: | ---: | ---: |
| Four separate raw calls, main | 119/120 | 20,370 | baseline |
| Four questions in one raw call per meeting, main | 119/120 | 6,990 | 65.7% fewer |
| Four separate raw calls, stress | 22/24 | 4,434 | baseline |
| Four questions in one raw call per meeting, stress | 22/24 | 1,482 | 66.6% fewer |

Batching used the same `gpt-4.1-mini-2025-04-14` model and a structured list of answer objects. It preserved the exact question-level success and failure pattern on both corpora, with no missing answer slots. Full prompts, answers, and API usage are in `batched_questions.json` and `batched_stress.json`. The main test split had been evaluated earlier, so this is an exploratory comparison rather than a new untouched holdout. Batching applies when multiple questions are known at request time. It cannot amortize future interactive questions that have not yet been asked.

| Other experiment | Correct | OpenAI input tokens | Finding |
| --- | ---: | ---: | --- |
| Current raw prompt, development + validation | 95/96 | 16,208 | Reference |
| Shorter instruction `s5`, development + validation | 93/96 | 13,150 | Two new unsupported `no` answers; fails critical-field gate |
| One-field structured output, development + validation | 92/96 | 14,777 | Three new unsupported `no` answers |
| Better plain-text prompt without output schema, development + validation | 67/96 | 8,240 | Unreliable answers despite much lower input |

The shortening experiments are in `short_prompt_rules.json`, `short_prompt_finalists.json`, `one_field_schema.json`, and `plain_output.json`. They show why removing instructions or schema tokens indiscriminately is not a validated solution.

## Product path

1. For bundled questions, answer the list in one request. This is the only approach tested here that cut actual input tokens substantially without changing the measured answers.
2. For one question about a short note, use the raw note until a smaller prompt passes the exact owner, date, decision, and unknown-answer checks. The content is already too small to yield substantial savings by compression.
3. For longer transcripts, segment locally and select question-relevant evidence before sending it to the answerer. Preserve literal names, dates, negation, and proposal or decision status. A local selection step can avoid another paid model call; if a model extracts facts, count that extraction input and output once per meeting. The break-even number of questions is `ceil(extraction tokens / input tokens saved per answer)` when per-answer savings are positive.
4. Compare query-aware selection, LLMLingua-2, and Bear-2 on real long meeting transcripts with checked questions. Measure full API input, downstream answer quality, and compressor cost separately. The existing notes average only 25 tokens, so they cannot establish long-transcript savings.

Research supports this direction. [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) and the [rate-distortion study](https://arxiv.org/abs/2407.15504) find that the question matters when selecting context. [RECOMP](https://arxiv.org/abs/2310.04408) studies extractive selection and selective augmentation. [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) learns to retain important original tokens and evaluates on MeetingBank. These papers study other tasks and models; their reported compression rates are not measured Tokenese results.

For precise accounting, the [OpenAI input-token counting API](https://developers.openai.com/api/docs/guides/token-counting) includes request components such as tool definitions, while actual response `usage.input_tokens` is the authoritative measurement after a call. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) can reduce billed cost or latency for repeated prefixes, but cached tokens remain part of input usage. [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state) also bills prior input in a `previous_response_id` chain. Neither substitutes for sending less context.
