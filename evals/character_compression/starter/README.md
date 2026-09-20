# Chinese-character prompt compression: evaluation starter kit

## What this kit does

Evaluate a compressor by the downstream LLM behavior it preserves, not by how short its output looks. Run each original request and its compressed counterpart against the same target LLM, and grade both against an independent expected answer.

This kit contains 40 original synthetic development fixtures in 10 categories, a provider-independent Python scorer, an empty prediction template, and scorer unit tests. It does not contain your compressor, your character dictionary, book excerpts, or actual model measurements. No claims about your model's performance have been made. The scorer tests use synthetic outcomes only to verify the scoring code.

The fixtures are now public to you and should be treated as development/smoke tests, not as a sequestered release benchmark. They also are not a comprehensive evaluation of free-form summaries, tool use, or custom-dictionary behavior.

## Files

- `eval_cases.jsonl`: original messages, independent gold outputs, graders, and source-cluster identifiers.
- `predictions.template.jsonl`: output/measurement fields initialized to null. Replace null outputs with actual model responses; do not copy gold answers into prediction fields.
- `score_evals.py`: deterministic grading, paired failure analysis, token/cost/latency accounting, category breakdowns, and exploratory paired source-cluster bootstrap intervals.
- `test_score_evals.py`: 12 unit tests of scorer behavior, not model evaluations.

Python 3.10 or later is required. The scorer uses only the standard library and makes no network calls.

## Collect actual predictions

For every fixture, extract `messages` only for inference. Do not expose `expected_output`, the grader, or other gold metadata to either the compressor or the target LLM.

1. Run the original messages through the target LLM and save the actual output.
2. Pass the original messages through your real compressor. Add exactly the dictionary/decoder instructions your deployed target will receive. Save the resulting full message sequence.
3. Run those compressed messages through the same target LLM with the same generation configuration. Save the actual output.
4. Record the target's full-request input token counts. They must include message framing, the dictionary, examples, and decoder instructions, not just the compressed passage. Save tokenizer/model/dictionary/compressor versions.
5. Repeat with the same run IDs for every case. Randomize arm order and use fresh, independent conversations. If the provider supports seeds, keep the paired seed the same, but do not assume this eliminates all variation.

Use one predictions file per target model, compressor/dictionary version, decoding configuration, and compression setting. Do not mix distinct budgets in the same aggregate report. The scorer checks some version fields, but it cannot verify that your logged metadata actually matches the requests.

For a failed request, record an empty string for that arm's output and a nonempty `baseline_error` or `compressed_error`. Such errors count as failures. Do not omit failed cases. Leave unavailable token, cost, or timing measurements null; missing measurements are never silently treated as zero. A completely unmeasured output remains null and the scorer will refuse to score it.

Each prediction record has this shape (all null fields below are placeholders, not observations):

```json
{
  "case_id": "conditional_only_if",
  "run_id": 0,
  "baseline_output": null,
  "compressed_output": null,
  "baseline_input_tokens": null,
  "compressed_input_tokens": null,
  "baseline_output_tokens": null,
  "compressed_output_tokens": null,
  "baseline_latency_ms": null,
  "compressed_pipeline_latency_ms": null,
  "baseline_cost_usd": null,
  "compressed_pipeline_cost_usd": null,
  "baseline_error": null,
  "compressed_error": null,
  "compressed_messages": null,
  "compression_metadata": null,
  "model_id": null,
  "tokenizer_id": null,
  "dictionary_version": null,
  "compression_version": null,
  "token_count_source": null
}
```

`compressed_messages` must be the actual full request, including the codebook if supplied in context. Preserve roles and quoted-data boundaries. Put the requested compression budget, substitutions, escaping decisions, fallback status, and effective generation settings in `compression_metadata` or your experiment manifest.

For token counts, use provider-reported input usage or an exact model-specific tokenizer and chat template. A character count or token count from another model is not a substitute. Record the count source explicitly. Cached input is not automatically absent from the logical context: report logical input size and actual billed cost separately. This scorer consumes the input-token fields you provide and does not independently validate them.

Measure compressed pipeline latency from compression through the final answer. Pipeline cost must include compressor calls, target input/output, retries, and fallbacks. For a no-retry/no-fallback experiment the accounting is simplest. For production trials, also save attempt-level logs and distinguish first-attempt logical prompt size from aggregate input usage across attempts. A report based on aggregate usage should be labeled accordingly. Never report only the final successful attempt's cost.

## Score

After collecting predictions into `predictions.jsonl`:

```bash
python score_evals.py \
  --cases eval_cases.jsonl \
  --predictions predictions.jsonl \
  --out report.json \
  --details scored_cases.jsonl
```

The default run requires all cases and the same run IDs for each case. `--allow-partial` explicitly permits a diagnostic subset, with coverage reported. It does not allow unequal repetition counts. Null outputs are rejected even with this option.

Run scorer self-tests:

```bash
python -m unittest -v
```

## Interpret the metrics

- **Compressed accuracy:** fraction of compressed responses that pass their independently specified graders.
- **Accuracy delta (percentage points):** 100 times compressed accuracy minus original-prompt accuracy.
- **Correct-to-wrong regression rate:** original-correct/compressed-wrong pairs divided by all original-correct pairs. Improvements elsewhere do not erase these regressions.
- **Net input savings:** 1 minus total compressed input tokens divided by total original input tokens. The input counts must include the full dictionary/decoding overhead.
- **Expansion rate:** fraction of requests whose compressed version uses more input tokens than the original.
- **Correct answers per 1,000 input tokens:** an efficiency diagnostic, not a standalone objective. Enforce a quality floor first.
- **Literal output agreement:** exact agreement after stripping outer whitespace. It is not semantic equivalence and is not evidence of correctness. The same wrong answer can appear in both arms.
- **Cost and latency:** optional end-to-end metrics. They remain null unless their measurements are complete within the reported aggregate.

The four paired outcome counts distinguish both-correct, correct-to-wrong, wrong-to-correct, and both-wrong. A single observed transition is a regression signal, not proof of deterministic compression causation; compare repeated trials and original-versus-original variability.

For `exact` graders, only the stated outer-whitespace handling is allowed. No lowercasing, punctuation removal, Unicode normalization, or translation is applied. Strict format and literal-Unicode cases do not strip outer whitespace. For `json`, the response must be bare valid JSON, with no duplicate keys, no Markdown fences, and no extra prose. Object key order is ignored; array order and value types are enforced. In this kit, `1`, `1.0`, and `true` are intentionally distinct.

The bootstrap resamples source clusters and keeps related variants and repeated trials together. It returns exploratory 95% percentile intervals for accuracy delta and net token savings. These are not release certification: a small unrepresentative suite, especially with all-equal outcomes, can produce misleadingly narrow intervals. Zero observed errors is not evidence of zero future risk. The scorer does not implement a full power analysis, judge-based scoring, or every production release gate.

## Required extensions for your library

For every actual dictionary entry, build both isolated and downstream-in-context tests. Vary grammatical position, singular/plural forms, negation, nearby punctuation, overlapping phrases, infrequent meanings, and combinations with other entries. Keep exact phrases and their minimally different variants distinct. A code for “at least three” must not silently cover “more than three.”

Test literal occurrences of every assigned character in native Chinese text, names, quotations, code, and identifiers. They must remain literal unless explicitly marked as aliases. Evaluate your escaping/namespace convention, including its token overhead. The two Unicode fixtures in this kit are examples, not comprehensive coverage of your dictionary.

Test the supported decoder configurations separately: full codebook, relevant entries only, a mapping learned by a fine-tuned target, and cold-start requests. Define expected behavior for missing aliases, conflicting dictionaries, stale versions, and truncated histories. Do not assume training the compressor also trains a separate target LLM to understand custom aliases.

If you change the downstream LLM through fine-tuning, evaluate original and compressed prompts on that same fine-tuned model. Also report the original deployed model separately; otherwise model adaptation and compression effects are confounded.

Optional round-trip reconstruction is a debugging aid. Exact reconstruction is appropriate for a genuinely lossless dictionary code. For task-oriented lossy compression, evaluate preservation of relevant propositions and downstream answers instead of requiring identical recovered wording.

## Book and production benchmark design

Keep entire books, related editions, translations, series, and authors together when practical. Deduplicate related passages before splitting. Build the phrase library only from training material. Freeze dictionary, model, prompts, and thresholds before the final held-out evaluation.

For each held-out passage, store the source identifier, evidence spans, task, independent expected answer or rubric, required constraints, and failure severity. Include factual retrieval, cross-paragraph inference, chronology, relationship tracking, calculations, and explicitly unanswerable questions. Create counterfactual twins by changing names, values, or facts, and update the gold accordingly. This tests use of the supplied source rather than familiarity with a book.

Include deployment-style instructions and documents outside the book domain. A library learned from narrative prose must be tested on the actual prompts it will compress. Sample real workload length and task distributions for aggregate savings, and report high-risk slices separately.

The six long-context fixtures here vary a fact's position within 80 or 800 filler paragraphs. Those are paragraph counts, not claimed token counts. Real long-context evaluation should span the target's measured token lengths, multiple facts, realistic distractors, and many independent source documents.

Sweep several requested compression levels and compare achieved full-request savings at comparable quality. Include original English, concise English, relevant Chinese-language or ASCII-shorthand controls, and an established prompt-compression baseline at comparable token budgets. An English prompt that does not fit the context window belongs in a separate capacity-expansion experiment, not the ordinary same-content comparison.

## Research grounding

These are supporting references for the evaluation design, not measured results for your model:

- OpenAI, *How to count tokens with Tiktoken*: https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken . The page is archived; use current target-specific counting rather than hard-coding its example model table.
- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023): https://arxiv.org/abs/2305.15425 . Supports measuring tokenization by language/model, not assuming character-count savings.
- Zhou et al., *Instruction-Following Evaluation for Large Language Models*: https://arxiv.org/abs/2311.07911 . Supports objectively verifiable instruction checks.
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*: https://arxiv.org/abs/2307.03172 . Motivates position-sensitive retrieval tests.
- Pan et al., *LLMLingua-2* (Findings of ACL 2024): https://arxiv.org/abs/2403.12968 . A prompt-compression comparison method with downstream, in-domain, and out-of-domain evaluation.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023): https://arxiv.org/abs/2306.05685 . Documents judge biases; motivates blinded, calibrated judging for open-ended extensions.
