# Reusing cached work after text compaction

**Status: promising approximate prototype; the general problem remains unsolved.**
This experiment actually replaces already-cached history with the 149M pruner's
compressed text, retokenizes that text, and repairs part of the old Qwen cache.
It addresses a different question from the earlier KV-eviction experiment.

With about 10% of compressed-history positions recomputed, the prototype reused
89.95% of their old per-layer K/V entries and reduced median cache-transition
time from 115 ms with ordinary exact prefix reuse to 70 ms. Both methods scored
4/12 on factual/abstention questions. This does not establish equivalence: only
10/14 generated answers were identical, the first-token distributions differed,
and the compression itself already discarded important evidence.

## What is being solved

When earlier text changes, a later token's cached state can change even if that
token's spelling survives. Shifting its position is only part of the issue:
deeper states encode attention to the old preceding text. Exact ordinary prefix
reuse can keep the unchanged beginning but rebuilds the affected suffix.

The proposed approach is to reuse surviving states approximately, adjust rotary
positions for the actual shortened prompt, and recompute a subset of states.
This draws on the selective-recomputation idea in
[CacheBlend](https://arxiv.org/abs/2405.16444), which studies reuse of cached RAG
chunks. Our fixed edit-distance selector is **not a reproduction of CacheBlend**,
and applying this idea to text compaction is the hypothesis tested here.

## Implementation

1. Build Qwen's original history cache once, before the final question.
2. Use the previous experiment's saved outputs from the unchanged 149,015,041
   parameter ModernBERT pruner, checked against the checkpoint hashes. Replace
   each historical session with its actual `PruneResult.notes` string.
3. Tokenize the complete new prompt normally. Track source-character provenance
   to align surviving tokens; newly inserted joins and retokenized boundary
   tokens are always recomputed. No new token is assigned an unrelated old state.
4. Rephase surviving keys from their old RoPE positions to their new positions.
   Values are reused unchanged unless selected for repair. This positional
   correction alone does not repair contextual differences in deeper layers.
5. At every layer, recompute selected tokens' attention and feed-forward states
   against the combined reused/recomputed cache. The fixed selector prioritizes
   positions immediately following edits. It never reads the final question,
   reference answer, evidence labels, or a freshly rebuilt reference cache.
6. Process the final question and generate with Qwen using positions in the new,
   shortened prompt. The old cache remains read-only during each comparison.

The 150M model chooses the compressed text. The new repair policy is a heuristic;
it is not a trained cache-repair head. Qwen2.5-1.5B remains frozen.

## Results

Same 14 previously inspected public LongMemEval oracle examples, seven arms,
98 completed generations. This is a development diagnostic, not a new holdout
or the full-history benchmark. Every arm receives identical compressed prompt
token IDs. All arms have the same smaller final cache; the comparison measures
the work required to obtain it after text has already been processed.

| Method | Compressed-history KV entries reused | Median transition time | Answers identical to fresh compressed prefill | Factual correct |
|---|---:|---:|---:|---:|
| Fresh compressed prefill | 0% | 122 ms | 14/14 | 4/12 |
| Exact common-prefix reuse | 13.72% | 115 ms | 14/14 | 4/12 |
| Repair only new/boundary tokens | 93.53% | 68 ms | 9/14 | 4/12 |
| Repair about 10% | 89.95% | 70 ms | 10/14 | 4/12 |
| Repair about 25% | 74.96% | 76 ms | 10/14 | 4/12 |
| Repair about 50% | 49.97% | 99 ms | 9/14 | 4/12 |
| Repair 100%: correctness control | 0% | 152 ms | 14/14 | 4/12 |

Reuse is a token-weighted fraction of positions in the **compressed history**,
not the original history, input-token billing, or a fraction of all FLOPs. Keys
at shifted positions still incur rotation/copy work. Mandatory new tokens can
exceed a requested repair budget. Across the 14 cases, history lengths totaled
53,671 original and 10,374 compressed tokens; final questions add 705 tokens.

The 10% arm was faster than exact prefix reuse in 11/14 cases. The ratio of
summed transition times was 1.78x; the displayed medians have a different ratio.
The smallest prompts can be slower with repair because it adds overhead.

Median first-token KL divergence from fresh compressed prefill was 0.2223 for
mandatory-only repair, 0.0947 at 10%, 0.0462 at 25%, and 0.0152 at 50%. This
measures only the first generated-token distribution, not complete-answer
fidelity. Exact prefix reuse and full repair matched the reference's generated
token sequence in all 14 cases. More repair did not monotonically increase
verbatim answer agreement; some differences were harmless paraphrases.

The four correct factual cases were father-gift abstention, Sacramento-booking
abstention, two meetings with Alex, and Arati Prabhakar. All methods missed the
other eight. The Japan/Chicago answer changed from two days with fresh prefill
to three with partial repair; both are wrong. The two preference cases are
qualitative: reunion responses lacked personal details; evening suggestions
omitted the before-9:30 preference. The evening output hit the 128-token cap in
fresh prefill, exact prefix reuse, 25%, 50%, and full repair.

Quality was reviewed manually by the Codex agent against public reference
answers. It is not independent human grading or the official LongMemEval judge.
The review is bound to the results-file hash.

This compression is not identical to the prior eviction experiment: actual
`PruneResult.notes` assembly changes whitespace/headings and does not apply the
old eviction policy's 128-token protection. Its 4/12 cannot be used as a direct
head-to-head comparison against that experiment's 5/12.

## Timing and correctness checks

Pinned Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16 on Mac MPS,
PyTorch 2.14.0, Transformers 4.57.6. No hosted inference or judge calls.
One warmup for each case/arm shape precedes three timed transitions; the median
of those three is used. Arm order is randomized with seed 17. The full/prefix
baselines use one-pass native prefill, rather than an artificially chunked
baseline. Repair includes its source-alignment preparation time.

Times cover cache transition plus final-question processing. They exclude the
already-paid initial prefill, common compression work, model loading, and
generation. Compressor outputs were reused from the earlier run; we did not
rerun or retime the pruner. These are local component timings, not whole-chat
latency, provider charges, peak memory, or production throughput.

All **44 pruning-project tests pass**, including nine new tests that check:

- Full repair agrees with fresh shortened-prompt cache tensors and continuation.
- Rephasing alone fixes first-layer positions but leaves deeper-state differences.
- Original caches are not mutated; identity edits preserve every value.
- New tokens are always repaired and invalid alignments are rejected.
- Actual compressed text, source alignment, and withheld-question boundaries.

## Remaining work

The present code supports only batch-one, full-attention Qwen2 with default
RoPE. It does not test repeated compactions, production batching, other models,
or arbitrary abstractive summaries. Deep states that are reused can retain
influence from deleted text. Partial repair offers no exact-equivalence
guarantee, and equal scores on twelve mostly failed questions are weak evidence.

The next useful evaluation needs new conversations, substantially stronger
compressed-answer quality, and repeated compaction over complete sessions.
It should compare a model-informed repair selector with this boundary heuristic
at equal work budgets and include compressor/runtime overhead.

This requires inference-engine access. It does not make an ordinary hosted
text-completion API accept edited old KV tensors; OpenRouter's documented
[prompt-caching controls](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
provide routing and provider cache hints, not this repair operation. No claim
about The Token Company's implementation or comparative performance is made.

## Reproduce

From `pruning-model/`, with the prior public-data run and pinned model cached:

```sh
.venv/bin/python -m pruning_model.cache_repair_benchmark \
  --prior-run runs/kv-longmemeval-20260920 \
  --checkpoint runs/model \
  --output runs/cache-repair-new \
  --device mps --repetitions 3

.venv/bin/python -m pytest -q
```

Use a new output directory. Completed artifacts are in the Git-ignored
`runs/cache-repair-main-20260920/`: `manifest.json`, `edits.jsonl`,
`results.jsonl`, `summary.json`, `quality_review.json`, and `executed_source/`.
The two-case pilot used an earlier timing protocol and is excluded from this
report. The final results SHA-256 is
`6370044aface461ccc01933749ca85ef2372d307549e80716390bbee11e96158`.

Implementation: [cache repair](../../src/pruning_model/cache_repair.py),
[benchmark](../../src/pruning_model/cache_repair_benchmark.py).
