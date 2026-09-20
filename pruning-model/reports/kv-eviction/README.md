# Actual KV eviction with the 149M pruner

This experiment tests selective KV deletion. It does **not** solve cache reuse
after rewriting already-cached text; that separate question is tested in the
[cache-repair experiment](../cache-repair/README.md).

The prototype removes selected entries from a local answering model's real KV
cache and continues inference without replaying history. In this small diagnostic,
it reduced retained KV storage by **73.2%**, but factual answer correctness fell
from **8/12 to 5/12** and median pipeline time increased from **1.62 s to 1.82 s**
including the pruner. The current checkpoint does not demonstrate a favorable
quality/latency tradeoff for this use.

The target is **Qwen/Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, running FP16 with PyTorch SDPA on
the Mac's MPS GPU. The unchanged fine-tuned ModernBERT checkpoint has
**149,015,041 parameters**, its saved **0.48** threshold, and its normal
8,192-token overflow/no-selection fallbacks. No threshold was selected from
these test results. No DeepSeek, OpenRouter, or other hosted inference/judge
calls were made.

## Measured results

Fourteen public examples, three arms, **42 completed generations**:

| Policy | Factual answers correct | Retained prompt KV slots, summed | Mean prompt KV storage | Median pipeline time |
|---|---:|---:|---:|---:|
| Full KV | 8/12 (66.7%) | 54,376 | 106.2 MiB | 1,620 ms |
| Neural KV eviction | 5/12 (41.7%) | 14,549 | 28.4 MiB | 1,815 ms |
| Recency eviction, matched cache sizes | 4/12 (33.3%) | 14,549 | 28.4 MiB | 1,277 ms |

The two open-ended preference cases are reviewed qualitatively, not included in
the factual/abstention denominator. All three policies gave generic high-school
reunion advice without the requested personalization. On the evening-activities
case, full and neural output hit the 128-token limit, and all omitted the 9:30 pm
preference. These are disclosed incomplete generations, not API failures.

Quality was reviewed by the Codex agent against the released reference answers
and source evidence. This is **not independent human grading or the official
LongMemEval model judge**. It scores the requested answer, not every incidental
claim. The hash-bound review is saved with the run.

Four factual questions changed from correct with full KV to wrong with neural KV:

| Public case ID | Evidence lost / resulting error |
|---|---|
| `d7c942c3` | The mother's later switch to the same grocery-list app was removed; the model abstained. |
| `70b3e69b` | The Manolo García example was removed; the model denied having the example. |
| `29f2956b` | The user's 30-minute daily guitar practice was removed; the model abstained. |
| `caf9ead2` | The five-hour move was removed; the model substituted a two-hour drive to the parents' house. |

Neural eviction also fixed one full-cache error by correctly abstaining on an
unstated gift from the user's father. It outperformed matched recency on one
additional device-order question. This tiny sample does not establish a general
advantage over recency. The small target model already fails four of the twelve
factual/abstention questions with full context.

Median local pruner time was **475 ms** per example; median KV-selection/copy
time was **8.7 ms**. Neural median prefill time was 895 ms versus 1,074 ms for
full KV, but compressor overhead offset that improvement. Compression was
prepared separately, then its measured time was added to target time. Loading
is excluded. Each arm ran once with randomized arm order; output lengths differ.
These are diagnostic timings, not controlled production throughput estimates.
Recency receives matched budgets from the saved neural plan. Its time excludes
the scoring used to derive those budgets; it is a control for selection quality,
not an independently measured adaptive-budget controller.

## Public data and test protocol

Source: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and its official
[cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(MIT). The pinned dataset revision is
`98d7416c24c778c2fee6e6f3006e7a073259d48f`; the oracle JSON SHA-256 is
`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`.

This uses **oracle histories containing only evidence sessions**. It is not a
score on the standard full-history LongMemEval benchmark. Before compression or
generation, examples were selected deterministically with seed 17, two each from
abstention, knowledge updates, multi-session reasoning, assistant facts, user
preferences, user facts, and temporal reasoning. The local prompt limit was
6,144 Qwen tokens; 264 of the 500 examples were eligible. No selected example
was removed for compression failure or answer quality.

The fourteen cases contain twenty historical sessions. Sessions are sorted by
timestamp. Each session is prefetched in chunks, then scored by the pruner using
that session's last historical user message. The **final evaluation question,
reference answer, and evidence labels never guide retention**. All twenty
sessions produced a selected subset; none used a passthrough fallback.

Retained source spans are mapped by character offsets into the Qwen tokenizer's
positions. The model instructions, external session-date scaffolding, final
question, and recent 128-token window are protected. Tokens that straddle a
retained character are kept conservatively. The same token selection applies to
all layers and KV heads. At every session boundary, recency retains the same
number of slots and the same protected context as neural eviction.

## Implementation and correctness

- [KV session implementation](../../src/pruning_model/kv_cache.py): gathers
  surviving K/V rows from each full-attention DynamicCache layer. It preserves
  their values and original RoPE positions while compacting physical slots.
- [Benchmark](../../src/pruning_model/kv_benchmark.py): pinned dataset/model,
  label-free retention preparation, matched recency control, native tensor-byte
  accounting, run manifests, source snapshots, and explicit resumability checks.
- [Correctness tests](../../tests/test_kv_cache.py): chunked prefill matches the
  ordinary model; eviction preserves tensor values; continuation matches an
  independent reference that retains the original cache and masks evicted
  entries; repeated eviction never restores discarded positions.
- [Protocol tests](../../tests/test_kv_benchmark.py): chronological ordering,
  unchanged source data, and no influence of the future query/gold on historical
  retention.

**35 pruning-project tests pass**, including 11 new tests. Each arm prefetched
the same 54,376 original prompt tokens in aggregate, exactly once. The model
processed **zero replayed history tokens** to rebuild cache after eviction.

This is a batch-one, full-attention Qwen2 research prototype. It does not support
production prefix sharing, arbitrary model architectures, paged attention,
batched serving, or dropping KV during answer decoding. The benchmark incrementally
prefills a recorded archive; it is not a live agent executing tools. Reduced KV
changes inference, and surviving hidden states can still encode influence from
earlier discarded tokens. It is not equivalent to re-encoding shortened text.

Reported memory is **KV tensor storage**, not total or peak device memory.
Weights, activations, allocator reservations, and transient old-plus-new cache
copies are excluded. Initial input processing is still paid, and original
position/context limits still apply.

## Reproduce

Run from `pruning-model/` with its existing environment:

```sh
mkdir -p data/imported/longmemeval
curl -fL \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json \
  -o data/imported/longmemeval/longmemeval_oracle.json

.venv/bin/python -m pruning_model.kv_benchmark \
  --dataset data/imported/longmemeval/longmemeval_oracle.json \
  --checkpoint runs/model --output runs/kv-new-run \
  --per-category 2 --max-prompt-tokens 6144 \
  --max-new-tokens 128 --recent-tokens 128 --device mps

.venv/bin/python -m pytest -q
```

The Qwen checkpoint downloads on first use; no hosted inference key is needed.
Use a new output directory for changed settings. `--resume` requires identical
settings, checkpoint hashes, and source hashes and reuses completed local rows;
it is not a fresh timing experiment.

The completed run is in `pruning-model/runs/kv-longmemeval-20260920/` (Git-ignored):
`manifest.json`, `cases.json`, `plans.jsonl`, `results.jsonl`, `report.md`,
`report.json`, `quality_review.json`, `reviewed_report.json`, and the exact
`executed_source/` snapshot. Public source data and model provenance are also
saved in `pruning-model/runs/kv-feasibility-20260920/`. The answer review is a
separate manual artifact; future runs need their own quality review.
