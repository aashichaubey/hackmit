# Meeting-note pruner

Fine-tune ModernBERT-base to select source sentences/bullets given a question, then
measure whether a fixed answering LLM still answers correctly. This is a separate
experiment from the English–Mandarin dictionary compressor.

The completed local pilot, measured comparison, and artifact links are in
[the pilot report](reports/pilot/README.md). The initial checkpoint
loses answer accuracy; it is an experimental model, not a lossless compressor.

An [actual KV-eviction prototype and public-data test](reports/kv-eviction/README.md)
uses this saved 149M checkpoint to remove KV entries inside local
Qwen2.5-1.5B-Instruct. It preserves surviving cache tensors without replaying
history, but the initial diagnostic loses answer accuracy. No hosted inference
is involved; this is separate from text compression and provider prompt caching.

Run all commands below from this folder:

```sh
cd /Users/kaisong/code/hackmit/pruning-model
```

Code is in `src/pruning_model/`, tests in `tests/`, prepared examples in
`data/imported/`, trained checkpoints in `runs/model/`, and benchmark artifacts in
`reports/pilot/`. This project has its own `pyproject.toml`, `uv.lock`, and `.venv`.
It reuses the parent project's evaluation adapters/storage through the editable
local `tokenmix` dependency. Keep this folder inside the repository.

## Install and source data

```sh
uv sync --extra baselines --dev
uv run pruning-model fetch
uv run pruning-model import-public
```

`fetch` pins the QMSum and ExplainMeetSum revisions. `import-public` checks question
identity, source sentence evidence, and meeting split separation. It produces
`transcript_{train,val,test}.jsonl`, `notes_sources.jsonl`, and a provenance manifest
under `data/imported/`.

The initial note sources are **published generic meeting summaries formatted as
bullets**, not AI-generated notes. Their human sentence evidence comes from
ExplainMeetSum. Construction never consults specific questions. They are a useful
public proxy; results do not establish performance on arbitrary AI note products.
The original source licenses remain with the downloaded repositories.

Sources: [QMSum](https://github.com/Yale-LILY/QMSum),
[ExplainMeetSum](https://github.com/hkim-etri/ExplainMeetSum),
[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).

## Prepare and review notes questions

Put `OPENROUTER_API_KEY` in this folder's `.env` or the environment. The existing
repository-level `.env` is also supported, without copying credentials. The
default `configs/deepseek-streamlake.json` preserves the original DeepSeek V4 Flash
settings with provider fallback disabled.

```sh
uv run pruning-model annotate \
  --output data/imported/pilot \
  --train-meetings 12 --val-meetings 0 --test-meetings 0 \
  --journal runs/data-preparation/api.jsonl --max-calls 72
```

Each selected meeting gets questions, reference answers, atomic required facts,
and exact evidence quotes. Quotes are checked locally; a separately prompted
model audits answerability and support. This is model verification, **not human
certification**. Review a sample before training, and all evaluation labels where
practical. In particular, inspect questions marked unanswerable, incomplete lists,
and queries whose answers require conditions or antecedents.

`apply-review` accepts an explicit JSON review containing `input_digest`,
`reviewer`, and `examples`. Each reviewed example has `id`, `verdict`
(`accept`, `reject`, or `correct`), `reason`, and optional `correction`. Corrections
can update questions, answers, facts, evidence ids, tags, and answerability, but
cannot change notes, meeting ids, or splits. The input digest covers the combined
train/val/test rows in that order. A stale review fails.

```sh
uv run pruning-model apply-review \
  --input data/imported/pilot \
  --review evals/pilot-review.json \
  --output data/imported/reviewed-train

uv run pruning-model curate \
  --sources data/imported/notes_sources.jsonl \
  --questions evals/curated-questions.json \
  --train data/imported/reviewed-train/notes_train.jsonl \
  --output data/imported/reviewed
```

The checked-in review describes the exact pilot annotation version; newly
generated annotations require a new review digest and checked corrections. The
curated file supplies 18 validation and 18 test questions authored directly from
six held-out public note sources by the agent. These labels were prepared before
training, independently of the answering model. This is agent review, not
independent human certification. To generate your own evaluation questions using
the API instead, set nonzero validation/test meeting limits and review them.

API calls are journaled before invocation. Successes are reused on resume; 429,
502, 503, and 504 receive at most three total attempts with bounded backoff.
Every attempt consumes the call cap. Interrupted calls with unknown consumption
are never silently retried. Raise the explicit cap and rerun the same command to
continue a capped job. An unresolved call requires an explicit new journal to
retry; retain the previous journal for accounting.
An exhausted transient failure can be explicitly retried by increasing
`--max-attempts`; the default is three lifetime attempts per logical request.
No provider is silently substituted. A different `--target-config` produces
separate request identities and must use a new benchmark output directory.

## Train locally

```sh
# Short real-data training check, separate checkpoint:
uv run pruning-model train \
  --train data/imported/reviewed/notes_train.jsonl \
  --validation data/imported/reviewed/notes_val.jsonl \
  --output runs/smoke-model \
  --device mps --max-length 512 --max-steps 2 --accumulation 1

# Fine-tune the encoder and span classifier:
uv run pruning-model train \
  --train data/imported/reviewed/notes_train.jsonl \
  --validation data/imported/reviewed/notes_val.jsonl \
  --output runs/model \
  --device mps --max-length 1024 --epochs 3 --accumulation 8
```

The batch size is one; gradient accumulation controls the effective batch.
Training uses weighted binary cross-entropy (keep weight 3), AdamW, gradient
clipping, and encoder gradient checkpointing. The initial learning rate is 2e-5.
Reference answers and labels never enter inference inputs. Unanswerable examples
are reserved for evaluation rather than being used to teach “drop everything.”

For optional transcript supervision, add
`--train data/imported/transcript_train.jsonl`. Complete utterances
are packed into windows with the unchanged question repeated. Overlong atomic
spans are recorded as skipped, not silently truncated. Cross-window dependencies
are a limitation of this training mode. Training on short windows does not itself
verify quality at the maximum 8,192-token inference length.

Calibration uses validation data only. It selects the greatest observed token
savings satisfying evidence-recall targets of 99%, 97%, and 95%. These are
**validation targets**, not guarantees on future inputs. Because no-selection
passes through the input, the threshold curve can be non-monotonic. Calibration
evaluates the actual selection/fallback behavior. Identical selected thresholds
are combined in the benchmark rather than presented as different experiments.

For an explicitly exploratory sweep, calibrate a separate copy of the checkpoint
with `--recall-targets .99 .90 .80`. The pilot's default targets all chose 0.48;
the exploratory targets chose 0.48, 0.49, and 0.50 from the same validation data.
The first run is preserved. Do not select or adjust thresholds using test answers.
With passthrough fallbacks, a higher threshold need not save more tokens on test.

Model weights, tokenizer, classifier, training metadata, loss/timing logs, and
calibration curves are saved in the output directory. Existing training directories
are not overwritten. Colab/CUDA can use the same commands with `--device cuda`;
it is optional, not required for the Mac workflow.

## Prune

```sh
uv run pruning-model prune \
  --checkpoint runs/model \
  --question 'Who owns the migration and when is it due?' --notes meeting-notes.txt
```

Output includes the unchanged question, selected notes, retained source ids, and
each original span's text, score, selection, and Unicode character offsets.
Bullets with indented continuations remain atomic. Selected passages keep their
heading ancestors. Source text is copied, not generated. Empty selections and
inputs beyond the supported context window pass through unchanged.

For Python integration, load the model once and reuse it:

```python
from pathlib import Path
from pruning_model.model import ModernBertPruner

pruner = ModernBertPruner(Path("runs/model"), device="mps")
result = pruner.prune("Who owns the migration?", Path("meeting-notes.txt").read_text())
# Send result.question and result.notes to the answering LLM.
```

`pruning-model` is the main CLI; `tokenmix-pruner` remains an alias in this
project's environment. Historical training metadata keeps its original path
labels. Saved report/journal location pointers were updated when the project
moved; `relocation.json` records the mapping and artifact integrity checks.

## Paired benchmark

```sh
uv run --extra baselines pruning-model benchmark \
  --checkpoint runs/model \
  --dataset data/imported/reviewed/notes_test.jsonl \
  --output runs/benchmark --max-calls 400
```

During the initial run, StreamLake, Baidu, and DeepInfra returned persistent
shared-pool rate limits. A separately recorded run pins Parasail for the same
DeepSeek V4 Flash model:

```sh
uv run --extra baselines pruning-model benchmark \
  --checkpoint runs/model \
  --dataset data/imported/reviewed/notes_test.jsonl \
  --target-config configs/deepseek-parasail.json \
  --output runs/benchmark-parasail --max-calls 240 --workers 2
```

Only independent remote calls run concurrently. Neural pruning runs on one thread
before those calls. Journal writes and the shared call budget are synchronized.

The benchmark compares lexical sentence retrieval, the released LLMLingua-2 mBERT
compressor, and the calibrated ModernBERT pruner against the same original request.
Lexical and LLMLingua baselines target roughly 50% note retention. Their actual
complete-request token savings are measured, not assumed equal to 50%. The first
three cases also repeat the uncompressed request as a noise control.

Both answer arms use identical instructions, model, provider, temperature, and
generation settings. Answer and judge output limits are explicitly 1,024 tokens,
overriding the older dictionary experiment's 128-token limit equally for both
arms. Reference facts, evidence labels, and expected answers stay outside the
pruner and answering interfaces. A separate anonymous-answer judge assesses each
response against independently prepared source-backed references. It accepts
paraphrases, checks conditions and contradictions, and detects unsupported claims.
The judge uses the same pinned DeepSeek provider; judge error and correlated model
bias remain limitations. Inspect the saved verdicts and regression examples.

Only provider-reported `usage.prompt_tokens` counts as measured input usage.
Counts cover the complete request. Missing usage stays unknown. Output tokens are
recorded separately. Loss is original accuracy minus compressed accuracy in
percentage points; positive values mean worse answers. Required-fact completeness
and strict supporting-span recall are separate measures. LLMLingua deletes words,
so strict whole-span recall is particularly conservative for that baseline and
must not be mistaken for measured semantic information loss.

Artifacts:

- `per_call.csv`, `per_call.json`, `per_call.jsonl`: input tokens before/after,
  absolute and percentage savings, answers, fact scores, regressions, and evidence.
- `report.md`, `report.json`: aggregate scores, mean/median per-call savings,
  actual journal consumption, and paired meeting-cluster bootstrap intervals.
- `tradeoff.png`: observed token savings versus answer-accuracy loss.
- `regressions.json`: cases that became wrong, with exact removed passages.
  `text_changes` records exact character deletions/replacements, including partial
  sentences from LLMLingua; `removed_passages` lists spans not fully retained.
- `api.jsonl`, `manifest.json`: raw paid-call accounting and experiment identity.

Interrupted runs are marked partial; a partial cohort is not a complete benchmark.
The original answer is never used as the compressed answer's gold. Before/after
comparison totals reuse the original arm across variants; actual experiment call
consumption is reported separately, including judge and retry calls. Latency is a
secondary diagnostic rather than an acceptance gate.

Model judges can make factual mistakes. `audit-grades` applies an explicit,
hash-checked review into a separate directory and preserves all raw reports:

```sh
uv run pruning-model audit-grades \
  --source runs/benchmark-sweep-parasail \
  --dataset data/imported/reviewed/notes_test.jsonl \
  --review evals/benchmark-sweep-parasail-grading-review.json \
  --output runs/benchmark-sweep-parasail-reviewed
```

Every review identifies the reviewer and exact saved answer/source/grade digest.
An original-answer correction applies to every variant's pair. The checked-in
review belongs only to the recorded pilot; new calls require a new review. Agent
review is disclosed as such and is not independent human certification.

Five separately marked synthetic edge cases are in
`evals/adversarial.jsonl`. They cover corrections, conditions,
attribution, multiple passages, absent answers, and negation. Run them as a
separate diagnostic experiment, never pool them into the public held-out results.

## Checks

```sh
uv run --extra baselines pytest -q
```

Tests cover preserved source offsets/headings, atomic conditions, split leakage,
source evidence, stale review rejection, real tiny-transformer gradient updates and
checkpoint reload, overflow passthrough, API retry accounting, native-token metrics,
missing usage, independent reference grading, and meeting-cluster uncertainty.
