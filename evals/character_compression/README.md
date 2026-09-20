# Character compression evaluation

This harness compares original and compressed requests to the same target model
under identical generation settings. Both actual answers are graded against
independent fixture golds by the **unchanged supplied scorer**. The original answer
is never used as the compressed answer's expected result.

The repository currently supplies a deterministic English–Mandarin phrase
compressor, a nine-entry illustrative translation dictionary, and no trained
character-code model. This task adds evaluation infrastructure around those
interfaces; it does not redesign them.

## Setup and offline execution

From the repository root:

```bash
uv sync --dev
uv run tokenmix-eval validate --config evals/character_compression/config.mock.json
uv run --offline tokenmix-eval run --config evals/character_compression/config.mock.json
uv run pytest -q
uv run python -m unittest discover -s evals/character_compression/starter -v
```

The default mock configuration uses an explicitly selected identity mock
compressor and a constant-response mock target. It runs **all 40 supplied cases**
without credentials or inference network access. It deliberately does not read
gold answers to construct outputs. Its synthetic byte counts are retained in
attempt logs and excluded from trustworthy-token metrics. A zero mock accuracy is
expected; mock accuracy is never a model-quality measurement.

`uv sync` needs package access on initial setup. The supplied scorer and default
mock execution need no tokenizer download. The real Tokenmix compressor needs
its configured tiktoken vocabulary cached; its first use can download that
vocabulary. For a fresh machine, provision the two existing compressor-test vocabularies once with `uv run python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]"`. Evaluation test fixtures disable network calls. After provisioning the cache, the real compressor can be evaluated offline against the mock target:

```bash
uv run tokenmix-eval run --config evals/character_compression/config.tokenmix-offline.json
uv run tokenmix-eval run --config evals/character_compression/config.dictionary-offline.json
```

The latter runs 98 additional, **separate development** probes: every enabled
dictionary entry in different grammatical contexts, native Chinese names and
quotes, identifiers, high density, cold requests, unseen combinations, unknown
mapping passthrough, absent information, name/number counterfactuals, and minimal
pairs for if/only-if, at-least/more-than, all/not-all, may/must, and transfer
direction. Their expected answers follow explicit synthetic facts or exact-copy
instructions. They are not generated from a model response.

Generate a new dictionary-specific development dataset without changing the
starter or an existing dataset:

```bash
uv run python -m tokenmix.evaluation.dictionary_cases \
  --dictionary src/tokenmix/seed.jsonl \
  --output evals/character_compression/fixtures/my-dictionary-v2.jsonl
```

## Bounded DeepSeek V4 Flash calls through OpenRouter

Set `OPENROUTER_API_KEY` in the process environment or a local `.env`. The CLI
reads simple `NAME=value` assignments without executing shell code. Credentials
are never included in configurations, manifests, or request logs. Authorization
headers are excluded; errors redact the active key and common bearer/key forms.

```bash
uv run tokenmix-eval validate --config evals/character_compression/config.openrouter.json
uv run tokenmix-eval run --config evals/character_compression/config.openrouter.json
uv run tokenmix-eval run --config evals/character_compression/config.openrouter-dictionary-smoke.json
```

The first config selects four unchanged starter fixtures, capped at eight target
calls. The second selects six unchanged dictionary probes, capped at twelve
calls. Both are development integration checks. The full suite has its own config, `config.openrouter-full-development.json`, with all 40 cases and an 80-call cap. Run it with the same `tokenmix-eval run --config` command. For extra repeats, copy that config **in this directory** and set `max_live_calls` to at least `80 * repeats`. This creates a separate experiment. Do not edit a running experiment's inputs.

The requested model is `deepseek/deepseek-v4-flash` and the provider is pinned to
`streamlake/fp8`, with `allow_fallbacks=false` and `require_parameters=true`.
The config uses temperature 0, top-p 1, reasoning disabled, and an output budget
of 128 tokens. These are experiment settings, not optimal settings for every
task. Actual returned model/provider IDs and finish reasons are logged. The
provider may not disclose tokenizer/weight revisions; those remain unknown.
Provider errors fail the case. No mock or identity compressor is substituted in
live mode.

Relevant provider references:

- [OpenRouter model entry](https://openrouter.ai/deepseek/deepseek-v4-flash)
- [Native usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
- [Provider pinning and fallback controls](https://openrouter.ai/docs/guides/routing/provider-selection)

The separate `.deepinfra.json` configs preserve the initial provider selection; its recorded live calls returned upstream-overload errors. Changing provider is a new experiment, never an unlogged fallback.

An original-versus-original control invokes the target independently twice with
the same request; it bypasses compression and uses a different fingerprint:

```bash
uv run tokenmix-eval run --config evals/character_compression/config.openrouter.json \
  --control original_vs_original
```

Concise English, same-mapping ASCII aliases, and alternative compressors are
explicitly marked unavailable. Configuring an unavailable variant fails rather
than pretending it was benchmarked.

## Protocol and inference boundaries

`Message` contains only `role` and `content`. The runner constructs immutable
message tuples before calling either adapter; fixture objects, golds, grader
definitions, tags, and evaluation results stay outside those interfaces. Existing
roles and turn order are preserved. Only user text is eligible for Tokenmix
rewrites; system and assistant messages stay intact. Any fixed
`boundary_instruction` is added identically to both arms.

`compression.scope` selects one of two distinct experiment types:

- `user_messages`: question-aware compression of each user message.
- `delimited_passages`: rewrite only text enclosed in the explicitly configured
  delimiters (default `<source>...</source>`). The compressor receives the passage
  without the question, and the same passage/configuration is compressed once,
  persisted, and reused across questions. Generation responses are never cached.

The Tokenmix adapter alone owns optional decoding overhead. It can attach a
versioned instruction, JSON-escaped used dictionary entries, and explicit
demonstrations. Those are actually included in the compressed request. Even if
the body is shorter, overhead may make the full request longer; the evaluator
reports expansion instead of discarding it. It never expands the compressed
passage back to English before sending it.

The existing compressor treats mappings as natural bilingual equivalents.
Its missing-mapping contract is to leave text unchanged; it has no invented-code
namespace, fine-tuned decoder, or custom escaping protocol. Its existing literal
protection heuristics are reused unchanged. Stale dictionary hashes fail
preflight. Conflicting/overlapping mappings follow the existing compressor's
behavior and are covered by repository unit tests. Deploying a new character-code
compressor requires a new explicit adapter and its actual decoding/missing-code
contract; these behaviors cannot be inferred from this prototype.

## Accounting and artifacts

Every configuration has a separate content fingerprint and results directory:

```text
runs/character_compression/<experiment-id>/
  manifest.json
  predictions.jsonl
  attempts.jsonl
  report.json
  report.md
  scored_cases.jsonl
  failures.jsonl
  trials.jsonl
  compression_cache.json
  pipeline_timings.json
```

The manifest records resolved configuration, fixture and source hashes, Git
revision, timestamps, separate dataset/dictionary/compressor/scorer/model
identities, optimization tokenizer, unknown target tokenizer revision, and known
limitations. Each prediction retains the starter-compatible field names and adds
original messages, independent expected answers, exact compressed messages,
substitutions, generation settings, and separate final-pipeline data.

Each call is journaled as started **before dispatch**, then completed with its
actual response, usage, resolved target IDs, provider request ID, latency, and
error. Call kinds are first/retry/fallback. Request JSON includes every transmitted
generation field and message; it excludes the authentication header. Sensitive
prompt contents remain local in ignored `runs/`.

`baseline_input_tokens` and `compressed_input_tokens` are complete **first-request
logical** counts from `usage.prompt_tokens`. They include chat framing and all
dictionary/instruction/example/tool-schema inputs actually transmitted. Cached
tokens are a subset and are never added to that total. There is no exact local
DeepSeek chat-template counter; missing provider usage stays null. The local
compressor's OpenAI tokenizer is never used to fill this gap. Mock byte estimates
are retained as synthetic raw measurements, never as measured target tokens.

OpenRouter's observed `usage.cost` is preserved as **credits**. Dollar costs stay
unknown because no versioned credit-to-dollar billing conversion or pricing
configuration is supplied. No prices are fetched or assumed. Local deterministic
compression has zero LLM input/output usage and zero API charge; its elapsed
compute time is recorded. This does not claim local compute is free.

Terminal errors have empty-string outputs and nonempty error fields and fail
grading. Null outputs are unmeasured. Missing input, cost, or latency measurements
remain unknown. The full-cohort token claim is withheld if any included pair lacks
trustworthy counts; partial-run subset diagnostics cannot qualify for release.

Default `first_attempt` mode forbids retries/fallback. To explore deployment
behavior in a **separate configuration**, use:

```json
{"mode": "deployment", "max_retries": 1, "fallback_to_original": true}
```

This is the value of the config's `pipeline` field. Retries and fallback trigger
only on terminal errors, never on gold answers or grades. Fallback makes a new
target call on the original request; it does not reuse the baseline output.
First-attempt metrics remain unchanged. Pipeline reports separately aggregate
all target attempts plus compressor usage, final accuracy, fallback frequency,
and active end-to-end latency p50/p95. Active latency includes compression and
the arm's retries/fallback; it excludes time spent running the other randomized
arm. Interrupted active latency is unknown. A fallback success is never counted
as successful compression.

## Scoring, uncertainty, and gates

```bash
uv run tokenmix-eval report runs/character_compression/EXPERIMENT_ID
uv run python evals/character_compression/starter/score_evals.py \
  --cases evals/character_compression/starter/eval_cases.jsonl \
  --predictions runs/character_compression/EXPERIMENT_ID/predictions.jsonl \
  --out /tmp/standalone-score.json --details /tmp/standalone-cases.jsonl
```

`tokenmix-eval report DIRECTORY --check-gate` exits 0 for PASS, 1 for FAIL, 3 for INCONCLUSIVE, and 4 for NOT_CONFIGURED.

The standalone supplied scorer requires all cases and equal repeats; use its
explicit `--allow-partial` only for balanced diagnostic subsets. The integrated
reporter handles an interrupted run by showing coverage and scoring complete
pairs separately per repeat. It never passes null outputs to the scorer or
silently drops terminal errors. It checks fixture/scorer hashes and reconciles
predictions with durable attempts before regenerating reports.

Exact grading only performs the fixture's explicitly allowed outer-whitespace
handling. JSON grading parses the whole response, rejects duplicate keys and
non-finite literals, preserves scalar types and array order, and ignores object
key order. Markdown fences and extra prose fail. Rubric/LLM-judge grading is
unsupported and rejected; no similarity score is substituted.

Reports include both accuracies, accuracy difference, correct-to-wrong regression
rate, all four paired outcomes, workload-wide full-request savings, request
savings distribution, expansion rate, and correct answers per 1,000 input tokens.
Slices include categories, source groups, character-length bins, source-character
substitution density, positions, and supplied tags. Character-based slice labels
are not token estimates. Each slice includes its sample count.

Critical checks are explicit assertions: exact expected literals, required output
contracts, opposite YES/NO answers, required UNKNOWN answers, and instruction
boundary outputs. Additional fixtures can declare `critical_checks` with a `type`
and an `assertion` of `equals_expected`, `not_in` (with `forbidden_outputs`), or
`strict_json`. Failures without a diagnostic assertion are `unclassified`. These
labels describe observed answers, not the cause of a model error. Per-entry
substitution exposure and co-occurring regressions are reported without causal
claims.

The original source-cluster bootstrap keeps related cases, both arms, and repeats
together. Reports record seed, resamples, independent clusters, and exploratory
95% intervals. Small samples and all-equal outcomes can produce degenerate
intervals. Zero observed failures does not prove zero future risk.

`release_policy.example.json` is **disabled and illustrative**, with 20% minimum
net savings and a 1 percentage-point non-inferiority margin. Application accuracy,
regression, sample-size, source-cluster, critical-check, and required-slice policies
are unset. Gate states are `NOT_CONFIGURED`, `INCONCLUSIVE`, `FAIL`, and `PASS`.
An enabled gate requires complete real held-out evidence, source splits preceding
dictionary learning, adequate samples/measurements, consistent resolved targets,
actual substitutions, and explicit thresholds. Non-inferiority uses the **lower
confidence bound**, not just the accuracy point estimate. Mock and development
runs cannot receive release qualification.

## Resumption and additional datasets

```bash
uv run tokenmix-eval run --config evals/character_compression/config.mock.json --resume
```

Configuration, fixture, dictionary, scorer, and implementation hashes must match.
Completed calls are not repeated. A started call without a completion becomes an
explicit interrupted failure with unknown provider outcome/usage; it is not
silently replayed. A separately configured deployment policy may retry such a
failure, with every call charged against its limit. A stale `.runner.lock` must
be removed only after confirming the owning process is gone. Malformed journals
are rejected, not silently repaired. Concurrency is currently exactly one; other
values fail validation.

New deterministic datasets use the preserved starter schema, unique `case_id`,
explicit `cluster_id`, `split`, source provenance, message sequences, and
independent expected outputs. Optional tags, source groups, and critical checks
never enter inference payloads. Keep entire related books/editions/translations
and their repeated questions in one source cluster; split before learning any
mappings. Freeze dictionary/settings/graders before release evaluation. No
held-out books or deployment data are supplied or claimed. Context-overflow
failures are flagged; capacity-expansion experiments need separate configurations
and analysis rather than being pooled with same-content comparisons.

To add a provider or compressor, implement the typed contracts in
`src/tokenmix/evaluation/adapters.py`, register it explicitly in config validation
and the runner, and add leakage/accounting/contract tests. Exactly one adapter
must own dictionary overhead. The target must return actual outputs or explicit
errors and must never infer missing usage.
