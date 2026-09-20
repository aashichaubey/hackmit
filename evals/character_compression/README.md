# Character-compression evaluation harness

Measures whether a compressed request still produces the **independently
expected answer**, and how many **complete input tokens** it saves — including
dictionary and decoder overhead.

The supplied starter kit is preserved byte-for-byte; this harness wraps it.
See [`README.kit.md`](README.kit.md) for the original schemas, accounting rules
and limitations, which remain authoritative.

---

## ⚠️ Missing integration (read first)

**The real compressor does not exist yet.** `src/token_language/encoder/` is
empty, so `adapters.RepoCompressor` raises `NotImplementedError` naming exactly
what is required. A live run using the mock compressor is **refused by the
runner**, not silently substituted.

To complete the live path:

1. Build a phrase dictionary → `data/dictionaries/phrases.jsonl`
   (schema: `token_language.schema.DictionaryEntry`).
2. Implement `token_language.encoder.encoder.Encoder` with
   `Encoder.from_dictionary(path)` and `.encode(text) -> EncodingResult`.
3. Implement `RepoCompressor.compress()` to call it and to append the
   dictionary/decoder preamble (the compressor **owns** that overhead — the
   runner never adds it, so it cannot be double-counted).
4. Point `config.live.example.json` at the dictionary and run live.

Everything else — target LLM, token accounting, grading, reporting, gating — is
implemented and verified.

| Component | Status |
|---|---|
| Target LLM (OpenRouter → `deepseek/deepseek-v4-flash`) | ✅ real, verified live |
| Token accounting (provider usage + local DeepSeek tokenizer) | ✅ real, verified live |
| Scorer (supplied, unmodified) | ✅ 12/12 tests pass |
| Runner / reporting / policy gate | ✅ implemented, 76 tests |
| **Compressor** | ❌ **missing — see above** |

---

## Commands

```bash
cd evals/character_compression

# 1. Validate fixtures (structure, roles, graders, duplicate ids)
python run_evals.py validate --config config.example.json

# 2. Offline mock smoke run — no credentials, no network. SYNTHETIC results.
python run_evals.py run --config config.example.json

# 3. Score (the supplied scorer, unmodified)
D=runs/mock-first_attempt-<fingerprint>
python score_evals.py --cases eval_cases.jsonl \
  --predictions $D/predictions.scorer.jsonl \
  --out $D/report.json --details $D/scored_cases.jsonl

# 4. Readable report + failure diagnostics + gate
python report.py --run-dir $D --policy release_policy.example.json

# 5. Bounded live run (requires the real compressor, see above)
export OPENROUTER_API_KEY=...        # never put credentials in the config
python run_evals.py run --config config.live.example.json --max-live-calls 20

# Tests — offline, no credentials, no network
python -m pytest tests/ test_score_evals.py -q
python -m unittest test_score_evals -v     # supplied kit tests, unmodified
```

## Artifacts

One directory per configuration fingerprint, `runs/<experiment-id>/`:

| file | contents |
|---|---|
| `manifest.json` | resolved config, config/fixture SHA-256, adapter identities & mock flags, live-call budget vs. used, harness revision, limitations |
| `predictions.jsonl` | full records incl. `baseline_messages` + `compressed_messages` as actually sent |
| `predictions.scorer.jsonl` | the same records in the supplied scorer's schema |
| `attempts.jsonl` | per-attempt log: `first_attempt` / `retry` / `fallback`, finish_reason, reasoning tokens |
| `report.json` / `scored_cases.jsonl` | scorer output |
| `report.md` | readable report |
| `failures.jsonl` | every non-both-correct pair with exact requests, both outputs, tokens, substitutions |
| `gate.json` | per-check gate result |

## Experiment modes

| mode | meaning |
|---|---|
| `first_attempt` | **default.** No retries, no fallback. Measures compression quality. |
| `deployment_pipeline` | Retries + fallback. Reports first-attempt *and* final behaviour separately; a fallback to the original prompt is never credited as successful compression. |
| `baseline_vs_baseline` | Control: runs the original prompt in both arms to measure ordinary response variability. |

## Token accounting

Two sources, and the difference matters:

- **`provider_usage`** — OpenRouter's reported `prompt_tokens`. **Exact.**
  Eligible to back a release gate.
- **`approximate_local_template`** — local DeepSeek tokenizer over a
  hand-reconstructed chat framing. DeepSeek publishes **no `chat_template`**,
  so whole-request framing must be approximated. Measured drift on a live
  request: **38 local vs 39 provider (−2.6%)**. Diagnostic only; the policy
  gate forces `INCONCLUSIVE` if a savings gate relies on it.

Per-string tokenization is exact either way; only the chat framing is estimated.

Missing measurements stay `null` (**unknown**), never `0`.

## Reasoning-model handling

`deepseek/deepseek-v4-flash` is a reasoning model. With a small `max_tokens` it
returns `content: null` and `finish_reason: "length"`, having spent the entire
budget on reasoning tokens. **Verified live:** at `max_tokens=32` the answer is
empty; at `2048` it returns `Mira` and grades PASS.

The adapter reports that truncation as an explicit
`truncated_before_answer` **error**, so it is counted as a failure with a cause
rather than silently graded as a wrong answer. Set `generation.max_tokens` high
enough to cover reasoning **plus** the answer; both arms always use the same
budget.

## Safety properties (each enforced by a test)

- **No gold leakage.** `compress()` and `generate()` take plain
  `{role, content}` messages. `assert_no_gold_fields` raises `GoldLeakageError`
  on any payload carrying `expected_output` / `grader` / `cluster_id` /
  `split` / `source`. The runner holds gold only for grading.
- **No silent mocking.** Every adapter declares `is_mock`; live mode refuses
  mock adapters outright.
- **Mock runs can never be release-qualified.** `synthetic_results: true`
  forces the gate to `INCONCLUSIVE`.
- **Credentials** are read only from the environment, never from config, and
  are redacted from error text.
- **Independent conversations.** One fresh payload per case/repeat; nothing
  cached or reused. Arm order is shuffled from a recorded seed.

## Policy gate

States: `PASS`, `FAIL`, `INCONCLUSIVE`, `NOT_CONFIGURED`.

`release_policy.example.json` ships **disabled** and is illustrative only — its
20% savings / 1pp decline figures are examples from the brief, not an approved
standard. With no policy the gate returns `NOT_CONFIGURED`.

- Accuracy non-inferiority compares the **lower 95% bootstrap bound** of the
  accuracy delta against the permitted decline, not the point estimate.
- Quality and savings are **never** collapsed into one weighted score.
- Evidence adequacy (mock, partial coverage, approximate tokens, too few cases
  or clusters, development-split fixtures, missing required slices) forces
  `INCONCLUSIVE` regardless of how good the numbers look.

## Current measured status

The 40 fixtures are `development_smoke` — a smoke suite, **not** a release
benchmark. The only live measurements obtained so far are adapter-verification
calls (documented above). **No compression has been measured**, because the
compressor does not exist yet.

## Extending

- **New compressor** — implement the `Compressor` protocol in `adapters.py`
  (`name`, `is_mock`, `compress`), register it in `COMPRESSORS`.
- **New provider** — implement `Target` (`name`, `is_mock`, `generate`),
  register in `TARGETS`. Return `token_count_source="provider_usage"` when the
  provider reports input usage.
- **New fixtures** — append to a JSONL file matching the existing schema and
  point `fixtures` at it. `validate` checks structure; keep `cluster_id`
  grouped so the bootstrap keeps related cases together.
