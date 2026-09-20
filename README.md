# InputCompressor

We investigate whether LLM context can be represented using fewer tokenizer tokens while retaining the information needed for downstream reasoning.

This project was built as a HackMIT research prototype for The Token Company challenge. It is an exploratory study, not a production compression system or an endorsed product.

## Motivation

LLMs operate over tokens, yet most context is written for humans in ordinary, often redundant natural language. We began with a deliberately provocative hypothesis:

> Why assume ordinary English is the optimal representation for an LLM?

We tested concise English, Mandarin, symbolic and hybrid notation, canonical formats, token-aware representation search, structured semantic representations, and risk-aware compression. The project became an empirical investigation rather than an attempt to confirm the initial hypothesis: several seemingly compact formats performed poorly, and the strongest practical baseline was straightforward concise English.

## Core research question

Can we reduce the number of tokenizer tokens required to represent context while preserving downstream information—and what tradeoffs arise from doing so?

Token count alone is not a sufficient objective. A useful system must consider:

- representation token count;
- downstream semantic fidelity;
- preprocessing and compression cost;
- representation stability; and
- possible cache and system effects.

We measured the first four in small experiments. We did **not** measure actual provider prompt-cache hits, KV-cache reuse, KV-cache bandwidth, latency, or cached billing.

## Experimental pipeline

```text
Original context
  → question-independent compression
  → target-tokenizer measurement
  → compressed representation
  → downstream QA on frozen questions
  → blinded LLM judging + deterministic matching
  → compression/fidelity analysis
```

The compressor does not receive downstream questions or expected answers. In the primary benchmark, both `COMPACT_ENGLISH` and generated `TOKEN_OPTIMIZED` candidates operate from a shared, question-independent `FACTS` extraction. `TOKEN_OPTIMIZED` generates three candidates, measures their actual `o200k_base` token counts, semantically validates them without questions, and selects the shortest candidate marked valid. The untouched Original is Candidate 0, so the optimizer never selects a longer generated representation.

The later `RISK_AWARE_COMPACT` experiment first identifies potentially fragile context spans—conditions, exceptions, negations, requirements, permissions, quantities, times, thresholds, ordering, dependencies, entities, and locations—then asks for concise natural language that preserves those spans' meaning. That experiment is post-hoc and exploratory.

## Approaches explored

| Strategy | Intended test | Observed outcome |
|---|---|---|
| `ORIGINAL` | Unmodified reference context | Highest primary-benchmark QA; no compression. |
| `FACTS` | Question-independent proposition extraction | Useful shared input, but extraction itself can omit or alter details and was not token-efficient in the three-context study. |
| `COMPACT_ENGLISH` | Preserve useful content with concise English | Best practical primary-benchmark tradeoff among the tested compressors. |
| `SYMBOLIC` | Logic, arrows, relations, and code-like notation | Did not consistently improve token efficiency or QA in the exploratory study. |
| `MANDARIN` | Test whether a different natural language tokenizes more densely | Expanded token count under `o200k_base` in the three-context study. |
| `HYBRID` | Mix English, Mandarin, abbreviations, and symbols | Inconsistent and less reliable than concise English. |
| `TOKEN_OPTIMIZED` | Generate, validate, and select the shortest actual-token candidate | Expensive preprocessing; did not beat Compact English on the primary benchmark. |
| `DENSE_CANONICAL` | Prompt an LLM to emit a stable compact format | Generative output was not reliably exact-token deterministic. |
| `SEMANTIC_COMPILER` | LLM extraction into structured IR followed by deterministic Python serialization | Serializer was deterministic for identical IR, but extraction varied and the representation expanded. |
| `RISK_AWARE_COMPACT` | Protect semantically fragile propositions before rewriting | Recovered some failures on a selected subset but sacrificed compression and introduced new failures. |

Failed and negative experiments are retained under `results/`; they are part of the research history.

## Primary benchmark

`data/benchmark_v1.json` is a frozen synthetic development benchmark with:

- 50 context instances;
- 5 questions per context, or 250 QA pairs;
- 10 categories; and
- approximately 200–229 words per context.

The 50 instances are generated from approximately ten underlying category templates with five variants each. They must not be interpreted as 50 fully independent natural documents.

Frozen SHA-256:

```text
ad7b34f0308183d28de4e0f67184bda72a50c918dcbe6e0dfc38668fc74739d6
```

### Main results

These values were recomputed from `results/benchmark_v1/full_50/representations.csv` and `qa_results.csv`:

| Strategy | Mean representation tokens | Mean representation reduction | Official blinded-judge QA |
|---|---:|---:|---:|
| `ORIGINAL` | 252.92 | 0.00% | 235/250 (94.0%) |
| `COMPACT_ENGLISH` | 179.06 | 29.08% | 216/250 (86.4%) |
| `TOKEN_OPTIMIZED` | 194.04 | 23.21% | 212/250 (84.8%) |

The raw artifacts report **194.04** mean tokens for `TOKEN_OPTIMIZED`; this repository uses that recomputed value rather than an earlier draft figure of 194.22.

The corrected question-level paired analysis finds:

| Strategy | Original-correct → compressed-wrong | Original-wrong → compressed-correct | Contexts with ≥1 additional failure | Saved tokens with no additional question failure |
|---|---:|---:|---:|---:|
| `COMPACT_ENGLISH` | 25 | 6 | 20/50 | 30/50 |
| `TOKEN_OPTIMIZED` | 28 | 5 | 21/50 | 26/50 |

An earlier analysis compared only aggregate correct counts within each context. That hid cases where one lost answer was offset by a gain on another question and incorrectly reported 16 Compact English failure contexts. Raw answers and judge decisions have not been changed; only the derived paired accounting was corrected.

> **What 29.08% means:** the Compact English context representation used 29.08% fewer `o200k_base` tokens on average than Original.

It does **not** mean 29% lower total inference cost, latency, KV-cache use, API cost, or memory bandwidth. Questions, system prompts, chat framing, generated answers, compression calls, and judging are outside this representation-only percentage.

## Key findings

- Concise English was surprisingly competitive and outperformed the more expensive representation-search method on this development benchmark.
- Mandarin and heavy symbolic or structured encodings did not automatically improve token efficiency for the target tokenizer.
- TOKEN_OPTIMIZED search consumed substantial preprocessing tokens and did not beat direct concise rewriting in the primary comparison.
- Plausible compressed text can still lose conditions, exceptions, distinctions, quantities, dates, ordering, or causal relationships.
- Generative canonicalization was not reliably exact-token deterministic, even at temperature 0.
- A deterministic serializer was perfectly stable given identical structured input, but the upstream LLM extraction was not stable.
- Optimizing representation length alone is insufficient.

A more realistic conceptual objective is:

```text
total utility / cost =
  representation token cost
  + compression and validation cost
  + semantic loss
  + deployment and cache effects
```

The final term is future work; this repository does not measure real cache behavior.

## Negative results

- Mandarin did not provide the expected tokenizer advantage.
- Formal and symbolic representations were not consistently more token-efficient.
- Explicit semantic IR expanded the representation in the three-context experiment.
- Brute-force representation search was expensive and weaker than Compact English in the primary benchmark.
- LLM-generated canonical text was unstable across repeated identical requests.
- Semantic validation accepted some candidates that altered untested details.
- Risk-aware compression showed a tradeoff, not an unconditional improvement.

## Risk-aware compression

The completed experiment uses 16 contexts selected post-hoc because Compact English had a lower **net context QA score** than Original. It is not a held-out evaluation and, because of the original paired-analysis bug, it is not the complete set of 20 contexts with at least one question-level loss.

| Strategy on selected 16 contexts | Mean tokens | Mean reduction | Official judged QA |
|---|---:|---:|---:|
| `ORIGINAL` | 254.56 | 0.00% | 75/80 (93.75%) |
| `COMPACT_ENGLISH` | 171.19 | 32.68% | 55/80 (68.75%) |
| `RISK_AWARE_COMPACT` | 194.00 | 23.61% | 59/80 (73.75%) |

Risk-Aware recovered 9 of the 21 compression-attributable failures inside this selected subset, while introducing 6 new failures relative to Original. It retained roughly 72% of Compact English's token savings. This is evidence worth following up on, not validation that the method solves semantic fidelity.

Because this run resumed from a local API cache, the usage block in `risk_aware_v1/report.json` covers only the final invocation. An audit of the complete local cache found 192 calls, 44,418 input tokens, 11,371 output tokens, and approximately $0.03596 at the recorded list-price assumptions. The cache itself is intentionally not published.

## Methodology

- **Generation, QA, judge:** `gpt-4.1-mini`; cached responses identify the concrete snapshot as `gpt-4.1-mini-2025-04-14`.
- **Tokenizer:** `o200k_base`, resolved for the configured target model with `tiktoken`.
- **Temperatures:** primary FACTS and Compact English generation 0; token-search generation 0.8; downstream QA, semantic validation, and judging 0.
- **Benchmark:** 50 synthetic template-derived contexts × 5 questions.
- **Question independence:** compression functions accept context or shared facts, never downstream questions or answers.
- **Evaluation:** deterministic normalized matching plus an LLM judge given question, expected answer, and model answer, but not strategy or token count.
- **Token search:** three generated candidates plus Candidate 0/Original; shortest candidate marked semantically valid wins.
- **Statistics:** 10,000 bootstrap draws with seed 1729, resampling contexts. Because template variants are correlated, these intervals are descriptive rather than population-level evidence.
- **Local caching:** completed API responses are cached for resumability during execution. Caches are excluded from Git; published result artifacts contain the representations and decisions needed for inspection.
- **Cost accounting:** benchmark `usage.json` records calls incurred by that invocation and can exclude work reused from an earlier local cache. It is not an API invoice or a full lifecycle-cost measurement.

## Limitations

- The primary benchmark is synthetic and template-derived.
- The 50 contexts come from approximately ten underlying templates/categories with multiple variants, not 50 independent natural documents.
- Five QA items probe only a subset of the information in each context.
- Questions and expected answers were authored alongside the synthetic contexts; they are hidden from compressors at runtime but do not constitute an independently created test set.
- The LLM judge is noisy. Original itself received only 94% official judged accuracy, and some correct answers with harmless detail were rejected.
- Deterministic matching can accept incomplete substring answers or reject valid paraphrases.
- FACTS extraction and semantic validation use LLMs and can omit, alter, or overlook details.
- Several later methods, especially Risk-Aware, were designed after examining earlier failures and are exploratory rather than held-out validation.
- Bootstrap intervals treat contexts as independent even though variants share templates.
- Compression and validation overhead can exceed downstream token savings at the tested reuse count.
- We measured representation tokens, not full production inference cost.
- We did not directly measure provider prompt-cache hits, KV-cache reuse, latency, memory bandwidth, or cached billing.
- Fresh API reproduction may vary despite temperature 0; local result artifacts are the historical record.
- These are hackathon-scale exploratory findings, not production benchmarks.

## What we learned

We started by asking:

> What is the shortest language an LLM can understand?

The experiments pushed us toward a better question:

> What representation minimizes total system cost while preserving the information needed downstream?

That objective must include semantic fidelity and preprocessing cost—not only the number of tokens in the final string.

## Running the project

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For real API runs, edit `.env` without committing it:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
GENERATION_MODEL=gpt-4.1-mini
ANSWER_MODEL=gpt-4.1-mini
JUDGE_MODEL=gpt-4.1-mini
TARGET_MODEL=gpt-4.1-mini
TOKENIZER_ENCODING=
GENERATION_TEMPERATURE=0
TOKEN_SEARCH_TEMPERATURE=0.8
```

Run non-API tests and validate the frozen benchmark:

```bash
python -m pytest -q
python -m src.benchmark --validate-only
```

Run the three-context exploratory pipeline:

```bash
python -m src.run_experiment --limit 3
```

Run a five-context benchmark pilot or the full benchmark:

```bash
python -m src.benchmark --limit 5
python -m src.benchmark --all
```

These commands make real API calls when `LLM_PROVIDER=openai`. The full benchmark is expensive; existing raw outputs are already committed for inspection. API generation is nondeterministic, so a fresh run need not reproduce text byte-for-byte.

Rebuild derived benchmark summaries and plots without API calls:

```bash
python -m src.benchmark_report
```

Exploratory stability/compiler runs are available as:

```bash
python -m src.dense_canonical
python -m src.semantic_compiler
```

They also call the API and should not be rerun merely to inspect the existing results.

## Repository structure

```text
data/
  examples.json                 # original three-context exploratory set
  benchmark_v1.json             # frozen 50-instance synthetic benchmark
  benchmark_v1.sha256           # benchmark integrity hash
src/
  representations.py            # FACTS, encoders, token search, semantic validation
  tokenizer.py                  # target-model token counting
  evaluate.py                   # deterministic and LLM-judge evaluation
  run_experiment.py             # three-context exploratory runner
  benchmark.py                  # cached primary benchmark runner
  benchmark_stats.py            # summaries, paired transitions, bootstrap, plots
  benchmark_report.py           # API-free post-run reporting
  dense_canonical.py            # generative stability experiment
  semantic_compiler.py          # structured extraction + deterministic serializer
  risk_aware.py                 # post-hoc risk-aware experiment
tests/                           # integrity and question-leakage regression tests
results/
  benchmark_v1/full_50/         # primary frozen raw outputs and derived reports
  benchmark_v1/pilot_5/         # five-context pilot history
  risk_aware_v1/                # post-hoc selected-subset experiment
  *.json, *.csv, *.png          # three-context exploratory history
```

## Reproducing and interpreting existing results

The result directories preserve raw generated representations, downstream model answers, judge decisions, summaries, and plots. API response caches are intentionally excluded because they are execution caches rather than research artifacts.

Use `python -m src.benchmark_report` to recompute the full benchmark's derived summaries from existing raw CSV files without spending API credits. Do not manually edit judge outcomes; apparent evaluator errors are recorded separately while official scores remain unchanged.

## Neural pruning and KV-cache experiments

The separate [pruning-model project](pruning-model/README.md) contains the fine-tuned
149M ModernBERT pruner and a local Qwen2.5-1.5B KV-eviction prototype.
The [KV-eviction report](pruning-model/reports/kv-eviction/README.md) records
73.2% lower retained KV storage with lower answer accuracy in a small public-data
diagnostic. These results are separate from the representation experiments above.

The [Bear-2 cache-repair experiment](pruning-model/reports/cache-repair/bear2-20260920.md)
reuses Qwen KV states after real text compression. Selective repair reused 66.7%
of compressed-history states and reduced median transition time from 587 ms to
305 ms in a 14-case diagnostic. Fresh and repaired caches both scored 7/12
on factual questions, but outputs differed; this remains approximate.

## Tokenmix dictionary experiment

The meeting-note pruning project lives in [pruning-model](pruning-model/README.md),
with its own code, environment, data, checkpoints, and benchmark reports.

An experimental English–Mandarin dictionary compressor and a paired evaluation
harness. The compressor uses exact phrase matches; the evaluator measures actual
task correctness and complete target-request usage. These are separate components.

```bash
uv sync --dev
uv run tokenmix compress 'Explain artificial intelligence as soon as possible.' --json
uv run tokenmix-eval validate --config evals/character_compression/config.mock.json
uv run --offline tokenmix-eval run --config evals/character_compression/config.mock.json
uv run pytest -q
```

See [the evaluation guide](evals/character_compression/README.md) for OpenRouter /
DeepSeek V4 Flash configuration, logs, scoring, controls, resumption, and gates.
The supplied 40 development fixtures and scorer are preserved byte-for-byte in
`evals/character_compression/starter/`. Mock runs are explicitly synthetic.

The current nine-entry dictionary contains illustrative natural translations,
not a trained character-code system. Unknown phrases remain unchanged. The
compressor's `o200k_base` optimization heuristic is **not** DeepSeek's tokenizer;
live evaluations use OpenRouter's native prompt-token usage. Earlier files in
`reports/` contain only OpenAI-tokenizer pair counts, not DeepSeek model-quality
measurements or complete-chat-request savings.

```bash
# Local phrase-pair exploration, not a downstream correctness evaluation:
uv run tokenmix benchmark --summary
# Imported pairs remain disabled until explicitly reviewed:
uv run tokenmix import-po path/to/translations.po --source 'URL/revision/license' --output candidates.jsonl
```

Credentials belong in `.env` or the process environment. `.env` and `runs/` are
ignored by Git; evaluation artifacts can contain exact source prompts and answers.
