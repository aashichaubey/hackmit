# Implementation and measured results

Implemented the paired evaluation CLI around the existing Tokenmix compressor. The supplied fixtures, prediction template, scorer, scorer tests, and README are unchanged; byte equality with the handoff was checked.

## Actual live measurements

All successful runs below used OpenRouter `deepseek/deepseek-v4-flash`, pinned to StreamLake, temperature 0, top-p 1, reasoning disabled, max output 128, one repeat, and no retries or fallback. These are development/integration results, not release evidence.

| Experiment | Pairs | Original correct | Comparison correct | Original input tokens | Comparison input tokens | Full-request savings | Changed pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Initial starter subset](../../runs/character_compression/70c29f9e9a40a510267e5278/report.md) | 4 | 4 | 4 | 347 | 347 | 0.00% | 0 |
| [Complete 40-case starter](../../runs/character_compression/5b911225f3eda961179a989c/report.md) | 40 | 39 | 37 | 69266 | 69266 | 0.00% | 0 |
| [Selected dictionary probes](../../runs/character_compression/159c537bcc68760189178696/report.md) | 6 | 6 | 6 | 217 | 209 | 3.69% | 4 |
| [Explicit original-vs-original control](../../runs/character_compression/46345f97f3eadc55eeb20c7d/report.md) | 4 | 4 | 4 | 347 | 347 | 0.00% | 0 |

The complete starter run had **zero substitutions and 40/40 identical paired request bodies**. Its 39/40 versus 37/40 answers therefore do not establish a text-compression effect. The observed original-correct/comparison-wrong cases were `negation_double` (NO → YES) and `boundary_quoted_instruction` (4 → 999); `conditional_only_if` was wrong in both arms. Gold answers and actual responses were not altered. [Exact failures](../../runs/character_compression/5b911225f3eda961179a989c/failures.jsonl).

The six dictionary probes exercised only the `asap` mapping, in four prompts. The other two checked literal passthrough. Observed complete-request savings were **8 tokens out of 217 (3.69%)**, with 6/6 correct in each arm. There were three source clusters; the exploratory 95% savings interval was 0%–4.26%. The degenerate accuracy interval [0, 0] is not proof of zero risk.

The initial DeepInfra experiments recorded **22 terminal HTTP 429 calls**, all with empty-string failure outputs and unknown token usage. Error details identify upstream shared-pool engine overload. They remain separate experiments; switching to StreamLake was an explicit configuration change, not hidden fallback.

Every run has gate **NOT_CONFIGURED** because the illustrative policy is disabled. Even with that policy enabled, these development datasets cannot qualify for release. Observed billing credits are retained; dollar costs and undisclosed target tokenizer/weight revisions remain unknown.

## Offline verification actually run

- `uv run pytest -q`: **69 passed**, including the 12 unchanged starter scorer tests. Evaluation adapter tests block network access and clear live credentials.
- `uv run python -m unittest discover -s evals/character_compression/starter -v`: **12 passed** (the same starter tests, not an additional model evaluation).
- `uv run --offline tokenmix-eval run --config evals/character_compression/config.mock.json`: all 40 cases, explicit mock compressor/target.
- `uv run --offline tokenmix-eval run --config evals/character_compression/config.tokenmix-offline.json`: all 40 cases, actual compressor and explicit mock target; zero matched substitutions.
- `uv run --offline tokenmix-eval run --config evals/character_compression/config.dictionary-offline.json`: 98 separate development probes, 19 changed pairs and seven substituted entries; target outputs are synthetic.
- `uv run --offline tokenmix-eval run --config evals/character_compression/config.mock.json --resume`: verified completed requests were reused without duplicate inference calls.
- Standalone supplied `score_evals.py` scored the 40-case mock predictions with the original schemas; output in `/tmp/tokenmix-standalone-score.json` and `/tmp/tokenmix-standalone-cases.jsonl`.
- Validation commands ran for mock and full live configurations. `git diff --check`, starter byte equality, Git ignore coverage for credentials/runs, and credential permissions (0600) were checked.

Live commands actually run:

```bash
uv run tokenmix-eval run --config evals/character_compression/config.openrouter.json
uv run tokenmix-eval run --config evals/character_compression/config.openrouter-dictionary-smoke.json
uv run tokenmix-eval run --config evals/character_compression/config.openrouter-full-development.json
uv run tokenmix-eval run --config evals/character_compression/config.openrouter.json --control original_vs_original
```

The first two commands were first attempted with DeepInfra (preserved in the `.deepinfra.json` configs), then with StreamLake. A separate `--max-cases 1` DeepInfra diagnostic recorded two additional failed calls.

## Files added or changed

- Reused `src/tokenmix/core.py` and `src/tokenmix/seed.jsonl`; no retraining or compressor-algorithm changes for this task.
- Added evaluation implementation:
  - `src/tokenmix/evaluation/__init__.py`
  - `src/tokenmix/evaluation/adapters.py`
  - `src/tokenmix/evaluation/cli.py`
  - `src/tokenmix/evaluation/config.py`
  - `src/tokenmix/evaluation/dictionary_cases.py`
  - `src/tokenmix/evaluation/policy.py`
  - `src/tokenmix/evaluation/reporting.py`
  - `src/tokenmix/evaluation/runner.py`
  - `src/tokenmix/evaluation/scoring.py`
  - `src/tokenmix/evaluation/storage.py`
- Added preserved starter, fixture hashes, configurations, policy, development probes, and tests:
  - `evals/character_compression/config.dictionary-offline.json`
  - `evals/character_compression/config.mock.json`
  - `evals/character_compression/config.openrouter-dictionary-smoke.deepinfra.json`
  - `evals/character_compression/config.openrouter-dictionary-smoke.json`
  - `evals/character_compression/config.openrouter-full-development.json`
  - `evals/character_compression/config.openrouter.deepinfra.json`
  - `evals/character_compression/config.openrouter.json`
  - `evals/character_compression/config.tokenmix-offline.json`
  - `evals/character_compression/fixtures/dictionary-development.jsonl`
  - `evals/character_compression/fixtures/integration-probes.jsonl`
  - `evals/character_compression/release_policy.example.json`
  - `evals/character_compression/starter/eval_cases.jsonl`
  - `evals/character_compression/starter/predictions.template.jsonl`
  - `evals/character_compression/starter/score_evals.py`
  - `evals/character_compression/starter/test_score_evals.py`
  - `evals/character_compression/starter.sha256.json`
  - `evals/character_compression/tests/conftest.py`
  - `evals/character_compression/tests/test_harness.py`
- Updated `pyproject.toml` with the `tokenmix-eval` command and evaluation test discovery.
- Updated `README.md`; added `evals/character_compression/README.md` and this report.
- Updated `.gitignore`; added `.env.example`. The actual credential is only in ignored `.env` with mode 0600.
- Generated exact local run artifacts under ignored `runs/character_compression/`; these include raw requests/responses and may contain sensitive prompt text.

## Remaining integration requirements

1. **Production character-code system:** the current adapter uses the existing nine natural-translation mappings. Supply the actual production dictionary/compressor version and its decoding, escaping, ambiguity, and missing-code contract to evaluate a different deployed code system. Relevant dictionary/instruction/example overhead must be configured explicitly.
2. **Release evidence:** supply source-split, independently labeled held-out book/deployment fixtures, a frozen dictionary, and application-approved accuracy/regression/sample/critical-slice thresholds. The 40-case starter and added development probes are not release tests.
3. **Future live runs:** keep a valid OpenRouter key and a supported pinned provider in configuration. The supplied temporary key is already integrated locally; its expiry will require replacing the environment value.
4. **Optional exact offline accounting / dollar costs:** no exact local DeepSeek chat template or versioned USD conversion is configured. Provider-native prompt usage and observed credits already work; missing provider data remains unknown.
5. **Unsupported extensions:** open-ended rubric/LLM judges, tool execution, concurrency above one, concise-English/ASCII comparison adapters, and capacity-expansion protocols require explicit implementations and validation. They are not silently approximated.

No further credential or provider integration is needed to reproduce the current bounded StreamLake runs while the supplied key remains valid.

## Artifact index

- [full_mock](../../runs/character_compression/e1480fda15999a75c1250cc0/report.md) (`e1480fda15999a75c1250cc0`)
- [real_compressor_mock_target](../../runs/character_compression/d42ec7351e5e7bbd8154a9a9/report.md) (`d42ec7351e5e7bbd8154a9a9`)
- [dictionary_mock](../../runs/character_compression/22579ab7daffbf45416399e5/report.md) (`22579ab7daffbf45416399e5`)
- [live_initial](../../runs/character_compression/70c29f9e9a40a510267e5278/report.md) (`70c29f9e9a40a510267e5278`)
- [live_full](../../runs/character_compression/5b911225f3eda961179a989c/report.md) (`5b911225f3eda961179a989c`)
- [live_dictionary](../../runs/character_compression/159c537bcc68760189178696/report.md) (`159c537bcc68760189178696`)
- [live_control](../../runs/character_compression/46345f97f3eadc55eeb20c7d/report.md) (`46345f97f3eadc55eeb20c7d`)
- [failed_deepinfra_starter](../../runs/character_compression/6358d9466f923fc57d87e2d5/report.md) (`6358d9466f923fc57d87e2d5`)
- [failed_deepinfra_dictionary](../../runs/character_compression/dc095ef307dfb7e34a154083/report.md) (`dc095ef307dfb7e34a154083`)
- [failed_deepinfra_diagnostic](../../runs/character_compression/15b4154180be21ca146aa4eb/report.md) (`15b4154180be21ca146aa4eb`)
