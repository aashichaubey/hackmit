# Tokenmix

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
