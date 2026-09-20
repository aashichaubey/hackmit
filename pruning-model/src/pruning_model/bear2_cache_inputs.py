"""Prepare real Bear-2 outputs for the local Qwen cache-repair benchmark.

Only historical session notes go to TTC. Final evaluation questions, references,
and evidence labels are never sent to the compressor. No fallback compressor or
automatic retries are used. Credentials are excluded from every saved artifact.
"""
from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import math
import os
import time
from pathlib import Path

from transformers import AutoTokenizer

from tokenmix.evaluation.config import load_dotenv
from tokenmix.evaluation.storage import append_jsonl, digest, file_hash, strict_json, write_json
from .cache_repair_benchmark import prepare_external_edit
from .kv_benchmark import MODEL, MODEL_REVISION, clean_sessions


def compress_sessions(case: dict, client, *, aggressiveness: float, record) -> tuple[list[str], float]:
    """Compress historical text only; a recorder journals each completed call."""
    replacements, total_ms = [], 0.0
    for session in clean_sessions(case):
        request = {"input": session["notes"], "model": "bear-2", "aggressiveness": aggressiveness}
        started = time.perf_counter()
        try:
            result = client.compress(session["notes"], model="bear-2", aggressiveness=aggressiveness)
        except Exception as exc:
            record({"case_id": case["question_id"], "session_id": session["id"],
                    "request_sha256": digest(request), "status": "failed",
                    "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                    "latency_ms": (time.perf_counter() - started) * 1000})
            raise RuntimeError(f"Bear-2 request failed ({type(exc).__name__}); no automatic retry") from None
        elapsed = (time.perf_counter() - started) * 1000
        if not isinstance(result.output, str):
            raise ValueError("Bear-2 response output must be a string")
        record({"case_id": case["question_id"], "session_id": session["id"],
                "request_sha256": digest(request), "status": "success", "latency_ms": elapsed,
                "model_requested": "bear-2", "aggressiveness": aggressiveness,
                "input_text": session["notes"], "output": result.output,
                "ttc_input_tokens": result.input_tokens, "ttc_output_tokens": result.output_tokens,
                "ttc_tokens_saved": result.tokens_saved})
        replacements.append(result.output)
        total_ms += elapsed
    return replacements, total_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--aggressiveness", type=float, default=.2)
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--env-file", type=Path, default=Path("../.env"))
    parser.add_argument("--prompt-key", action="store_true", help="Read the key without terminal echo")
    args = parser.parse_args()
    if (args.limit < 1 or args.max_calls < 1 or not math.isfinite(args.aggressiveness)
            or not 0 <= args.aggressiveness <= 1):
        parser.error("positive limits and aggressiveness in [0, 1] required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a new empty output directory; completed calls are never silently replayed")
    cases_file = args.prior_run / "cases.json"
    cases = strict_json(cases_file.read_text())[:args.limit]
    needed = sum(len(clean_sessions(case)) for case in cases)
    if not cases or needed > args.max_calls:
        parser.error(f"selected cases need {needed} API calls, outside configured limits")
    load_dotenv(args.env_file)
    key = getpass.getpass("TTC API key: ") if args.prompt_key else os.environ.get("TTC_API_KEY", "")
    if not key.strip():
        parser.error("set TTC_API_KEY in the environment/.env or use --prompt-key")
    from thetokencompany import TheTokenCompany

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    manifest = {
        "status": "running", "compressor": "bear-2", "aggressiveness": args.aggressiveness,
        "sdk_version": importlib.metadata.version("the-token-company"),
        "endpoint": "https://api.thetokencompany.com/v1/compress",
        "target_model": MODEL, "target_revision": MODEL_REVISION,
        "source_cases_sha256": file_hash(cases_file), "case_ids": [c["question_id"] for c in cases],
        "source_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "bear2_cache_inputs.py", "cache_repair_benchmark.py", "kv_benchmark.py")},
        "expected_calls": needed, "max_calls": args.max_calls, "successful_calls": 0,
        "failed_calls": 0, "ttc_input_tokens": 0, "ttc_output_tokens": 0,
        "compression_api_ms": 0, "completed_cases": 0,
        "selection": "same fixed 14-case public LongMemEval diagnostic; not a new holdout",
        "alignment": "inferred monotone equal-character blocks; ambiguous deletion provenance unknown",
        "cost": None, "cost_note": "SDK token counts recorded; monetary charges not returned",
    }
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "manifest.json", manifest)

    def record(row):
        append_jsonl(args.output / "compression.jsonl", row)
        manifest["successful_calls" if row["status"] == "success" else "failed_calls"] += 1
        manifest["compression_api_ms"] += row["latency_ms"]
        if row["status"] == "success":
            manifest["ttc_input_tokens"] += row["ttc_input_tokens"]
            manifest["ttc_output_tokens"] += row["ttc_output_tokens"]
        write_json(args.output / "manifest.json", manifest)

    try:
        with TheTokenCompany(api_key=key, timeout=60, app_id="qwen-cache-repair-benchmark") as client:
            for case in cases:
                outputs, elapsed = compress_sessions(case, client, aggressiveness=args.aggressiveness,
                                                     record=record)
                edit = prepare_external_edit(case, outputs, tokenizer)
                edit["compression_api_ms"] = elapsed
                append_jsonl(args.output / "edits.jsonl", edit)
                manifest["completed_cases"] += 1
                write_json(args.output / "manifest.json", manifest)
                print(f"{case['question_id']}: {len(edit['old_ids'])} -> {len(edit['new_ids'])} Qwen history tokens; "
                      f"TTC {elapsed:.0f}ms, alignment {edit['alignment_ms']:.0f}ms", flush=True)
    except BaseException:
        manifest["status"] = "incomplete"
        write_json(args.output / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest["artifacts"] = {name: file_hash(args.output / name)
                             for name in ("cases.json", "compression.jsonl", "edits.jsonl")}
    write_json(args.output / "manifest.json", manifest)
    print(args.output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
