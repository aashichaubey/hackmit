"""Diagnostic of cache reuse AFTER real text compaction, using saved 149M outputs.

All arms receive exactly the same newly tokenized compressed prompt. Selection
uses edit locations only. Fresh compressed prefill is the reference, and exact
common-prefix reuse is the practical baseline. No hosted model calls are made.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenmix.evaluation.storage import append_jsonl, file_hash, read_jsonl, write_json
from .cache_repair import continue_after_edit
from .kv_benchmark import MODEL, MODEL_REVISION, clean_sessions, render_prompt
from .kv_cache import QwenKVSession, cache_bytes, select_cache, synchronize


def prepare_edit(case, plan, tokenizer):
    """Build the pruner's actual shortened text with exact source provenance.

    Newline joins and tokens retokenized at joins are not assumed reusable.
    We reuse the previous run's fixed pruning outputs, not its KV selection.
    """
    started = time.perf_counter()
    original, ranges = render_prompt(case, clean_sessions(case), tokenizer)
    if original != plan["prompt"] or len(ranges) != len(plan["pruning"]):
        raise ValueError("saved pruning plan does not match public case")
    pieces, provenance = [], []

    def copy(start, end):
        pieces.append(original[start:end])
        provenance.extend(range(start, end))

    cursor = 0
    for bounds, result in zip(ranges, plan["pruning"]):
        start, end = bounds["start"], bounds["end"]
        copy(cursor, start)
        if result.get("reason") != "selected":
            copy(start, end)
        else:
            kept = [s for s in result["spans"] if s["kept"]]
            if "\n".join(s["text"] for s in kept) != result["notes"]:
                raise ValueError("saved compressed text differs from retained spans")
            for i, span in enumerate(kept):
                if i:
                    pieces.append("\n")
                    provenance.append(None)
                left, right = start + span["start"], start + span["end"]
                if original[left:right] != span["text"]:
                    raise ValueError("source span mismatch")
                copy(left, right)
        cursor = end
    copy(cursor, len(original))
    compressed = "".join(pieces)
    return align_edit(case, plan["category"], original, compressed, provenance, tokenizer,
                      started=started)


def prepare_external_edit(case, replacements, tokenizer):
    """Align a compressor's exact output without changing its text.

    APIs without deletion offsets require inferred alignment. Equal character
    blocks provide a conservative monotone mapping, not the compressor's true
    deletion mask when repeated source text is ambiguous. Unmatched characters
    and retokenized boundaries are rebuilt, never assigned arbitrary cache rows.
    """
    from .kv_benchmark import category

    started = time.perf_counter()
    original, ranges = render_prompt(case, clean_sessions(case), tokenizer)
    if len(ranges) != len(replacements):
        raise ValueError("one compressor output is required per historical session")
    pieces, provenance, cursor = [], [], 0
    for bounds, replacement in zip(ranges, replacements):
        if not isinstance(replacement, str):
            raise ValueError("compressor output must be text")
        start, end = bounds["start"], bounds["end"]
        pieces.append(original[cursor:start])
        provenance.extend(range(cursor, start))
        pieces.append(replacement)
        aligned = [None] * len(replacement)
        # autojunk bounds work on long repetitive text; excluded matches become
        # mandatory recomputation, preserving correctness of every reused match.
        for block in difflib.SequenceMatcher(None, original[start:end], replacement).get_matching_blocks():
            aligned[block.b:block.b + block.size] = range(start + block.a, start + block.a + block.size)
        provenance.extend(aligned)
        cursor = end
    pieces.append(original[cursor:])
    provenance.extend(range(cursor, len(original)))
    return align_edit(case, category(case), original, "".join(pieces), provenance,
                      tokenizer, started=started)


def align_edit(case, category, original, compressed, provenance, tokenizer, *, started):
    assert len(provenance) == len(compressed)
    old = tokenizer(original, add_special_tokens=False, return_offsets_mapping=True)
    new = tokenizer(compressed, add_special_tokens=False, return_offsets_mapping=True)
    # Cache only the archive. The final question is processed AFTER compaction.
    marker = f"Question date: {case['question_date']}\nFinal question:"
    old_cut = next(i for i, (a, _) in enumerate(old["offset_mapping"]) if a >= original.rindex(marker))
    new_cut = next(i for i, (a, _) in enumerate(new["offset_mapping"]) if a >= compressed.rindex(marker))
    if old["input_ids"][old_cut:] != new["input_ids"][new_cut:]:
        raise ValueError("final question tokenization differs between histories")
    # Some tokenizers give multiple byte tokens the same Unicode character span.
    # Reuse only unambiguous offset/ID triples; rebuild ambiguous occurrences.
    offsets = {}
    for i, ((a, b), token) in enumerate(zip(old["offset_mapping"][:old_cut], old["input_ids"])):
        key = (a, b, token)
        offsets[key] = None if key in offsets else i
    mapping = []
    last_source = -1
    for i, (a, b) in enumerate(new["offset_mapping"][:new_cut]):
        source = provenance[a:b]
        match = None
        if source and source[0] is not None and source == list(range(source[0], source[0] + len(source))):
            match = offsets.get((source[0], source[-1] + 1, new["input_ids"][i]))
        if match is not None and match <= last_source:
            match = None
        if match is not None:
            last_source = match
        mapping.append(match)
    old_ids, new_ids = old["input_ids"][:old_cut], new["input_ids"][:new_cut]
    common = 0
    for a, b in zip(old_ids, new_ids):
        if a != b:
            break
        common += 1
    return {"case_id": case["question_id"], "category": category,
            "original_text": original, "compressed_text": compressed,
            "old_ids": old_ids, "new_ids": new_ids, "mapping": mapping,
            "query_ids": new["input_ids"][new_cut:], "common_prefix_tokens": common,
            "alignment_ms": (time.perf_counter() - started) * 1000}


def run_transition(model, original, edit, arm):
    new = edit["new_ids"]
    if arm == "fresh_compressed":
        session = QwenKVSession(model, chunk_size=len(new) + len(edit["query_ids"]))
        logits = session.append(new + edit["query_ids"])
        return session, {"recomputed_tokens": len(new), "reused_tokens": 0}, logits
    if arm == "exact_prefix":
        return continue_after_edit(model, original, edit["old_ids"], new, edit["query_ids"])
    fraction = int(arm.removeprefix("repair_")) / 100
    return continue_after_edit(model, original, edit["old_ids"], new, edit["query_ids"],
                               mapping=edit["mapping"], fraction=fraction)


def summarize(rows):
    arms = sorted({r["arm"] for r in rows})
    reference = {r["case_id"]: r for r in rows if r["arm"] == "fresh_compressed"}
    return {arm: {"cases": len(selected),
                  "reused_history_fraction": sum(r["reused_tokens"] for r in selected) / sum(r["new_history_tokens"] for r in selected),
                  "median_transition_ms": statistics.median(r["transition_ms"] for r in selected),
                  "median_first_token_kl": statistics.median(r["first_token_kl"] for r in selected),
                  "median_decode_ms": statistics.median(r["decode_ms"] for r in selected),
                  "median_transition_plus_decode_ms": statistics.median(r["transition_ms"] + r["decode_ms"] for r in selected),
                  "paired_transition_speedup": (sum(reference[r["case_id"]]["transition_ms"] for r in selected)
                                                 / sum(r["transition_ms"] for r in selected)
                                                 if all(r["case_id"] in reference for r in selected) else None),
                  "first_token_matches": sum(r["first_token_matches"] for r in selected),
                  "identical_answers": sum(r["identical_answer"] for r in selected),
                  "truncated": sum(r["finish_reason"] == "length" for r in selected)}
            for arm in arms if (selected := [r for r in rows if r["arm"] == arm])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prior-run", type=Path)
    source.add_argument("--bear2-run", type=Path, help="Completed Bear-2 preparation directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--arms", nargs="+", choices=("fresh_compressed", "exact_prefix", "repair_0",
                        "repair_10", "repair_25", "repair_50", "repair_100"),
                        default=["fresh_compressed", "exact_prefix", "repair_0", "repair_10",
                                 "repair_25", "repair_50", "repair_100"])
    args = parser.parse_args()
    if min(args.limit, args.max_new_tokens, args.repetitions) < 1:
        parser.error("limits must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a new, empty output directory")
    if len(args.arms) != len(set(args.arms)) or "fresh_compressed" not in args.arms:
        parser.error("arms must be unique and include fresh_compressed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    if args.bear2_run:
        prior = json.loads((args.bear2_run / "manifest.json").read_text())
        if (prior.get("status") != "complete" or prior.get("compressor") != "bear-2"
                or prior.get("target_revision") != MODEL_REVISION):
            parser.error("Bear-2 preparation must be complete and match the Qwen revision")
        for name, digest in prior["artifacts"].items():
            if file_hash(args.bear2_run / name) != digest:
                parser.error("Bear-2 preparation artifact changed")
        edits = read_jsonl(args.bear2_run / "edits.jsonl")[:args.limit]
        provenance = {"preparation_manifest": file_hash(args.bear2_run / "manifest.json"),
                      "artifacts": prior["artifacts"], "compressor": "bear-2",
                      "aggressiveness": prior["aggressiveness"],
                      "compression_api_calls": prior["successful_calls"]}
    else:
        prior = json.loads((args.prior_run / "manifest.json").read_text())
        if any(file_hash(args.checkpoint / name) != digest for name, digest in prior["checkpoint_hashes"].items()):
            parser.error("checkpoint differs from the saved pruning run")
        cases = json.loads((args.prior_run / "cases.json").read_text())[:args.limit]
        plans = {p["case_id"]: p for p in read_jsonl(args.prior_run / "plans.jsonl")}
        edits = [prepare_edit(c, plans[c["question_id"]], tokenizer) for c in cases]
        provenance = {"artifacts": {n: file_hash(args.prior_run / n)
                      for n in ("manifest.json", "cases.json", "plans.jsonl")},
                      "checkpoint_hashes": prior["checkpoint_hashes"], "compressor": "saved_modernbert"}
    if not edits or len({e["case_id"] for e in edits}) != len(edits):
        parser.error("nonempty unique cases are required")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ("cache_repair.py", "cache_repair_benchmark.py", "kv_cache.py", "kv_benchmark.py")
    write_json(args.output / "manifest.json", {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": MODEL, "revision": MODEL_REVISION, "torch": torch.__version__,
        "transformers": transformers.__version__, "api_calls": 0,
        "compression_provenance": provenance,
        "source_hashes": {n: file_hash(Path(__file__).with_name(n)) for n in sources},
        "selection": "first cases in the already fixed, previously inspected diagnostic set; not a new holdout",
        "compression_rerun": False,
        "quality": "answers saved for separate source-aware grading; reference agreement is not correctness",
        "arms": args.arms,
        "timing_scope": "one shape-specific warmup per case/arm then repeated transitions; excludes already-paid original prefill, common compression, generation; includes alignment for repair arms; full/prefix baseline use one-pass prefill",
    })
    snapshot = args.output / "executed_source"
    snapshot.mkdir()
    for name in sources:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    for edit in edits:
        append_jsonl(args.output / "edits.jsonl", edit)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    warmup = QwenKVSession(model)
    warmup.append(tokenizer.encode("A brief warmup.", add_special_tokens=False))
    del warmup
    rows, rng = [], random.Random(17)
    for edit in edits:
        original = QwenKVSession(model)
        original.append(edit["old_ids"])
        # Original-history quality control; never fed back to the repair policy.
        original_copy = QwenKVSession(
            model, cache=select_cache(original.cache, list(range(len(edit["old_ids"])))),
            positions=list(range(len(edit["old_ids"]))), seen_tokens=len(edit["old_ids"]),
        )
        synchronize(original_copy.device)
        original_query_started = time.perf_counter()
        full_logits = original_copy.append(edit["query_ids"])
        synchronize(original_copy.device)
        original_query_ms = (time.perf_counter() - original_query_started) * 1000
        original_decode_started = time.perf_counter()
        full_ids, full_finish = original_copy.generate(
            full_logits, max_new_tokens=args.max_new_tokens, eos_token_ids=eos,
        )
        synchronize(original_copy.device)
        original_decode_ms = (time.perf_counter() - original_decode_started) * 1000
        append_jsonl(args.output / "original_answers.jsonl", {
            "case_id": edit["case_id"], "output": tokenizer.decode(full_ids, skip_special_tokens=True),
            "output_tokens": len(full_ids), "finish_reason": full_finish,
            "cached_query_ms": original_query_ms, "decode_ms": original_decode_ms,
            "timing_scope": "single warm-cache continuation, excludes isolation copy and prior prefill",
        })
        del original_copy
        # Build an independent reference before timing randomized candidate arms.
        reference = QwenKVSession(model, chunk_size=len(edit["new_ids"]) + len(edit["query_ids"]))
        ref_logits = reference.append(edit["new_ids"] + edit["query_ids"]).float().cpu()
        ref_ids, _ = reference.generate(ref_logits, max_new_tokens=args.max_new_tokens, eos_token_ids=eos)
        del reference
        ref_logp = ref_logits.log_softmax(-1)
        arms = list(args.arms)
        rng.shuffle(arms)
        for arm in arms:
            warm, _, _ = run_transition(model, original, edit, arm)
            synchronize(warm.device)
            del warm
            timings = []
            for repetition in range(args.repetitions):
                synchronize(torch.device(args.device))
                started = time.perf_counter()
                session, stats, logits = run_transition(model, original, edit, arm)
                synchronize(session.device)
                timings.append((time.perf_counter() - started) * 1000)
                if repetition + 1 < args.repetitions:
                    del session
            elapsed = statistics.median(timings)
            logp = logits.float().cpu().log_softmax(-1)
            kl = max(0.0, float((ref_logp.exp() * (ref_logp - logp)).sum()))
            prompt_cache_bytes = cache_bytes(session.cache)
            synchronize(session.device)
            decode_started = time.perf_counter()
            output, finish = session.generate(logits, max_new_tokens=args.max_new_tokens, eos_token_ids=eos)
            synchronize(session.device)
            decode_ms = (time.perf_counter() - decode_started) * 1000
            row = {"case_id": edit["case_id"], "category": edit["category"], "arm": arm,
                   "original_history_tokens": len(edit["old_ids"]), "new_history_tokens": len(edit["new_ids"]),
                   "query_tokens": len(edit["query_ids"]), "common_prefix_tokens": edit["common_prefix_tokens"],
                   **stats, "transition_ms": elapsed + (edit["alignment_ms"] if arm.startswith("repair_") else 0),
                   "transition_measurements_ms": timings,
                   "alignment_ms": edit["alignment_ms"] if arm.startswith("repair_") else 0,
                   "compression_api_ms": edit.get("compression_api_ms"),
                   "decode_ms": decode_ms, "output_tokens": len(output),
                   "prompt_cache_bytes": prompt_cache_bytes,
                   "source_cache_bytes": cache_bytes(original.cache),
                   "first_token_max_logit_error": float((logits.float().cpu() - ref_logits).abs().max()),
                   "first_token_kl": kl, "first_token_matches": int(logits.argmax()) == int(ref_logits.argmax()),
                   "identical_answer": output == ref_ids, "output": tokenizer.decode(output, skip_special_tokens=True),
                   "reference_output": tokenizer.decode(ref_ids, skip_special_tokens=True), "finish_reason": finish}
            append_jsonl(args.output / "results.jsonl", row)
            rows.append(row)
            write_json(args.output / "summary.json", summarize(rows))
            print(f"{edit['case_id']} {arm}: reused={stats['reused_tokens']}/{len(edit['new_ids'])} KL={kl:.5f} transition={row['transition_ms']:.0f}ms", flush=True)
            del session
        del original
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
