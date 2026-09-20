from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from tokenmix.evaluation.storage import (
    append_jsonl,
    digest,
    file_hash,
    read_jsonl,
    write_json,
)

from .data import validate_examples
from .model import ModernBertPruner, build_model, encode, save_checkpoint
from .spans import Span, assemble, segment


def examples_to_windows(rows, tokenizer, max_length: int):
    """Keep complete source units; repeat the unchanged query in each window.

    Windowed supervision is explicitly recorded: distant dependencies cannot be
    jointly observed in these windows. Inference never silently windows a note.
    """
    result, skipped = [], []
    for row in rows:
        if not row.get("answerable", True):
            # Absence of evidence is not a license to erase everything; the
            # production no-selection fallback retains the original notes.
            continue
        spans = [s for s in segment(row["notes"]) if not s.is_heading]
        group = []
        for span in spans:
            trial = group + [span]
            text = row["notes"][trial[0].start : trial[-1].end]
            local = [
                Span(s.id, s.start - trial[0].start, s.end - trial[0].start, s.text)
                for s in trial
            ]
            if (
                encode(tokenizer, row["question"], text, local, max_length=max_length)
                is not None
            ):
                group = trial
                continue
            if group:
                result.append(_window(row, group, tokenizer, max_length))
            group = [span]
            local = [Span(span.id, 0, len(span.text), span.text)]
            if (
                encode(
                    tokenizer, row["question"], span.text, local, max_length=max_length
                )
                is None
            ):
                skipped.append(
                    {
                        "id": row["id"],
                        "span": span.id,
                        "reason": "atomic span or query too long",
                    }
                )
                group = []
        if group:
            result.append(_window(row, group, tokenizer, max_length))
    return result, skipped


def _window(row, group, tokenizer, max_length):
    start, end = group[0].start, group[-1].end
    local = [Span(s.id, s.start - start, s.end - start, s.text) for s in group]
    inputs, memberships = encode(
        tokenizer,
        row["question"],
        row["notes"][start:end],
        local,
        max_length=max_length,
    )
    return {
        "inputs": inputs,
        "memberships": memberships,
        "labels": torch.tensor([float(s.id in row["keep_ids"]) for s in group]),
        "example_id": row["id"],
        "meeting_id": row["meeting_id"],
        "kind": row["kind"],
    }


def train(
    train_paths: list[Path],
    validation_path: Path,
    output: Path,
    *,
    model_name="answerdotai/ModernBERT-base",
    device="auto",
    max_length=1024,
    epochs=3,
    max_steps=0,
    accumulation=8,
    lr=2e-5,
    keep_weight=3.0,
    seed=17,
    max_examples=0,
):
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "training output must be new/empty; checkpoints are not overwritten"
        )
    if (
        max_length < 32
        or epochs < 1
        or accumulation < 1
        or lr <= 0
        or keep_weight <= 0
        or max_steps < 0
        or max_examples < 0
    ):
        raise ValueError("invalid training settings")
    rows = [r for path in train_paths for r in read_jsonl(path)]
    validation = read_jsonl(validation_path)
    validate_examples(rows, require_split="train")
    validate_examples(validation, require_split="val")
    validate_examples(rows + validation)
    if max_examples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_examples]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, model = build_model(model_name, device=device)
    if max_length > model.encoder.config.max_position_embeddings:
        raise ValueError("training context exceeds model support")
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    windows, skipped = examples_to_windows(rows, tokenizer, max_length)
    if not windows:
        raise ValueError("no trainable windows")
    metadata = {
        "architecture": "mean-pooled source-span binary classifier",
        "model_name": model_name,
        "model_revision": getattr(model.encoder.config, "_commit_hash", None),
        "max_length": model.encoder.config.max_position_embeddings,
        "training_max_length": max_length,
        "seed": seed,
        "train_files": {str(p): file_hash(p) for p in train_paths},
        "validation_file": file_hash(validation_path),
        "train_examples_digest": digest(rows),
        "training_meetings": sorted({r["meeting_id"] for r in rows}),
        "validation_meetings": sorted({r["meeting_id"] for r in validation}),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device),
        "windows": len(windows),
        "skipped": skipped,
        "hyperparameters": {
            "lr": lr,
            "epochs": epochs,
            "accumulation": accumulation,
            "keep_weight": keep_weight,
        },
        "torch_version": torch.__version__,
        "status": "training",
    }
    write_json(output / "training_manifest.json", metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    dev = next(model.parameters()).device
    pos_weight = torch.tensor(keep_weight, device=dev)
    model.train()
    step, started = 0, time.perf_counter()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(windows)
        for offset in range(0, len(windows), accumulation):
            group = windows[offset : offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for window in group:
                inputs = {k: v.to(dev) for k, v in window["inputs"].items()}
                logits = model(inputs, window["memberships"])
                loss = F.binary_cross_entropy_with_logits(
                    logits, window["labels"].to(dev), pos_weight=pos_weight
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                (loss / len(group)).backward()
                total_loss += float(loss.detach().cpu()) / len(group)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            row = {
                "step": step,
                "epoch": epoch + 1,
                "loss": total_loss,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if dev.type == "mps":
                row["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            append_jsonl(output / "training.jsonl", row)
            if step == 1 or step % 5 == 0:
                print(
                    f"step={step} loss={total_loss:.4f} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    if dev.type == "mps":
        torch.mps.synchronize()
    metadata.update(
        status="trained",
        optimizer_steps=step,
        training_seconds=time.perf_counter() - started,
        limited_run=bool(max_steps and step >= max_steps),
        training_examples=len(rows),
    )
    save_checkpoint(output, tokenizer, model, metadata)
    del optimizer, model
    if dev.type == "mps":
        torch.mps.empty_cache()
    calibrate(output, validation_path, device=device)
    return metadata


def calibrate(
    checkpoint: Path,
    validation_path: Path,
    *,
    device="auto",
    recall_targets=(0.99, 0.97, 0.95),
):
    if len(recall_targets) != 3 or not all(0 < r <= 1 for r in recall_targets):
        raise ValueError("three recall targets in (0, 1] are required")
    rows = read_jsonl(validation_path)
    validate_examples(rows, require_split="val")
    pruner = ModernBertPruner(checkpoint, device=device)
    if {r["meeting_id"] for r in rows} & set(pruner.metadata["training_meetings"]):
        raise ValueError("validation overlaps training meetings")
    scored = [(r, *pruner.scores(r["question"], r["notes"])) for r in rows]
    curve = []
    for i in range(101):
        threshold = i / 100
        retained, total, before, after = 0, 0, 0, 0
        for row, spans, scores, reason in scored:
            result = assemble(
                row["question"], row["notes"], spans, scores, threshold, reason=reason
            )
            total += len(row["keep_ids"])
            retained += len(set(row["keep_ids"]) & set(result.retained_ids))
            before += len(pruner.tokenizer.encode(row["notes"]))
            after += len(pruner.tokenizer.encode(result.notes))
        curve.append(
            {
                "threshold": threshold,
                "evidence_recall": retained / total if total else None,
                "compressor_token_savings": 1 - after / before if before else 0,
            }
        )
    if not any(r["evidence_recall"] is not None for r in curve):
        raise ValueError("validation needs answerable cases")
    # No-selection passthrough makes the curve non-monotonic. Maximize observed
    # validation savings subject to each recall target, never blindly raise t.
    targets = dict(zip(("conservative", "balanced", "aggressive"), recall_targets))
    selected = {
        name: max(
            (r for r in curve if r["evidence_recall"] >= recall),
            key=lambda r: (r["compressor_token_savings"], -r["threshold"]),
        )["threshold"]
        for name, recall in targets.items()
    }
    pruner.metadata.update(
        thresholds=selected,
        validation_file=file_hash(validation_path),
        validation_meetings=sorted({r["meeting_id"] for r in rows}),
    )
    write_json(checkpoint / "pruner.json", pruner.metadata)
    write_json(
        checkpoint / "calibration.json",
        {
            "split": "val",
            "recall_targets": targets,
            "thresholds": selected,
            "curve": curve,
            "note": "Compressor tokens are for tuning only; benchmark uses native target usage",
        },
    )
    print("Validation-selected thresholds:", selected, flush=True)
    return selected
