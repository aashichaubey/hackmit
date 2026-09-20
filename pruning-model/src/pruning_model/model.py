from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .spans import PruneResult, Span, assemble, segment


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


class SpanClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, inputs: dict, memberships: list[list[int]]) -> torch.Tensor:
        hidden = self.encoder(**inputs).last_hidden_state[0]
        pooled = torch.stack([hidden[indices].mean(dim=0) for indices in memberships])
        return self.classifier(pooled).squeeze(-1)


def build_model(name: str, *, device: str = "auto", revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    encoder = AutoModel.from_pretrained(
        name, revision=revision, attn_implementation="sdpa", reference_compile=False
    )
    model = SpanClassifier(encoder).to(select_device(device))
    return tokenizer, model


def encode(tokenizer, question: str, notes: str, spans: list[Span], *, max_length: int):
    # Never silently truncate a question or a note. sequence_ids distinguishes
    # identical character offsets in the question and in the source notes.
    encoded = tokenizer(question, notes, return_offsets_mapping=True, truncation=False)
    if len(encoded["input_ids"]) > max_length:
        return None
    sequence_ids = encoded.sequence_ids()
    offsets = encoded.pop("offset_mapping")
    memberships = []
    for span in spans:
        indices = [
            i
            for i, ((a, b), seq) in enumerate(zip(offsets, sequence_ids))
            if seq == 1 and b > a and a < span.end and b > span.start
        ]
        if not indices:
            raise ValueError("content span has no tokenizer coverage")
        memberships.append(indices)
    inputs = {
        key: torch.tensor([values], dtype=torch.long)
        for key, values in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    return inputs, memberships


def save_checkpoint(path: Path, tokenizer, model: SpanClassifier, metadata: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(path / "encoder")
    save_file(
        {
            k: v.detach().cpu().contiguous()
            for k, v in model.classifier.state_dict().items()
        },
        str(path / "classifier.safetensors"),
    )
    from tokenmix.evaluation.storage import write_json

    write_json(path / "pruner.json", metadata)


class ModernBertPruner:
    def __init__(self, checkpoint: Path, *, device: str = "auto"):
        self.metadata = json.loads((checkpoint / "pruner.json").read_text())
        self.tokenizer, self.model = build_model(
            str(checkpoint / "encoder"), device=device
        )
        self.model.classifier.load_state_dict(
            load_file(str(checkpoint / "classifier.safetensors"))
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_length = min(
            self.metadata["max_length"],
            self.model.encoder.config.max_position_embeddings,
        )

    def scores(self, question: str, notes: str):
        spans = segment(notes)
        content = [s for s in spans if not s.is_heading]
        if not content:
            return spans, [1.0] * len(spans), "empty_content_passthrough"
        encoded = encode(
            self.tokenizer, question, notes, content, max_length=self.max_length
        )
        if encoded is None:
            return spans, [1.0] * len(spans), "context_overflow_passthrough"
        inputs, memberships = encoded
        with torch.inference_mode():
            probs = (
                self.model(
                    {k: v.to(self.device) for k, v in inputs.items()}, memberships
                )
                .sigmoid()
                .cpu()
                .tolist()
            )
        by_id = dict(zip([s.id for s in content], probs))
        return spans, [by_id.get(s.id, 1.0) for s in spans], None

    def prune(
        self, question: str, notes: str, *, threshold: float | None = None
    ) -> PruneResult:
        started = time.perf_counter()
        spans, scores, reason = self.scores(question, notes)
        if threshold is None:
            threshold = self.metadata.get("thresholds", {}).get("conservative", 0.5)
        result = assemble(question, notes, spans, scores, threshold, reason=reason)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
