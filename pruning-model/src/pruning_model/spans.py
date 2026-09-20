from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Span:
    id: int
    start: int
    end: int
    text: str
    heading_ids: tuple[int, ...] = ()
    is_heading: bool = False


@dataclass
class PruneResult:
    question: str
    notes: str
    spans: list[dict]
    retained_ids: list[int]
    reason: str = "selected"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def segment(notes: str) -> list[Span]:
    """Keep bullets (including indented continuations) atomic; attach headings.

    Offsets are Python Unicode character offsets into the unmodified source.
    Sentence splitting is deliberately conservative around abbreviations.
    """
    spans: list[Span] = []
    headings: list[tuple[int, int]] = []
    lines = list(re.finditer(r"[^\n]+(?:\n|$)", notes))
    index = 0
    while index < len(lines):
        line = lines[index]
        raw = line.group().rstrip("\r\n")
        if not raw.strip():
            index += 1
            continue
        start = line.start() + len(raw) - len(raw.lstrip())
        end = line.start() + len(raw.rstrip())
        stripped = notes[start:end]
        heading = re.match(r"^(#{1,6})\s+", stripped)
        bare_heading = (
            not re.match(r"^[-*+]\s|^\d+[.)]\s", stripped)
            and len(stripped) < 100
            and stripped.endswith(":")
        )
        if heading or bare_heading:
            level = len(heading[1]) if heading else 6
            headings = [(depth, hid) for depth, hid in headings if depth < level]
            sid = len(spans)
            spans.append(
                Span(
                    sid,
                    start,
                    end,
                    notes[start:end],
                    tuple(hid for _, hid in headings),
                    True,
                )
            )
            headings.append((level, sid))
        else:
            bullet = bool(re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped))
            if bullet:
                while index + 1 < len(lines):
                    following = lines[index + 1]
                    text = following.group().rstrip("\r\n")
                    if (
                        not text.strip()
                        or not text.startswith(("  ", "\t"))
                        or re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", text)
                    ):
                        break
                    end = following.start() + len(text.rstrip())
                    index += 1
                bounds = [(start, end)]
            else:
                bounds = []
                cursor = start
                for match in re.finditer(
                    r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", notes[start:end]
                ):
                    boundary = start + match.start()
                    if re.search(
                        r"\b(?:Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e|[A-Z])\.$",
                        notes[cursor:boundary],
                    ):
                        continue
                    bounds.append((cursor, boundary))
                    cursor = start + match.end()
                bounds.append((cursor, end))
            for left, right in bounds:
                spans.append(
                    Span(
                        len(spans),
                        left,
                        right,
                        notes[left:right],
                        tuple(hid for _, hid in headings),
                    )
                )
        index += 1
    return spans


def assemble(
    question: str,
    notes: str,
    spans: list[Span],
    scores: list[float],
    threshold: float,
    *,
    reason: str | None = None,
) -> PruneResult:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if len(scores) != len(spans) or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in scores
    ):
        raise ValueError("one finite probability per source span is required")
    kept = {
        s.id
        for s, score in zip(spans, scores)
        if not s.is_heading and score >= threshold
    }
    for span in spans:
        if span.id in kept:
            kept.update(span.heading_ids)
    if reason or not kept:
        kept = {s.id for s in spans}
        output = notes
        reason = reason or "no_selection_passthrough"
    elif len(kept) == len(spans):
        output = notes
        reason = "all_retained"
    else:
        output = "\n".join(s.text for s in spans if s.id in kept)
        reason = "selected"
    return PruneResult(
        question,
        output,
        [
            dict(asdict(s), score=float(p), kept=s.id in kept)
            for s, p in zip(spans, scores)
        ],
        sorted(kept),
        reason,
    )
