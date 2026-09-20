"""Small, data-only languages. Round trips prove structure, not LLM understanding."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .facts import MeetingFacts

KINDS = ("action", "proposal", "decision", "agreement", "disagreement")
VOCAB = {
    "action": ("action", "do", "→", "!"),
    "proposal": ("proposal", "proposed", "?", "~", "proposal (unconfirmed)"),
    "decision": ("decision", "decided", "=", "✓"),
    "agreement": ("agreement", "agreed", "+", "yes"),
    "disagreement": ("disagreement", "disagreed", "−", "no"),
}
OWNER_MARKERS = ("", "owner:", "by:")
DUE_MARKERS = ("@", "due:", "by:")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GrammarSpec:
    kind_forms: tuple[str, ...] = ("action", "proposal", "decision", "agreement", "disagreement")
    layout: Literal["actor", "kind", "content", "prose"] = "kind"
    owner_marker: str = ""
    deadline_marker: str = "@"
    separator: str = ":"
    grouping: bool = False
    repairs: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "kind_forms", tuple(self.kind_forms))
        object.__setattr__(self, "repairs", tuple(self.repairs))
        if len(self.kind_forms) != 5 or any(v not in VOCAB[k] for k, v in zip(KINDS, self.kind_forms)):
            raise ValueError("Unbounded or invalid structural vocabulary")
        if len(set(self.kind_forms)) != 5:
            raise ValueError("Kind markers must be distinct")
        if self.layout not in ("actor", "kind", "content", "prose") or self.separator not in (":", "|", ";", "\t"):
            raise ValueError("Unsupported layout or separator")
        if self.owner_marker not in OWNER_MARKERS or self.deadline_marker not in DUE_MARKERS:
            raise ValueError("Invalid field marker")
        allowed = {"owner", "deadline", "boundaries", *KINDS}
        if len(self.repairs) > 4 or len(set(self.repairs)) != len(self.repairs) or any(r not in allowed for r in self.repairs):
            raise ValueError("Repairs must be bounded structural rules")
        if self.layout == "prose" and self.grouping:
            raise ValueError("Prose has explicit per-fact fields")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def id(self) -> str:
        return digest(self.to_dict())[:16]

    def edit(self, **changes) -> GrammarSpec:
        return replace(self, **changes)


@dataclass(frozen=True)
class SemanticFact:
    kind: str
    person: str
    text: str
    deadline: str


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    fact_index: int
    field: str
    source_quote: str


@dataclass(frozen=True)
class CompiledText:
    text: str
    trace: tuple[Segment, ...]
    grammar_id: str


def projection(facts: MeetingFacts) -> list[SemanticFact]:
    return [SemanticFact(f.kind, f.person, f.text, f.deadline) for f in facts.facts]


def _escape(value: str, separator: str) -> str:
    # Escape every grammar boundary. Literal words and dates are never rewritten.
    return "".join("\\" + c if c in {"\\", separator, "[", "]"} else c for c in value)


def _unescape(value: str, separator: str) -> str:
    out, i = [], 0
    while i < len(value):
        c = value[i]
        if c == "\\":
            i += 1
            if i == len(value) or value[i] not in {"\\", separator, "[", "]"}:
                raise ValueError("Invalid escape")
            c = value[i]
        out.append(c)
        i += 1
    return "".join(out)


def _split(value: str, separator: str) -> list[str]:
    out, start, escaped = [], 0, False
    for i, c in enumerate(value):
        if escaped:
            escaped = False
        elif c == "\\":
            escaped = True
        elif c == separator:
            out.append(value[start:i])
            start = i + 1
    if escaped:
        raise ValueError("Dangling escape")
    out.append(value[start:])
    return out


def _ordered(grammar: GrammarSpec) -> tuple[str, ...]:
    return {"actor": ("person", "kind", "text"), "kind": ("kind", "person", "text"),
            "content": ("kind", "text", "person"), "prose": ("kind", "person", "text", "deadline")}[grammar.layout]


def compile_facts(facts: MeetingFacts, grammar: GrammarSpec) -> CompiledText:
    chunks, trace, offset = [], [], 0
    previous = None
    for index, fact in enumerate(facts.facts):
        sep = grammar.separator
        group = grammar.grouping and previous == (fact.kind, fact.person)
        fields = ("text",) if group else _ordered(grammar)
        pieces = []
        prefix = "]" if group else ""
        if prefix:
            pieces.append(prefix)
        for field in fields:
            if len(pieces) > (1 if group else 0):
                pieces.append(sep)
            value = grammar.kind_forms[KINDS.index(fact.kind)] if field == "kind" else getattr(fact, field)
            marker = ""
            if grammar.layout == "prose":
                marker = {"kind": "Type: ", "person": " Person: ", "text": " Content: ", "deadline": " Deadline: "}[field]
            elif field == "person":
                marker = grammar.owner_marker
            pieces.append(_escape(marker, sep))
            start = offset + sum(map(len, pieces))
            encoded = _escape(value, sep)
            pieces.append(encoded)
            trace.append(Segment(start, start + len(encoded), index, field, fact.source_quote))
        if grammar.layout != "prose" and fact.deadline:
            # A separate structural field, never inferred from task wording.
            pieces.extend([sep, _escape(grammar.deadline_marker, sep)])
            start = offset + sum(map(len, pieces))
            value = _escape(fact.deadline, sep)
            pieces.append(value)
            trace.append(Segment(start, start + len(value), index, "deadline", fact.source_quote))
        line = "".join(pieces)
        chunks.append(line)
        offset += len(line) + 1
        previous = (fact.kind, fact.person)
    return CompiledText("\n".join(chunks), tuple(trace), grammar.id)


def parse_compiled(text: str, grammar: GrammarSpec) -> list[SemanticFact]:
    if not text:
        return []
    out = []
    for line in text.split("\n"):
        group = line.startswith("]")
        if group and (not grammar.grouping or not out):
            raise ValueError("Orphan inherited record")
        parts = _split(line[1:] if group else line, grammar.separator)
        fields = ("text",) if group else _ordered(grammar)
        if len(parts) not in ({len(fields)} if grammar.layout == "prose" else {len(fields), len(fields) + 1}):
            raise ValueError("Wrong number of fields")
        values = {"deadline": ""}
        if group:
            values.update(kind=out[-1].kind, person=out[-1].person)
        for field, value in zip(fields, parts):
            marker = ({"kind": "Type: ", "person": " Person: ", "text": " Content: ", "deadline": " Deadline: "}[field]
                      if grammar.layout == "prose" else grammar.owner_marker if field == "person" else "")
            marker = _escape(marker, grammar.separator)
            if not value.startswith(marker):
                raise ValueError("Missing field marker")
            value = _unescape(value[len(marker):], grammar.separator)
            if field == "kind":
                if value not in grammar.kind_forms:
                    raise ValueError("Unknown kind")
                value = KINDS[grammar.kind_forms.index(value)]
            values[field] = value
        if len(parts) > len(fields):
            due_marker = _escape(grammar.deadline_marker, grammar.separator)
            if not parts[-1].startswith(due_marker):
                raise ValueError("Missing deadline marker")
            values["deadline"] = _unescape(parts[-1][len(due_marker):], grammar.separator)
            if not values["deadline"]:
                raise ValueError("Noncanonical empty deadline")
        if not values["text"] or any("\r" in v for v in values.values()):
            raise ValueError("Invalid literal")
        out.append(SemanticFact(**values))
    return out


def integrity_errors(facts: MeetingFacts, compiled: CompiledText, grammar: GrammarSpec) -> list[str]:
    try:
        if compiled.grammar_id != grammar.id:
            return ["Grammar hash mismatch"]
        if parse_compiled(compiled.text, grammar) != projection(facts):
            return ["Semantic projection changed (including order or multiplicity)"]
    except ValueError as exc:
        return [str(exc)]
    return []
