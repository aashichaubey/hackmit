"""Session-scoped compile-once support for the original Tokenese demo."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import MutableMapping

from . import MODEL, PROMPT_VERSION
from .facts import MeetingFacts, Usage
from .model import extract_facts
from .prompts import EXTRACTION_INSTRUCTION


@dataclass(frozen=True)
class CompilationResult:
    facts: MeetingFacts
    usage: Usage
    reused: bool


def compilation_key(notes: str, model: str = MODEL) -> str:
    identity = "\n".join((model, PROMPT_VERSION, EXTRACTION_INSTRUCTION, notes))
    return hashlib.sha256(identity.encode()).hexdigest()


def compile_once(
    state: MutableMapping[str, object],
    notes: str,
    client,
    model: str = MODEL,
) -> CompilationResult:
    """Extract facts once for unchanged notes and reuse them for follow-up questions."""
    key = compilation_key(notes, model)
    if state.get("v1_compilation_key") == key:
        facts = state.get("v1_facts")
        usage = state.get("v1_extraction_usage")
        if isinstance(facts, MeetingFacts) and isinstance(usage, Usage):
            return CompilationResult(facts=facts, usage=usage, reused=True)

    facts, usage = extract_facts(client, notes, model)
    state["v1_compilation_key"] = key
    state["v1_facts"] = facts
    state["v1_extraction_usage"] = usage
    return CompilationResult(facts=facts, usage=usage, reused=False)
