"""Experimental reversible source grammar; no fact selection or model extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .evo_grammar import digest
from .tokens import count_tokens


@dataclass(frozen=True)
class SourceRule:
    kind: str
    source: str
    emitted: str


# Structural verb phrases only. All remaining source characters stay literal.
RULES = (
    SourceRule('decision', ' decided to ', ' decided '),
    SourceRule('agreement', ' agreed to ', ' agreed '),
    SourceRule('proposal', ' proposed to ', ' proposed '),
    SourceRule('disagreement', ' disagreed with ', ' disagreed '),
)
GRAMMAR_ID = digest([vars(rule) for rule in RULES])
PROTECTED = re.compile(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”]*”|(?<!\w)\x27[^\x27\n]*\x27')


@dataclass(frozen=True)
class SourceEdit:
    start: int
    end: int
    kind: str
    source: str
    emitted: str


@dataclass(frozen=True)
class SourceEncoding:
    text: str
    route: str
    reason: str
    source_tokens: int
    encoded_tokens: int
    edits: tuple[SourceEdit, ...]
    grammar_id: str = GRAMMAR_ID


def decode_source(encoded: str) -> str:
    """Decode using grammar alone, never an edit trace or the original source.

    Applicable only to an encoded route. Raw fallback strings are not codewords.
    """
    for rule in RULES:
        encoded = encoded.replace(rule.emitted, rule.source)
    return encoded


def compile_source(notes: str) -> SourceEncoding:
    tokens = count_tokens(notes)
    def raw(reason):
        return SourceEncoding(notes,'raw',reason,tokens,tokens,())
    # Unclosed code/quote syntax is conservatively left unchanged.
    if notes.count('```') % 2 or notes.count('`') % 2 or notes.count('"') % 2:
        return raw('unclosed_literal')
    protected = [(m.start(),m.end()) for m in PROTECTED.finditer(notes)]
    edits = []
    for rule in RULES:
        for match in re.finditer(re.escape(rule.source),notes):
            if not any(match.start() < end and match.end() > start for start,end in protected):
                edits.append(SourceEdit(match.start(),match.end(),rule.kind,rule.source,rule.emitted))
    edits.sort(key=lambda e:e.start)
    if not edits:
        return raw('no_supported_structure')
    pieces, cursor = [], 0
    for edit in edits:
        if edit.start < cursor:
            return raw('overlapping_structure')
        pieces.extend((notes[cursor:edit.start],edit.emitted))
        cursor = edit.end
    pieces.append(notes[cursor:])
    encoded = ''.join(pieces)
    if decode_source(encoded) != notes:
        return raw('literal_marker_collision_or_integrity_failure')
    encoded_tokens = count_tokens(encoded)
    if encoded_tokens >= tokens:
        return raw('no_token_saving')
    return SourceEncoding(encoded,'source_grammar','reversible_structural_edit',tokens,encoded_tokens,tuple(edits))
