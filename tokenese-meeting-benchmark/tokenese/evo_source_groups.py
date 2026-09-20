"""Experimental consecutive actor inheritance over complete source paragraphs."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .evo_grammar import digest
from .tokens import count_tokens


CONTRACT = {'version':1,'source_separator':' - ','group':'{actor:clause;clause}',
            'minimum_run':3,'maximum_actor_words':8,'literal_policy':'preserve_all'}
GRAMMAR_ID = digest(CONTRACT)


@dataclass(frozen=True)
class GroupEncoding:
    text: str
    route: str
    reason: str
    source_tokens: int
    encoded_tokens: int
    groups: int
    grammar_id: str = GRAMMAR_ID


def actor_prefix(clause: str) -> str:
    words = []
    for word in clause.split(' '):
        if len(words) == 8 or not word or not word[0].isupper() or not all(c.isalpha() or c in "'-" for c in word):
            break
        words.append(word)
    prefix = ' '.join(words) + ' ' if words else ''
    return prefix if prefix and clause.startswith(prefix) and len(prefix) < len(clause) else ''


def decode_groups(text: str) -> str:
    """Recover original source using the group grammar alone."""
    def expand(match):
        prefix, content = match.group(1),match.group(2)
        return ' - '.join(prefix+' '+body for body in content.split(';'))
    return re.sub(r'\{([^{}:;\n]+):([^{}\n]+)\}',expand,text)


def compile_groups(notes: str) -> GroupEncoding:
    original_tokens = count_tokens(notes)
    def raw(reason):
        return GroupEncoding(notes,'raw',reason,original_tokens,original_tokens,0)
    if any(char in notes for char in '{}'):
        return raw('literal_marker_collision')
    if notes.count('```') % 2:
        return raw('unclosed_code_block')
    code = [(m.start(),m.end()) for m in re.finditer(r'```[\s\S]*?```',notes)]
    output, groups = [], 0
    offset = 0
    for line in notes.splitlines(keepends=True):
        start,offset = offset,offset+len(line)
        # Work only on explicit inline bullet sequences, never reflow paragraphs.
        if not line.startswith('- ') or ';' in line or any(start < b and offset > a for a,b in code):
            output.append(line)
            continue
        ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        text = line[:-len(ending)] if ending else line
        clauses = text[2:].split(' - ')
        revised, index = [], 0
        while index < len(clauses):
            prefix = actor_prefix(clauses[index])
            end = index+1
            if prefix:
                while end < len(clauses) and actor_prefix(clauses[end]) == prefix:
                    end += 1
            if prefix and end-index >= 3:
                original = ' - '.join(clauses[index:end])
                candidate = '{'+prefix[:-1]+':'+';'.join(c[len(prefix):] for c in clauses[index:end])+'}'
                if count_tokens(candidate) < count_tokens(original):
                    revised.append(candidate)
                    groups += 1
                    index = end
                    continue
            revised.append(clauses[index])
            index += 1
        output.append('- '+' - '.join(revised)+ending)
    encoded = ''.join(output)
    if decode_groups(encoded) != notes:
        return raw('integrity_failure')
    encoded_tokens = count_tokens(encoded)
    if encoded_tokens >= original_tokens:
        return raw('no_token_saving')
    return GroupEncoding(encoded,'source_groups','consecutive_actor_inheritance',original_tokens,encoded_tokens,groups)
