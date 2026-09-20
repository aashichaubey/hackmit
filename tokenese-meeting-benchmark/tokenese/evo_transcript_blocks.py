"""Unqualified local grammar: consecutive speaker turns as indented list items.

All utterance characters survive. Syntax recognition is not semantic speaker
verification, and exact reversibility is not evidence of model comprehension.
"""
from __future__ import annotations

from collections import Counter
import json
import re

from .evo_grammar import digest
from .evo_public_data import REPORTS, load_training
from .evo_transcript import TranscriptEncoding
from .tokens import count_tokens

BULLET = '  -'
NAMED = r"(?m)^[A-Z][A-Za-z'-]{0,30}(?: [A-Z][A-Za-z'-]{0,30}){0,2}:"
PATTERNS = {'MeeQA': r'& SPEAKER_\d+:',
            'meetingqa': r'(?m)^Speaker \d+:|(?<=\n) Speaker \d+:',
            'named': NAMED}
GRAMMAR = {'version': 1, 'style': 'consecutive_speaker_indented_list',
           'speaker_headers': 'verbatim', 'continuation': BULLET,
           'utterances': 'all_characters_unchanged', 'legend': None}
GRAMMAR_ID = digest(GRAMMAR)


def decode_blocks(text: str, profile: str) -> str:
    if profile == 'raw':
        return text
    if profile not in PATTERNS:
        raise ValueError('Unsupported block profile')
    previous = None
    if profile == 'MeeQA':
        pattern = r'\n  -|\n& SPEAKER_\d+:|& SPEAKER_\d+:'
    elif profile == 'meetingqa':
        pattern = r'Speaker \d+:|  -'
    else:
        pattern = NAMED+r'|  -'

    def expand(match):
        nonlocal previous
        value = match[0]
        if value in (BULLET, '\n'+BULLET):
            if previous is None:
                raise ValueError('Continuation without speaker')
            return previous
        if profile == 'MeeQA':
            value = value.removeprefix('\n')
        previous = value
        return value

    return re.sub(pattern, expand, text)


def compile_blocks(notes: str) -> TranscriptEncoding:
    size = count_tokens(notes)

    def raw(reason):
        return TranscriptEncoding(notes, 'raw', size, size, 0, reason, GRAMMAR_ID)

    if BULLET in notes:
        return raw('literal_list_marker_collision')
    matches = {name: list(re.finditer(pattern, notes)) for name, pattern in PATTERNS.items()}
    dataset_profiles = [name for name in ('MeeQA', 'meetingqa') if matches[name]]
    if len(dataset_profiles) > 1:
        return raw('mixed_framing')
    profile = dataset_profiles[0] if dataset_profiles else 'named'
    turns = matches[profile]
    if not turns or turns[0].start() != 0:
        return raw('unsupported_framing_or_preface')
    # General named-speaker transcripts must actually alternate identifiable
    # names. This conservative syntax guard cannot establish semantic identity.
    if profile == 'named' and (len(turns) < 3 or len({m[0] for m in turns}) < 2):
        return raw('insufficient_named_turn_structure')
    previous = None
    first = True

    def shrink(match):
        nonlocal previous, first
        header = match[0].lstrip(' ')
        prefix = match[0][:len(match[0])-len(header)]
        if previous == header:
            result = ('\n' if profile == 'MeeQA' else prefix)+BULLET
        else:
            result = ('\n' if profile == 'MeeQA' and not first else prefix)+header
        previous = header
        first = False
        return result

    text = re.sub(PATTERNS[profile], shrink, notes)
    if decode_blocks(text, profile) != notes:
        return raw('integrity_failure')
    tokens = count_tokens(text)
    if tokens >= size:
        return raw('not_cheaper')
    return TranscriptEncoding(text, profile, size, tokens, len(turns),
                              'reversible_indented_speaker_blocks', GRAMMAR_ID)


def decode_blocks_quote(text: str, profile: str, quote: str) -> str:
    """Ground every occurrence using full speaker context, without references."""
    if profile != 'raw' and profile not in PATTERNS:
        raise ValueError('unsupported_quote_profile')
    if not quote:
        raise ValueError('empty_quote')
    occurrences = []
    start = text.find(quote)
    while start >= 0:
        occurrences.append((start, start+len(quote)))
        start = text.find(quote, start+1)
    if not occurrences:
        raise ValueError('ungrounded_quote')
    if profile == 'raw':
        return quote
    pattern = (r'\n  -|\n& SPEAKER_\d+:|& SPEAKER_\d+:' if profile == 'MeeQA'
               else r'Speaker \d+:|  -' if profile == 'meetingqa' else NAMED+r'|  -')
    boundaries, pieces = [0], []
    cursor = offset = 0
    previous = None
    for match in re.finditer(pattern, text):
        untouched = text[cursor:match.start()]
        pieces.append(untouched)
        boundaries.extend(range(offset+1, offset+len(untouched)+1))
        offset += len(untouched)
        value = match[0]
        if value in (BULLET, '\n'+BULLET):
            if previous is None:
                raise ValueError('continuation_without_speaker')
            replacement = previous
            if value.startswith('\n'):
                boundaries.append(offset)
            # Unlike unchanged labels, synthetic bullet interiors have no
            # character-for-character correspondence to the original speaker.
            boundaries.extend([None]*(len(BULLET)-1))
            boundaries.append(offset+len(replacement))
        else:
            replacement = value.removeprefix('\n') if profile == 'MeeQA' else value
            if len(replacement) != len(value):
                boundaries.append(offset)
            boundaries.extend(range(offset+1, offset+len(replacement)+1))
            previous = replacement
        pieces.append(replacement)
        offset += len(replacement)
        cursor = match.end()
    suffix = text[cursor:]
    pieces.append(suffix)
    boundaries.extend(range(offset+1, offset+len(suffix)+1))
    source = ''.join(pieces)
    decoded = set()
    for start, end in occurrences:
        if boundaries[start] is None or boundaries[end] is None:
            raise ValueError('partial_synthetic_marker_quote')
        decoded.add(source[boundaries[start]:boundaries[end]])
    if len(decoded) != 1:
        raise ValueError('ambiguous_inherited_quote')
    return decoded.pop()


def screen_training() -> dict:
    datasets, _, sources = load_training()
    result = {'study': 'local_only_speaker_blocks', 'grammar': GRAMMAR,
              'grammar_id': GRAMMAR_ID, 'sources': sources, 'splits': ['train'],
              'qualified': False, 'model_calls': 0,
              'scope': 'All unique retained training contexts; adapter exclusions apply. No label, question, or model-outcome selection.',
              'limitation': 'Structural token savings only; comprehension untested. Named-speaker support has synthetic integrity tests only.',
              'datasets': {}}
    for name in ('meetingqa', 'MeeQA'):
        contexts = {c['context'] for c in datasets[name]}
        counters = Counter()
        reasons = Counter()
        for context in contexts:
            encoded = compile_blocks(context)
            counters.update(source_tokens=encoded.source_tokens, encoded_tokens=encoded.encoded_tokens,
                            changed_contexts=int(encoded.profile != 'raw'),
                            round_trip_failures=int(decode_blocks(encoded.text, encoded.profile) != context))
            reasons[encoded.reason] += 1
        result['datasets'][name] = {**dict(counters), 'unique_contexts': len(contexts),
            'saved_tokens': counters['source_tokens']-counters['encoded_tokens'],
            'saved_fraction': 1-counters['encoded_tokens']/counters['source_tokens'],
            'reason_counts': dict(reasons)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS/'local-blocks.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(screen_training(), indent=2))
