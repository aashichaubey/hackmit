"""Local-only screening of reversible consecutive-speaker inheritance.

This measures serialization, not model comprehension. No labels or questions are
used by the compiler, and no model calls or held-out data are read by the study.
"""
from __future__ import annotations

from collections import Counter
import json
import re

from .evo_grammar import digest
from .evo_public_data import REPORTS, load_training
from .evo_transcript import PATTERNS, TranscriptEncoding
from .tokens import count_tokens

CONTINUATION = '↳'
GRAMMAR = {'version': 1, 'policy': 'consecutive_speaker_inheritance',
           'continuation': CONTINUATION, 'first_and_changed_speakers': 'verbatim',
           'utterance_policy': 'all_characters_unchanged'}
GRAMMAR_ID = digest(GRAMMAR)


def decode_inheritance(text: str, profile: str) -> str:
    if profile == 'raw':
        return text
    if profile not in PATTERNS:
        raise ValueError('Unknown transcript framing')
    # The prefix space belongs to framing, but is kept outside the substituted
    # marker. Scan labels independently of line starts so ↳ is also recognized.
    header = r'Speaker \d+:' if profile == 'meetingqa' else r'& SPEAKER_\d+:'
    previous = None

    def expand(match):
        nonlocal previous
        if match[0] != CONTINUATION:
            previous = match[0]
            return match[0]
        if previous is None:
            raise ValueError('Continuation before a speaker')
        return previous

    return re.sub(header+'|'+re.escape(CONTINUATION), expand, text)


def compile_inheritance(notes: str) -> TranscriptEncoding:
    size = count_tokens(notes)

    def raw(reason):
        return TranscriptEncoding(notes, 'raw', size, size, 0, reason, GRAMMAR_ID)

    if CONTINUATION in notes:
        return raw('literal_continuation_collision')
    profiles = [name for name, pattern in PATTERNS.items() if re.search(pattern, notes)]
    if len(profiles) != 1:
        return raw('unsupported_or_mixed_framing')
    profile = profiles[0]
    previous = None

    def shrink(match):
        nonlocal previous
        header = match[0].lstrip(' ')
        prefix = match[0][ :len(match[0])-len(header)]
        replacement = CONTINUATION if previous == header else header
        previous = header
        return prefix+replacement

    text, turns = re.subn(PATTERNS[profile], shrink, notes)
    # A literal label inside an utterance can confuse the decoder's state. The
    # exact round trip rejects that case rather than interpreting it as a turn.
    if decode_inheritance(text, profile) != notes:
        return raw('integrity_failure')
    encoded_size = count_tokens(text)
    if encoded_size >= size:
        return raw('not_cheaper')
    return TranscriptEncoding(text, profile, size, encoded_size, turns,
                              'reversible_consecutive_speaker_inheritance', GRAMMAR_ID)


def decode_quote(text: str, profile: str, quote: str) -> str:
    """Ground an exact quote and expand inheritance using its full context.

    Raises ValueError for absent or ambiguous quotes.
    Every occurrence must yield the same source string; references are never
    consulted. A body-only quote can safely repeat across different speakers.
    """
    if profile != 'raw' and profile not in PATTERNS:
        raise ValueError('unsupported_quote_profile')
    if not quote:
        raise ValueError('empty_quote')
    occurrences = []
    offset = text.find(quote)
    while offset >= 0:
        occurrences.append((offset, offset+len(quote)))
        offset = text.find(quote, offset+1)
    if not occurrences:
        raise ValueError('ungrounded_quote')
    if profile == 'raw':
        return quote

    header = r'Speaker \d+:' if profile == 'meetingqa' else r'& SPEAKER_\d+:'
    matches = list(re.finditer(header+'|'+re.escape(CONTINUATION), text))
    # Mapping stores original-source offsets for every encoded boundary. An
    # inherited marker is atomic: its end advances by the full original label.
    boundaries = [0]
    pieces = []
    previous = None
    cursor = source_offset = 0
    for match in matches:
        untouched = text[cursor:match.start()]
        pieces.append(untouched)
        boundaries.extend(range(source_offset+1, source_offset+len(untouched)+1))
        source_offset += len(untouched)
        if match[0] == CONTINUATION:
            if previous is None:
                raise ValueError('continuation_without_speaker')
            replacement = previous
            boundaries.append(source_offset+len(replacement))
        else:
            previous = replacement = match[0]
            boundaries.extend(range(source_offset+1, source_offset+len(replacement)+1))
        pieces.append(replacement)
        source_offset += len(replacement)
        cursor = match.end()
    suffix = text[cursor:]
    pieces.append(suffix)
    boundaries.extend(range(source_offset+1, source_offset+len(suffix)+1))
    source = ''.join(pieces)
    decoded = set()
    for start, end in occurrences:
        decoded.add(source[boundaries[start]:boundaries[end]])
    if len(decoded) != 1:
        raise ValueError('ambiguous_inherited_quote')
    return decoded.pop()


def screen_training() -> dict:
    datasets, _, sources = load_training()
    report = {'study': 'local_only_consecutive_speaker_screen', 'grammar': GRAMMAR,
              'grammar_id': GRAMMAR_ID, 'sources': sources, 'splits': ['train'],
              'model_calls': 0, 'qualified': False,
              'context_scope': 'Unique contexts retained by the published-label adapter; invalid offsets and answerability-disagreement groups are excluded upstream. No model-outcome filtering.',
              'limitation': 'Serialization savings only. Continuation comprehension is untested; no quality improvement claim.',
              'datasets': {}}
    for dataset in ('meetingqa', 'MeeQA'):
        # Count each released context once, not once per annotation/question.
        contexts = sorted({case['context'] for case in datasets[dataset]})
        source_tokens = encoded_tokens = changed = failures = 0
        reasons = Counter()
        for context in contexts:
            encoded = compile_inheritance(context)
            source_tokens += encoded.source_tokens
            encoded_tokens += encoded.encoded_tokens
            changed += encoded.profile != 'raw'
            failures += decode_inheritance(encoded.text, encoded.profile) != context
            reasons[encoded.reason] += 1
        report['datasets'][dataset] = {
            'unique_contexts': len(contexts), 'changed_contexts': changed,
            'source_tokens': source_tokens, 'encoded_tokens': encoded_tokens,
            'saved_tokens': source_tokens-encoded_tokens,
            'saved_fraction': 1-encoded_tokens/source_tokens if source_tokens else 0,
            'round_trip_failures': failures, 'reason_counts': dict(reasons)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS/'local-search.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(screen_training(), indent=2))
