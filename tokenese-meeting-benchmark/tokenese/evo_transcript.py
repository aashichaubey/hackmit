"""Question-independent structural classification and reversible turn notation."""
from dataclasses import dataclass
import re

from .evo_grammar import digest
from .tokens import count_tokens

GRAMMAR = {'version':1,'speaker_marker':'@','speaker_id':'unchanged_digits',
           'utterance_policy':'all_characters_unchanged','inheritance':False}
GRAMMAR_ID = digest(GRAMMAR)
PATTERNS = {'meetingqa':r'(?m)^Speaker (\d+):|(?<=\n) Speaker (\d+):',
            'MeeQA':r'& SPEAKER_(\d+):'}


@dataclass(frozen=True)
class TranscriptEncoding:
    text: str
    profile: str
    source_tokens: int
    encoded_tokens: int
    turns: int
    reason: str
    grammar_id: str = GRAMMAR_ID


def decode_transcript(text: str, profile: str) -> str:
    if profile == 'raw':
        return text
    if profile == 'MeeQA-minimal':
        return re.sub(r'SPEAKER_(\d+):',lambda m:'& SPEAKER_'+m[1]+':',text)
    if profile == 'MeeQA-readable':
        return decode_readable_transcript(text)
    if profile not in PATTERNS:
        raise ValueError('Unknown transcript framing')
    replacement = (lambda m:'Speaker '+m[1]+':') if profile == 'meetingqa' else (lambda m:'& SPEAKER_'+m[1]+':')
    return re.sub(r'@(\d+):',replacement,text)


def compile_transcript(notes: str) -> TranscriptEncoding:
    size = count_tokens(notes)
    def raw(reason):
        return TranscriptEncoding(notes,'raw',size,size,0,reason)
    if re.search(r'@\d+:',notes):
        return raw('literal_marker_collision')
    profiles = [name for name,pattern in PATTERNS.items() if re.search(pattern,notes)]
    if len(profiles) != 1:
        return raw('unsupported_or_mixed_framing')
    profile = profiles[0]
    def replace(match):
        # Preserve the literal space before a MeetingQA speaker label.
        prefix = ' ' if match[0].startswith(' ') else ''
        identifier = next(g for g in match.groups() if g is not None)
        return prefix+'@'+identifier+':'
    encoded,turns = re.subn(PATTERNS[profile],replace,notes)
    if decode_transcript(encoded,profile) != notes:
        return raw('integrity_failure')
    tokens = count_tokens(encoded)
    if tokens >= size:
        return raw('not_cheaper')
    return TranscriptEncoding(encoded,profile,size,tokens,turns,'reversible_speaker_structure')


def compile_readable_transcript(notes: str) -> TranscriptEncoding:
    """Second candidate: keep the word Speaker; compact only verbose MeeQA framing."""
    original = compile_transcript(notes)
    if original.profile != 'MeeQA' or re.search(r'Speaker \d+:',notes):
        size = count_tokens(notes)
        return TranscriptEncoding(notes,'raw',size,size,0,'readable_profile_not_applicable',digest({'version':2,'marker':'Speaker '}))
    text = re.sub(PATTERNS['MeeQA'],lambda m:'Speaker '+m[1]+':',notes)
    size = count_tokens(text)
    if decode_readable_transcript(text) != notes or size >= original.source_tokens:
        return TranscriptEncoding(notes,'raw',original.source_tokens,original.source_tokens,0,'readable_integrity_or_cost',digest({'version':2,'marker':'Speaker '}))
    return TranscriptEncoding(text,'MeeQA-readable',original.source_tokens,size,original.turns,'reversible_readable_speaker_structure',digest({'version':2,'marker':'Speaker '}))


def decode_readable_transcript(text: str) -> str:
    return re.sub(r'Speaker (\d+):',lambda m:'& SPEAKER_'+m[1]+':',text)


def compile_minimal_transcript(notes: str) -> TranscriptEncoding:
    """Third candidate: remove only the decorative ampersand, keeping speaker IDs."""
    original = compile_transcript(notes)
    grammar_id = digest({'version':3,'remove':'& ','keep':'SPEAKER_N:'})
    if original.profile != 'MeeQA' or re.search(r'(?<!& )SPEAKER_\d+:',notes):
        size=count_tokens(notes)
        return TranscriptEncoding(notes,'raw',size,size,0,'minimal_profile_not_applicable',grammar_id)
    text=re.sub(PATTERNS['MeeQA'],lambda m:'SPEAKER_'+m[1]+':',notes)
    size=count_tokens(text)
    if decode_transcript(text,'MeeQA-minimal') != notes or size >= original.source_tokens:
        return TranscriptEncoding(notes,'raw',original.source_tokens,original.source_tokens,0,'minimal_integrity_or_cost',grammar_id)
    return TranscriptEncoding(text,'MeeQA-minimal',original.source_tokens,size,original.turns,'reversible_minimal_speaker_structure',grammar_id)
