"""Source-only quotation restoration shared by the workspace and product audit.

This policy never consults questions or gold answers. It preserves provider
metadata and applies identical answer-shape and grounding checks to every arm.
"""
from __future__ import annotations

from .evo_transcript import TranscriptEncoding, decode_transcript
from .evo_transcript_blocks import decode_blocks_quote


def restore_response(source: str, encoding: TranscriptEncoding, result: dict, method: str) -> dict:
    """Return a new response with source quotations, or an explicit policy error."""
    if method not in ('raw', 'minimal', 'blocks'):
        raise ValueError('Unknown transcript method')
    restored = dict(result)
    if result.get('error'):
        return restored
    answer = result.get('answer')
    valid = (isinstance(answer, dict) and type(answer.get('found')) is bool
             and isinstance(answer.get('spans'), list)
             and all(isinstance(span, str) and span.strip() for span in answer['spans'])
             and answer['found'] == bool(answer['spans']))
    if not valid:
        restored['error'] = 'Invalid quotation answer: found must agree with nonempty string spans.'
        return restored
    spans = []
    try:
        for span in answer['spans']:
            if span in source:
                # A model may already have expanded a structural speaker label.
                decoded = span
            elif method == 'blocks':
                decoded = decode_blocks_quote(encoding.text, encoding.profile, span)
            elif method == 'minimal':
                decoded = decode_transcript(span, encoding.profile)
            else:
                decoded = span
            if not decoded.strip() or decoded not in source:
                raise ValueError('quotation_not_in_source')
            spans.append(decoded)
    except ValueError as exc:
        restored['error'] = 'Quote restoration: ' + str(exc)
        return restored
    restored['answer'] = {**answer, 'spans': spans}
    return restored
