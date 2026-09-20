"""Import aligned gettext translations without enabling them for replacement."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polib

from .core import Entry


def import_po(path: str | Path, *, source: str, domain: str) -> list[Entry]:
    result: list[Entry] = []
    seen: set[str] = set()
    for item in polib.pofile(str(path)):
        # Plurals, fuzzy, obsolete, and untranslated entries are not trustworthy
        # one-to-one pairs. Preserve markup for subsequent explicit review.
        if item.obsolete or "fuzzy" in item.flags or item.msgid_plural:
            continue
        if not item.msgid.strip() or not item.msgstr.strip():
            continue
        digest = hashlib.sha256(
            f"{source}\0{item.msgctxt or ''}\0{item.msgid}\0{item.msgstr}".encode()
        ).hexdigest()[:24]
        if digest in seen:
            continue
        seen.add(digest)
        context = f"; msgctxt={item.msgctxt}" if item.msgctxt else ""
        result.append(Entry(digest, item.msgid, item.msgstr, domain, source + context, False))
    return result
