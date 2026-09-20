"""Corpus loaders.

Datasets are registered, not hardcoded, so experiments can add domains without
touching pipeline code. Every loader yields raw `(english, mandarin)` pairs and
leaves normalization/filtering to the caller.

All loaders use only the standard library for network access so that the basic
pipeline never requires the heavy `datasets` stack.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_UA = {"User-Agent": "token-language/0.1 (research)"}
_CACHE = Path("data/raw")


@dataclass(frozen=True)
class CorpusSpec:
    """Declarative description of a parallel corpus."""

    name: str
    domain: str
    license: str
    loader: str  # "opus_moses" | "hf_rows"
    #: loader-specific location (OPUS zip URL, or HF "dataset:config:split")
    location: str
    #: Corpora known to ship pre-tokenized text needing space repair.
    pretokenized: bool = False
    #: Corpora that ship Traditional Chinese.
    traditional: bool = False
    notes: str = ""


# Verified reachable and license-checked. Sizes are the compressed download.
REGISTRY: dict[str, CorpusSpec] = {
    "wmt_news": CorpusSpec(
        name="wmt_news",
        domain="news",
        license="OPUS redistribution of WMT test sets (free for research)",
        loader="opus_moses",
        location="https://object.pouta.csc.fi/OPUS-WMT-News/v2019/moses/en-zh.txt.zip",
        notes="2.2MB / ~20k pairs. Professional human translation, cleanest source found.",
    ),
    "opus100": CorpusSpec(
        name="opus100",
        domain="general",
        license="OPUS-derived; 'unknown' tag on the HF hub",
        loader="hf_rows",
        location="Helsinki-NLP/opus-100:en-zh:test",
        notes="2000-row held-out split, fetched over REST with no download.",
    ),
    "kde4": CorpusSpec(
        name="kde4",
        domain="technical",
        license="same as original KDE sources (GPL/LGPL docs)",
        loader="opus_moses",
        location="https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-zh_CN.txt.zip",
        pretokenized=True,
        notes="3MB / ~140k pairs. Software UI strings. Heavy filtering required.",
    ),
    "tatoeba": CorpusSpec(
        name="tatoeba",
        domain="conversational",
        license="CC-BY 2.0 FR",
        loader="opus_moses",
        location="https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/cmn-en.txt.zip",
        notes="1.4MB / ~50k pairs. Short conversational sentences. Note 'cmn' code.",
    ),
    "bible": CorpusSpec(
        name="bible",
        domain="books",
        license="CC0 1.0",
        loader="opus_moses",
        location="https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-zh.txt.zip",
        notes="Verse-aligned, genuinely sentence-level. Archaic register.",
    ),
    "tldr": CorpusSpec(
        name="tldr",
        domain="technical",
        license="CC-BY (tldr-pages project)",
        loader="opus_moses",
        location="https://object.pouta.csc.fi/OPUS-tldr-pages/v2026-07-07/moses/en-zh.txt.zip",
        traditional=True,
        notes="0.5MB / ~17k pairs. CLI docs. Ships TRADITIONAL Chinese.",
    ),
}

# Deliberately excluded, with reasons, so the choice is auditable:
#   OpenSubtitles - OPUS lists no license, only a takedown notice. The underlying
#                   subtitles are user-uploaded transcripts of copyrighted films.
#   TED2020       - CC-BY-NC-ND: the no-derivatives clause conflicts with
#                   publishing a derived dictionary.
#   WikiMatrix    - LASER-mined and visibly misaligned on inspection.
#   FLORES        - gated on the HF hub; needs an authenticated account.
EXCLUDED: dict[str, str] = {
    "opensubtitles": "no license grant, only a takedown notice; copyrighted source material",
    "ted2020": "CC-BY-NC-ND - no-derivatives blocks a published dictionary",
    "wikimatrix": "machine-mined alignment, verified misaligned rows",
    "flores": "gated on HuggingFace, requires authenticated acceptance",
}


def _fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _cached_download(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once; reuse thereafter."""
    if dest.exists():
        log.info("using cached %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(_fetch(url))
    tmp.replace(dest)
    return dest


def load_opus_moses(
    spec: CorpusSpec, limit: int | None = None, cache_dir: Path = _CACHE
) -> Iterator[tuple[str, str]]:
    """Read an OPUS 'moses' zip: two line-parallel plaintext files.

    Also extracts the bundled README/LICENSE so provenance is recorded from the
    archive itself rather than trusted from a table.
    """
    dest = cache_dir / "opus" / f"{spec.name}.zip"
    _cached_download(spec.location, dest)

    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
        _save_provenance(zf, names, cache_dir / "opus" / f"{spec.name}.provenance.txt")

        # Side files end in the language code, e.g. ".en" / ".zh" / ".zh_CN" / ".cmn".
        en_name = _pick_side(names, ("en",))
        zh_name = _pick_side(names, ("zh_CN", "zh", "cmn", "zh_TW"))
        if not en_name or not zh_name:
            raise ValueError(f"{spec.name}: could not identify side files in {names}")

        with zf.open(en_name) as ef, zf.open(zh_name) as zfh:
            en_lines = io.TextIOWrapper(ef, encoding="utf-8", errors="replace")
            zh_lines = io.TextIOWrapper(zfh, encoding="utf-8", errors="replace")
            for i, (en, zh) in enumerate(zip(en_lines, zh_lines)):
                if limit is not None and i >= limit:
                    break
                yield en.rstrip("\n"), zh.rstrip("\n")


def _pick_side(names: list[str], suffixes: tuple[str, ...]) -> str | None:
    """Pick the data file whose extension matches one of `suffixes`.

    Ordered by preference, so zh_CN beats zh_TW when both exist.
    """
    for suf in suffixes:
        for n in names:
            if n.endswith(f".{suf}") and not n.endswith((".zip", ".xml")):
                return n
    return None


def _save_provenance(zf: zipfile.ZipFile, names: list[str], dest: Path) -> None:
    """Persist the README/LICENSE bundled in the OPUS archive."""
    parts = []
    for n in names:
        if Path(n).name.upper() in {"README", "LICENSE", "README.TXT", "LICENSE.TXT"}:
            try:
                parts.append(f"===== {n} =====\n{zf.read(n).decode('utf-8', 'replace')}")
            except KeyError:  # pragma: no cover
                continue
    if parts:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n\n".join(parts), encoding="utf-8")


def load_hf_rows(
    spec: CorpusSpec, limit: int | None = None, cache_dir: Path = _CACHE
) -> Iterator[tuple[str, str]]:
    """Stream rows from the HuggingFace datasets-server REST API.

    Avoids the `datasets` dependency entirely. The endpoint caps `length` at
    100 rows per request, so this paginates on `offset`.
    """
    dataset, config, split = spec.location.split(":")
    cache = cache_dir / "hf" / f"{spec.name}.jsonl"

    # Only trust the cache if it holds at least as many rows as requested.
    # A cache written by an earlier small-`limit` run would otherwise silently
    # cap every later run to that smaller size.
    if cache.exists():
        with cache.open(encoding="utf-8") as fh:
            cached = [json.loads(line) for line in fh if line.strip()]
        if limit is None or len(cached) >= limit:
            for row in cached[:limit] if limit else cached:
                yield row["en"], row["zh"]
            return
        log.info(
            "cache for %s holds %d rows but %d requested; refetching",
            spec.name,
            len(cached),
            limit,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    target = limit if limit is not None else 2000
    collected: list[dict[str, str]] = []
    offset, page = 0, 100

    while len(collected) < target:
        q = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(page, target - len(collected)),
            }
        )
        payload = json.loads(_fetch(f"https://datasets-server.huggingface.co/rows?{q}"))
        rows = payload.get("rows", [])
        if not rows:
            break  # split exhausted
        for r in rows:
            tr = r["row"]["translation"]
            collected.append({"en": tr["en"], "zh": tr["zh"]})
        offset += len(rows)

    with cache.open("w", encoding="utf-8") as fh:
        for row in collected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in collected:
        yield row["en"], row["zh"]


_LOADERS = {"opus_moses": load_opus_moses, "hf_rows": load_hf_rows}


def load_corpus(
    name: str, limit: int | None = None, cache_dir: Path = _CACHE
) -> Iterator[tuple[str, str]]:
    """Load a registered corpus by name."""
    if name in EXCLUDED:
        raise ValueError(f"corpus {name!r} is excluded: {EXCLUDED[name]}")
    if name not in REGISTRY:
        raise KeyError(f"unknown corpus {name!r}; registered: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    return _LOADERS[spec.loader](spec, limit=limit, cache_dir=cache_dir)
