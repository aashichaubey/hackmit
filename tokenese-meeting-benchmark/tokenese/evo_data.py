"""Auditable split loading. Search cannot load a held-out file accidentally."""
from __future__ import annotations

import json
from pathlib import Path

from .facts import MeetingFacts
from .evo_grammar import digest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "evolution"
REPORTS = ROOT / "reports" / "evolution"


def load_development(path: str | Path = DATA / "dev.json") -> list[dict]:
    cases = load_cases(path, "dev")
    counterexamples = Path(path).parent / "counterexamples.json"
    if counterexamples.exists():
        from .evo_contrasts import contrast_cases
        payload = json.loads(counterexamples.read_text())
        if payload["split"] != "dev" or len(payload["pairs"]) > 20:
            raise ValueError("Development contrast budget exceeded")
        cases += contrast_cases(payload["pairs"])
    return cases


def load_cases(path: str | Path, expected_split: str) -> list[dict]:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("split") != expected_split or expected_split not in ("dev", "validation", "test"):
        raise ValueError("Split access mismatch")
    cases = payload["cases"]
    seen = set()
    for case in cases:
        if case["id"] in seen or case["split"] != expected_split:
            raise ValueError("Duplicate case or mixed split")
        seen.add(case["id"])
        notes = case["notes"]
        provenance = case["provenance"]
        if not all(provenance.get(k) for k in ("source_id", "series_id", "url", "annotation_status")):
            raise ValueError("Missing source or review provenance")
        facts = MeetingFacts(facts=case["facts"])
        if any(f.source_quote not in notes for f in facts.facts):
            raise ValueError("Broken fact source quote")
        for q in case["questions"]:
            if q["id"] in seen or not q["answers"] or type(q["expected_found"]) is not bool:
                raise ValueError("Duplicate question or missing expected answer")
            seen.add(q["id"])
            if (not q["expected_found"]) != (q["answers"] == ["not found"]):
                raise ValueError("Inconsistent found label")
            if not q.get("category") or not q.get("evidence"):
                raise ValueError("Missing category or evidence")
            for span in q["evidence"]:
                if not (0 <= span["start"] < span["end"] <= len(notes)) or notes[span["start"]:span["end"]] != span["quote"]:
                    raise ValueError("Broken question evidence span")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["splits"][expected_split]
        if entry["sha256"] != digest(payload):
            raise ValueError("Dataset hash changed")
        sources, series = set(), set()
        for split_entry in manifest["splits"].values():
            current = set(split_entry["source_ids"])
            groups = set(split_entry["series_ids"])
            if sources & current or series & groups:
                raise ValueError("Source/series overlap across splits")
            sources |= current
            series |= groups
        if set(entry["source_ids"]) != {c["provenance"]["source_id"] for c in cases} or set(entry["series_ids"]) != {c["provenance"]["series_id"] for c in cases}:
            raise ValueError("Provenance does not match manifest")
    return cases


def write_manifest(directory: Path = DATA) -> dict:
    result = {"version": 1, "splits": {}}
    for split in ("dev", "validation", "test"):
        payload = json.loads((directory / f"{split}.json").read_text())
        result["splits"][split] = {"sha256": digest(payload), "cases": len(payload["cases"]),
            "questions": sum(len(c["questions"]) for c in payload["cases"]),
            "source_ids": sorted({c["provenance"]["source_id"] for c in payload["cases"]}),
            "series_ids": sorted({c["provenance"]["series_id"] for c in payload["cases"]})}
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_contrast_pair(pair: dict) -> None:
    if pair.get("split") != "dev" or pair.get("review_status") != "deterministic_checked":
        raise ValueError("Only reviewed deterministic development contrasts")
    before, after = pair["before"], pair["after"]
    field = pair["field"]
    if field not in ("kind", "person", "text", "deadline"):
        raise ValueError("Unsupported distinction")
    left = MeetingFacts(facts=before["facts"])
    right = MeetingFacts(facts=after["facts"])
    if len(left.facts) != len(right.facts):
        raise ValueError("Contrast changes fact multiplicity")
    changes = []
    for i, (a, b) in enumerate(zip(left.facts, right.facts)):
        for key in ("kind", "person", "text", "deadline"):
            if getattr(a, key) != getattr(b, key):
                changes.append((i, key))
        if a.source_quote not in before["notes"] or b.source_quote not in after["notes"]:
            raise ValueError("Contrast source edit is inconsistent")
    if len(changes) != 1 or changes[0][1] != field:
        raise ValueError("Contrast must change exactly one semantic field")
    if before["question"] != after["question"] or before["answers"] == after["answers"]:
        raise ValueError("Contrast must preserve question and change gold answer")
