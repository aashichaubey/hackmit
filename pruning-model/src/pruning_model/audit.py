"""Versioned corrections to model grading, preserving the original paid outputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from tokenmix.evaluation.storage import (
    digest,
    file_hash,
    read_jsonl,
    strict_json,
    write_json,
)

from .benchmark import score_verdict, write_reports


def grade_fingerprint(row: dict, arm: str) -> str:
    return digest(
        {
            "question": row["question"],
            "source": row["original_notes"],
            "answer": row[f"{arm}_answer"],
            "grade": row[f"{arm}_grade"],
        }
    )


def audit_grades(source: Path, dataset: Path, review_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "audited output must be new/empty; raw results are never overwritten"
        )
    manifest = strict_json((source / "manifest.json").read_text())
    rows = deepcopy(read_jsonl(source / "per_call.jsonl"))
    review = strict_json(review_path.read_text())
    if review["input_sha256"] != file_hash(source / "per_call.jsonl"):
        raise ValueError("grade review is stale")
    if file_hash(dataset) != manifest["dataset_sha256"]:
        raise ValueError("grade review needs the original reference dataset")
    if not review.get("reviewer", "").strip():
        raise ValueError("grade review requires an identified reviewer")
    cases = {r["id"]: r for r in read_jsonl(dataset)}
    seen = set()
    for entry in review["reviews"]:
        arm = entry["arm"]
        if arm not in {"original", "compressed"}:
            raise ValueError("review arm must be original or compressed")
        key = (entry["case_id"], arm, entry.get("variant"))
        if key in seen or (arm == "original" and entry.get("variant") is not None):
            raise ValueError("duplicate review or variant-specific original grade")
        seen.add(key)
        matched = [
            r
            for r in rows
            if r["case_id"] == entry["case_id"]
            and (arm == "original" or r["variant"] == entry.get("variant"))
        ]
        if not matched or any(
            grade_fingerprint(r, arm) != entry["grade_sha256"] for r in matched
        ):
            raise ValueError("review answer/source/grade does not match the saved run")
        if not entry.get("reason", "").strip():
            raise ValueError("every review needs an explanation")
        for row in matched:
            if "verdict" in entry:
                grade = score_verdict(cases[row["case_id"]], entry["verdict"])
                row[f"raw_{arm}_grade"] = row[f"{arm}_grade"]
                row[f"{arm}_grade"] = grade
                row[f"{arm}_correct"] = grade["correct"]
                row[f"{arm}_completeness"] = grade["completeness"]
            row.setdefault("grade_review", []).append(
                {"arm": arm, "reason": entry["reason"], "reviewer": review["reviewer"]}
            )
    for row in rows:
        row["regression"] = row["original_correct"] and not row["compressed_correct"]
        row["improvement"] = not row["original_correct"] and row["compressed_correct"]
    manifest.update(
        raw_results=str(source.resolve()),
        grade_review_sha256=file_hash(review_path),
        scoring=f"Model judge with explicit grade review by {review['reviewer']}; "
        "raw grades preserved separately; not independent human certification.",
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "grading_review.json", review)
    return write_reports(output, rows, manifest, complete=True)
