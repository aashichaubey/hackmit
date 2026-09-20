"""Human-readable report, failure diagnostics, and gate evaluation.

    python report.py --run-dir runs/<experiment-id> [--policy release_policy.example.json]

Consumes the scorer's `report.json`/`scored_cases.jsonl` plus the runner's
`manifest.json`/`predictions.jsonl`, and emits `report.md`, `failures.jsonl`,
and `gate.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from policy import evaluate, load_policy  # noqa: E402
from score_evals import load_jsonl  # noqa: E402


def _pct(v: float | None, digits: int = 1) -> str:
    return "unknown" if v is None else f"{100 * v:.{digits}f}%"


def _num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "unknown"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def build_failures(scored: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full diagnostics for every non-both-correct pair.

    Includes the exact compressed request so a regression can be reproduced
    without re-running the harness.
    """
    pred_index = {(p["case_id"], p.get("run_id", 0)): p for p in predictions}
    out = []
    for row in scored:
        b, c = row["baseline_pass"], row["compressed_pass"]
        if b and c:
            continue
        key = (row["case_id"], row.get("run_id", 0))
        pred = pred_index.get(key, {})
        if b and not c:
            kind = "regression_baseline_correct_compressed_wrong"
        elif not b and c:
            kind = "recovery_baseline_wrong_compressed_correct"
        else:
            kind = "both_wrong"
        out.append({
            "case_id": row["case_id"],
            "run_id": row.get("run_id", 0),
            "outcome": kind,
            "category": row["category"],
            "cluster_id": row["cluster_id"],
            "expected_output": row["expected_output"],
            "grader": row["grader"],
            "baseline_output": row["baseline_output"],
            "compressed_output": row["compressed_output"],
            "baseline_error": row.get("baseline_error"),
            "compressed_error": row.get("compressed_error"),
            "baseline_input_tokens": row.get("baseline_input_tokens"),
            "compressed_input_tokens": row.get("compressed_input_tokens"),
            "token_count_source": row.get("token_count_source"),
            "baseline_messages": pred.get("baseline_messages"),
            "compressed_messages": row.get("compressed_messages"),
            "compression_metadata": row.get("compression_metadata"),
            # The harness observes the transition; it does not assert a cause.
            "diagnosis": "unclassified",
        })
    return out


def render_markdown(report: dict[str, Any], manifest: dict[str, Any],
                    gate: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    o = report["overall"]
    t = o.get("tokens")
    u = report.get("uncertainty", {})
    L: list[str] = []

    L.append(f"# Character-compression evaluation - `{manifest.get('experiment_id')}`\n")

    if manifest.get("synthetic_results"):
        L.append("> **SYNTHETIC RESULTS.** This run used mock adapters. The numbers below "
                 "exercise the harness; they are **not** measurements of any model or "
                 "compressor and must never be quoted as performance.\n")

    L.append("## Run\n")
    ad = manifest.get("adapters", {})
    L.append(f"| field | value |\n|---|---|")
    L.append(f"| mode | `{manifest.get('run_mode')}` |")
    L.append(f"| experiment mode | `{manifest.get('experiment_mode')}` |")
    L.append(f"| compressor | `{ad.get('compressor')}`{' **(MOCK)**' if ad.get('compressor_is_mock') else ''} |")
    L.append(f"| target | `{ad.get('target')}`{' **(MOCK)**' if ad.get('target_is_mock') else ''} |")
    L.append(f"| token counter | `{ad.get('token_counter')}` |")
    L.append(f"| fixtures | `{Path(manifest.get('fixture_path','')).name}` "
             f"({manifest.get('fixture_count')} cases, sha256 `{str(manifest.get('fixture_sha256'))[:12]}`) |")
    L.append(f"| splits | {', '.join(report.get('fixture_splits', []))} |")
    L.append(f"| repeats | {manifest.get('repeats')} |")
    L.append(f"| paired trials | {o.get('pairs')} |")
    L.append(f"| live calls | {manifest.get('live_calls_made')} (budget {manifest.get('live_call_budget')}) |")
    L.append(f"| harness revision | `{manifest.get('implementation_revision')}` |\n")

    L.append("## Gate\n")
    L.append(f"**`{gate['gate_status']}`**\n")
    if gate.get("reasons"):
        L.append("Reasons:\n")
        for r in gate["reasons"]:
            L.append(f"- {r}")
        L.append("")
    if gate.get("checks"):
        L.append("| check | status | observed | threshold |\n|---|---|---|---|")
        for c in gate["checks"]:
            L.append(f"| {c['check']} | {c['status']} | `{c['observed']}` | `{c['threshold']}` |")
        L.append("")

    L.append("## Paired quality\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| baseline accuracy | {_pct(o.get('baseline_accuracy'))} |")
    L.append(f"| compressed accuracy | {_pct(o.get('compressed_accuracy'))} |")
    L.append(f"| accuracy delta | {_num(o.get('accuracy_delta_pp'))} pp |")
    L.append(f"| both correct | {o.get('both_correct')} |")
    L.append(f"| **regression** (base ok -> comp wrong) | {o.get('baseline_correct_compressed_wrong')} |")
    L.append(f"| recovery (base wrong -> comp ok) | {o.get('baseline_wrong_compressed_correct')} |")
    L.append(f"| both wrong | {o.get('both_wrong')} |")
    rr = o.get("regression_rate_given_baseline_correct")
    L.append(f"| regression rate | {'undefined (no baseline-correct pairs)' if rr is None else _pct(rr)} |")
    L.append(f"| baseline errors | {o.get('baseline_error_count')} |")
    L.append(f"| compressed errors | {o.get('compressed_error_count')} |")
    L.append(f"| literal output agreement | {_pct(o.get('literal_output_agreement'))} "
             f"(not evidence of correctness) |\n")

    L.append("## Input-token accounting\n")
    L.append(f"Measurement coverage: **{_pct(o.get('token_measurement_coverage'))}**\n")
    if t is None:
        L.append("Token totals are **unknown** - at least one pair lacked a count. "
                 "The complete-cohort savings claim is withheld.\n")
    else:
        L.append("| metric | value |\n|---|---|")
        L.append(f"| baseline total input | {t['baseline_total_input']:,} |")
        L.append(f"| compressed total input | {t['compressed_total_input']:,} |")
        L.append(f"| **net input savings** | {_pct(t['net_input_token_savings'])} |")
        L.append(f"| compression ratio | {_num(t['full_request_compression_ratio'])}x |")
        L.append(f"| mean per-request savings | {_pct(t['mean_per_request_savings'])} |")
        L.append(f"| p50 per-request savings | {_pct(t['p50_per_request_savings'])} |")
        L.append(f"| p05 per-request savings | {_pct(t['p05_per_request_savings'])} |")
        L.append(f"| **expansion rate** | {_pct(t['expansion_rate'])} |")
        L.append(f"| compressed correct / 1k input tokens | {_num(t['compressed_correct_per_1000_input_tokens'], 3)} |\n")
    srcs = gate.get("evidence", {}).get("token_count_sources", {})
    L.append(f"Token count sources: `{srcs}`. "
             "Only `provider_usage`/`exact_local_tokenizer` may back a release gate.\n")

    L.append("## Uncertainty\n")
    L.append(f"- method: {u.get('method')} ({u.get('resamples')} resamples, seed {u.get('seed')}, "
             f"{u.get('clusters')} independent clusters)")
    ai = u.get("accuracy_delta_pp_95pct_interval")
    L.append(f"- accuracy delta 95% interval: {ai}")
    L.append(f"- net token savings 95% interval: {u.get('net_token_savings_95pct_interval')}")
    L.append(f"\n> {u.get('caution')}\n")

    L.append("## By category\n")
    L.append("| category | n | base acc | comp acc | delta pp | regressions | net savings |\n|---|---|---|---|---|---|---|")
    for cat, m in sorted(report.get("by_category", {}).items()):
        mt = m.get("tokens")
        L.append(f"| {cat} | {m['pairs']} | {_pct(m['baseline_accuracy'],0)} | "
                 f"{_pct(m['compressed_accuracy'],0)} | {_num(m['accuracy_delta_pp'],1)} | "
                 f"{m['baseline_correct_compressed_wrong']} | "
                 f"{_pct(mt['net_input_token_savings']) if mt else 'unknown'} |")
    L.append("")

    regressions = [f for f in failures if f["outcome"].startswith("regression")]
    L.append(f"## Failure diagnostics\n")
    L.append(f"{len(failures)} non-both-correct pairs ({len(regressions)} regressions). "
             f"Full detail with exact requests in `failures.jsonl`.\n")
    for f in regressions[:10]:
        L.append(f"- **{f['case_id']}** ({f['category']}): expected `{f['expected_output']}`, "
                 f"compressed produced `{f['compressed_output'][:60]}`"
                 f"{' / error: ' + str(f['compressed_error']) if f['compressed_error'] else ''}")
    L.append("")

    L.append("## Limitations\n")
    for lim in manifest.get("limitations", []):
        L.append(f"- {lim}")
    for note in report.get("notes", []):
        L.append(f"- {note}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--policy", type=Path, default=None)
    args = ap.parse_args()

    d = args.run_dir if args.run_dir.is_absolute() else HERE / args.run_dir
    try:
        report = json.loads((d / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        scored = load_jsonl(d / "scored_cases.jsonl")
        predictions = load_jsonl(d / "predictions.jsonl")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Run `run_evals.py run` then `score_evals.py` first.", file=sys.stderr)
        return 2

    policy = load_policy(args.policy)
    gate = evaluate(report, manifest, predictions, policy).as_dict()
    failures = build_failures(scored, predictions)

    (d / "gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    with (d / "failures.jsonl").open("w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n")
    md = render_markdown(report, manifest, gate, failures)
    (d / "report.md").write_text(md, encoding="utf-8")

    print(f"gate: {gate['gate_status']}")
    for r in gate.get("reasons", []):
        print(f"  - {r}")
    print(f"failures: {len(failures)}")
    print(f"wrote {d/'report.md'}, {d/'failures.jsonl'}, {d/'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
