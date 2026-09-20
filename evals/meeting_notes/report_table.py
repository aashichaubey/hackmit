"""Render the final evaluation table (Section 32) from results/report.json.

    python report_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

LABELS = {
    "raw": "Raw",
    "en_zh": "EN/ZH",
    "simple_compression": "Simple compression",
    "new_language": "New language",
}


def pct(v, digits=1):
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="results/report.json")
    ap.add_argument("--out", default="results/FINAL_TABLE.md")
    args = ap.parse_args()

    r = json.loads((HERE / args.report).read_text(encoding="utf-8"))
    econ = r.get("layer5_economics", {})
    facts = r.get("layer1_facts", {})
    blind = r.get("layer4_qa_blind", {}).get("by_method", {})
    expl = r.get("layer2_qa_explained", {}).get("by_method", {})
    judges = r.get("layer3_judges", {}).get("by_method", {})
    stress = r.get("stress", {}).get("by_method", {})

    L: list[str] = []
    L.append("# Meeting-note compression: final evaluation\n")
    L.append(f"- downstream model: `{r.get('model')}`  (identical across all methods)")
    L.append(f"- notes: {r.get('notes')}   stress cases: {r.get('stress_cases')}")
    L.append(f"- layers run: {', '.join(r.get('layers_run', []))}")
    L.append("- fact ground truth is *planted*, not extracted; the Raw row is a "
             "calibration control and must read 100%.\n")

    L.append("## Main table\n")
    L.append("| Method | Input Tokens | Token Savings | QA Accuracy (blind) | "
             "Fact Accuracy | Compiler Cost | Net Savings |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in r.get("methods", []):
        e, f = econ.get(m, {}), facts.get(m, {})
        b = blind.get(m, {})
        L.append(
            f"| {LABELS.get(m, m)} | {e.get('context_tokens','n/a')} | "
            f"{pct(e.get('gross_savings'))} | {pct(b.get('qa_accuracy'))} | "
            f"{pct(f.get('fact_accuracy'))} | {e.get('grammar_tokens', 0)} | "
            f"{pct(e.get('net_savings_amortized'))} |"
        )
    L.append("\n*Compiler cost = grammar/legend tokens that must accompany the "
             "representation. Net savings amortizes that cost once across all "
             f"{r.get('notes')} notes - the most generous honest framing.*\n")

    L.append("## Net savings under both accounting models\n")
    L.append("| Method | Gross | Net (grammar amortized) | Net (grammar per request) | Break-even |")
    L.append("|---|---:|---:|---:|---:|")
    for m in r.get("methods", []):
        e = econ.get(m, {})
        be = e.get("breakeven_original_tokens")
        L.append(f"| {LABELS.get(m,m)} | {pct(e.get('gross_savings'))} | "
                 f"{pct(e.get('net_savings_amortized'))} | "
                 f"{pct(e.get('net_savings_per_request'))} | "
                 f"{str(be) + ' tok' if be else 'n/a'} |")
    L.append("")

    if blind or expl:
        L.append("## Layer 2 vs Layer 4: does explaining the grammar help?\n")
        L.append("| Method | QA blind | QA with grammar supplied | delta |")
        L.append("|---|---:|---:|---:|")
        for m in r.get("methods", []):
            b, x = blind.get(m, {}), expl.get(m, {})
            ba, xa = b.get("qa_accuracy"), x.get("qa_accuracy")
            d = "n/a" if (ba is None or xa is None) else f"{100*(xa-ba):+.1f} pp"
            L.append(f"| {LABELS.get(m,m)} | {pct(ba)} | {pct(xa)} | {d} |")
        L.append("")

    if stress:
        L.append("## Stress tests (adversarial minimal pairs)\n")
        L.append("A *collapsed pair* means both sides of a minimal pair produced the "
                 "same answer although the gold answers differ - the representation "
                 "destroyed the distinction.\n")
        L.append("| Method | Accuracy | Pairs | Collapsed pairs |")
        L.append("|---|---:|---:|---:|")
        for m in r.get("methods", []):
            s = stress.get(m, {})
            L.append(f"| {LABELS.get(m,m)} | {pct(s.get('stress_accuracy'))} | "
                     f"{s.get('pairs','n/a')} | {s.get('collapsed_pairs','n/a')} |")
        L.append("")
        collapses = r.get("stress", {}).get("collapses", [])
        if collapses:
            L.append("Collapsed:\n")
            for c in collapses[:15]:
                L.append(f"- `{c['method']}` / {c['pair']} ({c['kind']}): both sides answered "
                         f"`{c['both_answered']}`")
            L.append("")

    if judges:
        L.append("## Layer 3: judges vs deterministic ground truth\n")
        L.append("`false_pass` = the judge said information was preserved when the "
                 "deterministic checker proves a planted fact was lost. This is the "
                 "quadrant that justifies not trusting a judge alone.\n")
        L.append("| Method | Judged preserved | Mean severity | false_pass | JEv/LLM agreement |")
        L.append("|---|---:|---:|---:|---:|")
        for m in r.get("methods", []):
            j = judges.get(m)
            if not j:
                continue
            L.append(f"| {LABELS.get(m,m)} | {j['judged_preserved']}/{j['judged']} | "
                     f"{j.get('mean_severity','n/a')} | **{j.get('false_pass',0)}** | "
                     f"{pct(j.get('jev_llm_agreement')) if j.get('jev_llm_agreement') is not None else 'n/a'} |")
        L.append("")
        L.append("Quadrants per method (judge vs deterministic):\n")
        for m in r.get("methods", []):
            j = judges.get(m)
            if j:
                L.append(f"- `{m}`: {j.get('quadrants')}")
        L.append("")

    L.append("## Where it fails\n")
    fl = facts.get("new_language", {}).get("failures", [])
    if fl:
        L.append("`new_language` fact losses (sample):\n")
        for f in fl[:8]:
            L.append(f"- {f.get('note')}: {f.get('fact')} missing {f.get('missing')}")
        L.append("")

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
