"""Run the layered meeting-note evaluation.

    python run_meeting_evals.py --layers 1,5            # offline only, free
    python run_meeting_evals.py --layers 1,2,3,4,5      # full, uses API calls

Layers
  1  deterministic fact recoverability (offline, ground truth by construction)
  2  downstream QA with the grammar/legend supplied  ("explained")
  3  semantic-equivalence judging (JEv + OpenRouter LLM judge)
  4  downstream QA with NO hint that the input is compressed ("blind")
  5  token economics: gross and net, plus amortization break-even

Every layer uses the SAME downstream model and the SAME questions across all
four representations, so differences are attributable to the representation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "character_compression"))

from facts import check_facts  # noqa: E402
from judge import (  # noqa: E402
    JevJudge,
    LlmJudge,
    agreement_with_deterministic,
    combine,
)
from representations import METHODS, build_renderers  # noqa: E402

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Downstream model
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    output: str
    input_tokens: int | None
    error: str | None = None


def ask(context: str, question: str, grammar: str, model: str,
        timeout: float = 90.0, max_tokens: int = 2048) -> QAResult:
    """Ask one question about one representation.

    When `grammar` is empty the model is given no hint that the input is
    compressed - that is the Layer 4 blind condition.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    system = "Answer using only the provided notes. Give the shortest possible answer, no explanation."
    if grammar:
        system += "\n\n" + grammar
    user = f"NOTES:\n{context}\n\nQUESTION:\n{question}"

    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            p = json.loads(resp.read())
        choice = p["choices"][0]
        content = choice["message"].get("content") or ""
        usage = p.get("usage") or {}
        err = None
        if not content.strip() and choice.get("finish_reason") == "length":
            err = "truncated_before_answer"
        return QAResult(content.strip(), usage.get("prompt_tokens"), err)
    except Exception as exc:
        return QAResult("", None, f"{type(exc).__name__}: {str(exc)[:150]}")


def graded(answer: str, expected: str) -> bool:
    """Lenient containment match, case-insensitive.

    Deliberately lenient: we are measuring whether the *representation*
    destroyed the fact, not whether the model formatted its reply nicely.
    Minimal-pair stress cases still discriminate because their expected
    answers differ.

    KNOWN LIMITATION (found during live testing): this is an English-literal
    match. A model answering correctly in Chinese (e.g. "周五" for "Friday")
    will be graded wrong here even though the answer is right. Treat scores
    against non-English representations as a lower bound; verify surprising
    misses by hand before concluding the representation lost information.
    """
    a, e = answer.strip().lower(), expected.strip().lower()
    if not a:
        return False
    if a == e or e in a:
        return True
    # YES/NO answers must not be satisfied by substring accident.
    if e in ("yes", "no"):
        first = a.replace("*", "").split()[:2]
        return any(t.strip(".,:;").lower() == e for t in first)
    return False


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def layer5_economics(notes, renderers, counter) -> dict:
    """Token economics: gross and net, plus the amortization break-even."""
    ctx = defaultdict(int)
    grammar_tok: dict[str, int] = {}
    per_note: dict[str, list] = defaultdict(list)

    for n in notes:
        for m in METHODS:
            r = renderers[m](n["text"])
            t = counter.count_text(r.text)
            ctx[m] += t
            per_note[m].append(t)
            if r.grammar:
                grammar_tok[m] = counter.count_text(r.grammar)

    base = ctx["raw"]
    out = {}
    for m in METHODS:
        g = grammar_tok.get(m, 0)
        gross = 1 - ctx[m] / base if base else 0.0
        # Net when the grammar is paid ONCE for the whole corpus (amortized).
        net_amortized = 1 - (ctx[m] + g) / base if base else 0.0
        # Net when the grammar is paid per request (the realistic single-note case).
        net_per_request = 1 - (ctx[m] + g * len(notes)) / base if base else 0.0
        # Original-token volume needed for the grammar to pay for itself.
        breakeven = (g / gross) if gross > 0 else None
        out[m] = {
            "context_tokens": ctx[m],
            "grammar_tokens": g,
            "gross_savings": round(gross, 4),
            "net_savings_amortized": round(net_amortized, 4),
            "net_savings_per_request": round(net_per_request, 4),
            "breakeven_original_tokens": round(breakeven) if breakeven else None,
            "compression_ratio": round(base / ctx[m], 3) if ctx[m] else None,
        }
    out["_base_tokens"] = base
    out["_notes"] = len(notes)
    return out


def layer1_facts(notes, renderers) -> dict:
    out: dict[str, Any] = {}
    for m in METHODS:
        tot = rec = crit = neg_broken = 0
        failures = []
        for n in notes:
            r = renderers[m](n["text"])
            rep = check_facts(n["facts"], r.text, r.grammar, original=n["text"])
            tot += rep.total
            rec += rep.recovered
            crit += rep.critical_failures
            if not rep.negation_preserved:
                neg_broken += 1
                failures.append({"note": n["id"], "issue": rep.negation_detail})
            for fr in rep.results:
                if not fr.recovered:
                    failures.append({"note": n["id"], "fact": fr.fact.get("type"),
                                     "missing": fr.missing})
        out[m] = {
            "facts_total": tot,
            "facts_recovered": rec,
            "fact_accuracy": round(rec / tot, 4) if tot else None,
            "critical_failures": crit,
            "notes_with_broken_negation": neg_broken,
            "failures": failures[:20],
        }
    return out


def layer24_qa(notes, renderers, model, blind: bool, workers: int) -> dict:
    """Downstream QA. blind=True withholds the grammar (Layer 4)."""
    jobs = []
    for n in notes:
        for m in METHODS:
            r = renderers[m](n["text"])
            grammar = "" if blind else r.grammar
            for q in n["questions"]:
                jobs.append((m, n["id"], q, r.text, grammar))

    def run(job):
        m, nid, q, text, grammar = job
        res = ask(text, q["question"], grammar, model)
        return m, nid, q, res, graded(res.output, q["answer"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, jobs))

    agg: dict[str, Any] = {}
    detail = []
    for m in METHODS:
        rows = [r for r in results if r[0] == m]
        correct = sum(r[4] for r in rows)
        agg[m] = {
            "questions": len(rows),
            "correct": correct,
            "qa_accuracy": round(correct / len(rows), 4) if rows else None,
            "errors": sum(bool(r[3].error) for r in rows),
        }
    for m, nid, q, res, ok in results:
        if not ok:
            detail.append({"method": m, "note": nid, "question": q["question"],
                           "expected": q["answer"], "got": res.output[:120],
                           "error": res.error})
    return {"by_method": agg, "failures": detail}


def layer_stress(stress, renderers, model, workers: int) -> dict:
    """Adversarial minimal pairs, blind. Collapse = semantic corruption."""
    jobs = []
    for c in stress:
        for m in METHODS:
            r = renderers[m](c["text"])
            jobs.append((m, c, r.text))

    def run(job):
        m, c, text = job
        res = ask(text, c["question"], "", model)
        return m, c, res, graded(res.output, c["answer"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, jobs))

    agg: dict[str, Any] = {}
    collapses: list[dict] = []
    for m in METHODS:
        rows = [r for r in results if r[0] == m]
        correct = sum(r[3] for r in rows)
        # A pair "collapses" when both sides produce the same answer even
        # though the gold answers differ - the representation destroyed the
        # distinction.
        by_pair: dict[str, list] = defaultdict(list)
        for _m, c, res, ok in rows:
            by_pair[c["pair_id"]].append((c, res, ok))
        collapsed = 0
        for pid, items in by_pair.items():
            if len(items) == 2:
                a, b = items
                if a[1].output.strip().lower() == b[1].output.strip().lower():
                    collapsed += 1
                    collapses.append({"method": m, "pair": pid,
                                      "kind": a[0]["kind"],
                                      "both_answered": a[1].output[:60]})
        agg[m] = {
            "cases": len(rows),
            "correct": correct,
            "stress_accuracy": round(correct / len(rows), 4) if rows else None,
            "pairs": len(by_pair),
            "collapsed_pairs": collapsed,
        }
    return {"by_method": agg, "collapses": collapses}


def layer3_judges(notes, renderers, facts_by_method, model, workers: int,
                  use_jev: bool = True) -> dict:
    jev = None
    if use_jev:
        try:
            jev = JevJudge()
        except Exception as exc:
            print(f"  JEv unavailable: {exc}", file=sys.stderr)
    try:
        llm = LlmJudge(model=model)
    except Exception as exc:
        print(f"  LLM judge unavailable: {exc}", file=sys.stderr)
        llm = None

    jobs = []
    for n in notes:
        task = "; ".join(q["question"] for q in n["questions"])
        for m in METHODS:
            if m == "raw":
                continue
            r = renderers[m](n["text"])
            rep = check_facts(n["facts"], r.text, r.grammar, original=n["text"])
            deterministic_ok = (rep.recovered == rep.total) and rep.negation_preserved
            jobs.append((m, n["id"], n["text"], r.text, task, deterministic_ok))

    def run(job):
        m, nid, orig, comp, task, det_ok = job
        jv = jev.judge(orig, comp, task) if jev else None
        lv = llm.judge(orig, comp, task) if llm else None
        merged = combine(jv, lv)
        merged["agreement_with_deterministic"] = agreement_with_deterministic(
            merged.get("preserves_task_information"), det_ok)
        merged["deterministic_facts_ok"] = det_ok
        return m, nid, merged

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, jobs))

    agg: dict[str, Any] = {}
    for m in METHODS:
        if m == "raw":
            continue
        rows = [r[2] for r in results if r[0] == m]
        if not rows:
            continue
        quad = defaultdict(int)
        for r in rows:
            quad[r["agreement_with_deterministic"]] += 1
        agree = [r["judges_agree"] for r in rows if r.get("judges_agree") is not None]
        agg[m] = {
            "judged": len(rows),
            "judged_preserved": sum(bool(r.get("preserves_task_information")) for r in rows),
            "mean_severity": round(
                sum(r["severity"] for r in rows if isinstance(r.get("severity"), (int, float)))
                / max(1, sum(1 for r in rows if isinstance(r.get("severity"), (int, float)))), 3),
            "quadrants": dict(quad),
            "false_pass": quad.get("false_pass", 0),
            "jev_llm_agreement": round(sum(agree) / len(agree), 3) if agree else None,
        }
    return {"by_method": agg,
            "detail": [{"method": m, "note": nid, **r} for m, nid, r in results]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="1,5")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--notes", default="meeting_notes.jsonl")
    ap.add_argument("--stress-file", default="stress_tests.jsonl")
    ap.add_argument("--dictionary", default="../../data/dictionaries/phrases.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--no-stress", action="store_true")
    args = ap.parse_args()

    layers = {s.strip() for s in args.layers.split(",") if s.strip()}
    notes = [json.loads(l) for l in (HERE / args.notes).read_text(encoding="utf-8").splitlines() if l.strip()]
    stress = [json.loads(l) for l in (HERE / args.stress_file).read_text(encoding="utf-8").splitlines() if l.strip()]

    from adapters import DeepSeekTokenCounter
    counter = DeepSeekTokenCounter()
    renderers = build_renderers(HERE / args.dictionary)

    report: dict[str, Any] = {
        "model": args.model,
        "notes": len(notes),
        "stress_cases": len(stress),
        "methods": METHODS,
        "layers_run": sorted(layers),
    }

    if "5" in layers:
        print("Layer 5: token economics ...")
        report["layer5_economics"] = layer5_economics(notes, renderers, counter)

    facts_by_method = {}
    if "1" in layers:
        print("Layer 1: deterministic fact checks ...")
        facts_by_method = layer1_facts(notes, renderers)
        report["layer1_facts"] = facts_by_method

    if "4" in layers:
        print("Layer 4: BLIND downstream QA ...")
        report["layer4_qa_blind"] = layer24_qa(notes, renderers, args.model, True, args.workers)

    if "2" in layers:
        print("Layer 2: downstream QA with grammar supplied ...")
        report["layer2_qa_explained"] = layer24_qa(notes, renderers, args.model, False, args.workers)

    if "3" in layers:
        print("Layer 3: judges (JEv + LLM) ...")
        report["layer3_judges"] = layer3_judges(
            notes, renderers, facts_by_method, args.model, args.workers,
            use_jev=not args.no_jev)

    if not args.no_stress and ({"2", "4"} & layers):
        print("Stress tests: adversarial minimal pairs ...")
        report["stress"] = layer_stress(stress, renderers, args.model, args.workers)

    outdir = HERE / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {outdir/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
