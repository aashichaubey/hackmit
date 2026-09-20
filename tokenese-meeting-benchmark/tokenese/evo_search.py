"""Resumeable development-only search; held-out selection is a separate command."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .evo_data import DATA, REPORTS, load_development
from .evo_eval import evaluate, experiment_identity, summarize
from .evo_grammar import GrammarSpec, digest
from .evo_mutate import seed_population, mutate, crossover, screen
from .evo_repair import propose_repairs, accept_repair, RepairRecord
from .evo_usage import Ledger, Runner, BudgetExceeded


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pareto(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not any(o["summary"]["correct"] >= r["summary"]["correct"] and
            o["summary"]["visible_prompt_tokens"] <= r["summary"]["visible_prompt_tokens"] and
            (o["summary"]["correct"] > r["summary"]["correct"] or o["summary"]["visible_prompt_tokens"] < r["summary"]["visible_prompt_tokens"])
            for o in rows)]


def probe_ids(cases: list[dict], failures: list[str], generation: int) -> set[str]:
    questions = [q for c in cases for q in c["questions"]]
    # Two questions from each of the first four public meetings stay fixed.
    anchors = [q["id"] for c in cases[:4] for q in c["questions"][:2]]
    lookup = {q["id"]: q for q in questions}
    # Contrast pairs belong in the pilot, not at the end of a long easy corpus.
    pool = list(dict.fromkeys([q['id'] for q in questions if q.get('contrast_id')] + [q["id"] for q in questions]))
    rotate = pool[generation*8 % max(1, len(pool)):] + pool[:generation*8 % max(1, len(pool))]
    selected = list(dict.fromkeys(anchors))[:8]
    for qid in (list(failures) if generation >= 2 else []) + rotate:
        additions = [qid]
        if lookup[qid].get("contrast_id") in lookup:
            additions.append(lookup[qid]["contrast_id"])
        additions = [i for i in additions if i not in selected]
        if len(selected) + len(additions) <= 16:
            selected.extend(additions)
        if len(selected) >= 16:
            break
    return set(selected)


def run_search(*, dev_path=DATA / "dev.json", output=REPORTS / "search.json", seed=42,
               generations=6, candidates_per_generation=6, runner=None, repair=True, local=False,
               reserve_calls=1800, reserve_usd=3.0, max_finalists=4, probe_only=False) -> dict:
    if not 1 <= generations <= 6 or not 1 <= candidates_per_generation <= 6 or not 1 <= max_finalists <= 4:
        raise ValueError("Search exceeds staged limits")
    cases = load_development(dev_path)
    output = Path(output)
    # Validation use permanently closes development mutation for this run.
    if (output.parent / "selection-lock.json").exists() and not local:
        raise ValueError("Validation has opened; mutation is frozen")
    identity = {**experiment_identity(), "dataset_hash": digest(cases), "seed": seed,
                "repair": repair, "candidates_per_generation": candidates_per_generation, "max_finalists": max_finalists,
                "probe_policy": "eight-public-anchors-eight-contrasts-v2", "grammar_code_hash": digest(Path(__file__).with_name('evo_grammar.py').read_text())}
    report = {"identity": identity, "status": "local_only" if local else "running", "generations": [],
              "repairs": [], "failure_archive": [], "finalists": [], "seed_population": [], 'loaded_splits':['dev']}
    if output.exists() and not local:
        report = json.loads(output.read_text())
        if report["identity"] != identity:
            raise ValueError("Search identity changed; do not mix experiments")
    seeds = seed_population()
    report["seed_population"] = [screen(g, cases) for g in seeds]
    if local:
        report["local_frontier"] = sorted(report["seed_population"], key=lambda s: (s["visible_input_tokens"], s["grammar_id"]))
        save(output, report)
        return report
    if runner is None:
        raise ValueError("API runner required")
    evaluated = {r["grammar_id"]: r for gen in report["generations"] for r in gen["candidates"]}
    parents = [GrammarSpec(**r["grammar"]) for r in report["generations"][-1]["candidates"]] if report["generations"] else seeds
    try:
        for generation in range(len(report["generations"]), generations):
            rng = random.Random(seed + generation)
            ids = probe_ids(cases, report["failure_archive"], generation)
            pool = {g.id: (g, None, "seed" if generation == 0 else "survivor") for g in parents}
            for _ in range(256-len(pool)):
                parent = rng.choice(parents)
                child = mutate(parent, rng) if rng.random() < .8 else crossover(parent, rng.choice(parents))
                pool[child.id] = (child, parent.id, "mutation")
            screens, rendered = [], set()
            for grammar, parent_id, origin in pool.values():
                candidate = screen(grammar, cases)
                if candidate["integrity_errors"] or candidate["output_hash"] in rendered:
                    continue
                rendered.add(candidate["output_hash"])
                screens.append({**candidate, "parent_id": parent_id, "origin": origin})
            screens.sort(key=lambda r: (r["visible_input_tokens"], len(r["grammar"]["repairs"]), r["grammar_id"]))
            # Always compare an explicit English seed on this round's same questions.
            chosen = [{**screen(seeds[0], cases), "parent_id": None, "origin": "english_anchor"}]
            if generation == 0:
                chosen.append({**screen(seeds[2], cases), "parent_id": None, "origin": "readable_seed"})
            # Preserve a useful near-neighbor slot: shortest-only selection kept
            # rediscovering unreadable punctuation and starved accurate parents.
            accurate_ids = {r['grammar_id'] for r in evaluated.values() if r['summary']['correct'] == r['summary']['total']} | {seeds[0].id}
            def atomic_distance(candidate):
                parent = evaluated.get(candidate['parent_id'], {}).get('grammar', seeds[0].to_dict())
                child = candidate['grammar']
                return sum(a != b for a,b in zip(parent['kind_forms'], child['kind_forms'])) + sum(parent[k] != child[k] for k in ('layout','owner_marker','deadline_marker','separator','grouping'))
            neighbor = next((s for s in screens if s['parent_id'] in accurate_ids and s['grammar_id'] not in evaluated
                             and s['grammar_id'] not in {x['grammar_id'] for x in chosen} and atomic_distance(s) == 1), None)
            if neighbor is not None and generation >= 2:
                chosen.append({**neighbor, 'origin':'accurate_parent_mutation'})
            for s in screens:
                if s["grammar_id"] not in {x["grammar_id"] for x in chosen}:
                    chosen.append(s)
                if len(chosen) >= max(2, candidates_per_generation-2):
                    break
            results = []
            pending = chosen[:candidates_per_generation]
            while pending and len(results) < candidates_per_generation:
                s = pending.pop(0)
                grammar = GrammarSpec(**s["grammar"])
                rows = evaluate(grammar, cases, runner, question_ids=ids, leave_calls=reserve_calls, leave_usd=reserve_usd)
                item = {**s, "summary": summarize(rows), "rows": rows, "probe_ids": sorted(ids)}
                results.append(item)
                evaluated[grammar.id] = item
                bad = [r for r in rows if not r["correct"]]
                report["failure_archive"] = list(dict.fromkeys(report["failure_archive"] + [r["question_id"] for r in bad]))[:40]
                if repair and bad and len(results)+len(pending) < candidates_per_generation:
                    for child, category in propose_repairs(grammar, bad):
                        if child.id not in {x["grammar_id"] for x in results+pending}:
                            pending.append({**screen(child, cases), "parent_id": grammar.id, "origin": "repair", "repair_category": category})
                        if len(results)+len(pending) >= candidates_per_generation:
                            break
                if s["origin"] == "repair":
                    parent = next(r for r in results if r["grammar_id"] == s["parent_id"])
                    acceptance = accept_repair(parent["rows"], rows, {"parent": parent["summary"]["visible_prompt_tokens"], "child": item["summary"]["visible_prompt_tokens"]})
                    record = RepairRecord(parent["grammar_id"], grammar.id, s["repair_category"], grammar.to_dict(), acceptance, parent["rows"], rows)
                    report["repairs"].append(record.to_dict())
                if not pending and len(results) < candidates_per_generation:
                    # Diversity slot chooses a different layout before cosmetic variants.
                    options = [r for r in screens if r["grammar_id"] not in {x["grammar_id"] for x in results}]
                    if options:
                        layouts = {x["grammar"]["layout"] for x in results}
                        pending.append(next((r for r in options if r["grammar"]["layout"] not in layouts), options[0]))
            frontier = pareto(results)
            ranked = sorted(results, key=lambda r: (-r["summary"]["correct"], r["summary"]["visible_prompt_tokens"], len(r["grammar"]["repairs"]), r["grammar_id"]))
            parents = [GrammarSpec(**r["grammar"]) for r in ranked[:3] + frontier[:3]]
            report["generations"].append({"generation": generation, "probe_ids": sorted(ids), "screened": len(screens),
                                          "candidates": results, "frontier_ids": [r["grammar_id"] for r in frontier]})
            save(output, report)
        # Probe scores across generations differ. Re-evaluate all nominees on one full suite.
        if probe_only:
            report['status'] = 'probe_ablation_complete'
            report['ledger'] = runner.ledger.summary() if runner.ledger else None
            save(output, report)
            return report
        nominees = {r["grammar_id"]: r for gen in report["generations"] for r in gen["candidates"] if r["summary"]["correct"] == r["summary"]["total"]}
        if not nominees:
            # Diagnostic finalists may fail. They cannot become deployable by nomination.
            nominees = {r["grammar_id"]: r for r in report["generations"][-1]["candidates"]
                        if r['summary']['correct'] == max(x['summary']['correct'] for x in report['generations'][-1]['candidates'])}
        ordered = sorted(nominees.values(), key=lambda r: (r["visible_input_tokens"], len(r["grammar"]["repairs"]), r["grammar_id"]))[:max(1,max_finalists-1)]
        seed_item = {"grammar": seeds[0].to_dict(), "grammar_id": seeds[0].id}
        if seeds[0].id not in {r["grammar_id"] for r in ordered}:
            ordered.append(seed_item)
        finalists = []
        for nominee in ordered[:max_finalists]:
            grammar = GrammarSpec(**nominee["grammar"])
            rows = evaluate(grammar, cases, runner, purpose="finalist", leave_calls=reserve_calls, leave_usd=reserve_usd)
            failed_archive = [r["question_id"] for r in rows if r["question_id"] in report["failure_archive"] and not r["correct"]]
            finalists.append({"grammar": grammar.to_dict(), "grammar_id": grammar.id, "rows": rows, "summary": summarize(rows),
                              "archive_pass": not failed_archive, "unresolved_archive": failed_archive})
        report["finalists"] = finalists
        report["status"] = "development_complete"
    except BudgetExceeded as exc:
        report["status"] = "incomplete_evaluation"
        report["stop_reason"] = str(exc)
    if runner.ledger:
        report["ledger"] = runner.ledger.summary()
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--max-calls", type=int, default=3500)
    parser.add_argument("--max-usd", type=float, default=10)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument("--probe-only", action="store_true", help="Matched-budget diagnostic ablation; cannot select a product grammar")
    args = parser.parse_args()
    runner = None
    if args.run_api:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        runner = Runner(OpenAI(), Ledger(REPORTS / "ledger.sqlite", args.max_calls, args.max_usd))
    local = args.local or not args.run_api
    filename = "local.json" if local else "search-no-repair.json" if args.no_repair else "search.json"
    result = run_search(output=REPORTS / filename, seed=args.seed, generations=args.generations, candidates_per_generation=args.candidates,
                        runner=runner, repair=not args.no_repair, local=local, max_finalists=args.finalists, probe_only=args.probe_only)
    print(json.dumps({"status": result["status"], "generations": len(result["generations"]), "ledger": result.get("ledger")}, indent=2))


if __name__ == "__main__":
    main()
