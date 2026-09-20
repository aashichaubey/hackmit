"""Failures suggest hypotheses. Controlled paired probes decide whether they help."""
from dataclasses import asdict, dataclass

from .evo_grammar import GrammarSpec, KINDS


def classify_failure(row: dict) -> str:
    if row.get("error"):
        return "provider"
    category = row["category"]
    return {"owner": "owner", "speaker_assignee": "owner", "deadline": "deadline", "date_modifier": "deadline",
            "proposal": "proposal", "decision": "decision", "unknown": "proposal", "negation": "disagreement",
            "agreement": "agreement", "disagreement": "disagreement", "grouping": "boundaries"}.get(category, "boundaries")


def propose_repairs(grammar: GrammarSpec, failures: list[dict]) -> list[tuple[GrammarSpec, str]]:
    if len(grammar.repairs) >= 4:
        return []
    edits = []
    hypotheses = {classify_failure(r) for r in failures if not r['correct']}
    # Owner failures can come from an unreadable kind marker, not just a missing
    # owner label. Test those as separate one-rule hypotheses.
    for row in failures:
        if not row['correct']:
            for kind in KINDS:
                if f'the {kind} about' in row.get('question',''):
                    hypotheses.add(kind)
    for rule in sorted(hypotheses):
        if rule in grammar.repairs or rule == "provider":
            continue
        changes = {}
        if rule in KINDS:
            forms = list(grammar.kind_forms)
            forms[KINDS.index(rule)] = "proposal (unconfirmed)" if rule == "proposal" and forms[KINDS.index(rule)] in ('proposal', 'proposed') else rule
            if tuple(forms) == grammar.kind_forms:
                continue
            changes["kind_forms"] = tuple(forms)
        elif rule == "owner":
            if grammar.owner_marker == "owner:":
                continue
            changes["owner_marker"] = "owner:"
        elif rule == "deadline":
            if grammar.deadline_marker == "due:":
                continue
            changes["deadline_marker"] = "due:"
        elif rule == "boundaries":
            if not grammar.grouping:
                continue
            changes["grouping"] = False
        edits.append((grammar.edit(repairs=grammar.repairs + (rule,), **changes), rule))
    return edits


def accept_repair(parent_rows: list[dict], child_rows: list[dict], costs: dict) -> dict:
    parent = {r["question_id"]: r for r in parent_rows}
    child = {r["question_id"]: r for r in child_rows}
    if set(parent) != set(child) or len(parent) != len(parent_rows) or len(child) != len(child_rows):
        raise ValueError("Repair comparison must share exactly the same probe")
    fixed = [qid for qid in parent if not parent[qid]["correct"] and child[qid]["correct"]]
    regressions = [qid for qid in parent if parent[qid]["correct"] and not child[qid]["correct"]]
    sources = {child[qid]["source_id"] for qid in fixed}
    paired = []
    for qid in fixed:
        contrast = parent[qid].get("contrast_id")
        if contrast and contrast in child and child[contrast]["correct"]:
            paired.append(qid)
    accepted = bool(paired) and not regressions
    return {"accepted": accepted, "confirmed": accepted and len(sources) >= 2,
            "fixed": fixed, "paired_fixes": paired, "regressions": regressions,
            "source_meetings_improved": len(sources), "additional_tokens": costs["child"]-costs["parent"],
            "status": "confirmed" if accepted and len(sources) >= 2 else "provisional" if accepted else "unconfirmed"}


@dataclass(frozen=True)
class RepairRecord:
    parent_id: str
    child_id: str
    category: str
    edit: dict
    result: dict
    before: list[dict]
    after: list[dict]

    def to_dict(self):
        return asdict(self)
