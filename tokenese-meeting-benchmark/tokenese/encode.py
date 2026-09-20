import json
from pathlib import Path
from .facts import MeetingFacts

CANDIDATES = Path(__file__).resolve().parent.parent / "data" / "candidates.json"

def profiles():
    return json.loads(CANDIDATES.read_text())

def render_english(facts: MeetingFacts) -> str:
    lines = []
    for fact in facts.facts:
        parts = [fact.kind, fact.person, fact.text, fact.deadline]
        lines.append(" | ".join(part for part in parts if part))
    return "\n".join(lines)

def render_tokenese(facts: MeetingFacts, profile: str) -> str:
    item = profiles()[profile]
    sep = item["separator"]
    lines = []
    for fact in facts.facts:
        parts = [item["labels"][fact.kind]]
        if fact.kind != "decision" or fact.person:
            parts.append(fact.person)
        parts.append(fact.text)
        if fact.deadline:
            parts.append(item["deadline_marker"] + fact.deadline)
        lines.append(sep.join(parts))
    return "\n".join(lines)

def tokenese_legend(profile: str) -> str:
    item = profiles()[profile]
    labels = "; ".join(f"{label} {kind}" for kind, label in item["labels"].items())
    return f"{labels}. {item['separator']} fields: label, person, text, {item['deadline_marker']}deadline. Decision may omit person."
