"""Generate the meeting-note evaluation dataset.

Notes are built from templates with *planted* facts, so the ground truth is
known by construction rather than extracted afterwards. That matters: Layer 1
checks fact recoverability, and a fact-extractor used to build the gold set
would make its own errors invisible.

Covers the required case types: simple action items, multiple people,
conflicting statements, changed deadlines, numbers, percentages, dates,
relative dates, project names, similar names, pronouns, decisions,
disagreements, conditional statements, questions, blockers, dependencies,
and negation.

    python generate_dataset.py --out meeting_notes.jsonl --stress stress_tests.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------
# Fact vocabulary
# --------------------------------------------------------------------------

FactT = dict[str, str]


def F(type_: str, **kw: str) -> FactT:
    return {"type": type_, **kw}


# --------------------------------------------------------------------------
# Meeting notes. Each: id, tags, text, facts (planted), questions (gold).
#
# NOTE: fact values must be literal spans that appear in `text` (not
# paraphrases and not descriptive labels like a `metric` name) - the
# deterministic checker in facts.py does substring matching, and the `raw`
# control representation must score 100% fact accuracy or the checker itself
# is miscalibrated.
# --------------------------------------------------------------------------

MEETINGS: list[dict] = [
    {
        "id": "mn_simple_action",
        "tags": ["action_item", "deadline", "project"],
        "text": (
            "Platform sync, 12 March.\n"
            "Sarah will finish the API integration by Thursday.\n"
            "Dan will review the migration script afterwards."
        ),
        "facts": [
            F("person", value="Sarah"), F("person", value="Dan"),
            F("project", value="API integration"),
            F("action_item", person="Sarah", action="finish",
              object="API integration", deadline="Thursday"),
            F("deadline", person="Sarah", value="Thursday"),
            F("date", value="12 March"),
        ],
        "questions": [
            {"question": "Who owns the API integration?", "answer": "Sarah"},
            {"question": "By when must the API integration be finished?", "answer": "Thursday"},
        ],
    },
    {
        "id": "mn_changed_deadline",
        "tags": ["changed_deadline", "status_change", "date"],
        "text": (
            "Release planning, 4 June.\n"
            "The deadline for the billing rollout moved from Thursday to Friday.\n"
            "Priya confirmed the new date with the vendor."
        ),
        "facts": [
            F("person", value="Priya"),
            F("project", value="billing rollout"),
            F("status_change", object="billing rollout deadline",
              from_value="Thursday", to_value="Friday"),
            F("deadline", project="billing rollout", value="Friday"),
        ],
        "questions": [
            {"question": "What is the current deadline for the billing rollout?", "answer": "Friday"},
            {"question": "What was the previous deadline for the billing rollout?", "answer": "Thursday"},
        ],
    },
    {
        "id": "mn_negation_agreement",
        "tags": ["negation", "disagreement", "decision"],
        "text": (
            "Architecture review, 9 April.\n"
            "John did not agree with Sarah about the caching layer.\n"
            "No decision was recorded; the team will revisit next week."
        ),
        "facts": [
            F("person", value="John"), F("person", value="Sarah"),
            F("disagreement", between="John", and_="Sarah", topic="caching layer"),
            F("negation", statement="John did not agree with Sarah"),
            F("decision", value="revisit next week"),
        ],
        "questions": [
            {"question": "Did John agree with Sarah about the caching layer? Answer YES or NO.", "answer": "NO"},
            {"question": "Was a decision recorded about the caching layer? Answer YES or NO.", "answer": "NO"},
        ],
    },
    {
        "id": "mn_numbers_percent",
        "tags": ["numbers", "percentages", "status_change"],
        "text": (
            "Metrics review, 21 May.\n"
            "Checkout conversion rose from 13% to 30% after the redesign.\n"
            "Infrastructure spend was 1.5M dollars for the quarter."
        ),
        "facts": [
            F("percentage", metric="checkout conversion", from_value="13%", to_value="30%"),
            F("number", metric="infrastructure spend", value="1.5M"),
            F("status_change", object="checkout conversion",
              from_value="13%", to_value="30%"),
        ],
        "questions": [
            {"question": "What is the current checkout conversion rate?", "answer": "30%"},
            {"question": "What was the infrastructure spend for the quarter?", "answer": "1.5M"},
        ],
    },
    {
        "id": "mn_similar_names",
        "tags": ["similar_names", "multiple_people", "owner"],
        "text": (
            "Staffing sync, 2 February.\n"
            "Alice owns the search reindex. Alicia owns the billing export.\n"
            "They report status on alternate weeks."
        ),
        "facts": [
            F("person", value="Alice"), F("person", value="Alicia"),
            F("owner", person="Alice", object="search reindex"),
            F("owner", person="Alicia", object="billing export"),
        ],
        "questions": [
            {"question": "Who owns the search reindex?", "answer": "Alice"},
            {"question": "Who owns the billing export?", "answer": "Alicia"},
        ],
    },
    {
        "id": "mn_blocker_dependency",
        "tags": ["blocker", "dependency", "project"],
        "text": (
            "Delivery standup, 17 July.\n"
            "The data export is blocked on the schema migration.\n"
            "Miguel is unblocked and continues on the search reindex."
        ),
        "facts": [
            F("person", value="Miguel"),
            F("blocker", object="data export", blocked_on="schema migration"),
            F("dependency", object="data export", depends_on="schema migration"),
            F("project", value="search reindex"),
        ],
        "questions": [
            {"question": "What is the data export blocked on?", "answer": "schema migration"},
            {"question": "Is Miguel blocked? Answer YES or NO.", "answer": "NO"},
        ],
    },
    {
        "id": "mn_conditional",
        "tags": ["conditional", "decision", "deadline"],
        "text": (
            "Go/no-go, 8 October.\n"
            "If load testing passes by Wednesday, the team ships on Friday.\n"
            "Otherwise the release slips to the following sprint."
        ),
        "facts": [
            F("conditional", condition="load testing passes by Wednesday",
              consequence="ships on Friday"),
            F("deadline", object="load testing", value="Wednesday"),
            F("decision", value="ships on Friday"),
        ],
        "questions": [
            {"question": "On what condition does the team ship on Friday?",
             "answer": "load testing passes"},
        ],
    },
    {
        "id": "mn_pronoun_attribution",
        "tags": ["pronouns", "attribution", "multiple_people"],
        "text": (
            "Vendor call, 14 January.\n"
            "Sarah said John would finish the contract review.\n"
            "She will send the summary once he is done."
        ),
        "facts": [
            F("person", value="Sarah"), F("person", value="John"),
            F("action_item", person="John", action="finish", object="contract review"),
            F("action_item", person="Sarah", action="send", object="summary"),
            F("attribution", speaker="Sarah", about="John"),
        ],
        "questions": [
            {"question": "Who will finish the contract review?", "answer": "John"},
            {"question": "Who will send the summary?", "answer": "Sarah"},
        ],
    },
    {
        "id": "mn_relative_date",
        "tags": ["relative_dates", "deadline"],
        "text": (
            "Ops review, 3 June.\n"
            "The audit is due next Thursday, not this Thursday.\n"
            "Nadia will circulate the checklist beforehand."
        ),
        "facts": [
            F("person", value="Nadia"),
            F("deadline", object="audit", value="next Thursday"),
            F("negation", statement="not this Thursday"),
        ],
        "questions": [
            {"question": "When is the audit due?", "answer": "next Thursday"},
        ],
    },
    {
        "id": "mn_conflicting",
        "tags": ["conflicting_statements", "status_change"],
        "text": (
            "Incident debrief, 30 June.\n"
            "Initially the team reported the outage lasted 13 minutes.\n"
            "The final report corrected this to 30 minutes."
        ),
        "facts": [
            F("number", metric="outage duration initial", value="13 minutes"),
            F("number", metric="outage duration final", value="30 minutes"),
            F("status_change", object="outage",
              from_value="13 minutes", to_value="30 minutes"),
        ],
        "questions": [
            {"question": "What was the final reported outage duration?", "answer": "30 minutes"},
        ],
    },
    {
        "id": "mn_org_agreement",
        "tags": ["organizations", "agreement", "decision"],
        "text": (
            "Partnership sync, 11 September.\n"
            "Acme and Northwind agreed to a joint pilot starting in Q4.\n"
            "Legal sign-off is owned by Acme."
        ),
        "facts": [
            F("organization", value="Acme"), F("organization", value="Northwind"),
            F("agreement", between="Acme", and_="Northwind", topic="joint pilot"),
            F("decision", value="joint pilot"),
            F("owner", person="Acme", object="legal sign-off"),
        ],
        "questions": [
            {"question": "Which two organizations agreed to the joint pilot?",
             "answer": "Acme and Northwind"},
            {"question": "Who owns legal sign-off?", "answer": "Acme"},
        ],
    },
    {
        "id": "mn_project_ab",
        "tags": ["project_names", "numbers", "owner"],
        "text": (
            "Portfolio review, 25 August.\n"
            "Project A is 40% complete and owned by Rafael.\n"
            "Project B is 4% complete and owned by Hana."
        ),
        "facts": [
            F("person", value="Rafael"), F("person", value="Hana"),
            F("project", value="Project A"), F("project", value="Project B"),
            F("percentage", metric="Project A completion", value="40%"),
            F("percentage", metric="Project B completion", value="4%"),
            F("owner", person="Rafael", object="Project A"),
            F("owner", person="Hana", object="Project B"),
        ],
        "questions": [
            {"question": "Who owns Project B?", "answer": "Hana"},
            {"question": "What percent complete is Project A?", "answer": "40%"},
        ],
    },
]


# --------------------------------------------------------------------------
# Adversarial minimal pairs (Section 31). Each pair differs minimally and the
# correct answers MUST differ; a representation that collapses them is corrupt.
# --------------------------------------------------------------------------

STRESS_PAIRS: list[dict] = [
    {
        "id": "st_attribution",
        "kind": "attribution",
        "a": "Sarah said John would finish it.",
        "b": "Sarah would finish it, not John.",
        "question": "Who will finish it? Return only the name.",
        "answer_a": "John",
        "answer_b": "Sarah",
    },
    {
        "id": "st_deadline_move",
        "kind": "negation",
        "a": "The deadline moved from Thursday to Friday.",
        "b": "The deadline did not move from Thursday to Friday.",
        "question": "Is the deadline now Friday? Answer YES or NO.",
        "answer_a": "YES",
        "answer_b": "NO",
    },
    {
        "id": "st_agreement",
        "kind": "negation",
        "a": "John agreed with Sarah.",
        "b": "John did not agree with Sarah.",
        "question": "Did John agree with Sarah? Answer YES or NO.",
        "answer_a": "YES",
        "answer_b": "NO",
    },
    {
        "id": "st_13_30",
        "kind": "number",
        "a": "The batch contains 13 samples.",
        "b": "The batch contains 30 samples.",
        "question": "How many samples does the batch contain? Return only the number.",
        "answer_a": "13",
        "answer_b": "30",
    },
    {
        "id": "st_money",
        "kind": "number",
        "a": "The contract is worth $1.5M.",
        "b": "The contract is worth $15M.",
        "question": "What is the contract worth? Return only the amount.",
        "answer_a": "$1.5M",
        "answer_b": "$15M",
    },
    {
        "id": "st_june",
        "kind": "date",
        "a": "The filing is due June 3.",
        "b": "The filing is due June 30.",
        "question": "When is the filing due? Return only the date.",
        "answer_a": "June 3",
        "answer_b": "June 30",
    },
    {
        "id": "st_thursday",
        "kind": "relative_date",
        "a": "The audit is due Thursday.",
        "b": "The audit is due next Thursday.",
        "question": "When is the audit due? Return only the day.",
        "answer_a": "Thursday",
        "answer_b": "next Thursday",
    },
    {
        "id": "st_percent",
        "kind": "percentage",
        "a": "Coverage increased to 10%.",
        "b": "Coverage increased to 100%.",
        "question": "What is the coverage now? Return only the percentage.",
        "answer_a": "10%",
        "answer_b": "100%",
    },
    {
        "id": "st_names",
        "kind": "similar_names",
        "a": "Alice approved the budget.",
        "b": "Alicia approved the budget.",
        "question": "Who approved the budget? Return only the name.",
        "answer_a": "Alice",
        "answer_b": "Alicia",
    },
    {
        "id": "st_projects",
        "kind": "project_names",
        "a": "Project A shipped on time.",
        "b": "Project B shipped on time.",
        "question": "Which project shipped on time? Return only the project name.",
        "answer_a": "Project A",
        "answer_b": "Project B",
    },
    {
        "id": "st_owner_swap",
        "kind": "owner",
        "a": "Dan owns the migration; Priya owns the rollback.",
        "b": "Priya owns the migration; Dan owns the rollback.",
        "question": "Who owns the migration? Return only the name.",
        "answer_a": "Dan",
        "answer_b": "Priya",
    },
    {
        "id": "st_blocker",
        "kind": "negation",
        "a": "The export is blocked on the migration.",
        "b": "The export is not blocked on the migration.",
        "question": "Is the export blocked on the migration? Answer YES or NO.",
        "answer_a": "YES",
        "answer_b": "NO",
    },
    {
        "id": "st_conditional",
        "kind": "conditional",
        "a": "We ship on Friday if testing passes.",
        "b": "We ship on Friday only if testing passes.",
        "question": "Does testing passing guarantee a Friday ship? Answer YES or NO.",
        "answer_a": "YES",
        "answer_b": "NO",
    },
    {
        "id": "st_quantifier",
        "kind": "quantifier",
        "a": "At least 3 reviewers approved.",
        "b": "More than 3 reviewers approved.",
        "question": "Could exactly 3 reviewers have approved? Answer YES or NO.",
        "answer_a": "YES",
        "answer_b": "NO",
    },
]


def build_stress_cases() -> list[dict]:
    """Expand each minimal pair into two independent cases."""
    out = []
    for p in STRESS_PAIRS:
        for side in ("a", "b"):
            out.append({
                "id": f"{p['id']}_{side}",
                "pair_id": p["id"],
                "kind": p["kind"],
                "text": p[side],
                "question": p["question"],
                "answer": p[f"answer_{side}"],
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="meeting_notes.jsonl")
    ap.add_argument("--stress", default="stress_tests.jsonl")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = here / args.out
    with out.open("w", encoding="utf-8") as fh:
        for m in MEETINGS:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    stress = build_stress_cases()
    sp = here / args.stress
    with sp.open("w", encoding="utf-8") as fh:
        for c in stress:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    tags = sorted({t for m in MEETINGS for t in m["tags"]})
    print(f"meetings   : {len(MEETINGS)}  -> {out.name}")
    print(f"  facts    : {sum(len(m['facts']) for m in MEETINGS)}")
    print(f"  questions: {sum(len(m['questions']) for m in MEETINGS)}")
    print(f"  tags     : {tags}")
    print(f"stress     : {len(stress)} cases from {len(STRESS_PAIRS)} minimal pairs -> {sp.name}")
    print(f"  kinds    : {sorted({p['kind'] for p in STRESS_PAIRS})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
