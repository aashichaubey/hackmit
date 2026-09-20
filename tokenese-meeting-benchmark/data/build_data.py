import json
from pathlib import Path

root = Path(__file__).parent
names = ["David", "John", "Maya", "Lena", "Omar", "Priya", "Alex", "Nina", "Sam", "Iris", "Leo", "Ana", "Ben", "Tara", "Noah", "Eva", "Kai", "Ruth", "Jules", "Amir", "Kim", "Zoe", "Paul", "Uma", "Luis", "Jo", "Mina", "Raj", "Ella", "Theo"]
tasks = ["finish the API", "review the budget", "update the website", "prepare the slides", "test the mobile app", "draft the proposal", "check the database", "send the invitations", "fix the login flow", "write the release notes", "audit the dashboard", "plan the workshop", "verify the metrics", "design the landing page", "book the venue", "ship the prototype", "clean the dataset", "run the survey", "approve the copy", "schedule the demo", "review the contract", "publish the guide", "test the payment flow", "update the roadmap", "prepare the report", "inspect the logs", "finish the integration", "draft the email", "review the mockups", "set up the server"]
dates = ["Thursday", "Friday", "Monday", "Tuesday", "Wednesday", "September 25", "next Friday", "October 2", "tomorrow", "end of month"]
meetings = []
for i in range(30):
    owner = names[i]
    other = names[(i + 7) % 30]
    task = tasks[i]
    due = dates[i % len(dates)]
    status = "proposed" if i % 2 == 0 else "decided"
    reaction = "agreed" if i % 3 else "disagreed"
    topic = f"launch plan {i + 1}"
    lines = [f"{other} {status} the {topic}.", f"{owner} will {task} by {due}.", f"{names[(i + 11) % 30]} {reaction} with the {topic}."]
    notes = " ".join(lines)
    facts = [
        {"kind": "proposal" if status == "proposed" else "decision", "person": other if status == "proposed" else "", "text": f"the {topic}", "deadline": "", "source_quote": lines[0]},
        {"kind": "action", "person": owner, "text": task, "deadline": due, "source_quote": lines[1]},
        {"kind": "agreement" if reaction == "agreed" else "disagreement", "person": names[(i + 11) % 30], "text": f"the {topic}", "deadline": "", "source_quote": lines[2]},
    ]
    questions = [
        {"question": f"Who owns {task}?", "answers": [owner], "kind": "owner"},
        {"question": f"When is {task} due?", "answers": [due], "kind": "deadline"},
        {"question": f"Was the {topic} decided?", "answers": ["yes"] if status == "decided" else ["not found"], "kind": "decision" if status == "decided" else "unknown"},
        {"question": f"Who {reaction} with the {topic}?", "answers": [names[(i + 11) % 30]], "kind": "agreement" if reaction == "agreed" else "disagreement"},
    ]
    split = "dev" if i < 18 else "validation" if i < 24 else "test"
    meetings.append({"id": f"meeting-{i + 1:02}", "split": split, "notes": notes, "facts": facts, "questions": questions})
def paired_case(index, owner, due, reaction, status):
    task = "finish the API"
    topic = "launch plan"
    first = f"Sarah {status} the {topic}."
    second = f"{owner} will {task} by {due}."
    third = f"Maya {reaction} with the {topic}."
    case = meetings[index]
    case["notes"] = " ".join((first, second, third))
    case["facts"] = [
        {"kind": "proposal" if status == "proposed" else "decision", "person": "Sarah" if status == "proposed" else "", "text": f"the {topic}", "deadline": "", "source_quote": first},
        {"kind": "action", "person": owner, "text": task, "deadline": due, "source_quote": second},
        {"kind": "agreement" if reaction == "agreed" else "disagreement", "person": "Maya", "text": f"the {topic}", "deadline": "", "source_quote": third},
    ]
    case["questions"] = [
        {"question": "Who owns the API?", "answers": [owner], "kind": "owner"},
        {"question": "When is the API due?", "answers": [due], "kind": "deadline"},
        {"question": "Was the launch plan decided?", "answers": ["yes"] if status == "decided" else ["not found"], "kind": "decision" if status == "decided" else "unknown"},
        {"question": f"Who {reaction} with the launch plan?", "answers": ["Maya"], "kind": "agreement" if reaction == "agreed" else "disagreement"},
    ]

for first, second, a, b in [
    (0, 1, ("David", "Thursday", "agreed", "proposed"), ("John", "Thursday", "agreed", "proposed")),
    (2, 3, ("David", "Thursday", "agreed", "proposed"), ("David", "Friday", "agreed", "proposed")),
    (4, 5, ("David", "Thursday", "agreed", "proposed"), ("David", "Thursday", "disagreed", "proposed")),
    (6, 7, ("David", "Thursday", "agreed", "proposed"), ("David", "Thursday", "agreed", "decided")),
]:
    paired_case(first, *a)
    paired_case(second, *b)

missing = meetings[8]
missing["notes"] = "Nina will review the budget. No deadline was set. Omar proposed the audit."
missing["facts"] = [
    {"kind": "action", "person": "Nina", "text": "review the budget", "deadline": "", "source_quote": "Nina will review the budget."},
    {"kind": "proposal", "person": "Omar", "text": "the audit", "deadline": "", "source_quote": "Omar proposed the audit."},
]
missing["questions"] = [
    {"question": "Who will review the budget?", "answers": ["Nina"], "kind": "owner"},
    {"question": "When is the budget review due?", "answers": ["not found"], "kind": "unknown"},
    {"question": "Was the audit decided?", "answers": ["not found"], "kind": "unknown"},
    {"question": "Who proposed the audit?", "answers": ["Omar"], "kind": "proposal"},
]
ambiguous = meetings[9]
ambiguous["notes"] = "Alex may take the migration, but nobody confirmed an owner. Priya decided to run the demo."
ambiguous["facts"] = [{"kind": "decision", "person": "", "text": "run the demo", "deadline": "", "source_quote": "Priya decided to run the demo."}]
ambiguous["questions"] = [
    {"question": "Who owns the migration?", "answers": ["not found"], "kind": "unknown"},
    {"question": "Was the demo decided?", "answers": ["yes"], "kind": "decision"},
    {"question": "When is the demo due?", "answers": ["not found"], "kind": "unknown"},
    {"question": "What was decided?", "answers": ["run the demo"], "kind": "decision"},
]

(root / "meetings.json").write_text(json.dumps({"version": "1", "meetings": meetings}, indent=2) + "\n")
pairs = []
for i in range(40):
    field = ["owner", "deadline", "negation", "decision"][i % 4]
    person = names[i % len(names)]
    other = names[(i + 1) % len(names)]
    due = dates[i % len(dates)]
    other_due = dates[(i + 1) % len(dates)]
    topic = f"project {i + 1}"
    original = {"owner": f"{person} owns {topic}", "deadline": f"{topic} is due {due}", "negation": f"{person} disagreed with {topic}", "decision": f"{person} proposed {topic}"}[field]
    corrupted = {"owner": f"{other} owns {topic}", "deadline": f"{topic} is due {other_due}", "negation": f"{person} agreed with {topic}", "decision": f"{person} decided {topic}"}[field]
    preserved = i % 2 == 0
    pairs.append({"id": f"pilot-{i + 1:02}", "field": field, "original": original, "encoded": original if preserved else corrupted, "preserved": preserved, "false_accept_if_preserved": not preserved})
(root / "judge_pilot.json").write_text(json.dumps({"version": "1", "pairs": pairs}, indent=2) + "\n")
