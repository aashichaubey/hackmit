"""Generate the fixed three-context, five-question benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "examples.json"

EXAMPLES = [
    {
        "id": 1,
        "category": "academic_notes",
        "context": (
            "Professor Kim's Tuesday cell-biology lecture opened with a reminder that the optional microscopy workshop will be held Friday in Room 214. She explained that mitochondria produce ATP, the usable energy that powers many cellular processes. Unlike mitochondria, ribosomes assemble proteins by translating messenger RNA. The class then compared plant and animal cells: plant cells have cellulose cell walls and chloroplasts, whereas animal cells have neither feature. Kim noted that mature human red blood cells lack nuclei, an adaptation that leaves more space for hemoglobin. During a historical aside, she credited Robert Hooke with publishing observations of cork cells in 1665. The lecture also distinguished passive and active transport. Passive transport moves substances down a concentration gradient without direct energy expenditure; active transport can move them against a gradient and requires energy. The sodium-potassium pump was given as an example of active transport. Kim ended by assigning chapter 7 for Monday and moving her Wednesday office hours from 2:00 p.m. to 3:30 p.m. because of a faculty meeting."
        ),
        "questions": [
            {"question": "What do ribosomes assemble?", "expected_answer": "proteins"},
            {"question": "Which two features mentioned in the lecture do plant cells have that animal cells lack?", "expected_answer": "cellulose cell walls and chloroplasts"},
            {"question": "Why do mature human red blood cells lack nuclei?", "expected_answer": "to leave more space for hemoglobin"},
            {"question": "Who published observations of cork cells in 1665?", "expected_answer": "Robert Hooke"},
            {"question": "What time were Wednesday office hours moved to?", "expected_answer": "3:30 p.m."},
        ],
    },
    {
        "id": 2,
        "category": "policy_rules",
        "context": (
            "The North Campus laboratory handbook applies to the wet lab, machine shop, greenhouse, and instrument room. Trained researchers may enter the wet lab by scanning an active badge. Visitors without safety training may enter only when escorted by a certified technician, who must stay with them until they leave. Food and drinks are prohibited in all experimental rooms but may be stored in the second-floor lounge. Chemical deliveries arrive at loading dock C and must be logged by a staff member before storage. Reference samples normally cannot leave the building; a principal investigator may authorize an overnight loan by signing form RS-4. After three failed badge scans, access is disabled for twenty minutes. Any spill larger than 500 milliliters requires evacuation and a call to campus emergency services, while smaller spills may be handled by trained staff using the appropriate kit. The eyewash stations are tested every Monday. Incident reports must be submitted within 24 hours. The handbook's annual review occurs in March, and outdated printed copies should be placed in secure recycling rather than ordinary bins."
        ),
        "questions": [
            {"question": "Where may food and drinks be stored?", "expected_answer": "the second-floor lounge"},
            {"question": "Where do chemical deliveries arrive?", "expected_answer": "loading dock C"},
            {"question": "What form must a principal investigator sign to authorize an overnight reference-sample loan?", "expected_answer": "form RS-4"},
            {"question": "How long is access disabled after three failed badge scans?", "expected_answer": "twenty minutes"},
            {"question": "What spill size requires evacuation and a call to campus emergency services?", "expected_answer": "a spill larger than 500 milliliters"},
        ],
    },
    {
        "id": 3,
        "category": "schedule_and_facts",
        "context": (
            "Saturday's research conference begins with registration at 8:00 a.m. in the student center. The data-visualization tutorial starts at 9:00 in Room 32-123, while the hardware-prototyping tutorial meets at the same time in Maker Studio B. The machine-learning keynote begins at 10:30 a.m. in Kresge Auditorium; doors open fifteen minutes earlier. Lunch is served at noon in Walker Memorial, with vegetarian meals marked by green cards. The poster session runs from 1:15 to 2:45 p.m. in Johnson Track and includes 84 posters. Because of an equipment issue, the robotics demonstration was moved from Building 26 to the Stata Center and delayed until 3:20 p.m. Shuttle buses to the evening reception depart from Massachusetts Avenue every twenty minutes beginning at 5:40 p.m. The reception itself starts at 6:00 in the MIT Museum. Attendees who lose a badge should visit the welcome desk, but lost physical items are held at campus police until Monday. The best-poster award, a $500 equipment grant, will be announced during the reception at 7:15 p.m."
        ),
        "questions": [
            {"question": "Where is the hardware-prototyping tutorial held?", "expected_answer": "Maker Studio B"},
            {"question": "How many posters are included in the poster session?", "expected_answer": "84"},
            {"question": "Where was the robotics demonstration moved?", "expected_answer": "the Stata Center"},
            {"question": "When do shuttle buses begin departing for the reception?", "expected_answer": "5:40 p.m."},
            {"question": "What is the best-poster award?", "expected_answer": "a $500 equipment grant"},
        ],
    },
]


def validate_examples(examples: list[dict]) -> None:
    assert len(examples) == 3
    for example in examples:
        assert set(example) == {"id", "category", "context", "questions"}
        assert len(example["questions"]) == 5
        for item in example["questions"]:
            assert set(item) == {"question", "expected_answer"}
            assert item["question"] and item["expected_answer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    validate_examples(EXAMPLES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(EXAMPLES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(EXAMPLES)} contexts × 5 questions to {args.output}")


if __name__ == "__main__":
    main()
