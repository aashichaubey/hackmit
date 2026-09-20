"""Generate separate development probes; never change the supplied fixtures."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenmix.core import load_dictionary

from .storage import file_hash, write_jsonl


def build_cases(dictionary: Path) -> list[dict]:
    entries = load_dictionary(dictionary)
    version = file_hash(dictionary)
    cases = []

    def add(case_id, cluster, category, prompt, expected, tags=()):
        cases.append({"case_id": case_id, "cluster_id": cluster, "category": category,
                      "split": "development_dictionary", "source": "Original synthetic dictionary probe; independent expected outputs",
                      "source_group": "dictionary-development", "dictionary_sha256": version,
                      "messages": [{"role": "system", "content": "Use only the supplied facts and follow the requested output format."},
                                   {"role": "user", "content": prompt}],
                      "expected_output": expected, "grader": {"type": "exact", "strip_outer_whitespace": False},
                      "tags": list(tags)})

    for entry in entries:
        if not entry.enabled:
            continue
        tags = ["entry:" + entry.id]
        probes = [
            ("subject", "grounding", f"The topic {entry.en} has label CEDAR. Return only its label.", "CEDAR"),
            ("object", "grounding", f"Mira studies {entry.en}. Who studies that topic? Return only the name.", "Mira"),
            ("negated", "negation", f"Mira studies {entry.en}; Noro does not. Who does not study the topic? Return only the name.", "Noro"),
            ("chinese_name", "literal_fidelity", f"A person's name is {entry.zh}. Return that exact name and nothing else.", entry.zh),
            ("quote", "literal_fidelity", f'Copy exactly the text between double quotes, omitting the quotes: "{entry.en}"', entry.en),
            ("chinese_quote", "literal_fidelity", f'Copy exactly the text between double quotes, omitting the quotes: "{entry.zh}"', entry.zh),
            ("identifier", "literal_fidelity", f"Return this exact identifier: `record_{entry.id}_人工智能`", f"record_{entry.id}_人工智能"),
            ("high_density", "grounding", ((f"The subject is {entry.en}. " * 20) + "The subject's label is PINE. Return only that label."), "PINE"),
            ("cold_context", "grounding", f"<source>The topic is {entry.zh}. Its label is MAPLE.</source> Return only the topic's label.", "MAPLE"),
        ]
        for name, category, prompt, expected in probes:
            add(f"dict_{entry.id}_{name}", "dict_entry_" + entry.id, category, prompt, expected, [*tags, name])
    pairs = [
        ("if", "conditional_payment", "conditionals", "A parcel ships if payment clears. Payment cleared. Must it have shipped? Answer only YES or NO.", "YES"),
        ("only_if", "conditional_payment", "conditionals", "A parcel ships only if payment clears. Payment cleared. Must it have shipped? Answer only YES or NO.", "NO"),
        ("at_least", "threshold", "quantifiers", "Approve if there are at least three votes. There are three votes. Reply only APPROVE or REJECT.", "APPROVE"),
        ("more_than", "threshold", "quantifiers", "Approve if there are more than three votes. There are three votes. Reply only APPROVE or REJECT.", "REJECT"),
        ("all", "lamps", "negation", "All three lamps are on. Is any lamp off? Answer only YES or NO.", "NO"),
        ("not_all", "lamps", "negation", "Not all three lamps are on. Each lamp is either on or off. Is any lamp off? Answer only YES or NO.", "YES"),
        ("may", "attendance", "conditionals", "The rule says staff may attend. Does this rule require attendance? Answer only YES or NO.", "NO"),
        ("must", "attendance", "conditionals", "The rule says staff must attend. Does this rule require attendance? Answer only YES or NO.", "YES"),
        ("transfer_forward", "transfer", "reasoning", "Nora hands a token to Tavi. Who has the token after this transfer? Return only the name.", "Tavi"),
        ("transfer_reverse", "transfer", "reasoning", "Tavi hands a token to Nora. Who has the token after this transfer? Return only the name.", "Nora"),
        ("number_a", "numeric_twins", "reasoning", "Mira owns 8 beads and gives away 3. How many remain? Return only the number.", "5"),
        ("number_b", "numeric_twins", "reasoning", "Mira owns 9 beads and gives away 3. How many remain? Return only the number.", "6"),
        ("name_a", "name_twins", "grounding", "Mira owns the blue box. Return only the owner's name.", "Mira"),
        ("name_b", "name_twins", "grounding", "Tavi owns the blue box. Return only the owner's name.", "Tavi"),
        ("absent", "name_twins", "grounding", "Mira owns a box; its color is unstated. What color is it? Reply only UNKNOWN if unstated.", "UNKNOWN"),
        ("missing_mapping", "missing_mapping", "literal_fidelity", "Return exactly this identifier: zxv_unknown_phrase", "zxv_unknown_phrase"),
        ("unseen_combination", "unseen_combination", "grounding", "A report about artificial intelligence and an operating system is required as soon as possible. The report's owner is Lira. Return only its owner's name.", "Lira"),
    ]
    for case_id, cluster, category, prompt, expected in pairs:
        add("dev_" + case_id, "dev_" + cluster, category, prompt, expected, ["minimal_pair_or_contract"])
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new development-dataset version")
    rows = build_cases(args.dictionary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} synthetic development probes, not held-out release evidence, to {args.output}")


if __name__ == "__main__":
    main()
