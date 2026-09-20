import json
from pathlib import Path
from tokenese.benchmark import ROOT, summarize
from tokenese.encode import tokenese_legend
from tokenese.prompts import answer_prompt
from tokenese.tokens import count_tokens


def test_report_prompts_and_usage():
    report = json.loads((ROOT / "reports" / "latest.json").read_text())
    assert report["status"] == "API benchmark complete"
    assert len(report["runs"]) == 60
    for run in report["runs"]:
        assert len(run["questions"]) == 4
        for row in run["questions"]:
            for name, record in row["methods"].items():
                assert record["error"] is None
                legend = tokenese_legend(name) if name in ("symbols", "mixed") else ""
                assert record["prompt"] == answer_prompt(record["context"], row["question"], legend)
                assert record["full_input_tokens"] == count_tokens(record["prompt"], report["model"])
                assert record["context_tokens"] == count_tokens(record["context"], report["model"])
    for source in ("gold", "parsed"):
        for split in ("dev", "validation", "test"):
            runs = [run for run in report["runs"] if run["fact_source"] == source and run["split"] == split]
            assert summarize(runs) == report["summary"][source][split]


def test_stress_report_usage():
    report = json.loads((ROOT / "reports" / "stress.json").read_text())
    cases = json.loads((ROOT / "data" / "stress_meetings.json").read_text())["meetings"]
    assert len(report["runs"]) == 12
    for source in ("gold", "parsed"):
        runs = [run for run in report["runs"] if run["fact_source"] == source]
        assert summarize(runs, cases) == report["summary"][source]
