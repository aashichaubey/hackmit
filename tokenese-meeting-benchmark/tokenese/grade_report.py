import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .benchmark import ROOT, load_cases
from .cache import CachedClient
from .judges import judge_answer

def run():
    load_dotenv(ROOT / ".env")
    path = ROOT / "reports" / "latest.json"
    report = json.loads(path.read_text())
    cases = {case["id"]: case for case in load_cases()}
    client = CachedClient(OpenAI(), ROOT / "reports" / "judge_cache.json")
    reviewed = {}
    for result in report["runs"]:
        gold_facts = cases[result["id"]]["facts"]
        for row in result["questions"]:
            for record in row["methods"].values():
                answer = record.get("answer")
                if record.get("correct") or not answer or not answer["found"] or row["answers"] == ["not found"]:
                    continue
                key = (result["id"], row["question"], answer["answer"])
                if key not in reviewed:
                    reviewed[key] = judge_answer(client, gold_facts, row["question"], answer["answer"])
                record["judge"] = reviewed[key]
    verdicts = {name: sum(item["verdict"] == name for item in reviewed.values()) for name in ("correct", "incorrect", "uncertain")}
    report["judge_diagnostics"] = {"unique_nonexact_answers": len(reviewed), "verdicts": verdicts, "manual_review": [{"meeting": key[0], "question": key[1], "answer": key[2], "verdict": value["verdict"], "reason": value["reason"]} for key, value in reviewed.items() if value["verdict"] != "incorrect"]}
    human_reviews = json.loads((ROOT / "data" / "judge_review.json").read_text())["reviews"]
    for item in report["judge_diagnostics"]["manual_review"]:
        review = next((value for value in human_reviews if value["meeting"] == item["meeting"] and value["question"] == item["question"]), None)
        if review:
            item["human_verdict"] = review["human_verdict"]
            item["human_reason"] = review["reason"]
    report["judge_diagnostics"]["critical_false_accepts"] = sum(item.get("human_verdict") == "incorrect" and item["verdict"] == "correct" for item in report["judge_diagnostics"]["manual_review"])
    decisions = {(item["meeting"], item["question"]): item["human_verdict"] for item in human_reviews}
    adjudicated = {}
    for result in report["runs"]:
        bucket = adjudicated.setdefault(result["fact_source"], {}).setdefault(result["split"], {})
        for row in result["questions"]:
            for method, record in row["methods"].items():
                reviewed_correct = record.get("judge", {}).get("verdict") == "correct" and decisions.get((result["id"], row["question"])) == "correct"
                record["adjudicated_correct"] = bool(record["correct"] or reviewed_correct)
                counts = bucket.setdefault(method, {"correct": 0, "questions": 0})
                counts["correct"] += record["adjudicated_correct"]
                counts["questions"] += 1
    report["adjudicated_summary"] = adjudicated
    usage = report.setdefault("research_usage", {})
    usage["grader_calls"] = len(reviewed)
    usage["grader_input_tokens"] = sum(item["usage"]["input_tokens"] for item in reviewed.values() if item["usage"])
    usage["grader_output_tokens"] = sum(item["usage"]["output_tokens"] for item in reviewed.values() if item["usage"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["judge_diagnostics"], indent=2))

if __name__ == "__main__":
    run()
