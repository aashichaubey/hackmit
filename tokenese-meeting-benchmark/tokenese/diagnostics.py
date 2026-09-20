import json
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
from openai import OpenAI
from .benchmark import ROOT, load_cases
from .encode import render_english, render_tokenese
from .facts import MeetingFacts
from .judges import cosine_similarity, embeddings

def run():
    load_dotenv(ROOT / ".env")
    report_path = ROOT / "reports" / "latest.json"
    report = json.loads(report_path.read_text())
    pairs = []
    texts = []
    for case in load_cases():
        facts = MeetingFacts(facts=case["facts"])
        english = render_english(facts).splitlines()
        for method in ("symbols", "mixed"):
            encoded = render_tokenese(facts, method).splitlines()
            for left, right in zip(english, encoded):
                pairs.append((case["split"], method, left, right))
                texts.extend((left, right))
    pilot = json.loads((ROOT / "data" / "judge_pilot.json").read_text())["pairs"]
    for item in pilot:
        texts.extend((item["original"], item["encoded"]))
    vectors, usage = embeddings(OpenAI(), texts, ROOT / "reports" / "embedding_cache.json")
    results = {}
    for split in ("dev", "validation", "test"):
        results[split] = {}
        for method in ("symbols", "mixed"):
            values = [cosine_similarity(vectors[left], vectors[right]) for row_split, row_method, left, right in pairs if row_split == split and row_method == method]
            results[split][method] = {"fact_pairs": len(values), "mean_cosine": mean(values)}
    pilot_scores = [(item["preserved"], cosine_similarity(vectors[item["original"]], vectors[item["encoded"]])) for item in pilot]
    results["pilot"] = {"preserved_mean_cosine": mean(value for preserved, value in pilot_scores if preserved), "corrupted_mean_cosine": mean(value for preserved, value in pilot_scores if not preserved), "corrupted_above_0_9": sum(not preserved and value >= 0.9 for preserved, value in pilot_scores), "corrupted_pairs": sum(not preserved for preserved, _ in pilot_scores)}
    report["embedding_diagnostics"] = results
    report["research_usage"]["embedding_calls"] = usage["calls"]
    report["research_usage"]["embedding_tokens"] = usage["tokens"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"diagnostics": results, "usage": usage}, indent=2))

if __name__ == "__main__":
    run()
