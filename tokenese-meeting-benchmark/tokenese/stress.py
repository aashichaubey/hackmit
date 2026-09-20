import json
from dotenv import load_dotenv
from openai import OpenAI
from . import MODEL
from .baseline import DeletionBaseline
from .bear import BearBaseline
from .benchmark import ROOT, run_case, summarize
from .cache import CachedClient

def run():
    load_dotenv(ROOT / ".env")
    cases = json.loads((ROOT / "data" / "stress_meetings.json").read_text())["meetings"]
    client = CachedClient(OpenAI(), ROOT / "reports" / "cache.json")
    baselines = {"deletion": DeletionBaseline(), "bear-2": BearBaseline()}
    methods = ("raw", "english", "symbols", "mixed", "deletion", "bear-2")
    runs = [run_case(client, case, model=MODEL, fact_source=source, methods=methods, deletion=baselines) for source in ("gold", "parsed") for case in cases]
    report = {"status": "supplemental stress run", "model": MODEL, "dataset_version": "1", "runs": runs, "summary": {source: summarize([item for item in runs if item["fact_source"] == source], cases) for source in ("gold", "parsed")}}
    (ROOT / "reports" / "stress.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    run()
