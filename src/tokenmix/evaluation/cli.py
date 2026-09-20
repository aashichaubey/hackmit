from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_cases, load_config, load_dotenv
from .reporting import write_report
from .runner import prepare_manifest, run
from .scoring import load_scorer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paired, independently graded character-compression evaluations")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--max-cases", type=int)
        command.add_argument("--control", choices=["original_vs_compressed", "original_vs_original"])
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--env-file", type=Path, default=Path(".env"))
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--check-gate", action="store_true", help="Exit 1 for FAIL, 3 for INCONCLUSIVE, 4 for NOT_CONFIGURED")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            result = write_report(args.directory)
            print(json.dumps({"gate": result["gate"], "coverage": result["coverage"]}, indent=2))
            return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3, "NOT_CONFIGURED": 4}[result["gate"]["state"]] if args.check_gate else 0
        config = load_config(args.config.resolve())
        if args.max_cases is not None:
            if args.max_cases < 1:
                raise ValueError("--max-cases must be positive")
            config["max_cases"] = args.max_cases
        if args.control:
            config["experiment"] = args.control
        cases = load_cases(Path(config["fixture_path"]))
        load_scorer(Path(config["scorer"]["path"]))
        manifest = prepare_manifest(config, cases)
        selected_count = min(config["max_cases"] or len(cases), len(cases))
        if args.command == "validate":
            print(json.dumps({"valid": True, "fixture_count": len(cases), "selected_cases": selected_count,
                              "categories": sorted({c["category"] for c in cases}), "mode": config["mode"],
                              "first_attempt_target_calls": selected_count * config["repeats"] * 2,
                              "live_call_limit": config["max_live_calls"], "experiment_id": manifest["experiment_id"]}, indent=2))
        else:
            if config["mode"] == "live":
                load_dotenv(args.env_file)
            output = run(config, resume=args.resume, progress=lambda line: print(line, flush=True))
            print(f"Artifacts: {output}")
        return 0
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"tokenmix-eval: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
