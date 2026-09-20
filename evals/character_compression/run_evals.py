"""Paired original-vs-compressed evaluation runner.

    # validate fixtures only
    python run_evals.py validate --config config.example.json

    # offline mock smoke run (no credentials, no network)
    python run_evals.py run --config config.example.json

    # bounded live run
    python run_evals.py run --config config.live.json --max-live-calls 20

Design rules enforced here rather than left to convention:

* Gold answers are held by the runner for grading and never enter a payload.
* The compressor owns dictionary/decoder overhead; the runner never adds it.
* Arm execution order is randomized from a recorded seed.
* Each case/repeat is an independent conversation; nothing is cached or reused.
* A mock adapter in live mode is a hard error, never a silent substitution.
* Null measurements stay null (unknown); they never become zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapters import (  # noqa: E402
    COMPRESSORS,
    COUNTERS,
    TARGETS,
    CompressionResult,
    GenerationResult,
    GoldLeakageError,
    assert_no_gold_fields,
    sanitize_messages,
)
from score_evals import grade, load_jsonl  # noqa: E402

SCHEMA_VERSION = "1.0.0"

# Experiment modes.
MODE_FIRST_ATTEMPT = "first_attempt"  # no retries, no fallback (quality)
MODE_PIPELINE = "deployment_pipeline"  # retries + fallback (deployment policy)
MODE_CONTROL = "baseline_vs_baseline"  # variability control


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    mode: str = "mock"  # mock | live
    experiment_mode: str = MODE_FIRST_ATTEMPT
    fixtures: str = "eval_cases.jsonl"
    compressor: str = "mock"
    compressor_options: dict[str, Any] = field(default_factory=dict)
    target: str = "mock"
    target_options: dict[str, Any] = field(default_factory=dict)
    token_counter: str | None = "deepseek_local"
    token_counter_options: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 256})
    compression_config: dict[str, Any] = field(default_factory=lambda: {"scope": "user_only"})
    compression_scope: str = "user_only"
    repeats: int = 1
    seed: int = 17
    timeout_s: float = 60.0
    max_live_calls: int | None = None
    retry: dict[str, Any] = field(default_factory=lambda: {"max_retries": 0, "fallback_to_original": False})
    out_root: str = "runs"
    experiment_id: str | None = None
    pricing: dict[str, float] | None = None
    notes: str = ""

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def fingerprint(self) -> str:
        """Stable hash of the settings that make results incomparable if changed."""
        material = {
            "schema": SCHEMA_VERSION,
            "mode": self.mode,
            "experiment_mode": self.experiment_mode,
            "compressor": self.compressor,
            "compressor_options": self.compressor_options,
            "target": self.target,
            "target_options": self.target_options,
            "generation": self.generation,
            "compression_config": self.compression_config,
            "compression_scope": self.compression_scope,
            "repeats": self.repeats,
            "seed": self.seed,
            "retry": self.retry,
        }
        blob = json.dumps(material, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

REQUIRED_CASE_FIELDS = {"case_id", "category", "messages", "expected_output", "grader"}


def validate_fixtures(cases: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(cases):
        missing = REQUIRED_CASE_FIELDS - c.keys()
        if missing:
            problems.append(f"case {i}: missing {sorted(missing)}")
            continue
        if c["case_id"] in seen:
            problems.append(f"duplicate case_id: {c['case_id']}")
        seen.add(c["case_id"])
        if not isinstance(c["messages"], list) or not c["messages"]:
            problems.append(f"{c['case_id']}: messages must be a non-empty list")
            continue
        for m in c["messages"]:
            if set(m) != {"role", "content"}:
                problems.append(f"{c['case_id']}: message keys must be exactly role/content")
            if m.get("role") not in ("system", "user", "assistant"):
                problems.append(f"{c['case_id']}: bad role {m.get('role')!r}")
        if c["grader"].get("type") not in ("exact", "json"):
            problems.append(f"{c['case_id']}: unsupported grader {c['grader'].get('type')!r}")
    return problems


# ---------------------------------------------------------------------------
# Attempt logging
# ---------------------------------------------------------------------------


@dataclass
class Attempt:
    case_id: str
    run_id: int
    arm: str  # baseline | compressed
    attempt_index: int
    kind: str  # first_attempt | retry | fallback
    output: str
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    model_id: str
    token_count_source: str | None
    provider_request_id: str | None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    passed: int | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, config: Config, out_dir: Path, max_live_calls: int | None = None):
        self.cfg = config
        self.out_dir = out_dir
        self.live_budget = max_live_calls if max_live_calls is not None else config.max_live_calls
        self.live_calls = 0
        self.attempts: list[Attempt] = []

        self.counter = None
        if config.token_counter:
            self.counter = COUNTERS[config.token_counter](**config.token_counter_options)

        copts = dict(config.compressor_options)
        self.compressor = COMPRESSORS[config.compressor](**copts)

        topts = dict(config.target_options)
        if config.target == "mock" and self.counter is not None:
            topts.setdefault("counter", self.counter)
        if config.pricing and config.target != "mock":
            topts.setdefault("pricing", config.pricing)
        self.target = TARGETS[config.target](**topts)

        # Hard stop: a mock in a live run would fabricate compression.
        if config.mode == "live":
            mocked = [a.name for a in (self.compressor, self.target) if getattr(a, "is_mock", False)]
            if mocked:
                raise RuntimeError(
                    f"live mode refused: mock adapter(s) selected: {mocked}. "
                    "Configure real adapters, or set mode='mock' for an offline smoke run."
                )

    # -- token accounting ---------------------------------------------------

    def _count(self, messages: list[dict[str, str]], gen: GenerationResult) -> tuple[int | None, str | None]:
        """Prefer provider-reported usage; fall back to the local counter.

        Never invents a number: if neither is available the count stays None
        (unknown) and the scorer will report reduced measurement coverage.
        """
        if gen.input_tokens is not None:
            return gen.input_tokens, gen.token_count_source or "provider_usage"
        if self.counter is not None:
            tc = self.counter.count_request_tokens(messages)
            return tc.tokens, tc.source
        return None, None

    # -- one arm ------------------------------------------------------------

    def _call(self, messages: list[dict[str, str]], case_id: str, run_id: int, arm: str,
              attempt_index: int, kind: str) -> GenerationResult:
        if self.cfg.mode == "live":
            if self.live_budget is not None and self.live_calls >= self.live_budget:
                raise RuntimeError(
                    f"live call budget of {self.live_budget} exhausted at {case_id}/{run_id}/{arm}"
                )
            self.live_calls += 1

        assert_no_gold_fields(messages, f"{case_id}/{arm}")
        gen = self.target.generate(messages, self.cfg.generation)
        tokens, source = self._count(messages, gen)
        gen.input_tokens, gen.token_count_source = tokens, source

        self.attempts.append(Attempt(
            case_id=case_id, run_id=run_id, arm=arm, attempt_index=attempt_index, kind=kind,
            output=gen.output, error=gen.error, input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens, latency_ms=gen.latency_ms,
            model_id=gen.model_id, token_count_source=gen.token_count_source,
            provider_request_id=gen.provider_request_id,
            finish_reason=gen.finish_reason, reasoning_tokens=gen.reasoning_tokens,
        ))
        return gen

    def _run_arm_with_policy(self, messages, case, run_id, arm, original_messages):
        """Execute one arm, applying retry/fallback only in pipeline mode.

        Returns (first_attempt_result, final_result, fell_back).
        First-attempt quality is always preserved separately so that a
        fallback to the original prompt can never be reported as successful
        compression.
        """
        first = self._call(messages, case["case_id"], run_id, arm, 0, "first_attempt")
        first.passed = int(not first.failed and grade(case, first.output))

        if self.cfg.experiment_mode != MODE_PIPELINE:
            return first, first, False

        policy = self.cfg.retry
        final, fell_back = first, False
        attempt_index = 0
        for _ in range(int(policy.get("max_retries", 0))):
            if not final.failed:
                break
            attempt_index += 1
            final = self._call(messages, case["case_id"], run_id, arm, attempt_index, "retry")
        if final.failed and policy.get("fallback_to_original") and arm == "compressed":
            attempt_index += 1
            final = self._call(original_messages, case["case_id"], run_id, arm, attempt_index, "fallback")
            fell_back = True
        return first, final, fell_back

    # -- one case/repeat ----------------------------------------------------

    def run_case(self, case: dict[str, Any], run_id: int, rng: random.Random) -> dict[str, Any]:
        # Only messages cross into inference; gold stays here in the runner.
        original = sanitize_messages(case["messages"])

        comp: CompressionResult = self.compressor.compress(
            [dict(m) for m in original], self.cfg.compression_config
        )
        compressed = sanitize_messages(comp.messages)

        if self.cfg.experiment_mode == MODE_CONTROL:
            # Control arm: run the ORIGINAL prompt twice to measure ordinary
            # response variability. Never labelled as compression.
            compressed = original

        arms = ["baseline", "compressed"]
        rng.shuffle(arms)  # randomized, recorded via the seed

        results: dict[str, GenerationResult] = {}
        firsts: dict[str, GenerationResult] = {}
        fell_back = False
        comp_started = time.perf_counter()
        for arm in arms:
            msgs = original if arm == "baseline" else compressed
            first, final, fb = self._run_arm_with_policy(msgs, case, run_id, arm, original)
            firsts[arm], results[arm] = first, final
            if arm == "compressed":
                fell_back = fb
        compressed_pipeline_ms = (time.perf_counter() - comp_started) * 1000

        b, c = results["baseline"], results["compressed"]

        # Pipeline latency spans compression through the final answer.
        pipeline_latency = None
        if c.latency_ms is not None:
            pipeline_latency = c.latency_ms + (comp.latency_ms or 0.0)

        pred = {
            "case_id": case["case_id"],
            "run_id": run_id,
            "baseline_output": b.output,
            "compressed_output": c.output,
            "baseline_input_tokens": b.input_tokens,
            "compressed_input_tokens": c.input_tokens,
            "baseline_output_tokens": b.output_tokens,
            "compressed_output_tokens": c.output_tokens,
            "baseline_latency_ms": b.latency_ms,
            "compressed_pipeline_latency_ms": pipeline_latency,
            "baseline_cost_usd": b.cost_usd,
            "compressed_pipeline_cost_usd": c.cost_usd,
            "baseline_error": b.error,
            "compressed_error": c.error,
            "compressed_messages": compressed,
            "baseline_messages": original,
            "compression_metadata": {
                **comp.metadata,
                "fell_back_to_original": fell_back,
                "first_attempt_compressed_pass": firsts["compressed"].passed,
                "arm_order": arms,
                "experiment_mode": self.cfg.experiment_mode,
                "compression_scope": self.cfg.compression_scope,
            },
            "model_id": c.model_id,
            "tokenizer_id": getattr(self.counter, "tokenizer_id", None),
            "dictionary_version": comp.dictionary_version,
            "compression_version": comp.compressor_version,
            "token_count_source": c.token_count_source,
        }
        # Gold must not be present anywhere in the inference-facing fields.
        assert_no_gold_fields(pred["compressed_messages"], "prediction.compressed_messages")
        assert_no_gold_fields(pred["baseline_messages"], "prediction.baseline_messages")
        return pred

    def run(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preds = []
        for run_id in range(self.cfg.repeats):
            for case in cases:
                # Per-case RNG keeps arm-order deterministic under the seed
                # while remaining independent across cases and repeats.
                rng = random.Random(f"{self.cfg.seed}:{case['case_id']}:{run_id}")
                preds.append(self.run_case(case, run_id, rng))
        return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    cases = load_jsonl(HERE / cfg.fixtures if not Path(cfg.fixtures).is_absolute() else Path(cfg.fixtures))
    problems = validate_fixtures(cases)
    print(f"fixtures: {len(cases)}")
    print(f"categories: {sorted({c['category'] for c in cases})}")
    print(f"clusters: {len({c.get('cluster_id', c['case_id']) for c in cases})}")
    print(f"splits: {sorted({c.get('split', 'unspecified') for c in cases})}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:20]:
            print("  -", p)
        return 1
    print("\nOK: all fixtures structurally valid")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = Config.load(cfg_path)
    if args.mode:
        cfg.mode = args.mode

    fixtures_path = Path(cfg.fixtures)
    if not fixtures_path.is_absolute():
        fixtures_path = HERE / fixtures_path
    cases = load_jsonl(fixtures_path)
    problems = validate_fixtures(cases)
    if problems:
        print(f"Error: {len(problems)} fixture problems; run `validate` for detail", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]

    fp = cfg.fingerprint()
    exp_id = cfg.experiment_id or f"{cfg.mode}-{cfg.experiment_mode}-{fp}"
    out_dir = Path(cfg.out_root) / exp_id
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_hash = _sha256_file(fixtures_path)
    manifest_path = out_dir / "manifest.json"

    # Resume is permitted only when config AND fixtures are byte-identical.
    existing: list[dict[str, Any]] = []
    if args.resume and manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prev.get("config_fingerprint") != fp or prev.get("fixture_sha256") != fixture_hash:
            print("Error: cannot resume - config or fixture hash changed", file=sys.stderr)
            return 2
        pred_path = out_dir / "predictions.jsonl"
        if pred_path.exists():
            existing = load_jsonl(pred_path)
            done = {(r["case_id"], r["run_id"]) for r in existing}
            cases = [c for c in cases if not all((c["case_id"], r) in done for r in range(cfg.repeats))]
            print(f"resuming: {len(existing)} records already present")

    started = _now()
    try:
        runner = Runner(cfg, out_dir, max_live_calls=args.max_live_calls)
    except (RuntimeError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        preds = runner.run(cases)
    except GoldLeakageError as exc:
        print(f"FATAL gold leakage: {exc}", file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    preds = existing + preds
    _write_jsonl(out_dir / "predictions.jsonl", preds)
    _write_jsonl(out_dir / "attempts.jsonl", [asdict(a) for a in runner.attempts])

    # Scorer-compatible predictions: strip harness-only fields it does not expect.
    scorer_preds = [{k: v for k, v in p.items() if k != "baseline_messages"} for p in preds]
    _write_jsonl(out_dir / "predictions.scorer.jsonl", scorer_preds)

    is_mock = cfg.mode != "live"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp_id,
        "config_fingerprint": fp,
        "config": asdict(cfg),
        "config_sha256": _sha256_file(cfg_path),
        "fixture_path": str(fixtures_path),
        "fixture_sha256": fixture_hash,
        "fixture_count": len(load_jsonl(fixtures_path)),
        "cases_run": len({p["case_id"] for p in preds}),
        "repeats": cfg.repeats,
        "records": len(preds),
        "run_mode": cfg.mode,
        "experiment_mode": cfg.experiment_mode,
        "synthetic_results": is_mock,
        "adapters": {
            "compressor": runner.compressor.name,
            "compressor_is_mock": getattr(runner.compressor, "is_mock", False),
            "target": runner.target.name,
            "target_is_mock": getattr(runner.target, "is_mock", False),
            "token_counter": getattr(runner.counter, "name", None),
        },
        "live_calls_made": runner.live_calls,
        "live_call_budget": runner.live_budget,
        "started_at": started,
        "finished_at": _now(),
        "implementation_revision": _revision(),
        "limitations": [
            "The 40-case suite is a development/smoke set, not a held-out release benchmark.",
            "DeepSeek publishes no chat_template, so locally-computed full-request counts "
            "are labelled approximate_local_template and are excluded from release gates.",
            "Provider-reported usage (source=provider_usage) is the only exact full-request count.",
        ] + ([
            "SYNTHETIC: results come from mock adapters and are not model measurements."
        ] if is_mock else []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nrecords: {len(preds)}   attempts: {len(runner.attempts)}")
    if is_mock:
        print("MODE: mock - results are SYNTHETIC and must not be reported as model performance.")
    print(f"live calls: {runner.live_calls}")
    print(f"artifacts -> {out_dir}")
    print(f"\nnext:\n  python score_evals.py --cases {fixtures_path.name} "
          f"--predictions {out_dir/'predictions.scorer.jsonl'} "
          f"--out {out_dir/'report.json'} --details {out_dir/'scored_cases.jsonl'}")
    return 0


def _revision() -> str:
    """Hash the harness source so a report is traceable to its code."""
    h = hashlib.sha256()
    for name in sorted(["run_evals.py", "adapters.py", "score_evals.py"]):
        p = HERE / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="structurally validate fixtures")
    v.add_argument("--config", default=str(HERE / "config.example.json"))
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="execute the paired evaluation")
    r.add_argument("--config", default=str(HERE / "config.example.json"))
    r.add_argument("--mode", choices=["mock", "live"], help="override config mode")
    r.add_argument("--limit", type=int, help="run only the first N cases (smoke)")
    r.add_argument("--max-live-calls", type=int, help="hard cap on live API calls")
    r.add_argument("--resume", action="store_true")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
