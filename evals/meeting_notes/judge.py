"""Layer 3: semantic-equivalence judging.

Two judges, deliberately kept separate:

* **JEv** (TypeSafe "System One", `jev-latest`) supplies the typed fields it is
  actually built for - a calibrated boolean (Noul) and a bounded severity
  (Score). JEv **generates no text**, so `missing_information`,
  `introduced_information` and `reason` are structurally impossible for it;
  the SDK raises TypeError at decoration time for `str`/`list` fields.
* **An OpenRouter LLM judge** supplies those narrative fields.

Neither is trusted alone. `agreement_with_deterministic` compares every verdict
against the Layer-1 fact checker, which is ground truth by construction, so a
judge that waves through a dropped negation is visibly wrong rather than
quietly authoritative.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SEVERITY_LEVELS = ["none", "minor", "moderate", "severe"]


@dataclass
class JudgeVerdict:
    preserves_task_information: bool | None = None
    missing_information: list[str] = field(default_factory=list)
    introduced_information: list[str] = field(default_factory=list)
    severity: float | None = None
    reason: str = ""
    #: Provenance, so no field's origin is ambiguous.
    source: str = "unknown"
    #: JEv's calibrated probability, retained rather than only the threshold.
    noul: float | None = None
    confidence: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# JEv
# ---------------------------------------------------------------------------


class JevJudge:
    """Typed verdict from JEv. Supplies boolean + severity only."""

    name = "jev"
    #: Fields JEv cannot produce, documented rather than silently omitted.
    UNSUPPORTED = ("missing_information", "introduced_information", "reason")

    def __init__(self, threshold: float = 0.5, model: str = "jev-latest",
                 timeout: float = 30.0):
        from typesafe_sdk import TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Note the SDK reads TYPESAFE_API_KEY, "
                "not JEV_API_KEY."
            )
        self._client_cls = TypeSafeClient
        self.threshold = threshold
        self.model = model
        self.timeout = timeout

    def judge(self, original: str, compressed: str, task: str) -> JudgeVerdict:
        from typesafe_sdk import Noul, Score

        try:
            with self._client_cls(timeout=self.timeout) as c:
                r = c.system_one(
                    model=self.model,
                    state={"original": original, "compressed": compressed, "task": task},
                    questions={
                        "preserves_task_information": Noul(
                            instructions=(
                                "The COMPRESSED representation preserves all information "
                                "needed to perform the TASK correctly, with the same answer "
                                "as the ORIGINAL. Names, numbers, dates and negation must be "
                                "unchanged."
                            )
                        ),
                        "severity": Score(
                            instructions=(
                                "Severity of information loss or distortion in COMPRESSED "
                                "relative to ORIGINAL, for the purpose of the TASK."
                            ),
                            criteria=SEVERITY_LEVELS,
                        ),
                    },
                )
            noul = r.nouls["preserves_task_information"]
            score = r.scores["severity"]
            return JudgeVerdict(
                preserves_task_information=noul.noul >= self.threshold,
                severity=round(float(score.score), 3),
                noul=round(float(noul.noul), 4),
                confidence=round(float(getattr(score, "confidence", 0.0) or 0.0), 4),
                source="jev:jev-latest",
                reason="(JEv generates no text; see llm_judge for rationale)",
            )
        except Exception as exc:  # network / auth / schema
            return JudgeVerdict(source="jev:jev-latest",
                                error=f"{type(exc).__name__}: {str(exc)[:200]}")


# ---------------------------------------------------------------------------
# OpenRouter LLM judge (narrative fields)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are grading whether a COMPRESSED representation of a meeting note preserves the information needed for a TASK.

ORIGINAL:
{original}

COMPRESSED:
{compressed}

TASK:
{task}

Decide whether COMPRESSED preserves everything needed to perform TASK with the same answer as ORIGINAL. Pay specific attention to changes in: negation, numbers, percentages, dates, names, owners, and quantifiers.

Return ONLY a JSON object, no prose and no markdown fences:
{{"preserves_task_information": true or false,
  "missing_information": ["..."],
  "introduced_information": ["..."],
  "severity": 0,
  "reason": "one sentence"}}
severity: 0=none, 1=minor, 2=moderate, 3=severe."""


class LlmJudge:
    """Structured-JSON judge via OpenRouter."""

    name = "llm_judge"

    def __init__(self, model: str = "deepseek/deepseek-v4-flash",
                 timeout: float = 90.0, max_tokens: int = 2048):
        self._key = os.environ.get("OPENROUTER_API_KEY")
        if not self._key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def judge(self, original: str, compressed: str, task: str) -> JudgeVerdict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": _JUDGE_PROMPT.format(
                original=original, compressed=compressed, task=task)}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            content = (payload["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            return JudgeVerdict(source=f"openrouter:{self.model}",
                                error=f"{type(exc).__name__}: {str(exc)[:200]}")

        if content.startswith("```"):
            content = content.strip("`")
            content = content[content.find("{"):]
        try:
            start, end = content.index("{"), content.rindex("}") + 1
            data = json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            return JudgeVerdict(source=f"openrouter:{self.model}",
                                error=f"unparseable judge output: {content[:150]}")

        return JudgeVerdict(
            preserves_task_information=bool(data.get("preserves_task_information")),
            missing_information=list(data.get("missing_information") or []),
            introduced_information=list(data.get("introduced_information") or []),
            severity=data.get("severity"),
            reason=str(data.get("reason", ""))[:300],
            source=f"openrouter:{self.model}",
        )


# ---------------------------------------------------------------------------
# Combination + calibration against deterministic ground truth
# ---------------------------------------------------------------------------


def combine(jev: JudgeVerdict | None, llm: JudgeVerdict | None) -> dict[str, Any]:
    """Merge the two judges, keeping provenance for every field."""
    out: dict[str, Any] = {
        "preserves_task_information": None,
        "missing_information": [],
        "introduced_information": [],
        "severity": None,
        "reason": "",
        "field_sources": {},
        "judges_agree": None,
    }
    if jev and not jev.error:
        out["preserves_task_information"] = jev.preserves_task_information
        out["severity"] = jev.severity
        out["jev_noul"] = jev.noul
        out["field_sources"]["preserves_task_information"] = jev.source
        out["field_sources"]["severity"] = jev.source
    if llm and not llm.error:
        out["missing_information"] = llm.missing_information
        out["introduced_information"] = llm.introduced_information
        out["reason"] = llm.reason
        out["field_sources"]["missing_information"] = llm.source
        out["field_sources"]["reason"] = llm.source
        if out["preserves_task_information"] is None:
            out["preserves_task_information"] = llm.preserves_task_information
            out["field_sources"]["preserves_task_information"] = llm.source
        if jev and not jev.error:
            out["judges_agree"] = (
                jev.preserves_task_information == llm.preserves_task_information
            )
    out["errors"] = [v.error for v in (jev, llm) if v and v.error]
    return out


def agreement_with_deterministic(judge_says_preserved: bool | None,
                                 facts_fully_recovered: bool) -> str:
    """Classify a judge verdict against Layer-1 ground truth.

    `false_pass` is the dangerous quadrant: the judge approved a
    representation that demonstrably lost a planted fact.
    """
    if judge_says_preserved is None:
        return "unavailable"
    if judge_says_preserved and facts_fully_recovered:
        return "true_pass"
    if judge_says_preserved and not facts_fully_recovered:
        return "false_pass"
    if not judge_says_preserved and facts_fully_recovered:
        return "false_fail"
    return "true_fail"
