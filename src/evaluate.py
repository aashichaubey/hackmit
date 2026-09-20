"""Independent downstream QA with deterministic and blinded judge scoring."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .representations import LLMProvider


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s.]", " ", text.casefold().strip())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).rstrip(".")


def deterministic_score(expected: str, actual: str) -> tuple[bool, str]:
    e, a = normalize(expected), normalize(actual)
    if e == a:
        return True, "normalized_exact"
    if e in a or a in e:
        return True, "normalized_containment"
    ratio = SequenceMatcher(None, e, a).ratio()
    return ratio >= 0.86, f"fuzzy_{ratio:.3f}"


def _parse_judge(text: str) -> tuple[bool, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    verdict = json.loads(cleaned)
    if not isinstance(verdict.get("correct"), bool):
        raise ValueError("judge response lacks boolean 'correct'")
    return verdict["correct"], str(verdict.get("reason", ""))


def evaluate_question(context: str, item: dict, provider: LLMProvider) -> dict:
    answer_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{item['question']}"
    answer = provider.complete("Using only the provided context, answer the question. Give only the answer.", answer_prompt, purpose="answer", temperature=0)
    deterministic_correct, method = deterministic_score(item["expected_answer"], answer["text"])
    # Deliberately contains no representation name or metadata.
    judge_prompt = (
        f"QUESTION:\n{item['question']}\n\nEXPECTED ANSWER:\n{item['expected_answer']}\n\n"
        f"MODEL ANSWER:\n{answer['text']}\n\nReturn JSON only: {{\"correct\": true|false, \"reason\": \"brief explanation\"}}"
    )
    judge = provider.complete(
        "Judge whether the model answer is semantically correct relative to the expected answer. Accept equivalent wording and harmless extra detail. Reject contradictions or missing required parts.",
        judge_prompt,
        purpose="judge",
        temperature=0,
    )
    judge_correct, judge_reason = _parse_judge(judge["text"])
    return {
        "answer": answer["text"],
        "deterministic_correct": deterministic_correct,
        "deterministic_method": method,
        "judge_correct": judge_correct,
        "judge_reason": judge_reason,
        "answer_usage": answer["usage"],
        "judge_usage": judge["usage"],
    }
