"""Cached API access and blinded evaluation for benchmark_v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evaluate import deterministic_score, _parse_judge
from .representations import LLMProvider


class ApiCache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, payload: dict):
        return self.data.get(self.key(payload))

    def put(self, payload: dict, value: dict) -> None:
        self.data[self.key(payload)] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class CachedProvider:
    """Drop-in provider wrapper that persists every completed API response."""

    def __init__(self, provider: LLMProvider, cache: ApiCache):
        self.inner = provider
        self.cache = cache
        self.provider = provider.provider
        self.generation_model = provider.generation_model
        self.answer_model = provider.answer_model
        self.judge_model = provider.judge_model

    def complete(self, system: str, user: str, *, purpose: str, temperature: float = 0) -> dict:
        model = {"generation": self.generation_model, "answer": self.answer_model, "judge": self.judge_model, "semantic_judge": self.judge_model}[purpose]
        payload = {"v": 1, "model": model, "system": system, "user": user, "purpose": purpose, "temperature": temperature}
        cached = self.cache.get(payload)
        if cached is not None:
            return {**cached, "cached": True}
        result = self.inner.complete(system, user, purpose=purpose, temperature=temperature)
        self.cache.put(payload, result)
        return {**result, "cached": False}

    def complete_schema(self, system: str, user: str, *, schema_name: str, schema: dict, temperature: float = 0) -> dict:
        payload = {"v": 1, "model": self.generation_model, "system": system, "user": user, "purpose": "structured_generation", "temperature": temperature, "schema_name": schema_name, "schema": schema}
        cached = self.cache.get(payload)
        if cached is not None:
            return {**cached, "cached": True}
        result = self.inner.complete_schema(system, user, schema_name=schema_name, schema=schema, temperature=temperature)
        self.cache.put(payload, result)
        return {**result, "cached": False}


def evaluate_blinded(context: str, item: dict, provider: CachedProvider) -> dict:
    answer = provider.complete(
        "Using only the provided context, answer the question. Give only the answer.",
        f"CONTEXT:\n{context}\n\nQUESTION:\n{item['question']}", purpose="answer", temperature=0,
    )
    deterministic_correct, method = deterministic_score(item["expected_answer"], answer["text"])
    judge_prompt = (
        f"QUESTION:\n{item['question']}\n\nEXPECTED ANSWER:\n{item['expected_answer']}\n\nMODEL ANSWER:\n{answer['text']}\n\n"
        "Return JSON only: {\"correct\": true|false, \"reason\": \"brief explanation\"}"
    )
    judge = provider.complete(
        "Judge whether the model answer is semantically correct relative to the expected answer. Accept equivalent wording and harmless extra detail. Reject contradictions or missing required parts.",
        judge_prompt, purpose="judge", temperature=0,
    )
    correct, reason = _parse_judge(judge["text"])
    return {
        "model_answer": answer["text"], "deterministic_correct": deterministic_correct,
        "deterministic_method": method, "judge_correct": correct, "judge_reason": reason,
        "answer_usage": answer.get("usage", {}), "judge_usage": judge.get("usage", {}),
        "answer_cached": answer.get("cached", False), "judge_cached": judge.get("cached", False),
    }
