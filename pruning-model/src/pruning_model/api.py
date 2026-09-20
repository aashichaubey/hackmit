"""Reuse native request accounting; journal all paid calls before parsing them."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tokenmix.evaluation.adapters import Message, OpenRouterTarget
from tokenmix.evaluation.config import load_dotenv
from tokenmix.evaluation.storage import (
    append_jsonl,
    digest,
    read_jsonl,
    strict_json,
    utc_now,
)


class JournaledAPI:
    def __init__(
        self,
        config: Path,
        journal: Path,
        *,
        max_calls: int,
        target=None,
        max_attempts: int = 3,
    ):
        raw = strict_json(config.read_text())
        self.generation = raw["generation"].copy()
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        if target is None:
            provider = self.generation.get("provider", {})
            if (
                provider.get("allow_fallbacks") is not False
                or len(provider.get("only", [])) != 1
            ):
                raise ValueError(
                    "live experiments require one pinned provider with fallback disabled"
                )
        self.journal = journal
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.lock = threading.RLock()
        self.key_locks = {}
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        # The shared repository key remains available without copying credentials.
        load_dotenv(project_root.parent / ".env")
        self.target = target or OpenRouterTarget(timeout=120)
        self.events = read_jsonl(journal)
        self.used = sum(e["event"] == "started" for e in self.events)
        self.cache = {
            e.get("base_key", e["key"]): e["result"]
            for e in self.events
            if e["event"] == "completed"
        }
        self.attempts = Counter(
            e.get("base_key", e["key"]) for e in self.events if e["event"] == "started"
        )
        started = {e["key"] for e in self.events if e["event"] == "started"}
        completed = {e["key"] for e in self.events if e["event"] == "completed"}
        unresolved = started - completed
        self.uncertain = {
            e.get("base_key", e["key"]) for e in self.events if e["key"] in unresolved
        }

    def call(
        self, messages: tuple[Message, ...], *, purpose: str, max_tokens: int = 1024
    ) -> dict:
        generation = dict(self.generation, max_tokens=max_tokens)
        key = digest(
            {
                "messages": [asdict(m) for m in messages],
                "generation": generation,
                "purpose": purpose,
            }
        )
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            return self._call(messages, generation, purpose, key)

    def _call(self, messages, generation, purpose, key):
        if key in self.uncertain:
            raise ValueError(
                "An interrupted API call has unknown consumption; use a new journal to explicitly retry"
            )
        while True:
            cached = self.cache.get(key)
            if cached:
                transient = any(
                    token in (cached["error"] or "")
                    for token in ("HTTP 429", "HTTP 502", "HTTP 503", "HTTP 504")
                )
                if not transient or self.attempts[key] >= self.max_attempts:
                    return cached
                time.sleep(min(30, 4 * 2 ** self.attempts[key]))
            with self.lock:
                if self.used >= self.max_calls:
                    raise ValueError(
                        f"API call cap reached ({self.max_calls}); progress is resumable"
                    )
                attempt_key = digest(
                    {"base_key": key, "attempt": self.attempts[key] + 1}
                )
                append_jsonl(
                    self.journal,
                    {
                        "event": "started",
                        "key": attempt_key,
                        "base_key": key,
                        "purpose": purpose,
                        "at": utc_now(),
                        "request": self.target.request(messages, generation),
                    },
                )
                self.used += 1
                self.attempts[key] += 1
                self.uncertain.add(key)
            result = asdict(self.target.generate(messages, generation))
            with self.lock:
                append_jsonl(
                    self.journal,
                    {
                        "event": "completed",
                        "key": attempt_key,
                        "base_key": key,
                        "purpose": purpose,
                        "at": utc_now(),
                        "result": result,
                    },
                )
                self.cache[key] = result
                self.uncertain.remove(key)

    def json(self, system: str, payload: dict, *, purpose: str, max_tokens: int = 4096):
        import json

        result = self.call(
            (
                Message(
                    "system",
                    system + " Return only valid JSON, without Markdown fences.",
                ),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ),
            purpose=purpose,
            max_tokens=max_tokens,
        )
        if result["error"] or result["finish_reason"] != "stop":
            raise ValueError(
                f"{purpose} failed: {result['error'] or result['finish_reason']}"
            )
        return strict_json(result["output"])
