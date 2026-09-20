"""Transactional shared budgets, exact public request replay, conservative failures."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import MODEL
from .evo_grammar import digest
from .tokens import count_tokens

# USD / million tokens, pinned reference model, verified 2026-09-20.
RATES = {"input": 0.40, "cached_input": 0.10, "output": 1.60}


class BudgetExceeded(RuntimeError):
    pass


def dollars(usage: dict) -> float:
    cached = usage.get("cached_input_tokens", 0)
    return ((usage["input_tokens"] - cached) * RATES["input"] + cached * RATES["cached_input"] + usage["output_tokens"] * RATES["output"]) / 1_000_000


class Ledger:
    def __init__(self, path: str | Path, max_calls: int = 3500, max_usd: float = 10.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, max_calls INTEGER, max_usd REAL)")
            db.execute("INSERT OR IGNORE INTO config VALUES (1, ?, ?)", (max_calls, max_usd))
            # Restarting cannot expand the original budget.
            db.execute("UPDATE config SET max_calls=MIN(max_calls, ?), max_usd=MIN(max_usd, ?) WHERE id=1", (max_calls, max_usd))
            db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, key TEXT, purpose TEXT, status TEXT, reserved REAL, result TEXT, created REAL)")
            db.execute('CREATE TABLE IF NOT EXISTS pacing (id INTEGER PRIMARY KEY, next_dispatch REAL)')
            db.execute('INSERT OR IGNORE INTO pacing VALUES (1,0)')
            db.execute('CREATE TABLE IF NOT EXISTS discovery_allocations (purpose TEXT PRIMARY KEY, max_calls INTEGER, reserve_calls INTEGER, reason TEXT, created REAL)')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def allocate_successor_discovery(self, study: str, max_calls: int, reserve_calls: int, reason: str) -> str:
        """Explicit, immutable allocation within the unchanged global call/dollar limits."""
        if not study or not all(c.isascii() and (c.isalnum() or c in '_-') for c in study):
            raise ValueError('Use a simple nonempty study identifier')
        if type(max_calls) is not int or max_calls <= 0 or type(reserve_calls) is not int or reserve_calls < 0 or not reason.strip():
            raise ValueError('A positive allocation, nonnegative reserve and reason are required')
        purpose = 'successor_discovery:' + study
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM discovery_allocations WHERE purpose=?',(purpose,)).fetchone()
            if previous:
                if (previous['max_calls'],previous['reserve_calls'],previous['reason']) != (max_calls,reserve_calls,reason):
                    raise ValueError('Recorded successor allocations are immutable')
                return purpose
            remaining = db.execute('SELECT max_calls-(SELECT COUNT(*) FROM calls) FROM config WHERE id=1').fetchone()[0]
            outstanding = db.execute("SELECT COALESCE(SUM(MAX(0,a.max_calls-(SELECT COUNT(*) FROM calls c WHERE c.purpose=a.purpose))),0) FROM discovery_allocations a").fetchone()[0]
            previous_reserve = db.execute('SELECT COALESCE(MAX(reserve_calls),0) FROM discovery_allocations').fetchone()[0]
            if max_calls + outstanding + max(reserve_calls, previous_reserve) > remaining:
                raise BudgetExceeded('Successor allocation would consume the final-evaluation reserve')
            db.execute('INSERT INTO discovery_allocations VALUES(?,?,?,?,?)',(purpose,max_calls,reserve_calls,reason,time.time()))
        return purpose

    def check_budget(self, reserve: float, purpose: str, leave_calls: int = 0, leave_usd: float = 0) -> None:
        with self.connect() as db:
            self._check(db, reserve, purpose, leave_calls, leave_usd)

    @staticmethod
    def _check(db, reserve, purpose, leave_calls, leave_usd):
        cfg = db.execute("SELECT * FROM config WHERE id=1").fetchone()
        totals = db.execute("SELECT COUNT(*) n, COALESCE(SUM(reserved),0) usd FROM calls").fetchone()
        discovery = db.execute("SELECT COUNT(*) FROM calls WHERE purpose IN ('probe','finalist','validation','classifier_development')").fetchone()[0]
        if totals["n"] + 1 + leave_calls > cfg["max_calls"] or totals["usd"] + reserve + leave_usd > cfg["max_usd"]:
            raise BudgetExceeded("Global budget reached; unfinished evaluation remains incomplete")
        if purpose.startswith('successor_discovery:'):
            allocation = db.execute('SELECT * FROM discovery_allocations WHERE purpose=?',(purpose,)).fetchone()
            if allocation is None:
                raise BudgetExceeded('Successor discovery requires an explicit recorded allocation')
            used = db.execute('SELECT COUNT(*) FROM calls WHERE purpose=?',(purpose,)).fetchone()[0]
            held_back = db.execute('SELECT COALESCE(MAX(reserve_calls),0) FROM discovery_allocations').fetchone()[0]
            if used >= allocation['max_calls'] or totals['n'] + 1 + max(leave_calls,held_back) > cfg['max_calls']:
                raise BudgetExceeded('Successor discovery allocation or final-evaluation reserve reached')
        if purpose in ("probe", "finalist", "validation", "classifier_development") and discovery >= 1500:
            raise BudgetExceeded("Discovery call ceiling reached")

    def reserve(self, key: str, purpose: str, amount: float, leave_calls=0, leave_usd=0) -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check(db, amount, purpose, leave_calls, leave_usd)
            cursor = db.execute("INSERT INTO calls(key,purpose,status,reserved,created) VALUES(?,?,'pending',?,?)", (key, purpose, amount, time.time()))
            return cursor.lastrowid

    def record_call(self, call_id: int, result: dict):
        with self.connect() as db:
            if result.get("usage") is None:
                db.execute("UPDATE calls SET status='unknown',result=? WHERE id=?", (json.dumps(result), call_id))
            else:
                db.execute("UPDATE calls SET status='complete',reserved=?,result=? WHERE id=?", (dollars(result["usage"]), json.dumps(result), call_id))

    def pace(self, estimated_tokens: int, tokens_per_minute: int = 170000):
        """Shared leaky bucket, below the observed reference-model 200k TPM limit.

        This affects dispatch timing only, never prompts, labels, or retry counts.
        """
        now = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            next_slot = db.execute('SELECT next_dispatch FROM pacing WHERE id=1').fetchone()[0]
            dispatch = max(now, next_slot)
            db.execute('UPDATE pacing SET next_dispatch=? WHERE id=1', (dispatch + estimated_tokens * 60 / tokens_per_minute,))
        delay = dispatch - time.time()
        if delay > 0:
            time.sleep(delay)

    def replay(self, key: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT result FROM calls WHERE key=? AND status='complete' ORDER BY id LIMIT 1", (key,)).fetchone()
        if row:
            result = json.loads(row[0])
            result["replayed"] = True
            return result
        return None

    def summary(self) -> dict:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM calls").fetchall()
            config = dict(db.execute("SELECT * FROM config WHERE id=1").fetchone())
            allocations = [dict(r) for r in db.execute("SELECT * FROM discovery_allocations ORDER BY created")]
        totals = {"actual_calls": len(rows), "known_usd": 0.0, "reserved_or_spent_usd": sum(r["reserved"] for r in rows),
                  "unknown_usage_calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "by_purpose": {}, "limits": config, "successor_discovery_allocations": allocations}
        for row in rows:
            result = json.loads(row["result"]) if row["result"] else {}
            usage = result.get("usage")
            totals["by_purpose"].setdefault(row["purpose"], {"calls": 0, "known_usd": 0.0})
            totals["by_purpose"][row["purpose"]]["calls"] += 1
            if usage is None:
                totals["unknown_usage_calls"] += 1
            else:
                for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                    totals[key] += usage.get(key, 0)
                totals["known_usd"] += dollars(usage)
                totals["by_purpose"][row["purpose"]]["known_usd"] += dollars(usage)
        totals["accounting_status"] = "incomplete" if totals["unknown_usage_calls"] else "complete"
        return totals


class Runner:
    def __init__(self, client, ledger: Ledger | None = None, model: str = MODEL, public: bool = True):
        if model != MODEL:
            raise ValueError("A different model requires separate pricing and qualification")
        if not public and ledger is not None:
            raise ValueError("Private prompts must not enter disk-backed ledger")
        self.client, self.ledger, self.model, self.public = client, ledger, model, public
        self.memory_cache = {}

    def call(self, prompt: str, shape, purpose="probe", max_output_tokens=256, leave_calls=0, leave_usd=0, replicate=0) -> dict:
        if type(replicate) is not int or replicate < 0:
            raise ValueError('Replicate index must be a nonnegative integer')
        request = {"model": self.model, "input": prompt, "schema": shape.model_json_schema(), "max_output_tokens": max_output_tokens, "temperature": 0}
        if replicate:
            # Cache namespace only. Never change the prompt or provider parameters.
            request['replicate'] = replicate
        key = digest(request)
        cached = self.ledger.replay(key) if self.ledger else self.memory_cache.get(key)
        if cached:
            return {**cached, "replayed": True}
        # Conservative upper bound: UTF-8 bytes exceed tokens, plus schema and framing.
        upper_input = len(prompt.encode()) + len(json.dumps(request["schema"]).encode()) + 1024
        reserved = dollars({"input_tokens": upper_input, "output_tokens": max_output_tokens})
        call_id = self.ledger.reserve(key, purpose, reserved, leave_calls, leave_usd) if self.ledger else None
        if self.ledger:
            self.ledger.pace(count_tokens(prompt) + count_tokens(json.dumps(request['schema'])) + 512 + max_output_tokens)
        result = {"request_hash": key, "request": request, "usage": None, "answer": None, "error": None,
                  "response_id": None, "replayed": False, "visible_input_tokens": count_tokens(prompt, self.model)}
        try:
            response = self.client.with_options(max_retries=0, timeout=60).responses.parse(
                model=self.model, input=prompt, text_format=shape, max_output_tokens=max_output_tokens, temperature=0)
            result["response_id"] = response.id
            if response.usage is not None:
                details = response.usage.input_tokens_details
                result["usage"] = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                                   "cached_input_tokens": getattr(details, "cached_tokens", 0) if details else 0}
            if response.output_parsed is None:
                result["error"] = "Missing/invalid structured response or refusal"
            else:
                result["answer"] = response.output_parsed.model_dump()
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        if self.ledger:
            self.ledger.record_call(call_id, result)
        elif not result["error"]:
            self.memory_cache[key] = result
        return result
