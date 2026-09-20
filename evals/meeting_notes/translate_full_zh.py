"""Whole-note Mandarin translation (real language, model-translated).

Tests the lever that actually worked in the Phase-1 corpus measurement:
translate entire sentences, not single words mid-sentence, and keep names,
dates, numbers and percentages in their original form so facts stay literally
checkable. No legend - it's real Chinese, verified in-context against the
target tokenizer exactly like the corpus dictionary.

MEASURED RESULTS (live, 12 notes, deepseek/deepseek-v4-flash via OpenRouter).
Translation is not perfectly deterministic even at temperature 0, so gross
savings vary run-to-run (observed range so far: 5.6%-7.8%). Consistent across
runs:
  grammar cost    : 0     (real language needs no legend)
  net savings     : == gross (no overhead to subtract)
  blind QA        : 21/21 = 100% TRUE accuracy on one run, once the grader's
                    English-only literal matching is corrected by hand (some
                    answers came back correct-but-in-Chinese, e.g. "周五" for
                    "Friday", and the naive substring grader marked those
                    wrong). Re-verify this by hand on any new translation run
                    rather than trusting the raw grader score.

This is smaller than the `simple_compression` baseline's 17.8% gross, but it
is the only tested method with zero grammar/legend overhead and no measured
task-accuracy loss. Treat the exact percentage as noisy, not a fixed constant.

    python translate_full_zh.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "character_compression"))

from adapters import DeepSeekTokenCounter  # noqa: E402

PROMPT = """Translate this meeting note into natural Simplified Chinese.

Rules:
- Keep every person name, organization name, project name (e.g. "Project A"), date, day-of-week, number, and percentage EXACTLY as written in the original (do not translate or reformat them).
- Preserve negation exactly - do not drop or add "not".
- Preserve every fact: who does what, deadlines, blockers, decisions, agreements.
- Natural fluent Chinese, not word-for-word.
- Return ONLY the translated text, no explanation, no quotes.

NOTE:
{text}"""


def translate(text: str, api_key: str, model: str = "deepseek/deepseek-v4-flash") -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "temperature": 0.0, "max_tokens": 2048,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read())
    return (payload["choices"][0]["message"].get("content") or "").strip()


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    here = Path(__file__).resolve().parent
    notes = [json.loads(l) for l in (here / "meeting_notes.jsonl").read_text().splitlines() if l.strip()]
    tc = DeepSeekTokenCounter()

    out = []
    raw_tot = zh_tot = 0
    for n in notes:
        zh = translate(n["text"], api_key)
        rt, zt = tc.count_text(n["text"]), tc.count_text(zh)
        raw_tot += rt
        zh_tot += zt
        out.append({**n, "full_zh": zh, "raw_tokens": rt, "zh_tokens": zt})
        print(f"[{rt:>3}->{zt:<3} ({100*(1-zt/rt):+5.1f}%)] {n['id']}")
        print(f"    {zh}")

    print(f"\nTOTAL {raw_tot} -> {zh_tot}   gross = {100*(1-zh_tot/raw_tot):.1f}%   grammar = 0")

    (here / "full_zh_translations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"\nwrote full_zh_translations.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
