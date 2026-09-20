"""Build a token-efficient English->Mandarin phrase dictionary.

Mined from the PARALLEL CORPUS only, never from the evaluation fixtures, so
that eval results measure generalization rather than memorization.

Pipeline per candidate phrase:
  1. mine frequent English n-grams from the corpus
  2. obtain a Mandarin rendering (LLM, batched)
  3. reject unless invariants hold (numbers/negation/URLs/code preserved)
  4. reject unless LaBSE cross-lingual similarity >= threshold
  5. keep only if the Mandarin side is strictly cheaper in TARGET tokens

    python -m scripts.build_dictionary --max-candidates 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_language.data.datasets import read_jsonl  # noqa: E402
from token_language.data.normalization import normalize_chinese  # noqa: E402
from token_language.schema import DictionaryEntry  # noqa: E402
from token_language.semantics.embeddings import EmbeddingModel  # noqa: E402
from token_language.semantics.invariants import check_invariants  # noqa: E402
from token_language.tokenizer import HFTokenizer  # noqa: E402

log = logging.getLogger("build_dictionary")

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+|[.,;:!?]")
# Phrases that are pure function words carry no standalone meaning.
_STOP_ONLY = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are",
              "was", "were", "be", "for", "on", "at", "it", "that", "this",
              "with", "as", "by", "from", "but", "not", "no"}


def mine_phrases(texts: list[str], n_min: int = 2, n_max: int = 6,
                 min_freq: int = 3) -> list[tuple[str, int]]:
    """Frequent English n-grams that look like meaningful phrases."""
    counts: Counter[str] = Counter()
    for text in texts:
        toks = _WORD.findall(text)
        for n in range(n_min, n_max + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i : i + n]
                if any(t in ".,;:!?" for t in gram):
                    continue
                if all(t.lower() in _STOP_ONLY for t in gram):
                    continue
                if gram[0].lower() in _STOP_ONLY or gram[-1].lower() in _STOP_ONLY:
                    continue
                counts[" ".join(gram)] += 1
    return [(p, c) for p, c in counts.most_common() if c >= min_freq]


TRANSLATE_PROMPT = (
    "Translate each numbered English phrase into Simplified Chinese.\n"
    "Rules: preserve meaning exactly; keep every number, name, URL and code "
    "identifier unchanged; keep negation; be as concise as natural Chinese allows; "
    "do not add explanation.\n"
    "Return ONLY a JSON object mapping the number (as a string) to the Chinese "
    'translation, e.g. {"1": "数据库连接超时"}.\n\n'
)


def translate_batch(phrases: list[str], model: str, api_key: str,
                    timeout: float = 120.0) -> dict[str, str]:
    """Ask the target model for Mandarin renderings. Returns {phrase: chinese}."""
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(phrases))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TRANSLATE_PROMPT + numbered}],
        "temperature": 0.0,
        "max_tokens": 8000,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    content = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    try:
        mapping = json.loads(content)
    except json.JSONDecodeError:
        log.warning("unparseable translation batch; skipping %d phrases", len(phrases))
        return {}
    out = {}
    for key, zh in mapping.items():
        try:
            idx = int(key) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(phrases) and isinstance(zh, str) and zh.strip():
            out[phrases[idx]] = normalize_chinese(zh)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", default="data/processed/train.jsonl",
                    help="TRAIN split only - never the eval fixtures")
    ap.add_argument("--out", default="data/dictionaries/phrases.jsonl")
    ap.add_argument("--tokenizer", default="deepseek-v4-flash")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--max-candidates", type=int, default=300)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--embedding-model", default="labse")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        return 2

    examples = list(read_jsonl(args.examples))
    log.info("mining phrases from %d TRAIN examples", len(examples))
    candidates = mine_phrases([e.english for e in examples], min_freq=args.min_freq)
    candidates = candidates[: args.max_candidates]
    log.info("%d candidate phrases", len(candidates))
    if not candidates:
        log.error("no candidates; lower --min-freq")
        return 1

    tk = HFTokenizer.from_pretrained(args.tokenizer)

    translations: dict[str, str] = {}
    phrases = [p for p, _ in candidates]
    for i in range(0, len(phrases), args.batch):
        chunk = phrases[i : i + args.batch]
        log.info("translating %d-%d of %d", i + 1, i + len(chunk), len(phrases))
        translations.update(translate_batch(chunk, args.model, api_key))

    log.info("received %d translations", len(translations))
    if not translations:
        return 1

    freq = dict(candidates)
    keys = list(translations)
    with EmbeddingModel(args.embedding_model) as emb:
        sims = emb.similarity(keys, [translations[k] for k in keys])

    entries: list[DictionaryEntry] = []
    stats = Counter()
    for phrase, sim in zip(keys, sims):
        zh = translations[phrase]
        stats["translated"] += 1

        report = check_invariants(phrase, zh)
        if not report.ok:
            stats["reject_invariant"] += 1
            continue
        if float(sim) < args.threshold:
            stats["reject_similarity"] += 1
            continue

        en_tok = tk.count_tokens(phrase)
        zh_tok = tk.count_tokens(zh)
        if zh_tok >= en_tok:
            # Kept out of the dictionary but counted: this is the evidence for
            # "English sometimes wins", which is why this is a hybrid and not a
            # translation system.
            stats["reject_not_cheaper"] += 1
            continue

        entries.append(DictionaryEntry(
            concept_id=hashlib.sha1(phrase.encode()).hexdigest()[:12],
            english=phrase,
            mandarin=zh,
            english_tokens=en_tok,
            mandarin_tokens=zh_tok,
            tokenizer_id=tk.id,
            semantic_score=round(float(sim), 4),
            frequency=freq.get(phrase, 1),
        ))
        stats["kept"] += 1

    # Longest first so the encoder's greedy match prefers specific phrases.
    entries.sort(key=lambda e: (-len(e.english), -e.savings))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.model_dump_json() + "\n")

    version = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    out.with_suffix(".meta.json").write_text(json.dumps({
        "dictionary_version": version,
        "entries": len(entries),
        "tokenizer_id": tk.id,
        "translation_model": args.model,
        "embedding_model": args.embedding_model,
        "semantic_threshold": args.threshold,
        "source": args.examples,
        "provenance": "mined from the TRAIN split of the parallel corpus; "
                      "evaluation fixtures were never used",
        "stats": dict(stats),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ncandidates translated : {stats['translated']}")
    print(f"  rejected invariants : {stats['reject_invariant']}")
    print(f"  rejected similarity : {stats['reject_similarity']}")
    print(f"  rejected not cheaper: {stats['reject_not_cheaper']}  <- English already wins")
    print(f"  KEPT                : {stats['kept']}")
    print(f"\ndictionary_version={version}  -> {out}")
    if entries:
        print("\ntop savings:")
        for e in sorted(entries, key=lambda x: -x.savings)[:10]:
            print(f"  {e.english_tokens:>2}->{e.mandarin_tokens:<2} (-{e.savings}) "
                  f"sim={e.semantic_score}  {e.english[:40]} = {e.mandarin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
