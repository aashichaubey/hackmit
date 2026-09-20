from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

import tiktoken

from .core import compress, load_dictionary
from .corpus import import_po


SEED = Path(__file__).with_name("seed.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental English–Mandarin token optimization")
    commands = parser.add_subparsers(dest="command", required=True)
    rewrite = commands.add_parser("compress", help="Rewrite exact, enabled dictionary matches")
    rewrite.add_argument("text", nargs="?", help="Text; omit to read stdin")
    rewrite.add_argument("--dictionary", type=Path, default=SEED)
    rewrite.add_argument("--encoding", default="o200k_base", choices=tiktoken.list_encoding_names())
    rewrite.add_argument("--domain", default="general")
    rewrite.add_argument("--prefix", default="", help="Optional decoding/output-language instruction, included in token cost")
    rewrite.add_argument("--protect", action="append", default=[], help="Literal to keep unchanged; repeatable")
    rewrite.add_argument("--max-changes", type=int, default=100)
    rewrite.add_argument("--json", action="store_true")

    benchmark = commands.add_parser("benchmark", help="Count supplied parallel texts; not a semantic evaluation")
    benchmark.add_argument("--dictionary", type=Path, default=SEED)
    benchmark.add_argument("--encodings", nargs="+", default=["cl100k_base", "o200k_base"], choices=tiktoken.list_encoding_names())
    benchmark.add_argument("--summary", action="store_true", help="Omit individual text pairs from the report")

    importer = commands.add_parser("import-po", help="Convert a local English→Mandarin gettext file into disabled candidates")
    importer.add_argument("path", type=Path)
    importer.add_argument("--source", required=True, help="Source URL/revision and license provenance")
    importer.add_argument("--domain", default="technical")
    importer.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "compress":
            text = args.text if args.text is not None else sys.stdin.read()
            result = compress(text, load_dictionary(args.dictionary), encoding=args.encoding,
                              domain=args.domain, prefix=args.prefix, protect=args.protect,
                              max_changes=args.max_changes)
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(result.prefix + result.text)
                print(f"{result.original_tokens} → {result.payload_tokens} tokens; "
                      f"saved {result.saved_tokens} ({result.to_dict()['savings_percent']}%)", file=sys.stderr)
        elif args.command == "benchmark":
            entries = load_dictionary(args.dictionary)
            report = {
                "kind": "parallel-pair token counts; assumes translation equivalence",
                "tiktoken_version": importlib.metadata.version("tiktoken"),
                "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest(),
                "encodings": {},
            }
            for name in args.encodings:
                enc = tiktoken.get_encoding(name)
                rows = []
                for entry in entries:
                    en = len(enc.encode_ordinary(entry.en))
                    zh = len(enc.encode_ordinary(entry.zh))
                    rows.append({"id": entry.id, "en": entry.en, "zh": entry.zh,
                                 "en_tokens": en, "zh_tokens": zh,
                                 "preferred": "en" if en < zh else "zh" if zh < en else "tie",
                                 "enabled": entry.enabled})
                en_total = sum(r["en_tokens"] for r in rows)
                best_total = sum(min(r["en_tokens"], r["zh_tokens"]) for r in rows)
                report["encodings"][name] = {
                    "pairs": len(rows), "english_tokens": en_total,
                    "mandarin_tokens": sum(r["zh_tokens"] for r in rows),
                    "per_pair_min_tokens": best_total,
                    "per_pair_savings_vs_english_percent": round(100 * (en_total - best_total) / en_total, 2) if en_total else 0,
                    "english_wins": sum(r["preferred"] == "en" for r in rows),
                    "mandarin_wins": sum(r["preferred"] == "zh" for r in rows),
                    "ties": sum(r["preferred"] == "tie" for r in rows),
                }
                if not args.summary:
                    report["encodings"][name]["rows"] = rows
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            entries = import_po(args.path, source=args.source, domain=args.domain)
            # Exclusive creation prevents accidentally overwriting a reviewed dictionary.
            with args.output.open("x", encoding="utf-8") as stream:
                for entry in entries:
                    stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            print(f"Imported {len(entries)} disabled candidates into {args.output}")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"tokenmix: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
