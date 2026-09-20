import json

import polib
import pytest

from tokenmix.cli import main
from tokenmix.corpus import import_po


def test_import_retains_only_nonempty_nonfuzzy_singular_pairs(tmp_path):
    path = tmp_path / "sample.po"
    po = polib.POFile()
    po.metadata = {"Language": "zh_CN", "Content-Type": "text/plain; charset=UTF-8"}
    po.extend([
        polib.POEntry(msgid="operating system", msgstr="操作系统"),
        polib.POEntry(msgid="missing", msgstr=""),
        polib.POEntry(msgid="fuzzy", msgstr="模糊", flags=["fuzzy"]),
        polib.POEntry(msgid="old", msgstr="旧", obsolete=True),
        polib.POEntry(msgid="item", msgid_plural="items", msgstr_plural={0: "项目"}),
        polib.POEntry(msgid="context", msgstr="上下文", msgctxt="docs"),
    ])
    po.save(str(path))
    entries = import_po(path, source="test revision; fixture", domain="technical")
    assert [e.en for e in entries] == ["operating system", "context"]
    assert not any(e.enabled for e in entries)
    assert entries[1].source.endswith("msgctxt=docs")
    assert entries == import_po(path, source="test revision; fixture", domain="technical")


def test_cli_compression_json(capsys):
    assert main(["compress", "as soon as possible", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["text"] == "尽快"
    assert result["saved_tokens"] == 2


def test_cli_benchmark_totals(capsys):
    assert main(["benchmark"]) == 0
    report = json.loads(capsys.readouterr().out)
    for value in report["encodings"].values():
        assert value["english_tokens"] == sum(r["en_tokens"] for r in value["rows"])
        assert value["per_pair_min_tokens"] == sum(min(r["en_tokens"], r["zh_tokens"]) for r in value["rows"])
    assert main(["benchmark", "--summary"]) == 0
    assert "rows" not in json.loads(capsys.readouterr().out)["encodings"]["o200k_base"]


def test_cli_does_not_overwrite_existing_dictionary(tmp_path, capsys):
    source = tmp_path / "empty.po"
    source.write_text("", encoding="utf-8")
    output = tmp_path / "existing.jsonl"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["import-po", str(source), "--source", "fixture", "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text() == "existing"
