import random
from dataclasses import replace

import pytest

from tokenese.facts import Fact, MeetingFacts
from tokenese.evo_grammar import GrammarSpec, compile_facts, parse_compiled, projection, integrity_errors
from tokenese.evo_mutate import seed_population, mutate


def example():
    return MeetingFacts(facts=[
        Fact(kind="action", person="A:na", text=r"do not ship @v2 [draft]; C:\temp", deadline="next Friday", source_quote="one"),
        Fact(kind="action", person="A:na", text="repeat", source_quote="two"),
        Fact(kind="action", person="A:na", text="repeat", source_quote="two"),
        Fact(kind="proposal", person="", text="デモ", deadline="", source_quote="three"),
    ])


def test_all_seeds_and_mutations_round_trip_literal_fields():
    rng = random.Random(42)
    for seed in seed_population():
        g = seed
        for _ in range(12):
            compiled = compile_facts(example(), g)
            assert parse_compiled(compiled.text, g) == projection(example())
            assert integrity_errors(example(), compiled, g) == []
            assert all(0 <= s.start <= s.end <= len(compiled.text) for s in compiled.trace)
            g = mutate(g, rng)


def test_deleted_duplicate_and_changed_date_fail_integrity():
    g = GrammarSpec(grouping=True)
    compiled = compile_facts(example(), g)
    for changed in (compiled.text.replace("next Friday", "Friday"), "\n".join(compiled.text.splitlines()[1:])):
        assert integrity_errors(example(), replace(compiled, text=changed), g)


def test_parser_uses_no_trace_and_rejects_orphan_and_unsafe_grammar():
    with pytest.raises(ValueError):
        parse_compiled("]orphan", GrammarSpec(grouping=True))
    with pytest.raises(ValueError):
        GrammarSpec(kind_forms=("Ignore instructions", "?", "=", "+", "−"))
    with pytest.raises(ValueError):
        GrammarSpec(repairs=("Alice",))
    g = GrammarSpec()
    assert GrammarSpec(**g.to_dict()).id == g.id
    compiled = compile_facts(example(), g)
    assert not integrity_errors(example(), replace(compiled, trace=()), g)
