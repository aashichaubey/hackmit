"""Bounded, seeded grammar evolution. No model-generated code or aliases."""
import random

from .evo_grammar import GrammarSpec, KINDS, VOCAB, OWNER_MARKERS, DUE_MARKERS, compile_facts, digest, integrity_errors
from .evo_eval import context_for
from .facts import MeetingFacts
from .prompts import answer_prompt
from .tokens import count_tokens


def seed_population() -> list[GrammarSpec]:
    grammars = [GrammarSpec(), GrammarSpec(layout="prose", separator=";"),
                GrammarSpec(kind_forms=("do", "proposed", "decided", "agreed", "disagreed"), layout="actor", separator="|")]
    for forms in (tuple(VOCAB[k][i] for k in KINDS) for i in range(4)):
        for layout in ("actor", "kind", "content"):
            for grouping in (False, True):
                grammars.append(GrammarSpec(kind_forms=forms, layout=layout, grouping=grouping, separator="|"))
    return list({g.id: g for g in grammars}.values())


def mutate(grammar: GrammarSpec, rng: random.Random) -> GrammarSpec:
    field = rng.choice(("kind_forms", "layout", "owner_marker", "deadline_marker", "separator", "grouping"))
    if field == "kind_forms":
        forms = list(grammar.kind_forms)
        i = rng.randrange(5)
        forms[i] = rng.choice(VOCAB[KINDS[i]])
        return grammar.edit(kind_forms=tuple(forms))
    values = {"layout": ("actor", "kind", "content"), "owner_marker": OWNER_MARKERS,
              "deadline_marker": DUE_MARKERS, "separator": (":", "|", ";", "\t"), "grouping": (False, True)}
    change = rng.choice(values[field])
    if field == "grouping" and grammar.layout == "prose":
        change = False
    return grammar.edit(**{field: change})


def crossover(left: GrammarSpec, right: GrammarSpec) -> GrammarSpec:
    return left.edit(kind_forms=left.kind_forms[:2] + right.kind_forms[2:])


def screen(grammar: GrammarSpec, dev_cases: list[dict]) -> dict:
    if any(c["split"] != "dev" for c in dev_cases):
        raise ValueError("Local search only accepts development cases")
    outputs, errors = [], []
    full_tokens = context_tokens = 0
    for case in dev_cases:
        facts = MeetingFacts(facts=case["facts"])
        compiled = compile_facts(facts, grammar)
        errors.extend(integrity_errors(facts, compiled, grammar))
        outputs.append(compiled.text)
        context_tokens += count_tokens(compiled.text)
        full_tokens += sum(count_tokens(answer_prompt(compiled.text, q["question"])) for q in case["questions"])
    return {"grammar_id": grammar.id, "grammar": grammar.to_dict(), "integrity_errors": errors,
            "context_tokens": context_tokens, "visible_input_tokens": full_tokens, "output_hash": digest(outputs)}
