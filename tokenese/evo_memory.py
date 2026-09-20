"""Session-only extraction/compilation; explicit qualification and raw fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .facts import Answer, Fact, MeetingFacts
from .prompts import answer_prompt
from .tokens import count_tokens
from .evo_eval import EXTRACTION_INSTRUCTION, experiment_identity
from .evo_grammar import GrammarSpec, CompiledText, compile_facts, digest, integrity_errors
from .evo_usage import Runner, dollars


class ClassifiedMeeting(BaseModel):
    facts: list[Fact]
    unsupported_content: bool
    ambiguity_notes: list[str]


CLASSIFIER_INSTRUCTION = EXTRACTION_INSTRUCTION + (
    " Also report unsupported_content=true if meaningful information (status, background, metrics, open questions, "
    "or other non-fact content) cannot be represented in these five kinds. Be conservative: extraction cannot "
    "guarantee answers to arbitrary future questions when such content exists. List unresolved ownership, "
    "date, contradiction or scope ambiguities in ambiguity_notes."
)


def classifier_prompt(notes: str) -> str:
    return f"{CLASSIFIER_INSTRUCTION}\n<meeting>\n{notes}\n</meeting>"


def product_identity() -> dict:
    return {"classifier_prompt_hash": digest(CLASSIFIER_INSTRUCTION),
            "classifier_schema_hash": digest(ClassifiedMeeting.model_json_schema()),
            "fallback_policy": "raw-on-unqualified-unsupported-ambiguous-invalid-empty-or-not-cheaper-v1"}


def code_hashes() -> dict:
    directory = Path(__file__).parent
    return {name: digest((directory / name).read_text()) for name in
            ("evo_grammar.py", "evo_eval.py", "evo_memory.py", "facts.py", "prompts.py")}


def verify_frozen(manifest: dict, require_qualified=True) -> GrammarSpec:
    if manifest.get("identity") != experiment_identity() or manifest.get("product_identity") != product_identity():
        raise ValueError("Frozen prompts, schema, model, or tokenizer are incompatible")
    if manifest.get("code_hashes") != code_hashes():
        raise ValueError("Frozen compiler/parser/product code changed")
    grammar = GrammarSpec(**manifest["grammar"])
    if manifest.get("grammar_id") != grammar.id:
        raise ValueError("Frozen grammar hash mismatch")
    if 'freeze_hash' not in manifest or manifest['freeze_hash'] != freeze_digest(manifest):
        raise ValueError('Frozen selection contract changed')
    if require_qualified and (manifest.get("language_status") != "qualified" or manifest.get("product_status") != "qualified"):
        raise ValueError("No qualified product grammar; raw notes required")
    return grammar


def freeze_digest(manifest: dict) -> str:
    keys = ('identity','product_identity','code_hashes','dataset_manifest','candidate_ids','search_hash','grammar')
    return digest({k:manifest[k] for k in keys})


@dataclass
class EvolutionMemory:
    key: str
    notes: str
    facts: MeetingFacts | None = None
    compiled: CompiledText | None = None
    extraction: dict | None = None
    route: str = "raw"
    reason: str = "No qualified grammar"
    ambiguity_notes: list[str] = field(default_factory=list)
    answers: list[dict] = field(default_factory=list)


def memory_key(notes: str, frozen_grammar: dict, research=False) -> str:
    return digest({"notes": notes, "manifest": frozen_grammar, "experiment": experiment_identity(),
                   "product": product_identity(), "code": code_hashes(), "research": research})


def compile_meeting(notes: str, frozen_grammar: dict, client, *, research=False) -> EvolutionMemory:
    if not notes.strip():
        raise ValueError("Meeting notes are blank")
    runner = client if isinstance(client, Runner) else Runner(client, public=False)
    if runner.public or runner.ledger is not None:
        raise ValueError("Unseen private notes require a session-only runner")
    memory = EvolutionMemory(memory_key(notes, frozen_grammar, research), notes)
    try:
        grammar = verify_frozen(frozen_grammar, require_qualified=not research)
    except (ValueError, KeyError) as exc:
        memory.reason = str(exc)
        return memory
    extraction = runner.call(classifier_prompt(notes), ClassifiedMeeting, purpose="extraction", max_output_tokens=4096)
    memory.extraction = extraction
    if extraction.get("error") or not extraction.get("answer"):
        memory.reason = "Extraction failed; raw notes preserve the original evidence"
        return memory
    try:
        classified = ClassifiedMeeting.model_validate(extraction["answer"])
        memory.facts = MeetingFacts(facts=classified.facts)
        memory.ambiguity_notes = classified.ambiguity_notes
        if not classified.facts or any(f.source_quote not in notes for f in classified.facts):
            memory.reason = "Empty extraction or unmatched source evidence"
            return memory
        memory.compiled = compile_facts(memory.facts, grammar)
        if integrity_errors(memory.facts, memory.compiled, grammar):
            memory.reason = "Compiler integrity check failed"
            return memory
        if classified.unsupported_content or classified.ambiguity_notes:
            memory.reason = "Original notes retained for unsupported or ambiguous content"
            return memory
        if count_tokens(answer_prompt(memory.compiled.text, "")) >= count_tokens(answer_prompt(notes, "")):
            memory.reason = "Encoding does not reduce full visible input"
            return memory
        memory.route = "research_encoded" if research else "encoded"
        memory.reason = "Explicit unqualified research mode" if research else "Qualified compile-once route"
    except ValueError as exc:
        memory.reason = f"Invalid extraction: {exc}"
    return memory


def answer_from_memory(memory: EvolutionMemory, question: str, client) -> dict:
    if not question.strip():
        raise ValueError("Question is blank")
    runner = client if isinstance(client, Runner) else Runner(client, public=False)
    if runner.public or runner.ledger is not None:
        raise ValueError("Private answers must remain session-scoped")
    context = memory.compiled.text if memory.route.endswith("encoded") and memory.compiled else memory.notes
    # Question-dependent overhead is identical, but recheck before dispatch.
    route = memory.route
    if context != memory.notes and count_tokens(answer_prompt(context, question)) >= count_tokens(answer_prompt(memory.notes, question)):
        context, route = memory.notes, "raw"
    result = runner.call(answer_prompt(context, question), Answer, purpose="product_answer")
    result["route"] = route
    result["question"] = question
    memory.answers.append(result)
    return result


def workflow_usage(memory: EvolutionMemory) -> dict:
    calls = ([memory.extraction] if memory.extraction else []) + memory.answers
    usage = [r.get("usage") for r in calls]
    if any(u is None for u in usage):
        return {"accounting_status": "incomplete", "questions": len(memory.answers), "usd": None}
    actual = [r['usage'] for r in calls if not r.get('replayed')]
    return {"accounting_status": "complete", "questions": len(memory.answers),
            "input_tokens": sum(u["input_tokens"] for u in actual),
            "output_tokens": sum(u["output_tokens"] for u in actual),
            "usd": sum(dollars(u) for u in actual), "extraction_calls": int(memory.extraction is not None and not memory.extraction.get('replayed')),
            "replayed_requests": sum(bool(r.get('replayed')) for r in calls),
            "logical_request_input_tokens": sum(u['input_tokens'] for u in usage)}
