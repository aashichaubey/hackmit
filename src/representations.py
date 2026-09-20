"""Question-independent semantic extraction and representation generation."""

from __future__ import annotations

import os

STRATEGIES = ("ORIGINAL", "FACTS", "COMPACT_ENGLISH", "SYMBOLIC", "MANDARIN", "HYBRID", "CANONICAL")
GENERATED_STRATEGIES = STRATEGIES[2:]
SEARCH_COMPARISON_STRATEGIES = ("ORIGINAL", "FACTS", "COMPACT_ENGLISH", "SYMBOLIC", "MANDARIN", "HYBRID", "TOKEN_OPTIMIZED")

FACTS_SYSTEM = (
    "Extract factual/propositional content from a source for reuse in unknown future question-answering tasks. "
    "Do not summarize around a presumed question. Preserve names, entities, attributes, quantities, dates, times, "
    "locations, comparisons, causal statements, conditions, exceptions, and relations. Output only a concise bullet list."
)

# Frozen before the real run. Every strategy receives the exact same FACTS text.
STRATEGY_INSTRUCTIONS = {
    "COMPACT_ENGLISH": "Encode every proposition in the FACTS input using extremely concise English. Do not omit or add facts. Output only the encoding.",
    "SYMBOLIC": "Encode every proposition in the FACTS input using compact logic, relations, arrows, abbreviations, or code-like notation. Do not omit or add facts. Output only the encoding.",
    "MANDARIN": "Encode every proposition in the FACTS input in concise Mandarin. Preserve proper nouns and technical terms where useful. Do not omit or add facts. Output only the encoding.",
    "HYBRID": "Encode every proposition in the FACTS input using any compact mix of English, Mandarin, symbols, logic, abbreviations, and code-like syntax. Do not omit or add facts. Output only the encoding.",
    "CANONICAL": "Encode every proposition in deterministic lines formatted SUBJECT|RELATION|OBJECT. Use uppercase snake_case English relation names, one proposition per line, stable source order, literal values, and no stylistic prose. Do not omit or add facts. Output only the encoding.",
}


class LLMProvider:
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if self.provider not in {"openai", "mock"}:
            raise RuntimeError("LLM_PROVIDER must be 'openai' or 'mock'")
        self.generation_model = os.getenv("GENERATION_MODEL", "gpt-4.1-mini")
        self.answer_model = os.getenv("ANSWER_MODEL", self.generation_model)
        self.judge_model = os.getenv("JUDGE_MODEL", self.answer_model)
        self.base_url = os.getenv("OPENAI_BASE_URL") or None
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def complete(self, system: str, user: str, *, purpose: str, temperature: float = 0) -> dict:
        if self.provider == "mock":
            return {"text": user, "usage": {"input_tokens": 0, "output_tokens": 0}, "model": "mock"}
        model = {"generation": self.generation_model, "answer": self.answer_model, "judge": self.judge_model, "semantic_judge": self.judge_model}[purpose]
        request = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        if purpose in {"judge", "semantic_judge"}:
            request["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**request)
        usage = response.usage
        return {
            "text": (response.choices[0].message.content or "").strip(),
            "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens} if usage else {},
            "model": response.model,
        }

    def complete_schema(self, system: str, user: str, *, schema_name: str, schema: dict, temperature: float = 0) -> dict:
        """Generate strict JSON-schema output for question-independent extraction."""
        if self.provider != "openai":
            raise RuntimeError("structured extraction requires LLM_PROVIDER=openai")
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        )
        usage = response.usage
        import json
        return {
            "data": json.loads(response.choices[0].message.content or "{}"),
            "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens} if usage else {},
            "model": response.model,
        }


def extract_facts(context: str, provider: LLMProvider) -> dict:
    """The only semantic-pruning call; structurally accepts no questions."""
    assert isinstance(context, str) and context.strip()
    return provider.complete(FACTS_SYSTEM, f"SOURCE CONTEXT:\n{context}", purpose="generation", temperature=0)


def encode_facts(facts: str, strategy: str, provider: LLMProvider) -> dict:
    """Encode shared facts; structurally accepts no context, questions, or answers."""
    assert strategy in GENERATED_STRATEGIES
    assert isinstance(facts, str) and facts.strip()
    system = "You encode a supplied fact set without selecting among facts. Retain approximately the same semantic content."
    temperature = 0 if strategy == "CANONICAL" else float(os.getenv("GENERATION_TEMPERATURE", "0"))
    return provider.complete(system, f"INSTRUCTION:\n{STRATEGY_INSTRUCTIONS[strategy]}\n\nFACTS:\n{facts}", purpose="generation", temperature=temperature)


def generate_representations(context: str, provider: LLMProvider) -> dict:
    """Generate all representations without any access to the example's questions."""
    facts_result = extract_facts(context, provider)
    result = {"ORIGINAL": {"text": context, "usage": {}}, "FACTS": facts_result}
    for strategy in GENERATED_STRATEGIES:
        result[strategy] = encode_facts(facts_result["text"], strategy, provider)
    return result


TOKEN_SEARCH_SYSTEM = (
    "Encode supplied semantic content for a capable language model. Minimize tokens under an external tokenizer while "
    "preserving every materially important proposition. You do not know future questions. Choose any representation "
    "you judge effective: concise natural language, abbreviations, formal notation, code-like relations, Unicode, other "
    "human languages, invented notation, or mixtures. No form is preferred. Output only the candidate encoding."
)


def generate_token_candidate(facts: str, attempt: int, provider: LLMProvider) -> dict:
    """Generate one question-independent search candidate from shared facts."""
    assert isinstance(facts, str) and facts.strip()
    assert 1 <= attempt <= 10
    prompt = (
        f"SEARCH ATTEMPT {attempt} OF 10. Produce one self-contained encoding of all supplied propositions. "
        "Token count will be measured externally; do not estimate or report it.\n\nSOURCE SEMANTIC CONTENT:\n"
        f"{facts}"
    )
    return provider.complete(TOKEN_SEARCH_SYSTEM, prompt, purpose="generation", temperature=float(os.getenv("TOKEN_SEARCH_TEMPERATURE", "0.8")))


def validate_semantics(facts: str, candidate: str, provider: LLMProvider) -> dict:
    """Blind semantic validation; accepts no strategy identity, token count, or questions."""
    assert facts.strip() and candidate.strip()
    prompt = (
        f"SOURCE SEMANTIC CONTENT:\n{facts}\n\nCANDIDATE REPRESENTATION:\n{candidate}\n\n"
        "Determine whether every materially important proposition in the source remains recoverable from the candidate. "
        "Return JSON only: {\"valid\": true|false, \"reason\": \"brief explanation\"}"
    )
    result = provider.complete(
        "You are a strict semantic-completeness judge. Ignore stylistic differences. Reject omissions, contradictions, changed values, or ambiguous encodings that prevent recovery. You are not given and must not infer token counts or strategy names.",
        prompt,
        purpose="semantic_judge",
        temperature=0,
    )
    import json
    import re
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", result["text"].strip(), flags=re.I)
    verdict = json.loads(cleaned)
    if not isinstance(verdict.get("valid"), bool):
        raise ValueError("semantic judge response lacks boolean 'valid'")
    return {"valid": verdict["valid"], "reason": str(verdict.get("reason", "")), "usage": result["usage"], "cached": result.get("cached", False)}


def search_token_optimized(original: str, facts: str, tokenizer, provider: LLMProvider, candidate_count: int = 10) -> dict:
    """Generate, count, validate, and select candidates without downstream information."""
    assert candidate_count == 10, "the preregistered search uses exactly 10 candidates"
    assert original.strip()
    candidates = [{
        "candidate": 0, "text": original, "token_count": tokenizer.count(original), "valid": True,
        "validation_reason": "Original source baseline; valid by construction.", "source": "original_baseline",
        "generation_usage": {}, "validation_usage": {},
    }]
    for attempt in range(1, candidate_count + 1):
        generated = generate_token_candidate(facts, attempt, provider)
        token_count = tokenizer.count(generated["text"])
        validation = validate_semantics(facts, generated["text"], provider)
        candidates.append({
            "candidate": attempt, "text": generated["text"], "token_count": token_count, "source": "generated",
            "valid": validation["valid"], "validation_reason": validation["reason"],
            "generation_usage": generated["usage"], "validation_usage": validation["usage"],
        })
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise RuntimeError("TOKEN_OPTIMIZED search produced no semantically valid candidate")
    winner = min(valid, key=lambda candidate: (candidate["token_count"], candidate["candidate"]))
    return {"winner": winner, "candidates": candidates}
