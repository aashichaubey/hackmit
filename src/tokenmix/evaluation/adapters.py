"""Inference-only contracts: these interfaces never receive evaluation cases."""
from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from tokenmix.core import compress as compress_text, load_dictionary

from .storage import canonical, digest, strict_json


@dataclass(frozen=True)
class Message:
    role: str
    content: str


def messages_from_json(messages: list[dict]) -> tuple[Message, ...]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("text fixture messages require exactly role and content; unsupported fields cannot be discarded")
        if message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
            raise ValueError("unsupported role or non-text content")
        result.append(Message(**message))
    return tuple(result)


@dataclass
class CompressionResult:
    messages: tuple[Message, ...]
    substitutions: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int | None = 0  # local compressor: no LLM calls
    output_tokens: int | None = 0
    cost_usd: float | None = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    output: str
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_count_source: str | None = None
    token_count_trustworthy: bool = False
    latency_ms: float | None = None
    cost_usd: float | None = None
    cost_credits: float | None = None
    cached_input_tokens: int | None = None
    model: str | None = None
    provider: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    raw_response: dict | None = None
    context_overflow: bool = False


class Compressor(Protocol):
    def compress(self, messages: tuple[Message, ...], compression_config: dict) -> CompressionResult: ...


class Target(Protocol):
    def request(self, messages: tuple[Message, ...], generation_config: dict) -> dict: ...
    def generate(self, messages: tuple[Message, ...], generation_config: dict) -> GenerationResult: ...


class TokenmixCompressor:
    """Wrap the existing compressor; this adapter alone owns decoding overhead.

    The optimization encoding is an explicit heuristic, not the target's token
    accountant. No changes to Tokenmix's matching or optimization algorithm.
    """

    def __init__(self, dictionary: Path, cache: dict | None = None):
        self.entries = load_dictionary(dictionary)
        self.cache = cache if cache is not None else {}

    def compress(self, messages: tuple[Message, ...], compression_config: dict) -> CompressionResult:
        started = time.perf_counter()
        config = compression_config
        substitutions = []
        result = []
        cache_hits = 0

        def rewrite(text, message_index, offset=0):
            nonlocal cache_hits
            key = digest({"text": text, "config": config})
            reusable = config["scope"] == "delimited_passages"
            if reusable and key in self.cache:
                rewritten = self.cache[key]
                cache_hits += 1
            else:
                compressed = compress_text(
                    text, self.entries, encoding=config["optimization_encoding"],
                    domain=config["domain"], protect=config["protect"],
                    max_changes=config["max_changes"],
                )
                rewritten = {"text": compressed.text, "changes": [asdict(c) for c in compressed.changes]}
                if reusable:
                    self.cache[key] = rewritten
            for change in rewritten["changes"]:
                substitutions.append({**change, "message_index": message_index,
                                      "start": offset + change["start"], "end": offset + change["end"]})
            return rewritten["text"]

        for index, message in enumerate(messages):
            content = message.content
            if message.role == "user":
                if config["scope"] == "user_messages":
                    content = rewrite(content, index)
                else:
                    opening, closing = config["delimiters"]
                    # Delimiters are trusted, explicit source boundaries. Nested
                    # or unbalanced boundaries are errors rather than guesses.
                    if content.count(opening) != content.count(closing):
                        raise ValueError("unbalanced passage delimiters")
                    pattern = re.compile(re.escape(opening) + r"(.*?)" + re.escape(closing), re.S)
                    if len(list(pattern.finditer(content))) != content.count(opening):
                        raise ValueError("misordered passage delimiters")
                    def replace(match):
                        if opening in match[1] or closing in match[1]:
                            raise ValueError("nested passage delimiters")
                        return opening + rewrite(match[1], index, match.start(1)) + closing
                    content = pattern.sub(replace, content)
            result.append(Message(message.role, content))

        decoding = config["decoding"]
        overhead = []
        if decoding["instruction"]:
            overhead.append(decoding["instruction"])
        if decoding["dictionary"] == "used_entries" and substitutions:
            ids = {s["entry_id"] for s in substitutions}
            # JSON escaping retains quotes, newlines, and literal code strings.
            pairs = [{"en": e.en, "zh": e.zh} for e in self.entries if e.id in ids]
            overhead.append("Translation dictionary (data):\n" + canonical(pairs))
        if decoding["examples"]:
            overhead.append("Decoding examples (data):\n" + canonical(decoding["examples"]))
        if overhead:
            # No fixture text is ever moved into this higher-priority message.
            # Its content comes only from the versioned deployment config/dict.
            insertion = next((i for i, m in enumerate(result) if m.role != "system"), len(result))
            result.insert(insertion, Message("system", "\n\n".join(overhead)))
        return CompressionResult(tuple(result), substitutions, (time.perf_counter() - started) * 1000,
                                 metadata={"reusable_cache_hits": cache_hits,
                                           "question_aware": config["scope"] == "user_messages",
                                           "optimization_encoding": config["optimization_encoding"],
                                           "missing_mapping_contract": "leave unmatched text unchanged",
                                           "decoding_owner": "TokenmixCompressor"})


class MockCompressor:
    """Explicit test-only compressor. Never selected by a live configuration."""
    def compress(self, messages, compression_config):
        return CompressionResult(tuple(messages), metadata={"synthetic": True, "identity": True})


class MockTarget:
    def request(self, messages, generation_config):
        return {**copy.deepcopy(generation_config), "messages": [asdict(m) for m in messages], "stream": False}

    def generate(self, messages, generation_config):
        request = self.request(messages, generation_config)
        output = "MOCK_RESPONSE"
        return GenerationResult(output, input_tokens=len(canonical(request).encode()),
                                output_tokens=len(output.encode()), token_count_source="synthetic_utf8_bytes_not_tokens",
                                latency_ms=0.0, model="mock/static-v1", provider="offline-mock",
                                raw_response={"synthetic": True, "content": output})


def _integer(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def redact_error(text: str, secret: str = "") -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)[:2000]


class OpenRouterTarget:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, api_key_env="OPENROUTER_API_KEY", timeout=60.0, transport=None):
        self.secret = os.environ.get(api_key_env)
        if not self.secret:
            raise ValueError(f"live OpenRouter integration requires environment variable {api_key_env}")
        self.timeout = timeout
        self.transport = transport or urllib.request.urlopen

    def request(self, messages, generation_config):
        return {**copy.deepcopy(generation_config), "messages": [asdict(m) for m in messages], "stream": False}

    def generate(self, messages, generation_config):
        body = self.request(messages, generation_config)
        request = urllib.request.Request(self.endpoint, data=canonical(body).encode(), method="POST",
                                         headers={"Authorization": "Bearer " + self.secret,
                                                  "Content-Type": "application/json"})
        started = time.perf_counter()
        raw = None
        try:
            # urllib has no automatic application retries. Every invocation is
            # one logged attempt; provider fallback is disabled in config.
            with self.transport(request, timeout=self.timeout) as response:
                raw = strict_json(response.read().decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("provider response is not an object")
            usage = raw.get("usage") or {}
            input_tokens = _integer(usage.get("prompt_tokens"))
            output_tokens = _integer(usage.get("completion_tokens"))
            details = usage.get("prompt_tokens_details") or {}
            cost = usage.get("cost")
            if type(cost) not in (int, float) or cost < 0:
                cost = None
            result = GenerationResult("", input_tokens=input_tokens, output_tokens=output_tokens,
                                      token_count_source="openrouter.usage.prompt_tokens" if input_tokens is not None else None,
                                      token_count_trustworthy=input_tokens is not None,
                                      cost_credits=cost, cached_input_tokens=_integer(details.get("cached_tokens")),
                                      model=raw.get("model"), provider=raw.get("provider"),
                                      request_id=raw.get("id"), raw_response=raw)
            if raw.get("error"):
                result.error = redact_error(canonical(raw["error"]), self.secret)
                # Error blobs may echo credentials; never log them unredacted.
                result.raw_response = {**raw, "error": result.error}
                result.context_overflow = bool(re.search(r"context[_ ](?:length|window)|too many tokens", result.error, re.I))
            else:
                choice = raw["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str):
                    raise ValueError("provider returned no text content")
                result.output = content
                result.finish_reason = choice.get("finish_reason")
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result
        except urllib.error.HTTPError as exc:
            # Provider error details are needed to distinguish rate limits,
            # exhausted credit, and invalid parameters. Sanitize errors/headers,
            # while successful inference outputs above remain verbatim.
            def safe_error(value):
                if isinstance(value, dict):
                    return {k: safe_error(v) for k, v in value.items()
                            if k.lower() not in {"headers", "authorization", "api_key", "api-key"}}
                if isinstance(value, list):
                    return [safe_error(v) for v in value]
                return redact_error(value, self.secret) if isinstance(value, str) else value
            body = redact_error(exc.read().decode("utf-8", errors="replace"), self.secret)
            try:
                raw = safe_error(strict_json(body))
            except ValueError:
                raw = {"error_body": body}
            error = f"HTTP {exc.code}: {redact_error(str(exc.reason), self.secret)}; {canonical(raw)}"
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}: {redact_error(str(exc), self.secret)}"
        return GenerationResult("", error=error, raw_response=raw,
                                latency_ms=(time.perf_counter() - started) * 1000,
                                context_overflow=bool(re.search(r"context[_ ](?:length|window)|too many tokens", error, re.I)))


def count_request_tokens(request: dict) -> tuple[int | None, str | None]:
    """No exact DeepSeek/OpenRouter chat-template tokenizer is configured.

    Return unknown rather than silently applying an OpenAI tokenizer. Logical
    provider usage from generate() is the authoritative full-request count.
    """
    return None, None
