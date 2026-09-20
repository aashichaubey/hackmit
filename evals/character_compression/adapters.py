"""Adapters isolating the compressor, target LLM, and token accounting.

Three contracts, each with a real implementation and an explicitly-labelled
offline mock:

    compress(messages, compression_config)  -> CompressionResult
    generate(messages, generation_config)   -> GenerationResult
    count_request_tokens(messages)          -> TokenCount

Two invariants are enforced structurally rather than by convention:

1. **No gold leakage.** `compress` and `generate` accept a list of plain
   message dicts, never a fixture object. `assert_no_gold_fields` rejects any
   payload carrying `expected_output`/`grader`/`cluster_id`, so a leak is a
   crash rather than a silently optimistic score.

2. **No silent mocking in live runs.** Every adapter declares `is_mock`. The
   runner refuses to start a live experiment if any selected adapter is a mock,
   because a mock compressor would report the identity transform as compression.

Ownership note: the *compressor* alone appends dictionary and decoder
instructions. The runner never adds them, so overhead cannot be double-counted.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

Messages = list[dict[str, str]]

# Fields that exist only in the fixture and must never reach inference.
GOLD_FIELDS = frozenset({"expected_output", "grader", "cluster_id", "split", "source"})

UNKNOWN = None  # explicit: a missing measurement is unknown, never zero


class GoldLeakageError(AssertionError):
    """Raised when evaluation-only data reaches an inference payload."""


def assert_no_gold_fields(payload: Any, where: str) -> None:
    """Recursively verify no gold/grading metadata is present."""
    if isinstance(payload, dict):
        leaked = GOLD_FIELDS & payload.keys()
        if leaked:
            raise GoldLeakageError(
                f"{where}: inference payload contains gold field(s) {sorted(leaked)}"
            )
        for v in payload.values():
            assert_no_gold_fields(v, where)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            assert_no_gold_fields(v, where)


def sanitize_messages(messages: Messages) -> Messages:
    """Return a defensive copy containing only role/content."""
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    assert_no_gold_fields(clean, "sanitize_messages")
    return clean


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class CompressionResult:
    messages: Messages
    #: Free-form, recorded verbatim into predictions.compression_metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = UNKNOWN
    compressor_version: str = "unknown"
    dictionary_version: str = "unknown"
    #: True when the compressor fell back to emitting the original messages.
    fell_back: bool = False


@dataclass
class GenerationResult:
    output: str
    input_tokens: int | None = UNKNOWN
    output_tokens: int | None = UNKNOWN
    latency_ms: float | None = UNKNOWN
    model_id: str = "unknown"
    provider_request_id: str | None = UNKNOWN
    token_count_source: str | None = UNKNOWN
    cost_usd: float | None = UNKNOWN
    error: str | None = None
    #: Reasoning-model bookkeeping. `finish_reason == "length"` with an empty
    #: answer means the output budget was consumed before any answer was
    #: emitted - a measurement artifact, recorded as an error so it is never
    #: silently graded as a wrong answer.
    finish_reason: str | None = UNKNOWN
    reasoning_tokens: int | None = UNKNOWN

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass
class TokenCount:
    tokens: int
    #: One of: provider_usage | exact_local_tokenizer | approximate_local_template
    source: str

    @property
    def is_exact(self) -> bool:
        """Only exact counts may back a release gate."""
        return self.source in ("provider_usage", "exact_local_tokenizer")


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Compressor(Protocol):
    name: str
    is_mock: bool

    def compress(self, messages: Messages, config: dict[str, Any]) -> CompressionResult: ...


class Target(Protocol):
    name: str
    is_mock: bool

    def generate(self, messages: Messages, config: dict[str, Any]) -> GenerationResult: ...


class TokenCounter(Protocol):
    name: str

    def count_request_tokens(self, messages: Messages) -> TokenCount: ...


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

# DeepSeek does not publish a chat_template in tokenizer_config.json, so a
# locally-computed *full request* count must reconstruct the framing by hand.
# That reconstruction is an approximation and is labelled as such; it is
# excluded from release gates. Provider-reported usage is the exact path.
_DS_BOS = "<｜begin▁of▁sentence｜>"
_DS_USER = "<｜User｜>"
_DS_ASSISTANT = "<｜Assistant｜>"
_DS_EOS = "<｜end▁of▁sentence｜>"


class DeepSeekTokenCounter:
    """Exact per-string tokenization; approximate whole-request framing."""

    name = "deepseek_local_tokenizer"

    def __init__(self, model: str = "deepseek-v4-flash", tokenizer_dir: str | Path | None = None):
        import sys

        repo_src = Path(__file__).resolve().parents[2] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from token_language.tokenizer import HFTokenizer  # noqa: E402

        # Resolve the cache against the repository root, not the current working
        # directory, so running from evals/ does not create a second 6MB copy
        # of the tokenizer.
        cache = Path(tokenizer_dir) if tokenizer_dir else repo_src.parent / "data/raw/tokenizers"
        self._tk = HFTokenizer.from_pretrained(model, cache_dir=cache)
        self.tokenizer_id = self._tk.id

    def render(self, messages: Messages) -> str:
        """Reconstruct the DeepSeek chat framing around the messages."""
        parts = [_DS_BOS]
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"{_DS_USER}{content}")
            elif role == "assistant":
                parts.append(f"{_DS_ASSISTANT}{content}{_DS_EOS}")
        parts.append(_DS_ASSISTANT)
        return "".join(parts)

    def count_request_tokens(self, messages: Messages) -> TokenCount:
        return TokenCount(
            tokens=self._tk.count_tokens(self.render(messages)),
            source="approximate_local_template",
        )

    def count_text(self, text: str) -> int:
        """Exact count for a bare string (no chat framing involved)."""
        return self._tk.count_tokens(text)


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------


class RepoCompressor:
    """Adapter for this repository's real hybrid encoder.

    The encoder is not implemented yet (`src/token_language/encoder/` is empty),
    so construction fails loudly with the precise missing integration rather
    than degrading to an identity transform.
    """

    name = "repo_hybrid_encoder"
    is_mock = False

    def __init__(self, dictionary_path: str | Path, **kwargs: Any) -> None:
        import sys

        repo_src = Path(__file__).resolve().parents[2] / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        try:
            from token_language.encoder.encoder import Encoder  # type: ignore # noqa
        except ImportError as exc:
            raise NotImplementedError(
                "RepoCompressor requires token_language.encoder.encoder.Encoder, which "
                "does not exist yet. Missing integration: (1) build the phrase "
                "dictionary via scripts/build_dictionary.py, (2) implement the "
                "Encoder with .encode(text) -> EncodingResult. Until then use "
                "--compressor mock for offline harness tests; live mode is refused."
            ) from exc
        self._encoder = Encoder.from_dictionary(dictionary_path, **kwargs)
        self.dictionary_path = str(dictionary_path)

    def compress(self, messages: Messages, config: dict[str, Any]) -> CompressionResult:
        raise NotImplementedError  # pragma: no cover - unreachable until Encoder lands


class MockCompressor:
    """Deterministic, offline stand-in used ONLY for harness tests.

    Applies a tiny fixed English->Chinese codebook and prepends a decoder
    instruction, so the harness exercises realistic dictionary overhead,
    substitution metadata, and expansion behaviour. It is NOT a real
    compressor and is refused in live mode.
    """

    name = "mock_compressor"
    is_mock = True
    VERSION = "mock-0.1.0"
    DICTIONARY_VERSION = "mock-dict-0.1.0"

    CODEBOOK: dict[str, str] = {
        "Return only the name.": "只回名。",
        "Answer only YES or NO.": "只答YES或NO。",
        "source data": "源数据",
        "the following": "下述",
    }

    def __init__(self, scope: str = "user_only", include_dictionary: bool = True):
        self.scope = scope
        self.include_dictionary = include_dictionary

    def _decoder_preamble(self, used: dict[str, str]) -> str:
        lines = [f"{zh} = {en}" for en, zh in used.items()]
        return (
            "Some phrases below use Chinese-character codes. Decode them with "
            "this codebook before answering:\n" + "\n".join(lines)
        )

    def compress(self, messages: Messages, config: dict[str, Any]) -> CompressionResult:
        assert_no_gold_fields(messages, "MockCompressor.compress")
        started = time.perf_counter()

        out: Messages = []
        used: dict[str, str] = {}
        substitutions = 0
        for m in sanitize_messages(messages):
            content = m["content"]
            # Scope control: only rewrite the roles the config allows.
            if self.scope == "user_only" and m["role"] != "user":
                out.append(m)
                continue
            for en, zh in self.CODEBOOK.items():
                if en in content:
                    content = content.replace(en, zh)
                    used[en] = zh
                    substitutions += 1
            out.append({"role": m["role"], "content": content})

        if used and self.include_dictionary:
            preamble = self._decoder_preamble(used)
            if out and out[0]["role"] == "system":
                out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + preamble}
            else:
                out.insert(0, {"role": "system", "content": preamble})

        return CompressionResult(
            messages=out,
            metadata={
                "substitutions": substitutions,
                "codebook_entries_used": len(used),
                "scope": self.scope,
                "dictionary_included_by": "compressor",
                "synthetic": True,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
            compressor_version=self.VERSION,
            dictionary_version=self.DICTIONARY_VERSION,
        )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


class OpenRouterTarget:
    """Real target-LLM adapter (OpenRouter chat completions).

    Reads the key from the environment only; it is never accepted via config
    and never written to logs.
    """

    name = "openrouter"
    is_mock = False
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-flash",
        timeout: float = 60.0,
        api_key_env: str = "OPENROUTER_API_KEY",
        pricing: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._key = os.environ.get(api_key_env)
        if not self._key:
            raise RuntimeError(
                f"{api_key_env} is not set. Export it (or put it in .env) to run live mode; "
                "credentials are never read from the config file."
            )
        # Pricing must be supplied explicitly; never fetched or assumed.
        self.pricing = pricing

    def generate(self, messages: Messages, config: dict[str, Any]) -> GenerationResult:
        assert_no_gold_fields(messages, "OpenRouterTarget.generate")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": sanitize_messages(messages),
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 256),
        }
        if (seed := config.get("seed")) is not None:
            body["seed"] = seed

        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = _redact(exc.read().decode("utf-8", "replace")[:400])
            return GenerationResult(
                output="",
                latency_ms=(time.perf_counter() - started) * 1000,
                model_id=self.model,
                error=f"HTTP {exc.code}: {detail}",
            )
        except Exception as exc:  # network/timeout/parse
            return GenerationResult(
                output="",
                latency_ms=(time.perf_counter() - started) * 1000,
                model_id=self.model,
                error=f"{type(exc).__name__}: {_redact(str(exc))[:300]}",
            )
        latency = (time.perf_counter() - started) * 1000

        if err := payload.get("error"):
            return GenerationResult(
                output="", latency_ms=latency, model_id=self.model,
                error=f"provider_error: {_redact(json.dumps(err))[:300]}",
            )
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return GenerationResult(
                output="", latency_ms=latency, model_id=self.model,
                error=f"malformed_response: {_redact(json.dumps(payload))[:300]}",
            )

        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        finish_reason = choice.get("finish_reason")
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

        # DeepSeek V4 Flash is a reasoning model: it may return content=null
        # having spent the entire output budget on reasoning tokens. That is a
        # truncation artifact of the harness's max_tokens setting, not a wrong
        # answer, so it is surfaced as an explicit error rather than being
        # graded as an incorrect response.
        output = content if isinstance(content, str) else ""
        error = None
        if not output.strip() and finish_reason == "length":
            error = (
                f"truncated_before_answer: finish_reason=length with empty content "
                f"(reasoning_tokens={reasoning_tokens}). Raise generation.max_tokens."
            )

        return GenerationResult(
            output=output,
            input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else UNKNOWN,
            output_tokens=usage.get("completion_tokens") if isinstance(
                usage.get("completion_tokens"), int) else UNKNOWN,
            latency_ms=latency,
            model_id=payload.get("model", self.model),
            provider_request_id=payload.get("id"),
            token_count_source="provider_usage" if isinstance(prompt_tokens, int) else UNKNOWN,
            cost_usd=self._cost(usage),
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
            error=error,
        )

    def _cost(self, usage: dict[str, Any]) -> float | None:
        """Actual billed cost when the provider reports it; else supplied pricing.

        Provider-reported `usage.cost` is observed billing data and is
        preferred. Pricing is never fetched or assumed - if neither source is
        available the cost stays unknown.
        """
        observed = usage.get("cost")
        if isinstance(observed, (int, float)):
            return float(observed)
        if not self.pricing:
            return UNKNOWN
        pin, pout = self.pricing.get("input_per_token"), self.pricing.get("output_per_token")
        tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if None in (pin, pout, tin, tout):
            return UNKNOWN
        return tin * pin + tout * pout


class MockTarget:
    """Deterministic offline target used ONLY for harness tests.

    Answers from the message text with simple rules. It is intentionally
    imperfect so the harness exercises all four paired outcomes. Refused in
    live mode.
    """

    name = "mock_target"
    is_mock = True

    def __init__(self, counter: TokenCounter | None = None, wrong_case_ids: set[str] | None = None):
        self._counter = counter
        self._wrong = wrong_case_ids or set()

    def generate(self, messages: Messages, config: dict[str, Any]) -> GenerationResult:
        assert_no_gold_fields(messages, "MockTarget.generate")
        started = time.perf_counter()
        text = "\n".join(m["content"] for m in messages)
        forced = config.get("_mock_force_output")
        output = forced if forced is not None else _mock_answer(text)
        tokens = self._counter.count_request_tokens(messages) if self._counter else None
        return GenerationResult(
            output=output,
            input_tokens=tokens.tokens if tokens else UNKNOWN,
            output_tokens=len(output.split()) if output else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model_id="mock-target-v1",
            token_count_source=tokens.source if tokens else UNKNOWN,
        )


def _mock_answer(text: str) -> str:
    """Crude deterministic responder - synthetic, not a model."""
    if "JSON object" in text:
        return '{"Mira": 2, "Tovan": 1}'
    if "YES or NO" in text:
        return "YES" if "Not all" in text or "not all" in text else "NO"
    if "APPROVE or REJECT" in text:
        return "APPROVE" if "at least" in text else "REJECT"
    return "UNKNOWN"


_SECRET_HINTS = ("sk-or-", "Bearer ", "api_key", "Authorization")


def _redact(text: str) -> str:
    """Strip anything resembling a credential from error text."""
    out = text
    for hint in _SECRET_HINTS:
        idx = 0
        while (idx := out.find(hint, idx)) != -1:
            end = idx + len(hint)
            while end < len(out) and (out[end].isalnum() or out[end] in "-_."):
                end += 1
            out = out[:idx] + hint + "[REDACTED]" + out[end:]
            idx = idx + len(hint) + len("[REDACTED]")
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPRESSORS = {"mock": MockCompressor, "repo": RepoCompressor}
TARGETS = {"mock": MockTarget, "openrouter": OpenRouterTarget}
COUNTERS = {"deepseek_local": DeepSeekTokenCounter}
