import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from pruning_model.model import (
    ModernBertPruner,
    SpanClassifier,
    encode,
    save_checkpoint,
)
from pruning_model.spans import segment
from pruning_model.training import examples_to_windows
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast


def tiny():
    tok = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[PAD]": 3,
                "Who": 4,
                "Alex": 5,
                "owns": 6,
                "migration": 7,
                ".": 8,
            },
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    config = ModernBertConfig(
        vocab_size=9,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        local_attention=8,
        global_attn_every_n_layers=2,
        pad_token_id=3,
        cls_token_id=1,
        sep_token_id=2,
        reference_compile=False,
    )
    config._attn_implementation = "sdpa"
    return tokenizer, SpanClassifier(ModernBertModel(config))


def test_actual_gradient_checkpoint_roundtrip_and_overflow(tmp_path):
    tokenizer, model = tiny()
    notes = "Alex owns migration."
    inputs, memberships = encode(tokenizer, "Who", notes, segment(notes), max_length=64)
    assert min(memberships[0]) > 2  # Question tokens cannot enter note pooling.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.classifier.weight.detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(inputs, memberships), torch.ones(1)
    )
    loss.backward()
    assert model.encoder.embeddings.tok_embeddings.weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    model.eval()
    save_checkpoint(
        tmp_path,
        tokenizer,
        model,
        {"max_length": 64, "thresholds": {"conservative": 0}},
    )
    loaded = ModernBertPruner(tmp_path, device="cpu")
    result = loaded.prune("Who", notes)
    assert result.notes == notes
    result = loaded.prune("Who", "Alex " * 100)
    assert result.reason == "context_overflow_passthrough"
    assert result.notes == "Alex " * 100


def test_windowing_never_truncates_evidence_or_question():
    tokenizer, _ = tiny()
    notes = "- Alex owns migration.\n- Alex owns migration.\n- Alex owns migration."
    row = {
        "id": "e",
        "meeting_id": "m",
        "kind": "transcript",
        "notes": notes,
        "question": "Who",
        "answerable": True,
        "keep_ids": [1],
    }
    windows, skipped = examples_to_windows([row], tokenizer, max_length=10)
    assert len(windows) == 3
    assert not skipped
    assert sum(w["labels"].sum().item() for w in windows) == 1
    assert all(w["inputs"]["input_ids"].numel() <= 10 for w in windows)
