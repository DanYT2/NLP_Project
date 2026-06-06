"""Tests for src/nlp_project/llama_classify.py.

Fast tests exercise the pure-Python helpers (prompt construction, demo
selection, label parsing). The single slow smoke test wires the full
single-document classification path through a tiny GPT-2 with a hand-set
chat template — it produces garbage output but verifies the plumbing
(tokenizer → generate → decode → parse) holds together end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlp_project import llama_classify as lc


LABEL_NAMES = [f"class.{i}" for i in range(20)]


# ---------------- fast unit tests ----------------


def test_build_prompt_zero_shot_has_no_assistant_turn() -> None:
    msgs = lc.build_prompt("hello", LABEL_NAMES, demos=None)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Allowed labels" in msgs[0]["content"]
    assert "hello" in msgs[1]["content"]


def test_build_prompt_few_shot_alternates_user_assistant() -> None:
    demos = [("demo doc one", 3), ("demo doc two", 7)]
    msgs = lc.build_prompt("query doc", LABEL_NAMES, demos=demos)
    # system + 2 demos * (user+assistant) + final user = 6
    assert len(msgs) == 6
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert msgs[2]["content"] == LABEL_NAMES[3]
    assert msgs[4]["content"] == LABEL_NAMES[7]
    assert "query doc" in msgs[5]["content"]


def test_build_prompt_truncates_long_docs() -> None:
    long_doc = "x" * 5000
    msgs = lc.build_prompt(long_doc, LABEL_NAMES, truncate_chars=1500)
    # The user content is "<truncated>\n\nLabel:" — length is truncate + suffix.
    assert len(msgs[-1]["content"]) <= 1500 + len("\n\nLabel:") + 1


def test_select_demos_returns_k_per_class_and_correct_label() -> None:
    rng = np.random.default_rng(0)
    train_labels = np.repeat(np.arange(20), 50)
    train_docs = [f"doc {i}" for i in range(len(train_labels))]
    rng.shuffle(train_labels)
    out = lc.select_demos(train_docs, train_labels, LABEL_NAMES, k_per_class=3)
    assert len(out) == 60
    by_class: dict[int, int] = {}
    for _, label_idx in out:
        by_class[label_idx] = by_class.get(label_idx, 0) + 1
    assert all(v == 3 for v in by_class.values())
    assert set(by_class.keys()) == set(range(20))


def test_select_demos_is_deterministic() -> None:
    train_labels = np.repeat(np.arange(20), 10)
    train_docs = [f"d{i}" for i in range(len(train_labels))]
    a = lc.select_demos(train_docs, train_labels, LABEL_NAMES, k_per_class=2, seed=123)
    b = lc.select_demos(train_docs, train_labels, LABEL_NAMES, k_per_class=2, seed=123)
    assert a == b


def test_parse_label_exact_and_substring() -> None:
    idx, invalid = lc.parse_label("class.5", LABEL_NAMES)
    assert (idx, invalid) == (5, False)
    # Llama-style verbose output that still contains the label as substring.
    idx, invalid = lc.parse_label("The label is class.12", LABEL_NAMES)
    assert (idx, invalid) == (12, False)


def test_parse_label_fuzzy_fallback() -> None:
    # "clas 9" should fuzzy-match "class.9" (normalized to "class 9").
    idx, invalid = lc.parse_label("class 9", LABEL_NAMES)
    assert (idx, invalid) == (9, False)


def test_parse_label_invalid() -> None:
    idx, invalid = lc.parse_label("not_a_label_at_all", LABEL_NAMES)
    assert invalid is True
    assert 0 <= idx < len(LABEL_NAMES)


# ---------------- slow integration smoke ----------------


@pytest.mark.slow
def test_classify_one_smoke_with_tiny_gpt2(tmp_path) -> None:
    """Wire ``classify_one`` through a tiny model end-to-end.

    We use ``sshleifer/tiny-gpt2`` (random-init style tiny model, ~5 MB) and
    install a minimal Jinja2 chat template manually — GPT-2 ships without
    one. The generated output will be nonsense, so we don't assert on the
    label index *value*; we assert on the return shape and that the
    ``invalid`` flag is a bool. This catches dtype/device wiring bugs and
    chat-template regressions without needing a multi-GB Llama download.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    model.eval()

    pred_idx, raw, invalid = lc.classify_one(
        model, tok,
        doc="The space shuttle launched yesterday from Cape Canaveral.",
        label_names=LABEL_NAMES,
        max_new_tokens=5,
        truncate_chars=200,
    )
    assert isinstance(pred_idx, int)
    assert 0 <= pred_idx < len(LABEL_NAMES)
    assert isinstance(raw, str)
    assert isinstance(invalid, (bool, np.bool_))
