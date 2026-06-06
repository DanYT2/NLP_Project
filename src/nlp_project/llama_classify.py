"""Llama-3 zero-shot and few-shot classification on 20 Newsgroups (Q4).

Design notes
------------

* **Why a generative classifier?** The spec asks for *zero-shot and few-shot
  with Llama-3*, which in modern usage means prompt the instruction-tuned
  model and parse its free-text output. A logit-scoring alternative (compute
  ``log P(label | prompt)`` over the 20 labels and argmax) is always valid by
  construction but requires one forward pass per label per document and does
  not test the model's instruction-following — which is the actual capability
  under evaluation. We go generative and track an ``invalid_rate`` diagnostic.

* **Chat template, not raw text.** ``tokenizer.apply_chat_template`` keeps the
  prompt in the exact format the instruct model was post-trained on. Building
  raw ``<|start_header_id|>`` strings by hand is brittle across Llama versions.

* **Greedy decoding for reproducibility.** ``do_sample=False, num_beams=1``
  means a fixed seed isn't even strictly necessary for generation, but we
  still call :func:`nlp_project.set_seed` upstream for consistency with the
  rest of the project.

* **One doc at a time in ``classify_batch``.** Batching chat-template inputs
  requires left-padding and per-row attention masks; the bookkeeping is
  fiddly and the speedup is modest on a single 6 GB GPU. We iterate. For 200
  test docs at ~2-3 s/doc on a 3060 in 4-bit nf4 this is ~10 min/run.

* **bitsandbytes is optional.** ``load_llama(quantize_4bit=True)`` builds a
  :class:`BitsAndBytesConfig` only when both ``bitsandbytes`` is importable
  *and* CUDA is available — otherwise it falls back to bf16/fp16. This keeps
  the slow smoke test runnable on macOS / Colab CPU.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import SEED

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a text classifier for the 20 Newsgroups dataset. Given a Usenet "
    "post, respond with exactly one label from the following list and nothing "
    "else — no explanation, no punctuation around the label.\n\n"
    "Allowed labels: {labels}"
)


def build_prompt(
    doc: str,
    label_names: list[str],
    demos: list[tuple[str, int]] | None = None,
    truncate_chars: int = 1500,
) -> list[dict[str, str]]:
    """Build the chat-template messages list for one document.

    Parameters
    ----------
    doc:
        Raw document text. Truncated to ``truncate_chars`` characters before
        insertion so the prompt stays comfortably within the model's context
        window even at k=3/class (60 demos).
    label_names:
        The 20 newsgroup category names; inserted verbatim into the system
        message so the model sees exactly the strings the parser will match.
    demos:
        Optional list of ``(demo_doc, demo_label_index)`` tuples. Each demo
        becomes two messages — a user turn with the demo doc and an
        assistant turn with the corresponding label string. Demo docs are
        truncated to the same character budget.
    truncate_chars:
        Per-document character cap.

    Returns
    -------
    A list of ``{"role": ..., "content": ...}`` dicts ready for
    :meth:`PreTrainedTokenizerBase.apply_chat_template`.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(labels=", ".join(label_names))},
    ]
    for demo_doc, demo_label in (demos or []):
        messages.append({"role": "user", "content": demo_doc[:truncate_chars] + "\n\nLabel:"})
        messages.append({"role": "assistant", "content": label_names[demo_label]})
    messages.append({"role": "user", "content": doc[:truncate_chars] + "\n\nLabel:"})
    return messages


# ---------------------------------------------------------------------------
# Few-shot demonstration selection
# ---------------------------------------------------------------------------


def select_demos(
    train_docs: Sequence[str],
    train_labels: np.ndarray | Sequence[int],
    label_names: list[str],
    k_per_class: int,
    seed: int = SEED,
) -> list[tuple[str, int]]:
    """Stratified demo selection — ``k_per_class`` docs per newsgroup.

    Deterministic given ``seed``. We seed a *local* ``np.random.default_rng``
    so callers can sample with different seeds without disturbing the global
    NumPy state that the rest of the pipeline depends on.
    """
    rng = np.random.default_rng(seed)
    train_labels = np.asarray(train_labels)
    out: list[tuple[str, int]] = []
    for class_idx in range(len(label_names)):
        candidates = np.flatnonzero(train_labels == class_idx)
        if len(candidates) == 0:
            continue
        take = min(k_per_class, len(candidates))
        chosen = rng.choice(candidates, size=take, replace=False)
        for idx in chosen:
            out.append((train_docs[int(idx)], int(class_idx)))
    return out


# ---------------------------------------------------------------------------
# Label parsing (generative output -> class index)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip, and replace dots/underscores with spaces for matching."""
    return text.strip().lower().replace(".", " ").replace("_", " ")


def parse_label(raw_output: str, label_names: list[str]) -> tuple[int, bool]:
    """Map a generated string to a label index.

    Returns ``(label_index, invalid_flag)``. When no match is found, returns
    ``(0, True)`` — caller treats the prediction as wrong, but we still need a
    valid index for the metrics functions. The boolean flag is used to
    compute an ``invalid_rate`` diagnostic.

    Matching strategy (in order):

    1. Exact normalized match.
    2. Substring match — label normalized form appears in the output's
       normalized form (or vice versa). Longest-match wins to avoid
       prefer-shorter bias.
    3. ``difflib.get_close_matches(cutoff=0.6)`` fuzzy match.
    """
    norm_out = _normalize(raw_output)
    norm_labels = [_normalize(l) for l in label_names]

    for i, nl in enumerate(norm_labels):
        if norm_out == nl:
            return i, False

    matches: list[tuple[int, int]] = []
    for i, nl in enumerate(norm_labels):
        if nl in norm_out or (norm_out and norm_out in nl):
            matches.append((i, len(nl)))
    if matches:
        matches.sort(key=lambda x: -x[1])
        return matches[0][0], False

    close = difflib.get_close_matches(norm_out, norm_labels, n=1, cutoff=0.6)
    if close:
        return norm_labels.index(close[0]), False

    return 0, True


# ---------------------------------------------------------------------------
# Model loading (CUDA + bitsandbytes, with safe CPU/MPS fallback)
# ---------------------------------------------------------------------------


def load_llama(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    quantize_4bit: bool = True,
    device: str = "auto",
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Load an instruction-tuned Llama-3 model + its tokenizer.

    On a CUDA host with ``bitsandbytes`` installed and ``quantize_4bit=True``
    we use 4-bit nf4 with bf16 compute (≈ 2 GB for the 3B model, ≈ 5 GB for
    the 8B model). Anywhere else (CPU, MPS, missing bnb), we fall back to
    a plain bf16 / fp16 load so dev-box smoke tests still work.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # Llama-3 instruct tokenizers ship without a pad token; reuse EOS for
        # left-padded generation (we don't actually batch, but generate() warns).
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(quantize_4bit and torch.cuda.is_available())
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map=device,
            )
        except ImportError:
            use_4bit = False  # fall through

    if not use_4bit:
        # Fallback: bf16 on CUDA/MPS if dtype unspecified, else fp32 on CPU.
        if dtype is None:
            if torch.cuda.is_available() or (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device if device != "auto" else None,
        )
        if device == "auto":
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                model = model.to("mps")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-document and batched classification
# ---------------------------------------------------------------------------


def classify_one(
    model: Any,
    tokenizer: Any,
    doc: str,
    label_names: list[str],
    demos: list[tuple[str, int]] | None = None,
    max_new_tokens: int = 15,
    truncate_chars: int = 1500,
) -> tuple[int, str, bool]:
    """Classify one document. Returns ``(pred_idx, raw_output, invalid_flag)``.

    Greedy decoding (``do_sample=False, num_beams=1``) — no temperature, no
    top-p; identical inputs always produce identical outputs.
    """
    import torch

    messages = build_prompt(doc, label_names, demos=demos, truncate_chars=truncate_chars)
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pred_idx, invalid = parse_label(raw, label_names)
    return pred_idx, raw, invalid


def classify_batch(
    model: Any,
    tokenizer: Any,
    docs: Sequence[str],
    label_names: list[str],
    demos: list[tuple[str, int]] | None = None,
    max_new_tokens: int = 15,
    truncate_chars: int = 1500,
    progress: bool = True,
) -> dict[str, Any]:
    """Classify every doc in ``docs`` one at a time.

    Iterating per-document is intentional — see the module docstring for the
    rationale. Returns a dict with ``y_pred`` (int array), ``raw_outputs``
    (list of strings) and ``invalid_mask`` (bool array). The ``invalid_rate``
    diagnostic is ``invalid_mask.mean()``.
    """
    y_pred = np.zeros(len(docs), dtype=np.int64)
    invalid_mask = np.zeros(len(docs), dtype=bool)
    raw_outputs: list[str] = []

    iterator: Any = enumerate(docs)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = enumerate(tqdm(docs, desc="classifying"))
        except ImportError:
            pass

    for i, doc in iterator:
        pred_idx, raw, invalid = classify_one(
            model, tokenizer, doc, label_names,
            demos=demos, max_new_tokens=max_new_tokens, truncate_chars=truncate_chars,
        )
        y_pred[i] = pred_idx
        raw_outputs.append(raw)
        invalid_mask[i] = invalid

    return {"y_pred": y_pred, "raw_outputs": raw_outputs, "invalid_mask": invalid_mask}
