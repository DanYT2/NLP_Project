# Q2 — Fine-tune BERT-base on 20 Newsgroups: Implementation Plan

> **For agentic workers:** mirror the Q1 plan's structure. Steps use `- [ ]` checkboxes.

**Goal:** Fine-tune `bert-base-uncased` on 20 Newsgroups, run a small set of ablations to populate the "parameters and potential experiments" discussion the spec asks for, and produce an apples-to-apples comparison against Q1's best result.

**Architecture:** Two new modules in `src/nlp_project/` — `bert_data.py` (tokenization + dataset class + tri-split builder) and `bert_train.py` (HuggingFace `Trainer` wrapper with MPS-safe defaults). The existing `eval.py` is refactored to expose `metrics_from_predictions(...)`, the single source of truth for accuracy / macro-F1 / per-class F1 / confusion matrix — used by both Q1 and Q2 so the comparison is provably equivalent.

**Tech Stack:** Python 3.13, `uv`, PyTorch (MPS), HuggingFace `transformers` + `accelerate` + `datasets`, Weights & Biases, pytest, matplotlib, pandas.

**Spec:** `project_description/NALAPRO Project.pdf` §2.

---

## File Structure

**Created:**

```
src/nlp_project/
    bert_data.py         # TextClassificationDataset, tokenize(), build_splits()
    bert_train.py        # make_training_args, freeze_encoder, compute_metrics, run_finetune
tests/
    test_bert_data.py    # fast unit + slow integration (uses bert-base-uncased tokenizer)
    test_bert_train.py   # fast unit + slow smoke (tiny BertConfig, no Hub weight download)
notebooks/
    q2a_bert_setup.ipynb         # token-length EDA, decision on max_length
    q2b_bert_baseline.ipynb      # bert-base-uncased baseline, full eval, CM
    q2c_bert_experiments.ipynb   # LR sweep / seq-len / linear probe / cased ablations
    q2d_q1_vs_q2.ipynb           # cross-experiment comparison table + figures
scripts/
    build_q2_notebooks.py        # one-shot notebook scaffolder
docs/superpowers/plans/
    2026-05-15-q2-bert-finetune.md   # this file
```

**Modified:**

- `pyproject.toml` — add `transformers>=5.8.1`, `accelerate>=1.13.0`, `datasets>=4.8.5`.
- `src/nlp_project/eval.py` — extract `metrics_from_predictions(y_true, y_pred, label_names)` from inside `evaluate()`; the existing `evaluate()` becomes a thin wrapper.
- `CLAUDE.md` — append "Q2 status" subsection.
- `README.md` — note Q2 completion.

**Not touched:** Q1 modules (`data.py`, `embeddings.py`, `vectorizers.py`, `model.py`, `train.py`, `viz.py`) and Q1 notebooks. Q1's branch (`ft-qn-1`) and Q2's branch (`ft-qn-2`) stay independently runnable.

---

## Reusable utilities

| Utility | File | Why reused |
|---|---|---|
| `set_seed(SEED)` | `nlp_project/__init__.py` | Same RNG init across Q1 and Q2; Trainer's `seed=SEED` arg also wired |
| `load_20ng(remove=True)` | `nlp_project/data.py` | Identical header/footer/quote stripping — avoids label leakage |
| `train_val_split(seed=42)` | `nlp_project/data.py` | Same val indices in Q1 and Q2 (validated by `test_build_splits_matches_q1_val_indices`) |
| `metrics_from_predictions(...)` | `nlp_project/eval.py` (new) | Single metrics implementation — both Q1 and Q2 call it |
| `plot_confusion(...)` | `nlp_project/eval.py` | Same CM rendering for q2b and q2d |

---

## Tasks

### 1. Dependencies + eval refactor

- [ ] `uv add transformers accelerate datasets`
- [ ] TDD `metrics_from_predictions` in `tests/test_eval.py` (3 new tests + 1 regression on `evaluate()`'s return shape).
- [ ] Extract the helper from `evaluate()`; keep `evaluate()`'s signature and return shape identical.
- [ ] `uv run pytest tests/test_eval.py` → 7 passed.

### 2. `bert_data.py` (TDD)

- [ ] Fast tests: `TextClassificationDataset` length, `__getitem__` keys, label-order preservation.
- [ ] Slow tests: truncation at `max_length`, padding-free `attention_mask`, 3-split shape, Q1↔Q2 val-index parity.
- [ ] Implement `TextClassificationDataset`, `tokenize(docs, tokenizer, max_length)`, `build_splits(tokenizer_name, max_length, val_frac=0.1, seed=42)`.
- [ ] All 7 tests green (3 fast + 4 slow).

### 3. `bert_train.py` (TDD)

- [ ] Fast tests: `make_training_args` propagates hparams; MPS-safe flags off; load-best-on-eval-macro-f1; `compute_metrics` returns accuracy & macro-F1 from both `EvalPrediction` and `(logits, labels)` tuple; `freeze_encoder()` only freezes non-classifier params.
- [ ] Slow test: end-to-end `run_finetune` on a tiny `BertConfig`-built model + 12-doc toy dataset, asserting the returned dict shape.
- [ ] Implement `compute_metrics`, `freeze_encoder`, `make_training_args`, `_make_model`, `run_finetune`.
- [ ] Full suite: `uv run pytest` → all green.

### 4. Notebooks (via `scripts/build_q2_notebooks.py`)

- [ ] `q2a_bert_setup.ipynb` — load 20NG, tokenize, plot length histogram + quantiles, save `figures/q2a_token_length_hist.png`, justify `max_length=256`.
- [ ] `q2b_bert_baseline.ipynb` — W&B `q2b-bert-base-uncased-baseline` run at lr=2e-5, seq=256, bs=16, 3 epochs; save `figures/q2b_confusion_matrix.png`; serialize results to `models/q2_results/q2b_baseline.json`.
- [ ] `q2c_bert_experiments.ipynb` — LR sweep × seq_len × frozen × cased. Each run is its own W&B run inside group `q2`; rows serialized to `models/q2_results/q2c_sweep.json`; save bar-chart figure.
- [ ] `q2d_q1_vs_q2.ipynb` — read both JSON outputs, paste Q1's already-known numbers (q1b=0.6321/0.6098, q1c=0.6964/0.6895, q1d=0.6202/0.6004), render comparison table + bars + CM.

### 5. Docs + housekeeping

- [ ] Append a "Q2 status" subsection to `CLAUDE.md` mirroring "Q1 status".
- [ ] Update `README.md` with a Q2 line.
- [ ] Commit incrementally per task; one commit ≈ one Task above.

---

## Architecture notes for future maintainers

- **HF Trainer (Q2) vs hand-rolled (Q1).** Q1's CLAUDE.md note ("training loop is intentionally hand-rolled") is a deliberate educational choice — the point there was the loop. For Q2 the point is the fine-tuning recipe (warmup, AdamW grouping, scheduler), so we delegate to `Trainer`. Both are defensible in the 15-minute oral.
- **MPS-safe flags.** `fp16=False`, `bf16=False`, `dataloader_pin_memory=False` are non-negotiable on Apple Silicon — fp16 either crashes or silently degrades accuracy in some kernels, and pinned host memory is meaningless without CUDA.
- **Trainer `tokenizer=` → `processing_class=`.** In transformers 5.x the `tokenizer=` kwarg was renamed; we use `processing_class=tokenizer` so the warning doesn't print and so the eventual `Trainer.predict()` keeps tokenizer-aware decoding available.
- **Linear probe LR.** The frozen-encoder run uses lr=1e-3, not 2e-5 — only the classifier head trains, and an MLP head wants the classic MLP LR, not the fine-tuning LR.

## Verification

```bash
uv run pytest -q                   # 55+ passed (Q1's 37 + Q2's new ones)
uv run python scripts/build_q2_notebooks.py    # idempotent regen of notebooks
# then user-driven:
wandb login                        # one-time
uv run jupyter lab                 # execute q2a → q2b → q2c → q2d
```
