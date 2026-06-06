# Q3 — MLM Pretraining then Classification Fine-tune on 20 Newsgroups: Implementation Plan

> **For agentic workers:** mirror the Q2 plan's structure. Steps use `- [ ]` checkboxes.

**Goal:** Continue BERT's masked-language-modelling objective on the 20 Newsgroups training corpus (domain-adaptive pretraining), then fine-tune the resulting checkpoint on the 20-way classification task with identical hyperparameters to Q2b. The single research question the deliverable must answer: *does MLM pretraining on the in-domain corpus improve downstream classification compared to fine-tuning `bert-base-uncased` directly (Q2b)?*

**Architecture:** One new module in `src/nlp_project/` — `mlm_pretrain.py` (HuggingFace `Trainer` wrapper for `BertForMaskedLM`, MPS-safe defaults, same seed wiring as `bert_train.py`). One new test file. **One** single end-to-end notebook with comprehensive markdown documentation between every step. Stage B of the notebook reuses `bert_train.run_finetune` verbatim — only the encoder init changes.

**Tech Stack:** Python 3.13, `uv`, PyTorch (MPS), HuggingFace `transformers` + `accelerate` + `datasets`, Weights & Biases, pytest, matplotlib, pandas. All dependencies already installed for Q2.

**Spec:** `project_description/NALAPRO Project.pdf` §2, Question 3:
> "Fine-tune a BERT model by masking some words out instead of fine tuning the classification task. After that, now finetune the model on the classification task. Evaluate and discuss the results."

---

## File Structure

**Created:**

```
src/nlp_project/
    mlm_pretrain.py        # build_mlm_dataset, make_mlm_training_args, run_mlm_pretrain
tests/
    test_mlm_pretrain.py   # fast unit + slow smoke (tiny BertConfig, no Hub weight download)
notebooks/
    q3_mlm_then_finetune.ipynb   # single end-to-end notebook (Stage A + Stage B + comparison)
docs/superpowers/plans/
    2026-05-18-q3-mlm-pretrain.md   # this file
models/q3_results/         # gitignored runtime outputs
    mlm_ckpt/              # MLM checkpoint (HF format)
    q3_pretrain_log.json   # serialized log_history with per-epoch losses + perplexity
    q3_finetune_results.json   # serialized run_finetune output
figures/
    q3_mlm_loss.png
    q3_confusion_matrix.png
    q3_vs_q2_bars.png
```

**Modified:**

- `.gitignore` — add `models/q3_results/` (matches Q2 pattern).
- `CLAUDE.md` — append "Q3 status" subsection.
- `README.md` — note Q3 completion.

**Not touched:** Q1/Q2 source modules, Q1/Q2 notebooks, Q1/Q2 result JSONs.

---

## Reusable utilities

| Utility | File | Why reused |
|---|---|---|
| `set_seed(SEED)` | `nlp_project/__init__.py:17` | Same RNG init across Q1/Q2/Q3 |
| `load_20ng(remove=True)` | `nlp_project/data.py:18` | Identical header/footer/quote stripping — avoids label leakage in Stage B |
| `train_val_split(seed=42)` | `nlp_project/data.py:95` | Same val indices as Q1 and Q2 — apples-to-apples comparison |
| `build_splits`, `tokenize`, `TextClassificationDataset` | `nlp_project/bert_data.py:74,50,17` | Stage B reuses verbatim |
| `run_finetune`, `make_training_args`, `compute_metrics` | `nlp_project/bert_train.py:130,66,33` | Stage B reuses verbatim — pass `model=<MLM-pretrained encoder>` |
| `metrics_from_predictions` | `nlp_project/eval.py:21` | Single source of truth for metrics |
| `plot_confusion` | `nlp_project/eval.py:70` | Stage B CM + comparison plots |

---

## Tasks

### 1. `mlm_pretrain.py` (TDD)

- [ ] Fast tests: `make_mlm_training_args` propagates hparams; metric_for_best_model is `"eval_loss"`, greater_is_better is False; MPS-safe flags off (fp16=bf16=pin_memory=False); `build_mlm_dataset` returns expected `input_ids`/`attention_mask` keys and length.
- [ ] Slow test: end-to-end `run_mlm_pretrain` on a tiny `BertForMaskedLM` built from a fresh `BertConfig` (no Hub download) + 16-doc toy corpus, asserting the checkpoint directory contains `config.json` + a weights file and that `perplexity == math.exp(best_eval_loss)`.
- [ ] Implement `build_mlm_dataset`, `make_mlm_training_args`, `run_mlm_pretrain`.
- [ ] Full suite green via `uv run pytest`.

### 2. `notebooks/q3_mlm_then_finetune.ipynb`

Single notebook, alternating markdown + code cells. Each major step gets a "What & Why" markdown header explaining the decision *and* the alternative considered.

Sections:
0. Title, abstract, W&B link placeholder, AI-tool disclosure.
1. Setup — imports, autoreload, `set_seed`, output dirs.
2. Data — `load_20ng(remove=True)`, corpus statistics, one sample doc.
3. Stage A: MLM pretraining — tokenizer, internal 90/10 MLM val split (independent of classification val), `DataCollatorForLanguageModeling(mlm_probability=0.15)`, `run_mlm_pretrain`, serialize log + plot loss/perplexity.
4. Stage B: classification fine-tune — `build_splits`, load MLM checkpoint as `AutoModelForSequenceClassification`, `run_finetune` with Q2b hparams (lr=2e-5, seq=256, bs=16, 3 epochs), serialize results + plot CM.
5. Comparison — read `models/q2_results/q2b_baseline.json`, side-by-side table + bar chart + dual CMs.
6. Discussion — quantitative answer, per-class winners/losers, "why or why not", limitations.
7. Artifacts & reproducibility.

### 3. Docs + housekeeping

- [ ] Update `.gitignore` (add `models/q3_results/`).
- [ ] Append "Q3 status" to `CLAUDE.md`.
- [ ] Add a Q3 line to `README.md`.

---

## Architecture notes

- **HF Trainer for MLM is the right choice.** `DataCollatorForLanguageModeling` does the 80/10/10 mask/random/keep token logic for us; reimplementing it adds zero educational value vs. Q2's HF-Trainer story.
- **MLM-val ≠ classification-val.** We do an internal 90/10 split on the train docs *inside* `run_mlm_pretrain` to give the Trainer something to compute `eval_loss` on for early-stopping. This is independent of the classification val set (carved by `data.train_val_split`) — keeping them separate prevents indirect leakage of the classification val into MLM training.
- **Identical Stage B hparams to Q2b.** The whole point of the comparison is to isolate the effect of the MLM stage. lr=2e-5, seq=256, bs=16, 3 epochs are taken verbatim from `models/q2_results/q2b_baseline.json`.
- **`perplexity = exp(eval_loss)`.** Standard MLM convention. We report it alongside `eval_loss` because perplexity is the more conventional language-modelling metric (and easier to compare to BERT pretraining papers).
- **`mlm_probability=0.15`.** Original BERT default. Not tuned — the question is "does MLM help at all," not "what is the optimal mask rate."

## Verification

```bash
uv run pytest -q                                  # ≥ 60 passed (Q1+Q2+Q3 fast suites)
uv run pytest tests/test_mlm_pretrain.py          # all green
# then user-driven:
wandb login                                       # one-time
uv run jupyter lab notebooks/q3_mlm_then_finetune.ipynb
# Stage A ≈ 30-60 min on MPS; Stage B ≈ 15-30 min.
```
