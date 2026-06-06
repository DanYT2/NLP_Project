"""One-shot script to scaffold notebooks/q3_mlm_then_finetune.ipynb.

The notebook is *not* executed here — that requires W&B credentials and
~45-90 minutes of MPS compute. This script writes the cells; the user
runs the notebook via `uv run jupyter lab`.

Re-running this script overwrites the notebook. After it is executed and
committed with output, do not run this script again without merging your
output cells back in.

Mirrors the structure of `scripts/build_q2_notebooks.py`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
KERNEL_META = {
    "kernelspec": {"display_name": ".venv (3.13.5)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.5",
    },
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": KERNEL_META, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1))


# ---------------------------------------------------------------------------
# Q3 — single end-to-end notebook
# ---------------------------------------------------------------------------

cells: list[dict] = []

# --- 0. Title & abstract --------------------------------------------------

cells += [
    md(
        "# Q3 — MLM Pretraining then Classification Fine-tuning on 20 Newsgroups\n\n"
        "**HSLU NALAPRO project — Question 3.** _Author: Dan Waititu (`danwwaititu@gmail.com`)._\n\n"
        "## Research question\n\n"
        "> _\"Fine-tune a BERT model by masking some words out instead of fine tuning the classification "
        "task. After that, now finetune the model on the classification task. Evaluate and discuss the "
        "results.\"_ — `project_description/NALAPRO Project.pdf`, §2 Question 3.\n\n"
        "In modern terminology this is **domain-adaptive pretraining** "
        "(Gururangan et al., 2020, _Don't Stop Pretraining_). We run a two-stage pipeline:\n\n"
        "| Stage | Objective | Inputs | Output |\n"
        "|---|---|---|---|\n"
        "| **A — MLM pretraining** | Masked language modelling | 20NG train texts (labels ignored) | Adapted encoder checkpoint |\n"
        "| **B — Classification fine-tune** | 20-way cross-entropy | 20NG train/val/test with labels | Classifier + test metrics |\n\n"
        "The Stage-B hyperparameters are **identical** to the Q2 baseline "
        "(`models/q2_results/q2b_baseline.json`), so the only variable changing between Q2 and Q3 is the "
        "encoder's starting weights. Any delta in test metrics is attributable to the MLM stage.\n\n"
        "## Deliverables produced by this notebook\n\n"
        "- `models/q3_results/mlm_ckpt/` — Stage-A MLM checkpoint (HF format).\n"
        "- `models/q3_results/q3_pretrain_log.json` — per-epoch MLM train/eval loss + perplexity.\n"
        "- `models/q3_results/q3_finetune_results.json` — Stage-B test metrics + W&B run names.\n"
        "- `figures/q3_mlm_loss.png` — MLM loss & perplexity curve.\n"
        "- `figures/q3_confusion_matrix.png` — Stage-B confusion matrix on the 20NG test set.\n"
        "- `figures/q3_vs_q2_bars.png` — side-by-side accuracy / macro-F1 / per-class F1 comparison vs Q2b.\n\n"
        "## W&B runs\n\n"
        "Project: `hslu-nalapro` · group: `q3` · two runs — `q3-mlm-pretrain` (Stage A) and "
        "`q3-mlm-then-finetune` (Stage B). Paste the run URLs into the placeholder below once executed.\n\n"
        "- Stage A: <https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro?nw=nwuserdanwwaititu> _(group `q3`)_\n"
        "- Stage B: same group.\n\n"
        "## AI tool disclosure\n\n"
        "Per spec §7, this notebook (cells + markdown) was co-authored with **Claude Code** "
        "(Anthropic, model `claude-opus-4-7`). All design decisions, the MLM corpus choice, the "
        "hyperparameter freeze, and the discussion section reflect the author's judgement. The model "
        "weights and metrics are reproduced from a single local execution on Apple M-series (MPS); "
        "the seed is `nlp_project.SEED = 42` throughout."
    ),
]

# --- 1. Setup -------------------------------------------------------------

cells += [
    md(
        "## 1. Setup\n\n"
        "**What:** import everything, seed all RNGs, declare output directories.\n\n"
        "**Why centralise seeding:** the project's `set_seed` (in `src/nlp_project/__init__.py`) seeds "
        "Python `random`, NumPy, `PYTHONHASHSEED`, and PyTorch from `SEED = 42`. HuggingFace's `Trainer` "
        "additionally receives `seed=SEED` and `data_seed=SEED` through `make_mlm_training_args` / "
        "`make_training_args`, so Stage A and Stage B both fire from the same RNG state.\n\n"
        "**Why `autoreload`:** the notebook is a thin orchestrator; if I edit the package modules in "
        "another editor, autoreload picks the changes up without restarting the kernel."
    ),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n\n"
        "import json\n"
        "import math\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "import wandb\n"
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n\n"
        "from nlp_project import SEED, set_seed\n"
        "from nlp_project.bert_data import build_splits\n"
        "from nlp_project.bert_train import run_finetune\n"
        "from nlp_project.data import load_20ng\n"
        "from nlp_project.eval import plot_confusion\n"
        "from nlp_project.mlm_pretrain import run_mlm_pretrain\n\n"
        "set_seed()\n\n"
        "DEVICE = (\n"
        "    'mps' if torch.backends.mps.is_available()\n"
        "    else 'cuda' if torch.cuda.is_available()\n"
        "    else 'cpu'\n"
        ")\n"
        "print(f'device = {DEVICE}; seed = {SEED}')\n\n"
        "RESULTS_DIR = Path('../models/q3_results'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MLM_CKPT_DIR = RESULTS_DIR / 'mlm_ckpt'\n"
        "FT_OUT_DIR = RESULTS_DIR / 'finetune'\n"
        "FIG_DIR = Path('../figures'); FIG_DIR.mkdir(exist_ok=True)\n"
        "Q2_RESULTS = Path('../models/q2_results/q2b_baseline.json')\n"
    ),
]

# --- 2. Data --------------------------------------------------------------

cells += [
    md(
        "## 2. Data\n\n"
        "### 2.1 Load 20 Newsgroups\n\n"
        "**What:** fetch the 20NG train/test splits via `nlp_project.data.load_20ng(remove=True)`.\n\n"
        "**Why `remove=True`:** the spec (§3 *label leakage*) explicitly notes that the message headers, "
        "footers, and quoted text contain the newsgroup name and similar tells that trivialise the task. "
        "We strip them. Q1 and Q2 use exactly the same setting — keeping it consistent makes the Q1↔Q2↔Q3 "
        "comparison meaningful.\n\n"
        "### 2.2 Which corpus is used in MLM?\n\n"
        "We pretrain on the **train split only** — no test, no labels. The decision was discussed with the "
        "user and recorded in the plan file `docs/superpowers/plans/2026-05-18-q3-mlm-pretrain.md`:\n\n"
        "> *Why train-split-only for MLM*: keeps the Q2 ↔ Q3 comparison clean — Q2's encoder also never "
        "saw the test text during training. Domain-adaptive pretraining literature supports either "
        "choice, but for our small corpus the cleaner experimental contrast outweighs the small extra "
        "unlabeled data.\n"
    ),
    code(
        "train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)\n"
        "print(f'train docs : {len(train_docs):>6d}')\n"
        "print(f'test docs  : {len(test_docs):>6d}')\n"
        "print(f'n_classes  : {len(label_names):>6d}')\n"
        "print()\n"
        "print('Example doc (first 500 chars):')\n"
        "print('-' * 60)\n"
        "print(train_docs[0][:500])\n"
        "print('-' * 60)\n"
        "print(f'label = {label_names[train_labels[0]]!r}')\n"
    ),
    code(
        "# Per-class document counts (sanity check — 20NG is approximately balanced).\n"
        "counts = pd.Series(train_labels).value_counts().sort_index()\n"
        "counts.index = label_names\n"
        "counts.rename('n_train_docs').to_frame()\n"
    ),
]

# --- 3. Stage A: MLM pretraining ------------------------------------------

cells += [
    md(
        "## 3. Stage A — MLM pretraining\n\n"
        "### 3.1 What MLM is doing here\n\n"
        "We continue BERT's original pretraining objective on the 20NG train corpus. For each input "
        "sequence, HuggingFace's `DataCollatorForLanguageModeling` does the canonical Devlin-et-al.\n"
        "masking on the fly:\n\n"
        "- Select **15%** of the WordPiece tokens uniformly at random.\n"
        "- Of those, replace 80% with `[MASK]`, 10% with a random vocabulary token, and 10% leave unchanged.\n"
        "- The labels tensor is `-100` everywhere except the selected positions, where it carries the "
        "original token id. PyTorch's cross-entropy ignores the `-100` positions.\n\n"
        "**Why `mlm_probability=0.15`:** BERT's original default. We are not tuning the mask rate — the "
        "research question is *does the extra stage help at all*, not *what is the optimal mask rate*.\n\n"
        "### 3.2 Why we report `eval_loss` and perplexity (not accuracy / F1)\n\n"
        "MLM has no classification labels, so accuracy and macro-F1 do not apply. The two standard metrics "
        "are cross-entropy loss on held-out masked tokens and its exponentiation, **perplexity** "
        "(`PPL = exp(eval_loss)`). Lower is better for both. We use `eval_loss` as the "
        "`metric_for_best_model` so the Trainer's `load_best_model_at_end=True` restores the lowest-loss "
        "epoch before saving.\n\n"
        "### 3.3 Why a separate MLM eval split\n\n"
        "Inside `run_mlm_pretrain` we carve a fresh 90/10 split out of the train docs for MLM "
        "train/eval. This split is *independent* of the classification val set carved by `train_val_split` "
        "in Stage B — if we reused the classification val for MLM, Stage A would indirectly leak "
        "information into Stage B's early-stopping signal. The two stages use disjoint validation pools."
    ),
    code(
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n"
        "\n"
        "# Stage A hyperparameters — chosen to match Q2's training budget per epoch.\n"
        "MLM_CONFIG = {\n"
        "    'model': 'bert-base-uncased',\n"
        "    'lr': 5e-5,\n"
        "    'epochs': 3,\n"
        "    'batch_size': 16,\n"
        "    'max_length': 256,\n"
        "    'mlm_probability': 0.15,\n"
        "    'mlm_val_frac': 0.1,\n"
        "    'weight_decay': 0.01,\n"
        "    'warmup_ratio': 0.1,\n"
        "    'seed': SEED,\n"
        "}\n"
        "MLM_CONFIG\n"
    ),
    md(
        "### 3.4 Run Stage A\n\n"
        "On Apple Silicon (M-series, MPS) this takes roughly 30–60 minutes for ~11k docs × 256 tokens × 3 "
        "epochs. The Trainer logs train loss every `logging_steps=50` updates and `eval_loss` once per "
        "epoch; W&B picks both up automatically through `report_to=['wandb']`.\n\n"
        "We `wandb.init` before the call and `wandb.finish` after, mirroring the pattern used in Q2 — "
        "`run_mlm_pretrain` itself is pure with respect to the W&B session."
    ),
    code(
        "wandb.init(\n"
        "    project='hslu-nalapro',\n"
        "    group='q3',\n"
        "    name='q3-mlm-pretrain',\n"
        "    config=MLM_CONFIG,\n"
        "    reinit=True,\n"
        ")\n"
        "\n"
        "stage_a = run_mlm_pretrain(\n"
        "    tokenizer=tokenizer,\n"
        "    train_docs=train_docs,\n"
        "    out_dir=MLM_CKPT_DIR,\n"
        "    model_name=MLM_CONFIG['model'],\n"
        "    lr=MLM_CONFIG['lr'],\n"
        "    epochs=MLM_CONFIG['epochs'],\n"
        "    batch_size=MLM_CONFIG['batch_size'],\n"
        "    max_length=MLM_CONFIG['max_length'],\n"
        "    mlm_probability=MLM_CONFIG['mlm_probability'],\n"
        "    mlm_val_frac=MLM_CONFIG['mlm_val_frac'],\n"
        "    weight_decay=MLM_CONFIG['weight_decay'],\n"
        "    warmup_ratio=MLM_CONFIG['warmup_ratio'],\n"
        "    run_name='q3-mlm-pretrain',\n"
        "    report_to=['wandb'],\n"
        ")\n"
        "wandb.finish()\n"
        "\n"
        "print(f'best eval_loss = {stage_a[\"best_eval_loss\"]:.4f}')\n"
        "print(f'perplexity     = {stage_a[\"perplexity\"]:.4f}')\n"
    ),
    md(
        "### 3.5 Serialize Stage-A logs and plot the loss curve\n\n"
        "We persist the Trainer's `log_history` (a list of dicts, one per logged step or eval) to JSON so "
        "the comparison cell at the end can re-render the plot without re-running training. The figure "
        "is saved to `figures/q3_mlm_loss.png` for the report."
    ),
    code(
        "log = stage_a['log_history']\n"
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "(RESULTS_DIR / 'q3_pretrain_log.json').write_text(json.dumps({\n"
        "    'config': MLM_CONFIG,\n"
        "    'best_eval_loss': stage_a['best_eval_loss'],\n"
        "    'perplexity': stage_a['perplexity'],\n"
        "    'log_history': log,\n"
        "}, indent=2, default=float))\n"
        "\n"
        "train_steps = [e['step'] for e in log if 'loss' in e and 'eval_loss' not in e]\n"
        "train_loss = [e['loss'] for e in log if 'loss' in e and 'eval_loss' not in e]\n"
        "eval_epochs = [e['epoch'] for e in log if 'eval_loss' in e]\n"
        "eval_loss = [e['eval_loss'] for e in log if 'eval_loss' in e]\n"
        "eval_ppl = [math.exp(l) for l in eval_loss]\n"
        "\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax1.plot(train_steps, train_loss, label='train loss (per logged step)', color='tab:blue', alpha=0.7)\n"
        "if eval_loss:\n"
        "    # Map eval points onto the step axis by piecewise-linear interpolation.\n"
        "    steps_per_epoch = train_steps[-1] / log[-1]['epoch'] if train_steps else 1.0\n"
        "    eval_x = [e * steps_per_epoch for e in eval_epochs]\n"
        "    ax1.plot(eval_x, eval_loss, label='eval loss', color='tab:red', marker='o', linewidth=2)\n"
        "ax1.set_xlabel('step'); ax1.set_ylabel('loss'); ax1.set_title('MLM training loss')\n"
        "ax1.legend(); ax1.grid(alpha=0.3)\n"
        "\n"
        "ax2.plot(eval_epochs, eval_ppl, marker='o', color='tab:red')\n"
        "ax2.set_xlabel('epoch'); ax2.set_ylabel('perplexity'); ax2.set_title('Eval perplexity per epoch')\n"
        "ax2.grid(alpha=0.3)\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q3_mlm_loss.png', dpi=150)\n"
        "plt.show()\n"
    ),
]

# --- 4. Stage B: classification fine-tune ---------------------------------

cells += [
    md(
        "## 4. Stage B — Classification fine-tune from the MLM checkpoint\n\n"
        "### 4.1 What changes vs Q2b, and what stays the same\n\n"
        "Everything except the encoder's starting weights stays the same. The point of an experiment is "
        "to isolate the effect of one change at a time. Q2b's run config (read from "
        "`models/q2_results/q2b_baseline.json`) was:\n\n"
        "| Setting | Q2b baseline | Q3 Stage B |\n"
        "|---|---|---|\n"
        "| Encoder init | `bert-base-uncased` (Hub) | `models/q3_results/mlm_ckpt/` (our MLM-pretrained) |\n"
        "| max_length | 256 | **256** |\n"
        "| Learning rate | 2e-5 | **2e-5** |\n"
        "| Batch size | 16 | **16** |\n"
        "| Epochs | 3 | **3** |\n"
        "| Weight decay | 0.01 | **0.01** |\n"
        "| Warmup ratio | 0.1 | **0.1** |\n"
        "| Seed | 42 | **42** |\n"
        "| Trainer | HF `Trainer` (MPS-safe flags) | **same** |\n\n"
        "Because `build_splits` reuses `data.train_val_split(seed=42)`, the val indices are identical to "
        "Q1 and Q2 — `tests/test_bert_data.py::test_build_splits_matches_q1_val_indices` enforces this.\n\n"
        "### 4.2 Loading the MLM checkpoint into a classification head\n\n"
        "`AutoModelForSequenceClassification.from_pretrained(<mlm_ckpt>, num_labels=20, ...)` reuses the "
        "encoder layers from the MLM checkpoint and *randomly initializes* a fresh classifier head on top "
        "(the MLM head is discarded). The classifier head's random init is seeded by `set_seed()`, so the "
        "head start is identical to a Q2b re-run."
    ),
    code(
        "splits = build_splits(tokenizer_name='bert-base-uncased', max_length=256)\n"
        "print(f'train: {len(splits[\"train\"])}, val: {len(splits[\"val\"])}, test: {len(splits[\"test\"])}')\n"
        "print(f'n classes: {len(splits[\"label_names\"])}')\n"
    ),
    code(
        "FT_CONFIG = {\n"
        "    'init_from': 'q3-mlm-ckpt',\n"
        "    'mlm_ckpt': str(MLM_CKPT_DIR),\n"
        "    'max_length': 256,\n"
        "    'lr': 2e-5,\n"
        "    'epochs': 3,\n"
        "    'batch_size': 16,\n"
        "    'weight_decay': 0.01,\n"
        "    'warmup_ratio': 0.1,\n"
        "    'seed': SEED,\n"
        "}\n"
        "\n"
        "# Re-seed before head init so the classifier weights are deterministic.\n"
        "set_seed()\n"
        "model = AutoModelForSequenceClassification.from_pretrained(\n"
        "    str(MLM_CKPT_DIR),\n"
        "    num_labels=len(splits['label_names']),\n"
        "    id2label={i: n for i, n in enumerate(splits['label_names'])},\n"
        "    label2id={n: i for i, n in enumerate(splits['label_names'])},\n"
        ")\n"
        "type(model).__name__\n"
    ),
    md(
        "### 4.3 Run Stage B\n\n"
        "We delegate to `bert_train.run_finetune` — the exact same function Q2 used. The model arg lets us "
        "pass our MLM-pretrained encoder instead of having `run_finetune` download fresh weights from the "
        "Hub. On MPS, Stage B takes roughly 15–30 minutes."
    ),
    code(
        "wandb.init(\n"
        "    project='hslu-nalapro',\n"
        "    group='q3',\n"
        "    name='q3-mlm-then-finetune',\n"
        "    config=FT_CONFIG,\n"
        "    reinit=True,\n"
        ")\n"
        "\n"
        "stage_b = run_finetune(\n"
        "    splits=splits,\n"
        "    model=model,\n"
        "    out_dir=FT_OUT_DIR,\n"
        "    lr=FT_CONFIG['lr'],\n"
        "    epochs=FT_CONFIG['epochs'],\n"
        "    batch_size=FT_CONFIG['batch_size'],\n"
        "    run_name='q3-mlm-then-finetune',\n"
        "    weight_decay=FT_CONFIG['weight_decay'],\n"
        "    warmup_ratio=FT_CONFIG['warmup_ratio'],\n"
        "    report_to=['wandb'],\n"
        ")\n"
        "wandb.finish()\n"
        "\n"
        "tm = stage_b['test_metrics']\n"
        "print(f'accuracy   = {tm[\"accuracy\"]:.4f}')\n"
        "print(f'macro_f1   = {tm[\"macro_f1\"]:.4f}')\n"
        "print(f'best eval  = {stage_b[\"best_eval_metric\"]:.4f}')\n"
    ),
    md(
        "### 4.4 Serialize Stage-B results and render the confusion matrix\n\n"
        "We dump the same JSON shape as Q2b so the comparison cell can read both files with identical "
        "code. The CM PNG goes into `figures/` for the scientific report."
    ),
    code(
        "tm = stage_b['test_metrics']\n"
        "(RESULTS_DIR / 'q3_finetune_results.json').write_text(json.dumps({\n"
        "    'name': 'q3-mlm-then-finetune',\n"
        "    'config': FT_CONFIG,\n"
        "    'accuracy': float(tm['accuracy']),\n"
        "    'macro_f1': float(tm['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in tm['per_class_f1']],\n"
        "    'confusion_matrix': tm['confusion_matrix'].tolist(),\n"
        "    'best_eval_macro_f1': float(stage_b['best_eval_metric']),\n"
        "}, indent=2))\n"
        "\n"
        "plot_confusion(\n"
        "    cm=tm['confusion_matrix'],\n"
        "    label_names=splits['label_names'],\n"
        "    save_path=FIG_DIR / 'q3_confusion_matrix.png',\n"
        "    title='Q3 — MLM-pretrained then fine-tuned (test set)',\n"
        ")\n"
        "print('Saved figure to', FIG_DIR / 'q3_confusion_matrix.png')\n"
    ),
]

# --- 5. Comparison vs Q2 --------------------------------------------------

cells += [
    md(
        "## 5. Comparison — Q2 baseline vs Q3 (MLM-then-fine-tune)\n\n"
        "**What:** load the Q2b baseline JSON and the Q3 Stage-B JSON we just wrote, and present "
        "head-to-head numbers + plots.\n\n"
        "**Why:** the spec asks us to \"evaluate and discuss the results.\" The numbers in this section "
        "are what the report's results table and discussion will cite."
    ),
    code(
        "q2 = json.loads(Q2_RESULTS.read_text())\n"
        "q3 = json.loads((RESULTS_DIR / 'q3_finetune_results.json').read_text())\n"
        "label_names = splits['label_names']\n"
        "\n"
        "summary = pd.DataFrame({\n"
        "    'metric': ['accuracy', 'macro_f1'],\n"
        "    'Q2b (bert-base-uncased)': [q2['accuracy'], q2['macro_f1']],\n"
        "    'Q3 (MLM -> fine-tune)':   [q3['accuracy'], q3['macro_f1']],\n"
        "})\n"
        "summary['delta (Q3 - Q2)'] = summary['Q3 (MLM -> fine-tune)'] - summary['Q2b (bert-base-uncased)']\n"
        "summary\n"
    ),
    code(
        "# Per-class F1 deltas — which classes did MLM pretraining help / hurt the most?\n"
        "per_class = pd.DataFrame({\n"
        "    'label': label_names,\n"
        "    'Q2b_f1': q2['per_class_f1'],\n"
        "    'Q3_f1':  q3['per_class_f1'],\n"
        "})\n"
        "per_class['delta'] = per_class['Q3_f1'] - per_class['Q2b_f1']\n"
        "print('Top 3 classes where MLM pretraining helped:')\n"
        "print(per_class.nlargest(3, 'delta').to_string(index=False))\n"
        "print()\n"
        "print('Bottom 3 classes (MLM hurt or made no difference):')\n"
        "print(per_class.nsmallest(3, 'delta').to_string(index=False))\n"
    ),
    code(
        "# Bar chart: aggregate metrics + per-class F1 deltas.\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "ax1.bar(['accuracy', 'macro_f1'],\n"
        "        [q2['accuracy'], q2['macro_f1']], width=0.4, align='edge', label='Q2b', color='tab:gray')\n"
        "ax1.bar(['accuracy', 'macro_f1'],\n"
        "        [q3['accuracy'], q3['macro_f1']], width=-0.4, align='edge', label='Q3', color='tab:blue')\n"
        "ax1.set_ylim(0, 1); ax1.set_ylabel('score'); ax1.set_title('Aggregate test metrics')\n"
        "ax1.legend(); ax1.grid(alpha=0.3, axis='y')\n"
        "\n"
        "order = per_class.sort_values('delta')['label']\n"
        "pc = per_class.set_index('label').loc[order]\n"
        "colors = ['tab:red' if d < 0 else 'tab:green' for d in pc['delta']]\n"
        "ax2.barh(pc.index, pc['delta'], color=colors)\n"
        "ax2.axvline(0, color='black', linewidth=0.8)\n"
        "ax2.set_xlabel('per-class F1 delta (Q3 - Q2b)')\n"
        "ax2.set_title('Per-class F1 — Q3 minus Q2b')\n"
        "ax2.grid(alpha=0.3, axis='x')\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q3_vs_q2_bars.png', dpi=150)\n"
        "plt.show()\n"
    ),
    code(
        "# Side-by-side confusion matrices.\n"
        "import seaborn as sns\n"
        "q2_cm = np.array(q2['confusion_matrix'])\n"
        "q3_cm = np.array(q3['confusion_matrix'])\n"
        "fig, axes = plt.subplots(1, 2, figsize=(22, 10))\n"
        "for ax, cm, title in zip(axes, [q2_cm, q3_cm], ['Q2b — bert-base-uncased', 'Q3 — MLM then fine-tune']):\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "                xticklabels=label_names, yticklabels=label_names,\n"
        "                cbar=False, ax=ax)\n"
        "    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)\n"
        "    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q3_vs_q2_confusion_side_by_side.png', dpi=150)\n"
        "plt.show()\n"
    ),
]

# --- 6. Discussion --------------------------------------------------------

cells += [
    md(
        "## 6. Discussion\n\n"
        "_The placeholders below are written so the reader can drop in the actual deltas after the "
        "notebook executes. The phrasing intentionally hedges — whether MLM helps on a "
        "~10k-document in-domain corpus is genuinely uncertain a priori._\n\n"
        "### 6.1 Did MLM pretraining help, quantitatively?\n\n"
        "The aggregate test metrics (section 5) show **Δaccuracy = Q3 − Q2b = `{fill in}`** and "
        "**Δmacro-F1 = `{fill in}`**. With 7532 test documents, a difference in accuracy of more than "
        "~1.0 percentage points is outside the binomial noise envelope at 95% confidence "
        "(`1.96 × √(0.71·0.29/7532) ≈ 0.010`). Read the printed delta and judge accordingly.\n\n"
        "### 6.2 Per-class winners and losers\n\n"
        "The per-class F1 delta chart (`figures/q3_vs_q2_bars.png`, right panel) ranks newsgroups by how "
        "much MLM pretraining helped or hurt them. Two patterns to look for:\n\n"
        "- **Tail-of-domain newsgroups** (`talk.religion.misc`, `talk.politics.misc`, `talk.politics.mideast`) "
        "  often have idiosyncratic vocabulary that `bert-base-uncased` undertrains on. If MLM helps, "
        "  these are the classes most likely to benefit.\n"
        "- **Technical newsgroups** (`sci.crypt`, `comp.os.ms-windows.misc`, `sci.electronics`) tend to "
        "  share vocabulary with Wikipedia-trained BERT, so the marginal value of in-domain MLM is "
        "  smaller there.\n\n"
        "### 6.3 Why might MLM pretraining help (or not)?\n\n"
        "**Arguments for a positive effect:**\n"
        "- 20NG includes a fair amount of usenet-jargon and pre-2000 technical writing that `bert-base-uncased`'s "
        "  Wikipedia/BookCorpus training set under-represents. MLM on the in-domain corpus moves the "
        "  encoder's representation manifold toward that distribution before the classifier head is even introduced.\n"
        "- Even when the vocabulary overlaps, the *topic priors* differ — Wikipedia rarely sees the "
        "  three-quote nested replies that survived header-stripping; MLM teaches the model that those "
        "  patterns are normal.\n\n"
        "**Arguments for a null or negative effect:**\n"
        "- The corpus is small (~10k docs). Three MLM epochs at lr=5e-5 may overfit to corpus quirks "
        "  without giving the encoder enough signal to update its general representations.\n"
        "- Q2b is already strong (≈ 0.71 accuracy on a 20-way task) — there is limited headroom for an "
        "  unsupervised stage to recover from.\n"
        "- We share the tokenizer between stages, so the only thing the MLM stage can adjust is the *encoder*. "
        "  If the bottleneck is really the classifier head's interaction with the embedding manifold, MLM "
        "  may not move that bottleneck.\n\n"
        "### 6.4 Limitations and threats to validity\n\n"
        "- **Single seed.** Both Stage A and Stage B run at seed 42. A proper variance estimate requires "
        "  3–5 seeds per condition; for this graded project we accept a single seed and treat any delta "
        "  smaller than the binomial confidence interval as inconclusive.\n"
        "- **MLM-val ≠ classification-val.** Required to prevent leakage, but it means we cannot directly "
        "  compare MLM eval loss progression to classification val F1 progression.\n"
        "- **Same hyperparameters as Q2b.** A fair argument exists that Q3 deserves its *own* "
        "  hyperparameter sweep — MLM-pretrained encoders may want different LRs. We hold them fixed to "
        "  isolate the encoder-init effect, but the absolute Q3 accuracy reported here may not be the best "
        "  achievable with this approach.\n"
        "- **`bert-base-uncased` only.** A heavier base (`bert-large-uncased`, or a domain-pretrained "
        "  variant like SciBERT) would be more informative for the broader claim. Out of scope here."
    ),
]

# --- 7. Artifacts ---------------------------------------------------------

cells += [
    md(
        "## 7. Artifacts & reproducibility\n\n"
        "### Files produced by this notebook\n\n"
        "| Path | Content |\n"
        "|---|---|\n"
        "| `models/q3_results/mlm_ckpt/` | Stage-A MLM checkpoint (HF model + tokenizer) |\n"
        "| `models/q3_results/q3_pretrain_log.json` | Stage-A log history + best eval loss + perplexity |\n"
        "| `models/q3_results/finetune/` | Stage-B Trainer output (best-restored checkpoint only; saved by `save_total_limit=1`) |\n"
        "| `models/q3_results/q3_finetune_results.json` | Stage-B test metrics + config |\n"
        "| `figures/q3_mlm_loss.png` | MLM loss & perplexity curve |\n"
        "| `figures/q3_confusion_matrix.png` | Stage-B confusion matrix |\n"
        "| `figures/q3_vs_q2_bars.png` | Aggregate + per-class F1 comparison vs Q2b |\n"
        "| `figures/q3_vs_q2_confusion_side_by_side.png` | Two CMs side by side |\n\n"
        "### Re-running\n\n"
        "1. `wandb login` (one-time).\n"
        "2. `uv run jupyter lab notebooks/q3_mlm_then_finetune.ipynb`.\n"
        "3. Execute all cells top to bottom. Stage A ≈ 30–60 min on MPS, Stage B ≈ 15–30 min.\n"
        "4. After execution, paste the two W&B run URLs into the placeholder lines in section 0.\n\n"
        "### Test suite invariants\n\n"
        "Regardless of whether Stage A / Stage B were run, the package contract holds:\n\n"
        "```bash\n"
        "uv run pytest -m 'not slow'   # all fast tests pass\n"
        "uv run pytest                 # full suite incl. the MLM smoke test\n"
        "```\n"
    ),
]


# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    out = NB_DIR / "q3_mlm_then_finetune.ipynb"
    write_notebook(out, cells)
    print(f"Wrote {out.relative_to(NB_DIR.parent)} ({len(cells)} cells).")
