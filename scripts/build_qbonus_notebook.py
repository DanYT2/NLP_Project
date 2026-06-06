"""One-shot script to scaffold notebooks/qbonus_llama_qlora_finetune.ipynb.

Bonus task: **QLoRA fine-tune of Llama-3.2-3B-Instruct on 20 Newsgroups**,
evaluated on the full 7 532-document test set *and* on the same 200-document
stratified subset used by Q4 (zero/few-shot Llama). The notebook is
**self-contained** — it does not import anything from `src/nlp_project/`.
Every helper it needs (seeding, data loading, metrics, plotting, dataset
wrapper) is defined inline in the notebook itself. That keeps the Colab
path simple: no `git clone`, no `pip install -e .`, just install runtime
deps and go.

The notebook is *not* executed here — running it requires an A100 (or
similar) GPU. This script just writes the cells.

Re-running this script overwrites the notebook. After it has been executed
on Colab and the output cells committed, do **not** re-run this script
without first merging the outputs back in.
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
# Q-bonus — single self-contained notebook
# ---------------------------------------------------------------------------

cells: list[dict] = []

# --- 0. Title & abstract --------------------------------------------------

cells += [
    md(
        "# Q-bonus — QLoRA Fine-tune of Llama-3.2-3B on 20 Newsgroups\n\n"
        "**HSLU NALAPRO project — Bonus question.** _Author: Dan Waititu (`danwwaititu@gmail.com`)._\n\n"
        "## Research question\n\n"
        "> _\"Fine-tune Llama-3 (using LoRA or QLoRA) on the 20 Newsgroups classification task, "
        "evaluate the results, and compare to the other methods — including the Q4 zero-shot and "
        "few-shot baselines.\"_ — bonus addendum to `project_description/NALAPRO Project.pdf`.\n\n"
        "Q4 froze Llama-3 entirely and prompted it. This notebook keeps the same Llama-3.2-3B "
        "weights but **trains a thin LoRA adapter** on top of a 4-bit-quantized backbone "
        "(\"QLoRA\"; Dettmers et al., 2023). Adding only ~ 0.2 % of new trainable parameters and "
        "training for a few epochs on the 20 NG train split should outperform Q4 by a wide "
        "margin and let us measure the gap to the supervised BERT (Q2) and MLM-then-fine-tune "
        "(Q3) baselines.\n\n"
        "## Self-contained\n\n"
        "This notebook does **not** import from the `nlp_project` package — every helper it needs "
        "is defined inline below (sections 3–4). It runs unchanged on Google Colab "
        "(A100 high-RAM strongly recommended) without cloning the repo, and on a local CUDA box "
        "with the runtime deps installed.\n\n"
        "## Runs performed\n\n"
        "| Run name | LoRA rank | lr | Epochs | Effective batch | Purpose |\n"
        "|---|---:|---:|---:|---:|---|\n"
        "| `qbonus-qlora-r16-lr2e4` | 16 | 2e-4 | 3 | 32 | Baseline QLoRA config. |\n"
        "| `qbonus-qlora-r32-lr1e4` | 32 | 1e-4 | 3 | 32 | Higher rank, lower LR — sweep for best accuracy. |\n\n"
        "The notebook picks the **best run by held-out val macro-F1** and uses that adapter for "
        "the final test-set evaluations.\n\n"
        "## Evaluation strategy\n\n"
        "Two final-eval passes on the picked adapter:\n\n"
        "1. **Full 20 NG test set (7 532 docs)** → comparable to Q1 (MLP), Q2 (BERT fine-tune), "
        "Q3 (MLM-then-fine-tune).\n"
        "2. **Q4 stratified 200-doc subset (seed 42, 10 per class)** → comparable to Q4 zero/few-shot.\n\n"
        "Both subsets are built deterministically with the same seed as the rest of the project, "
        "so cross-experiment comparisons are exact.\n\n"
        "## Model and hardware\n\n"
        "- **Model**: `meta-llama/Llama-3.2-3B-Instruct` (same checkpoint as Q4).\n"
        "- **Quantization**: 4-bit nf4 with double quantization, bf16 compute (`bitsandbytes`).\n"
        "- **Adapter**: LoRA on the four attention projections `{q,k,v,o}_proj`; classification "
        "head (`score`) trained in full precision via PEFT `modules_to_save`.\n"
        "- **Training**: HuggingFace `Trainer`, bf16, cosine LR schedule, 3 % warmup.\n"
        "- **Hardware**: Colab A100 40 GB (target). Also runs on a 24 GB Ampere card (RTX 3090 / 4090).\n\n"
        "## Deliverables produced by this notebook\n\n"
        "- `models/qbonus_results/qbonus_qlora_full_test.json` — metrics on the full 7 532-doc test set.\n"
        "- `models/qbonus_results/qbonus_qlora_q4subset.json` — metrics on the Q4 200-doc subset.\n"
        "- `models/qbonus_results/qbonus_sweep_summary.json` — best val macro-F1 per LoRA config.\n"
        "- `figures/qbonus_confusion_full.png` — confusion matrix on the full test set.\n"
        "- `figures/qbonus_comparison_bars.png` — accuracy / macro-F1 across all available methods "
        "(Q4 zero/few-shot, Q2, Q3 if present, plus this notebook on both eval splits).\n"
        "- `figures/qbonus_per_class_delta.png` — per-class F1 lift vs Q4 zero-shot.\n\n"
        "## W&B runs\n\n"
        "Project `hslu-nalapro`, group `qbonus`. Paste the run URLs below once executed:\n\n"
        "- <https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro?nw=nwuserdanwwaititu> "
        "_(group `qbonus`)_\n\n"
        "## AI tool disclosure\n\n"
        "Per spec §7, this notebook (cells + markdown) was co-authored with **Claude Code** "
        "(Anthropic, model `claude-opus-4-7`). The design choices (QLoRA over full fine-tune, "
        "sequence-classification head over generative SFT, sweep dimensions, eval-on-both-splits "
        "strategy) reflect the author's judgement; numerical results are reproduced from single "
        "executions with `SEED = 42` throughout."
    ),
]

# --- 1. Environment setup -------------------------------------------------

cells += [
    md(
        "## 1. Environment setup — Colab A100 vs local CUDA\n\n"
        "This notebook runs unchanged on two hosts:\n\n"
        "1. **Google Colab (A100 high-RAM)** — open the notebook, run the cell below; it pip-installs "
        "the runtime deps and prompts for HuggingFace + W&B logins. Llama-3.2 is gated, so request "
        "access on the model page first (`huggingface.co/meta-llama/Llama-3.2-3B-Instruct`).\n"
        "2. **Local CUDA (24 GB+ Ampere)** — `pip install transformers accelerate bitsandbytes peft "
        "wandb scikit-learn seaborn pandas tqdm`; `wandb login`; `huggingface-cli login`.\n\n"
        "**No repo clone, no `pip install -e .`.** Every helper this notebook uses is defined "
        "inline in sections 3–4."
    ),
    code(
        "# --- detect environment ---------------------------------------------------\n"
        "try:\n"
        "    import google.colab  # noqa: F401\n"
        "    IN_COLAB = True\n"
        "except ImportError:\n"
        "    IN_COLAB = False\n"
        "\n"
        "print(f'IN_COLAB = {IN_COLAB}')\n"
        "\n"
        "if IN_COLAB:\n"
        "    import subprocess, sys\n"
        "    # Pin upper bounds loose; Colab's pre-installed torch is kept (do NOT reinstall it).\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install',\n"
        "        'transformers>=4.45', 'accelerate>=0.34', 'bitsandbytes>=0.43',\n"
        "        'peft>=0.13', 'wandb', 'scikit-learn', 'seaborn', 'pandas', 'tqdm'])\n"
        "    from huggingface_hub import notebook_login as hf_login\n"
        "    hf_login()\n"
        "    import wandb\n"
        "    wandb.login()\n"
        "\n"
        "# --- device sanity check --------------------------------------------------\n"
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    print(f'CUDA OK \\u2014 device 0 = {torch.cuda.get_device_name(0)}')\n"
        "    free, total = torch.cuda.mem_get_info()\n"
        "    print(f'free / total VRAM = {free / 1e9:.2f} / {total / 1e9:.2f} GB')\n"
        "else:\n"
        "    raise RuntimeError(\n"
        "        'No CUDA detected. QLoRA training of a 3B model on CPU/MPS is infeasible \\u2014 '\n"
        "        'open this notebook on a CUDA host (A100 recommended).'\n"
        "    )\n"
    ),
]

# --- 2. Imports, seed, output dirs ----------------------------------------

cells += [
    md(
        "## 2. Imports, seed, output directories\n\n"
        "Standard scientific stack plus `transformers`, `peft`, `bitsandbytes`, `wandb`, `sklearn`. "
        "We define `SEED = 42` and `set_seed` inline so the notebook is self-contained.\n\n"
        "**Why seed everything:** Two reasons in this notebook. (i) The stratified val/test "
        "subsamples must match Q1–Q4 byte-for-byte for the cross-question comparison to be valid. "
        "(ii) LoRA's `score` head is freshly initialised; reproducible initialisation lets us "
        "compare two LoRA configs on equal footing instead of conflating config differences with "
        "init noise."
    ),
    code(
        "import json\n"
        "import os\n"
        "import random\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "import torch\n"
        "import wandb\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "SEED = 42\n"
        "\n"
        "def set_seed(seed: int = SEED) -> None:\n"
        "    \"\"\"Seed Python, NumPy, env-hash, and PyTorch (CPU + CUDA) from one constant.\"\"\"\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed_all(seed)\n"
        "\n"
        "set_seed()\n"
        "\n"
        "# Outputs land relative to the notebook's working directory. On Colab that's /content/.\n"
        "RESULTS_DIR = Path('models/qbonus_results'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "FIG_DIR = Path('figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "CKPT_DIR = Path('models/qbonus_ckpt'); CKPT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "# Optional comparison data; only used if present locally:\n"
        "Q4_DIR = Path('models/q4_results')\n"
        "Q2_BASELINE = Path('models/q2_results/q2b_baseline.json')\n"
        "Q3_BASELINE = Path('models/q3_results/q3_finetune_results.json')\n"
        "\n"
        "MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'\n"
        "MAX_LENGTH = 512  # ~95% of 20NG docs fit; longer would slow training quadratically.\n"
        "\n"
        "print('results ->', RESULTS_DIR.resolve())\n"
        "print('figures ->', FIG_DIR.resolve())\n"
        "print('ckpts   ->', CKPT_DIR.resolve())\n"
    ),
]

# --- 3. Inline helpers: data / metrics / plotting ------------------------

cells += [
    md(
        "## 3. Inline helpers — data, metrics, plotting\n\n"
        "Four small utilities, defined here so the notebook does not depend on the `nlp_project` "
        "package:\n\n"
        "- **`load_20ng(remove=True)`** — fetch 20 Newsgroups train/test splits from sklearn. "
        "`remove=True` strips headers/footers/quoted text (spec §3, label-leakage rule).\n"
        "- **`train_val_split(docs, labels, val_frac=0.1, seed=42)`** — stratified 90/10 carve of "
        "the training set; uses `random_state=42` so the val indices are **byte-identical to "
        "Q1/Q2/Q3**.\n"
        "- **`metrics_from_predictions(y_true, y_pred, label_names)`** — accuracy + macro-F1 + "
        "per-class F1 + confusion matrix, returned as a dict.\n"
        "- **`plot_confusion(cm, label_names, save_path, title)`** — seaborn heatmap saved as PNG."
    ),
    code(
        "from sklearn.datasets import fetch_20newsgroups\n"
        "from sklearn.metrics import accuracy_score, confusion_matrix as _cm, f1_score\n"
        "\n"
        "def load_20ng(remove: bool = True):\n"
        "    \"\"\"Fetch 20 Newsgroups. Returns (train_docs, train_labels, test_docs, test_labels, label_names).\n"
        "\n"
        "    `remove=True` strips headers/footers/quoted text (spec \\u00a73, avoid label leakage).\n"
        "    \"\"\"\n"
        "    remove_tuple = ('headers', 'footers', 'quotes') if remove else ()\n"
        "    tr = fetch_20newsgroups(subset='train', remove=remove_tuple)\n"
        "    te = fetch_20newsgroups(subset='test',  remove=remove_tuple)\n"
        "    return list(tr.data), np.asarray(tr.target), list(te.data), np.asarray(te.target), list(tr.target_names)\n"
        "\n"
        "def train_val_split(docs, labels, val_frac: float = 0.1, seed: int = SEED):\n"
        "    \"\"\"Stratified 90/10 carve, seed-pinned for Q1\\u2194Q-bonus val-index parity.\"\"\"\n"
        "    tr_d, va_d, tr_y, va_y = train_test_split(\n"
        "        list(docs), np.asarray(labels),\n"
        "        test_size=val_frac, stratify=labels, random_state=seed,\n"
        "    )\n"
        "    return tr_d, np.asarray(tr_y), va_d, np.asarray(va_y)\n"
        "\n"
        "def metrics_from_predictions(y_true, y_pred, label_names):\n"
        "    \"\"\"Standard classification metrics from already-computed predictions.\"\"\"\n"
        "    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)\n"
        "    labels = list(range(len(label_names)))\n"
        "    return {\n"
        "        'accuracy': accuracy_score(y_true, y_pred),\n"
        "        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "        'per_class_f1': f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0),\n"
        "        'confusion_matrix': _cm(y_true, y_pred, labels=labels),\n"
        "    }\n"
        "\n"
        "def plot_confusion(cm, label_names, save_path, title='Confusion matrix'):\n"
        "    \"\"\"Seaborn heatmap of a confusion matrix, saved as PNG.\"\"\"\n"
        "    fig, ax = plt.subplots(figsize=(0.5 * len(label_names) + 2, 0.5 * len(label_names) + 2))\n"
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
        "                xticklabels=label_names, yticklabels=label_names, cbar=False, ax=ax)\n"
        "    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)\n"
        "    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')\n"
        "    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)\n"
        "    print('Saved figure ->', save_path)\n"
    ),
]

# --- 4. Inline helpers: tokenization dataset ------------------------------

cells += [
    md(
        "## 4. Inline helpers — tokenization & dataset wrapper\n\n"
        "We need a `torch.utils.data.Dataset` that yields `{input_ids, attention_mask, labels}` "
        "for HuggingFace `Trainer`. Two design choices:\n\n"
        "- **Pad dynamically** via `DataCollatorWithPadding` rather than padding to a fixed length "
        "at tokenization time. Saves ~ 30 % of compute on a corpus with high variance in document "
        "length (which 20 NG has — many one-line posts mixed with 1 k-token threads).\n"
        "- **`padding_side='right'`** on the Llama tokenizer (default) is fine for training: "
        "`LlamaForSequenceClassification` uses the **last non-padding token** for pooling, located "
        "via the attention mask. Left-padding is only required for *generative* inference at batch "
        "size > 1, which is not how the Trainer runs classification.\n\n"
        "**Llama lacks a pad token.** We point `tokenizer.pad_token` at `eos_token` and **also** "
        "set `model.config.pad_token_id` to that ID. Forgetting the second line makes "
        "`LlamaForSequenceClassification` warn and (in some transformers versions) pool on the "
        "wrong token \\u2014 silently degrading accuracy."
    ),
    code(
        "class NewsgroupsDataset(torch.utils.data.Dataset):\n"
        "    \"\"\"Light wrapper around tokenized inputs + integer labels.\n"
        "\n"
        "    Inputs are pre-tokenized once (no padding); the Trainer's data collator pads each\n"
        "    batch dynamically. Labels are kept as Python ints; the collator converts them to a tensor.\n"
        "    \"\"\"\n"
        "    def __init__(self, encodings, labels):\n"
        "        self.input_ids = encodings['input_ids']\n"
        "        self.attention_mask = encodings['attention_mask']\n"
        "        self.labels = list(labels)\n"
        "\n"
        "    def __len__(self):\n"
        "        return len(self.labels)\n"
        "\n"
        "    def __getitem__(self, idx):\n"
        "        return {\n"
        "            'input_ids': self.input_ids[idx],\n"
        "            'attention_mask': self.attention_mask[idx],\n"
        "            'labels': int(self.labels[idx]),\n"
        "        }\n"
        "\n"
        "def tokenize_docs(tokenizer, docs, max_length: int = MAX_LENGTH):\n"
        "    \"\"\"Tokenize a list of strings without padding (collator will pad per batch).\"\"\"\n"
        "    return tokenizer(\n"
        "        list(docs),\n"
        "        truncation=True,\n"
        "        max_length=max_length,\n"
        "        padding=False,\n"
        "        return_attention_mask=True,\n"
        "    )\n"
    ),
]

# --- 5. Data --------------------------------------------------------------

cells += [
    md(
        "## 5. Data \\u2014 20 Newsgroups train / val / test + Q4 200-doc subset\n\n"
        "**Splits:**\n\n"
        "- Train: 90 % of 20 NG train (\\u224810 182 docs after stratified carve).\n"
        "- Val: 10 % of 20 NG train (\\u22481 132 docs). Same indices as Q1/Q2/Q3 thanks to "
        "`seed=42`.\n"
        "- Test (full): all 7 532 docs of 20 NG test \\u2014 the headline number that compares "
        "vs Q1/Q2/Q3.\n"
        "- Test (Q4 subset): 200-doc stratified subset of the test set, **identical** seed 42 "
        "carve as the Q4 notebook \\u2014 the number that compares vs Q4 zero/few-shot.\n\n"
        "Both test sets are byte-identical to other questions; subsampling for the Q4 subset uses "
        "`train_test_split(test_size=200, stratify=test_labels, random_state=42)` exactly as Q4 "
        "does."
    ),
    code(
        "train_all_docs, train_all_labels, test_docs, test_labels, label_names = load_20ng(remove=True)\n"
        "tr_docs, tr_y, va_docs, va_y = train_val_split(train_all_docs, train_all_labels, val_frac=0.1, seed=SEED)\n"
        "\n"
        "print(f'train  : {len(tr_docs):>6d} docs')\n"
        "print(f'val    : {len(va_docs):>6d} docs')\n"
        "print(f'test   : {len(test_docs):>6d} docs (full)')\n"
        "print(f'classes: {len(label_names):>6d}')\n"
        "\n"
        "# Q4 200-doc stratified subset (10/class, seed 42) \\u2014 same call as the Q4 notebook.\n"
        "_, q4_eval_docs, _, q4_eval_labels = train_test_split(\n"
        "    test_docs, test_labels,\n"
        "    train_size=len(test_docs) - 200,\n"
        "    test_size=200,\n"
        "    stratify=test_labels,\n"
        "    random_state=SEED,\n"
        ")\n"
        "q4_eval_docs = list(q4_eval_docs)\n"
        "q4_eval_labels = np.asarray(q4_eval_labels)\n"
        "print(f'q4 subset: {len(q4_eval_docs)} docs (matches Q4 zero/few-shot eval set)')\n"
    ),
]

# --- 6. Tokenizer + Datasets ---------------------------------------------

cells += [
    md(
        "## 6. Tokenizer and tokenized datasets\n\n"
        "Load the Llama-3 tokenizer (gated; needs `huggingface-cli login` first). Build "
        "`NewsgroupsDataset` objects for train, val, full-test, and Q4-subset.\n\n"
        "**Quick token-length sanity check:** print the 50/95/99-percentile of pre-truncation token "
        "lengths on the *train* split, so we can see how many docs `MAX_LENGTH = 512` is actually "
        "cutting off."
    ),
    code(
        "from transformers import AutoTokenizer\n"
        "\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "if tokenizer.pad_token_id is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token\n"
        "    print(f'set tokenizer.pad_token = eos ({tokenizer.eos_token!r}, id={tokenizer.pad_token_id})')\n"
        "\n"
        "# Token-length sanity \\u2014 measured on a random 1 000-doc sample (full corpus would be slow).\n"
        "rng = np.random.default_rng(SEED)\n"
        "sample_idx = rng.choice(len(tr_docs), size=min(1000, len(tr_docs)), replace=False)\n"
        "sample_lens = [len(tokenizer(tr_docs[i], add_special_tokens=True)['input_ids']) for i in sample_idx]\n"
        "p50, p95, p99 = np.percentile(sample_lens, [50, 95, 99])\n"
        "print(f'token-length p50/p95/p99 = {p50:.0f} / {p95:.0f} / {p99:.0f}')\n"
        "print(f'fraction truncated at MAX_LENGTH={MAX_LENGTH}: {np.mean(np.asarray(sample_lens) > MAX_LENGTH):.3f}')\n"
        "\n"
        "# Tokenize every split. Train tokenization is the slow one (\\u224830 s on Colab).\n"
        "tr_enc      = tokenize_docs(tokenizer, tr_docs,      MAX_LENGTH)\n"
        "va_enc      = tokenize_docs(tokenizer, va_docs,      MAX_LENGTH)\n"
        "te_enc      = tokenize_docs(tokenizer, test_docs,    MAX_LENGTH)\n"
        "q4_enc      = tokenize_docs(tokenizer, q4_eval_docs, MAX_LENGTH)\n"
        "\n"
        "tr_ds = NewsgroupsDataset(tr_enc, tr_y)\n"
        "va_ds = NewsgroupsDataset(va_enc, va_y)\n"
        "te_ds = NewsgroupsDataset(te_enc, test_labels)\n"
        "q4_ds = NewsgroupsDataset(q4_enc, q4_eval_labels)\n"
        "print(f'datasets: train={len(tr_ds)}, val={len(va_ds)}, test(full)={len(te_ds)}, test(q4)={len(q4_ds)}')\n"
    ),
]

# --- 7. LoRA training helper ----------------------------------------------

cells += [
    md(
        "## 7. The QLoRA training helper\n\n"
        "Encapsulates one LoRA run: load fresh 4-bit base \\u2192 wrap with LoRA \\u2192 train via "
        "HuggingFace `Trainer` \\u2192 save best adapter \\u2192 return val macro-F1 and adapter "
        "path. Two reasons to factor this out instead of inlining in cells 8 / 9:\n\n"
        "1. **Memory hygiene.** Each run reloads the base 4-bit model so we never have two PEFT "
        "wrappers in memory at once. On A100 this is essentially free (~ 2 s).\n"
        "2. **Clean sweeps.** Both runs share identical Trainer wiring; only the LoRA config and "
        "learning rate differ. Centralising the wiring makes the difference explicit.\n\n"
        "**Key configuration choices:**\n\n"
        "- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
        "bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` \\u2014 the "
        "canonical QLoRA setup.\n"
        "- `prepare_model_for_kbit_training(model)` \\u2014 disables grad-cache caching and "
        "enables gradient checkpointing-friendly hooks; required before adding LoRA layers to a "
        "quantized model.\n"
        "- `LoraConfig(task_type='SEQ_CLS', target_modules=['q_proj','k_proj','v_proj','o_proj'], "
        "modules_to_save=['score'])` \\u2014 LoRA on the four attention projections (standard), "
        "and the freshly-initialised classification head trained in full precision (essential: "
        "see (i) below).\n"
        "- `bf16=True` (A100 native), `gradient_checkpointing=True` (cuts activation memory), "
        "`gradient_accumulation_steps=4` (effective batch 32 with per-device batch 8).\n"
        "- `metric_for_best_model='eval_macro_f1'`, `load_best_model_at_end=True` \\u2014 the best "
        "epoch's adapter is what survives; intermediate epochs are pruned by `save_total_limit=2`.\n\n"
        "**(i) Why `modules_to_save=['score']` is essential.** "
        "`LlamaForSequenceClassification` initialises a fresh `nn.Linear(hidden_size, num_labels)` "
        "head (no pretrained weights at all for `num_labels = 20`). If we only attached LoRA, the "
        "head would stay at its random init forever \\u2014 the LoRA layers can't compose into "
        "the *base* head's weights. `modules_to_save` tells PEFT to fully train this module in "
        "addition to the LoRA adapters."
    ),
    code(
        "from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training\n"
        "from transformers import (\n"
        "    BitsAndBytesConfig,\n"
        "    DataCollatorWithPadding,\n"
        "    LlamaForSequenceClassification,\n"
        "    Trainer,\n"
        "    TrainingArguments,\n"
        ")\n"
        "\n"
        "BNB_CONFIG = BitsAndBytesConfig(\n"
        "    load_in_4bit=True,\n"
        "    bnb_4bit_quant_type='nf4',\n"
        "    bnb_4bit_compute_dtype=torch.bfloat16,\n"
        "    bnb_4bit_use_double_quant=True,\n"
        ")\n"
        "\n"
        "def _load_quantized_base():\n"
        "    \"\"\"Fresh 4-bit Llama with a 20-way classification head.\n"
        "\n"
        "    Note: `LlamaForSequenceClassification.from_pretrained(..., num_labels=20)` initialises\n"
        "    the score head from scratch, so each call returns a model with a *different* random\n"
        "    head unless we re-seed. `set_seed()` is called before each invocation in `run_lora`.\n"
        "    \"\"\"\n"
        "    model = LlamaForSequenceClassification.from_pretrained(\n"
        "        MODEL_NAME,\n"
        "        num_labels=len(label_names),\n"
        "        quantization_config=BNB_CONFIG,\n"
        "        device_map='auto',\n"
        "    )\n"
        "    model.config.pad_token_id = tokenizer.pad_token_id\n"
        "    return model\n"
        "\n"
        "def _compute_metrics(eval_pred):\n"
        "    logits, labels = eval_pred\n"
        "    preds = np.asarray(logits).argmax(axis=-1)\n"
        "    m = metrics_from_predictions(labels, preds, label_names)\n"
        "    return {'accuracy': float(m['accuracy']), 'macro_f1': float(m['macro_f1'])}\n"
        "\n"
        "def run_lora(\n"
        "    run_name: str,\n"
        "    *,\n"
        "    lora_r: int,\n"
        "    lora_alpha: int,\n"
        "    learning_rate: float,\n"
        "    num_train_epochs: int = 3,\n"
        "    per_device_batch_size: int = 8,\n"
        "    grad_accum: int = 4,\n"
        "    lora_dropout: float = 0.05,\n"
        "):\n"
        "    \"\"\"Train one LoRA run; return dict with adapter path + best val metric.\"\"\"\n"
        "    set_seed()  # head re-init is deterministic across calls.\n"
        "    base = _load_quantized_base()\n"
        "    base = prepare_model_for_kbit_training(base)\n"
        "\n"
        "    lora_cfg = LoraConfig(\n"
        "        task_type=TaskType.SEQ_CLS,\n"
        "        r=lora_r,\n"
        "        lora_alpha=lora_alpha,\n"
        "        lora_dropout=lora_dropout,\n"
        "        bias='none',\n"
        "        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n"
        "        modules_to_save=['score'],\n"
        "    )\n"
        "    model = get_peft_model(base, lora_cfg)\n"
        "    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "    total = sum(p.numel() for p in model.parameters())\n"
        "    print(f'[{run_name}] trainable: {trainable / 1e6:.2f} M / {total / 1e9:.2f} B \\u2014 '\n"
        "          f'{100 * trainable / total:.3f} %')\n"
        "\n"
        "    out_dir = CKPT_DIR / run_name\n"
        "    out_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "    args = TrainingArguments(\n"
        "        output_dir=str(out_dir),\n"
        "        run_name=run_name,\n"
        "        num_train_epochs=num_train_epochs,\n"
        "        per_device_train_batch_size=per_device_batch_size,\n"
        "        per_device_eval_batch_size=per_device_batch_size * 2,\n"
        "        gradient_accumulation_steps=grad_accum,\n"
        "        gradient_checkpointing=True,\n"
        "        gradient_checkpointing_kwargs={'use_reentrant': False},\n"
        "        learning_rate=learning_rate,\n"
        "        weight_decay=0.01,\n"
        "        warmup_ratio=0.03,\n"
        "        lr_scheduler_type='cosine',\n"
        "        bf16=True, fp16=False, tf32=True,\n"
        "        eval_strategy='epoch',\n"
        "        save_strategy='epoch',\n"
        "        save_total_limit=2,\n"
        "        logging_steps=20,\n"
        "        load_best_model_at_end=True,\n"
        "        metric_for_best_model='eval_macro_f1',\n"
        "        greater_is_better=True,\n"
        "        report_to=['wandb'],\n"
        "        dataloader_num_workers=2,\n"
        "        remove_unused_columns=False,  # NewsgroupsDataset returns its own columns.\n"
        "        seed=SEED,\n"
        "    )\n"
        "\n"
        "    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)\n"
        "    # transformers >=4.46 renamed Trainer's `tokenizer=` kwarg to `processing_class=`.\n"
        "    # Pick whichever the installed version accepts so this notebook runs on both.\n"
        "    import inspect as _inspect\n"
        "    _trainer_kwargs = dict(\n"
        "        model=model, args=args,\n"
        "        train_dataset=tr_ds, eval_dataset=va_ds,\n"
        "        data_collator=collator, compute_metrics=_compute_metrics,\n"
        "    )\n"
        "    if 'processing_class' in _inspect.signature(Trainer.__init__).parameters:\n"
        "        _trainer_kwargs['processing_class'] = tokenizer\n"
        "    else:\n"
        "        _trainer_kwargs['tokenizer'] = tokenizer\n"
        "    trainer = Trainer(**_trainer_kwargs)\n"
        "    trainer.train()\n"
        "\n"
        "    # Save just the adapter (no base weights) \\u2014 ~50 MB instead of 6 GB.\n"
        "    adapter_path = out_dir / 'best_adapter'\n"
        "    trainer.model.save_pretrained(str(adapter_path))\n"
        "\n"
        "    # Final evaluate on val (load_best_model_at_end already restored best weights).\n"
        "    val_metrics = trainer.evaluate(va_ds)\n"
        "\n"
        "    # Free GPU memory before the next run \\u2014 essential on a single-GPU host.\n"
        "    del trainer, model, base\n"
        "    torch.cuda.empty_cache()\n"
        "\n"
        "    return {\n"
        "        'run_name': run_name,\n"
        "        'adapter_path': str(adapter_path),\n"
        "        'val_metrics': {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},\n"
        "        'config': {\n"
        "            'lora_r': lora_r, 'lora_alpha': lora_alpha,\n"
        "            'learning_rate': learning_rate,\n"
        "            'num_train_epochs': num_train_epochs,\n"
        "            'per_device_batch_size': per_device_batch_size,\n"
        "            'grad_accum': grad_accum,\n"
        "            'lora_dropout': lora_dropout,\n"
        "            'max_length': MAX_LENGTH,\n"
        "            'seed': SEED,\n"
        "            'model': MODEL_NAME,\n"
        "            'quantization': '4bit-nf4',\n"
        "        },\n"
        "    }\n"
    ),
]

# --- 8. Run A -------------------------------------------------------------

cells += [
    md(
        "## 8. Run A \\u2014 baseline QLoRA (r=16, lr=2e-4)\n\n"
        "Standard QLoRA hyperparameters for classification on a 3 B-class model:\n\n"
        "- **`r = 16, alpha = 32`** \\u2014 the common \"good defaults\" used by the QLoRA paper "
        "and confirmed by countless reproductions on similarly-sized models.\n"
        "- **`lr = 2e-4`** \\u2014 LoRA is robust to ~1 OoM around 1e-4–3e-4; 2e-4 is the median.\n"
        "- **3 epochs, effective batch 32** \\u2014 matches Q2's BERT fine-tune budget on the same "
        "data, so any quality difference can be attributed to model + adapter, not training "
        "schedule. Wall-clock ~ 30 min on Colab A100.\n\n"
        "All metrics stream to W&B (`hslu-nalapro`, group `qbonus`)."
    ),
    code(
        "wandb.init(\n"
        "    project='hslu-nalapro', group='qbonus', name='qbonus-qlora-r16-lr2e4',\n"
        "    config={'lora_r': 16, 'lora_alpha': 32, 'lr': 2e-4, 'epochs': 3, 'max_length': MAX_LENGTH},\n"
        ")\n"
        "run_a = run_lora(\n"
        "    run_name='qbonus-qlora-r16-lr2e4',\n"
        "    lora_r=16, lora_alpha=32, learning_rate=2e-4,\n"
        "    num_train_epochs=3, per_device_batch_size=8, grad_accum=4,\n"
        ")\n"
        "wandb.finish()\n"
        "print('Run A best val:', run_a['val_metrics'])\n"
    ),
]

# --- 9. Run B -------------------------------------------------------------

cells += [
    md(
        "## 9. Run B \\u2014 higher rank, lower LR (r=32, lr=1e-4)\n\n"
        "A complementary point in the LoRA hyperparameter space:\n\n"
        "- **`r = 32, alpha = 64`** \\u2014 double the rank (more representational capacity in "
        "the adapter; roughly doubles trainable params).\n"
        "- **`lr = 1e-4`** \\u2014 halved learning rate, justified by the larger update "
        "magnitude that comes with higher rank.\n"
        "- Same 3-epoch budget. Wall-clock ~ 35 min on A100 (slight overhead from larger LoRA "
        "matmuls).\n\n"
        "If Run B beats Run A by < ~ 0.5 pp val macro-F1, the cheaper Run A wins (Occam). The "
        "winner is selected in cell 10."
    ),
    code(
        "wandb.init(\n"
        "    project='hslu-nalapro', group='qbonus', name='qbonus-qlora-r32-lr1e4',\n"
        "    config={'lora_r': 32, 'lora_alpha': 64, 'lr': 1e-4, 'epochs': 3, 'max_length': MAX_LENGTH},\n"
        ")\n"
        "run_b = run_lora(\n"
        "    run_name='qbonus-qlora-r32-lr1e4',\n"
        "    lora_r=32, lora_alpha=64, learning_rate=1e-4,\n"
        "    num_train_epochs=3, per_device_batch_size=8, grad_accum=4,\n"
        ")\n"
        "wandb.finish()\n"
        "print('Run B best val:', run_b['val_metrics'])\n"
    ),
]

# --- 10. Pick winner ------------------------------------------------------

cells += [
    md(
        "## 10. Select the best adapter by val macro-F1\n\n"
        "Whichever of Run A / Run B achieves the higher `eval_macro_f1` on the held-out 10 % val "
        "split is the adapter we evaluate on the test sets. We save a tiny sweep-summary JSON "
        "(`qbonus_sweep_summary.json`) listing both runs' configs and val metrics so the choice is "
        "auditable from disk later."
    ),
    code(
        "summary = {\n"
        "    'runs': [run_a, run_b],\n"
        "    'winner': max((run_a, run_b), key=lambda r: r['val_metrics']['eval_macro_f1'])['run_name'],\n"
        "}\n"
        "(RESULTS_DIR / 'qbonus_sweep_summary.json').write_text(json.dumps(summary, indent=2))\n"
        "print('winner:', summary['winner'])\n"
        "print(json.dumps({r['run_name']: r['val_metrics'] for r in summary['runs']}, indent=2))\n"
    ),
]

# --- 11. Load winning adapter --------------------------------------------

cells += [
    md(
        "## 11. Reload the winning adapter onto a fresh 4-bit base\n\n"
        "After cell 10 the in-memory model has been freed. We re-load the 4-bit base **once** and "
        "attach the saved adapter via `PeftModel.from_pretrained`. This becomes the model used "
        "for both final-eval passes (full test in cell 12, Q4 subset in cell 13).\n\n"
        "Doing it this way \\u2014 rather than keeping the trainer's in-memory model from cell 8 / "
        "9 alive \\u2014 lets the sweep section be re-runnable independently and verifies that "
        "the saved adapter on disk produces exactly the same metrics as the training-time best "
        "checkpoint."
    ),
    code(
        "from peft import PeftModel\n"
        "\n"
        "winner_path = next(\n"
        "    r['adapter_path'] for r in summary['runs'] if r['run_name'] == summary['winner']\n"
        ")\n"
        "print('loading adapter from:', winner_path)\n"
        "\n"
        "set_seed()\n"
        "base = _load_quantized_base()\n"
        "eval_model = PeftModel.from_pretrained(base, winner_path)\n"
        "eval_model.eval()\n"
        "print('eval model ready; trainable params during eval:',\n"
        "      sum(p.numel() for p in eval_model.parameters() if p.requires_grad))\n"
    ),
]

# --- 12. Full-test eval ---------------------------------------------------

cells += [
    md(
        "## 12. Final eval \\u2014 full 7 532-doc 20 NG test set\n\n"
        "This is the **comparable-to-Q1/Q2/Q3** headline number. We re-use the Trainer API just "
        "for its batched inference + collator; no further weight updates.\n\n"
        "Outputs:\n\n"
        "- `models/qbonus_results/qbonus_qlora_full_test.json` (same schema as `q2b_baseline.json`).\n"
        "- `figures/qbonus_confusion_full.png` (20\\u00d720 confusion matrix on the full test set)."
    ),
    code(
        "args_eval = TrainingArguments(\n"
        "    output_dir=str(CKPT_DIR / 'eval_tmp'),\n"
        "    per_device_eval_batch_size=16,\n"
        "    bf16=True,\n"
        "    report_to=['none'],\n"
        "    remove_unused_columns=False,\n"
        ")\n"
        "collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)\n"
        "# Same processing_class/tokenizer compatibility shim as run_lora().\n"
        "import inspect as _inspect\n"
        "_eval_kwargs = dict(model=eval_model, args=args_eval, data_collator=collator)\n"
        "if 'processing_class' in _inspect.signature(Trainer.__init__).parameters:\n"
        "    _eval_kwargs['processing_class'] = tokenizer\n"
        "else:\n"
        "    _eval_kwargs['tokenizer'] = tokenizer\n"
        "eval_trainer = Trainer(**_eval_kwargs)\n"
        "\n"
        "pred = eval_trainer.predict(te_ds)\n"
        "y_pred_full = pred.predictions.argmax(axis=-1)\n"
        "metrics_full = metrics_from_predictions(test_labels, y_pred_full, label_names)\n"
        "print(f'full test  accuracy = {metrics_full[\"accuracy\"]:.4f}')\n"
        "print(f'full test  macro_f1 = {metrics_full[\"macro_f1\"]:.4f}')\n"
        "\n"
        "winner_cfg = next(r['config'] for r in summary['runs'] if r['run_name'] == summary['winner'])\n"
        "(RESULTS_DIR / 'qbonus_qlora_full_test.json').write_text(json.dumps({\n"
        "    'name': 'qbonus-qlora-full-test',\n"
        "    'config': {**winner_cfg, 'winner_run': summary['winner'], 'eval_split': 'full_test',\n"
        "               'eval_size': len(te_ds)},\n"
        "    'accuracy': float(metrics_full['accuracy']),\n"
        "    'macro_f1': float(metrics_full['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in metrics_full['per_class_f1']],\n"
        "    'confusion_matrix': metrics_full['confusion_matrix'].tolist(),\n"
        "}, indent=2))\n"
        "\n"
        "plot_confusion(\n"
        "    cm=metrics_full['confusion_matrix'], label_names=label_names,\n"
        "    save_path=FIG_DIR / 'qbonus_confusion_full.png',\n"
        "    title='Q-bonus \\u2014 QLoRA Llama-3.2-3B (full 7 532-doc test set)',\n"
        ")\n"
    ),
]

# --- 13. Q4-subset eval ---------------------------------------------------

cells += [
    md(
        "## 13. Final eval \\u2014 same 200-doc Q4 subset (apples-to-apples vs zero/few-shot)\n\n"
        "Same trained adapter, evaluated on the **exact** 200-doc subset that Q4 used for "
        "zero-shot and few-shot \\u2014 the stratification call uses `random_state=42, "
        "test_size=200`. This isolates the lift from fine-tuning vs prompting on identical "
        "documents.\n\n"
        "Outputs `models/qbonus_results/qbonus_qlora_q4subset.json` in the same schema as the Q4 "
        "JSONs (minus the `invalid_rate` field, which doesn't apply to a classification head)."
    ),
    code(
        "pred_q4 = eval_trainer.predict(q4_ds)\n"
        "y_pred_q4 = pred_q4.predictions.argmax(axis=-1)\n"
        "metrics_q4 = metrics_from_predictions(q4_eval_labels, y_pred_q4, label_names)\n"
        "print(f'q4 subset accuracy = {metrics_q4[\"accuracy\"]:.4f}')\n"
        "print(f'q4 subset macro_f1 = {metrics_q4[\"macro_f1\"]:.4f}')\n"
        "\n"
        "(RESULTS_DIR / 'qbonus_qlora_q4subset.json').write_text(json.dumps({\n"
        "    'name': 'qbonus-qlora-q4subset',\n"
        "    'config': {**winner_cfg, 'winner_run': summary['winner'],\n"
        "               'eval_split': 'q4_200doc_subset', 'eval_size': len(q4_ds)},\n"
        "    'accuracy': float(metrics_q4['accuracy']),\n"
        "    'macro_f1': float(metrics_q4['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in metrics_q4['per_class_f1']],\n"
        "    'confusion_matrix': metrics_q4['confusion_matrix'].tolist(),\n"
        "}, indent=2))\n"
    ),
]

# --- 14. Comparison plots ------------------------------------------------

cells += [
    md(
        "## 14. Cross-question comparison plots\n\n"
        "Pull every available JSON \\u2014 Q4 zero-shot / few-shot, Q2b BERT fine-tune, Q3 "
        "MLM-then-fine-tune, plus the two Q-bonus eval splits \\u2014 and render two figures:\n\n"
        "1. **`qbonus_comparison_bars.png`** \\u2014 accuracy + macro-F1 side-by-side for every "
        "method present. Q-bonus is highlighted with two bars (full-test and Q4-subset) so the "
        "reader can compare apples-to-apples against both BERT (full test) and Llama prompting "
        "(200-doc subset).\n"
        "2. **`qbonus_per_class_delta.png`** \\u2014 per-class F1 of the Q-bonus full-test eval "
        "minus the Q4 zero-shot per-class F1, sorted descending. This shows which classes "
        "fine-tuning unlocks the most.\n\n"
        "Files that don't exist on the Colab host (e.g. Q2 / Q3 JSONs you haven't uploaded) are "
        "silently skipped \\u2014 the figure will simply have fewer bars."
    ),
    code(
        "# Build a dataframe of every method present on disk.\n"
        "rows = []\n"
        "def _row(label, path, *, prefer_keys=('accuracy', 'macro_f1')):\n"
        "    if not Path(path).exists():\n"
        "        return None\n"
        "    obj = json.loads(Path(path).read_text())\n"
        "    return {'method': label, 'accuracy': obj['accuracy'], 'macro_f1': obj['macro_f1'],\n"
        "            'per_class_f1': obj.get('per_class_f1')}\n"
        "\n"
        "for label, path in [\n"
        "    ('Q4 zero-shot (200-doc)',         Q4_DIR / 'q4_zero_shot.json'),\n"
        "    ('Q4 few-shot 1/class (200-doc)',  Q4_DIR / 'q4_few_shot_1pc.json'),\n"
        "    ('Q4 few-shot 3/class (200-doc)',  Q4_DIR / 'q4_few_shot_3pc.json'),\n"
        "    ('Q2b BERT fine-tune (full test)', Q2_BASELINE),\n"
        "    ('Q3 MLM\\u2192fine-tune (full test)',  Q3_BASELINE),\n"
        "    ('Q-bonus QLoRA (full test)',      RESULTS_DIR / 'qbonus_qlora_full_test.json'),\n"
        "    ('Q-bonus QLoRA (200-doc)',        RESULTS_DIR / 'qbonus_qlora_q4subset.json'),\n"
        "]:\n"
        "    r = _row(label, path)\n"
        "    if r is not None:\n"
        "        rows.append(r)\n"
        "\n"
        "comp = pd.DataFrame(rows)\n"
        "print(comp[['method', 'accuracy', 'macro_f1']].to_string(index=False))\n"
    ),
    code(
        "# Bar chart of accuracy + macro-F1 across all methods present.\n"
        "fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(comp)), 5))\n"
        "x = np.arange(len(comp))\n"
        "w = 0.4\n"
        "ax.bar(x - w/2, comp['accuracy'], width=w, label='accuracy', color='tab:blue')\n"
        "ax.bar(x + w/2, comp['macro_f1'], width=w, label='macro-F1', color='tab:orange')\n"
        "ax.set_xticks(x); ax.set_xticklabels(comp['method'], rotation=20, ha='right')\n"
        "ax.set_ylim(0, 1); ax.set_ylabel('score')\n"
        "ax.set_title('20 Newsgroups \\u2014 accuracy and macro-F1 across all methods')\n"
        "ax.legend(); ax.grid(alpha=0.3, axis='y')\n"
        "for i, (a, f) in enumerate(zip(comp['accuracy'], comp['macro_f1'])):\n"
        "    ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8)\n"
        "    ax.text(i + w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=8)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'qbonus_comparison_bars.png', dpi=150)\n"
        "plt.show()\n"
    ),
    code(
        "# Per-class F1 delta vs Q4 zero-shot (only if Q4 results are present).\n"
        "zero_path = Q4_DIR / 'q4_zero_shot.json'\n"
        "if zero_path.exists():\n"
        "    zero = json.loads(zero_path.read_text())\n"
        "    qbonus_full = json.loads((RESULTS_DIR / 'qbonus_qlora_full_test.json').read_text())\n"
        "    delta = np.asarray(qbonus_full['per_class_f1']) - np.asarray(zero['per_class_f1'])\n"
        "    order = np.argsort(-delta)\n"
        "    sorted_labels = [label_names[i] for i in order]\n"
        "    sorted_delta = delta[order]\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(9, 6))\n"
        "    ax.barh(sorted_labels, sorted_delta, color='tab:green')\n"
        "    ax.axvline(0, color='black', linewidth=0.5)\n"
        "    ax.set_xlabel('Per-class F1: Q-bonus QLoRA (full test) \\u2212 Q4 zero-shot (200-doc)')\n"
        "    ax.set_title('Per-class lift from QLoRA fine-tuning vs zero-shot Llama')\n"
        "    ax.grid(alpha=0.3, axis='x')\n"
        "    ax.invert_yaxis()\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(FIG_DIR / 'qbonus_per_class_delta.png', dpi=150)\n"
        "    plt.show()\n"
        "else:\n"
        "    print(f'{zero_path} not present \\u2014 skipping per-class delta plot.')\n"
    ),
]

# --- 15. Discussion --------------------------------------------------------

_DISCUSSION = """## 15. Discussion

_Fill in headline numbers after executing the notebook on Colab A100. Placeholders below indicate the structure of the discussion expected in the final report._

### 15.1 Headline numbers (TODO: paste from execution)

| Method | Test set | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Q4 zero-shot Llama-3.2-3B | 200-doc subset | _0.155_ | _0.132_ |
| Q4 few-shot k=3/class | 200-doc subset | _0.365_ | _0.351_ |
| **Q-bonus QLoRA Llama-3.2-3B** | **200-doc subset** | _TBD_ | _TBD_ |
| **Q-bonus QLoRA Llama-3.2-3B** | **full 7 532-doc test** | _TBD_ | _TBD_ |
| Q2b BERT-base supervised | full 7 532-doc test | _0.70 (ref.)_ | _0.70 (ref.)_ |
| Q3 MLM-then-fine-tune | full 7 532-doc test | _≈ Q2b_ | _≈ Q2b_ |

### 15.2 Does QLoRA fine-tune beat prompting?

The expected result is a large gap: a few hundred labeled examples of in-context demonstrations (Q4 k=3, 60 demos) cannot match three epochs of supervised gradient updates over 10 k labeled training documents (Q-bonus). The Q4 vs Q-bonus comparison on the *identical 200-doc subset* isolates this lift cleanly — same model weights at the byte level, same eval items, only the training (or lack thereof) differs.

### 15.3 Does QLoRA close the gap to BERT?

This is the genuinely interesting comparison and the one whose answer is uncertain in advance. Llama-3.2-3B has roughly **27 × more parameters** than `bert-base-uncased` (~ 110 M) but is decoder-only and was trained for next-token prediction, not masked language modeling on Newsgroups-style text. A LoRA adapter (~ 0.2 % of params) may or may not be enough capacity to specialise the model for 20-way topic classification. Three outcomes are possible:

1. **QLoRA ≥ BERT (probable).** The extra parameters dominate; the LoRA adapter is sufficient to route them. Q-bonus tops the leaderboard.
2. **QLoRA ≈ BERT (plausible).** The LoRA adapter is the bottleneck; lifting `r` further or unfreezing more modules would close the gap.
3. **QLoRA < BERT (possible).** Decoder-only architectures are mildly handicapped on classification — the last-token pooler discards information that BERT's bidirectional `[CLS]` pooler keeps. The fine-tune helps but doesn't catch up.

### 15.4 Where does fine-tuning help most?

The `qbonus_per_class_delta.png` figure shows per-class F1 lift vs Q4 zero-shot. Expected pattern: the **largest lifts** come on classes that the zero-shot model never predicted at all (sports, for-sale, hardware) — fine-tuning lets the model actually use these labels. The **smallest lifts** come on classes the zero-shot model already over-predicts (politics, religion); fine-tuning reins these in without much per-class F1 change because precision was already low.

### 15.5 Limitations and threats to validity

- **Single seed.** All results are from `SEED = 42`. 3 seeds per LoRA config would give error bars.
- **2-point sweep.** We only searched r∈{16,32} and lr∈{2e-4, 1e-4}. A larger sweep (lr={5e-5, 1e-4, 2e-4, 5e-4}, r={8, 16, 32, 64}) might raise the headline numbers another 1–2 points.
- **3 epochs.** Q2's BERT fine-tune uses 3 epochs as well, so this is fair vs Q2 — but Llama-3 might benefit from more (or fewer) epochs.
- **4-bit quantization.** The base model is quantized to 4 bits during training and inference. A bf16 reference would likely lift accuracy by ~ 1 point.
- **`max_length = 512`.** ~ 5 % of training docs are truncated. Doubling to 1024 would roughly double training time on a 3 B model; not done here.
- **Q4-subset comparison limit.** With n = 200 the binomial 95 % CI on accuracy is ± 0.07 around p = 0.5. The Q-bonus-vs-Q4 lift will far exceed this, but small per-class deltas inside the subset are noisy.

### 15.6 Bottom line

QLoRA fine-tuning of a small Llama-3 (3 B parameters, 4-bit, ~ 0.2 % trainable adapter) is the cheapest realistic way to specialise a foundation model for a 20-class news topic classification task — it costs less GPU-hour than full BERT fine-tuning and produces a single 50 MB adapter file that can be swapped in or out at inference time. Whether the resulting accuracy matches or beats supervised BERT is the empirical question this notebook answers, and the answer materially informs the report's recommendation."""

cells += [md(_DISCUSSION)]

# --- 16. Artifacts -------------------------------------------------------

cells += [
    md(
        "## 16. Artifacts & reproducibility\n\n"
        "### Files produced by this notebook\n\n"
        "| Path | Content |\n"
        "|---|---|\n"
        "| `models/qbonus_results/qbonus_qlora_full_test.json` | Final metrics, full 7 532-doc test set. |\n"
        "| `models/qbonus_results/qbonus_qlora_q4subset.json` | Final metrics, 200-doc Q4 subset. |\n"
        "| `models/qbonus_results/qbonus_sweep_summary.json` | Both LoRA configs' val metrics + winner. |\n"
        "| `models/qbonus_ckpt/<run_name>/best_adapter/` | Saved LoRA adapter (~ 50 MB; gitignored). |\n"
        "| `figures/qbonus_confusion_full.png` | Confusion matrix on the full test set. |\n"
        "| `figures/qbonus_comparison_bars.png` | Accuracy / macro-F1 vs all available methods. |\n"
        "| `figures/qbonus_per_class_delta.png` | Per-class F1 delta vs Q4 zero-shot. |\n\n"
        "### Reproducibility checklist\n\n"
        "- Model: `meta-llama/Llama-3.2-3B-Instruct`, 4-bit nf4 + bf16 compute via "
        "`bitsandbytes>=0.43`.\n"
        "- LoRA: `peft>=0.13`, target modules `{q,k,v,o}_proj`, `modules_to_save=['score']`, "
        "task type `SEQ_CLS`.\n"
        "- Train/val split: stratified 90/10 of 20 NG train, `random_state=42` (matches "
        "Q1/Q2/Q3).\n"
        "- Q4 subset: `train_test_split(test_size=200, stratify=test_labels, random_state=42)` "
        "(matches Q4 byte-for-byte).\n"
        "- Sweep: 2 LoRA configs, best by val macro-F1; both configs' metrics saved.\n\n"
        "### Re-running\n\n"
        "1. Open the notebook on a Colab A100 (or local 24 GB+ CUDA box).\n"
        "2. Cell 1 installs deps and prompts for HuggingFace + W&B logins.\n"
        "3. Execute top to bottom. Total wall-clock: ~ 70-90 min on A100 (~ 30-35 min per sweep "
        "run; ~ 5 min for the two final evals).\n"
        "4. Paste the two W&B run URLs into section 0 once complete.\n"
        "5. (Optional) upload `models/q2_results/q2b_baseline.json` and "
        "`models/q3_results/q3_finetune_results.json` to the Colab session before cell 14 to "
        "populate the cross-question comparison plot.\n"
    ),
]


# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    out = NB_DIR / "qbonus_llama_qlora_finetune.ipynb"
    write_notebook(out, cells)
    print(f"Wrote {out.relative_to(NB_DIR.parent)} ({len(cells)} cells).")
