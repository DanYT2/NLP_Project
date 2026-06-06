"""One-shot script to scaffold notebooks/q4_llama_zero_few_shot.ipynb.

The notebook is **self-contained** — it does not import anything from
`src/nlp_project/`. Every helper it needs (seeding, data loading, metrics,
plotting, prompt construction, demo selection, label parsing, model
loading, classification) is defined inline in the notebook itself. This
keeps the Colab path simple: no `git clone`, no `pip install -e .`, just
install runtime deps and go.

The notebook is *not* executed here — running it requires a CUDA GPU
(3060 laptop or Colab T4) and a Llama-3 download (~2 GB for the 3B model
in 4-bit). This script just writes the cells.

Re-running this script overwrites the notebook. After it is executed and
committed with output, do not run this script again without merging your
output cells back in.
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
# Q4 — single self-contained notebook
# ---------------------------------------------------------------------------

cells: list[dict] = []

# --- 0. Title & abstract --------------------------------------------------

cells += [
    md(
        "# Q4 — Llama-3 Zero-shot and Few-shot Classification on 20 Newsgroups\n\n"
        "**HSLU NALAPRO project — Question 4.** _Author: Dan Waititu (`danwwaititu@gmail.com`)._\n\n"
        "## Research question\n\n"
        "> _\"Using Llama-3 run zero-shot and few-shot learning experiments. Evaluate and discuss "
        "the results.\"_ — `project_description/NALAPRO Project.pdf`, §2 Question 4.\n\n"
        "Whereas Q2 (BERT fine-tune) and Q3 (MLM-then-fine-tune) update model weights with task "
        "supervision, this notebook freezes Llama-3 entirely and asks the model to classify each "
        "post by following an instruction prompt. The only training-set information the model gets "
        "is whatever appears in the prompt itself: in the **zero-shot** condition that is just the "
        "list of 20 newsgroup names; in the **few-shot** conditions we add 1 or 3 example documents "
        "per class so the model can pattern-match.\n\n"
        "## Self-contained\n\n"
        "This notebook does **not** import from the `nlp_project` package — every helper it needs "
        "is defined inline below (sections 3 and 4). That means it runs unchanged on Google Colab "
        "without cloning the repo, and on a local NVIDIA 3060 laptop with just the runtime "
        "dependencies installed. The trade-off is some duplication versus the rest of the project; "
        "for Q4 we judged portability more important than DRY.\n\n"
        "## Conditions tested in this notebook\n\n"
        "| Run name | k (demos / class) | Total demos in prompt | W&B run |\n"
        "|---|---:|---:|---|\n"
        "| Zero-shot | 0 | 0 | `q4-zero-shot` |\n"
        "| Few-shot (1/class) | 1 | 20 | `q4-few-shot-1pc` |\n"
        "| Few-shot (3/class) | 3 | 60 | `q4-few-shot-3pc` |\n\n"
        "All three are evaluated on the **same stratified 200-document test subsample** (10 per "
        "class, seed 42) so the comparison isolates the effect of in-context examples.\n\n"
        "## Model and hardware\n\n"
        "- **Model**: `meta-llama/Llama-3.2-3B-Instruct`. The smallest Llama-3 family member with "
        "a chat template; fits in ≈ 2 GB of VRAM under 4-bit nf4 quantization, so the notebook "
        "runs comfortably on a 6 GB NVIDIA 3060 laptop or a Colab free T4.\n"
        "- **Quantization**: 4-bit nf4 with bf16 compute (via `bitsandbytes`). Same on both hosts "
        "so the numbers are comparable between runs.\n"
        "- **Decoding**: greedy (`do_sample=False`), `max_new_tokens=15` — labels are short; "
        "greedy means identical inputs always give identical outputs.\n\n"
        "## Deliverables produced by this notebook\n\n"
        "- `models/q4_results/q4_zero_shot.json`, `q4_few_shot_1pc.json`, `q4_few_shot_3pc.json` "
        "— per-run metrics (accuracy, macro-F1, per-class F1, confusion matrix, invalid-rate).\n"
        "- `figures/q4_confusion_{zero,1pc,3pc}.png` — confusion matrices per condition.\n"
        "- `figures/q4_comparison_bars.png` — accuracy + macro-F1 across the three Llama runs.\n"
        "- `figures/q4_vs_q2b_bars.png` — Llama runs vs the Q2b supervised baseline (if "
        "`models/q2_results/q2b_baseline.json` is present locally).\n\n"
        "## W&B runs\n\n"
        "Project `hslu-nalapro`, group `q4`, three runs (one per condition). Paste the URLs below "
        "once executed:\n\n"
        "- <https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro?nw=nwuserdanwwaititu> "
        "_(group `q4`)_\n\n"
        "## AI tool disclosure\n\n"
        "Per spec §7, this notebook (cells + markdown) was co-authored with **Claude Code** "
        "(Anthropic, model `claude-opus-4-7`). The choice of model size, quantization scheme, "
        "subsample size, few-shot strategy, label-parsing heuristic, and the discussion section "
        "below reflect the author's judgement. The numerical results are reproduced from single "
        "local executions; seed is `SEED = 42` throughout."
    ),
]

# --- 1. Environment setup -------------------------------------------------

cells += [
    md(
        "## 1. Environment setup — local vs Google Colab\n\n"
        "This notebook runs unchanged on two hosts:\n\n"
        "1. **Local NVIDIA 3060 laptop** — Python 3.10+, `pip install transformers accelerate "
        "bitsandbytes wandb scikit-learn seaborn pandas tqdm matplotlib numpy torch`. (The HSLU "
        "repo also has a `uv` setup — `uv sync --extra llm` installs the same set — but you don't "
        "need any project file to run this notebook.)\n"
        "2. **Google Colab (free GPU)** — open the notebook, run the cell below; it installs "
        "the runtime deps and prompts for Hugging Face + Weights & Biases logins.\n\n"
        "**No repo clone, no `pip install -e .`.** Every helper this notebook uses is defined "
        "inline in sections 3 and 4. The cell below detects the environment, installs deps if "
        "needed, and prints a device summary so you know GPU acceleration is actually active."
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
        "    # On Colab we install runtime deps + interactively log in to HF (gated model) and W&B.\n"
        "    import subprocess, sys\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install',\n"
        "        'transformers>=4.45', 'accelerate>=0.34', 'bitsandbytes>=0.43',\n"
        "        'wandb', 'scikit-learn', 'seaborn', 'pandas', 'tqdm'])\n"
        "    from huggingface_hub import notebook_login as hf_login\n"
        "    hf_login()\n"
        "    import wandb\n"
        "    wandb.login()\n"
        "\n"
        "# --- device sanity check (both hosts) -------------------------------------\n"
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    print(f'CUDA OK — device 0 = {torch.cuda.get_device_name(0)}')\n"
        "    free, total = torch.cuda.mem_get_info()\n"
        "    print(f'free / total VRAM = {free / 1e9:.2f} / {total / 1e9:.2f} GB')\n"
        "else:\n"
        "    print('WARNING: no CUDA detected. The notebook will fall back to bf16/fp32 on CPU/MPS,')\n"
        "    print('which is far too slow for the 200-doc test subsample — use a GPU host.')\n"
    ),
]

# --- 2. Imports, seeding, output dirs -------------------------------------

cells += [
    md(
        "## 2. Imports, seed, output directories\n\n"
        "Standard scientific stack plus `transformers`, `wandb`, and `sklearn`. We define `SEED` "
        "and `set_seed` inline (instead of importing from the project package) so the notebook is "
        "self-contained.\n\n"
        "**Why seed everything:** greedy decoding doesn't strictly need a seed, but stratified "
        "subsampling and demo selection do — different seeds would change which 200 test docs and "
        "which demos the model sees, and therefore the headline numbers. We use `SEED = 42` "
        "throughout, matching Q1–Q3."
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
        "DEVICE = (\n"
        "    'cuda' if torch.cuda.is_available()\n"
        "    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()\n"
        "    else 'cpu'\n"
        ")\n"
        "print(f'device = {DEVICE}; seed = {SEED}')\n"
        "\n"
        "# Outputs land relative to the notebook's working directory. On Colab that's /content;\n"
        "# locally it depends on where Jupyter was launched. Both paths are created if missing.\n"
        "RESULTS_DIR = Path('models/q4_results'); RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "FIG_DIR = Path('figures'); FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
        "# Optional comparison-vs-Q2b: only used if the file already exists locally (it won't on Colab).\n"
        "Q2_BASELINE = Path('models/q2_results/q2b_baseline.json')\n"
        "print('results ->', RESULTS_DIR.resolve())\n"
        "print('figures ->', FIG_DIR.resolve())\n"
    ),
]

# --- 3. Inline helpers: data + metrics + plotting ------------------------

cells += [
    md(
        "## 3. Inline helpers — data, metrics, plotting\n\n"
        "Three small utilities, defined here so the notebook does not depend on the `nlp_project` "
        "package:\n\n"
        "- **`load_20ng(remove=True)`** — fetch the 20 Newsgroups train/test splits from sklearn. "
        "  `remove=True` strips headers/footers/quoted text (those leak the label; spec §3).\n"
        "- **`metrics_from_predictions(y_true, y_pred, label_names)`** — accuracy, macro-F1, "
        "  per-class F1, and confusion matrix in one shot. Uses the full label range so classes "
        "  that the model never predicts still occupy a slot.\n"
        "- **`plot_confusion(cm, label_names, save_path, title)`** — seaborn heatmap with class "
        "  names on both axes, saved as PNG.\n\n"
        "These are byte-for-byte equivalent to the implementations in "
        "`src/nlp_project/{data,eval}.py` — copied inline only to make this notebook portable."
    ),
    code(
        "from sklearn.datasets import fetch_20newsgroups\n"
        "from sklearn.metrics import accuracy_score, confusion_matrix as _cm, f1_score\n"
        "\n"
        "def load_20ng(remove: bool = True):\n"
        "    \"\"\"Fetch 20 Newsgroups. Returns (train_docs, train_labels, test_docs, test_labels, label_names).\n"
        "\n"
        "    When `remove=True` (default), strips headers, footers, and quoted text — those\n"
        "    contain the newsgroup name and similar tells that trivialise the task.\n"
        "    \"\"\"\n"
        "    remove_tuple = ('headers', 'footers', 'quotes') if remove else ()\n"
        "    tr = fetch_20newsgroups(subset='train', remove=remove_tuple)\n"
        "    te = fetch_20newsgroups(subset='test',  remove=remove_tuple)\n"
        "    return list(tr.data), np.asarray(tr.target), list(te.data), np.asarray(te.target), list(tr.target_names)\n"
        "\n"
        "def metrics_from_predictions(y_true, y_pred, label_names):\n"
        "    \"\"\"Standard classification metrics from already-computed predictions.\n"
        "\n"
        "    Returns dict with `accuracy`, `macro_f1`, `per_class_f1` (length n_classes),\n"
        "    and `confusion_matrix` (n_classes x n_classes).\n"
        "    \"\"\"\n"
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

# --- 4. Inline helpers: Llama classification ------------------------------

cells += [
    md(
        "## 4. Inline helpers — Llama classification\n\n"
        "Five small functions, defined inline so this notebook is self-contained:\n\n"
        "- **`SYSTEM_PROMPT`** — the system message template listing the 20 allowed labels.\n"
        "- **`build_prompt(doc, label_names, demos=None, truncate_chars=1500)`** — assembles the "
        "  chat-template messages list (system + optional alternating user/assistant demo turns + "
        "  final user query). Documents are truncated to keep the prompt short.\n"
        "- **`select_demos(train_docs, train_labels, label_names, k_per_class, seed=42)`** — "
        "  stratified per-class demo selection from the training set, deterministic given `seed`. "
        "  Uses a local RNG (`np.random.default_rng(seed)`) so callers can vary the seed without "
        "  disturbing global NumPy state.\n"
        "- **`parse_label(raw_output, label_names)`** — maps the model's generated text to a label "
        "  index. Strategy: lowercase normalize → exact match → substring match (longest wins) → "
        "  `difflib.get_close_matches(cutoff=0.6)` fuzzy fallback. Returns `(idx, invalid_flag)`; "
        "  unmatched outputs return `(0, True)` so the metrics functions still get a valid index.\n"
        "- **`load_llama(model_name, quantize_4bit=True, device='auto')`** — loads "
        "  `AutoModelForCausalLM` + tokenizer. On CUDA with `bitsandbytes`, uses 4-bit nf4 with "
        "  bf16 compute (~ 2 GB for the 3B model). Anywhere else, falls back to bf16/fp32 — this "
        "  keeps the notebook importable on a CPU/MPS box for code inspection.\n"
        "- **`classify_one`** / **`classify_batch`** — single-doc greedy generation + the loop. We "
        "  iterate one document at a time because batched chat-template inputs require left-padding "
        "  + per-row attention masks; the bookkeeping is fiddly and the speedup is modest on a "
        "  single 6 GB GPU.\n\n"
        "**Why generative + fuzzy parse rather than logit scoring?** Logit scoring (compute "
        "`log P(label | prompt)` for each of 20 labels and argmax) is always valid but slow (one "
        "forward pass per label per document) and does not test the model's instruction-following "
        "— which is the actual capability under evaluation. We go generative and track an "
        "`invalid_rate` diagnostic for outputs the parser cannot map to any label."
    ),
    code(
        "import difflib\n"
        "from collections.abc import Sequence\n"
        "from typing import Any\n"
        "\n"
        "SYSTEM_PROMPT = (\n"
        "    'You are a text classifier for the 20 Newsgroups dataset. Given a Usenet '\n"
        "    'post, respond with exactly one label from the following list and nothing '\n"
        "    'else \\u2014 no explanation, no punctuation around the label.\\n\\n'\n"
        "    'Allowed labels: {labels}'\n"
        ")\n"
        "\n"
        "def build_prompt(doc, label_names, demos=None, truncate_chars=1500):\n"
        "    \"\"\"Chat-template messages list for one document. `demos` is a list of\n"
        "    `(doc, label_index)` tuples that become user/assistant turn pairs.\n"
        "    \"\"\"\n"
        "    messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(labels=', '.join(label_names))}]\n"
        "    for demo_doc, demo_label in (demos or []):\n"
        "        messages.append({'role': 'user', 'content': demo_doc[:truncate_chars] + '\\n\\nLabel:'})\n"
        "        messages.append({'role': 'assistant', 'content': label_names[demo_label]})\n"
        "    messages.append({'role': 'user', 'content': doc[:truncate_chars] + '\\n\\nLabel:'})\n"
        "    return messages\n"
        "\n"
        "def select_demos(train_docs, train_labels, label_names, k_per_class, seed=SEED):\n"
        "    \"\"\"Stratified demo selection — `k_per_class` docs per newsgroup. Deterministic given `seed`.\"\"\"\n"
        "    rng = np.random.default_rng(seed)\n"
        "    train_labels = np.asarray(train_labels)\n"
        "    out = []\n"
        "    for class_idx in range(len(label_names)):\n"
        "        candidates = np.flatnonzero(train_labels == class_idx)\n"
        "        if len(candidates) == 0:\n"
        "            continue\n"
        "        take = min(k_per_class, len(candidates))\n"
        "        chosen = rng.choice(candidates, size=take, replace=False)\n"
        "        for idx in chosen:\n"
        "            out.append((train_docs[int(idx)], int(class_idx)))\n"
        "    return out\n"
        "\n"
        "def _normalize(text: str) -> str:\n"
        "    \"\"\"Lowercase, strip, replace dots/underscores with spaces for fuzzy matching.\"\"\"\n"
        "    return text.strip().lower().replace('.', ' ').replace('_', ' ')\n"
        "\n"
        "def parse_label(raw_output, label_names):\n"
        "    \"\"\"Map generated text -> (label_index, invalid_flag).\n"
        "\n"
        "    Strategy: exact normalized match -> substring (longest wins) -> difflib fuzzy.\n"
        "    Unmatched outputs return (0, True); the bool flag drives the `invalid_rate` diagnostic.\n"
        "    \"\"\"\n"
        "    norm_out = _normalize(raw_output)\n"
        "    norm_labels = [_normalize(l) for l in label_names]\n"
        "    for i, nl in enumerate(norm_labels):\n"
        "        if norm_out == nl:\n"
        "            return i, False\n"
        "    matches = []\n"
        "    for i, nl in enumerate(norm_labels):\n"
        "        if nl in norm_out or (norm_out and norm_out in nl):\n"
        "            matches.append((i, len(nl)))\n"
        "    if matches:\n"
        "        matches.sort(key=lambda x: -x[1])\n"
        "        return matches[0][0], False\n"
        "    close = difflib.get_close_matches(norm_out, norm_labels, n=1, cutoff=0.6)\n"
        "    if close:\n"
        "        return norm_labels.index(close[0]), False\n"
        "    return 0, True\n"
    ),
    code(
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "def load_llama(model_name='meta-llama/Llama-3.2-3B-Instruct',\n"
        "               quantize_4bit=True, device='auto', dtype=None):\n"
        "    \"\"\"Load an instruct Llama-3 + tokenizer.\n"
        "\n"
        "    On CUDA with `bitsandbytes`, uses 4-bit nf4 + bf16 compute (~ 2 GB for the 3B model).\n"
        "    Anywhere else, falls back to bf16/fp32 so the notebook stays importable on dev boxes.\n"
        "    \"\"\"\n"
        "    tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
        "    if tokenizer.pad_token_id is None:\n"
        "        # Llama-3 instruct tokenizers ship without a pad token; reuse EOS.\n"
        "        tokenizer.pad_token = tokenizer.eos_token\n"
        "\n"
        "    use_4bit = bool(quantize_4bit and torch.cuda.is_available())\n"
        "    if use_4bit:\n"
        "        try:\n"
        "            import bitsandbytes  # noqa: F401\n"
        "            from transformers import BitsAndBytesConfig\n"
        "            bnb = BitsAndBytesConfig(\n"
        "                load_in_4bit=True, bnb_4bit_quant_type='nf4',\n"
        "                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,\n"
        "            )\n"
        "            model = AutoModelForCausalLM.from_pretrained(\n"
        "                model_name, quantization_config=bnb, device_map=device,\n"
        "            )\n"
        "        except ImportError:\n"
        "            use_4bit = False\n"
        "\n"
        "    if not use_4bit:\n"
        "        if dtype is None:\n"
        "            if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):\n"
        "                dtype = torch.bfloat16\n"
        "            else:\n"
        "                dtype = torch.float32\n"
        "        model = AutoModelForCausalLM.from_pretrained(\n"
        "            model_name, torch_dtype=dtype,\n"
        "            device_map=device if device != 'auto' else None,\n"
        "        )\n"
        "        if device == 'auto':\n"
        "            if torch.cuda.is_available():\n"
        "                model = model.to('cuda')\n"
        "            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
        "                model = model.to('mps')\n"
        "    model.eval()\n"
        "    return model, tokenizer\n"
        "\n"
        "def classify_one(model, tokenizer, doc, label_names, demos=None,\n"
        "                 max_new_tokens=15, truncate_chars=1500):\n"
        "    \"\"\"Greedy classification of one document. Returns (pred_idx, raw_output, invalid_flag).\"\"\"\n"
        "    messages = build_prompt(doc, label_names, demos=demos, truncate_chars=truncate_chars)\n"
        "    prompt_text = tokenizer.apply_chat_template(\n"
        "        messages, add_generation_prompt=True, tokenize=False,\n"
        "    )\n"
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)\n"
        "    with torch.no_grad():\n"
        "        out = model.generate(\n"
        "            **inputs, max_new_tokens=max_new_tokens,\n"
        "            do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id,\n"
        "        )\n"
        "    new_tokens = out[0, inputs['input_ids'].shape[1]:]\n"
        "    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()\n"
        "    pred_idx, invalid = parse_label(raw, label_names)\n"
        "    return pred_idx, raw, invalid\n"
        "\n"
        "def classify_batch(model, tokenizer, docs, label_names, demos=None,\n"
        "                   max_new_tokens=15, truncate_chars=1500, progress=True):\n"
        "    \"\"\"Loop classify_one over `docs`. Returns dict with y_pred, raw_outputs, invalid_mask.\"\"\"\n"
        "    y_pred = np.zeros(len(docs), dtype=np.int64)\n"
        "    invalid_mask = np.zeros(len(docs), dtype=bool)\n"
        "    raw_outputs = []\n"
        "    iterator = enumerate(docs)\n"
        "    if progress:\n"
        "        try:\n"
        "            from tqdm.auto import tqdm\n"
        "            iterator = enumerate(tqdm(docs, desc='classifying'))\n"
        "        except ImportError:\n"
        "            pass\n"
        "    for i, doc in iterator:\n"
        "        pred_idx, raw, invalid = classify_one(\n"
        "            model, tokenizer, doc, label_names,\n"
        "            demos=demos, max_new_tokens=max_new_tokens, truncate_chars=truncate_chars,\n"
        "        )\n"
        "        y_pred[i] = pred_idx\n"
        "        raw_outputs.append(raw)\n"
        "        invalid_mask[i] = invalid\n"
        "    return {'y_pred': y_pred, 'raw_outputs': raw_outputs, 'invalid_mask': invalid_mask}\n"
    ),
]

# --- 5. Data --------------------------------------------------------------

cells += [
    md(
        "## 5. Data — 20 Newsgroups + stratified test subsample\n\n"
        "**Load.** Same call as Q1–Q3: `load_20ng(remove=True)`. Headers/footers/quoted text are "
        "stripped because they leak the label (spec §3, *label leakage*). Keeping this consistent "
        "across all four questions is what makes the cross-experiment comparison meaningful.\n\n"
        "**Why subsample.** The 20NG test set has 7 532 documents. At ~2–3 s per document with "
        "Llama-3.2-3B in 4-bit on a 3060, a single condition would take ~5–6 hours; three "
        "conditions would be a full day. We take a **stratified subsample of 200 documents (10 per "
        "class)** — fast enough to run all three conditions in roughly an hour, while still giving "
        "us 10 samples per class to estimate per-class F1.\n\n"
        "**Statistical limitation.** A 200-doc subsample has a binomial 95% CI of "
        "±1.96 · √(p·(1-p)/200) ≈ ±6 percentage points around accuracy = 0.5. So small "
        "between-condition deltas should be read with caution; we flag this in the discussion."
    ),
    code(
        "train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)\n"
        "print(f'train docs: {len(train_docs):>6d}')\n"
        "print(f'test docs : {len(test_docs):>6d}')\n"
        "print(f'classes   : {len(label_names):>6d}')\n"
        "\n"
        "# Stratified 200-doc subsample (10 / class). test_size=200 + stratify=labels\n"
        "# guarantees 10 docs per class because 200 / 20 == 10.\n"
        "_, eval_docs, _, eval_labels = train_test_split(\n"
        "    test_docs, test_labels,\n"
        "    train_size=len(test_docs) - 200,\n"
        "    test_size=200,\n"
        "    stratify=test_labels,\n"
        "    random_state=SEED,\n"
        ")\n"
        "eval_docs = list(eval_docs)\n"
        "eval_labels = np.asarray(eval_labels)\n"
        "print(f'eval subset: {len(eval_docs)} docs')\n"
        "print('per-class counts in eval subset:')\n"
        "pd.Series(eval_labels).value_counts().sort_index().rename(\n"
        "    index={i: n for i, n in enumerate(label_names)}\n"
        ")\n"
    ),
]

# --- 6. Model load --------------------------------------------------------

cells += [
    md(
        "## 6. Load Llama-3.2-3B-Instruct in 4-bit\n\n"
        "**Why 4-bit nf4.** Full-precision Llama-3.2-3B is ≈ 6 GB; in bf16 ≈ 6 GB; in 4-bit nf4 "
        "with double quantization it is ≈ 2 GB of weights plus activation memory. That fits "
        "comfortably on a 6 GB 3060 even at our k=3/class context length.\n\n"
        "**Why this matters for fairness.** Quantization changes the numerical behaviour of the "
        "model slightly versus the bf16 reference. We run *all three conditions* with the same "
        "4-bit configuration so the comparison between zero/few-shot is internally consistent.\n\n"
        "**Gated model.** Llama-3.2-3B-Instruct is gated on Hugging Face; the first run will fail "
        "if you haven't requested access on the model page and logged in. The Colab setup cell "
        "above (`hf_login`) handles this interactively."
    ),
    code(
        "MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'\n"
        "\n"
        "model, tokenizer = load_llama(MODEL_NAME, quantize_4bit=True, device='auto')\n"
        "n_params = sum(p.numel() for p in model.parameters()) / 1e9\n"
        "print(f'loaded {MODEL_NAME}: {n_params:.2f} B parameters')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')\n"
        "    print(f'VRAM reserved : {torch.cuda.memory_reserved() / 1e9:.2f} GB')\n"
    ),
]

# --- 7. Prompt walkthrough ------------------------------------------------

cells += [
    md(
        "## 7. Prompt walkthrough — what does the model actually see?\n\n"
        "It is easy to get prompting wrong silently, so we render one full chat-template prompt "
        "for the first eval document and print it. This is the literal string fed to the model "
        "(after BPE tokenization).\n\n"
        "**Structure for zero-shot:**\n"
        "1. A *system* message listing the 20 allowed labels and instructing the model to reply "
        "with exactly one of them.\n"
        "2. A *user* message containing the (truncated to 1500 chars) document and a trailing "
        "`Label:` cue.\n\n"
        "**For few-shot,** demonstration documents are inserted as alternating user / assistant "
        "turns before the final user query — the standard chat-template way to do in-context "
        "learning with an instruction-tuned model."
    ),
    code(
        "# Render the zero-shot prompt for the first eval doc.\n"
        "msgs_zero = build_prompt(eval_docs[0], label_names, demos=None, truncate_chars=1500)\n"
        "prompt_zero = tokenizer.apply_chat_template(msgs_zero, add_generation_prompt=True, tokenize=False)\n"
        "print('=' * 72)\n"
        "print('ZERO-SHOT PROMPT (first eval doc):')\n"
        "print('=' * 72)\n"
        "print(prompt_zero[:2000] + ('\\n... [truncated]' if len(prompt_zero) > 2000 else ''))\n"
        "n_tokens_zero = tokenizer(prompt_zero, return_tensors='pt')['input_ids'].shape[1]\n"
        "print(f'\\nprompt length: {n_tokens_zero} tokens')\n"
    ),
    code(
        "# Build 1/class and 3/class demo sets and report their prompt token lengths, so we can\n"
        "# eyeball that the few-shot machinery stays well within the model's context window.\n"
        "demos_preview = select_demos(train_docs, train_labels, label_names, k_per_class=1, seed=SEED)\n"
        "msgs_few = build_prompt(eval_docs[0], label_names, demos=demos_preview, truncate_chars=1500)\n"
        "prompt_few = tokenizer.apply_chat_template(msgs_few, add_generation_prompt=True, tokenize=False)\n"
        "n_tokens_few = tokenizer(prompt_few, return_tensors='pt')['input_ids'].shape[1]\n"
        "print(f'few-shot 1/class: {len(demos_preview)} demos, {n_tokens_few} tokens')\n"
        "\n"
        "demos_3 = select_demos(train_docs, train_labels, label_names, k_per_class=3, seed=SEED)\n"
        "msgs_3 = build_prompt(eval_docs[0], label_names, demos=demos_3, truncate_chars=1500)\n"
        "prompt_3 = tokenizer.apply_chat_template(msgs_3, add_generation_prompt=True, tokenize=False)\n"
        "n_tokens_3 = tokenizer(prompt_3, return_tensors='pt')['input_ids'].shape[1]\n"
        "print(f'few-shot 3/class: {len(demos_3)} demos, {n_tokens_3} tokens')\n"
        "print('\\nAll three prompt lengths must be well under Llama-3.2 context (128K tokens).')\n"
    ),
]

# --- 8. Zero-shot run -----------------------------------------------------

cells += [
    md(
        "## 8. Zero-shot run\n\n"
        "Classify all 200 eval documents with no in-context examples. The model is given only "
        "the 20 label names in the system message and must follow the instruction on instinct.\n\n"
        "**What we log to W&B:** accuracy, macro-F1, invalid-rate, plus the full config "
        "(model name, quantization, max_new_tokens, seed, subsample size). The metrics JSON "
        "written to disk mirrors the Q2/Q3 schema (with an added `invalid_rate` field) so the "
        "cross-experiment comparison cell can read all of them uniformly."
    ),
    code(
        "ZS_CONFIG = {\n"
        "    'model': MODEL_NAME,\n"
        "    'quantization': '4bit-nf4',\n"
        "    'k_per_class': 0,\n"
        "    'n_demos': 0,\n"
        "    'eval_size': len(eval_docs),\n"
        "    'max_new_tokens': 15,\n"
        "    'truncate_chars': 1500,\n"
        "    'seed': SEED,\n"
        "}\n"
        "\n"
        "wandb.init(\n"
        "    project='hslu-nalapro', group='q4', name='q4-zero-shot',\n"
        "    config=ZS_CONFIG, reinit=True,\n"
        ")\n"
        "\n"
        "out_zs = classify_batch(\n"
        "    model, tokenizer, eval_docs, label_names,\n"
        "    demos=None, max_new_tokens=15, truncate_chars=1500, progress=True,\n"
        ")\n"
        "metrics_zs = metrics_from_predictions(eval_labels, out_zs['y_pred'], label_names)\n"
        "invalid_rate_zs = float(out_zs['invalid_mask'].mean())\n"
        "\n"
        "wandb.log({\n"
        "    'accuracy': metrics_zs['accuracy'],\n"
        "    'macro_f1': metrics_zs['macro_f1'],\n"
        "    'invalid_rate': invalid_rate_zs,\n"
        "})\n"
        "wandb.finish()\n"
        "\n"
        "print(f'accuracy     = {metrics_zs[\"accuracy\"]:.4f}')\n"
        "print(f'macro_f1     = {metrics_zs[\"macro_f1\"]:.4f}')\n"
        "print(f'invalid_rate = {invalid_rate_zs:.4f}')\n"
    ),
    code(
        "# Persist results and render the confusion matrix.\n"
        "(RESULTS_DIR / 'q4_zero_shot.json').write_text(json.dumps({\n"
        "    'name': 'q4-zero-shot',\n"
        "    'config': ZS_CONFIG,\n"
        "    'accuracy': float(metrics_zs['accuracy']),\n"
        "    'macro_f1': float(metrics_zs['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in metrics_zs['per_class_f1']],\n"
        "    'confusion_matrix': metrics_zs['confusion_matrix'].tolist(),\n"
        "    'invalid_rate': invalid_rate_zs,\n"
        "}, indent=2))\n"
        "\n"
        "plot_confusion(\n"
        "    cm=metrics_zs['confusion_matrix'], label_names=label_names,\n"
        "    save_path=FIG_DIR / 'q4_confusion_zero.png',\n"
        "    title='Q4 — Llama-3.2-3B zero-shot (200-doc test subsample)',\n"
        ")\n"
    ),
]

# --- 9. Few-shot k=1/class -----------------------------------------------

cells += [
    md(
        "## 9. Few-shot (1 example per class)\n\n"
        "Same setup as zero-shot, but with 20 in-context demonstrations — one document per class, "
        "selected stratified from the *training* set with seed 42 (so demos are reproducible and "
        "never overlap the eval subsample, which comes from the *test* set).\n\n"
        "**Hypothesis.** With one example per class the model should at least see what each label "
        "looks like in practice, which should help on classes whose names alone are ambiguous "
        "(e.g. `talk.religion.misc` vs `soc.religion.christian`). Whether the lift is large "
        "depends on how much the demo style matches the eval document style."
    ),
    code(
        "DEMOS_1PC = select_demos(train_docs, train_labels, label_names, k_per_class=1, seed=SEED)\n"
        "print(f'selected {len(DEMOS_1PC)} demos (1 per class)')\n"
        "\n"
        "FS1_CONFIG = {\n"
        "    'model': MODEL_NAME, 'quantization': '4bit-nf4',\n"
        "    'k_per_class': 1, 'n_demos': len(DEMOS_1PC),\n"
        "    'eval_size': len(eval_docs), 'max_new_tokens': 15,\n"
        "    'truncate_chars': 1500, 'seed': SEED,\n"
        "}\n"
        "\n"
        "wandb.init(\n"
        "    project='hslu-nalapro', group='q4', name='q4-few-shot-1pc',\n"
        "    config=FS1_CONFIG, reinit=True,\n"
        ")\n"
        "\n"
        "out_fs1 = classify_batch(\n"
        "    model, tokenizer, eval_docs, label_names,\n"
        "    demos=DEMOS_1PC, max_new_tokens=15, truncate_chars=1500, progress=True,\n"
        ")\n"
        "metrics_fs1 = metrics_from_predictions(eval_labels, out_fs1['y_pred'], label_names)\n"
        "invalid_rate_fs1 = float(out_fs1['invalid_mask'].mean())\n"
        "\n"
        "wandb.log({\n"
        "    'accuracy': metrics_fs1['accuracy'],\n"
        "    'macro_f1': metrics_fs1['macro_f1'],\n"
        "    'invalid_rate': invalid_rate_fs1,\n"
        "})\n"
        "wandb.finish()\n"
        "\n"
        "print(f'accuracy     = {metrics_fs1[\"accuracy\"]:.4f}')\n"
        "print(f'macro_f1     = {metrics_fs1[\"macro_f1\"]:.4f}')\n"
        "print(f'invalid_rate = {invalid_rate_fs1:.4f}')\n"
    ),
    code(
        "(RESULTS_DIR / 'q4_few_shot_1pc.json').write_text(json.dumps({\n"
        "    'name': 'q4-few-shot-1pc',\n"
        "    'config': FS1_CONFIG,\n"
        "    'accuracy': float(metrics_fs1['accuracy']),\n"
        "    'macro_f1': float(metrics_fs1['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in metrics_fs1['per_class_f1']],\n"
        "    'confusion_matrix': metrics_fs1['confusion_matrix'].tolist(),\n"
        "    'invalid_rate': invalid_rate_fs1,\n"
        "}, indent=2))\n"
        "\n"
        "plot_confusion(\n"
        "    cm=metrics_fs1['confusion_matrix'], label_names=label_names,\n"
        "    save_path=FIG_DIR / 'q4_confusion_1pc.png',\n"
        "    title='Q4 — Llama-3.2-3B few-shot 1/class (200-doc test subsample)',\n"
        ")\n"
    ),
]

# --- 10. Few-shot k=3/class ----------------------------------------------

cells += [
    md(
        "## 10. Few-shot (3 examples per class)\n\n"
        "60 in-context demonstrations — three documents per class. The prompt is significantly "
        "longer (~10–25 k tokens depending on demo length), so each forward pass is slower; on a "
        "3060 expect ~5–8 s per document. The cell still completes in roughly 25 minutes for 200 "
        "docs.\n\n"
        "**Hypothesis.** More examples per class should help disambiguate the close ones "
        "(`comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`, etc.). Whether the marginal gain "
        "from 1→3 is larger or smaller than the 0→1 gain is genuinely uncertain a priori — "
        "the standard intuition is diminishing returns, but for a small model with a strong "
        "class-bias prior, additional demos may instead be needed to keep broadening the "
        "prediction repertoire. We measure it."
    ),
    code(
        "DEMOS_3PC = select_demos(train_docs, train_labels, label_names, k_per_class=3, seed=SEED)\n"
        "print(f'selected {len(DEMOS_3PC)} demos (3 per class)')\n"
        "\n"
        "FS3_CONFIG = {\n"
        "    'model': MODEL_NAME, 'quantization': '4bit-nf4',\n"
        "    'k_per_class': 3, 'n_demos': len(DEMOS_3PC),\n"
        "    'eval_size': len(eval_docs), 'max_new_tokens': 15,\n"
        "    'truncate_chars': 1500, 'seed': SEED,\n"
        "}\n"
        "\n"
        "wandb.init(\n"
        "    project='hslu-nalapro', group='q4', name='q4-few-shot-3pc',\n"
        "    config=FS3_CONFIG, reinit=True,\n"
        ")\n"
        "\n"
        "out_fs3 = classify_batch(\n"
        "    model, tokenizer, eval_docs, label_names,\n"
        "    demos=DEMOS_3PC, max_new_tokens=15, truncate_chars=1500, progress=True,\n"
        ")\n"
        "metrics_fs3 = metrics_from_predictions(eval_labels, out_fs3['y_pred'], label_names)\n"
        "invalid_rate_fs3 = float(out_fs3['invalid_mask'].mean())\n"
        "\n"
        "wandb.log({\n"
        "    'accuracy': metrics_fs3['accuracy'],\n"
        "    'macro_f1': metrics_fs3['macro_f1'],\n"
        "    'invalid_rate': invalid_rate_fs3,\n"
        "})\n"
        "wandb.finish()\n"
        "\n"
        "print(f'accuracy     = {metrics_fs3[\"accuracy\"]:.4f}')\n"
        "print(f'macro_f1     = {metrics_fs3[\"macro_f1\"]:.4f}')\n"
        "print(f'invalid_rate = {invalid_rate_fs3:.4f}')\n"
    ),
    code(
        "(RESULTS_DIR / 'q4_few_shot_3pc.json').write_text(json.dumps({\n"
        "    'name': 'q4-few-shot-3pc',\n"
        "    'config': FS3_CONFIG,\n"
        "    'accuracy': float(metrics_fs3['accuracy']),\n"
        "    'macro_f1': float(metrics_fs3['macro_f1']),\n"
        "    'per_class_f1': [float(x) for x in metrics_fs3['per_class_f1']],\n"
        "    'confusion_matrix': metrics_fs3['confusion_matrix'].tolist(),\n"
        "    'invalid_rate': invalid_rate_fs3,\n"
        "}, indent=2))\n"
        "\n"
        "plot_confusion(\n"
        "    cm=metrics_fs3['confusion_matrix'], label_names=label_names,\n"
        "    save_path=FIG_DIR / 'q4_confusion_3pc.png',\n"
        "    title='Q4 — Llama-3.2-3B few-shot 3/class (200-doc test subsample)',\n"
        ")\n"
    ),
]

# --- 11. Comparison -------------------------------------------------------

cells += [
    md(
        "## 11. Comparison — zero-shot vs few-shot vs supervised baseline\n\n"
        "Bring together the three Q4 runs and (if available) the Q2b BERT supervised baseline. "
        "The Q2b numbers were computed on the *full* 7 532-doc test set, while Q4 numbers are on "
        "a 200-doc stratified subsample, so the comparison vs Q2b is **indicative not exact** — "
        "we flag this in the discussion. The Q2b-vs-Q4 plot only renders if "
        "`models/q2_results/q2b_baseline.json` exists in the working directory (it won't on "
        "Colab unless you upload it manually).\n\n"
        "Two figures produced:\n\n"
        "1. `q4_comparison_bars.png` — accuracy + macro-F1 across the three Llama conditions.\n"
        "2. `q4_vs_q2b_bars.png` — best Llama condition vs Q2b supervised fine-tune (conditional)."
    ),
    code(
        "# Reload from JSON so this cell is independent of in-memory state and can be re-run\n"
        "# without re-executing the expensive inference cells.\n"
        "zs = json.loads((RESULTS_DIR / 'q4_zero_shot.json').read_text())\n"
        "fs1 = json.loads((RESULTS_DIR / 'q4_few_shot_1pc.json').read_text())\n"
        "fs3 = json.loads((RESULTS_DIR / 'q4_few_shot_3pc.json').read_text())\n"
        "\n"
        "summary = pd.DataFrame({\n"
        "    'condition': ['zero-shot', 'few-shot 1/class', 'few-shot 3/class'],\n"
        "    'accuracy': [zs['accuracy'], fs1['accuracy'], fs3['accuracy']],\n"
        "    'macro_f1': [zs['macro_f1'], fs1['macro_f1'], fs3['macro_f1']],\n"
        "    'invalid_rate': [zs['invalid_rate'], fs1['invalid_rate'], fs3['invalid_rate']],\n"
        "})\n"
        "summary\n"
    ),
    code(
        "# Bar chart: three Llama conditions side by side.\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "x = np.arange(len(summary))\n"
        "w = 0.35\n"
        "ax.bar(x - w/2, summary['accuracy'], width=w, label='accuracy', color='tab:blue')\n"
        "ax.bar(x + w/2, summary['macro_f1'], width=w, label='macro-F1', color='tab:orange')\n"
        "ax.set_xticks(x); ax.set_xticklabels(summary['condition'])\n"
        "ax.set_ylim(0, 1); ax.set_ylabel('score')\n"
        "ax.set_title('Q4 — Llama-3.2-3B across zero/few-shot conditions (200-doc subset)')\n"
        "ax.legend(); ax.grid(alpha=0.3, axis='y')\n"
        "for i, (a, f) in enumerate(zip(summary['accuracy'], summary['macro_f1'])):\n"
        "    ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)\n"
        "    ax.text(i + w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q4_comparison_bars.png', dpi=150)\n"
        "plt.show()\n"
    ),
    code(
        "# Comparison vs Q2b supervised baseline (only if the file is present locally).\n"
        "if Q2_BASELINE.exists():\n"
        "    q2b = json.loads(Q2_BASELINE.read_text())\n"
        "    candidates = [\n"
        "        ('zero-shot',         zs['accuracy'],  zs['macro_f1']),\n"
        "        ('few-shot 1/class',  fs1['accuracy'], fs1['macro_f1']),\n"
        "        ('few-shot 3/class',  fs3['accuracy'], fs3['macro_f1']),\n"
        "    ]\n"
        "    best_q4_name, best_q4_acc, best_q4_f1 = max(candidates, key=lambda x: x[1])\n"
        "    cmp = pd.DataFrame({\n"
        "        'model': [f'Q2b (BERT fine-tune, full test)', f'Q4 best ({best_q4_name}, 200-doc subset)'],\n"
        "        'accuracy': [q2b['accuracy'], best_q4_acc],\n"
        "        'macro_f1': [q2b['macro_f1'], best_q4_f1],\n"
        "    })\n"
        "    print(cmp.to_string(index=False))\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "    x = np.arange(len(cmp)); w = 0.35\n"
        "    ax.bar(x - w/2, cmp['accuracy'], width=w, label='accuracy', color='tab:blue')\n"
        "    ax.bar(x + w/2, cmp['macro_f1'], width=w, label='macro-F1', color='tab:orange')\n"
        "    ax.set_xticks(x); ax.set_xticklabels(cmp['model'])\n"
        "    ax.set_ylim(0, 1); ax.set_ylabel('score')\n"
        "    ax.set_title('Q4 best Llama-3 vs Q2b supervised baseline')\n"
        "    ax.legend(); ax.grid(alpha=0.3, axis='y')\n"
        "    for i, (a, f) in enumerate(zip(cmp['accuracy'], cmp['macro_f1'])):\n"
        "        ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)\n"
        "        ax.text(i + w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9)\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(FIG_DIR / 'q4_vs_q2b_bars.png', dpi=150)\n"
        "    plt.show()\n"
        "else:\n"
        "    print(f'Q2b baseline not found at {Q2_BASELINE}; skipping vs-Q2b plot.')\n"
        "    print('(That file is produced by the Q2 notebook; on Colab you can upload it manually.)')\n"
    ),
    code(
        "# Per-class F1 across the three Llama conditions — which classes does few-shot help?\n"
        "per_class = pd.DataFrame({\n"
        "    'label': label_names,\n"
        "    'zero_f1': zs['per_class_f1'],\n"
        "    'fs1_f1': fs1['per_class_f1'],\n"
        "    'fs3_f1': fs3['per_class_f1'],\n"
        "})\n"
        "per_class['delta_fs1_minus_zero'] = per_class['fs1_f1'] - per_class['zero_f1']\n"
        "per_class['delta_fs3_minus_zero'] = per_class['fs3_f1'] - per_class['zero_f1']\n"
        "print('Top 3 classes helped most by few-shot (k=1/class):')\n"
        "print(per_class.nlargest(3, 'delta_fs1_minus_zero')[['label', 'zero_f1', 'fs1_f1', 'delta_fs1_minus_zero']].to_string(index=False))\n"
        "print('\\nTop 3 classes hurt by few-shot (k=1/class):')\n"
        "print(per_class.nsmallest(3, 'delta_fs1_minus_zero')[['label', 'zero_f1', 'fs1_f1', 'delta_fs1_minus_zero']].to_string(index=False))\n"
    ),
]

# --- 12. Discussion -------------------------------------------------------

_DISCUSSION = """## 12. Discussion

_All numbers below are taken from this notebook's actual run on Google Colab (Tesla T4, 4-bit nf4 Llama-3.2-3B-Instruct, 200-document stratified test subsample, seed = 42)._

### 12.1 Headline numbers

| Condition | Accuracy | Macro-F1 | Invalid rate | Distinct classes predicted (of 20) |
|---|---:|---:|---:|---:|
| Zero-shot         | **0.155** | **0.132** | 0.005 | 14 |
| Few-shot 1/class  | **0.240** | **0.246** | 0.040 | 18 |
| Few-shot 3/class  | **0.365** | **0.351** | 0.035 | 19 |

For reference, random-choice on 20 classes is 0.05, and a binomial 95% CI at p ≈ 0.2 / n = 200 is roughly ± 0.06 accuracy. So all three between-condition differences (+0.085 zero → fs1, +0.125 fs1 → fs3) sit well above the noise floor.

### 12.2 Did few-shot help over zero-shot? Yes — and the returns _accelerate_

Accuracy roughly **doubles** from zero-shot (0.155) to k=1/class (0.240), then jumps by another +0.125 to 0.365 at k=3/class. That contradicts the diminishing-returns hypothesis sketched in §9 — for this small model on a 20-way task with awkward dotted labels, **more demos keep helping**, at least up to k=3. The plausible reason is that one demo per class is barely enough to break the model's class-bias prior (see §12.3); three demos start to teach actual surface patterns.

### 12.3 The dominant zero-shot failure is _class bias_, not parser noise

The invalid-rate diagnostic — the fraction of generations the parser cannot map to any label — is **0.5%** in zero-shot. The model is faithfully emitting valid label strings; the parser is not the bottleneck. The real failure mode is that Llama-3.2-3B has strong topical priors and falls back to a tiny set of \"favourite\" categories when uncertain:

- In zero-shot, 147 / 200 predictions (≈ 74%) go to just three labels: `talk.religion.misc` (52), `talk.politics.misc` (50), `talk.politics.guns` (45).
- Nine of the 20 classes (`alt.atheism`, `comp.os.ms-windows.misc`, `comp.windows.x`, `misc.forsale`, `rec.autos`, `rec.sport.baseball`, `rec.sport.hockey`, `sci.crypt`, `sci.med`) are **never predicted at all** in zero-shot — their per-class F1 is identically zero.
- The corresponding confusion matrix (`figures/q4_confusion_zero.png`) shows the symptom visually: a vertical band on the right-hand side (the model funnels everything into the politics/religion columns) and a near-empty diagonal.

Few-shot demonstrations directly attack this failure mode. With one demo per class the model expands its prediction repertoire from 14/20 to 18/20 distinct labels; at three demos per class it covers 19/20. The number of \"F1 = 0\" classes drops from 9 → 4 → 2 across the three conditions.

### 12.4 Where does few-shot help most (and where does it hurt)?

The classes that benefit most from a single demo per class are exactly those that the zero-shot model had never predicted at all:

| Class | F1 zero-shot | F1 few-shot 1/class | Δ |
|---|---:|---:|---:|
| `rec.sport.hockey`   | 0.000 | 0.667 | **+0.667** |
| `misc.forsale`       | 0.000 | 0.571 | **+0.571** |
| `rec.sport.baseball` | 0.000 | 0.471 | **+0.471** |

A single demonstration is enough to make the model aware that \"hockey\", \"baseball\" and \"for sale\" are valid output categories; once aware, it routes obvious documents correctly.

A handful of classes get **worse** with few-shot:

| Class | F1 zero-shot | F1 few-shot 1/class | Δ |
|---|---:|---:|---:|
| `comp.graphics`            | 0.421 | 0.000 | −0.421 |
| `talk.politics.guns`       | 0.218 | 0.000 | −0.218 |
| `soc.religion.christian`   | 0.154 | 0.125 | −0.029 |

A plausible mechanism: the demo for `comp.graphics` (a single training document) anchored the model's internal definition of that class on a narrow style; subsequent eval docs about graphics no longer matched and were re-routed to the neighbouring computer classes. This is a known fragility of single-demo in-context learning — the *one* example becomes a stereotype. Increasing k mitigates it (the model averages over multiple anchors), which is consistent with the broader k=3 lift.

### 12.5 How does Llama-3 compare to supervised BERT (Q2b)?

The Q2b baseline JSON was not present on the Colab host at run time, so `q4_vs_q2b_bars.png` was skipped (see the printed message under §11). For context: Q2b's `bert-base-uncased` reaches ≈ 0.71 accuracy / 0.70 macro-F1 on the **full** 7 532-doc test set after one epoch of supervised fine-tuning. Even our best Llama-3.2-3B condition (0.365 / 0.351 on a 200-doc subset) comes in roughly **35 absolute accuracy points below the supervised baseline**.

That is the expected ordering, and the size of the gap is the interesting part. A frozen 3 B-parameter LLM, prompted with at most 60 examples, recovers about **half** of what 11 k labeled training documents plus parameter updates buys you. The Q4 numbers are interesting as a measurement of what Llama-3.2-3B *already knows about Usenet topics* without any task-specific training — not as a competitive bid on 20 Newsgroups accuracy.

To produce the missing comparison plot, copy `models/q2_results/q2b_baseline.json` from a local run (or from the `ft-qn-2` branch) into the same path on the Colab session and re-execute §11.

### 12.6 Invalid-rate as a diagnostic

Invalid rate is low everywhere (0.5% zero-shot; 4.0% / 3.5% with few-shot). The mild *increase* with few-shot is expected: longer prompts with many in-context turns occasionally cause the model to drift into commentary (\"This appears to be about …\"). Even at the highest rate (4%) the parser fallback is rare enough that it does not dominate the metrics.

### 12.7 Limitations and threats to validity

- **200-document subsample.** Binomial 95% CI is ≈ ± 0.06 on accuracy at p = 0.2; per-class F1 has only 10 documents per class to estimate from, so any single per-class delta below ~ 0.2 should be read as anecdotal rather than statistically significant.
- **Single seed.** Demo selection, the 200-doc subsample, and the model are all seeded at 42. Different demonstrations would yield (probably better-or-worse-by-a-few-points) different numbers; 3–5 seeds per condition would be the rigorous setup.
- **4-bit quantization.** All three runs use 4-bit nf4 with bf16 compute. A bf16 reference run on an A100 would likely score ~ 1–2 points higher on accuracy.
- **Small model.** Llama-3.2-3B is the smallest instruction-tuned member of the family. An 8 B-class model (e.g. `unsloth/Meta-Llama-3.1-8B-Instruct` in 4-bit, ≈ 5 GB) typically lifts zero-shot 20 NG accuracy another 10–15 points. We chose 3 B to keep the notebook runnable on a 6 GB laptop 3060; the trade-off is honest.
- **Dotted label strings.** Showing the model labels like `comp.sys.ibm.pc.hardware` is token-hostile. Mapping the 20 raw labels to human-readable names (\"IBM PC hardware\") before prompting, and mapping back after, would likely lift both zero-shot and few-shot by another 5–10 points. We kept the raw labels to make every label string round-trip without ambiguity; this is a deliberate trade-off for measurement cleanliness over headline numbers.
- **Comparison to Q2 not at parity.** Q2b accuracy is on the full 7 532-doc test set; Q4 accuracy is on a stratified 200-doc subset. We report both but do not claim statistical equivalence.
- **Greedy decoding only.** Sampling with temperature > 0 followed by majority voting over k samples could lift accuracy a few points but is out of scope.

### 12.8 Bottom line

On 20 Newsgroups, a frozen Llama-3.2-3B-Instruct goes from chance × 3 (zero-shot) to chance × 7 (k=3 demos per class) without any parameter updates. Most of that lift comes from **breaking a strong \"favourite class\" prior** the base model carries; the in-context demonstrations function less as task instructions and more as a permission slip to predict the under-represented categories. Supervised fine-tuning (Q2/Q3) remains decisively better — but a frozen small LLM with a handful of examples is doing real work, and the trend across k=0, 1, 3 suggests there is still headroom at larger k or with better label strings."""

cells += [md(_DISCUSSION)]

# --- 13. Artifacts --------------------------------------------------------

cells += [
    md(
        "## 13. Artifacts & reproducibility\n\n"
        "### Files produced by this notebook\n\n"
        "| Path | Content |\n"
        "|---|---|\n"
        "| `models/q4_results/q4_zero_shot.json` | Zero-shot config + metrics |\n"
        "| `models/q4_results/q4_few_shot_1pc.json` | Few-shot k=1/class config + metrics |\n"
        "| `models/q4_results/q4_few_shot_3pc.json` | Few-shot k=3/class config + metrics |\n"
        "| `figures/q4_confusion_zero.png` | Confusion matrix, zero-shot |\n"
        "| `figures/q4_confusion_1pc.png` | Confusion matrix, k=1/class |\n"
        "| `figures/q4_confusion_3pc.png` | Confusion matrix, k=3/class |\n"
        "| `figures/q4_comparison_bars.png` | Accuracy + macro-F1 across the 3 Llama conditions |\n"
        "| `figures/q4_vs_q2b_bars.png` | Best Llama vs Q2b supervised baseline (if available) |\n\n"
        "### Reproducibility checklist\n\n"
        "- Model: `meta-llama/Llama-3.2-3B-Instruct`, 4-bit nf4 + bf16 compute via "
        "  `bitsandbytes>=0.43`.\n"
        "- Subsample: 200 stratified test docs, `random_state=SEED=42` via "
        "  `sklearn.model_selection.train_test_split`.\n"
        "- Demos: stratified per-class via inline `select_demos`, seed 42.\n"
        "- Decoding: greedy (`do_sample=False, num_beams=1`), `max_new_tokens=15`.\n"
        "- Parser: lowercased exact → substring → `difflib.get_close_matches(cutoff=0.6)`.\n\n"
        "### Re-running\n\n"
        "1. (Local) install `transformers accelerate bitsandbytes wandb scikit-learn seaborn pandas tqdm`; "
        "   `wandb login`; `huggingface-cli login`.\n"
        "2. (Colab) open the notebook; cell 1 installs everything and prompts for both logins.\n"
        "3. Run top to bottom. Sections 8–10 each take 5–25 minutes on a 3060; sections 11–13 are fast.\n"
        "4. Paste the three W&B run URLs into section 0 once complete.\n"
    ),
]


# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    out = NB_DIR / "q4_llama_zero_few_shot.ipynb"
    write_notebook(out, cells)
    print(f"Wrote {out.relative_to(NB_DIR.parent)} ({len(cells)} cells).")
