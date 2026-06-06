"""One-shot script to scaffold notebooks/q2{a,b,c,d}_*.ipynb.

The notebooks are not executed here — that requires W&B credentials and a
few minutes of MPS compute. This script writes the cells; the user runs
the notebooks via `uv run jupyter lab`.

Re-running this script overwrites the notebooks. After they are executed
and committed with output, do not run this script again.
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


# --- Q2a: token-length EDA ---

q2a = [
    md(
        "# Q2a — BERT tokenization setup\n\n"
        "**W&B run:** _none — pure EDA._\n\n"
        "This notebook tokenizes the 20 Newsgroups training set with `bert-base-uncased` and inspects\n"
        "token-length quantiles. Its only output is a decision on `max_length` for q2b/q2c."
    ),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from transformers import AutoTokenizer\n\n"
        "from nlp_project import set_seed\n"
        "from nlp_project.data import load_20ng\n\n"
        "set_seed()\n"
        "FIG_DIR = Path('../figures'); FIG_DIR.mkdir(exist_ok=True)\n"
    ),
    code(
        "train_docs, _, _, _, _ = load_20ng(remove=True)\n"
        "tok = AutoTokenizer.from_pretrained('bert-base-uncased')\n"
        "# Truncate-disabled count of WordPiece tokens per doc.\n"
        "lens = np.array([len(tok.encode(d, add_special_tokens=True, truncation=False)) for d in train_docs])\n"
        "print(f'n_docs = {len(lens)}')\n"
        "for q in [0.5, 0.75, 0.9, 0.95, 0.99]:\n"
        "    print(f'p{int(q*100):>2d}: {int(np.quantile(lens, q))}')\n"
        "print(f'max: {int(lens.max())}')\n"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.hist(np.clip(lens, 0, 1024), bins=50, color='#3a6ea5')\n"
        "ax.axvline(128, color='orange', ls='--', label='128')\n"
        "ax.axvline(256, color='red', ls='--', label='256')\n"
        "ax.axvline(512, color='black', ls='--', label='512 (BERT max)')\n"
        "ax.set_xlabel('WordPiece tokens per document (clipped at 1024)')\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title('20 Newsgroups — token length, bert-base-uncased')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q2a_token_length_hist.png', dpi=150)\n"
        "plt.show()\n"
    ),
    md(
        "## Decision\n\n"
        "`max_length=256` covers ~p90 of the corpus and is the largest size we can comfortably sweep on MPS\n"
        "within an afternoon. 512 is noted as a future ablation but skipped here for compute reasons.\n"
    ),
]

# --- Q2b: baseline fine-tune ---

q2b = [
    md(
        "# Q2b — BERT fine-tuning baseline\n\n"
        "**W&B run:** _filled in after first run_\n\n"
        "Single tuned fine-tuning run on `bert-base-uncased` with the hyperparameters that the report\n"
        "treats as the Q2 baseline. The saved metrics here feed the cross-experiment comparison in q2d."
    ),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import wandb\n\n"
        "from nlp_project import SEED, set_seed\n"
        "from nlp_project.bert_data import build_splits\n"
        "from nlp_project.bert_train import run_finetune\n"
        "from nlp_project.eval import plot_confusion\n\n"
        "set_seed()\n"
        "FIG_DIR = Path('../figures'); FIG_DIR.mkdir(exist_ok=True)\n"
        "MODEL_DIR = Path('../models'); MODEL_DIR.mkdir(exist_ok=True)\n"
        "RESULTS_DIR = Path('../models') / 'q2_results'; RESULTS_DIR.mkdir(exist_ok=True)\n\n"
        "os.environ.setdefault('WANDB_PROJECT', 'hslu-nalapro')\n"
        "DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'\n"
        "print(f'device: {DEVICE}')\n"
    ),
    code(
        "MODEL_NAME = 'bert-base-uncased'\n"
        "MAX_LENGTH = 256\n"
        "splits = build_splits(tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH, seed=SEED)\n"
        "print(f\"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}\")\n"
    ),
    code(
        "RUN_NAME = 'q2b-bert-base-uncased-baseline'\n"
        "config = {\n"
        "    'model': MODEL_NAME,\n"
        "    'max_length': MAX_LENGTH,\n"
        "    'lr': 2e-5,\n"
        "    'batch_size': 16,\n"
        "    'epochs': 3,\n"
        "    'warmup_ratio': 0.1,\n"
        "    'weight_decay': 0.01,\n"
        "    'freeze_encoder': False,\n"
        "    'seed': SEED,\n"
        "}\n"
        "run = wandb.init(project='hslu-nalapro', name=RUN_NAME, group='q2', config=config)\n"
    ),
    code(
        "result = run_finetune(\n"
        "    splits=splits,\n"
        "    out_dir=MODEL_DIR / 'q2b',\n"
        "    model_name=MODEL_NAME,\n"
        "    lr=config['lr'],\n"
        "    epochs=config['epochs'],\n"
        "    batch_size=config['batch_size'],\n"
        "    run_name=RUN_NAME,\n"
        "    weight_decay=config['weight_decay'],\n"
        "    warmup_ratio=config['warmup_ratio'],\n"
        "    report_to=['wandb'],\n"
        ")\n"
        "metrics = result['test_metrics']\n"
        "print(f\"test accuracy: {metrics['accuracy']:.4f}\")\n"
        "print(f\"test macro-F1: {metrics['macro_f1']:.4f}\")\n"
        "print(f\"best val macro-F1: {result['best_eval_metric']:.4f}\")\n"
    ),
    code(
        "plot_confusion(\n"
        "    metrics['confusion_matrix'], splits['label_names'],\n"
        "    save_path=FIG_DIR / 'q2b_confusion_matrix.png',\n"
        "    title='Q2b — bert-base-uncased baseline',\n"
        ")\n"
        "run.log({\n"
        "    'test_accuracy': metrics['accuracy'],\n"
        "    'test_macro_f1': metrics['macro_f1'],\n"
        "})\n"
        "run.finish()\n"
    ),
    code(
        "# Persist the baseline metrics for q2d to pick up.\n"
        "out = {\n"
        "    'name': RUN_NAME,\n"
        "    'config': config,\n"
        "    'accuracy': float(metrics['accuracy']),\n"
        "    'macro_f1': float(metrics['macro_f1']),\n"
        "    'per_class_f1': metrics['per_class_f1'].tolist(),\n"
        "    'confusion_matrix': metrics['confusion_matrix'].tolist(),\n"
        "}\n"
        "(RESULTS_DIR / 'q2b_baseline.json').write_text(json.dumps(out, indent=2))\n"
        "print('saved', RESULTS_DIR / 'q2b_baseline.json')\n"
    ),
    md(
        "## Notes for the report\n\n"
        "- This run is the Q2 baseline. Compare against Q1's TF-IDF and word2vec-mean numbers in q2d.\n"
        "- Hyperparameters were not tuned at this stage; q2c sweeps them.\n"
    ),
]

# --- Q2c: ablations ---

q2c = [
    md(
        "# Q2c — Parameter & ablation sweep\n\n"
        "**W&B group:** `q2`\n\n"
        "Runs the four ablations called out in the plan:\n\n"
        "1. **LR sweep** — {1e-5, 2e-5, 3e-5, 5e-5}\n"
        "2. **Seq-len ablation** — {128, 256} at the best LR\n"
        "3. **Frozen encoder (linear probe)** — head-only training, higher LR\n"
        "4. **Cased vs uncased** — `bert-base-cased` at the best (LR, seq_len)\n\n"
        "All runs write a row to `models/q2_results/q2c_sweep.json` for the comparison in q2d.\n"
    ),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import torch\n"
        "import wandb\n\n"
        "from nlp_project import SEED, set_seed\n"
        "from nlp_project.bert_data import build_splits\n"
        "from nlp_project.bert_train import run_finetune\n\n"
        "set_seed()\n"
        "FIG_DIR = Path('../figures'); FIG_DIR.mkdir(exist_ok=True)\n"
        "MODEL_DIR = Path('../models'); MODEL_DIR.mkdir(exist_ok=True)\n"
        "RESULTS_DIR = Path('../models') / 'q2_results'; RESULTS_DIR.mkdir(exist_ok=True)\n\n"
        "os.environ.setdefault('WANDB_PROJECT', 'hslu-nalapro')\n"
        "print(f\"mps={torch.backends.mps.is_available()}\")\n"
    ),
    code(
        "results: list[dict] = []\n\n"
        "def _record(run_name: str, cfg: dict, res: dict) -> None:\n"
        "    tm = res['test_metrics']\n"
        "    results.append({\n"
        "        'name': run_name,\n"
        "        **cfg,\n"
        "        'accuracy': float(tm['accuracy']),\n"
        "        'macro_f1': float(tm['macro_f1']),\n"
        "        'best_val_macro_f1': float(res['best_eval_metric'] or 0.0),\n"
        "    })\n"
    ),
    code(
        "# 1. LR sweep at seq_len=256, uncased, 3 epochs.\n"
        "MODEL = 'bert-base-uncased'\n"
        "splits_256 = build_splits(tokenizer_name=MODEL, max_length=256, seed=SEED)\n\n"
        "for lr in [1e-5, 2e-5, 3e-5, 5e-5]:\n"
        "    cfg = {'model': MODEL, 'max_length': 256, 'lr': lr, 'batch_size': 16,\n"
        "           'epochs': 3, 'freeze_encoder': False, 'experiment': 'lr_sweep'}\n"
        "    run_name = f'q2c-lr-{lr:.0e}'\n"
        "    run = wandb.init(project='hslu-nalapro', name=run_name, group='q2', config=cfg, reinit=True)\n"
        "    res = run_finetune(\n"
        "        splits=splits_256, out_dir=MODEL_DIR / f'q2c/{run_name}',\n"
        "        model_name=MODEL, lr=lr, epochs=3, batch_size=16,\n"
        "        run_name=run_name, report_to=['wandb'],\n"
        "    )\n"
        "    _record(run_name, cfg, res)\n"
        "    run.log({'test_accuracy': res['test_metrics']['accuracy'],\n"
        "             'test_macro_f1': res['test_metrics']['macro_f1']})\n"
        "    run.finish()\n"
    ),
    code(
        "# Best LR from the sweep so far.\n"
        "lr_rows = [r for r in results if r.get('experiment') == 'lr_sweep']\n"
        "best_lr = max(lr_rows, key=lambda r: r['macro_f1'])['lr']\n"
        "print(f'best LR: {best_lr}')\n"
    ),
    code(
        "# 2. Seq-len ablation at best LR.\n"
        "splits_128 = build_splits(tokenizer_name=MODEL, max_length=128, seed=SEED)\n"
        "for seq_len, splits in [(128, splits_128), (256, splits_256)]:\n"
        "    cfg = {'model': MODEL, 'max_length': seq_len, 'lr': best_lr, 'batch_size': 16,\n"
        "           'epochs': 3, 'freeze_encoder': False, 'experiment': 'seqlen_ablation'}\n"
        "    run_name = f'q2c-seqlen-{seq_len}'\n"
        "    run = wandb.init(project='hslu-nalapro', name=run_name, group='q2', config=cfg, reinit=True)\n"
        "    res = run_finetune(\n"
        "        splits=splits, out_dir=MODEL_DIR / f'q2c/{run_name}',\n"
        "        model_name=MODEL, lr=best_lr, epochs=3, batch_size=16,\n"
        "        run_name=run_name, report_to=['wandb'],\n"
        "    )\n"
        "    _record(run_name, cfg, res)\n"
        "    run.log({'test_accuracy': res['test_metrics']['accuracy'],\n"
        "             'test_macro_f1': res['test_metrics']['macro_f1']})\n"
        "    run.finish()\n"
    ),
    code(
        "# 3. Linear probe (frozen encoder, higher LR for the head).\n"
        "cfg = {'model': MODEL, 'max_length': 256, 'lr': 1e-3, 'batch_size': 16,\n"
        "       'epochs': 5, 'freeze_encoder': True, 'experiment': 'linear_probe'}\n"
        "run_name = 'q2c-frozen-linearprobe'\n"
        "run = wandb.init(project='hslu-nalapro', name=run_name, group='q2', config=cfg, reinit=True)\n"
        "res = run_finetune(\n"
        "    splits=splits_256, out_dir=MODEL_DIR / f'q2c/{run_name}',\n"
        "    model_name=MODEL, lr=1e-3, epochs=5, batch_size=16,\n"
        "    run_name=run_name, freeze_encoder_=True, report_to=['wandb'],\n"
        ")\n"
        "_record(run_name, cfg, res)\n"
        "run.log({'test_accuracy': res['test_metrics']['accuracy'],\n"
        "         'test_macro_f1': res['test_metrics']['macro_f1']})\n"
        "run.finish()\n"
    ),
    code(
        "# 4. Cased vs uncased at best LR + seq=256.\n"
        "splits_cased = build_splits(tokenizer_name='bert-base-cased', max_length=256, seed=SEED)\n"
        "cfg = {'model': 'bert-base-cased', 'max_length': 256, 'lr': best_lr, 'batch_size': 16,\n"
        "       'epochs': 3, 'freeze_encoder': False, 'experiment': 'cased_vs_uncased'}\n"
        "run_name = 'q2c-cased'\n"
        "run = wandb.init(project='hslu-nalapro', name=run_name, group='q2', config=cfg, reinit=True)\n"
        "res = run_finetune(\n"
        "    splits=splits_cased, out_dir=MODEL_DIR / f'q2c/{run_name}',\n"
        "    model_name='bert-base-cased', lr=best_lr, epochs=3, batch_size=16,\n"
        "    run_name=run_name, report_to=['wandb'],\n"
        ")\n"
        "_record(run_name, cfg, res)\n"
        "run.log({'test_accuracy': res['test_metrics']['accuracy'],\n"
        "         'test_macro_f1': res['test_metrics']['macro_f1']})\n"
        "run.finish()\n"
    ),
    code(
        "# Save sweep + render the comparison table and bar chart.\n"
        "(RESULTS_DIR / 'q2c_sweep.json').write_text(json.dumps(results, indent=2))\n\n"
        "df = pd.DataFrame(results).sort_values('macro_f1', ascending=False).reset_index(drop=True)\n"
        "display(df[['name', 'experiment', 'lr', 'max_length', 'freeze_encoder', 'accuracy', 'macro_f1']])\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.bar(df['name'], df['macro_f1'], color='#3a6ea5')\n"
        "ax.set_ylabel('test macro-F1')\n"
        "ax.set_title('Q2c — ablations (test macro-F1)')\n"
        "plt.setp(ax.get_xticklabels(), rotation=45, ha='right')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q2c_sweep_bars.png', dpi=150)\n"
        "plt.show()\n"
    ),
    md(
        "## Notes for the report\n\n"
        "- The LR sweep typically peaks around 2e-5–3e-5 for BERT-base.\n"
        "- The linear probe is the floor — anything close to its accuracy means fine-tuning didn't help.\n"
        "- Cased vs uncased is small in newsgroup text (lowercase-dominant); report the delta.\n"
    ),
]

# --- Q2d: cross-experiment comparison ---

q2d = [
    md(
        "# Q2d — Q1 vs Q2 comparison\n\n"
        "**Not a training run.** Aggregates the JSON outputs of q1c/q1d (paste in below) and q2b/q2c into\n"
        "the headline comparison table and figures the report needs.\n"
    ),
    code(
        "%load_ext autoreload\n"
        "%autoreload 2\n\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from nlp_project import set_seed\n"
        "from nlp_project.data import load_20ng\n"
        "from nlp_project.eval import plot_confusion\n\n"
        "set_seed()\n"
        "FIG_DIR = Path('../figures'); FIG_DIR.mkdir(exist_ok=True)\n"
        "RESULTS_DIR = Path('../models') / 'q2_results'\n"
        "_, _, _, _, label_names = load_20ng(remove=True)\n"
    ),
    code(
        "# Q1 final numbers pulled from the executed q1b/q1c/q1d notebooks.\n"
        "q1_results = [\n"
        "    {'name': 'Q1b word2vec mean-pool',     'accuracy': 0.6321, 'macro_f1': 0.6098},\n"
        "    {'name': 'Q1c TF-IDF',                  'accuracy': 0.6964, 'macro_f1': 0.6895},\n"
        "    {'name': 'Q1d word2vec mean+max-pool', 'accuracy': 0.6202, 'macro_f1': 0.6004},\n"
        "]\n"
        "q2b = json.loads((RESULTS_DIR / 'q2b_baseline.json').read_text())\n"
        "q2c_rows = json.loads((RESULTS_DIR / 'q2c_sweep.json').read_text())\n"
        "q2c_best = max(q2c_rows, key=lambda r: r['macro_f1'])\n"
        "print('Q2 baseline:', q2b['name'], q2b['macro_f1'])\n"
        "print('Q2 best    :', q2c_best['name'], q2c_best['macro_f1'])\n"
    ),
    code(
        "table = pd.DataFrame(\n"
        "    q1_results + [\n"
        "        {'name': 'Q2b BERT baseline', 'accuracy': q2b['accuracy'], 'macro_f1': q2b['macro_f1']},\n"
        "        {'name': f\"Q2c best ({q2c_best['name']})\", 'accuracy': q2c_best['accuracy'],\n"
        "         'macro_f1': q2c_best['macro_f1']},\n"
        "    ]\n"
        ")\n"
        "display(table)\n"
        "table.to_csv(RESULTS_DIR / 'q1_vs_q2_table.csv', index=False)\n"
    ),
    code(
        "# Side-by-side bar chart.\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "x = np.arange(len(table))\n"
        "w = 0.35\n"
        "ax.bar(x - w/2, table['accuracy'], w, label='accuracy', color='#3a6ea5')\n"
        "ax.bar(x + w/2, table['macro_f1'], w, label='macro-F1', color='#e08820')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(table['name'], rotation=30, ha='right')\n"
        "ax.set_ylim(0, 1)\n"
        "ax.set_title('Q1 vs Q2 — accuracy and macro-F1')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'q2d_comparison_bars.png', dpi=150)\n"
        "plt.show()\n"
    ),
    code(
        "# Q2 baseline confusion matrix — already saved in q2b; re-render here for completeness.\n"
        "cm = np.array(q2b['confusion_matrix'])\n"
        "plot_confusion(cm, label_names,\n"
        "               save_path=FIG_DIR / 'q2d_q2b_cm.png',\n"
        "               title='Q2 baseline — confusion matrix')\n"
    ),
    md(
        "## Notes for the report\n\n"
        "- BERT is expected to beat Q1's best by a clear margin on macro-F1; the per-class breakdown\n"
        "  often shows the biggest gains on classes that share vocabulary (e.g. talk.* groups).\n"
        "- Discuss compute trade-off: BERT-base has ~110M params vs Q1 MLP's small head — accuracy gain\n"
        "  comes at a >100× training cost.\n"
    ),
]


def main() -> None:
    write_notebook(NB_DIR / "q2a_bert_setup.ipynb", q2a)
    write_notebook(NB_DIR / "q2b_bert_baseline.ipynb", q2b)
    write_notebook(NB_DIR / "q2c_bert_experiments.ipynb", q2c)
    write_notebook(NB_DIR / "q2d_q1_vs_q2.ipynb", q2d)
    print("wrote q2{a,b,c,d}*.ipynb")


if __name__ == "__main__":
    main()
