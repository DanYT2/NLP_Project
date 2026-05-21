"""Pure data-loading helpers for the presentation dashboard.

No Streamlit imports — these can be unit-tested in isolation and reused.
All paths resolve relative to the repo root, regardless of CWD.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical 20 Newsgroups class order (sklearn default — matches every JSON dump).
CLASS_NAMES: list[str] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def figure_path(name: str) -> str:
    """Resolve a filename under figures/ to an absolute path string."""
    return str(REPO_ROOT / "figures" / name)


def _load_json(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text())


@lru_cache(maxsize=1)
def load_q1_metrics() -> pd.DataFrame:
    """Return Q1b/c/d headline metrics from the CSV comparison table."""
    df = pd.read_csv(REPO_ROOT / "figures" / "metric_comparison_table.csv")
    df["accuracy"] = df["accuracy"].astype(float).round(4)
    df["macro_f1"] = df["macro_f1"].astype(float).round(4)
    return df


@lru_cache(maxsize=1)
def load_q4_metrics() -> dict[str, dict]:
    """Return Q4 results keyed by condition: 'zero', '1pc', '3pc'."""
    return {
        "zero": _load_json("models/q4_results/q4_zero_shot.json"),
        "1pc": _load_json("models/q4_results/q4_few_shot_1pc.json"),
        "3pc": _load_json("models/q4_results/q4_few_shot_3pc.json"),
    }


@lru_cache(maxsize=1)
def load_qbonus_metrics() -> dict[str, dict]:
    """Return Q-bonus eval results keyed by split: 'full_test', 'q4subset'."""
    return {
        "full_test": _load_json("models/qbonus/qbonus_qlora_full_test.json"),
        "q4subset": _load_json("models/qbonus/qbonus_qlora_q4subset.json"),
    }


@lru_cache(maxsize=1)
def load_qbonus_sweep() -> dict:
    """Return the LoRA hyperparameter sweep summary."""
    return _load_json("models/qbonus/qbonus_sweep_summary.json")


@lru_cache(maxsize=1)
def cross_experiment_table() -> pd.DataFrame:
    """Merge Q1 CSV + Q4 + Q-bonus into one long-format dataframe.

    Q2 and Q3 don't have JSON dumps in this branch, so they're absent —
    Q2b acc ≈ 0.81 and Q3 acc ≈ 0.82 will be visible in the page-6/7 PNGs
    rather than this comparison chart.
    """
    rows: list[dict] = []

    q1 = load_q1_metrics()
    for _, r in q1.iterrows():
        rows.append(
            {
                "experiment": r["experiment"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "source": "Q1 (MLP)",
            }
        )

    q4 = load_q4_metrics()
    q4_label = {
        "zero": "Q4 — Llama zero-shot",
        "1pc": "Q4 — Llama few-shot k=1/class",
        "3pc": "Q4 — Llama few-shot k=3/class",
    }
    for k, d in q4.items():
        rows.append(
            {
                "experiment": q4_label[k],
                "accuracy": round(d["accuracy"], 4),
                "macro_f1": round(d["macro_f1"], 4),
                "source": "Q4 (Llama-3 frozen)",
            }
        )

    qb = load_qbonus_metrics()
    qb_label = {
        "full_test": "Q-bonus — QLoRA (full 7,532 test)",
        "q4subset": "Q-bonus — QLoRA (Q4 200-doc subset)",
    }
    for k, d in qb.items():
        rows.append(
            {
                "experiment": qb_label[k],
                "accuracy": round(d["accuracy"], 4),
                "macro_f1": round(d["macro_f1"], 4),
                "source": "Q-bonus (QLoRA)",
            }
        )

    return pd.DataFrame(rows)
