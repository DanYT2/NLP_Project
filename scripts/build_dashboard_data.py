"""Consolidate every experiment metric the presentation needs into one JSON file.

Reads:
- figures/metric_comparison_table.csv (Q1 fallback scalars)
- notebooks/q*.ipynb cell outputs (Q1a stats, Q2/Q3 scalars + Q1+Q2 comparison table)
- models/q4_results/*.json, models/qbonus/*.json (verbatim copy-through)

Writes: presentation/data/dashboard.json
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
FIG_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
OUT_PATH = ROOT / "presentation" / "data" / "dashboard.json"

LABELS_20NG = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
]


def nb_text(name: str) -> str:
    """Concatenate all stdout / text/plain outputs from a notebook into one blob."""
    nb = json.loads((NB_DIR / name).read_text())
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            t = out.get("text") or out.get("data", {}).get("text/plain")
            if t:
                chunks.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(chunks)


def find_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def find_int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Q1a — preprocessing / dataset stats
# --------------------------------------------------------------------------- #
def extract_dataset() -> dict[str, Any]:
    txt = nb_text("q1a_preprocessing.ipynb")
    return {
        "train_docs": find_int(txt, r"train docs:\s*(\d+)"),
        "test_docs": find_int(txt, r"test docs:\s*(\d+)"),
        "n_classes": 20,
        "vocab_raw": find_int(txt, r"vocab size:\s*(\d+)"),
        "doc_len": {
            "median": find_float(txt, r"median:\s*([\d.]+)"),
            "mean": find_float(txt, r"mean:\s*([\d.]+)"),
            "p95": find_float(txt, r"p95:\s*([\d.]+)"),
            "max": find_float(txt, r"max:\s*([\d.]+)"),
        },
        "labels": LABELS_20NG,
    }


# --------------------------------------------------------------------------- #
# Q1 — scalars (preferred source: q2d comparison table, fallback: CSV)
# --------------------------------------------------------------------------- #
def extract_q1_and_q2() -> tuple[dict[str, Any], dict[str, Any]]:
    txt = nb_text("q2d_q1_vs_q2.ipynb")
    # Lines look like:
    #   "0      Q1b word2vec mean-pool  0.632100  0.609800"
    row_re = re.compile(
        r"^\s*\d+\s+(.+?)\s+([01]\.\d{4,})\s+([01]\.\d{4,})\s*$", re.MULTILINE
    )
    rows = {m.group(1).strip(): (float(m.group(2)), float(m.group(3)))
            for m in row_re.finditer(txt)}

    def pick(label: str) -> dict[str, float | None]:
        if label in rows:
            acc, f1 = rows[label]
            return {"accuracy": acc, "macro_f1": f1}
        return {"accuracy": None, "macro_f1": None}

    # CSV fallback for Q1 (in case q2d wasn't executed)
    csv_path = FIG_DIR / "metric_comparison_table.csv"
    csv_rows: dict[str, tuple[float, float]] = {}
    if csv_path.exists():
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                csv_rows[r["experiment"]] = (float(r["accuracy"]), float(r["macro_f1"]))

    def csv_fallback(key: str) -> dict[str, float | None]:
        if key in csv_rows:
            acc, f1 = csv_rows[key]
            return {"accuracy": acc, "macro_f1": f1}
        return {"accuracy": None, "macro_f1": None}

    q1 = {
        "q1b": pick("Q1b word2vec mean-pool") if pick("Q1b word2vec mean-pool")["accuracy"]
              else csv_fallback("Q1b — word2vec mean-pool"),
        "q1c": pick("Q1c TF-IDF") if pick("Q1c TF-IDF")["accuracy"]
              else csv_fallback("Q1c — TF-IDF"),
        "q1d": pick("Q1d word2vec mean+max-pool") if pick("Q1d word2vec mean+max-pool")["accuracy"]
              else csv_fallback("Q1d — word2vec mean+max-pool"),
    }
    # word2vec ablation: 1-epoch vs 20-epoch — vocab stays the same, the embeddings
    # don't. We surface vocab numbers as evidence that training matters even when
    # vocab does not.
    q1b_txt = nb_text("q1b_word2vec.ipynb")
    q1["q1b_meta"] = {
        "vocab_epoch1": find_int(q1b_txt, r"vocab\(epoch=1\):\s*(\d+)"),
        "vocab_epoch20": find_int(q1b_txt, r"vocab\(epoch=20\):\s*(\d+)"),
    }

    # Q2b headline pulled from its own notebook for the "best val macro-F1" extra
    q2b_txt = nb_text("q2b_bert_baseline.ipynb")
    q2 = {
        "q2b": {
            **pick("Q2b BERT baseline"),
            "best_val_macro_f1": find_float(q2b_txt, r"best val macro-?F1:\s*([01]\.\d+)"),
        },
        "q2c_best": pick("Q2c best (q2c-lr-5e-05)"),
    }
    q2c_txt = nb_text("q2c_bert_experiments.ipynb")
    q2["q2c_best"]["best_lr"] = (re.search(r"best LR:\s*(\S+)", q2c_txt) or [None, None])[1]
    return q1, q2


# --------------------------------------------------------------------------- #
# Q3 — MLM-then-finetune
# --------------------------------------------------------------------------- #
def extract_q3() -> dict[str, Any]:
    txt = nb_text("q3_mlm_then_finetune.ipynb")
    return {
        "mlm": {
            "best_eval_loss": find_float(txt, r"best eval_loss\s*=\s*([\d.]+)"),
            "perplexity": find_float(txt, r"perplexity\s*=\s*([\d.]+)"),
            "mask_probability": 0.15,
            "mlm_val_frac": 0.1,
        },
        "classification": {
            "accuracy": find_float(txt, r"accuracy\s*=\s*([01]\.\d+)"),
            "macro_f1": find_float(txt, r"macro_f1\s*=\s*([01]\.\d+)"),
        },
    }


# --------------------------------------------------------------------------- #
# Q4 + Q-bonus — copy JSON through
# --------------------------------------------------------------------------- #
def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def extract_q4() -> dict[str, Any]:
    d = MODELS_DIR / "q4_results"
    return {
        "zero_shot": load_json(d / "q4_zero_shot.json"),
        "few_shot_1pc": load_json(d / "q4_few_shot_1pc.json"),
        "few_shot_3pc": load_json(d / "q4_few_shot_3pc.json"),
    }


def extract_qbonus() -> dict[str, Any]:
    d = MODELS_DIR / "qbonus"
    return {
        "sweep": load_json(d / "qbonus_sweep_summary.json"),
        "full_test": load_json(d / "qbonus_qlora_full_test.json"),
        "q4_subset": load_json(d / "qbonus_qlora_q4subset.json"),
    }


# --------------------------------------------------------------------------- #
# Cross-question comparison block (the climax chart)
# --------------------------------------------------------------------------- #
def build_comparison(q1: dict, q2: dict, q3: dict, q4: dict, qb: dict) -> list[dict]:
    """One row per experiment, ordered for the cross-question chart."""
    return [
        {"key": "q1b", "label": "Q1b · word2vec",   "group": "Q1", **q1["q1b"]},
        {"key": "q1c", "label": "Q1c · TF-IDF",     "group": "Q1", **q1["q1c"]},
        {"key": "q1d", "label": "Q1d · mean+max",   "group": "Q1", **q1["q1d"]},
        {"key": "q2b", "label": "Q2b · BERT base",  "group": "Q2",
            "accuracy": q2["q2b"]["accuracy"], "macro_f1": q2["q2b"]["macro_f1"]},
        {"key": "q2c", "label": "Q2c · BERT tuned", "group": "Q2", **q2["q2c_best"]},
        {"key": "q3",  "label": "Q3 · MLM + FT",    "group": "Q3",
            "accuracy": q3["classification"]["accuracy"],
            "macro_f1": q3["classification"]["macro_f1"]},
        {"key": "q4_zero",  "label": "Q4 · zero-shot", "group": "Q4",
            "accuracy": q4["zero_shot"]["accuracy"],
            "macro_f1": q4["zero_shot"]["macro_f1"],
            "note": "200-doc subset"},
        {"key": "q4_k1",    "label": "Q4 · k=1/class", "group": "Q4",
            "accuracy": q4["few_shot_1pc"]["accuracy"],
            "macro_f1": q4["few_shot_1pc"]["macro_f1"],
            "note": "200-doc subset"},
        {"key": "q4_k3",    "label": "Q4 · k=3/class", "group": "Q4",
            "accuracy": q4["few_shot_3pc"]["accuracy"],
            "macro_f1": q4["few_shot_3pc"]["macro_f1"],
            "note": "200-doc subset"},
        {"key": "qb_full",  "label": "Q-bonus · QLoRA (full)", "group": "Qb",
            "accuracy": qb["full_test"]["accuracy"],
            "macro_f1": qb["full_test"]["macro_f1"]},
        {"key": "qb_sub",   "label": "Q-bonus · QLoRA (Q4 subset)", "group": "Qb",
            "accuracy": qb["q4_subset"]["accuracy"],
            "macro_f1": qb["q4_subset"]["macro_f1"],
            "note": "200-doc subset"},
    ]


# --------------------------------------------------------------------------- #
def main() -> None:
    dataset = extract_dataset()
    q1, q2 = extract_q1_and_q2()
    q3 = extract_q3()
    q4 = extract_q4()
    qb = extract_qbonus()
    comparison = build_comparison(q1, q2, q3, q4, qb)

    out = {
        "dataset": dataset,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "qbonus": qb,
        "comparison": comparison,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Loud confirmation that the numbers actually got picked up
    missing = [c["key"] for c in comparison if c.get("accuracy") is None]
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  comparison rows: {len(comparison)}")
    if missing:
        print(f"  WARNING: missing accuracy for: {missing}")
    else:
        print("  all comparison rows populated")
    print(f"  Q3 perplexity: {q3['mlm']['perplexity']}, acc: {q3['classification']['accuracy']}")
    print(f"  Q-bonus full: acc={qb['full_test']['accuracy']:.4f}, "
          f"subset: acc={qb['q4_subset']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
