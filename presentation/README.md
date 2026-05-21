# Presentation Dashboard

Interactive Streamlit dashboard that drives the 20-minute HSLU NALAPRO
final presentation. Sidebar navigates 12 sections covering the problem,
dataset, every experiment (Q1 → Q-bonus), the cross-experiment comparison,
and the appendix (W&B link, citations, AI disclosure, reproducibility).

## Run it

From the repo root:

```bash
uv run streamlit run presentation/app.py
```

Streamlit opens at `http://localhost:8501`. Use the sidebar to jump
between sections — every page is self-contained, so during Q&A you can
go straight to any experiment or any methodology rationale.

## File layout

```
presentation/
  app.py            # Streamlit entry point, 12 sidebar pages
  data_loader.py    # Pure helpers — load CSVs, JSONs, resolve figure paths
  content.py        # Rationale text + long-form prose (kept out of the UI)
  README.md         # This file
```

## Data sources

The dashboard is **read-only** over artifacts produced by the experiment
notebooks — it does not retrain anything.

| Source | Used for |
|---|---|
| `figures/metric_comparison_table.csv` | Q1b/c/d headline metrics |
| `figures/*.png` (23 files) | Confusion matrices, comparison bars, t-SNE, loss curves |
| `models/q4_results/q4_*.json` | Q4 zero/few-shot interactive panels |
| `models/qbonus/qbonus_*.json` | Q-bonus QLoRA sweep + final evals |

The W&B project URL is hard-coded in `content.py` and surfaced on every
page via the sidebar link and in the appendix.

## What's interactive

- **Q1 page** — dropdown switches confusion-matrix variant
- **Q4 page** — dropdown switches condition (zero / k=1 / k=3 per class), redraws CM + per-class F1 bars from JSON
- **Q-bonus page** — sweep summary table, interactive CM heatmap, side-by-side comparisons
- **Cross-experiment page** — toggle accuracy/macro-F1, toggle sort direction; live Plotly bars across all experiments
- **Every experiment page** — "🧠 Methodology decisions" panel with collapsible expanders for each design choice (what + why + source)
