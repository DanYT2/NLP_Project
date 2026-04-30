# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

HSLU NLP module ("NALAPRO") graded project. Single-student repo — see `project_description/NALAPRO Project.pdf` for the authoritative spec. Submission deadline: **EOD 2026-06-06**. Grading: 30% code, 30% report, 40% presentation.

The project is a sequence of four experiments on the **20 Newsgroups** dataset (`sklearn.datasets.fetch_20newsgroups`):

1. **Q1** — Two-layer MLP (Linear → ReLU → Linear) trained with three input representations and one extra experiment:
   - (a) preprocessing
   - (b) word2vec embeddings (compare 1-epoch vs many-epoch embedding spaces)
   - (c) TF-IDF inputs, compared against (b)
   - (d) one self-designed experiment that improves results without changing the network
2. **Q2** — Fine-tune `bert-base` on the classification task; compare to Q1.
3. **Q3** — First do MLM pretraining on the corpus, then fine-tune for classification; compare to Q2.
4. **Q4** — Zero-shot and few-shot classification with Llama-3.

Each question lives on its own branch, named `ft-qn-<n>` (current: `ft-qn-1`). Keep the four experiments isolated so they can be presented and compared independently.

## Hard constraints from the spec

These come from the assignment PDF and must be respected when generating code or docs:

- **Do not commit the dataset.** It is fetched at runtime via `fetch_20newsgroups`.
- **Track experiments with Weights & Biases or MLflow.** The final report and the code's top-of-file docstring must include the W&B/MLflow link.
- **Cite any third-party code** copied or adapted into the repo.
- **Disclose AI tool usage** (including Claude Code) in the report's "tools used" section.
- The report is a separate scientific write-up — no code in it. Plots/graphs and cross-experiment comparison are required.

## Tooling

- Python **3.13** (pinned in `.python-version`). Project metadata is in `pyproject.toml`; no `[build-system]` and no lockfile yet → managed with `uv`.
- Run the entry point: `uv run python main.py`
- Add a dependency: `uv add <pkg>` (this creates `uv.lock` on first use; commit it).
- No tests, linter, or formatter are configured yet. If you add one, wire it through `uv` (e.g. `uv run pytest`, `uv run ruff check`) and update this file.

## Repo layout

The repo is currently a stub (`main.py` is a placeholder). Real structure will grow per question — keep notebooks, training scripts, and saved artifacts scoped to the relevant question's branch/folder rather than scattering them at the root.
