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

- Python **3.13** (pinned in `.python-version`). Managed with `uv`; lock file is `uv.lock` (committed).
- Add a dependency: `uv add <pkg>` (or `uv add --dev <pkg>` for test-only).
- Run the test suite: `uv run pytest` (37 tests, ~7s incl. the slow data-fetch ones).
  - Single file / test: `uv run pytest tests/test_train.py` or `uv run pytest tests/test_train.py::test_name`.
  - Skip the network/training-heavy ones: `uv run pytest -m "not slow"`. Run only the slow ones: `-m slow`.
  - Parallelize: `uv run pytest -n auto` (pytest-xdist is in the dev group).
- Open the notebooks: `uv run jupyter lab`.
- One-time setup before first notebook run: `uv run python scripts/setup_nltk.py` (downloads NLTK English stopwords).

## Repo layout (Q1)

```
src/nlp_project/   # reusable package: data, embeddings, vectorizers, model, train, eval, viz
tests/             # pytest suite (slow tests gated by -m slow)
scripts/           # one-time setup scripts (e.g. NLTK download)
notebooks/         # q1a–q1d notebooks; thin wrappers around src/nlp_project
figures/           # static plots referenced by the report (gitkeep'd; outputs gitignored)
models/            # word2vec checkpoints (gitignored)
docs/superpowers/  # spec and implementation plan for Q1
```

## Architecture notes

A few things span multiple files and are easier to internalize up front than to rediscover:

- **Package vs. notebook split.** All non-trivial logic lives in `src/nlp_project/`. Notebooks are thin orchestrators — they wire functions together, log to W&B, and save figures. New logic goes in the package with a test; do **not** grow logic inside `.ipynb` cells.
- **Determinism is centralized.** `nlp_project.set_seed()` (in `src/nlp_project/__init__.py`) is the single entry point that seeds Python `random`, NumPy, `PYTHONHASHSEED`, and PyTorch from `SEED = 42`. `tests/conftest.py` runs it via an `autouse` fixture, so every test starts from the same RNG state. Word2vec has its own `seed=` kwarg (passed explicitly in `embeddings.train_word2vec`) and is pinned to `workers=1` because gensim's multi-threaded training is non-deterministic — required for the 1-epoch vs many-epoch comparison to be meaningful.
- **Two non-obvious preprocessing gotchas** baked into `data.py` / `embeddings.py`:
  1. `load_20ng(remove=True)` is the default and the *correct* setting — leaving headers/footers/quotes in leaks the label (spec §3). Don't flip this without a good reason.
  2. `preprocess(..., drop_stopwords=False)` for word2vec input; `drop_stopwords=True` for TF-IDF / classifier input. Word2vec learns better when frequent function words remain in the context window.
- **Training loop is intentionally hand-rolled** (`train.py`) — no Lightning/Trainer. It does early stopping on val loss with `patience=5` and restores best-epoch weights before returning. `wandb_run` is optional so tests can call `train(...)` without a W&B session.

## Q1 status

Foundation package + tests are in place on `ft-qn-1`. Notebooks are drafted but not executed — they need the user's W&B credentials and a few minutes of CPU to run. To run them:

1. `wandb login` (one-time, uses your W&B account).
2. `uv run jupyter lab` and execute Q1a → Q1b → Q1c → Q1d in order.
3. Q1d's "comparison table" cell needs the accuracy/F1 numbers from Q1b and Q1c pasted in before re-running.
4. Paste the resulting W&B project URL into the line below.

- W&B project for Q1 runs: <https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro?nw=nwuserdanwwaititu>

## Q2 status

`ft-qn-2` branch. Foundation package + tests are in place; notebooks scaffolded but not executed (W&B credentials + MPS compute required). Q2 uses HuggingFace `Trainer` (not the hand-rolled Q1 loop) — see `docs/superpowers/plans/2026-05-15-q2-bert-finetune.md` for the design rationale.

- New modules: `src/nlp_project/bert_data.py` (tokenization + tri-split builder; reuses `data.train_val_split(seed=42)` for Q1↔Q2 val-index parity) and `src/nlp_project/bert_train.py` (HF Trainer wrapper with MPS-safe flags, encoder-freeze for the linear probe).
- New tests: `tests/test_bert_data.py`, `tests/test_bert_train.py` (full suite at 55+ green).
- `eval.py` refactored: `metrics_from_predictions(y_true, y_pred, label_names)` is now the single source of truth for accuracy/macro-F1/per-class F1/CM; both Q1 and Q2 use it.

To run the Q2 notebooks:

1. `wandb login` (one-time, same project as Q1).
2. `uv run jupyter lab` and execute Q2a → Q2b → Q2c → Q2d in order.
3. Q2a picks `max_length` from the token-length histogram (default 256).
4. Q2c writes its sweep to `models/q2_results/q2c_sweep.json`; Q2d reads it.
5. If notebooks need to be regenerated from `scripts/build_q2_notebooks.py`, do that *before* re-executing — running the script overwrites the executed `.ipynb`s.

- W&B project for Q2 runs: same as Q1 (group `q2`) — <https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro?nw=nwuserdanwwaititu>
