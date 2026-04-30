# Q1 — Two-Layer MLP on 20 Newsgroups: Design

**Date:** 2026-04-30
**Branch:** `ft-qn-1`
**Spec author:** Dan + Claude (brainstormed in this session)
**Scope:** Task 1 (a–d) of the NALAPRO project. Q2/Q3/Q4 are out of scope but inform reusable boundaries.

## 1. Goal

Implement Task 1 from `project_description/NALAPRO Project.pdf`: a two-layer MLP (Linear → ReLU → Linear) trained on the 20 Newsgroups classification task with three input representations plus a bonus experiment.

The spec calls for four sub-deliverables:

- **1a.** Data preprocessing.
- **1b.** Train word2vec embeddings; feed mean-pooled document vectors to the MLP. Visualize and compare embedding space after **1 epoch** vs **20 epochs** of word2vec training.
- **1c.** Use TF-IDF as the document vectorizer instead of word2vec. Compare against 1b.
- **1d.** One additional experiment that *could improve results* with the network unchanged. We will use **mean-pool concatenated with max-pool** of the same word2vec vectors.

The MLP architecture is fixed across all four sub-tasks. Only the input representation changes.

## 2. Non-goals

- Hyperparameter sweeps. We pick reasonable defaults, log them, and discuss in the report.
- Implementing word2vec from scratch. We use `gensim`.
- Cross-validation. The spec doesn't ask for it; a single train/val/test split is enough for the report's discussion.
- Anything Q2/Q3/Q4 needs (BERT, MLM, Llama). The Q1 modules are designed so the loader, training loop, eval, and W&B plumbing carry over; the BERT/Llama-specific code lands on later branches.

## 3. Hard constraints from the assignment

These come from `project_description/NALAPRO Project.pdf` and must be preserved:

- The dataset is **never committed**. It is fetched at runtime via `sklearn.datasets.fetch_20newsgroups`.
- All experiments are tracked in **Weights & Biases**. The W&B run/project URL goes in the top-of-file docstring of every entry-point script and in the report header.
- Any third-party code adapted into the repo is **cited** in the source file and in the report.
- AI tooling (Claude Code) is **disclosed** in the report's "tools used" section.
- The report contains **no code** — only results, plots, and discussion. Plots therefore must be saved as static image files (`figures/`) in addition to being logged to W&B.

## 4. Architecture

### 4.1 Repo layout

```
src/nlp_project/
    __init__.py
    data.py              # 20NG loader + preprocessing + train/val split
    embeddings.py        # word2vec training, mean-pool, mean+max-pool
    vectorizers.py       # TF-IDF wrapper
    model.py             # the two-layer MLP
    train.py             # generic training loop + W&B logging
    eval.py              # metrics + confusion matrix plotting
    viz.py               # t-SNE plots for embeddings
notebooks/
    q1a_preprocessing.ipynb
    q1b_word2vec.ipynb
    q1c_tfidf.ipynb
    q1d_mean_max_pool.ipynb
figures/                 # saved plots (only "final" plots tracked in git)
models/                  # saved word2vec checkpoints (.gitignored)
```

The `src/` package holds reusable logic. Notebooks are thin: they configure a run, call into `src/`, and render plots inline for the report. This same package backs Q2–Q4 with new modules added (e.g. `bert.py`) on later branches.

### 4.2 Module responsibilities

| Module | Public surface | Notes |
| --- | --- | --- |
| `data.py` | `load_20ng(remove=True) -> (train_docs, train_labels, test_docs, test_labels, label_names)`; `preprocess(docs, drop_stopwords=True) -> list[list[str]]`; `train_val_split(docs, labels, val_frac=0.1, seed=42)` | `preprocess` returns tokenized documents (list of lists). `drop_stopwords` is a flag because word2vec sometimes benefits from keeping them — see §8. Stopword list is loaded once at module import. |
| `embeddings.py` | `train_word2vec(token_lists, epochs, vector_size=100, window=5, min_count=5, sg=1, seed=42) -> Word2Vec`; `mean_pool(token_lists, w2v) -> np.ndarray`; `mean_max_pool(token_lists, w2v) -> np.ndarray` | Both pooling functions skip OOV tokens; documents with zero in-vocab tokens get a zero vector and a warning logged. |
| `vectorizers.py` | `fit_tfidf(train_docs, **kwargs) -> (TfidfVectorizer, X_train)`; `transform_tfidf(vec, docs) -> X` | Wraps `sklearn.feature_extraction.text.TfidfVectorizer` with our default settings. Input is a list of *strings* (post-preprocessing, joined back). |
| `model.py` | `class MLP(nn.Module)` with `__init__(in_dim, hidden_dim=256, num_classes=20, dropout=0.3)` | Linear → ReLU → Dropout → Linear. Dropout is included; the spec says "ReLU in between" and is silent on dropout, and without it the model overfits on TF-IDF inputs. |
| `train.py` | `train(model, train_loader, val_loader, *, epochs, lr, device, wandb_run, patience=5) -> dict[str, list]` | Adam optimizer, cross-entropy, early stopping on val loss. Returns history dict for plotting. Logs per-epoch metrics to W&B. |
| `eval.py` | `evaluate(model, loader, label_names, device) -> dict`; `plot_confusion(cm, label_names, save_path)` | Returns accuracy, macro-F1, per-class F1, confusion matrix. Saves PNG. |
| `viz.py` | `plot_word_neighborhood(w2v_models: dict[str, Word2Vec], words: list[str], save_path)` | Runs t-SNE on the union of vectors from each model, plots side-by-side subplots. Used to compare 1-epoch vs 20-epoch checkpoints. |

### 4.3 Data flow per sub-task

All four sub-tasks share the same skeleton:

```
raw_docs --preprocess--> tokens --vectorize--> X (n_docs × in_dim)
                                                   |
                                                   v
                              train(MLP, X_train, y_train, X_val, y_val)
                                                   |
                                                   v
                                          evaluate(MLP, X_test, y_test)
                                                   |
                                                   v
                            log to W&B + save metrics/plots to figures/
```

Only the `vectorize` step changes between sub-tasks:

| Sub-task | Vectorizer | `in_dim` |
| --- | --- | --- |
| 1b | `mean_pool` of word2vec | 100 |
| 1c | TF-IDF (top-N features) | configurable, default 20,000 |
| 1d | `mean_max_pool` of word2vec | 200 |

1a is "preprocessing only" — the notebook produces the cleaned token lists, sanity-checks vocab size, and saves a few descriptive plots (doc-length histogram, class balance) that the report will reference.

## 5. Concrete defaults (single source of truth)

These values are what the code defaults to. Any deviation lives in a notebook cell and gets logged to W&B.

**Preprocessing**

- `fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))`
- Tokenization: `gensim.utils.simple_preprocess` (lowercase, strip punctuation, drop tokens shorter than 3 characters).
- Stopwords: NLTK English stopword list (downloaded once via a setup script, not at import time).
- No stemming or lemmatization.

**Splits**

- Train (11,314) and test (7,532) splits as returned by sklearn.
- Carve **10% validation** from train, stratified by label, `random_state=42`.

**Word2vec**

- `gensim.models.Word2Vec(vector_size=100, window=5, min_count=5, sg=1, workers=1, seed=42)`. **`workers=1` is intentional**: gensim's multi-threaded training is non-deterministic even with a fixed seed, and we need byte-identical 1-epoch vs 20-epoch checkpoints for the comparison to be honest. (`workers=4` is fine for one-off exploration but the report's plots are generated with `workers=1`.)
- Train two checkpoints **from the same seeded init**: call `Word2Vec(...)` once with `epochs=1`, save to `models/w2v_epoch1.kv`; call again with `epochs=20`, save to `models/w2v_epoch20.kv`. Same seed, same corpus, same hyperparameters — only `epochs` changes.
- The 20-epoch model is the one fed into the MLP for 1b and 1d. The 1-epoch model exists only for the visual comparison required by the spec.

**TF-IDF**

- `TfidfVectorizer(max_features=20_000, ngram_range=(1,1), min_df=2, sublinear_tf=True)`.

**MLP**

- `hidden_dim=256`, `dropout=0.3`, `num_classes=20`.
- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-5`.
- Batch size 64, max 50 epochs, early stopping on val loss with `patience=5`.

**Visualization**

- t-SNE: `sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=42)`.
- Plot the **500 most frequent in-vocab tokens** that appear in *both* the 1-epoch and 20-epoch models. Side-by-side subplots so the report can show change visually.
- W&B Embedding Projector also receives the full vocab from both checkpoints.

**Metrics**

- Accuracy, macro-F1, per-class F1, confusion matrix.
- Reported on the held-out test set, single number per sub-task. Discussed across sub-tasks in the report.

## 6. Reproducibility

- One `seed=42` constant in `src/nlp_project/__init__.py`. Every randomized step (split, word2vec, torch, numpy) reads from it.
- `set_seed()` helper called at the top of each notebook.
- `uv.lock` committed once dependencies stabilize.

## 7. Report deliverables (Q1 only)

The notebooks should produce, and `figures/` should hold:

- `class_balance.png`, `doc_length_hist.png` — from 1a.
- `w2v_tsne_epoch1_vs_epoch20.png` — side-by-side t-SNE.
- `confusion_matrix_q1b.png`, `confusion_matrix_q1c.png`, `confusion_matrix_q1d.png`.
- `metric_comparison_table.csv` — accuracy + macro-F1 for 1b/1c/1d.
- A short markdown blurb per sub-task in the corresponding notebook that the report can quote.

## 8. Risks and pre-decided mitigations

- **Stopword removal too aggressive for word2vec.** Word2vec benefits from co-occurrence with stopwords; removing them weakens the embedding. *Mitigation:* `data.preprocess(docs, drop_stopwords=...)` takes the flag. Default `True` for the inputs we feed to TF-IDF and the MLP. For word2vec training the notebook calls `preprocess(..., drop_stopwords=False)` to keep the embedding signal — this is explicit, not magic.
- **t-SNE non-determinism across runs.** Even with a seed, t-SNE depends on initialization. *Mitigation:* save the resulting 2D coordinates as `.npy` alongside the PNG so plots can be regenerated.
- **W&B downtime around the deadline.** *Mitigation:* `train.py` always writes a local CSV of per-epoch metrics in addition to logging to W&B.
- **Class imbalance in 20NG is mild but not zero.** *Mitigation:* report macro-F1 alongside accuracy; the confusion matrix surfaces the rest.

## 9. Out of scope (explicitly)

- BERT / transformers (Q2, Q3).
- Llama-3 / few-shot prompting (Q4).
- Hyperparameter sweeps.
- Cross-validation.
- Building a CLI; notebooks are the entry points for Q1.
