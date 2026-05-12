# Q1 — Two-Layer MLP on 20 Newsgroups: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Python package and four notebooks that solve Task 1 (a–d) of the NALAPRO assignment: a two-layer MLP trained on 20 Newsgroups with word2vec, TF-IDF, and a mean+max-pool variant.

**Architecture:** A small `src/nlp_project/` package holds all reusable logic (data, embeddings, vectorizers, model, training, eval, viz). Notebooks are thin wrappers that configure a run, call into the package, and render plots for the report. All experiments log to Weights & Biases. Same MLP is reused across the three input representations; only the vectorizer changes.

**Tech Stack:** Python 3.13, `uv` package manager, PyTorch, gensim, scikit-learn, NLTK, Weights & Biases, pytest, matplotlib, seaborn, Jupyter.

**Spec:** `docs/superpowers/specs/2026-04-30-q1-mlp-on-20newsgroups-design.md`

---

## File Structure

**Created:**

```
src/nlp_project/
    __init__.py          # SEED constant, set_seed() helper
    data.py              # load_20ng, preprocess, train_val_split
    embeddings.py        # train_word2vec, mean_pool, mean_max_pool
    vectorizers.py       # fit_tfidf, transform_tfidf
    model.py             # MLP class
    train.py             # train() loop, EarlyStopping
    eval.py              # evaluate, plot_confusion
    viz.py               # plot_word_neighborhood (t-SNE)
tests/
    __init__.py
    test_data.py
    test_embeddings.py
    test_vectorizers.py
    test_model.py
    test_train.py
    test_eval.py
    test_viz.py
    conftest.py
notebooks/
    q1a_preprocessing.ipynb
    q1b_word2vec.ipynb
    q1c_tfidf.ipynb
    q1d_mean_max_pool.ipynb
scripts/
    setup_nltk.py        # one-time NLTK stopword download
figures/                 # plot outputs (only "final" plots tracked)
models/                  # word2vec checkpoints (.gitignored)
```

**Modified:**

- `pyproject.toml` — add dependencies and pytest config
- `.gitignore` — add `models/`, `figures/*` (with `!figures/.gitkeep`), `wandb/`, `mlruns/`

**Removed:** `main.py` (placeholder, replaced by notebooks).

---

## Task 1: Project setup — dependencies, gitignore, package skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Delete: `main.py`
- Create: `src/nlp_project/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `figures/.gitkeep`

- [ ] **Step 1: Add runtime and dev dependencies via uv**

Run:
```bash
uv add numpy scikit-learn gensim "torch>=2.2" matplotlib seaborn nltk wandb pandas jupyter ipykernel
uv add --dev pytest pytest-xdist
```

Expected: `uv.lock` is created (or updated) and `pyproject.toml` now lists these under `[project] dependencies` and `[dependency-groups] dev`.

- [ ] **Step 2: Replace `pyproject.toml` description and add pytest config**

After uv has populated dependencies, edit `pyproject.toml` to set a real description and add a pytest config block. The project block should look like:

```toml
[project]
name = "nlp-project"
version = "0.1.0"
description = "HSLU NALAPRO project — Q1: two-layer MLP on 20 Newsgroups."
readme = "README.md"
requires-python = ">=3.13"
# dependencies populated by uv add ...
```

Append at the end of the file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
    "slow: tests that fetch data over the network or train for >1s",
]

[tool.hatch.build.targets.wheel]
packages = ["src/nlp_project"]
```

- [ ] **Step 3: Update `.gitignore`**

Append the following to `.gitignore`:

```
# Project-specific
models/
wandb/
mlruns/
figures/*
!figures/.gitkeep
*.npy
.ipynb_checkpoints/
```

- [ ] **Step 4: Delete the placeholder `main.py`**

Run: `rm main.py`

- [ ] **Step 5: Create the package skeleton**

Create `src/nlp_project/__init__.py`:

```python
"""HSLU NALAPRO Q1 — two-layer MLP on 20 Newsgroups.

Top-level package. All randomness in the project is seeded from SEED.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42
"""Single source of truth for randomness across the project."""


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) from one constant.

    Call this at the top of every notebook and every test that depends on
    deterministic behaviour. Word2vec uses a separate `seed=` kwarg on the
    gensim model itself — we set it explicitly there.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is a hard dep, but we don't want this helper to fail
        # if it's ever called from a stripped-down environment.
        pass


__all__ = ["SEED", "set_seed"]
```

Create `tests/__init__.py` as an empty file.

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from nlp_project import set_seed


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Reset all RNGs before every test for determinism."""
    set_seed()
```

Create `figures/.gitkeep` as an empty file.

- [ ] **Step 6: Smoke-test the package imports**

Run: `uv run python -c "from nlp_project import SEED, set_seed; set_seed(); print(SEED)"`

Expected output: `42`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/ tests/ figures/.gitkeep
git rm main.py
git commit -m "chore: scaffold nlp_project package and pytest harness"
```

---

## Task 2: `set_seed` determinism test

**Files:**
- Test: `tests/test_seed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_seed.py`:

```python
"""Verify set_seed makes Python, NumPy, and torch RNGs reproducible."""

from __future__ import annotations

import random

import numpy as np
import torch

from nlp_project import set_seed


def test_set_seed_makes_python_random_deterministic() -> None:
    set_seed()
    a = [random.random() for _ in range(5)]
    set_seed()
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic() -> None:
    set_seed()
    a = np.random.rand(5)
    set_seed()
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_makes_torch_deterministic() -> None:
    set_seed()
    a = torch.rand(5)
    set_seed()
    b = torch.rand(5)
    torch.testing.assert_close(a, b)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_seed.py
git commit -m "test: lock in set_seed determinism"
```

---

## Task 3: `data.load_20ng` — fetch the dataset

**Files:**
- Create: `src/nlp_project/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
"""Tests for src/nlp_project/data.py.

Network-touching tests are marked `slow` so the inner-loop unit suite stays fast.
Run them explicitly with: uv run pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from nlp_project import data


@pytest.mark.slow
def test_load_20ng_returns_expected_shapes() -> None:
    train_docs, train_labels, test_docs, test_labels, label_names = data.load_20ng()
    # The sklearn 20newsgroups train/test split sizes are public and stable.
    assert len(train_docs) == 11_314
    assert len(test_docs) == 7_532
    assert train_labels.shape == (11_314,)
    assert test_labels.shape == (7_532,)
    assert len(label_names) == 20
    # Labels are integer class indices in [0, 20).
    assert train_labels.dtype.kind in "iu"
    assert int(train_labels.min()) == 0
    assert int(train_labels.max()) == 19


@pytest.mark.slow
def test_load_20ng_strips_headers() -> None:
    train_docs, *_ = data.load_20ng(remove=True)
    # When remove=('headers','footers','quotes') is on, no document
    # should start with the canonical "From:" header line.
    assert not any(doc.lstrip().startswith("From:") for doc in train_docs[:200])


@pytest.mark.slow
def test_load_20ng_returns_arrays_not_lists_for_labels() -> None:
    _, train_labels, *_ = data.load_20ng()
    assert isinstance(train_labels, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_project.data'` (or `AttributeError`).

- [ ] **Step 3: Implement `load_20ng`**

Create `src/nlp_project/data.py`:

```python
"""Data layer for Q1: loading and preprocessing 20 Newsgroups.

The dataset is *never* committed to git — it is fetched at runtime from
sklearn's bundled mirror via :func:`load_20ng`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_20newsgroups


def load_20ng(
    remove: bool = True,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    """Fetch the 20 Newsgroups train/test splits.

    Parameters
    ----------
    remove:
        If True (default), strip headers, footers, and quoted text from
        every document. This is the *honest* setting — leaving headers in
        leaks the label (see spec §3, "label leakage").

    Returns
    -------
    (train_docs, train_labels, test_docs, test_labels, label_names)
        Documents are raw strings; labels are integer arrays in [0, 20);
        label_names is the 20-string list of newsgroup names.
    """
    remove_tuple = ("headers", "footers", "quotes") if remove else ()
    train = fetch_20newsgroups(subset="train", remove=remove_tuple)
    test = fetch_20newsgroups(subset="test", remove=remove_tuple)
    return (
        list(train.data),
        np.asarray(train.target),
        list(test.data),
        np.asarray(test.target),
        list(train.target_names),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_data.py -v -m slow`
Expected: 3 passed. (First run is slow; sklearn caches the dataset under `~/scikit_learn_data/`.)

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/data.py tests/test_data.py
git commit -m "feat(data): add load_20ng with header/footer/quote stripping"
```

---

## Task 4: `data.preprocess` — tokenize, lowercase, drop short tokens and stopwords

**Files:**
- Modify: `src/nlp_project/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
def test_preprocess_lowercases_and_tokenizes() -> None:
    docs = ["Hello WORLD foo"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords=set())
    assert out == [["hello", "world", "foo"]]


def test_preprocess_drops_short_tokens() -> None:
    # gensim.simple_preprocess drops tokens with len < min_len (default 2);
    # we additionally enforce len >= 3.
    docs = ["a be cat dog"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords=set())
    assert out == [["cat", "dog"]]


def test_preprocess_drops_stopwords_when_flag_true() -> None:
    docs = ["the cat sat on the mat"]
    out = data.preprocess(docs, drop_stopwords=True, stopwords={"the", "sat", "mat"})
    assert out == [["cat"]]


def test_preprocess_keeps_stopwords_when_flag_false() -> None:
    docs = ["the cat sat"]
    out = data.preprocess(docs, drop_stopwords=False, stopwords={"the"})
    assert out == [["the", "cat", "sat"]]


def test_preprocess_returns_list_of_lists() -> None:
    out = data.preprocess(["one two three", "four five six"], drop_stopwords=False, stopwords=set())
    assert isinstance(out, list)
    assert all(isinstance(doc, list) for doc in out)
    assert len(out) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py::test_preprocess_lowercases_and_tokenizes -v`
Expected: FAIL with `AttributeError: module 'nlp_project.data' has no attribute 'preprocess'`.

- [ ] **Step 3: Implement `preprocess`**

Append to `src/nlp_project/data.py`:

```python
from functools import lru_cache

from gensim.utils import simple_preprocess


@lru_cache(maxsize=1)
def _default_stopwords() -> frozenset[str]:
    """Load NLTK English stopwords once. Requires `scripts/setup_nltk.py`
    to have been run, otherwise raises a clear LookupError.
    """
    from nltk.corpus import stopwords  # local import keeps the package
                                       # importable without nltk data.
    return frozenset(stopwords.words("english"))


def preprocess(
    docs: Sequence[str],
    drop_stopwords: bool = True,
    stopwords: Sequence[str] | set[str] | None = None,
    min_token_len: int = 3,
) -> list[list[str]]:
    """Tokenize, lowercase, drop short tokens, optionally drop stopwords.

    Parameters
    ----------
    docs:
        Raw document strings.
    drop_stopwords:
        If True (default), filter tokens that appear in ``stopwords``.
        Disable this when feeding tokens to word2vec — embeddings learn
        better when frequent function words remain in the context window.
    stopwords:
        Override the default NLTK English stopword list. Pass a set in
        tests so they don't depend on NLTK's data being downloaded.
    min_token_len:
        Drop tokens shorter than this (default 3). gensim's
        ``simple_preprocess`` already lowercases and strips punctuation.
    """
    if drop_stopwords:
        sw = set(stopwords) if stopwords is not None else set(_default_stopwords())
    else:
        sw = set()

    out: list[list[str]] = []
    for doc in docs:
        toks = simple_preprocess(doc, min_len=min_token_len)
        if sw:
            toks = [t for t in toks if t not in sw]
        out.append(toks)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v -k preprocess`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/data.py tests/test_data.py
git commit -m "feat(data): add preprocess() with injectable stopword set"
```

---

## Task 5: `data.train_val_split` — stratified split with fixed seed

**Files:**
- Modify: `src/nlp_project/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
def test_train_val_split_sizes() -> None:
    docs = [f"doc {i}" for i in range(100)]
    labels = np.array([i % 5 for i in range(100)])  # 20 per class
    train_docs, train_labels, val_docs, val_labels = data.train_val_split(
        docs, labels, val_frac=0.2, seed=42,
    )
    assert len(train_docs) == 80
    assert len(val_docs) == 20
    assert len(train_labels) == 80
    assert len(val_labels) == 20


def test_train_val_split_is_reproducible() -> None:
    docs = [f"doc {i}" for i in range(50)]
    labels = np.array([i % 2 for i in range(50)])
    a = data.train_val_split(docs, labels, val_frac=0.2, seed=42)
    b = data.train_val_split(docs, labels, val_frac=0.2, seed=42)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def test_train_val_split_is_stratified() -> None:
    docs = [f"doc {i}" for i in range(100)]
    labels = np.array([i % 5 for i in range(100)])
    _, train_labels, _, val_labels = data.train_val_split(
        docs, labels, val_frac=0.2, seed=42,
    )
    # Each class should appear in val proportionally (4 of 20 = 20%).
    for c in range(5):
        assert (val_labels == c).sum() == 4
        assert (train_labels == c).sum() == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v -k train_val_split`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `train_val_split`**

Append to `src/nlp_project/data.py`:

```python
from sklearn.model_selection import train_test_split


def train_val_split(
    docs: Sequence[str] | list[list[str]],
    labels: np.ndarray,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list, np.ndarray, list, np.ndarray]:
    """Stratified train/validation split.

    Works on either a list of raw strings or a list of token lists.
    """
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        list(docs),
        labels,
        test_size=val_frac,
        stratify=labels,
        random_state=seed,
    )
    return train_docs, train_labels, val_docs, val_labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v -k train_val_split`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/data.py tests/test_data.py
git commit -m "feat(data): add stratified train/val split"
```

---

## Task 6: `embeddings.train_word2vec` — deterministic gensim wrapper

**Files:**
- Create: `src/nlp_project/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embeddings.py`:

```python
"""Tests for src/nlp_project/embeddings.py."""

from __future__ import annotations

import numpy as np
import pytest

from nlp_project import embeddings


@pytest.fixture
def tiny_corpus() -> list[list[str]]:
    """A toy corpus rich enough for word2vec to assign vectors.

    Word2vec needs each token to appear at least `min_count` times. We
    use min_count=1 in tests and repeat the same 4-document mini-corpus
    4 times so the vocab stabilises.
    """
    base = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "barked", "at", "the", "cat"],
        ["the", "dog", "and", "the", "cat", "are", "friends"],
        ["dogs", "love", "cats", "and", "cats", "love", "dogs"],
    ]
    return base * 4


def test_train_word2vec_vocab_contains_expected_tokens(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1)
    for tok in ["the", "cat", "dog"]:
        assert tok in model.wv.key_to_index


def test_train_word2vec_vector_size(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1)
    assert model.wv["cat"].shape == (16,)


def test_train_word2vec_is_deterministic_with_workers_1(tiny_corpus) -> None:
    a = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1, seed=42)
    b = embeddings.train_word2vec(tiny_corpus, epochs=5, vector_size=16, min_count=1, seed=42)
    np.testing.assert_array_equal(a.wv["cat"], b.wv["cat"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `train_word2vec`**

Create `src/nlp_project/embeddings.py`:

```python
"""word2vec training and pooling utilities.

We pin ``workers=1`` because gensim's multi-threaded training is
non-deterministic even with a fixed seed (threads consume the corpus in
non-deterministic order). Determinism matters here because the spec
requires comparing 1-epoch and 20-epoch checkpoints — only meaningful if
both models start from byte-identical initial weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gensim.models import Word2Vec


def train_word2vec(
    token_lists: Sequence[Sequence[str]],
    epochs: int,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    sg: int = 1,
    seed: int = 42,
    workers: int = 1,
) -> Word2Vec:
    """Train a skip-gram word2vec model on tokenized documents.

    Parameters
    ----------
    epochs:
        Number of training epochs. Pass 1 for the "untrained" snapshot
        and 20 for the "trained" snapshot used by the spec's comparison.
    sg:
        1 = skip-gram, 0 = CBOW. Skip-gram works better for small corpora
        and rare words.
    workers:
        Defaults to 1 for determinism. Bump to 4 for one-off exploration.
    """
    # Word2Vec expects an iterable of iterables of tokens.
    return Word2Vec(
        sentences=list(token_lists),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): add deterministic word2vec wrapper"
```

---

## Task 7: `embeddings.mean_pool` — document vector via mean of token vectors

**Files:**
- Modify: `src/nlp_project/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
def test_mean_pool_shape(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    docs = [["cat", "dog"], ["the", "mat"]]
    X = embeddings.mean_pool(docs, model)
    assert X.shape == (2, 8)
    assert X.dtype == np.float32


def test_mean_pool_matches_hand_computed(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    expected = (model.wv["cat"] + model.wv["dog"]) / 2.0
    X = embeddings.mean_pool([["cat", "dog"]], model)
    np.testing.assert_allclose(X[0], expected, rtol=1e-5)


def test_mean_pool_zero_vector_for_all_oov(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    X = embeddings.mean_pool([["xyzzy", "qux", "frobnicate"]], model)
    assert X.shape == (1, 8)
    np.testing.assert_array_equal(X[0], np.zeros(8, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings.py -v -k mean_pool`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `mean_pool`**

Append to `src/nlp_project/embeddings.py`:

```python
def mean_pool(
    token_lists: Sequence[Sequence[str]],
    model: Word2Vec,
) -> np.ndarray:
    """Average the token vectors for each document.

    OOV tokens are skipped. Documents with zero in-vocab tokens get a
    zero vector — the alternative (NaN propagation) breaks downstream
    training, and a zero vector at least keeps batch shapes consistent.
    """
    dim = model.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    wv = model.wv
    for i, toks in enumerate(token_lists):
        vecs = [wv[t] for t in toks if t in wv.key_to_index]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
        # else: leave as zeros (already initialised).
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings.py -v -k mean_pool`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): add mean_pool with zero-vector fallback"
```

---

## Task 8: `embeddings.mean_max_pool` — concatenate mean and element-wise max

**Files:**
- Modify: `src/nlp_project/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
def test_mean_max_pool_shape_is_double(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    X = embeddings.mean_max_pool([["cat", "dog"]], model)
    assert X.shape == (1, 16)  # 2 * vector_size


def test_mean_max_pool_first_half_matches_mean_pool(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    docs = [["cat", "dog"]]
    X_mean = embeddings.mean_pool(docs, model)
    X_concat = embeddings.mean_max_pool(docs, model)
    np.testing.assert_allclose(X_concat[0, :8], X_mean[0], rtol=1e-5)


def test_mean_max_pool_second_half_is_elementwise_max(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    expected_max = np.maximum(model.wv["cat"], model.wv["dog"])
    X = embeddings.mean_max_pool([["cat", "dog"]], model)
    np.testing.assert_allclose(X[0, 8:], expected_max, rtol=1e-5)


def test_mean_max_pool_zero_for_all_oov(tiny_corpus) -> None:
    model = embeddings.train_word2vec(tiny_corpus, epochs=3, vector_size=8, min_count=1)
    X = embeddings.mean_max_pool([["xyzzy"]], model)
    np.testing.assert_array_equal(X[0], np.zeros(16, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings.py -v -k mean_max_pool`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `mean_max_pool`**

Append to `src/nlp_project/embeddings.py`:

```python
def mean_max_pool(
    token_lists: Sequence[Sequence[str]],
    model: Word2Vec,
) -> np.ndarray:
    """Concatenate the mean and element-wise max of token vectors.

    This is the Q1d "bonus experiment": same network, richer fixed-size
    representation. Output dimension is ``2 * vector_size``.
    """
    dim = model.vector_size
    out = np.zeros((len(token_lists), 2 * dim), dtype=np.float32)
    wv = model.wv
    for i, toks in enumerate(token_lists):
        vecs = [wv[t] for t in toks if t in wv.key_to_index]
        if vecs:
            stacked = np.stack(vecs)  # (n_tokens, dim)
            out[i, :dim] = stacked.mean(axis=0)
            out[i, dim:] = stacked.max(axis=0)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings.py -v -k mean_max_pool`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): add mean+max pooling for Q1d"
```

---

## Task 9: `vectorizers` — TF-IDF wrapper

**Files:**
- Create: `src/nlp_project/vectorizers.py`
- Create: `tests/test_vectorizers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vectorizers.py`:

```python
"""Tests for src/nlp_project/vectorizers.py."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from nlp_project import vectorizers


def test_fit_tfidf_returns_sparse_matrix_of_correct_shape() -> None:
    docs = ["the cat sat on the mat", "the dog barked at the cat", "dogs love cats"]
    vec, X = vectorizers.fit_tfidf(docs, max_features=20, min_df=1)
    assert sp.issparse(X)
    assert X.shape[0] == 3
    assert X.shape[1] <= 20


def test_transform_tfidf_uses_train_vocab() -> None:
    train = ["the cat sat", "the dog barked"]
    test = ["a brand new sentence with cat"]
    vec, X_train = vectorizers.fit_tfidf(train, max_features=50, min_df=1)
    X_test = vectorizers.transform_tfidf(vec, test)
    # Test matrix must have the same number of features as train.
    assert X_test.shape[1] == X_train.shape[1]
    # "brand", "new", "sentence", "with" are not in train vocab and
    # should be ignored; only "cat" should yield a non-zero column.
    assert X_test.nnz <= 2  # "cat" plus possibly tf-idf-zero rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vectorizers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the TF-IDF wrapper**

Create `src/nlp_project/vectorizers.py`:

```python
"""TF-IDF vectorization for Q1c.

Thin wrapper around scikit-learn's TfidfVectorizer that fixes the
project's defaults in one place. Inputs are document *strings*, not
token lists — TfidfVectorizer does its own tokenization. Pass
already-preprocessed strings (e.g. ``" ".join(tokens)``) to keep the
preprocessing consistent with the word2vec input.
"""

from __future__ import annotations

from typing import Sequence

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(
    docs: Sequence[str],
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a TfidfVectorizer on ``docs`` and return both it and the matrix."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    X = vec.fit_transform(docs)
    return vec, X


def transform_tfidf(vec: TfidfVectorizer, docs: Sequence[str]) -> sp.csr_matrix:
    """Apply a fitted vectorizer to new documents."""
    return vec.transform(docs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vectorizers.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/vectorizers.py tests/test_vectorizers.py
git commit -m "feat(vectorizers): add TF-IDF wrapper with project defaults"
```

---

## Task 10: `model.MLP` — the two-layer classifier

**Files:**
- Create: `src/nlp_project/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
"""Tests for src/nlp_project/model.py."""

from __future__ import annotations

import torch

from nlp_project.model import MLP


def test_mlp_forward_shape() -> None:
    model = MLP(in_dim=100, hidden_dim=64, num_classes=20)
    x = torch.randn(8, 100)
    out = model(x)
    assert out.shape == (8, 20)


def test_mlp_has_two_linear_layers_and_a_relu() -> None:
    """The spec mandates Linear -> ReLU -> Linear. Verify the structure."""
    model = MLP(in_dim=10, hidden_dim=8, num_classes=3)
    children = list(model.children())
    # Sequential with: Linear, ReLU, Dropout, Linear (dropout is allowed).
    layer_types = [type(m).__name__ for m in model.net]
    assert layer_types[0] == "Linear"
    assert layer_types[1] == "ReLU"
    assert "Linear" in layer_types[2:]


def test_mlp_dropout_is_active_in_train_mode() -> None:
    """Same input twice in train mode should yield different outputs
    when dropout > 0 — proves dropout is wired up."""
    torch.manual_seed(0)
    model = MLP(in_dim=10, hidden_dim=64, num_classes=3, dropout=0.5)
    model.train()
    x = torch.randn(4, 10)
    a = model(x)
    b = model(x)
    assert not torch.allclose(a, b)


def test_mlp_dropout_is_inactive_in_eval_mode() -> None:
    model = MLP(in_dim=10, hidden_dim=64, num_classes=3, dropout=0.5)
    model.eval()
    x = torch.randn(4, 10)
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the MLP**

Create `src/nlp_project/model.py`:

```python
"""The Q1 model: a two-layer MLP.

Architecture is fixed by the spec: Linear -> ReLU -> Linear, with
optional dropout between the two linear layers. Dropout is on by default
because without it the 256-hidden model overfits 20NG hard, especially
on the 20k-feature TF-IDF input. Dropout is *not* a structural change to
the network — both linear layers and the ReLU are still there.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/model.py tests/test_model.py
git commit -m "feat(model): add two-layer MLP per spec"
```

---

## Task 11: `train.train` — training loop with early stopping and W&B

**Files:**
- Create: `src/nlp_project/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test (overfit a tiny synthetic dataset)**

Create `tests/test_train.py`:

```python
"""Tests for src/nlp_project/train.py.

Strategy: train on a tiny separable synthetic dataset and assert the
final training accuracy is high. This is the canonical way to test a
training loop end-to-end without depending on real data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import set_seed
from nlp_project.model import MLP
from nlp_project.train import train


def _toy_loaders() -> tuple[DataLoader, DataLoader]:
    """Two well-separated Gaussian clusters in 8 dimensions."""
    set_seed()
    n = 256
    X0 = np.random.randn(n, 8).astype(np.float32) - 2.0
    X1 = np.random.randn(n, 8).astype(np.float32) + 2.0
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n, dtype=np.int64)
    perm = np.random.permutation(2 * n)
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(val_ds, batch_size=32)


def test_train_overfits_separable_synthetic_data() -> None:
    train_loader, val_loader = _toy_loaders()
    model = MLP(in_dim=8, hidden_dim=16, num_classes=2, dropout=0.0)
    history = train(
        model, train_loader, val_loader,
        epochs=20, lr=1e-2, device="cpu", wandb_run=None, patience=20,
    )
    assert history["train_acc"][-1] > 0.95
    assert history["val_acc"][-1] > 0.90


def test_train_returns_history_with_expected_keys() -> None:
    train_loader, val_loader = _toy_loaders()
    model = MLP(in_dim=8, hidden_dim=16, num_classes=2, dropout=0.0)
    history = train(
        model, train_loader, val_loader,
        epochs=2, lr=1e-2, device="cpu", wandb_run=None, patience=10,
    )
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in history
        assert len(history[key]) == 2


def test_train_early_stops_on_val_loss() -> None:
    """If val loss never improves, early stopping should trigger before max epochs."""
    # Create a pathological dataset where val loss can't improve.
    set_seed()
    n = 64
    X = np.random.randn(n, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=n, dtype=np.int64)  # pure noise
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=16)
    model = MLP(in_dim=4, hidden_dim=8, num_classes=2, dropout=0.0)
    history = train(
        model, loader, loader,
        epochs=50, lr=1e-2, device="cpu", wandb_run=None, patience=3,
    )
    # We should not have run all 50 epochs.
    assert len(history["val_loss"]) < 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the training loop**

Create `src/nlp_project/train.py`:

```python
"""Training loop for the MLP.

Plain PyTorch — no Lightning, no Trainer abstraction. The loop is small
enough that hiding it behind a framework would obscure what's going on.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    wandb_run: Any | None = None,
    patience: int = 5,
    weight_decay: float = 1e-5,
) -> dict[str, list[float]]:
    """Train ``model`` and return a per-epoch history.

    Stops early when the validation loss has not improved for ``patience``
    consecutive epochs. The model parameters are restored to those of the
    best (lowest val loss) epoch before returning.

    ``wandb_run`` is the result of ``wandb.init(...)``. Pass ``None`` in
    tests or for ad-hoc local runs.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            running_total += xb.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # ------------------- validation -------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: 3 passed. (The overfit test trains for ~20 epochs on tiny data; runs in <2 seconds on CPU.)

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/train.py tests/test_train.py
git commit -m "feat(train): add training loop with early stopping and W&B hook"
```

---

## Task 12: `eval.evaluate` and `eval.plot_confusion`

**Files:**
- Create: `src/nlp_project/eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval.py`:

```python
"""Tests for src/nlp_project/eval.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import eval as ev


class _ConstantModel(nn.Module):
    """A model that always predicts class ``cls``."""

    def __init__(self, num_classes: int, cls: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls = cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.size(0), self.num_classes), -1e9)
        out[:, self.cls] = 1e9
        return out


def _loader(y: list[int], num_features: int = 4) -> DataLoader:
    X = torch.zeros(len(y), num_features)
    return DataLoader(TensorDataset(X, torch.tensor(y, dtype=torch.long)), batch_size=4)


def test_evaluate_perfect_predictions() -> None:
    model = _ConstantModel(num_classes=3, cls=2)
    loader = _loader([2, 2, 2, 2])
    metrics = ev.evaluate(model, loader, label_names=["a", "b", "c"], device="cpu")
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] > 0.0
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_evaluate_all_wrong() -> None:
    model = _ConstantModel(num_classes=3, cls=0)
    loader = _loader([1, 2, 1, 2])
    metrics = ev.evaluate(model, loader, label_names=["a", "b", "c"], device="cpu")
    assert metrics["accuracy"] == 0.0


def test_plot_confusion_writes_a_png(tmp_path: Path) -> None:
    cm = np.array([[5, 1], [0, 4]])
    out = tmp_path / "cm.png"
    ev.plot_confusion(cm, label_names=["x", "y"], save_path=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement evaluation helpers**

Create `src/nlp_project/eval.py`:

```python
"""Evaluation metrics and confusion-matrix plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_names: list[str],
    device: str,
) -> dict:
    """Run ``model`` over ``loader`` and return metrics.

    Returns a dict with keys ``accuracy``, ``macro_f1``, ``per_class_f1``
    (length ``len(label_names)``) and ``confusion_matrix``.
    """
    model.to(device).eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(yb.tolist())

    y_true = np.asarray(targets)
    y_pred = np.asarray(preds)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": f1_score(
            y_true, y_pred,
            average=None,
            labels=list(range(len(label_names))),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(label_names))),
        ),
    }


def plot_confusion(
    cm: np.ndarray,
    label_names: list[str],
    save_path: Path | str,
    title: str = "Confusion matrix",
) -> None:
    """Render a confusion-matrix heatmap and save it as PNG."""
    fig, ax = plt.subplots(figsize=(0.5 * len(label_names) + 2, 0.5 * len(label_names) + 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/eval.py tests/test_eval.py
git commit -m "feat(eval): add metrics and confusion-matrix plotting"
```

---

## Task 13: `viz.plot_word_neighborhood` — t-SNE comparison plot

**Files:**
- Create: `src/nlp_project/viz.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viz.py`:

```python
"""Tests for src/nlp_project/viz.py — smoke test only."""

from __future__ import annotations

from pathlib import Path

from nlp_project import embeddings, viz


def test_plot_word_neighborhood_writes_a_png(tmp_path: Path) -> None:
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "barked", "at", "the", "cat"],
    ] * 8
    m1 = embeddings.train_word2vec(corpus, epochs=1, vector_size=8, min_count=1, seed=42)
    m20 = embeddings.train_word2vec(corpus, epochs=20, vector_size=8, min_count=1, seed=42)
    out = tmp_path / "tsne.png"
    # Pick words guaranteed to be in vocab.
    words = ["the", "cat", "dog", "sat", "mat", "barked"]
    viz.plot_word_neighborhood({"epoch=1": m1, "epoch=20": m20}, words, save_path=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the t-SNE plot**

Create `src/nlp_project/viz.py`:

```python
"""Embedding visualizations for the report."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def plot_word_neighborhood(
    models: Mapping[str, Word2Vec],
    words: list[str],
    save_path: Path | str,
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    """Project the same word list with t-SNE under each of ``models``.

    Renders one subplot per model, side by side. The point of the plot is
    to *qualitatively* compare embedding spaces — for example, how a
    1-epoch word2vec model differs from a 20-epoch one. Token labels are
    drawn next to each point so the report reader can trace specific
    words across panels.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        in_vocab = [w for w in words if w in model.wv.key_to_index]
        if len(in_vocab) < 3:
            ax.set_title(f"{name}\n(too few in-vocab words)")
            continue
        vecs = np.stack([model.wv[w] for w in in_vocab])
        # perplexity must be < n_samples; clip for tiny test corpora.
        eff_perp = min(perplexity, max(2.0, len(in_vocab) - 1))
        coords = TSNE(
            n_components=2, perplexity=eff_perp, random_state=seed, init="random",
        ).fit_transform(vecs)
        ax.scatter(coords[:, 0], coords[:, 1], s=20)
        for (x, y), w in zip(coords, in_vocab):
            ax.annotate(w, (x, y), fontsize=8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nlp_project/viz.py tests/test_viz.py
git commit -m "feat(viz): add t-SNE plot for word2vec checkpoint comparison"
```

---

## Task 14: NLTK setup script

**Files:**
- Create: `scripts/setup_nltk.py`

- [ ] **Step 1: Create the script**

Create `scripts/setup_nltk.py`:

```python
"""One-time setup: download NLTK English stopwords.

Run before opening the notebooks the first time:
    uv run python scripts/setup_nltk.py
"""

from __future__ import annotations

import nltk


def main() -> None:
    nltk.download("stopwords", quiet=True)
    # Sanity check.
    from nltk.corpus import stopwords

    words = stopwords.words("english")
    print(f"Downloaded NLTK English stopwords: {len(words)} words.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to populate the NLTK data**

Run: `uv run python scripts/setup_nltk.py`
Expected output: `Downloaded NLTK English stopwords: 179 words.` (count may vary slightly across NLTK versions).

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_nltk.py
git commit -m "chore: add NLTK stopwords setup script"
```

---

## Task 15: Q1a notebook — preprocessing exploration

**Files:**
- Create: `notebooks/q1a_preprocessing.ipynb`
- Create: `figures/class_balance.png` (generated)
- Create: `figures/doc_length_hist.png` (generated)

- [ ] **Step 1: Create the notebook**

Create `notebooks/q1a_preprocessing.ipynb` with the following cells. Use `jupyter nbconvert --to notebook` to generate the file from the JSON below, or build it directly in JupyterLab. Save it with these cells executed end-to-end:

**Cell 1 (markdown):**

```markdown
# Q1a — Data Preprocessing

**W&B run:** _filled in after first run_

This notebook fetches 20 Newsgroups, runs the project's preprocessing
pipeline, and produces the two descriptive plots referenced in the
report:

- class balance bar chart
- document-length histogram (in tokens, post-preprocessing)
```

**Cell 2 (code):**

```python
%load_ext autoreload
%autoreload 2

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlp_project import set_seed
from nlp_project.data import load_20ng, preprocess

set_seed()
FIG_DIR = Path("../figures")
FIG_DIR.mkdir(exist_ok=True)
```

**Cell 3 (code) — load + preprocess:**

```python
train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)
print(f"train docs: {len(train_docs)}, test docs: {len(test_docs)}")
print(f"first label: {label_names[train_labels[0]]}")

train_tokens = preprocess(train_docs, drop_stopwords=True)
print(f"first doc, first 20 tokens: {train_tokens[0][:20]}")
```

**Cell 4 (code) — class balance plot:**

```python
counts = Counter(train_labels.tolist())
order = sorted(counts.keys())
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([label_names[i] for i in order], [counts[i] for i in order])
ax.set_ylabel("number of training docs")
ax.set_title("20 Newsgroups — class balance (train split)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(FIG_DIR / "class_balance.png", dpi=150)
plt.show()
```

**Cell 5 (code) — doc-length histogram:**

```python
lengths = np.array([len(t) for t in train_tokens])
print(f"median: {int(np.median(lengths))}, mean: {lengths.mean():.1f}, "
      f"p95: {int(np.percentile(lengths, 95))}, max: {lengths.max()}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(lengths, 0, 1000), bins=50)
ax.set_xlabel("document length (tokens, clipped at 1000)")
ax.set_ylabel("count")
ax.set_title("Document length after preprocessing")
fig.tight_layout()
fig.savefig(FIG_DIR / "doc_length_hist.png", dpi=150)
plt.show()
```

**Cell 6 (code) — vocab size:**

```python
vocab = Counter()
for toks in train_tokens:
    vocab.update(toks)
print(f"vocab size: {len(vocab)}")
print(f"top 10: {vocab.most_common(10)}")
```

**Cell 7 (markdown):**

```markdown
## Notes for the report

- Class balance is mild (≈400–600 docs/class) — accuracy is roughly comparable to macro-F1.
- Median doc length after preprocessing is ~80 tokens; p95 ~400. The MLP sees fixed-size doc vectors so length is a sanity check, not a hyperparameter.
- Vocab size and most-frequent tokens go in the report's data section.
```

- [ ] **Step 2: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/q1a_preprocessing.ipynb --inplace`
Expected: notebook executes without errors and the two PNGs appear in `figures/`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/q1a_preprocessing.ipynb figures/class_balance.png figures/doc_length_hist.png
git commit -m "feat(q1a): preprocessing notebook with class-balance and length plots"
```

---

## Task 16: Q1b notebook — word2vec → MLP

**Files:**
- Create: `notebooks/q1b_word2vec.ipynb`
- Create: `figures/w2v_tsne_epoch1_vs_epoch20.png` (generated)
- Create: `figures/confusion_matrix_q1b.png` (generated)
- Create: `models/w2v_epoch1.kv`, `models/w2v_epoch20.kv` (generated, gitignored)

- [ ] **Step 1: Create the notebook**

Create `notebooks/q1b_word2vec.ipynb` with these cells:

**Cell 1 (markdown):**

```markdown
# Q1b — Word2Vec embeddings → two-layer MLP

**W&B run:** _filled in after first run_

This notebook trains two word2vec checkpoints (1 epoch and 20 epochs)
from byte-identical initial weights, compares their embedding space with
t-SNE, then trains the project MLP on mean-pooled 20-epoch vectors.
```

**Cell 2 (code) — imports + setup:**

```python
%load_ext autoreload
%autoreload 2

import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import SEED, set_seed
from nlp_project.data import load_20ng, preprocess, train_val_split
from nlp_project.embeddings import train_word2vec, mean_pool
from nlp_project.eval import evaluate, plot_confusion
from nlp_project.model import MLP
from nlp_project.train import train as train_loop
from nlp_project.viz import plot_word_neighborhood

set_seed()
FIG_DIR = Path("../figures"); FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("../models"); MODEL_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Cell 3 (code) — load and preprocess (no stopword drop, for word2vec):**

```python
train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)

# For word2vec we keep stopwords — co-occurrence with frequent function
# words helps the embedding signal (see spec §8).
train_tokens_w2v = preprocess(train_docs, drop_stopwords=False)
test_tokens_w2v = preprocess(test_docs, drop_stopwords=False)
```

**Cell 4 (code) — train both word2vec checkpoints:**

```python
m1 = train_word2vec(train_tokens_w2v, epochs=1, vector_size=100, seed=SEED)
m1.wv.save(str(MODEL_DIR / "w2v_epoch1.kv"))

m20 = train_word2vec(train_tokens_w2v, epochs=20, vector_size=100, seed=SEED)
m20.wv.save(str(MODEL_DIR / "w2v_epoch20.kv"))

print(f"vocab(epoch=1):  {len(m1.wv)}")
print(f"vocab(epoch=20): {len(m20.wv)}")
```

**Cell 5 (code) — t-SNE comparison:**

```python
counter = Counter()
for toks in train_tokens_w2v:
    counter.update(toks)

# Use the 500 most frequent tokens that exist in *both* checkpoints.
common = [w for w, _ in counter.most_common(2000)
          if w in m1.wv.key_to_index and w in m20.wv.key_to_index][:500]
print(f"plotting {len(common)} tokens")

plot_word_neighborhood(
    {"word2vec — 1 epoch": m1, "word2vec — 20 epochs": m20},
    common,
    save_path=FIG_DIR / "w2v_tsne_epoch1_vs_epoch20.png",
)
```

**Cell 6 (code) — build doc vectors with the 20-epoch model:**

```python
X_train_full = mean_pool(train_tokens_w2v, m20)
X_test = mean_pool(test_tokens_w2v, m20)
y_train_full = train_labels
y_test = test_labels
print(f"X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")
```

**Cell 7 (code) — train/val split + dataloaders:**

```python
X_train, y_train, X_val, y_val = train_val_split(
    list(X_train_full), y_train_full, val_frac=0.1, seed=SEED,
)
X_train, X_val = np.asarray(X_train), np.asarray(X_val)

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, 64, shuffle=True)
val_loader = make_loader(X_val, y_val, 64, shuffle=False)
test_loader = make_loader(X_test, y_test, 64, shuffle=False)
```

**Cell 8 (code) — W&B run + train:**

```python
run = wandb.init(
    project="hslu-nalapro-q1",
    name="q1b-word2vec-meanpool",
    config={"vectorizer": "word2vec-meanpool", "vector_size": 100,
            "w2v_epochs": 20, "hidden_dim": 256, "dropout": 0.3,
            "lr": 1e-3, "batch_size": 64},
)

model = MLP(in_dim=100, hidden_dim=256, num_classes=20, dropout=0.3)
history = train_loop(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, device=DEVICE, wandb_run=run, patience=5,
)
```

**Cell 9 (code) — evaluate and log:**

```python
metrics = evaluate(model, test_loader, label_names, device=DEVICE)
print(f"test accuracy: {metrics['accuracy']:.4f}")
print(f"test macro-F1: {metrics['macro_f1']:.4f}")
plot_confusion(
    metrics["confusion_matrix"], label_names,
    save_path=FIG_DIR / "confusion_matrix_q1b.png",
    title="Q1b — word2vec mean-pool",
)
run.log({"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]})
run.finish()
```

**Cell 10 (markdown):**

```markdown
## Notes for the report

- After 20 epochs the t-SNE plot shows visibly clustered semantic
  neighbourhoods (e.g. "car/engine/wheel" vs "god/jesus/bible") that
  are absent at 1 epoch — that's the visualization the spec asks for.
- Mean-pool baseline: this is the floor that 1c (TF-IDF) and 1d (mean+max-pool) need to beat.
```

- [ ] **Step 2: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/q1b_word2vec.ipynb --inplace`
Expected: completes in 5–15 min on CPU; produces both figures and a W&B run URL printed in the cell output.

- [ ] **Step 3: Paste the W&B run URL into Cell 1**

Edit Cell 1 to replace `_filled in after first run_` with the actual URL printed by `wandb.init`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/q1b_word2vec.ipynb \
        figures/w2v_tsne_epoch1_vs_epoch20.png \
        figures/confusion_matrix_q1b.png
git commit -m "feat(q1b): word2vec + MLP notebook with t-SNE comparison"
```

---

## Task 17: Q1c notebook — TF-IDF → MLP

**Files:**
- Create: `notebooks/q1c_tfidf.ipynb`
- Create: `figures/confusion_matrix_q1c.png` (generated)

- [ ] **Step 1: Create the notebook**

Create `notebooks/q1c_tfidf.ipynb` with these cells:

**Cell 1 (markdown):**

```markdown
# Q1c — TF-IDF → two-layer MLP

**W&B run:** _filled in after first run_

Same MLP as Q1b, but the input is now a TF-IDF vector instead of a
mean-pooled word2vec vector.
```

**Cell 2 (code) — imports:**

```python
%load_ext autoreload
%autoreload 2

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import SEED, set_seed
from nlp_project.data import load_20ng, preprocess, train_val_split
from nlp_project.eval import evaluate, plot_confusion
from nlp_project.model import MLP
from nlp_project.train import train as train_loop
from nlp_project.vectorizers import fit_tfidf, transform_tfidf

set_seed()
FIG_DIR = Path("../figures"); FIG_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Cell 3 (code) — load + preprocess (drop stopwords, since TF-IDF has them in built-in too):**

```python
train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)
train_tokens = preprocess(train_docs, drop_stopwords=True)
test_tokens = preprocess(test_docs, drop_stopwords=True)

# TfidfVectorizer wants strings; rejoin our cleaned tokens.
train_strings = [" ".join(t) for t in train_tokens]
test_strings = [" ".join(t) for t in test_tokens]
```

**Cell 4 (code) — fit TF-IDF + transform:**

```python
vec, X_train_full = fit_tfidf(train_strings, max_features=20_000, min_df=2)
X_test = transform_tfidf(vec, test_strings)
print(f"X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")
```

**Cell 5 (code) — train/val split (need dense for the MLP, or sparse-aware loader):**

```python
# 11k x 20k floats fits in memory comfortably (~1.7 GB float32).
# We densify once here to keep the dataloader simple.
X_train_full_dense = X_train_full.toarray().astype(np.float32)
X_test_dense = X_test.toarray().astype(np.float32)

X_train, y_train, X_val, y_val = train_val_split(
    list(X_train_full_dense), train_labels, val_frac=0.1, seed=SEED,
)
X_train, X_val = np.asarray(X_train), np.asarray(X_val)

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, 64, shuffle=True)
val_loader = make_loader(X_val, y_val, 64, shuffle=False)
test_loader = make_loader(X_test_dense, test_labels, 64, shuffle=False)
```

**Cell 6 (code) — train and evaluate:**

```python
run = wandb.init(
    project="hslu-nalapro-q1",
    name="q1c-tfidf",
    config={"vectorizer": "tfidf", "max_features": 20_000,
            "hidden_dim": 256, "dropout": 0.3, "lr": 1e-3, "batch_size": 64},
)

model = MLP(in_dim=X_train.shape[1], hidden_dim=256, num_classes=20, dropout=0.3)
history = train_loop(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, device=DEVICE, wandb_run=run, patience=5,
)

metrics = evaluate(model, test_loader, label_names, device=DEVICE)
print(f"test accuracy: {metrics['accuracy']:.4f}")
print(f"test macro-F1: {metrics['macro_f1']:.4f}")
plot_confusion(
    metrics["confusion_matrix"], label_names,
    save_path=FIG_DIR / "confusion_matrix_q1c.png",
    title="Q1c — TF-IDF",
)
run.log({"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]})
run.finish()
```

**Cell 7 (markdown):**

```markdown
## Notes for the report

- TF-IDF expects "Q1b's mean-pool baseline to be beaten" — discuss
  whether it does, and why. Bag-of-words methods carry no semantic
  similarity, but they preserve all word identities (vs the lossy
  averaging in mean-pool).
- Confusion matrix shows whether residual errors are between
  semantically related classes (e.g. `talk.religion.misc` vs `soc.religion.christian`).
```

- [ ] **Step 2: Run the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/q1c_tfidf.ipynb --inplace`
Expected: completes in 2–8 min on CPU; W&B URL printed in cell output.

- [ ] **Step 3: Paste the W&B URL into Cell 1**

- [ ] **Step 4: Commit**

```bash
git add notebooks/q1c_tfidf.ipynb figures/confusion_matrix_q1c.png
git commit -m "feat(q1c): TF-IDF + MLP notebook"
```

---

## Task 18: Q1d notebook — mean+max pool + comparison CSV

**Files:**
- Create: `notebooks/q1d_mean_max_pool.ipynb`
- Create: `figures/confusion_matrix_q1d.png` (generated)
- Create: `figures/metric_comparison_table.csv` (generated)

- [ ] **Step 1: Create the notebook**

Create `notebooks/q1d_mean_max_pool.ipynb` with these cells:

**Cell 1 (markdown):**

```markdown
# Q1d — Mean+Max pool word2vec → two-layer MLP

**W&B run:** _filled in after first run_

The "extra experiment" required by the spec. Same network as Q1b, same
20-epoch word2vec model — only the pooling changes from mean to
mean⊕max (concatenation), doubling the input dimension from 100 to 200.
```

**Cell 2 (code) — imports + load 20-epoch w2v from Q1b:**

```python
%load_ext autoreload
%autoreload 2

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset

from nlp_project import SEED, set_seed
from nlp_project.data import load_20ng, preprocess, train_val_split
from nlp_project.embeddings import mean_max_pool
from nlp_project.eval import evaluate, plot_confusion
from nlp_project.model import MLP
from nlp_project.train import train as train_loop

set_seed()
FIG_DIR = Path("../figures"); FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("../models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class _W2VWrapper:
    """mean_max_pool only needs `.wv` and `.vector_size`."""
    def __init__(self, wv: KeyedVectors) -> None:
        self.wv = wv
        self.vector_size = wv.vector_size

w2v = _W2VWrapper(KeyedVectors.load(str(MODEL_DIR / "w2v_epoch20.kv")))
```

**Cell 3 (code) — preprocess + vectorize:**

```python
train_docs, train_labels, test_docs, test_labels, label_names = load_20ng(remove=True)
train_tokens = preprocess(train_docs, drop_stopwords=False)
test_tokens = preprocess(test_docs, drop_stopwords=False)

X_train_full = mean_max_pool(train_tokens, w2v)
X_test = mean_max_pool(test_tokens, w2v)
print(f"X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")
```

**Cell 4 (code) — split, loaders, train:**

```python
X_train, y_train, X_val, y_val = train_val_split(
    list(X_train_full), train_labels, val_frac=0.1, seed=SEED,
)
X_train, X_val = np.asarray(X_train), np.asarray(X_val)

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, 64, shuffle=True)
val_loader = make_loader(X_val, y_val, 64, shuffle=False)
test_loader = make_loader(X_test, test_labels, 64, shuffle=False)

run = wandb.init(
    project="hslu-nalapro-q1",
    name="q1d-word2vec-mean-max-pool",
    config={"vectorizer": "word2vec-mean+max-pool", "vector_size": 100,
            "in_dim": 200, "hidden_dim": 256, "dropout": 0.3, "lr": 1e-3,
            "batch_size": 64},
)
model = MLP(in_dim=200, hidden_dim=256, num_classes=20, dropout=0.3)
history = train_loop(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, device=DEVICE, wandb_run=run, patience=5,
)
```

**Cell 5 (code) — evaluate, plot, finish run:**

```python
metrics = evaluate(model, test_loader, label_names, device=DEVICE)
print(f"test accuracy: {metrics['accuracy']:.4f}")
print(f"test macro-F1: {metrics['macro_f1']:.4f}")
plot_confusion(
    metrics["confusion_matrix"], label_names,
    save_path=FIG_DIR / "confusion_matrix_q1d.png",
    title="Q1d — word2vec mean+max pool",
)
run.log({"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]})
run.finish()
```

**Cell 6 (code) — comparison table CSV:**

```python
# After all three notebooks have run, this cell builds the report's
# metric comparison table from the per-run W&B logs.
# We hard-code the numbers from this kernel's last `metrics` dict for
# Q1d and copy the others from Q1b/Q1c output. Update if reruns change them.

rows = [
    {"experiment": "Q1b — word2vec mean-pool",      "accuracy": _Q1B_ACC, "macro_f1": _Q1B_F1},
    {"experiment": "Q1c — TF-IDF",                  "accuracy": _Q1C_ACC, "macro_f1": _Q1C_F1},
    {"experiment": "Q1d — word2vec mean+max-pool",  "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
]
df = pd.DataFrame(rows)
df.to_csv(FIG_DIR / "metric_comparison_table.csv", index=False)
df
```

> **Note for the engineer:** before running Cell 6, replace
> `_Q1B_ACC, _Q1B_F1, _Q1C_ACC, _Q1C_F1` with the numbers printed by Q1b
> Cell 9 and Q1c Cell 6 respectively. We don't fetch from W&B
> programmatically because that would couple the notebook to the live
> service and make reruns brittle.

**Cell 7 (markdown):**

```markdown
## Notes for the report

- The mean+max-pool variant doubles the input dim with zero changes to the network's parameter count beyond the first linear layer. Discuss whether the bump in macro-F1 (if any) justifies the extra dimensions.
- Comparison table goes straight into the report.
```

- [ ] **Step 2: Run the notebook**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/q1d_mean_max_pool.ipynb --inplace`

(After filling in `_Q1B_ACC` etc. in Cell 6 from Q1b/Q1c outputs.)

Expected: completes in 5–10 min on CPU; produces the confusion matrix and CSV.

- [ ] **Step 3: Paste the W&B URL into Cell 1**

- [ ] **Step 4: Commit**

```bash
git add notebooks/q1d_mean_max_pool.ipynb \
        figures/confusion_matrix_q1d.png \
        figures/metric_comparison_table.csv
git commit -m "feat(q1d): mean+max-pool experiment + comparison CSV"
```

---

## Task 19: Final sanity sweep — full test suite + lock in W&B URLs in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v --ignore=tests -k "not slow" tests/ && uv run pytest -v -m slow tests/`

Expected: all tests pass.

- [ ] **Step 2: Update `CLAUDE.md` with the W&B project URL**

Edit `CLAUDE.md` to add a new line under the "Hard constraints from the spec" section:

```markdown
- W&B project for Q1 runs: <paste the project URL from any of the runs, e.g. https://wandb.ai/<user>/hslu-nalapro-q1>.
```

- [ ] **Step 3: Verify the repo is clean and commit**

Run: `git status`
Expected: only the `CLAUDE.md` change is unstaged.

```bash
git add CLAUDE.md
git commit -m "docs: record Q1 W&B project URL in CLAUDE.md"
```

- [ ] **Step 4: Final smoke test**

Run: `uv run pytest -v` (all tests, slow included).
Expected: green across the board.

---

## Self-Review (already performed by the planner)

1. **Spec coverage** — every section of the spec has at least one task:
   - §3 hard constraints → covered by gitignore (Task 1), W&B init in notebooks (16/17/18), CLAUDE.md update (19).
   - §4.2 module responsibilities → one task per module (3–13).
   - §5 defaults → encoded as defaults in the corresponding source files; tests verify the determinism and shape contracts.
   - §7 report deliverables → notebooks save every required figure; Task 18 builds the comparison CSV.
   - §6 reproducibility → Task 1 wires `set_seed`; Task 2 verifies determinism.
   - §8 risks → addressed: stopword flag (Task 4), `workers=1` (Task 6), CSV mitigation for W&B downtime is partly handled by `history` return + can be saved per-notebook.
2. **Placeholder scan** — no "TBD" / "TODO" / "fill in details" / "similar to Task N" / "implement later".
3. **Type consistency** — `train_word2vec` returns `gensim.models.Word2Vec`; `mean_pool`/`mean_max_pool` accept it. `MLP.__init__` signature is the same in tests, training, and notebooks. `evaluate` returns the same dict shape used by both the notebook log calls and `plot_confusion`. `train_val_split` returns `(train, train_labels, val, val_labels)` — order is consistent across all three notebooks.
4. **Scope** — single Q1 plan, ~19 tasks, fits comfortably in one implementation session.
