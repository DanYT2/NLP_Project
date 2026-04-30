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
