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
