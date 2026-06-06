"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from nlp_project import set_seed


@pytest.fixture(autouse=True)
def _seed_everything() -> None:  # pyright: ignore[reportUnusedFunction]
    """Reset all RNGs before every test for determinism.

    Pytest discovers this fixture by collection; it is not called directly.
    """
    set_seed()
