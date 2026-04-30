"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from nlp_project import set_seed


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Reset all RNGs before every test for determinism."""
    set_seed()
