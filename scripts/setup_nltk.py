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
