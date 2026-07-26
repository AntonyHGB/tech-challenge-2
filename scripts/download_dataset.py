"""Download the raw dataset so it can be versioned with DVC.

Run with ``poetry run python scripts/download_dataset.py``. The file lands in
``data/raw/ratings.csv`` and is tracked by DVC rather than git.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from recsys.config import get_settings  # noqa: E402
from recsys.data.download import download_interactions  # noqa: E402


def main() -> int:
    """Download the dataset into the configured raw directory.

    Returns:
        ``0`` on success.
    """
    settings = get_settings()
    target = download_interactions(settings.data_raw_dir)
    size_mb = target.stat().st_size / 1_000_000
    print(f"Dataset ready at {target} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
