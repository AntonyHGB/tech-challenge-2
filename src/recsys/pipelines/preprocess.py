"""DVC stage 1: clean the raw interactions.

Run with ``poetry run python -m recsys.pipelines.preprocess``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.data.loader import clean_interactions, load_interactions
from recsys.pipelines.context import StageContext


def main() -> int:
    """Clean the raw dataset and store it for feature engineering.

    Returns:
        ``0`` on success.

    Raises:
        FileNotFoundError: If the DVC-tracked raw dataset is missing.
    """
    context = StageContext.load()
    source = context.layout.raw_interactions
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset '{source}' not found. Run 'dvc pull' or "
            "'python scripts/download_dataset.py' first."
        )
    raw = load_interactions(source)
    cleaned = clean_interactions(
        raw,
        min_user_interactions=context.params.data.min_user_interactions,
        min_item_interactions=context.params.data.min_item_interactions,
    )
    target = context.layout.interactions
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(target, index=False)
    _report(len(raw), cleaned, target)
    return 0


def _report(raw_rows: int, cleaned: pd.DataFrame, target: Path) -> None:
    """Print a short summary of the cleaning step.

    Args:
        raw_rows: Number of rows read from the raw file.
        cleaned: Cleaned interaction frame.
        target: Path the frame was written to.
    """
    print(f"[preprocess] raw interactions:     {raw_rows}")
    print(f"[preprocess] cleaned interactions: {len(cleaned)}")
    print(f"[preprocess] users:                {cleaned['user_id'].nunique()}")
    print(f"[preprocess] items:                {cleaned['item_id'].nunique()}")
    print(f"[preprocess] written to:           {target}")


if __name__ == "__main__":
    raise SystemExit(main())
