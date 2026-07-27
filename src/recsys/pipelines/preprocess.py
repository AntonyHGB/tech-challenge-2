"""Stage 1 do DVC: limpa as interações brutas.

Execute com ``poetry run python -m recsys.pipelines.preprocess``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.data.dataset import clean_interactions, load_interactions
from recsys.pipelines.context import StageContext


def main() -> int:
    """Limpa o dataset bruto e o grava para a engenharia de features.

    Returns:
        ``0`` em caso de sucesso.

    Raises:
        FileNotFoundError: Se o dataset bruto versionado no DVC não existir.
    """
    context = StageContext.load()
    source = context.layout.raw_interactions
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset bruto '{source}' não encontrado. Rode 'dvc pull' ou "
            "'python scripts/download_dataset.py' antes."
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
    """Imprime o resumo da limpeza no log do stage."""
    print(f"[preprocess] interações brutas:  {raw_rows}")
    print(f"[preprocess] interações limpas:  {len(cleaned)}")
    print(f"[preprocess] usuários:           {cleaned['user_id'].nunique()}")
    print(f"[preprocess] itens:              {cleaned['item_id'].nunique()}")
    print(f"[preprocess] gravado em:         {target}")


if __name__ == "__main__":
    raise SystemExit(main())
