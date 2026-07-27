"""Baixa o dataset bruto para que ele possa ser versionado com o DVC.

Execute com ``poetry run python scripts/download_dataset.py``. O arquivo é
gravado em ``data/raw/ratings.csv`` e versionado pelo DVC, não pelo git.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from recsys.config import get_settings  # noqa: E402
from recsys.data.dataset import download_interactions  # noqa: E402


def main() -> int:
    """Baixa o dataset para o diretório bruto configurado.

    Returns:
        ``0`` em caso de sucesso.
    """
    settings = get_settings()
    target = download_interactions(settings.data_raw_dir)
    size_mb = target.stat().st_size / 1_000_000
    print(f"Dataset disponível em {target} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
