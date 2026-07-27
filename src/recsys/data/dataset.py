"""Download, leitura e limpeza das interações usuário-item."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATINGS_MEMBER = "ml-latest-small/ratings.csv"
RAW_COLUMN_MAP: dict[str, str] = {
    "userId": "user_id",
    "movieId": "item_id",
    "rating": "rating",
    "timestamp": "timestamp",
}
INTERACTION_COLUMNS: tuple[str, ...] = ("user_id", "item_id", "rating", "timestamp")


def download_interactions(destination: Path, force: bool = False) -> Path:
    """Baixa o arquivo bruto de avaliações para ``destination``.

    Args:
        destination: Diretório que recebe o ``ratings.csv``.
        force: Baixa novamente mesmo que o arquivo já exista.

    Returns:
        Caminho do ``ratings.csv`` baixado.
    """
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "ratings.csv"
    if target.exists() and not force:
        return target
    with tempfile.TemporaryDirectory() as workdir:
        archive = Path(workdir) / "dataset.zip"
        with urllib.request.urlopen(MOVIELENS_URL, timeout=120) as response:
            archive.write_bytes(response.read())
        _extract(archive, target)
    return target


def load_interactions(path: Path) -> pd.DataFrame:
    """Lê o arquivo bruto e normaliza os nomes das colunas.

    Args:
        path: Localização do ``ratings.csv`` bruto.

    Returns:
        Frame com as colunas ``user_id``, ``item_id``, ``rating`` e
        ``timestamp``.

    Raises:
        ValueError: Se o arquivo bruto não tiver alguma coluna esperada.
    """
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_COLUMN_MAP) - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas ausentes no dataset bruto: {', '.join(missing)}.")
    return frame.rename(columns=RAW_COLUMN_MAP).loc[:, list(INTERACTION_COLUMNS)]


def clean_interactions(
    frame: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Remove linhas incompletas, duplicatas e entidades pouco observadas.

    Args:
        frame: Frame de interações normalizado.
        min_user_interactions: Mínimo de interações para manter um usuário.
        min_item_interactions: Mínimo de interações para manter um item.

    Returns:
        Frame limpo, ordenado cronologicamente e com índice reiniciado.
    """
    cleaned = frame.dropna(subset=list(INTERACTION_COLUMNS))
    cleaned = cleaned.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    cleaned = _filter_rare(cleaned, "item_id", min_item_interactions)
    cleaned = _filter_rare(cleaned, "user_id", min_user_interactions)
    return cleaned.sort_values("timestamp").reset_index(drop=True)


def _extract(archive: Path, target: Path) -> None:
    """Extrai o arquivo de avaliações de dentro do pacote compactado.

    Args:
        archive: Arquivo zip baixado.
        target: Arquivo de destino.

    Raises:
        KeyError: Se o pacote não contiver o arquivo de avaliações.
    """
    with zipfile.ZipFile(archive) as bundle:
        if RATINGS_MEMBER not in bundle.namelist():
            raise KeyError(f"'{RATINGS_MEMBER}' não encontrado em {archive.name}.")
        with bundle.open(RATINGS_MEMBER) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)


def _filter_rare(frame: pd.DataFrame, column: str, minimum: int) -> pd.DataFrame:
    """Mantém apenas as linhas cujo identificador aparece o bastante.

    Args:
        frame: Frame a filtrar.
        column: Coluna com o identificador da entidade.
        minimum: Número mínimo de ocorrências exigido.

    Returns:
        O frame filtrado.
    """
    if minimum <= 1:
        return frame
    counts = frame[column].value_counts()
    return frame[frame[column].isin(counts[counts >= minimum].index)]
