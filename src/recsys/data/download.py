"""Download of the raw MovieLens interaction dataset."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATINGS_MEMBER = "ml-latest-small/ratings.csv"


def download_interactions(
    destination: Path,
    url: str = MOVIELENS_URL,
    member: str = RATINGS_MEMBER,
    force: bool = False,
) -> Path:
    """Fetch the raw ratings file into ``destination``.

    Args:
        destination: Directory that receives ``ratings.csv``.
        url: Archive holding the dataset.
        member: Path of the ratings file inside the archive.
        force: Re-download even when the file is already present.

    Returns:
        Path of the downloaded ``ratings.csv``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "ratings.csv"
    if target.exists() and not force:
        return target
    with tempfile.TemporaryDirectory() as workdir:
        archive = _fetch(url, Path(workdir) / "dataset.zip")
        _extract_member(archive, member, target)
    return target


def _fetch(url: str, archive: Path) -> Path:
    """Download ``url`` to ``archive``.

    Args:
        url: Remote archive address.
        archive: Local path receiving the bytes.

    Returns:
        The local archive path.
    """
    with urllib.request.urlopen(url, timeout=120) as response:
        archive.write_bytes(response.read())
    return archive


def _extract_member(archive: Path, member: str, target: Path) -> None:
    """Extract a single member of ``archive`` into ``target``.

    Args:
        archive: Zip archive to read.
        member: Member path inside the archive.
        target: Destination file.

    Raises:
        KeyError: If ``member`` is absent from the archive.
    """
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise KeyError(f"'{member}' not found in {archive.name}.")
        with bundle.open(member) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
