"""Download the public PCOS dataset from Kaggle without requiring a token."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
ARCHIVE = RAW_DIR / "pcos-dataset.zip"
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "prasoonkottarathil/polycystic-ovary-syndrome-pcos"
)


def main() -> None:
    """Download and safely extract the two source files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Baixando dataset público do Kaggle...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        with ARCHIVE.open("wb") as destination:
            shutil.copyfileobj(response, destination)

    with zipfile.ZipFile(ARCHIVE) as zipped:
        for member in zipped.infolist():
            target = (RAW_DIR / member.filename).resolve()
            if RAW_DIR.resolve() not in target.parents:
                raise ValueError(f"Caminho inseguro no arquivo ZIP: {member.filename}")
        zipped.extractall(RAW_DIR)

    ARCHIVE.unlink()
    print(f"Dataset disponível em: {RAW_DIR}")


if __name__ == "__main__":
    main()
