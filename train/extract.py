"""
Étape 1 — Extraction du dataset depuis Ultralytics HUB.

Télécharge les images et annotations associées via le SDK Ultralytics,
en utilisant l'identifiant de dataset configuré dans config.py / .env.
"""

import os
import shutil
from pathlib import Path

from ultralytics import settings
from ultralytics.hub import login
from ultralytics.hub.utils import HUBDatasetStats

import config


def extract(dataset_dir: str = config.DATASET_DIR) -> Path:
    """
    Télécharge le dataset depuis Ultralytics HUB et le place dans dataset_dir.

    Args:
        dataset_dir: Chemin local où stocker le dataset.

    Returns:
        Chemin absolu vers le dossier du dataset téléchargé.

    Raises:
        RuntimeError: Si le téléchargement échoue.
    """
    os.environ["ULTRALYTICS_API_KEY"] = config.ULTRALYTICS_API_KEY

    dest = Path(dataset_dir)

    if dest.exists():
        print(f"[extract] Dataset déjà présent dans {dest}, téléchargement ignoré.")
        return dest.resolve()

    print(f"[extract] Connexion à Ultralytics HUB...")
    login(config.ULTRALYTICS_API_KEY)

    print(f"[extract] Téléchargement du dataset '{config.DATASET_ID}'...")

    try:
        from ultralytics.hub import dataset_download
        dataset_download(config.DATASET_ID, dest=str(dest))
    except Exception as e:
        raise RuntimeError(f"Échec du téléchargement du dataset : {e}") from e

    print(f"[extract] Dataset extrait dans : {dest.resolve()}")
    return dest.resolve()


if __name__ == "__main__":
    extract()
