"""
Étape 1 — Extraction du dataset depuis Ultralytics HUB.

Télécharge et prépare le dataset via l'URI Ultralytics configurée dans
config.py / .env.
"""

import os
from pathlib import Path

from ultralytics.data.utils import convert_ndjson_to_yolo_if_needed

import config


def _dataset_uri() -> str:
    """Retourne l'URI Ultralytics à résoudre.

    Supporte un override explicite via DATASET_URI, puis DATASET_ID si celui-ci
    est déjà au format ``ul://...``.
    """
    dataset_uri = getattr(config, "DATASET_URI", None) or config.DATASET_ID

    if not dataset_uri.startswith("ul://"):
        raise ValueError(
            "Le dataset doit être fourni sous la forme 'ul://user/datasets/slug' "
            "(par exemple: 'ul://antoine-germon/datasets/hands-digits')."
        )

    return dataset_uri


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

    dataset_uri = _dataset_uri()

    print(f"[extract] Conversion de l'export Ultralytics '{dataset_uri}'...")

    try:
        yaml_path = convert_ndjson_to_yolo_if_needed(dataset_uri)
    except Exception as e:
        raise RuntimeError(f"Échec de la préparation du dataset : {e}") from e

    extracted_root = Path(yaml_path).parent
    print(f"[extract] Dataset prêt dans : {extracted_root.resolve()}")
    return extracted_root.resolve()


if __name__ == "__main__":
    extract()
