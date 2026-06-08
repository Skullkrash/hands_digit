"""
Étape 3 — Préparation du dataset.

- Génère les splits train/val/test si absents (seed 42, proportions 60/20/20).
- Génère dynamiquement le fichier config.yaml attendu par Ultralytics si absent.
"""

import random
import shutil
from pathlib import Path

import yaml

import config

SPLITS: list[str] = ["train", "val", "test"]

CLASS_NAMES: dict[int, str] = {
    0: "0_doigt",
    1: "1_doigt",
    2: "2_doigts",
    3: "3_doigts",
    4: "4_doigts",
    5: "5_doigts",
}


def _splits_exist(root: Path) -> bool:
    """Vérifie si les trois splits sont déjà présents et non vides."""
    for split in SPLITS:
        split_dir = root / "images" / split
        if not split_dir.exists() or not any(split_dir.iterdir()):
            return False
    return True


def _generate_splits(root: Path) -> None:
    """
    Génère aléatoirement les splits train/val/test depuis un dossier images/all/
    ou directement depuis images/ si les images sont à la racine.

    Args:
        root: Chemin racine du dataset.
    """
    random.seed(config.SPLIT_SEED)

    # Cherche les images à la racine de images/ (pas encore dans des sous-dossiers)
    images_root = root / "images"
    labels_root = root / "labels"

    all_images = list(images_root.glob("*.jpg")) + \
                 list(images_root.glob("*.jpeg")) + \
                 list(images_root.glob("*.png"))

    if not all_images:
        print("[prepare] Aucune image à splitter trouvée à la racine de images/.")
        return

    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)

    split_map: dict[str, list[Path]] = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split, files in split_map.items():
        img_dest = images_root / split
        lbl_dest = labels_root / split
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.move(str(img_path), img_dest / img_path.name)

            label_src = labels_root / (img_path.stem + ".txt")
            if label_src.exists():
                shutil.move(str(label_src), lbl_dest / label_src.name)

        print(f"[prepare] Split '{split}' : {len(files)} images")


def _generate_yaml(root: Path) -> Path:
    """
    Génère le fichier config.yaml attendu par Ultralytics si absent.

    Args:
        root: Chemin racine du dataset.

    Returns:
        Chemin vers le fichier yaml généré ou existant.
    """
    yaml_path = root / "config.yaml"

    if yaml_path.exists():
        print(f"[prepare] config.yaml déjà présent : {yaml_path}")
        return yaml_path

    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[prepare] config.yaml généré : {yaml_path}")
    return yaml_path


def prepare(dataset_dir: str = config.DATASET_DIR) -> Path:
    """
    Prépare le dataset pour l'entraînement.

    Args:
        dataset_dir: Chemin vers le dossier racine du dataset.

    Returns:
        Chemin vers le fichier config.yaml à passer à model.train().
    """
    root = Path(dataset_dir)

    if not _splits_exist(root):
        print("[prepare] Splits absents, génération en cours...")
        _generate_splits(root)
    else:
        print("[prepare] Splits déjà présents.")

    yaml_path = _generate_yaml(root)
    return yaml_path


if __name__ == "__main__":
    prepare()
