"""
Étape 2 — Validation du dataset.

Vérifie que :
- Les images ne sont pas corrompues.
- Les fichiers d'annotations existent pour chaque image.
- Les coordonnées des bounding boxes sont valides (dans [0, 1], pas de négatifs).
"""

from pathlib import Path

import cv2

import config


# Classe attendues
EXPECTED_CLASSES: set[int] = {0, 1, 2, 3, 4, 5}
SPLITS: list[str] = ["train", "val", "test"]


def validate(dataset_dir: str = config.DATASET_DIR) -> None:
    """
    Valide l'intégrité du dataset YOLO.

    Args:
        dataset_dir: Chemin vers le dossier racine du dataset.

    Raises:
        ValueError: Si des erreurs critiques sont détectées.
    """
    root = Path(dataset_dir)
    errors: list[str] = []
    total_images = 0
    total_labels = 0

    for split in SPLITS:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split

        if not images_dir.exists():
            print(f"[validate] Split '{split}' absent, ignoré.")
            continue

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

        for img_path in image_files:
            total_images += 1

            # Vérification image non corrompue
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Image corrompue : {img_path}")
                continue

            # Vérification annotation associée
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                errors.append(f"Annotation manquante pour : {img_path.name}")
                continue

            total_labels += 1

            # Vérification contenu de l'annotation
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"{label_path.name} ligne {i} : format invalide ({line.strip()!r})")
                    continue

                try:
                    cls = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path.name} ligne {i} : valeurs non numériques")
                    continue

                if cls not in EXPECTED_CLASSES:
                    errors.append(f"{label_path.name} ligne {i} : classe {cls} inattendue")

                for coord in coords:
                    if coord < 0 or coord > 1:
                        errors.append(f"{label_path.name} ligne {i} : coordonnée hors de [0, 1] ({coord})")

    print(f"[validate] {total_images} images vérifiées, {total_labels} annotations trouvées.")

    if errors:
        print(f"[validate] {len(errors)} erreur(s) détectée(s) :")
        for err in errors:
            print(f"  ✗ {err}")
        raise ValueError(f"Validation échouée avec {len(errors)} erreur(s). Corrigez le dataset avant de continuer.")

    print("[validate] Dataset valide ✓")


if __name__ == "__main__":
    validate()
