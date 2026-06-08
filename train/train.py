"""
Étape 4 — Entraînement du modèle YOLO.

Configure et lance l'entraînement avec les hyperparamètres définis dans config.py.
Les métriques sont sauvegardées automatiquement par Ultralytics dans le dossier runs/.
"""

import random
from pathlib import Path

from ultralytics import YOLO

import config


def train(yaml_path: Path) -> Path:
    """
    Entraîne le modèle YOLO sur le dataset préparé.

    Args:
        yaml_path: Chemin vers le fichier config.yaml du dataset.

    Returns:
        Chemin vers les poids du meilleur modèle (best.pt).
    """
    seed = random.randint(0, 1000)
    print(f"[train] Seed utilisée : {seed}")

    model = YOLO(config.BASE_MODEL)

    model.train(
        data=str(yaml_path),

        # Training
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH,
        device=config.DEVICE,
        workers=config.WORKERS,

        # Optimisation
        optimizer=config.OPTIMIZER,
        lr0=config.LR0,
        cos_lr=config.COS_LR,

        seed=seed,
        cache="ram",

        # Augmentations
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        scale=config.SCALE,
        shear=config.SHEAR,
        fliplr=config.FLIPLR,
        flipud=config.FLIPUD,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,

        project=config.PROJECT,
        name=config.RUN_NAME,
    )

    best_weights = Path(config.PROJECT) / config.RUN_NAME / "weights" / "best.pt"
    print(f"[train] Entraînement terminé. Meilleurs poids : {best_weights}")
    return best_weights


if __name__ == "__main__":
    from prepare import prepare
    yaml_path = prepare()
    train(yaml_path)
