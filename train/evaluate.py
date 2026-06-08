"""
Étape 5 — Évaluation du modèle entraîné.

Évalue le meilleur modèle sur le jeu de test via model.val()
et affiche les métriques principales.
"""

from pathlib import Path

from ultralytics import YOLO

import config


def evaluate(weights_path: Path, yaml_path: Path) -> None:
    """
    Évalue le modèle sur le jeu de test et affiche les métriques.

    Args:
        weights_path: Chemin vers les poids du modèle à évaluer (best.pt).
        yaml_path: Chemin vers le fichier config.yaml du dataset.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    print(f"[evaluate] Chargement du modèle : {weights_path}")
    model = YOLO(str(weights_path))

    print("[evaluate] Évaluation sur le jeu de test...")
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
    )

    print("\n[evaluate] === Résultats ===")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Précision  : {metrics.box.mp:.4f}")
    print(f"  Rappel     : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    from prepare import prepare

    yaml_path = prepare()
    best_weights = Path(config.PROJECT) / config.RUN_NAME / "weights" / "best.pt"
    evaluate(best_weights, yaml_path)
