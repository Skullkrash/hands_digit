"""
Point d'entrée principal de la pipeline d'entraînement'.

Orchestre les étapes dans l'ordre :
    1. Extraction  — téléchargement du dataset
    2. Validation  — vérification de l'intégrité
    3. Préparation — splits + config.yaml
    4. Training    — entraînement du modèle
    5. Évaluation  — métriques sur le jeu de test

Usage :
    python pipeline.py
    python pipeline.py --dataset-id mon_dataset_id
    python pipeline.py --skip-extract   # si le dataset est déjà présent
"""

import argparse
import os
from pathlib import Path

import config
from extract import extract
from validate import validate
from prepare import prepare
from train import train
from evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement de modèle avec Ultralytics")

    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Identifiant du dataset Ultralytics HUB (override la config)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=config.DATASET_DIR,
        help="Dossier local où stocker le dataset",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Sauter l'étape d'extraction (dataset déjà présent localement)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Sauter l'entraînement (évaluer uniquement un modèle existant)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Chemin vers un best.pt existant (utilisé avec --skip-train)",
    )

    return parser.parse_args()


def main() -> None:
    """Exécute la pipeline complète."""
    args = parse_args()

    # Config écrasée si argument fourni
    if args.dataset_id:
        config.DATASET_ID = args.dataset_id

    print("\n------------------------------Pipeline d'entraînement — démarrage------------------------------\n")

    # Étape 1 — Extraction
    if not args.skip_extract:
        print("\n[1/5] Extraction du dataset...")
        extract(dataset_dir=args.dataset_dir)
    else:
        print("\n[1/5] Extraction ignorée (--skip-extract).")

    # Étape 2 — Validation
    print("\n[2/5] Validation du dataset...")
    validate(dataset_dir=args.dataset_dir)

    # Étape 3 — Préparation
    print("\n[3/5] Préparation du dataset...")
    yaml_path = prepare(dataset_dir=args.dataset_dir)

    # Étape 4 — Training
    if not args.skip_train:
        print("\n[4/5] Entraînement...")
        best_weights = train(yaml_path=yaml_path)
    else:
        if args.weights:
            best_weights = Path(args.weights)
        else:
            best_weights = Path(config.PROJECT) / config.RUN_NAME / "weights" / "best.pt"
        print(f"\n[4/5] Entraînement ignoré (--skip-train). Modèle : {best_weights}")

    # Étape 5 — Évaluation
    print("\n[5/5] Évaluation...")
    evaluate(weights_path=best_weights, yaml_path=yaml_path)

    print("\n------------------------------Pipeline terminée avec succès------------------------------\n")


if __name__ == "__main__":
    main()
