import os
import sys
from dotenv import load_dotenv

import train

def step_train() -> None:
    """Déclenche l'entraînement géré par train.py"""
    print("\n── Entraînement & Synchronisation HUB ─────────────")
    train.run_training()


def main() -> None:
    # Chargement des variables d'environnement
    load_dotenv()

    # Pas besoin d'extraction / validation avant car elle est gérée automatiquement par Ultralytics
    step_train()

    print("\n── Entraînement terminé avec succès ! ─────────────────────")


if __name__ == "__main__":
    main()