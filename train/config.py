"""
Configuration de la pipeline d'entraînement.

Les valeurs sensibles (clé API) sont lues depuis les variables d'environnement
ou un fichier .env à la racine du dossier training/.

Exemple de fichier .env :
    ULTRALYTICS_API_KEY=votre_cle_ici
    DATASET_ID=ul://votre_utilisateur/datasets/votre_dataset
"""

from decouple import config


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

ULTRALYTICS_API_KEY: str = config("ULTRALYTICS_API_KEY", default="ul_48ae8ce22c49669790c6e29673d424ebe9adeb9b")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_ID: str = config("DATASET_ID", default="ul://antoine-germon/datasets/hands-digits")

DATASET_DIR: str = config("DATASET_DIR", default="datasets/hands-digits")

# Proportions des splits si non fournis par la plateforme
SPLIT_SEED: int = 42
TRAIN_RATIO: float = 0.6
VAL_RATIO: float = 0.2
TEST_RATIO: float = 0.2

# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

BASE_MODEL: str = config("BASE_MODEL", default="yolo26m.pt")

# ---------------------------------------------------------------------------
# Hyperparamètres d'entraînement
# ---------------------------------------------------------------------------

EPOCHS: int = 120
PATIENCE: int = 10
IMG_SIZE: int = 640
BATCH: int = 8
DEVICE: int | str = 0
WORKERS: int = 6

OPTIMIZER: str = "AdamW"
LR0: float = 0.003
COS_LR: bool = True

# Augmentations
HSV_H: float = 0.03
HSV_S: float = 0.8
HSV_V: float = 0.6
DEGREES: float = 15.0
SCALE: float = 0.5
SHEAR: float = 2.0
FLIPLR: float = 0.5
FLIPUD: float = 0.0
MOSAIC: float = 0.0
MIXUP: float = 0.0

# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------

PROJECT: str = "runs/train"
RUN_NAME: str = "sharp_hands_digit"
