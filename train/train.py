import os
import yaml
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

DATASET_URL = "https://platform.ultralytics.com/api/v1/datasets/hands-digits/download"
DATASET_DIR = Path("datasets/hands-digits")

def download_dataset(api_key: str) -> Path:
    if DATASET_DIR.exists():
        print(f"[dataset] Déjà présent dans {DATASET_DIR}, skip.")
        return DATASET_DIR

    print("[dataset] Téléchargement depuis Ultralytics Platform...")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(DATASET_URL, headers=headers, stream=True)
    response.raise_for_status()

    zip_path = Path("datasets/hands-digits.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("datasets/")
    zip_path.unlink()

    print(f"[dataset] Extrait dans {DATASET_DIR}")
    return DATASET_DIR

def run_training():
    load_dotenv()
    api_key = os.environ.get("ULTRALYTICS_API_KEY")
    if not api_key:
        raise ValueError("ULTRALYTICS_API_KEY introuvable.")

    dataset_dir = download_dataset(api_key)
    yaml_path = next(dataset_dir.rglob("*.yaml"))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_type = config.pop("model", "yolo26n.pt")

    model = YOLO(model_type)
    model.train(data=str(yaml_path), **config)

if __name__ == "__main__":
    run_training()