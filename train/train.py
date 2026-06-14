import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO, hub

def run_training():
    load_dotenv()
    api_key = os.environ.get("ULTRALYTICS_API_KEY")
    
    if not api_key:
        raise ValueError("Erreur : La variable ULTRALYTICS_API_KEY est introuvable dans l'environnement.")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_type = config.pop("model", "yolo26n.pt")

    hub.login(api_key)

    # Télécharge le dataset HUB et récupère le chemin du yaml
    dataset_path = hub.download_dataset(
        "https://platform.ultralytics.com/antoine-germon/datasets/hands-digits",
        cache=True
    )
    yaml_path = next(Path(dataset_path).rglob("*.yaml"))

    model = YOLO(model_type)
    model.train(
        data=str(yaml_path),
        **config
    )

if __name__ == "__main__":
    run_training()