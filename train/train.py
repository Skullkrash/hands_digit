import os
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO, hub

def run_training():
    load_dotenv()
    api_key = os.environ.get("ULTRALYTICS_API_KEY")
    
    if not api_key:
        raise ValueError("Erreur : La variable ULTRALYTICS_API_KEY est introuvable dans l'environnement.")

    # Charger les paramètres d'entraînement depuis config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_type = config.pop("model", "yolo26n.pt")
    
    hub.login(api_key)
    model = YOLO(model_type)
    
    model.train(
        data="https://platform.ultralytics.com/antoine-germon/datasets/hands-digits",
        **config
    )

if __name__ == "__main__":
    run_training()
