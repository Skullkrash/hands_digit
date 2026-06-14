import os
import yaml
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

DATASET_URL = "ul://antoine-germon/datasets/hands-digits"

def run_training():
    load_dotenv()
    api_key = os.environ.get("ULTRALYTICS_API_KEY")
    if not api_key:
        raise ValueError("ULTRALYTICS_API_KEY introuvable.")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_type = config.pop("model", "yolo26n.pt")

    model = YOLO(model_type)
    model.train(data=DATASET_URL, **config)

if __name__ == "__main__":
    run_training()