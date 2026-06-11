import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import torch

import cv2
import numpy as np

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("ul://antoine-germon/train/exp-8").to(DEVICE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    data = await file.read()

    np_img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False, imgsz=640, device=DEVICE, max_det=2)

    detections = []
    print(results)

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "x": round(x1),
            "y": round(y1),
            "w": round(x2 - x1),
            "h": round(y2 - y1),
            "class_id": cls,
            "label": results[0].names[cls],
            "confidence": round(conf, 3)
        })
        print(detections)

    return {
        "detections": detections
    }