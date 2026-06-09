from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ultralytics import YOLO

import cv2
import numpy as np

app = FastAPI()

model = YOLO("model/best.pt")

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

    results = model(frame, verbose=False)

    plotted_frame = results[0].plot()

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detections.append({
            "label": results[0].names[cls],
        })

    _, encoded = cv2.imencode(".jpg", plotted_frame)
    jpg_bytes = encoded.tobytes()

    labels = [det["label"] for det in detections]

    response = Response(content=jpg_bytes, media_type="image/jpeg")
    response.headers["X-Detections"] = ",".join(labels)

    return response