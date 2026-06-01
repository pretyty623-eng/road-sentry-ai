from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ROAD-SENTRY AI Service",
    description="AI service untuk deteksi kerusakan jalan menggunakan YOLOv8",
    version="1.0.0"
)

# CORS — izinkan request dari backend Node.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model sekali saat server start
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"

print(" Loading ROAD-SENTRY AI model...")
model = YOLO(str(MODEL_PATH))
print(" Model siap!")


# HELPER: Interpretasi hasil deteksi YOLO
def interpret_detections(detections: list) -> dict:
    pothole_list = [d for d in detections if d["class"] == "pothole"]
    crack_list   = [d for d in detections if d["class"] == "crack"]
    manhole_list = [d for d in detections if d["class"] == "manhole"]
    damage_list  = pothole_list + crack_list

    damage_detected = len(damage_list) > 0

    # Hitung severity
    if len(pothole_list) >= 2 or (len(pothole_list) >= 1 and len(crack_list) >= 1):
        severity_hint = "high"
    elif len(pothole_list) == 1 or len(crack_list) >= 2:
        severity_hint = "medium"
    elif len(crack_list) == 1:
        severity_hint = "low"
    else:
        severity_hint = None

    max_confidence = max(
        [d["confidence"] for d in damage_list], default=0
    )

    return {
        "damage_detected": damage_detected,
        "pothole_count":   len(pothole_list),
        "crack_count":     len(crack_list),
        "manhole_count":   len(manhole_list),
        "severity_hint":   severity_hint,
        "max_confidence":  round(max_confidence, 3)
    }


# ENDPOINT: Health Check
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":  "ok",
        "model":   "yolov8s-road-sentry",
        "classes": ["pothole", "crack", "manhole"]
    }


# ENDPOINT: Predict
@app.post("/predict", tags=["AI"])
async def predict(image: UploadFile = File(...)):
    # Validasi tipe file
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File harus berupa gambar"
        )

    try:
        # Baca gambar dari upload
        image_bytes = await image.read()
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Jalankan deteksi YOLO
        results = model.predict(pil_image, conf=0.25, iou=0.45, verbose=False)

        # Proses deteksi
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class":      model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox":       [round(x, 1) for x in box.xyxy[0].tolist()]
                })

        # Generate gambar dengan bounding box 
        annotated_array = results[0].plot()  
        annotated_rgb   = cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
        annotated_pil   = Image.fromarray(annotated_rgb)

        # Convert ke base64 agar bisa langsung ditampilkan di frontend
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=85)
        annotated_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Interpretasi hasil
        summary = interpret_detections(detections)

        return {
            "success":         True,
            "is_valid_road":   True,
            "damage_detected": summary["damage_detected"],
            "detections":      detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "damage_summary":  summary
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/", tags=["System"])
def root():
    return {
        "service": "ROAD-SENTRY AI",
        "docs":    "http://localhost:5001/docs",
        "health":  "http://localhost:5001/health"
    }