import sys
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'best.pt'

try:
    from ultralytics import YOLO
    from PIL import Image

    # Load model
    model = YOLO(str(MODEL_PATH))

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({
            'success': False,
            'error': f'File gambar tidak ditemukan: {image_path}'
        }))
        sys.exit(1)

    results = model.predict(image_path, conf=0.25, iou=0.45, verbose=False)

    # Proses hasil
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'class':      model.names[int(box.cls)],
                'confidence': round(float(box.conf), 3),
                'bbox':       [round(x, 1) for x in box.xyxy[0].tolist()]
            })

    # Pisahkan per kelas
    pothole_list = [d for d in detections if d['class'] == 'pothole']
    crack_list   = [d for d in detections if d['class'] == 'crack']
    manhole_list = [d for d in detections if d['class'] == 'manhole']
    damage_list  = pothole_list + crack_list

    damage_detected = len(damage_list) > 0

    # Hitung severity
    if len(pothole_list) >= 2 or (len(pothole_list) >= 1 and len(crack_list) >= 1):
        severity_hint = 'high'
    elif len(pothole_list) == 1 or len(crack_list) >= 2:
        severity_hint = 'medium'
    elif len(crack_list) == 1:
        severity_hint = 'low'
    else:
        severity_hint = None

    max_confidence = max(
        [d['confidence'] for d in damage_list], default=0
    )

    print(json.dumps({
        'success':         True,
        'is_valid_road':   True,
        'damage_detected': damage_detected,
        'detections':      detections,
        'damage_summary': {
            'pothole_count':  len(pothole_list),
            'crack_count':    len(crack_list),
            'manhole_count':  len(manhole_list),
            'severity_hint':  severity_hint,
            'max_confidence': round(max_confidence, 3)
        }
    }))

except Exception as e:
    print(json.dumps({
        'success': False,
        'error':   str(e)
    }))
    sys.exit(1)