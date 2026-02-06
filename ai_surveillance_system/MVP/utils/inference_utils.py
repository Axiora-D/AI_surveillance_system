import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from typing import Any

# TensorFlow is optional: on Windows/Python >= 3.13 it's often unavailable.
# We keep the app runnable (YOLO-only) when TF isn't installed.
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

# Constants
IMG_SIZE = 224
CLASS_MAP = {0: "Male", 1: "Female"}


# -------- Model Loading --------
def load_keras_model(model_path: str, threshold_path: Optional[str] = None) -> Tuple[Any, float]:
    """
    Load the gender classifier and a decision threshold.
    If the threshold file is missing or invalid, default to 0.5.
    """
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed. Gender classification is unavailable in this environment."
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Keras model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)

    threshold = 0.5
    if threshold_path:
        try:
            threshold = float(Path(threshold_path).read_text().strip())
        except Exception:
            # Fallback to default threshold without crashing
            threshold = 0.5

    return model, threshold


def load_yolo_model(yolo_path: str = "yolo11n.pt") -> YOLO:
    """
    Load the YOLO model used as the face/person detector.
    """
    if not Path(yolo_path).exists():
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
    return YOLO(yolo_path)


# -------- Image Helpers --------
def crop_from_box(img_array: np.ndarray, box: List[float]) -> Optional[np.ndarray]:
    """
    Crop an image using an xyxy box. Returns None if the box is invalid after clipping.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img_array.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return img_array[y1:y2, x1:x2]


def preprocess_for_classifier(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Prepare a cropped BGR image for the Keras classifier.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
    crop_resized = crop_resized.astype("float32") / 255.0
    return np.expand_dims(crop_resized, axis=0)  # shape (1, H, W, 3)


def draw_annotations(pil_image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Draw boxes and labels onto a PIL image.
    """
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']}:{det['gender_score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text_pos = (x1, max(0, y1 - 18))
        draw.text(text_pos, label, fill="red", font=font)
    return pil_image

