from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
# OpenCV SFace cosine similarity. Higher is more alike.
MATCH_THRESHOLD = 0.363

_detector = None
_recognizer = None


def _models_ready() -> None:
    if not YUNET.exists() or not SFACE.exists():
        raise RuntimeError("Face models are missing from the models folder.")


def detector():
    global _detector
    _models_ready()
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 5000)
    return _detector


def recognizer():
    global _recognizer
    _models_ready()
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return _recognizer


def detect_faces(bgr: np.ndarray):
    height, width = bgr.shape[:2]
    face_detector = detector()
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(bgr)
    if faces is None:
        return []
    return faces


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: float(face[2]) * float(face[3]))


def embed_face(bgr: np.ndarray, face) -> np.ndarray:
    rec = recognizer()
    aligned = rec.alignCrop(bgr, face)
    feature = rec.feature(aligned)
    return np.asarray(feature, dtype=np.float32).reshape(-1)


def decode_image_bytes(payload: bytes) -> np.ndarray | None:
    from io import BytesIO

    from PIL import Image, ImageOps

    try:
        image = Image.open(BytesIO(payload))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def embed_image(bgr: np.ndarray) -> np.ndarray | None:
    face = largest_face(detect_faces(bgr))
    if face is None:
        return None
    return embed_face(bgr, face)


def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def best_name(embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> str:
    best_score = -1.0
    chosen = ""
    for name, stored in gallery:
        score = similarity(embedding, stored)
        if score > best_score:
            best_score = score
            chosen = name
    if chosen and best_score >= MATCH_THRESHOLD:
        return chosen
    return ""


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float32).reshape(1, -1)
    second = np.asarray(right, dtype=np.float32).reshape(1, -1)
    return float(recognizer().match(first, second, cv2.FaceRecognizerSF_FR_COSINE))


def face_inside_box(face, box) -> bool:
    x, y, width, height = [float(value) for value in face[:4]]
    center_x = x + width / 2
    center_y = y + height / 2
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= center_x <= x2 and y1 <= center_y <= y2
