import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBED_DIM = 512
# ArcFace cosine similarity on normalized buffalo_l embeddings.
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.08
MIN_DET_SCORE = 0.5

_app = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()


def model_ready() -> bool:
    return _app is not None


def warm() -> None:
    _analysis()


def _analysis():
    global _app
    with _init_lock:
        if _app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                root=str(ROOT),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=MIN_DET_SCORE)
            _app = app
        return _app


def _faces(bgr: np.ndarray):
    app = _analysis()
    with _infer_lock:
        found = app.get(bgr)
    if not found:
        return []
    return [face for face in found if float(face.det_score) >= MIN_DET_SCORE]


def _vector(face) -> np.ndarray:
    return np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)


def embed_image(bgr: np.ndarray) -> np.ndarray | None:
    faces = _faces(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda item: float(item.det_score))
    return _vector(face)


def embed_upper_body(frame: np.ndarray, coords) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Embed the face in the top of a person box. Bbox is in the original frame."""
    crop, origin = _upper_crop(frame, coords)
    if crop is None:
        return None
    faces = _faces(crop)
    if not faces:
        return None
    face = max(faces, key=lambda item: float(item.det_score))
    return _vector(face), _frame_box(face.bbox, origin, frame.shape)


def _upper_crop(frame: np.ndarray, coords):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in coords]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    box_height = y2 - y1
    if box_height < 24 or x2 <= x1:
        return None, (0, 0, 1.0)
    crop = frame[y1 : y1 + max(1, int(box_height * 0.55)), x1:x2]
    if crop.size == 0:
        return None, (0, 0, 1.0)
    scale = 1.0
    if crop.shape[0] < 180:
        scale = 180 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, (x1, y1, scale)


def _frame_box(bbox, origin, shape) -> tuple[int, int, int, int]:
    ox, oy, scale = origin
    height, width = shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    left = int(round(ox + x1 / scale))
    top = int(round(oy + y1 / scale))
    right = int(round(ox + x2 / scale))
    bottom = int(round(oy + y2 / scale))
    left = min(max(0, left), width - 1)
    top = min(max(0, top), height - 1)
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    return left, top, right, bottom


def decode_image_bytes(payload: bytes) -> np.ndarray | None:
    from io import BytesIO

    from PIL import Image, ImageOps

    try:
        image = Image.open(BytesIO(payload))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def current_vector(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if vector.size != EMBED_DIM:
        return None
    return vector


def prepare_embedding(blob: bytes | None, photo_bytes: bytes | None) -> tuple[bytes | None, bool]:
    """Return (embedding bytes, should_save). Rebuild from the photo when the vector is the wrong length."""
    if current_vector(blob) is not None:
        return blob, False
    if not photo_bytes:
        return None, False
    image = decode_image_bytes(photo_bytes)
    if image is None:
        return None, False
    vector = embed_image(image)
    if vector is None or vector.size != EMBED_DIM:
        return None, False
    return to_bytes(vector), True


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float32).reshape(-1)
    second = np.asarray(right, dtype=np.float32).reshape(-1)
    first = first / (np.linalg.norm(first) + 1e-8)
    second = second / (np.linalg.norm(second) + 1e-8)
    return float(np.dot(first, second))


def best_match(embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    scored: list[tuple[float, str]] = []
    for name, stored in gallery:
        vector = np.asarray(stored, dtype=np.float32).reshape(-1)
        if vector.size != query.size or vector.size != EMBED_DIM:
            continue
        scored.append((similarity(query, vector), name))
    if not scored:
        return "", 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, chosen = scored[0]
    if best_score < MATCH_THRESHOLD:
        return "", best_score
    if len(scored) > 1 and best_score - scored[1][0] < MATCH_MARGIN:
        return "", best_score
    return chosen, best_score


def best_name(embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> str:
    name, _score = best_match(embedding, gallery)
    return name


def next_identity(
    votes: dict,
    current: str,
    current_score: float,
    name: str,
    score: float,
) -> tuple[str, float]:
    """Lock a name after two agreeing matches. A later pair with a higher score can replace it."""
    if not name:
        return current, current_score
    entry = votes.setdefault(name, {"count": 0, "score": 0.0})
    entry["count"] += 1
    entry["score"] = max(float(entry["score"]), float(score))
    if entry["count"] < 2:
        return current, current_score
    if not current or (name != current and entry["score"] > current_score):
        return name, entry["score"]
    return current, current_score
