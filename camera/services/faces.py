import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBED_DIM = 512
# ArcFace cosine similarity on normalized buffalo_l embeddings.
MATCH_THRESHOLD = 0.45
MATCH_MARGIN = 0.08
MIN_DET_SCORE = 0.5
# Live labels can use a slightly softer face. Saving a capture is stricter.
LABEL_DET_SCORE = 0.55
LABEL_MIN_WIDTH = 48
LABEL_MAX_YAW = 40.0
SAVE_DET_SCORE = 0.62
SAVE_MIN_WIDTH = 64
SAVE_MAX_YAW = 30.0
MIN_BLUR = 45.0
BLACKLIST_THRESHOLD = 0.52
DEDUP_SIMILARITY = 0.55

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


def blur_score(image: np.ndarray) -> float:
    """Higher means a sharper crop. Flat frames score near zero."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_yaw(face) -> float | None:
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    values = np.asarray(pose, dtype=np.float32).reshape(-1)
    if values.size < 2:
        return None
    return float(values[1])


def assess_face(det_score: float, face_width: int, blur: float, yaw: float | None) -> tuple[bool, bool, float]:
    """Return (label_ok, save_ok, quality). save_ok is the bar for a recognition card."""
    yaw_abs = abs(float(yaw)) if yaw is not None else 0.0
    yaw_known = yaw is not None
    label_ok = (
        float(det_score) >= LABEL_DET_SCORE
        and int(face_width) >= LABEL_MIN_WIDTH
        and (not yaw_known or yaw_abs <= LABEL_MAX_YAW)
    )
    save_ok = (
        label_ok
        and float(det_score) >= SAVE_DET_SCORE
        and int(face_width) >= SAVE_MIN_WIDTH
        and float(blur) >= MIN_BLUR
        and (not yaw_known or yaw_abs <= SAVE_MAX_YAW)
    )
    if not save_ok:
        return label_ok, False, 0.0
    yaw_term = 1.0 if not yaw_known else max(0.0, 1.0 - yaw_abs / 90.0)
    quality = (
        float(det_score)
        * yaw_term
        * min(float(blur) / 200.0, 1.5)
        * min(int(face_width) / 120.0, 1.5)
    )
    return True, True, quality


def blacklist_alarm_ready(kind: str, score: float) -> bool:
    return kind == "blacklist" and float(score) >= BLACKLIST_THRESHOLD


@dataclass
class FaceHit:
    embedding: np.ndarray
    box: tuple[int, int, int, int]
    det_score: float
    yaw: float | None
    blur: float
    quality: float
    save_ok: bool
    label_ok: bool


def embed_image(bgr: np.ndarray) -> np.ndarray | None:
    faces = _faces(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda item: float(item.det_score))
    return _vector(face)


def embed_upper_body(frame: np.ndarray, coords) -> FaceHit | None:
    """Embed the face in the top of a person box. Bbox is in the original frame."""
    crop, origin = _upper_crop(frame, coords)
    if crop is None:
        return None
    faces = _faces(crop)
    if not faces:
        return None
    face = max(faces, key=lambda item: float(item.det_score))
    box = _frame_box(face.bbox, origin, frame.shape)
    face_crop = _crop_box(frame, _padded_box(box, frame.shape))
    blur = blur_score(face_crop)
    width = box[2] - box[0]
    yaw = face_yaw(face)
    det_score = float(face.det_score)
    label_ok, save_ok, quality = assess_face(det_score, width, blur, yaw)
    return FaceHit(
        embedding=_vector(face),
        box=box,
        det_score=det_score,
        yaw=yaw,
        blur=blur,
        quality=quality,
        save_ok=save_ok,
        label_ok=label_ok,
    )


def _padded_box(box, shape, pad: float = 0.18) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    left, top, right, bottom = box
    face_w = right - left
    face_h = bottom - top
    grow_x = int(face_w * pad)
    grow_y = int(face_h * pad)
    left = max(0, left - grow_x)
    top = max(0, top - grow_y)
    right = min(width, right + grow_x)
    bottom = min(height, bottom + grow_y)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _crop_box(frame: np.ndarray, box) -> np.ndarray:
    left, top, right, bottom = box
    return frame[top:bottom, left:right]


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


def best_person(embedding: np.ndarray, gallery: list[tuple]) -> tuple[object | None, float]:
    """Pick a person the same way as best_match. A close second name is rejected."""
    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    scored: list[tuple[float, object]] = []
    for person, stored in gallery:
        vector = np.asarray(stored, dtype=np.float32).reshape(-1)
        if vector.size != query.size or vector.size != EMBED_DIM:
            continue
        scored.append((similarity(query, vector), person))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, chosen = scored[0]
    if best_score < MATCH_THRESHOLD:
        return None, best_score
    if len(scored) > 1 and best_score - scored[1][0] < MATCH_MARGIN:
        return None, best_score
    return chosen, best_score


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
