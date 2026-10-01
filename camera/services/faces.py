import os
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
LABEL_MIN_WIDTH = 32
LABEL_MAX_YAW = 160.0
SAVE_DET_SCORE = 0.62
SAVE_MIN_WIDTH = 64
SAVE_MAX_YAW = 30.0
MIN_BLUR = 45.0
BLACKLIST_THRESHOLD = 0.52
DEDUP_SIMILARITY = 0.55
SAMPLE_SAME = 0.92
SAMPLE_LIMIT = 12
LIVE_DET_SIZE = (320, 320)
ENROLL_DET_SIZE = (640, 640)
_ORT_THREADS = 2

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
            os.environ.setdefault("OMP_NUM_THREADS", str(_ORT_THREADS))
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            options = ort.SessionOptions()
            options.intra_op_num_threads = _ORT_THREADS
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(ROOT),
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
            app.prepare(
                ctx_id=-1,
                det_size=[LIVE_DET_SIZE, ENROLL_DET_SIZE],
                det_thresh=MIN_DET_SCORE,
            )
            _app = app
        return _app


def _detect(image: np.ndarray, det_size: tuple[int, int]):
    app = _analysis()
    with _infer_lock:
        bboxes, kpss = app.det_model.detect(image, input_size=det_size, max_num=1)
    if bboxes is None or len(bboxes) == 0:
        return None
    index = int(np.argmax(bboxes[:, 4]))
    if float(bboxes[index, 4]) < MIN_DET_SCORE:
        return None
    kps = None if kpss is None else kpss[index]
    return bboxes[index], kps


def _recognize(image: np.ndarray, bbox, kps) -> np.ndarray | None:
    if kps is None:
        return None
    from insightface.app.common import Face

    face = Face(
        bbox=np.asarray(bbox[:4], dtype=np.float32),
        kps=np.asarray(kps, dtype=np.float32),
        det_score=float(bbox[4]),
    )
    app = _analysis()
    with _infer_lock:
        app.models["recognition"].get(image, face)
    if face.normed_embedding is None:
        return None
    vector = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBED_DIM:
        return None
    return vector


def blur_score(image: np.ndarray) -> float:
    """Higher means a sharper crop. Flat frames score near zero."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def yaw_from_kps(kps) -> float | None:
    """Estimate yaw in degrees from the detector's 5 points. Frontal is near zero."""
    if kps is None:
        return None
    points = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    left_eye, right_eye, nose = points[0], points[1], points[2]
    distance = float(np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
    if distance < 1.0:
        return None
    mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    offset = (float(nose[0]) - mid_x) / distance
    return offset * 70.0


def face_yaw(face) -> float | None:
    yaw = yaw_from_kps(getattr(face, "kps", None))
    if yaw is not None:
        return yaw
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
    photo: np.ndarray


def embed_image(bgr: np.ndarray) -> np.ndarray | None:
    detected = _detect(bgr, ENROLL_DET_SIZE)
    if detected is None:
        return None
    bbox, kps = detected
    return _recognize(bgr, bbox, kps)


def take_head(frame: np.ndarray, coords):
    """Copy only the head region the face worker needs."""
    crop, origin = _upper_crop(frame, coords)
    if crop is None:
        return None
    return crop.copy(), origin, tuple(frame.shape)


def embed_head(crop: np.ndarray, origin, frame_shape) -> FaceHit | None:
    """Detect on a head crop at 320. A turned face can still be named; only a clear face is saved."""
    detected = _detect(crop, LIVE_DET_SIZE)
    if detected is None:
        return None
    bbox, kps = detected
    box = _frame_box(bbox, origin, frame_shape)
    local = _frame_box(bbox, (0, 0, 1.0), crop.shape)
    photo = _crop_box(crop, _padded_box(local, crop.shape))
    blur = blur_score(photo)
    width = box[2] - box[0]
    yaw = yaw_from_kps(kps)
    det_score = float(bbox[4])
    label_ok, save_ok, quality = assess_face(det_score, width, blur, yaw)
    if not label_ok:
        return None
    embedding = _recognize(crop, bbox, kps)
    if embedding is None:
        return None
    return FaceHit(
        embedding=embedding,
        box=box,
        det_score=det_score,
        yaw=yaw,
        blur=blur,
        quality=quality,
        save_ok=save_ok,
        label_ok=True,
        photo=photo,
    )


def embed_upper_body(frame: np.ndarray, coords) -> FaceHit | None:
    """Embed the face in the top of a person box. Bbox is in the original frame."""
    taken = take_head(frame, coords)
    if taken is None:
        return None
    crop, origin, shape = taken
    return embed_head(crop, origin, shape)


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


def box_on_crop(frame_box, origin, crop_shape) -> tuple[int, int, int, int]:
    """Map a full-frame face box back onto the head crop."""
    ox, oy, scale = origin
    scale = scale or 1.0
    x1, y1, x2, y2 = frame_box
    height, width = crop_shape[:2]
    left = int(round((x1 - ox) * scale))
    top = int(round((y1 - oy) * scale))
    right = int(round((x2 - ox) * scale))
    bottom = int(round((y2 - oy) * scale))
    left = min(max(0, left), max(0, width - 1))
    top = min(max(0, top), max(0, height - 1))
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    return left, top, right, bottom


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
    return crop, (x1, y1, 1.0)


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


def _identity(label):
    pk = getattr(label, "pk", None)
    if pk is not None:
        return ("id", pk)
    return ("value", label)


def _best_by_identity(embedding: np.ndarray, gallery: list[tuple]) -> list[tuple[float, object]]:
    """Keep each person's best sample, then rank people."""
    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    best: dict = {}
    for label, stored in gallery:
        vector = np.asarray(stored, dtype=np.float32).reshape(-1)
        if vector.size != query.size or vector.size != EMBED_DIM:
            continue
        score = similarity(query, vector)
        key = _identity(label)
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, label)
    return sorted(best.values(), key=lambda item: item[0], reverse=True)


def _choose(ranked: list[tuple[float, object]]) -> tuple[object | None, float]:
    if not ranked:
        return None, 0.0
    best_score, chosen = ranked[0]
    if best_score < MATCH_THRESHOLD:
        return None, best_score
    if len(ranked) > 1 and best_score - ranked[1][0] < MATCH_MARGIN:
        return None, best_score
    return chosen, best_score


def best_match(embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
    chosen, score = _choose(_best_by_identity(embedding, gallery))
    return (chosen or ""), score


def best_name(embedding: np.ndarray, gallery: list[tuple[str, np.ndarray]]) -> str:
    name, _score = best_match(embedding, gallery)
    return name


def best_person(embedding: np.ndarray, gallery: list[tuple]) -> tuple[object | None, float]:
    """Pick a person by their best sample. A close second person is rejected."""
    return _choose(_best_by_identity(embedding, gallery))


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
