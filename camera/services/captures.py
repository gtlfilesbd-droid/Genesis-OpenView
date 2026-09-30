from datetime import timedelta

import cv2
import numpy as np
from django.core.files.base import ContentFile
from django.utils import timezone

from camera.models import FaceCapture

from .faces import DEDUP_SIMILARITY, current_vector, similarity, to_bytes

DEDUP_WINDOW = timedelta(minutes=3)
LINK_WINDOW = timedelta(minutes=10)


def stamp_face(crop: np.ndarray, when=None) -> np.ndarray:
    """Put the capture time on a bar under the face so the photo itself carries it."""
    moment = when or timezone.now()
    label = timezone.localtime(moment).strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), _baseline = cv2.getTextSize(label, font, scale, thickness)
    width = max(int(crop.shape[1]), text_width + 12)
    bar = text_height + 14
    canvas = np.zeros((int(crop.shape[0]) + bar, width, 3), dtype=np.uint8)
    canvas[:, :] = (16, 18, 22)
    canvas[0 : crop.shape[0], 0 : crop.shape[1]] = crop
    cv2.putText(
        canvas,
        label,
        (6, int(crop.shape[0]) + text_height + 6),
        font,
        scale,
        (232, 236, 240),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return b""
    return encoded.tobytes()


def _recent_duplicate(camera_number: int, embedding: np.ndarray, now):
    since = now - DEDUP_WINDOW
    recent = FaceCapture.objects.filter(camera_number=camera_number, created_at__gte=since).order_by("-created_at")[:30]
    for capture in recent:
        stored = current_vector(bytes(capture.embedding) if capture.embedding else None)
        if stored is not None and similarity(embedding, stored) >= DEDUP_SIMILARITY:
            return capture
    return None


def save_visit(
    camera_number: int,
    track_id: int,
    embedding: np.ndarray,
    crop: np.ndarray,
    det_score: float,
    quality: float,
    person,
    matched_name: str,
    match_score: float | None,
    known_id: int | None,
) -> int | None:
    """Keep one card per visit. A sharper frame replaces the photo; a new person does not."""
    if crop is None or crop.size == 0:
        return known_id
    now = timezone.now()
    row = None
    if known_id:
        row = FaceCapture.objects.filter(pk=known_id).first()
    if row is None:
        row = _recent_duplicate(camera_number, embedding, now)
    blob = to_bytes(embedding)
    if row is None:
        row = FaceCapture(
            camera_number=camera_number,
            track_id=int(track_id),
            embedding=blob,
            det_score=float(det_score),
            quality=float(quality),
            match_score=match_score,
            matched_name=matched_name or "",
            person=person,
        )
        jpeg = _jpeg(stamp_face(crop, now))
        if not jpeg:
            return None
        row.face_crop.save(f"{now.strftime('%Y%m%d-%H%M%S')}-{int(track_id)}.jpg", ContentFile(jpeg), save=False)
        row.save()
        return row.pk

    changed = ["track_id"]
    row.track_id = int(track_id)
    if float(quality) > float(row.quality):
        jpeg = _jpeg(stamp_face(crop, now))
        if jpeg:
            row.face_crop.save(
                f"{now.strftime('%Y%m%d-%H%M%S')}-{int(track_id)}.jpg",
                ContentFile(jpeg),
                save=False,
            )
            row.embedding = blob
            row.det_score = float(det_score)
            row.quality = float(quality)
            changed.extend(["face_crop", "embedding", "det_score", "quality"])
    if person is not None:
        row.person = person
        row.matched_name = matched_name or person.name
        row.match_score = match_score
        changed.extend(["person", "matched_name", "match_score"])
    row.save(update_fields=list(dict.fromkeys(changed)))
    return row.pk


def link_same_face(capture: FaceCapture, person) -> None:
    """Attach other recent looks of this face to the person just listed."""
    vector = current_vector(bytes(capture.embedding) if capture.embedding else None)
    if vector is None:
        return
    since = timezone.now() - LINK_WINDOW
    others = FaceCapture.objects.filter(created_at__gte=since).exclude(pk=capture.pk)
    for other in others:
        if other.person_id and other.person_id != person.pk:
            continue
        stored = current_vector(bytes(other.embedding) if other.embedding else None)
        if stored is None or similarity(vector, stored) < DEDUP_SIMILARITY:
            continue
        other.person = person
        other.matched_name = person.name
        other.save(update_fields=["person", "matched_name"])
