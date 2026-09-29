import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .dwell import DwellTracker
from .geom import point_in_polygon
from .rtsp import ROOT, base_rtsp_url, parse_channel, stream_url

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")


def _placeholder(text: str) -> bytes:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(image, text, (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else b""


def _draw_label(image, x: int, y: int, text: str, known: bool) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - 10)
    color = (46, 140, 60) if known else (40, 160, 200)
    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 8, top + text_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, top + text_height + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _embedding_for_person(frame, coords, faces):
    from .faces import detect_faces, embed_face, face_inside_box, largest_face

    face = next((item for item in faces if face_inside_box(item, coords)), None)
    if face is not None:
        try:
            return embed_face(frame, face)
        except cv2.error:
            return None
    crop = _head_crop(frame, coords)
    if crop is None:
        return None
    try:
        crop_face = largest_face(detect_faces(crop))
    except Exception:
        return None
    if crop_face is None:
        return None
    try:
        return embed_face(crop, crop_face)
    except cv2.error:
        return None


def _head_crop(frame, coords):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in coords]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    box_height = y2 - y1
    if box_height < 24 or x2 <= x1:
        return None
    head = frame[y1 : y1 + max(1, int(box_height * 0.45)), x1:x2]
    if head.size == 0:
        return None
    if head.shape[0] < 160:
        scale = 160 / head.shape[0]
        head = cv2.resize(head, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return head


def _draw_zone(image, points) -> None:
    height, width = image.shape[:2]
    polygon = np.array(
        [[int(float(x) * width), int(float(y) * height)] for x, y in points],
        dtype=np.int32,
    )
    if len(polygon) >= 2:
        cv2.polylines(image, [polygon], True, (40, 200, 240), 2)


class Engine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._model = None
        self._generation = 0
        self._defaults_ready = False
        self._dwell = DwellTracker()
        self._names: dict[int, str] = {}
        self._name_try_at: dict[int, float] = {}
        self._gallery: list[tuple[str, np.ndarray]] = []
        self._gallery_at = 0.0
        self.running = False
        self.starting = False
        self.camera_number = 1
        self.stream = "sub"
        self.channel = ""
        self.device = "cpu"
        self.error = ""
        self.persons: list[dict] = []
        self.latest_jpeg = b""
        self.raw_jpeg = b""
        self.raw_at = 0.0
        self.last_alarm: dict | None = None

    def ensure_defaults(self) -> None:
        if self._defaults_ready:
            return
        self._defaults_ready = True
        try:
            camera, stream = parse_channel(base_rtsp_url())
        except RuntimeError as exc:
            self.error = str(exc)
            return
        if not self.running:
            self.camera_number = camera
            self.stream = stream

    def start(self, camera_number: int, stream: str) -> None:
        self._stop.set()
        self._generation += 1
        generation = self._generation
        self._stop = threading.Event()
        stop_event = self._stop
        self._dwell = DwellTracker()
        self._names = {}
        self._name_try_at = {}
        with self._lock:
            self.camera_number = int(camera_number)
            self.stream = stream if stream in ("sub", "main") else "sub"
            self.starting = True
            self.running = False
            self.error = ""
            self.persons = []
            self.latest_jpeg = b""
            self.channel = ""
        self._thread = threading.Thread(
            target=self._run,
            args=(generation, stop_event),
            name="camera-engine",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._generation += 1
        with self._lock:
            self.running = False
            self.starting = False
            self.persons = []
            self.latest_jpeg = b""
            self.raw_jpeg = b""

    def status(self) -> dict:
        with self._lock:
            alarm = None
            if self.last_alarm and time.time() - self.last_alarm["at"] < 30:
                alarm = dict(self.last_alarm)
                alarm["active"] = True
            elif self.last_alarm:
                alarm = dict(self.last_alarm)
                alarm["active"] = False
            return {
                "running": self.running,
                "starting": self.starting,
                "camera": self.camera_number,
                "stream": self.stream,
                "channel": self.channel,
                "device": self.device,
                "error": self.error,
                "persons": list(self.persons),
                "alarm": alarm,
            }

    def current_jpeg(self) -> bytes:
        with self._lock:
            if self.latest_jpeg and (self.running or self.starting):
                return self.latest_jpeg
            if self.error:
                return _placeholder(self.error[:70])
            if self.starting:
                return _placeholder("Connecting...")
            return _placeholder("Camera stopped")

    def raw_frame(self, camera_number: int) -> bytes:
        with self._lock:
            fresh = self.raw_jpeg and time.time() - self.raw_at < 3
            same_camera = self.camera_number == camera_number and self.running
            if fresh and same_camera:
                return self.raw_jpeg
        return b""

    def _model_get(self):
        if self._model is None:
            self._model = YOLO(str(ROOT / "yolo11n.pt"))
        self._model.predictor = None
        return self._model

    def _run(self, generation: int, stop_event: threading.Event) -> None:
        from django.db import close_old_connections

        try:
            url, channel = stream_url(self.camera_number, self.stream)
        except Exception:
            self._fail(generation, "Set RTSP_URL in .env")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with self._lock:
            if generation != self._generation:
                return
            self.channel = channel
            self.device = device
        seen_frame = False
        while not stop_event.is_set() and generation == self._generation:
            try:
                model = self._model_get()
                results = model.track(
                    source=url,
                    stream=True,
                    persist=True,
                    classes=[0],
                    imgsz=640,
                    device=device,
                    verbose=False,
                )
                with self._lock:
                    if generation == self._generation:
                        self.running = True
                        self.starting = False
                        self.error = ""
                for result in results:
                    if stop_event.is_set() or generation != self._generation:
                        break
                    seen_frame = True
                    try:
                        self._handle_result(result, generation)
                    except Exception as exc:
                        print("frame error:", type(exc).__name__)
            except Exception as exc:
                print("camera engine error:", type(exc).__name__)
            if stop_event.is_set() or generation != self._generation:
                break
            with self._lock:
                if generation == self._generation:
                    self.running = False
                    self.starting = True
                    if not seen_frame:
                        self.error = "Could not read the camera."
            time.sleep(2)
        close_old_connections()
        with self._lock:
            if generation == self._generation:
                self.running = False
                self.starting = False

    def _fail(self, generation: int, message: str) -> None:
        with self._lock:
            if generation == self._generation:
                self.error = message
                self.running = False
                self.starting = False

    def _load_zone(self):
        from camera.models import Zone

        return Zone.objects.filter(camera_number=self.camera_number, active=True).first()

    def _handle_result(self, result, generation: int) -> None:
        from django.db import close_old_connections

        close_old_connections()
        if generation != self._generation or result.orig_img is None:
            return
        frame = result.orig_img.copy()
        plotted = result.plot()
        height, width = frame.shape[:2]
        boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            xyxy = result.boxes.xyxy.cpu().numpy()
            boxes = list(zip(ids, xyxy))

        zone = self._load_zone()
        now = time.monotonic()
        present: dict[int, bool] = {}
        polygon = []
        dwell_seconds = 60
        if zone and len(zone.points) >= 3:
            polygon = [(float(point[0]), float(point[1])) for point in zone.points]
            dwell_seconds = zone.dwell_seconds
            _draw_zone(plotted, polygon)
            for track_id, coords in boxes:
                x1, y1, x2, y2 = coords
                # Feet: bottom center of the person box.
                present[track_id] = point_in_polygon((x1 + x2) / 2 / width, y2 / height, polygon)
        else:
            present = {track_id: False for track_id, _ in boxes}
        alarm_ids = self._dwell.update(present, now, dwell_seconds)
        self._recognize(frame, boxes, now)

        persons = []
        for track_id, coords in boxes:
            elapsed = self._dwell.elapsed(track_id, now)
            inside = present.get(track_id, False)
            shown = round(elapsed, 1) if inside and elapsed is not None else 0
            name = self._names.get(int(track_id), "")
            persons.append({"id": int(track_id), "name": name, "dwell": shown, "inside": inside})
            if name or (inside and elapsed is not None):
                text = name or f"ID {int(track_id)}"
                if inside and elapsed is not None:
                    text = f"{text}  {int(elapsed)}s"
                _draw_label(plotted, int(coords[0]), int(coords[1]), text, known=bool(name))

        if alarm_ids and polygon:
            self._raise_alarms(frame, plotted, boxes, alarm_ids)

        ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        ok_raw, raw = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with self._lock:
            if generation != self._generation:
                return
            self.persons = persons
            if ok:
                self.latest_jpeg = encoded.tobytes()
            if ok_raw:
                self.raw_jpeg = raw.tobytes()
                self.raw_at = time.time()

    def _enrolled(self) -> list[tuple[str, np.ndarray]]:
        if self._gallery_at and time.monotonic() - self._gallery_at < 5:
            return self._gallery
        from camera.models import Person

        from .faces import from_bytes

        gallery = []
        for person in Person.objects.all():
            if not person.embedding:
                continue
            gallery.append((person.name, from_bytes(bytes(person.embedding))))
        self._gallery = gallery
        self._gallery_at = time.monotonic()
        return gallery

    def _recognize(self, frame, boxes, now: float) -> None:
        from .faces import best_name, detect_faces

        live_ids = {int(track_id) for track_id, _coords in boxes}
        self._names = {track_id: name for track_id, name in self._names.items() if track_id in live_ids}
        self._name_try_at = {track_id: tried for track_id, tried in self._name_try_at.items() if track_id in live_ids}
        gallery = self._enrolled()
        if not gallery:
            return
        pending = [
            (int(track_id), coords)
            for track_id, coords in boxes
            if int(track_id) not in self._names and now - self._name_try_at.get(int(track_id), 0) >= 2
        ][:1]
        if not pending:
            return
        try:
            faces = list(detect_faces(frame))
        except Exception:
            faces = []
        for track_id, coords in pending:
            self._name_try_at[track_id] = now
            embedding = _embedding_for_person(frame, coords, faces)
            if embedding is None:
                continue
            name = best_name(embedding, gallery)
            if name:
                self._names[track_id] = name

    def _raise_alarms(self, frame, plotted, boxes, alarm_ids: list[int]) -> None:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import Alarm, Person

        from .faces import (
            MATCH_THRESHOLD,
            detect_faces,
            embed_face,
            face_inside_box,
            from_bytes,
            similarity,
            to_bytes,
        )

        try:
            faces = detect_faces(frame)
        except Exception:
            faces = []
        people = list(Person.objects.all())
        located = {track_id: coords for track_id, coords in boxes}
        for track_id in alarm_ids:
            coords = located.get(track_id)
            face = None
            if coords is not None:
                for candidate in faces:
                    if face_inside_box(candidate, coords):
                        face = candidate
                        break
            embedding = None
            if face is not None:
                try:
                    embedding = embed_face(frame, face)
                except cv2.error:
                    embedding = None
            matched = None
            score = None
            name = "Unknown"
            if embedding is not None and people:
                best = -1.0
                best_person = None
                for person in people:
                    if not person.embedding:
                        continue
                    value = similarity(embedding, from_bytes(bytes(person.embedding)))
                    if value > best:
                        best = value
                        best_person = person
                score = best if best >= 0 else None
                if best_person is not None and best >= MATCH_THRESHOLD:
                    matched = best_person
                    name = best_person.name

            ok, jpeg = cv2.imencode(".jpg", plotted)
            if not ok:
                continue
            alarm = Alarm(
                camera_number=self.camera_number,
                track_id=int(track_id),
                person=matched,
                matched_name=name,
                score=score,
            )
            stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
            alarm.snapshot.save(
                f"{stamp}-{int(track_id)}.jpg",
                ContentFile(jpeg.tobytes()),
                save=False,
            )
            if embedding is not None and face is not None:
                alarm.face_embedding = to_bytes(embedding)
                x, y, box_w, box_h = [int(value) for value in face[:4]]
                y1 = max(0, y)
                x1 = max(0, x)
                crop = frame[y1 : y1 + box_h, x1 : x1 + box_w]
                if crop.size:
                    ok_crop, crop_jpeg = cv2.imencode(".jpg", crop)
                    if ok_crop:
                        alarm.face_crop.save(
                            f"{stamp}-{int(track_id)}-face.jpg",
                            ContentFile(crop_jpeg.tobytes()),
                            save=False,
                        )
            alarm.save()
            with self._lock:
                self.last_alarm = {
                    "id": alarm.id,
                    "name": name,
                    "track_id": int(track_id),
                    "camera": self.camera_number,
                    "at": time.time(),
                    "active": True,
                }


engine = Engine()


def grab_jpeg(camera_number: int, stream: str) -> bytes:
    cached = engine.raw_frame(camera_number)
    if cached:
        return cached
    try:
        url, _channel = stream_url(camera_number, stream)
    except Exception:
        return b""
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    ok, frame = False, None
    if capture.isOpened():
        ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return b""
    encoded_ok, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes() if encoded_ok else b""
