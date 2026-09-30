import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .classes import COCO_NAMES, DEFAULT_ENABLED, catalog, label_for
from .dwell import DwellTracker
from .geom import point_in_polygon
from .rtsp import ROOT, base_rtsp_url, parse_channel, stream_url

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")


def _placeholder(text: str) -> bytes:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(image, text, (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else b""


def _kind_name(kind) -> str:
    if kind is True:
        return "whitelist"
    if kind is False or kind is None:
        return ""
    return str(kind)


def _box_color(kind: str) -> tuple[int, int, int]:
    if kind == "blacklist":
        return (48, 48, 220)
    if kind == "whitelist":
        return (46, 140, 60)
    return (40, 160, 200)


def _draw_label(image, x: int, y: int, text: str, kind: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - 10)
    color = _box_color(kind)
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


def _draw_box(image, coords, text: str, kind="") -> None:
    kind = _kind_name(kind)
    x1, y1, x2, y2 = [int(value) for value in coords]
    color = _box_color(kind)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    _draw_label(image, x1, y1, text, kind)


def _draw_zone(image, points) -> None:
    height, width = image.shape[:2]
    polygon = np.array(
        [[int(float(x) * width), int(float(y) * height)] for x, y in points],
        dtype=np.int32,
    )
    if len(polygon) >= 2:
        cv2.polylines(image, [polygon], True, (40, 200, 240), 2)


def _open_capture(url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class Engine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warmed = False
        self._generation = 0
        self._defaults_ready = False
        self._dwell = DwellTracker()
        self._names: dict[int, str] = {}
        self._name_scores: dict[int, float] = {}
        self._kinds: dict[int, str] = {}
        self._person_ids: dict[int, int] = {}
        self._votes: dict[int, dict] = {}
        self._name_try_at: dict[int, float] = {}
        self._capture_ids: dict[int, int] = {}
        self._alarmed_tracks: set[int] = set()
        self._gallery: list[tuple] = []
        self._gallery_at = 0.0
        self.running = False
        self.starting = False
        self.camera_number = 1
        self.stream = "sub"
        self.channel = ""
        self.device = "cpu"
        self.error = ""
        self.persons: list[dict] = []
        self.detections: list[dict] = []
        self.enabled_classes = set(DEFAULT_ENABLED)
        self.latest_jpeg = b""
        self.raw_jpeg = b""
        self.raw_at = 0.0
        self.last_alarm: dict | None = None
        self._zone_cached = None
        self._zone_at = 0.0
        self._last_draw: list[tuple] = []
        self._last_polygon: list[tuple[float, float]] = []
        self._face_job = None
        self._face_lock = threading.Lock()
        threading.Thread(target=self._face_worker, name="face-match", daemon=True).start()

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
        self._name_scores = {}
        self._kinds = {}
        self._person_ids = {}
        self._votes = {}
        self._name_try_at = {}
        self._capture_ids = {}
        self._alarmed_tracks = set()
        self._last_draw = []
        self._last_polygon = []
        self._zone_at = 0.0
        with self._lock:
            self.camera_number = int(camera_number)
            self.stream = stream if stream in ("sub", "main") else "sub"
            self.starting = True
            self.running = False
            self.error = ""
            self.persons = []
            self.detections = []
            self.latest_jpeg = b""
            self.channel = ""
        self._warm_faces()
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
            self.detections = []
            self.latest_jpeg = b""
            self.raw_jpeg = b""

    def toggle_class(self, class_id: int) -> None:
        if class_id not in COCO_NAMES:
            return
        with self._lock:
            if class_id in self.enabled_classes:
                self.enabled_classes.remove(class_id)
            else:
                self.enabled_classes.add(class_id)

    def class_groups(self) -> list[dict]:
        with self._lock:
            enabled = set(self.enabled_classes)
        return catalog(enabled)

    def _enabled_ids(self) -> list[int]:
        with self._lock:
            return sorted(self.enabled_classes)

    def status(self) -> dict:
        with self._lock:
            alarm = None
            if self.last_alarm and time.time() - self.last_alarm["at"] < 30:
                alarm = dict(self.last_alarm)
                alarm["active"] = True
            elif self.last_alarm:
                alarm = dict(self.last_alarm)
                alarm["active"] = False
            snapshot = {
                "running": self.running,
                "starting": self.starting,
                "camera": self.camera_number,
                "stream": self.stream,
                "channel": self.channel,
                "device": self.device,
                "error": self.error,
                "persons": list(self.persons),
                "detections": list(self.detections),
                "enabled": set(self.enabled_classes),
                "alarm": alarm,
            }
        snapshot["classes"] = catalog(snapshot.pop("enabled"))
        return snapshot

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

    def _warm_faces(self) -> None:
        if self._warmed:
            return
        self._warmed = True

        def _load() -> None:
            try:
                from .faces import warm

                warm()
            except Exception as exc:
                print("face model:", type(exc).__name__, exc)

        threading.Thread(target=_load, name="face-model", daemon=True).start()

    def _run(self, generation: int, stop_event: threading.Event) -> None:
        from django.db import close_old_connections

        try:
            url, channel = stream_url(self.camera_number, self.stream)
        except Exception:
            self._fail(generation, "Set RTSP_URL in .env")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The small model keeps the CPU preview smooth. CUDA can carry the larger one.
        weight_name = "yolo11s.pt" if device == "cuda" else "yolo11n.pt"
        imgsz = 640 if device == "cuda" else 480
        with self._lock:
            if generation != self._generation:
                return
            self.channel = channel
            self.device = device
        try:
            model = YOLO(str(ROOT / weight_name))
        except Exception as exc:
            print("yolo load:", type(exc).__name__, exc)
            self._fail(generation, "Could not load the detection model.")
            return
        seen_frame = False
        while not stop_event.is_set() and generation == self._generation:
            capture = _open_capture(url)
            if not capture.isOpened():
                capture.release()
                self._mark_reconnect(generation, seen_frame)
                if stop_event.wait(2):
                    break
                continue
            with self._lock:
                if generation == self._generation:
                    self.running = True
                    self.starting = False
                    self.error = ""
            next_infer = 0.0
            infer_gap = 0.0 if device == "cuda" else 0.15
            while not stop_event.is_set() and generation == self._generation:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                seen_frame = True
                now = time.monotonic()
                try:
                    if now >= next_infer:
                        next_infer = now + infer_gap
                        self._handle_frame(model, frame, generation, device, imgsz)
                    else:
                        self._publish_preview(frame, generation)
                except Exception as exc:
                    print("frame error:", type(exc).__name__)
            capture.release()
            if stop_event.is_set() or generation != self._generation:
                break
            self._mark_reconnect(generation, seen_frame)
            if stop_event.wait(2):
                break
        close_old_connections()
        with self._lock:
            if generation == self._generation:
                self.running = False
                self.starting = False

    def _mark_reconnect(self, generation: int, seen_frame: bool) -> None:
        with self._lock:
            if generation == self._generation:
                self.running = False
                self.starting = True
                if not seen_frame:
                    self.error = "Could not read the camera."

    def _fail(self, generation: int, message: str) -> None:
        with self._lock:
            if generation == self._generation:
                self.error = message
                self.running = False
                self.starting = False

    def _load_zone(self):
        now = time.monotonic()
        if now - self._zone_at < 2:
            return self._zone_cached
        from camera.models import Zone

        self._zone_cached = Zone.objects.filter(camera_number=self.camera_number, active=True).first()
        self._zone_at = now
        return self._zone_cached

    def _publish_preview(self, frame, generation: int) -> None:
        """Show the newest camera frame with the last boxes, without running the model again."""
        if generation != self._generation:
            return
        plotted = frame.copy()
        if self._last_polygon:
            _draw_zone(plotted, self._last_polygon)
        for coords, text, known in self._last_draw:
            _draw_box(plotted, coords, text, known)
        ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        with self._lock:
            if generation == self._generation:
                self.latest_jpeg = encoded.tobytes()

    def _handle_frame(self, model, frame, generation: int, device: str, imgsz: int) -> None:
        from django.db import close_old_connections

        if time.monotonic() - self._zone_at >= 2:
            close_old_connections()
        if generation != self._generation:
            return
        plotted = frame.copy()
        height, width = frame.shape[:2]
        class_ids = self._enabled_ids()
        boxes = []
        if class_ids:
            results = model.track(
                frame,
                persist=True,
                classes=class_ids,
                imgsz=imgsz,
                conf=0.30,
                device=device,
                verbose=False,
            )
            if results:
                boxes = _read_boxes(results[0])

        zone = self._load_zone()
        now = time.monotonic()
        polygon = []
        dwell_seconds = 60
        if zone and len(zone.points) >= 3:
            polygon = [(float(point[0]), float(point[1])) for point in zone.points]
            dwell_seconds = zone.dwell_seconds
            _draw_zone(plotted, polygon)

        present: dict[int, bool] = {}
        person_boxes = []
        for track_id, coords, class_id, _conf in boxes:
            if class_id != 0 or track_id < 0:
                continue
            person_boxes.append((track_id, coords))
            if polygon:
                x1, y1, x2, y2 = coords
                present[track_id] = point_in_polygon((x1 + x2) / 2 / width, y2 / height, polygon)
            else:
                present[track_id] = False
        alarm_ids = self._dwell.update(present, now, dwell_seconds)
        self._queue_face(generation, frame, person_boxes, now)

        persons = []
        detections = []
        draw: list[tuple] = []
        with self._lock:
            names = dict(self._names)
            kinds = dict(self._kinds)
        for track_id, coords, class_id, conf in boxes:
            label = label_for(class_id) if class_id in COCO_NAMES else "Object"
            name = ""
            kind = ""
            inside = False
            shown = 0
            if class_id == 0 and track_id >= 0:
                elapsed = self._dwell.elapsed(track_id, now)
                inside = present.get(track_id, False)
                shown = round(elapsed, 1) if inside and elapsed is not None else 0
                name = names.get(int(track_id), "")
                kind = kinds.get(int(track_id), "")
                text = f"{name} · {kind}" if name and kind else (name or "Person")
                if inside and elapsed is not None:
                    text = f"{text}  {int(elapsed)}s"
                persons.append(
                    {"id": int(track_id), "name": name, "list": kind, "dwell": shown, "inside": inside}
                )
                _draw_box(plotted, coords, text, kind)
                draw.append((coords, text, kind))
            else:
                text = f"{label} {round(float(conf) * 100)}%"
                _draw_box(plotted, coords, text, "")
                draw.append((coords, text, ""))
            detections.append(
                {
                    "id": int(track_id),
                    "class_id": int(class_id),
                    "label": label,
                    "name": name,
                    "list": kind,
                    "confidence": round(float(conf), 3),
                    "dwell": shown,
                    "inside": inside,
                }
            )

        if alarm_ids and polygon:
            self._raise_alarms(frame, plotted, person_boxes, alarm_ids)

        self._last_draw = draw
        self._last_polygon = polygon
        ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        raw_due = time.time() - self.raw_at >= 1
        ok_raw, raw = (False, None)
        if raw_due:
            ok_raw, raw = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        with self._lock:
            if generation != self._generation:
                return
            self.persons = persons
            self.detections = detections
            if ok:
                self.latest_jpeg = encoded.tobytes()
            if ok_raw:
                self.raw_jpeg = raw.tobytes()
                self.raw_at = time.time()

    def _gallery_people(self) -> list[tuple]:
        if self._gallery_at and time.monotonic() - self._gallery_at < 5:
            return self._gallery
        from camera.models import Person

        from .gallery import load_person_vector

        gallery = []
        for person in Person.objects.all():
            vector = load_person_vector(person)
            if vector is None:
                continue
            gallery.append((person, vector))
        self._gallery = gallery
        self._gallery_at = time.monotonic()
        return gallery

    def _queue_face(self, generation: int, frame, person_boxes, now: float) -> None:
        from .faces import model_ready

        live_ids = {int(track_id) for track_id, _coords in person_boxes}
        with self._lock:
            self._names = {track_id: name for track_id, name in self._names.items() if track_id in live_ids}
            self._name_scores = {
                track_id: score for track_id, score in self._name_scores.items() if track_id in live_ids
            }
            self._kinds = {track_id: kind for track_id, kind in self._kinds.items() if track_id in live_ids}
            self._person_ids = {
                track_id: person_id
                for track_id, person_id in self._person_ids.items()
                if track_id in live_ids
            }
            self._votes = {track_id: votes for track_id, votes in self._votes.items() if track_id in live_ids}
            self._name_try_at = {
                track_id: tried for track_id, tried in self._name_try_at.items() if track_id in live_ids
            }
            self._capture_ids = {
                track_id: capture_id
                for track_id, capture_id in self._capture_ids.items()
                if track_id in live_ids
            }
            self._alarmed_tracks = {track_id for track_id in self._alarmed_tracks if track_id in live_ids}
            pending = []
            for track_id, coords in person_boxes:
                track_id = int(track_id)
                wait = 4.0 if track_id in self._names else 1.5
                if now - self._name_try_at.get(track_id, 0) >= wait:
                    pending.append((track_id, coords))
        if not pending or not model_ready():
            return
        with self._face_lock:
            if self._face_job is not None:
                return
        pending.sort(key=lambda item: self._name_try_at.get(item[0], 0))
        track_id, coords = pending[0]
        snapshot = frame.copy()
        with self._lock:
            self._name_try_at[track_id] = now
        with self._face_lock:
            if self._face_job is None:
                self._face_job = (generation, snapshot, track_id, coords)

    def _face_worker(self) -> None:
        while True:
            with self._face_lock:
                job = self._face_job
                self._face_job = None
            if job is None:
                time.sleep(0.05)
                continue
            generation, frame, track_id, coords = job
            if generation != self._generation:
                continue
            try:
                from django.db import close_old_connections

                close_old_connections()
                self._match_face(frame, track_id, coords)
            except Exception:
                print("face match error")

    def _match_face(self, frame, track_id: int, coords) -> None:
        from .captures import save_visit
        from .faces import (
            _crop_box,
            _padded_box,
            best_person,
            blacklist_alarm_ready,
            embed_upper_body,
            next_identity,
        )

        found = embed_upper_body(frame, coords)
        if found is None or not (found.label_ok or found.save_ok):
            return
        people = self._gallery_people()
        name = ""
        score = 0.0
        if found.label_ok and people:
            matched, score = best_person(found.embedding, people)
            if matched is not None:
                name = matched.name
        with self._lock:
            if name:
                votes = self._votes.setdefault(track_id, {})
                locked, locked_score = next_identity(
                    votes,
                    self._names.get(track_id, ""),
                    self._name_scores.get(track_id, 0.0),
                    name,
                    score,
                )
                if locked:
                    chosen = next((person for person, _vector in people if person.name == locked), None)
                    if chosen is not None:
                        self._names[track_id] = locked
                        self._name_scores[track_id] = locked_score
                        self._kinds[track_id] = chosen.list_status
                        self._person_ids[track_id] = chosen.pk
            locked_name = self._names.get(track_id, "")
            locked_score = self._name_scores.get(track_id, 0.0)
            kind = self._kinds.get(track_id, "")
            person_id = self._person_ids.get(track_id)
            known_capture = self._capture_ids.get(track_id)
            already_alarmed = track_id in self._alarmed_tracks
        person = next((item for item, _vector in people if item.pk == person_id), None)
        if found.save_ok:
            crop = _crop_box(frame, _padded_box(found.box, frame.shape))
            capture_id = save_visit(
                self.camera_number,
                track_id,
                found.embedding,
                crop,
                found.det_score,
                found.quality,
                person if locked_name else None,
                locked_name,
                locked_score if locked_name else None,
                known_capture,
            )
            if capture_id:
                with self._lock:
                    self._capture_ids[track_id] = capture_id
        if blacklist_alarm_ready(kind, locked_score) and not already_alarmed and person is not None:
            if self._raise_list_alarm(frame, track_id, coords, person, locked_score, found):
                with self._lock:
                    self._alarmed_tracks.add(track_id)

    def _raise_list_alarm(self, frame, track_id: int, coords, person, score: float, found) -> bool:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import Alarm

        from .faces import _crop_box, _padded_box, to_bytes

        plotted = frame.copy()
        _draw_box(plotted, coords, f"{person.name} · blacklist", "blacklist")
        ok, jpeg = cv2.imencode(".jpg", plotted)
        if not ok:
            return False
        alarm = Alarm(
            camera_number=self.camera_number,
            track_id=int(track_id),
            person=person,
            matched_name=person.name,
            score=score,
        )
        stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
        alarm.snapshot.save(f"{stamp}-{int(track_id)}-list.jpg", ContentFile(jpeg.tobytes()), save=False)
        alarm.face_embedding = to_bytes(found.embedding)
        crop = _crop_box(frame, _padded_box(found.box, frame.shape))
        if crop.size:
            ok_crop, crop_jpeg = cv2.imencode(".jpg", crop)
            if ok_crop:
                alarm.face_crop.save(
                    f"{stamp}-{int(track_id)}-list-face.jpg",
                    ContentFile(crop_jpeg.tobytes()),
                    save=False,
                )
        alarm.save()
        with self._lock:
            self.last_alarm = {
                "id": alarm.id,
                "name": person.name,
                "track_id": int(track_id),
                "camera": self.camera_number,
                "at": time.time(),
                "active": True,
            }
        return True

    def _raise_alarms(self, frame, plotted, person_boxes, alarm_ids: list[int]) -> None:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import Alarm

        from .faces import best_match, embed_upper_body, model_ready, to_bytes

        if not model_ready():
            faces_ready = False
        else:
            faces_ready = True
        people = self._gallery_people()
        located = {int(track_id): coords for track_id, coords in person_boxes}
        for track_id in alarm_ids:
            coords = located.get(int(track_id))
            embedding = None
            face_box = None
            if faces_ready and coords is not None:
                try:
                    found = embed_upper_body(frame, coords)
                except Exception:
                    found = None
                if found is not None:
                    embedding, face_box = found.embedding, found.box
            matched = None
            score = None
            name = "Unknown"
            if embedding is not None and people:
                chosen, best = best_match(embedding, [(person.name, vector) for person, vector in people])
                score = best
                if chosen:
                    name = chosen
                    matched = next((person for person, _vector in people if person.name == chosen), None)

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
            if embedding is not None and face_box is not None:
                alarm.face_embedding = to_bytes(embedding)
                x1, y1, x2, y2 = face_box
                crop = frame[y1:y2, x1:x2]
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


def _read_boxes(result) -> list[tuple]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
    else:
        ids = [-index - 1 for index in range(len(xyxy))]
    return list(zip(ids, xyxy, clss, confs))


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
