import os
import re
import threading
import time

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;8000000",
)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .classes import COCO_NAMES, DEFAULT_ENABLED, catalog, label_for
from .count import CrowdTimer, PeopleCounter
from .dwell import DwellTracker
from .geom import box_bottom_center, box_iou, foot_point, line_cross, point_in_polygon
from .rules import PERSON_KINDS
from .rtsp import ROOT, base_rtsp_url, configured_camera, stream_url
from .scene import FieldMonitor

_stderr_lock = threading.Lock()
_RTSP_SECRET = re.compile(r"rtsp://\S+", re.IGNORECASE)


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
        cv2.polylines(image, [polygon], len(polygon) >= 3, (40, 200, 240), 2)


def _draw_line(image, points) -> None:
    if len(points) < 2:
        return
    height, width = image.shape[:2]
    start = (int(float(points[0][0]) * width), int(float(points[0][1]) * height))
    end = (int(float(points[1][0]) * width), int(float(points[1][1]) * height))
    cv2.arrowedLine(image, start, end, (40, 200, 240), 2, tipLength=0.08)


def _draw_shapes(image, shapes) -> None:
    for kind, points in shapes:
        if kind == "line":
            _draw_line(image, points)
        else:
            _draw_zone(image, points)


def _draw_captions(image, lines) -> None:
    """Count text at the top, about the same height as a camera date stamp."""
    if not lines:
        return
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    probe = cv2.getTextSize("People: 8", font, 1, 1)[0][1]
    scale = max(1.0, (height * 0.06) / max(probe, 1))
    thickness = max(2, int(round(scale)))
    y = max(8, int(height * 0.02))
    for text in lines:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(8, (width - text_width) // 2)
        cv2.rectangle(
            image,
            (x - 12, y),
            (x + text_width + 12, y + text_height + baseline + 14),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (x, y + text_height + 6),
            font,
            scale,
            (80, 230, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += text_height + baseline + max(10, int(height * 0.015))


def _open_capture(url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.open(url, cv2.CAP_FFMPEG)
    return capture


def _with_ffmpeg_log(fn):
    """Run fn while collecting C-level stderr. ffmpeg does not use sys.stderr."""
    with _stderr_lock:
        read_fd, write_fd = os.pipe()
        saved = os.dup(2)
        chunks: list[bytes] = []

        def _read() -> None:
            try:
                while True:
                    data = os.read(read_fd, 4096)
                    if not data:
                        break
                    chunks.append(data)
            except OSError:
                pass

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        spare_write = write_fd
        try:
            os.dup2(write_fd, 2)
            os.close(write_fd)
            spare_write = -1
            result = fn()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            if spare_write != -1:
                os.close(spare_write)
            reader.join(timeout=1)
            os.close(read_fd)
        text = b"".join(chunks).decode("utf-8", "replace")
        return result, text


def rtsp_failure_message(log: str) -> str:
    """Short reason for the Live page. Never includes the RTSP password."""
    redacted = _RTSP_SECRET.sub("rtsp://nvr", log or "")
    lowered = redacted.lower()
    if "401" in lowered or "unauthorized" in lowered or "403" in lowered:
        return "NVR login was rejected."
    if "timed out" in lowered or "timeout" in lowered:
        return "The NVR did not answer."
    if any(
        phrase in lowered
        for phrase in (
            "connection refused",
            "no route",
            "unreachable",
            "could not connect",
            "failed to resolve",
            "name or service not known",
        )
    ):
        return "Could not reach the NVR."
    lines = [line.strip() for line in redacted.splitlines() if line.strip()]
    if not lines:
        return "Could not read the camera."
    last = lines[-1]
    if len(last) > 140:
        last = last[:137] + "..."
    return last


def _read_until_frame(capture, stop_event: threading.Event, seconds: float = 3):
    """Skip the broken packets an NVR sends before the first real frame."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not stop_event.is_set():
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            return True, frame
    return False, None


NVR_LOCK_MESSAGE = (
    "The NVR refused this login. Open the NVR in a browser on this computer and sign in, then save again."
)


def probe_stream(url: str) -> str:
    """Return '' when one frame arrives, otherwise a short public error.

    One attempt only. A second login check, or a retry every few seconds, is enough
    for a Hikvision NVR to lock this computer until someone signs in on its web page.
    """
    started = time.monotonic()
    capture = _open_capture(url)
    try:
        if not capture.isOpened():
            if time.monotonic() - started < 2:
                return NVR_LOCK_MESSAGE
            return "The NVR did not answer."
        ok, _frame = _read_until_frame(capture, threading.Event(), seconds=8)
        if not ok:
            return "The NVR accepted the login, but the camera sent no picture."
        return ""
    finally:
        capture.release()


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
        self.people_count = None
        self.crowd_count = None
        self.enabled_classes = set(DEFAULT_ENABLED)
        self.latest_jpeg = b""
        self.raw_jpeg = b""
        self.raw_at = 0.0
        self.last_alarm: dict | None = None
        self._rules_cached: list = []
        self._rules_at = 0.0
        self._rules_camera = 0
        self._counters: dict[int, PeopleCounter] = {}
        self._crowds: dict[int, CrowdTimer] = {}
        self._line_tracks: dict[int, dict] = {}
        self._line_alarmed: dict[int, set] = {}
        self._fields: dict[int, FieldMonitor] = {}
        self._field_stamps: dict[int, str] = {}
        self._pose_model = None
        self._pose_name = ""
        self._last_draw: list[tuple] = []
        self._last_shapes: list[tuple] = []
        self._last_captions: list[str] = []
        self._face_jobs: dict[int, tuple] = {}
        self._face_lock = threading.Lock()
        threading.Thread(target=self._face_worker, name="face-match", daemon=True).start()

    def ensure_defaults(self) -> None:
        if self._defaults_ready:
            return
        self._defaults_ready = True
        try:
            base_rtsp_url()
        except RuntimeError as exc:
            self.error = str(exc)
            return
        camera, stream = configured_camera()
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
        with self._face_lock:
            self._face_jobs = {}
        self._last_draw = []
        self._last_shapes = []
        self._last_captions = []
        self._rules_at = 0.0
        self._counters = {}
        self._crowds = {}
        self._line_tracks = {}
        self._line_alarmed = {}
        self._fields = {}
        self._field_stamps = {}
        with self._lock:
            self.camera_number = int(camera_number)
            self.stream = stream if stream in ("sub", "main") else "sub"
            self.starting = True
            self.running = False
            self.error = ""
            self.persons = []
            self.detections = []
            self.people_count = None
            self.crowd_count = None
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
            self.people_count = None
            self.crowd_count = None
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
                "people_count": self.people_count,
                "crowd_count": self.crowd_count,
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
            self._fail(generation, "Set the NVR in Settings")
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The small model keeps the CPU preview smooth. CUDA can carry the larger one.
        weight_name = "yolo11m.pt" if device == "cuda" else "yolo11n.pt"
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
            started = time.monotonic()
            capture = _open_capture(url)
            if not capture.isOpened():
                capture.release()
                if seen_frame:
                    self._mark_reconnect(generation, seen_frame)
                    if stop_event.wait(30):
                        break
                    continue
                reason = NVR_LOCK_MESSAGE if time.monotonic() - started < 2 else "The NVR did not answer."
                self._fail(generation, reason)
                return
            ok, frame = _read_until_frame(capture, stop_event)
            if not ok or frame is None:
                capture.release()
                if seen_frame:
                    self._mark_reconnect(generation, seen_frame)
                    if stop_event.wait(30):
                        break
                    continue
                self._fail(generation, "The NVR accepted the login, but the camera sent no picture.")
                return
            seen_frame = True
            with self._lock:
                if generation == self._generation:
                    self.running = True
                    self.starting = False
                    self.error = ""
            next_infer = 0.0
            infer_gap = 0.0 if device == "cuda" else 0.15
            while not stop_event.is_set() and generation == self._generation:
                now = time.monotonic()
                try:
                    if now >= next_infer:
                        next_infer = now + infer_gap
                        self._handle_frame(model, frame, generation, device, imgsz)
                    else:
                        self._publish_preview(frame, generation)
                except Exception as exc:
                    print("frame error:", type(exc).__name__)
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
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

    def _mark_reconnect(self, generation: int, seen_frame: bool, reason: str = "") -> None:
        with self._lock:
            if generation == self._generation:
                self.running = False
                self.starting = True
                if not seen_frame and reason:
                    self.error = reason
                elif not seen_frame and not self.error:
                    self.error = "Could not read the camera."

    def _fail(self, generation: int, message: str) -> None:
        with self._lock:
            if generation == self._generation:
                self.error = message
                self.running = False
                self.starting = False

    def _load_rules(self):
        now = time.monotonic()
        if now - self._rules_at < 2 and self._rules_camera == self.camera_number:
            return self._rules_cached
        from camera.models import AnalyticsRule

        self._rules_cached = list(
            AnalyticsRule.objects.filter(camera_number=self.camera_number, active=True)
        )
        self._rules_at = now
        self._rules_camera = self.camera_number
        return self._rules_cached

    def _ensure_pose(self, device: str):
        name = "yolo11s-pose.pt" if device == "cuda" else "yolo11n-pose.pt"
        if self._pose_model is not None and self._pose_name == name:
            return self._pose_model
        try:
            self._pose_model = YOLO(str(ROOT / name))
            self._pose_name = name
        except Exception as exc:
            print("pose load:", type(exc).__name__, exc)
            self._pose_model = None
        return self._pose_model

    def _publish_preview(self, frame, generation: int) -> None:
        """Show the newest camera frame with the last boxes, without running the model again."""
        if generation != self._generation:
            return
        plotted = frame.copy()
        _draw_shapes(plotted, self._last_shapes)
        for coords, text, known in self._last_draw:
            _draw_box(plotted, coords, text, known)
        _draw_captions(plotted, self._last_captions)
        ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        with self._lock:
            if generation == self._generation:
                self.latest_jpeg = encoded.tobytes()

    def _handle_frame(self, model, frame, generation: int, device: str, imgsz: int) -> None:
        from django.db import close_old_connections

        if time.monotonic() - self._rules_at >= 2:
            close_old_connections()
        if generation != self._generation:
            return
        plotted = frame.copy()
        height, width = frame.shape[:2]
        rules = self._load_rules()
        class_ids = self._enabled_ids()
        if any(rule.kind in PERSON_KINDS for rule in rules) and 0 not in class_ids:
            class_ids = sorted({*class_ids, 0})
        boxes = []
        if class_ids:
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=class_ids,
                imgsz=imgsz,
                conf=0.30,
                device=device,
                verbose=False,
            )
            if results:
                boxes = _read_boxes(results[0])

        now = time.monotonic()
        shapes = []
        zone_rule = None
        line_rule = None
        count_rule = None
        crowd_rule = None
        object_rules = []
        for rule in rules:
            points = rule.points or []
            if rule.kind == "line_cross" and len(points) >= 2:
                line_rule = rule
                shapes.append(("line", points[:2]))
            elif len(points) >= 3 and rule.kind == "zone":
                zone_rule = rule
                shapes.append(("poly", points))
            elif len(points) >= 3 and rule.kind == "people_count":
                count_rule = rule
                shapes.append(("poly", points))
            elif len(points) >= 3 and rule.kind == "crowd":
                crowd_rule = rule
                shapes.append(("poly", points))
            elif len(points) >= 3 and rule.kind in ("object_in", "object_removed"):
                object_rules.append(rule)
                shapes.append(("poly", points))
        _draw_shapes(plotted, shapes)

        person_boxes = []
        for track_id, coords, class_id, _conf in boxes:
            if class_id != 0 or track_id < 0:
                continue
            person_boxes.append((track_id, coords))
        need_pose = any(rule is not None for rule in (zone_rule, line_rule, count_rule, crowd_rule))
        if need_pose and person_boxes:
            feet = _feet_for_people(self._ensure_pose(device), frame, person_boxes, width, height, device, imgsz)
        else:
            feet = {
                int(track_id): box_bottom_center(coords, width, height)
                for track_id, coords in person_boxes
            }

        present: dict[int, bool] = {}
        dwell_seconds = 60
        if zone_rule is not None:
            dwell_seconds = zone_rule.duration_seconds
            present = _inside_map(person_boxes, feet, zone_rule.points, width, height)
        if zone_rule is None:
            present = {}
        alarm_ids = self._dwell.update(present, now, dwell_seconds)

        line_ids = []
        if line_rule is not None:
            line_ids = self._cross_line(line_rule, feet)

        captions = []
        people_count = None
        if count_rule is not None:
            people_count = self._count_people(count_rule, person_boxes, feet, width, height)
            captions.append(f"People: {people_count}")
        crowd_count = None
        crowd_hit = False
        if crowd_rule is not None:
            crowd_count = self._count_people(crowd_rule, person_boxes, feet, width, height)
            captions.append(f"Crowd: {crowd_count}")
            timer = self._crowds.get(crowd_rule.id)
            if timer is None:
                timer = CrowdTimer()
                self._crowds[crowd_rule.id] = timer
            crowd_hit = timer.update(crowd_count, now, crowd_rule.max_people, crowd_rule.duration_seconds)

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

        _draw_captions(plotted, captions)
        if alarm_ids and zone_rule is not None:
            self._raise_alarms(frame, plotted, person_boxes, alarm_ids, kind="zone")
        if line_ids:
            self._raise_alarms(frame, plotted, person_boxes, line_ids, kind="line_cross")
        if crowd_hit:
            self._raise_scene_alarm(plotted, "crowd")
        for rule in object_rules:
            stamp = rule.updated_at.isoformat() if rule.updated_at else ""
            monitor = self._fields.get(rule.id)
            if monitor is None or self._field_stamps.get(rule.id) != stamp:
                monitor = FieldMonitor()
                self._fields[rule.id] = monitor
                self._field_stamps[rule.id] = stamp
            for event in monitor.update(
                frame,
                rule.points,
                [coords for _track_id, coords in person_boxes],
                now,
                rule.duration_seconds,
                rule.kind == "object_in",
                rule.kind == "object_removed",
            ):
                self._raise_scene_alarm(plotted, event)

        self._last_draw = draw
        self._last_shapes = shapes
        self._last_captions = captions
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
            self.people_count = people_count
            self.crowd_count = crowd_count
            if ok:
                self.latest_jpeg = encoded.tobytes()
            if ok_raw:
                self.raw_jpeg = raw.tobytes()
                self.raw_at = time.time()

    def _count_people(self, rule, person_boxes, feet, width: int, height: int) -> int:
        counter = self._counters.get(rule.id)
        if counter is None:
            counter = PeopleCounter()
            self._counters[rule.id] = counter
        return counter.update(_inside_map(person_boxes, feet, rule.points, width, height))

    def _cross_line(self, rule, feet: dict[int, tuple]) -> list[int]:
        previous = self._line_tracks.setdefault(rule.id, {})
        alarmed = self._line_alarmed.setdefault(rule.id, set())
        line = [(float(point[0]), float(point[1])) for point in rule.points[:2]]
        alarms = []
        live = set()
        for track_id, point in feet.items():
            track_id = int(track_id)
            live.add(track_id)
            prior = previous.get(track_id)
            previous[track_id] = point
            if prior is None or track_id in alarmed:
                continue
            crossed = line_cross(prior, point, line[0], line[1])
            if crossed is None:
                continue
            if rule.direction != "any" and crossed != rule.direction:
                continue
            alarmed.add(track_id)
            alarms.append(track_id)
        for track_id in list(previous):
            if track_id not in live:
                previous.pop(track_id, None)
                alarmed.discard(track_id)
        return alarms

    def _gallery_people(self) -> list[tuple]:
        if self._gallery_at and time.monotonic() - self._gallery_at < 5:
            return self._gallery
        from camera.models import Person

        from .gallery import load_person_vectors

        gallery = []
        for person in Person.objects.prefetch_related("samples"):
            for vector in load_person_vectors(person):
                gallery.append((person, vector))
        self._gallery = gallery
        self._gallery_at = time.monotonic()
        return gallery

    def _queue_face(self, generation: int, frame, person_boxes, now: float) -> None:
        from .faces import model_ready, take_head

        live_ids = {int(track_id) for track_id, _coords in person_boxes}
        with self._face_lock:
            self._face_jobs = {
                track_id: job for track_id, job in self._face_jobs.items() if track_id in live_ids
            }
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
                wait = 2.0 if track_id in self._names else 0.35
                if now - self._name_try_at.get(track_id, 0) >= wait:
                    pending.append((track_id, coords))
        if not pending or not model_ready():
            return
        for track_id, coords in pending:
            taken = take_head(frame, coords)
            with self._lock:
                self._name_try_at[track_id] = now
            if taken is None:
                continue
            crop, origin, shape = taken
            with self._face_lock:
                self._face_jobs[track_id] = (generation, crop, origin, shape, track_id, now)

    def _pop_face_job(self):
        with self._face_lock:
            if not self._face_jobs:
                return None
            track_id = min(self._face_jobs, key=lambda item: self._face_jobs[item][5])
            return self._face_jobs.pop(track_id)

    def _face_worker(self) -> None:
        while True:
            job = self._pop_face_job()
            if job is None:
                time.sleep(0.02)
                continue
            generation, crop, origin, shape, track_id, _queued = job
            if generation != self._generation:
                continue
            try:
                from django.db import close_old_connections

                close_old_connections()
                self._match_face(crop, origin, shape, track_id)
            except Exception:
                print("face match error")

    def _match_face(self, crop, origin, shape, track_id: int) -> None:
        from .captures import save_visit
        from .faces import best_person, blacklist_alarm_ready, embed_head, next_identity

        found = embed_head(crop, origin, shape)
        if found is None:
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
        if found.save_ok and found.photo is not None and found.photo.size:
            capture_id = save_visit(
                self.camera_number,
                track_id,
                found.embedding,
                found.photo,
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
            if self._raise_list_alarm(crop, origin, track_id, person, locked_score, found):
                with self._lock:
                    self._alarmed_tracks.add(track_id)

    def _raise_list_alarm(self, crop, origin, track_id: int, person, score: float, found) -> bool:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import Alarm

        from .faces import box_on_crop, to_bytes

        plotted = crop.copy()
        _draw_box(plotted, box_on_crop(found.box, origin, crop.shape), f"{person.name} · blacklist", "blacklist")
        ok, jpeg = cv2.imencode(".jpg", plotted)
        if not ok:
            return False
        alarm = Alarm(
            camera_number=self.camera_number,
            track_id=int(track_id),
            person=person,
            matched_name=person.name,
            score=score,
            kind="blacklist",
        )
        stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
        alarm.snapshot.save(f"{stamp}-{int(track_id)}-list.jpg", ContentFile(jpeg.tobytes()), save=False)
        alarm.face_embedding = to_bytes(found.embedding)
        face = found.photo
        if face is not None and face.size:
            ok_crop, crop_jpeg = cv2.imencode(".jpg", face)
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
                "kind": "blacklist",
                "label": "Blacklist",
                "active": True,
            }
        return True

    def _raise_scene_alarm(self, plotted, kind: str) -> None:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import ALARM_KIND_LABELS, Alarm

        ok, jpeg = cv2.imencode(".jpg", plotted)
        if not ok:
            return
        label = ALARM_KIND_LABELS.get(kind, kind)
        alarm = Alarm(
            camera_number=self.camera_number,
            track_id=0,
            matched_name=label,
            kind=kind,
        )
        stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
        alarm.snapshot.save(f"{stamp}-{kind}.jpg", ContentFile(jpeg.tobytes()), save=False)
        alarm.save()
        with self._lock:
            self.last_alarm = {
                "id": alarm.id,
                "name": label,
                "track_id": 0,
                "camera": self.camera_number,
                "at": time.time(),
                "kind": kind,
                "label": label,
                "active": True,
            }

    def _raise_alarms(self, frame, plotted, person_boxes, alarm_ids: list[int], kind: str = "zone") -> None:
        from django.core.files.base import ContentFile
        from django.utils import timezone

        from camera.models import ALARM_KIND_LABELS, Alarm

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
                kind=kind,
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
                    "kind": kind,
                    "label": ALARM_KIND_LABELS.get(kind, "Zone"),
                    "active": True,
                }


def _inside_map(person_boxes, feet, points, width: int, height: int) -> dict[int, bool]:
    polygon = [(float(point[0]), float(point[1])) for point in points]
    present = {}
    for track_id, coords in person_boxes:
        if track_id < 0:
            continue
        point = feet.get(int(track_id)) or box_bottom_center(coords, width, height)
        present[int(track_id)] = point_in_polygon(point[0], point[1], polygon)
    return present


def _read_pose(result):
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return [], []
    return result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()


def _feet_for_people(pose_model, frame, person_boxes, width: int, height: int, device: str, imgsz: int):
    fallback = {
        int(track_id): box_bottom_center(coords, width, height)
        for track_id, coords in person_boxes
        if track_id >= 0
    }
    if pose_model is None or not person_boxes:
        return fallback
    try:
        results = pose_model(frame, conf=0.25, imgsz=imgsz, device=device, verbose=False)
    except Exception:
        return fallback
    if not results:
        return fallback
    pose_boxes, keypoints = _read_pose(results[0])
    if len(pose_boxes) == 0:
        return fallback
    used = set()
    feet = {}
    for track_id, coords in person_boxes:
        if track_id < 0:
            continue
        best_index = None
        best_score = 0.25
        for index, pose_box in enumerate(pose_boxes):
            if index in used:
                continue
            score = box_iou(coords, pose_box)
            if score > best_score:
                best_score = score
                best_index = index
        chosen = keypoints[best_index] if best_index is not None else None
        if best_index is not None:
            used.add(best_index)
        feet[int(track_id)] = foot_point(coords, chosen, width, height)
    return feet


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
    capture = _open_capture(url)
    ok, frame = False, None
    if capture.isOpened():
        ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return b""
    encoded_ok, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes() if encoded_ok else b""
