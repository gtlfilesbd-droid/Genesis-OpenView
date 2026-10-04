import cv2
import numpy as np

MIN_AREA_RATIO = 0.004
MIN_FRAME_RATIO = 0.0002
MAX_AREA_RATIO = 0.85
LIGHT_RATIO = 0.92
DIFF_THRESH = 28
CENTER_TOLERANCE = 0.05
PERSON_GROW = 0.08
PERSON_MASK_CONF = 0.55
SCENE_LONG_SIDE = 480
RED_HOLD = 8.0
MISS_LIMIT = 6
PATCH_MATCH = 0.55
PATCH_MARGIN = 0.15


def polygon_mask(height: int, width: int, points) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    polygon = np.array(
        [[int(float(point[0]) * width), int(float(point[1]) * height)] for point in points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def grow_box(coords, width: int, height: int, grow: float = PERSON_GROW):
    x1, y1, x2, y2 = [float(value) for value in coords]
    box_w = x2 - x1
    box_h = y2 - y1
    return (
        max(0, int(x1 - box_w * grow)),
        max(0, int(y1 - box_h * grow)),
        min(width, int(x2 + box_w * grow)),
        min(height, int(y2 + box_h * grow)),
    )


def apply_person_mask(mask: np.ndarray, boxes, width: int, height: int) -> np.ndarray:
    out = mask.copy()
    for coords in boxes:
        x1, y1, x2, y2 = grow_box(coords, width, height)
        out[y1:y2, x1:x2] = 0
    return out


def changed_mask(current: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(current, reference)
    changed = np.zeros_like(mask)
    changed[(diff > DIFF_THRESH) & (mask > 0)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
    return changed


def is_global_change(changed: np.ndarray, mask: np.ndarray) -> bool:
    roi = int(np.count_nonzero(mask))
    if roi < 50:
        return False
    return int(np.count_nonzero(changed)) / roi >= LIGHT_RATIO


def _same_spot(origin, center) -> bool:
    return (center[0] - origin[0]) ** 2 + (center[1] - origin[1]) ** 2 <= CENTER_TOLERANCE**2


def _object_still_inside(armed, current, blobs, mask) -> bool:
    if len(blobs) >= 2:
        return True
    if len(blobs) != 1:
        return False
    return _patch_elsewhere(armed, current, blobs[0], mask)


def _patch_elsewhere(armed, current, blob, mask) -> bool:
    x, y, width, height = blob["box"]
    if width < 8 or height < 8:
        return False
    if y + height > armed.shape[0] or x + width > armed.shape[1]:
        return False
    if current.shape[0] < height or current.shape[1] < width:
        return False
    patch = armed[y : y + height, x : x + width]
    scores = cv2.matchTemplate(current, patch, cv2.TM_CCOEFF_NORMED)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    y0 = max(0, y - height // 4)
    y1 = min(scores.shape[0], y + height // 4 + 1)
    x0 = max(0, x - width // 4)
    x1 = min(scores.shape[1], x + width // 4 + 1)
    scores[y0:y1, x0:x1] = -1.0
    _lowest, best, _low_at, best_at = cv2.minMaxLoc(scores)
    if not np.isfinite(best) or best < PATCH_MATCH:
        return False
    finite = scores[np.isfinite(scores)]
    if finite.size == 0 or best - float(np.percentile(finite, 90)) < PATCH_MARGIN:
        return False
    match_x, match_y = best_at
    window = mask[match_y : match_y + height, match_x : match_x + width]
    if window.size == 0:
        return False
    return int(np.count_nonzero(window)) / window.size >= 0.5


def local_blobs(current: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    changed = changed_mask(current, reference, mask)
    if is_global_change(changed, mask):
        return [], True
    height, width = current.shape[:2]
    roi = max(1, int(np.count_nonzero(mask)))
    floor = min(MIN_AREA_RATIO * roi, MIN_FRAME_RATIO * height * width)
    ceiling = MAX_AREA_RATIO * roi
    found = []
    contours, _hierarchy = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < floor or area > ceiling:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (moments["m10"] / moments["m00"] / width, moments["m01"] / moments["m00"] / height)
        x, y, box_w, box_h = cv2.boundingRect(contour)
        found.append({"center": center, "area": area, "box": (x, y, box_w, box_h)})
    return found, False


def _fit_scene(frame, person_boxes):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= SCENE_LONG_SIDE:
        return frame, person_boxes
    scale = SCENE_LONG_SIDE / longest
    small = cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    scaled = []
    for coords in person_boxes:
        x1, y1, x2, y2 = [float(value) for value in coords]
        scaled.append((x1 * scale, y1 * scale, x2 * scale, y2 * scale))
    return small, scaled


class FieldMonitor:
    """Stationary object appear and disappear inside a polygon.

    Each still spot keeps its own clock. A larger change elsewhere in the zone
    can move without resetting that clock. A brightness jump across the zone is
    a light change and does not alarm. People boxes are ignored.
    """

    def __init__(self) -> None:
        self.long_term = None
        self.armed = None
        self.in_center = None
        self.in_since = None
        self.in_alarmed = False
        self.in_misses = 0
        self.in_tracks = []
        self.out_center = None
        self.out_since = None
        self.out_alarmed = False
        self.out_misses = 0
        self.out_tracks = []
        self.elapsed = 0.0
        self.watching = False
        self.red_until = 0.0

    def showing_red(self, now: float) -> bool:
        return now < self.red_until

    def update(
        self,
        frame,
        points,
        person_boxes,
        now: float,
        duration: float,
        want_in: bool,
        want_removed: bool,
        outside: bool = False,
    ):
        frame, person_boxes = _fit_scene(frame, person_boxes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        height, width = gray.shape[:2]
        mask = apply_person_mask(polygon_mask(height, width, points), person_boxes, width, height)
        if self.long_term is None or self.long_term.shape != gray.shape:
            self.long_term = gray.copy()
            self.armed = gray.copy()
            self._clear("in")
            self._clear("out")
            return []
        alarms = []
        if want_in:
            alarms.extend(self._appear(gray, mask, now, duration))
        if want_removed and self.armed is not None:
            alarms.extend(self._removed(gray, mask, now, duration, outside))
        return alarms

    def _appear(self, gray, mask, now, duration):
        blobs, global_light = local_blobs(gray, self.long_term, mask)
        if global_light:
            self.long_term[mask > 0] = gray[mask > 0]
            self._clear("in")
            return []
        quiet = (cv2.absdiff(gray, self.long_term) < 12) & (mask > 0)
        self.long_term[quiet] = gray[quiet]
        return self._track("in", blobs, now, duration, "object_in")

    def _removed(self, gray, mask, now, duration, outside=False):
        blobs, global_light = local_blobs(gray, self.armed, mask)
        if global_light:
            self.armed[mask > 0] = gray[mask > 0]
            self._clear("out")
            return []
        if outside and _object_still_inside(self.armed, gray, blobs, mask):
            self._clear("out")
            return []
        return self._track("out", blobs, now, duration, "object_removed")

    def _track(self, which: str, blobs, now: float, duration: float, kind: str):
        tracks = getattr(self, f"{which}_tracks")
        used = set()
        for track in tracks:
            match_at = None
            for index, blob in enumerate(blobs):
                if index in used or not _same_spot(track["center"], blob["center"]):
                    continue
                if match_at is None or blob["area"] > blobs[match_at]["area"]:
                    match_at = index
            if match_at is None:
                track["misses"] += 1
            else:
                used.add(match_at)
                track["misses"] = 0
        for index, blob in enumerate(blobs):
            if index not in used:
                tracks.append({"center": blob["center"], "since": now, "alarmed": False, "misses": 0})
        kept = [track for track in tracks if track["misses"] < MISS_LIMIT]
        tracks[:] = kept
        if not tracks:
            self._clear(which)
            return []
        lead = max(tracks, key=lambda track: now - track["since"])
        self.elapsed = max(0.0, now - lead["since"])
        self.watching = True
        setattr(self, f"{which}_center", lead["center"])
        setattr(self, f"{which}_since", lead["since"])
        setattr(self, f"{which}_alarmed", lead["alarmed"])
        setattr(self, f"{which}_misses", lead["misses"])
        ready = [
            track
            for track in tracks
            if not track["alarmed"] and track["misses"] == 0 and now - track["since"] >= duration
        ]
        if not ready:
            return []
        for track in ready:
            track["alarmed"] = True
        setattr(self, f"{which}_alarmed", True)
        self.red_until = now + RED_HOLD
        return [kind]

    def _clear(self, which: str) -> None:
        setattr(self, f"{which}_center", None)
        setattr(self, f"{which}_since", None)
        setattr(self, f"{which}_alarmed", False)
        setattr(self, f"{which}_misses", 0)
        setattr(self, f"{which}_tracks", [])
        self.elapsed = 0.0
        self.watching = False
