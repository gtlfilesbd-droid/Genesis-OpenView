import cv2
import numpy as np

MIN_AREA_RATIO = 0.004
MAX_AREA_RATIO = 0.45
LIGHT_RATIO = 0.60
DIFF_THRESH = 28
CENTER_TOLERANCE = 0.02
PERSON_GROW = 0.18
MOVING_RATIO = 0.35


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


def edge_energy(gray: np.ndarray, blob: np.ndarray) -> float:
    edges = cv2.Canny(gray, 40, 120)
    pixels = int(np.count_nonzero(blob))
    if pixels == 0:
        return 0.0
    return float(np.count_nonzero((edges > 0) & (blob > 0))) / pixels


def local_blobs(current: np.ndarray, reference: np.ndarray, mask: np.ndarray):
    changed = changed_mask(current, reference, mask)
    if is_global_change(changed, mask):
        return [], True
    roi = max(1, int(np.count_nonzero(mask)))
    found = []
    contours, _hierarchy = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = current.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA_RATIO * roi or area > MAX_AREA_RATIO * roi:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (moments["m10"] / moments["m00"] / width, moments["m01"] / moments["m00"] / height)
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        current_edges = edge_energy(current, blob)
        reference_edges = edge_energy(reference, blob)
        found.append({"center": center, "area": area, "added": current_edges >= reference_edges})
    return found, False


class FieldMonitor:
    """Stationary object appear and disappear inside a polygon.

    MOG2 is the short-term model. A long-term reference keeps a placed object
    visible after the short-term model has absorbed it. A whole-region brightness
    jump is treated as a light change and does not alarm.
    """

    def __init__(self) -> None:
        self.mog = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=16, detectShadows=True)
        self.long_term = None
        self.armed = None
        self.in_center = None
        self.in_since = None
        self.in_alarmed = False
        self.out_center = None
        self.out_since = None
        self.out_alarmed = False

    def update(self, frame, points, person_boxes, now: float, duration: float, want_in: bool, want_removed: bool):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        height, width = gray.shape[:2]
        mask = apply_person_mask(polygon_mask(height, width, points), person_boxes, width, height)
        if self.long_term is None or self.long_term.shape != gray.shape:
            self.long_term = gray.copy()
            self.armed = gray.copy()
            self.mog.apply(gray, learningRate=0.05)
            return []
        foreground = self.mog.apply(gray, learningRate=0.05)
        foreground[foreground == 127] = 0
        moving = (foreground > 0) & (mask > 0)
        alarms = []
        if want_in:
            alarms.extend(self._appear(gray, mask, moving, now, duration, width, height))
        if want_removed and self.armed is not None:
            alarms.extend(self._removed(gray, mask, now, duration))
        return alarms

    def _appear(self, gray, mask, moving, now, duration, width, height):
        blobs, global_light = local_blobs(gray, self.long_term, mask)
        if global_light:
            self.long_term[mask > 0] = gray[mask > 0]
            self._clear("in")
            return []
        stationary = []
        for blob in blobs:
            if not blob["added"]:
                continue
            if self._moving_ratio(moving, blob["center"], width, height) > MOVING_RATIO:
                continue
            stationary.append(blob)
        quiet = (cv2.absdiff(gray, self.long_term) < 12) & (mask > 0)
        self.long_term[quiet] = gray[quiet]
        return self._track("in", stationary, now, duration, "object_in")

    def _removed(self, gray, mask, now, duration):
        blobs, global_light = local_blobs(gray, self.armed, mask)
        if global_light:
            self.armed[mask > 0] = gray[mask > 0]
            self._clear("out")
            return []
        missing = [blob for blob in blobs if not blob["added"]]
        return self._track("out", missing, now, duration, "object_removed")

    def _moving_ratio(self, moving, center, width: int, height: int) -> float:
        cx = int(center[0] * width)
        cy = int(center[1] * height)
        radius = 12
        patch = moving[max(0, cy - radius) : cy + radius, max(0, cx - radius) : cx + radius]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch)) / patch.size

    def _track(self, which: str, blobs, now: float, duration: float, kind: str):
        center_name = f"{which}_center"
        since_name = f"{which}_since"
        alarm_name = f"{which}_alarmed"
        if not blobs:
            self._clear(which)
            return []
        blob = max(blobs, key=lambda item: item["area"])
        center = blob["center"]
        previous = getattr(self, center_name)
        if previous is None or (center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2 > CENTER_TOLERANCE**2:
            setattr(self, center_name, center)
            setattr(self, since_name, now)
            setattr(self, alarm_name, False)
            return []
        if getattr(self, alarm_name):
            return []
        if now - getattr(self, since_name) >= duration:
            setattr(self, alarm_name, True)
            return [kind]
        return []

    def _clear(self, which: str) -> None:
        setattr(self, f"{which}_center", None)
        setattr(self, f"{which}_since", None)
        setattr(self, f"{which}_alarmed", False)
