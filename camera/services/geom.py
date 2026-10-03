LEFT_ANKLE = 15
RIGHT_ANKLE = 16
ANKLE_CONF = 0.4
LINE_MIN_MOVE = 0.004


def point_in_polygon(x: float, y: float, points: list) -> bool:
    if len(points) < 3:
        return False
    inside = False
    previous = len(points) - 1
    for index in range(len(points)):
        xi, yi = float(points[index][0]), float(points[index][1])
        xj, yj = float(points[previous][0]), float(points[previous][1])
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        previous = index
    return inside


def box_bottom_center(coords, width: float, height: float) -> tuple[float, float]:
    x1, _y1, x2, y2 = coords
    return (float(x1) + float(x2)) / 2 / width, float(y2) / height


def box_matches_area(coords, width: float, height: float, points, coverage: str = "touch") -> bool:
    """Touch counts any overlap. Inside counts only when the whole box is in the area."""
    x1, y1, x2, y2 = [float(value) for value in coords]
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return False
    if coverage == "inside":
        corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
        return all(point_in_polygon(x / width, y / height, points) for x, y in corners)
    for across in (0.0, 0.5, 1.0):
        for down in (0.0, 0.5, 1.0):
            if point_in_polygon(
                (x1 + (x2 - x1) * across) / width,
                (y1 + (y2 - y1) * down) / height,
                points,
            ):
                return True
    return False


def box_in_polygon(coords, width: float, height: float, points) -> bool:
    """True when the person's body sits in the area, even if their feet are outside it."""
    x1, y1, x2, y2 = [float(value) for value in coords]
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return False
    hits = 0
    for across in (0.3, 0.5, 0.7):
        for down in (0.35, 0.55, 0.75):
            if point_in_polygon(
                (x1 + (x2 - x1) * across) / width,
                (y1 + (y2 - y1) * down) / height,
                points,
            ):
                hits += 1
    return hits >= 2


def best_ankle(keypoints, width: float, height: float):
    if keypoints is None:
        return None
    points = list(keypoints)
    if len(points) <= RIGHT_ANKLE:
        return None
    chosen = None
    for index in (LEFT_ANKLE, RIGHT_ANKLE):
        row = points[index]
        x, y, conf = float(row[0]), float(row[1]), float(row[2])
        if conf < ANKLE_CONF:
            continue
        if chosen is None or conf > chosen[0]:
            chosen = (conf, x / width, y / height)
    if chosen is None:
        return None
    return chosen[1], chosen[2]


def foot_point(coords, keypoints, width: float, height: float) -> tuple[float, float]:
    ankle = best_ankle(keypoints, width, height)
    if ankle is not None:
        return ankle
    return box_bottom_center(coords, width, height)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    area = area_a + area_b - inter
    if area <= 0:
        return 0.0
    return inter / area


def side_of_line(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def segments_intersect(p1, p2, start, end) -> bool:
    def orient(ax, ay, bx, by, cx, cy) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    o1 = orient(p1[0], p1[1], p2[0], p2[1], start[0], start[1])
    o2 = orient(p1[0], p1[1], p2[0], p2[1], end[0], end[1])
    o3 = orient(start[0], start[1], end[0], end[1], p1[0], p1[1])
    o4 = orient(start[0], start[1], end[0], end[1], p2[0], p2[1])
    return o1 * o2 < 0 and o3 * o4 < 0


def line_cross(prev, curr, start, end, min_move: float = LINE_MIN_MOVE) -> str | None:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if dx * dx + dy * dy < min_move * min_move:
        return None
    before = side_of_line(prev[0], prev[1], start[0], start[1], end[0], end[1])
    after = side_of_line(curr[0], curr[1], start[0], start[1], end[0], end[1])
    if before == 0 or after == 0 or (before > 0) == (after > 0):
        return None
    if not segments_intersect(prev, curr, start, end):
        return None
    return "forward" if after > 0 else "backward"
