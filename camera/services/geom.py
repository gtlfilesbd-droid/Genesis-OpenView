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
