MAX_DURATION = 3600

DURATION_KINDS = {"zone", "object_in", "object_removed", "crowd"}
PERSON_KINDS = {"zone", "line_cross", "object_in", "object_removed", "people_count", "crowd"}


def total_seconds(minutes, seconds) -> int:
    try:
        whole = int(minutes)
    except (TypeError, ValueError):
        whole = 0
    try:
        part = int(seconds)
    except (TypeError, ValueError):
        part = 0
    if whole < 0:
        whole = 0
    if part < 0:
        part = 0
    return min(MAX_DURATION, max(1, whole * 60 + part))
