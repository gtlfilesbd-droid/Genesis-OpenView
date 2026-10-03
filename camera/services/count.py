class PeopleCounter:
    """Count a person only after their foot stays inside for a few frames."""

    def __init__(self, frames: int = 3) -> None:
        self.frames = frames
        self.inside_frames: dict[int, int] = {}

    def update(self, present: dict[int, bool]) -> int:
        count = 0
        for track_id, inside in present.items():
            if inside:
                self.inside_frames[track_id] = self.inside_frames.get(track_id, 0) + 1
            else:
                self.inside_frames[track_id] = 0
            if self.inside_frames[track_id] >= self.frames:
                count += 1
        for track_id in list(self.inside_frames):
            if track_id not in present:
                self.inside_frames.pop(track_id, None)
        return count


class CrowdTimer:
    """Alarm once when the count stays at or above the limit for the full wait.

    A short dip does not reset the timer. The wait starts again only after the
    count stays below the limit for a full second.
    """

    def __init__(self, grace: float = 1.0) -> None:
        self.grace = grace
        self.since: float | None = None
        self.alarmed = False
        self.below_since: float | None = None

    def update(self, count: int, now: float, max_people: int, duration: float) -> bool:
        if count >= max_people:
            self.below_since = None
            if self.since is None:
                self.since = now
                self.alarmed = False
            if not self.alarmed and now - self.since >= duration:
                self.alarmed = True
                return True
            return False
        if self.since is None:
            return False
        if self.below_since is None:
            self.below_since = now
        if now - self.below_since >= self.grace:
            self.since = None
            self.alarmed = False
            self.below_since = None
        return False
