class PeopleCounter:
    """Count each person whose foot is inside the area on this frame."""

    def update(self, present: dict[int, bool]) -> int:
        return sum(1 for inside in present.values() if inside)


class CrowdTimer:
    """Alarm once when the count stays at or above the limit for the full wait.

    One or two quiet updates do not reset the wait. The clock starts again only
    after the count stays below the limit for three updates in a row.
    """

    def __init__(self, misses: int = 3) -> None:
        self.misses = misses
        self.since: float | None = None
        self.alarmed = False
        self.below = 0
        self.elapsed = 0.0

    def update(self, count: int, now: float, max_people: int, duration: float) -> bool:
        if count >= max_people:
            self.below = 0
            if self.since is None:
                self.since = now
                self.alarmed = False
            self.elapsed = max(0.0, now - self.since)
            if not self.alarmed and self.elapsed >= duration:
                self.alarmed = True
                return True
            return False
        if self.since is None:
            self.elapsed = 0.0
            return False
        self.below += 1
        self.elapsed = max(0.0, now - self.since)
        if self.below >= self.misses:
            self.since = None
            self.alarmed = False
            self.below = 0
            self.elapsed = 0.0
        return False
