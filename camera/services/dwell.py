class DwellTracker:
    """Times how long each track id stays inside the door zone."""

    def __init__(self) -> None:
        self.visits: dict[int, dict] = {}

    def update(self, present: dict[int, bool], now: float, dwell_seconds: float) -> list[int]:
        alarms: list[int] = []
        for track_id, inside in present.items():
            if not inside:
                self.visits.pop(track_id, None)
                continue
            visit = self.visits.get(track_id)
            if visit is None:
                self.visits[track_id] = {"since": now, "alarmed": False}
                continue
            if not visit["alarmed"] and now - visit["since"] >= dwell_seconds:
                visit["alarmed"] = True
                alarms.append(track_id)
        for track_id in list(self.visits):
            if track_id not in present:
                self.visits.pop(track_id, None)
        return alarms

    def elapsed(self, track_id: int, now: float) -> float | None:
        visit = self.visits.get(track_id)
        if visit is None:
            return None
        return now - visit["since"]
