import uuid

from django.conf import settings
from django.db import models


def avatar_upload_to(instance, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        ext = "jpg"
    return f"avatars/{instance.user_id}-{uuid.uuid4().hex}.{ext}"


class Zone(models.Model):
    name = models.CharField(max_length=80, default="Door")
    camera_number = models.PositiveIntegerField(unique=True)
    points = models.JSONField(default=list)
    dwell_seconds = models.PositiveIntegerField(default=60)
    active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return f"Camera {self.camera_number} door"


class AnalyticsRule(models.Model):
    ZONE = "zone"
    LINE_CROSS = "line_cross"
    OBJECT_IN = "object_in"
    OBJECT_REMOVED = "object_removed"
    PEOPLE_COUNT = "people_count"
    CROWD = "crowd"
    KINDS = (
        (ZONE, "Zone"),
        (LINE_CROSS, "Line crossing"),
        (OBJECT_IN, "Object in field"),
        (OBJECT_REMOVED, "Remove object"),
        (PEOPLE_COUNT, "People counting"),
        (CROWD, "Crowd detection"),
    )
    FORWARD = "forward"
    BACKWARD = "backward"
    ANY = "any"
    DIRECTIONS = (
        (FORWARD, "Forward"),
        (BACKWARD, "Backward"),
        (ANY, "Any"),
    )
    TOUCH = "touch"
    INSIDE = "inside"
    COVERAGE = (
        (TOUCH, "Any part in the area"),
        (INSIDE, "Whole object inside the area"),
    )

    camera_number = models.PositiveIntegerField()
    kind = models.CharField(max_length=32, choices=KINDS)
    points = models.JSONField(default=list)
    direction = models.CharField(max_length=16, choices=DIRECTIONS, default=ANY)
    coverage = models.CharField(max_length=16, choices=COVERAGE, default=TOUCH)
    duration_seconds = models.PositiveIntegerField(default=60)
    max_people = models.PositiveIntegerField(default=5)
    active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["camera_number", "kind"], name="uniq_analytics_camera_kind"),
        ]

    def __str__(self) -> str:
        return f"Camera {self.camera_number} {self.kind}"


ALARM_KIND_LABELS = {
    "zone": "Zone",
    "line_cross": "Line crossing",
    "object_in": "Object in field",
    "object_removed": "Remove object",
    "people_count": "People counting",
    "crowd": "Crowd detection",
    "blacklist": "Blacklist",
}


class Nvr(models.Model):
    host = models.CharField(max_length=64)
    username = models.CharField(max_length=128)
    password = models.CharField(max_length=128)

    def __str__(self) -> str:
        return self.host


class Person(models.Model):
    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"

    name = models.CharField(max_length=120)
    photo = models.ImageField(upload_to="people/")
    embedding = models.BinaryField()
    list_status = models.CharField(max_length=16, default=WHITELIST)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name


class PersonSample(models.Model):
    person = models.ForeignKey(Person, related_name="samples", on_delete=models.CASCADE)
    photo = models.ImageField(upload_to="people/samples/")
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.person} sample"


class Alarm(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    camera_number = models.PositiveIntegerField()
    track_id = models.IntegerField()
    snapshot = models.ImageField(upload_to="alarms/")
    face_crop = models.ImageField(upload_to="alarms/faces/", blank=True)
    face_embedding = models.BinaryField(null=True, blank=True)
    person = models.ForeignKey(Person, null=True, blank=True, on_delete=models.SET_NULL)
    matched_name = models.CharField(max_length=120, default="Unknown")
    score = models.FloatField(null=True, blank=True)
    kind = models.CharField(max_length=32, default="zone")

    class Meta:
        ordering = ["-created_at"]

    def kind_label(self) -> str:
        return ALARM_KIND_LABELS.get(self.kind, "Zone")

    def __str__(self) -> str:
        return f"Camera {self.camera_number} track {self.track_id}"


class FaceCapture(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    camera_number = models.PositiveIntegerField()
    track_id = models.IntegerField()
    face_crop = models.ImageField(upload_to="captures/")
    embedding = models.BinaryField()
    det_score = models.FloatField(default=0)
    quality = models.FloatField(default=0)
    match_score = models.FloatField(null=True, blank=True)
    matched_name = models.CharField(max_length=120, blank=True, default="")
    person = models.ForeignKey(Person, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Camera {self.camera_number} {self.matched_name or 'Unknown'}"


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, related_name="profile", on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to=avatar_upload_to, blank=True)

    def __str__(self) -> str:
        return f"{self.user} profile"


class FeatureGrant(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name="feature_grants", on_delete=models.CASCADE)
    feature = models.CharField(max_length=32)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "feature"], name="uniq_user_feature"),
        ]

    def __str__(self) -> str:
        return f"{self.user} {self.feature}"
