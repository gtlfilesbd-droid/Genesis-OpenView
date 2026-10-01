from django.db import models


class Zone(models.Model):
    name = models.CharField(max_length=80, default="Door")
    camera_number = models.PositiveIntegerField(unique=True)
    points = models.JSONField(default=list)
    dwell_seconds = models.PositiveIntegerField(default=60)
    active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return f"Camera {self.camera_number} door"


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
    created_at = models.DateTimeField(auto_now_add=True)
    camera_number = models.PositiveIntegerField()
    track_id = models.IntegerField()
    snapshot = models.ImageField(upload_to="alarms/")
    face_crop = models.ImageField(upload_to="alarms/faces/", blank=True)
    face_embedding = models.BinaryField(null=True, blank=True)
    person = models.ForeignKey(Person, null=True, blank=True, on_delete=models.SET_NULL)
    matched_name = models.CharField(max_length=120, default="Unknown")
    score = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Camera {self.camera_number} track {self.track_id}"


class FaceCapture(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
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
