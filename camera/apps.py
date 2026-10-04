from django.apps import AppConfig


class CameraConfig(AppConfig):
    name = "camera"

    def ready(self):
        from . import signals  # noqa: F401
