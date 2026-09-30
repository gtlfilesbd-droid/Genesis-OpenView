from django.urls import path

from . import views

urlpatterns = [
    path("", views.live, name="live"),
    path("stream/", views.stream, name="stream"),
    path("status/", views.status, name="status"),
    path("control/", views.control, name="control"),
    path("objects/", views.objects, name="objects"),
    path("zone/", views.zone, name="zone"),
    path("zone/frame/", views.zone_frame, name="zone_frame"),
    path("alarms/", views.alarms, name="alarms"),
    path("recognition/", views.recognition, name="recognition"),
    path("recognition/<int:pk>/list/", views.capture_classify, name="capture_classify"),
    path("people/", views.people, name="people"),
    path("people/<int:pk>/delete/", views.person_delete, name="person_delete"),
    path("search/", views.search, name="search"),
]
