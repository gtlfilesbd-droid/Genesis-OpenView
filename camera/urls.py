from django.urls import path

from . import views

urlpatterns = [
    path("", views.live, name="live"),
    path("stream/", views.stream, name="stream"),
    path("status/", views.status, name="status"),
    path("control/", views.control, name="control"),
    path("objects/", views.objects, name="objects"),
    path("analytics/", views.analytics, name="analytics"),
    path("zone/", views.zone, name="zone"),
    path("zone/frame/", views.zone_frame, name="zone_frame"),
    path("alarms/", views.alarms, name="alarms"),
    path("recognition/", views.recognition, name="recognition"),
    path("recognition/<int:pk>/list/", views.capture_classify, name="capture_classify"),
    path("recognition/<int:pk>/sample/", views.capture_sample, name="capture_sample"),
    path("people/", views.people, name="people"),
    path("people/<int:pk>/delete/", views.person_delete, name="person_delete"),
    path("people/<int:pk>/list/", views.person_list, name="person_list"),
    path("people/<int:pk>/samples/", views.person_samples, name="person_samples"),
    path("people/<int:pk>/samples/<int:sample_pk>/delete/", views.sample_delete, name="sample_delete"),
    path("search/", views.search, name="search"),
    path("settings/", views.settings, name="settings"),
]
