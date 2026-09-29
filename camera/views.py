import json
import time

from django.contrib import messages
from django.core.files.base import ContentFile
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Alarm, Person, Zone
from .services.engine import engine, grab_jpeg
from .services.faces import (
    MATCH_THRESHOLD,
    current_vector,
    decode_image_bytes,
    embed_image,
    similarity,
    to_bytes,
)
from .services.gallery import load_person_vector


def _camera_number(value, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _clean_points(raw):
    try:
        points = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(points, list) or len(points) < 3:
        return None
    cleaned = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        cleaned.append([x, y])
    return cleaned


def _stream_name(value, default: str) -> str:
    return value if value in ("sub", "main") else default


def live(request):
    engine.ensure_defaults()
    return render(
        request,
        "camera/live.html",
        {"camera": engine.camera_number, "stream": engine.stream},
    )


def stream(request):
    def frames():
        while True:
            jpeg = engine.current_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.2)

    response = StreamingHttpResponse(frames(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-store"
    return response


def status(request):
    engine.ensure_defaults()
    return JsonResponse(engine.status())


@require_POST
def control(request):
    engine.ensure_defaults()
    action = request.POST.get("action", "")
    camera = _camera_number(request.POST.get("camera"), engine.camera_number)
    stream_name = _stream_name(request.POST.get("stream"), engine.stream)
    if action == "start":
        engine.start(camera, stream_name)
    elif action == "stop":
        engine.stop()
    return redirect("live")


@require_POST
def objects(request):
    try:
        class_id = int(request.POST.get("class_id", ""))
    except (TypeError, ValueError):
        return JsonResponse({"ok": False}, status=400)
    engine.toggle_class(class_id)
    return JsonResponse({"ok": True, "classes": engine.class_groups()})


def zone(request):
    engine.ensure_defaults()
    camera = _camera_number(request.GET.get("camera"), engine.camera_number)
    stream_name = _stream_name(request.GET.get("stream"), engine.stream)
    if request.method == "POST":
        camera = _camera_number(request.POST.get("camera"), camera)
        stream_name = _stream_name(request.POST.get("stream"), stream_name)
        try:
            dwell = int(request.POST.get("dwell_seconds", "60"))
        except ValueError:
            dwell = 60
        dwell = min(3600, max(1, dwell))
        if request.POST.get("action") == "remove":
            Zone.objects.filter(camera_number=camera).delete()
            messages.success(request, f"Door zone removed for camera {camera}.")
            return redirect(f"/zone/?camera={camera}&stream={stream_name}")
        points = _clean_points(request.POST.get("points"))
        if points is None:
            messages.error(request, "Click at least 3 points on the frame.")
        else:
            Zone.objects.update_or_create(
                camera_number=camera,
                defaults={
                    "name": "Door",
                    "points": points,
                    "dwell_seconds": dwell,
                    "active": True,
                },
            )
            messages.success(request, f"Door zone saved for camera {camera}.")
            return redirect(f"/zone/?camera={camera}&stream={stream_name}")
    saved = Zone.objects.filter(camera_number=camera).first()
    return render(
        request,
        "camera/zone.html",
        {
            "camera": camera,
            "stream": stream_name,
            "points": saved.points if saved else [],
            "dwell_seconds": saved.dwell_seconds if saved else 60,
        },
    )


def zone_frame(request):
    engine.ensure_defaults()
    camera = _camera_number(request.GET.get("camera"), engine.camera_number)
    stream_name = _stream_name(request.GET.get("stream"), engine.stream)
    jpeg = grab_jpeg(camera, stream_name)
    if not jpeg:
        jpeg = engine.current_jpeg()
    response = HttpResponse(jpeg, content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    return response


def alarms(request):
    return render(request, "camera/alarms.html", {"alarms": Alarm.objects.all()[:50]})


def people(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        upload = request.FILES.get("photo")
        if not name or upload is None:
            messages.error(request, "Enter a name and choose a clear face photo.")
        else:
            payload = upload.read()
            image = decode_image_bytes(payload)
            try:
                embedding = embed_image(image) if image is not None else None
            except Exception:
                embedding = None
            if embedding is None:
                messages.error(request, "No face found. Use a front-facing photo.")
            else:
                person = Person(name=name, embedding=to_bytes(embedding))
                person.photo.save(upload.name, ContentFile(payload), save=False)
                person.save()
                messages.success(request, f"Enrolled {name}.")
                return redirect("people")
    enrolled = list(Person.objects.order_by("-created_at"))
    for person in enrolled:
        person.needs_new_photo = load_person_vector(person) is None
    return render(request, "camera/people.html", {"people": enrolled})


@require_POST
def person_delete(request, pk):
    get_object_or_404(Person, pk=pk).delete()
    return redirect("people")


def search(request):
    found = None
    error = ""
    if request.method == "POST":
        upload = request.FILES.get("photo")
        if upload is None:
            error = "Choose a photo."
        else:
            image = decode_image_bytes(upload.read())
            try:
                embedding = embed_image(image) if image is not None else None
            except Exception:
                embedding = None
            if embedding is None:
                error = "No face found in that photo."
            else:
                people_scores = []
                for person in Person.objects.all():
                    stored = load_person_vector(person)
                    if stored is None:
                        continue
                    score = similarity(embedding, stored)
                    people_scores.append(
                        {"person": person, "score": score, "match": score >= MATCH_THRESHOLD}
                    )
                people_scores.sort(key=lambda item: item["score"], reverse=True)
                alarm_scores = []
                for alarm in Alarm.objects.exclude(face_embedding__isnull=True)[:200]:
                    stored = current_vector(bytes(alarm.face_embedding) if alarm.face_embedding else None)
                    if stored is None:
                        continue
                    score = similarity(embedding, stored)
                    if score >= MATCH_THRESHOLD:
                        alarm_scores.append({"alarm": alarm, "score": score})
                alarm_scores.sort(key=lambda item: item["score"], reverse=True)
                found = {"people": people_scores, "alarms": alarm_scores}
    return render(request, "camera/search.html", {"found": found, "error": error})
