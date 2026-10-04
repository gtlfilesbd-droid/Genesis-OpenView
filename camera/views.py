import ipaddress
import json
import time

from django.contrib import messages
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import Alarm, AnalyticsRule, FaceCapture, Nvr, Person, PersonSample
from .services.rules import DURATION_KINDS, total_seconds
from .services.rtsp import configured_camera, env_nvr_fields, stream_url
from .services.engine import engine, grab_jpeg, probe_stream
from .services.faces import (
    MATCH_THRESHOLD,
    current_vector,
    decode_image_bytes,
    embed_image,
    similarity,
    to_bytes,
)
from .services.gallery import add_sample, load_person_vector, load_person_vectors


def _ipv4(value: str) -> str:
    try:
        return str(ipaddress.IPv4Address(value.strip()))
    except (ValueError, ipaddress.AddressValueError):
        return ""


def _camera_number(value, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _clean_points(raw, exact=None):
    try:
        points = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(points, list):
        return None
    if exact is None and len(points) < 3:
        return None
    if exact is not None and len(points) != exact:
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


def _read_field(field) -> bytes:
    if not field:
        return b""
    field.open("rb")
    try:
        return field.read()
    finally:
        field.close()


def _list_status(value: str) -> str:
    if value in (Person.WHITELIST, Person.BLACKLIST):
        return value
    return ""


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
            time.sleep(0.05)

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


_ANALYTICS_HINTS = {
    "zone": "Click the area corners. A person is inside when an ankle, or the bottom of the box, is in the shape.",
    "line_cross": "Click the start and end of the line. The arrow is forward. A person alarms when they cross in the chosen direction.",
    "object_in": "Click the area corners. An alarm fires after something new stays in the area for the wait. People walking through are ignored.",
    "object_removed": "Click the area corners. An alarm fires if something that was in the area is gone for the wait.",
    "people_count": "Click the area corners. The live view shows how many people are standing in the shape.",
    "crowd": "Click the area corners. An alarm fires when this many people stay in the area for the wait.",
}


def zone(request):
    camera = request.GET.get("camera") or request.POST.get("camera") or ""
    stream_name = request.GET.get("stream") or request.POST.get("stream") or ""
    query = "kind=zone"
    if camera:
        query += f"&camera={camera}"
    if stream_name in ("sub", "main"):
        query += f"&stream={stream_name}"
    return redirect(f"/analytics/?{query}")


def analytics(request):
    engine.ensure_defaults()
    camera = _camera_number(request.GET.get("camera"), engine.camera_number)
    stream_name = _stream_name(request.GET.get("stream"), engine.stream)
    kind = request.GET.get("kind") or AnalyticsRule.ZONE
    if request.method == "POST":
        camera = _camera_number(request.POST.get("camera"), camera)
        stream_name = _stream_name(request.POST.get("stream"), stream_name)
        kind = request.POST.get("kind") or kind
        if kind not in dict(AnalyticsRule.KINDS):
            kind = AnalyticsRule.ZONE
        target = f"/analytics/?camera={camera}&stream={stream_name}&kind={kind}"
        if request.POST.get("action") == "remove":
            AnalyticsRule.objects.filter(camera_number=camera, kind=kind).delete()
            messages.success(request, f"{dict(AnalyticsRule.KINDS)[kind]} removed for camera {camera}.")
            return redirect(target)
        exact = 2 if kind == AnalyticsRule.LINE_CROSS else None
        points = _clean_points(request.POST.get("points"), exact=exact)
        if points is None:
            messages.error(
                request,
                "Click 2 points to draw the line." if exact == 2 else "Click at least 3 points on the frame.",
            )
        else:
            direction = request.POST.get("direction", AnalyticsRule.ANY)
            if direction not in dict(AnalyticsRule.DIRECTIONS):
                direction = AnalyticsRule.ANY
            try:
                max_people = int(request.POST.get("max_people", "5"))
            except (TypeError, ValueError):
                max_people = 5
            max_people = min(500, max(1, max_people))
            coverage = request.POST.get("coverage", AnalyticsRule.TOUCH)
            if kind in (AnalyticsRule.OBJECT_IN, AnalyticsRule.OBJECT_REMOVED):
                coverage = AnalyticsRule.TOUCH
            elif coverage not in dict(AnalyticsRule.COVERAGE):
                coverage = AnalyticsRule.TOUCH
            AnalyticsRule.objects.update_or_create(
                camera_number=camera,
                kind=kind,
                defaults={
                    "points": points,
                    "direction": direction,
                    "coverage": coverage,
                    "duration_seconds": total_seconds(request.POST.get("minutes"), request.POST.get("seconds")),
                    "max_people": max_people,
                    "active": True,
                },
            )
            messages.success(request, f"{dict(AnalyticsRule.KINDS)[kind]} saved for camera {camera}.")
            return redirect(target)
    if kind not in dict(AnalyticsRule.KINDS):
        kind = AnalyticsRule.ZONE
    saved = AnalyticsRule.objects.filter(camera_number=camera, kind=kind).first()
    duration = saved.duration_seconds if saved else 60
    return render(
        request,
        "camera/analytics.html",
        {
            "camera": camera,
            "stream": stream_name,
            "kind": kind,
            "kinds": [{"id": key, "label": label} for key, label in AnalyticsRule.KINDS],
            "points": saved.points if saved else [],
            "minutes": duration // 60,
            "seconds": duration % 60,
            "needs_duration": kind in DURATION_KINDS,
            "needs_people": kind == AnalyticsRule.CROWD,
            "is_line": kind == AnalyticsRule.LINE_CROSS,
            "needs_coverage": kind in (AnalyticsRule.ZONE, AnalyticsRule.PEOPLE_COUNT, AnalyticsRule.CROWD),
            "coverage": saved.coverage if saved else AnalyticsRule.TOUCH,
            "coverages": [{"id": key, "label": label} for key, label in AnalyticsRule.COVERAGE],
            "direction": saved.direction if saved else AnalyticsRule.ANY,
            "directions": [{"id": key, "label": label} for key, label in AnalyticsRule.DIRECTIONS],
            "max_people": saved.max_people if saved else 5,
            "hint": _ANALYTICS_HINTS[kind],
            "kind_label": dict(AnalyticsRule.KINDS)[kind],
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


ALARM_PAGE_SIZE = 50
RECOGNITION_PAGE_SIZE = 80


def _page_numbers(page, radius: int = 2) -> list:
    """Page links around the current page, with gaps marked as None."""
    last = page.paginator.num_pages
    if last <= 1:
        return []
    wanted = {1, last, page.number}
    for number in range(page.number - radius, page.number + radius + 1):
        if 1 <= number <= last:
            wanted.add(number)
    ordered = sorted(wanted)
    window = []
    previous = 0
    for number in ordered:
        if previous and number - previous > 1:
            window.append(None)
        window.append(number)
        previous = number
    return window


def _page(request, queryset, per_page):
    page = Paginator(queryset, per_page).get_page(request.GET.get("page"))
    return page, _page_numbers(page)


def _recognition_redirect(request):
    raw = (request.POST.get("page") or "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return redirect(f"{reverse('recognition')}?page={int(raw)}")
    return redirect("recognition")


def alarms(request):
    alarms, page_numbers = _page(request, Alarm.objects.defer("face_embedding"), ALARM_PAGE_SIZE)
    return render(request, "camera/alarms.html", {"alarms": alarms, "page_numbers": page_numbers})


def recognition(request):
    captures, page_numbers = _page(
        request,
        FaceCapture.objects.select_related("person").defer("embedding"),
        RECOGNITION_PAGE_SIZE,
    )
    people = Person.objects.order_by("name")
    return render(
        request,
        "camera/recognition.html",
        {"captures": captures, "people": people, "page_numbers": page_numbers},
    )


def _add_capture_sample(person, capture) -> None:
    vector = current_vector(bytes(capture.embedding) if capture.embedding else None)
    payload = _read_field(capture.face_crop)
    if vector is None or not payload:
        return
    add_sample(person, payload, vector, f"{capture.pk}.jpg")


@require_POST
def capture_classify(request, pk):
    from .services.captures import link_same_face

    capture = get_object_or_404(FaceCapture, pk=pk)
    name = (request.POST.get("name") or "").strip()
    status = _list_status(request.POST.get("list_status") or "")
    if not status or not name:
        messages.error(request, "Enter a name and choose whitelist or blacklist.")
        return _recognition_redirect(request)
    vector = current_vector(bytes(capture.embedding) if capture.embedding else None)
    if vector is None:
        messages.error(request, "This capture has no usable face vector.")
        return _recognition_redirect(request)
    payload = _read_field(capture.face_crop)
    person = Person.objects.filter(name__iexact=name).first()
    if person is None:
        if not payload:
            messages.error(request, "This capture has no face photo.")
            return _recognition_redirect(request)
        person = Person(name=name, list_status=status, embedding=to_bytes(vector))
        person.photo.save(f"{capture.pk}.jpg", ContentFile(payload), save=False)
        person.save()
    else:
        person.name = name
        person.list_status = status
        person.save(update_fields=["name", "list_status"])
    _add_capture_sample(person, capture)
    capture.person = person
    capture.matched_name = person.name
    capture.save(update_fields=["person", "matched_name"])
    link_same_face(capture, person)
    for other in FaceCapture.objects.filter(person=person).exclude(pk=capture.pk):
        _add_capture_sample(person, other)
    engine._gallery_at = 0.0
    messages.success(request, f"{person.name} is on the {status}.")
    return _recognition_redirect(request)


@require_POST
def capture_sample(request, pk):
    from .services.gallery import sample_skip_reason

    capture = get_object_or_404(FaceCapture, pk=pk)
    try:
        person_id = int(request.POST.get("person") or "")
    except ValueError:
        person_id = 0
    person = Person.objects.filter(pk=person_id).first()
    if person is None:
        messages.error(request, "Choose a person.")
        return _recognition_redirect(request)
    vector = current_vector(bytes(capture.embedding) if capture.embedding else None)
    payload = _read_field(capture.face_crop)
    if vector is None or not payload:
        messages.error(request, "This capture has no usable face photo.")
        return _recognition_redirect(request)
    reason = sample_skip_reason(person, vector)
    if reason == "duplicate":
        messages.error(request, f"This photo is already on {person.name}'s profile.")
        return _recognition_redirect(request)
    if reason == "full":
        messages.error(request, f"{person.name} already has 12 sample photos.")
        return _recognition_redirect(request)
    if reason or not add_sample(person, payload, vector, f"{capture.pk}.jpg"):
        messages.error(request, "This snap was not added.")
        return _recognition_redirect(request)
    capture.person = person
    capture.matched_name = person.name
    capture.save(update_fields=["person", "matched_name"])
    engine._gallery_at = 0.0
    messages.success(request, f"Added this snap to {person.name}.")
    return _recognition_redirect(request)


def _embed_upload(upload):
    payload = upload.read()
    image = decode_image_bytes(payload)
    try:
        embedding = embed_image(image) if image is not None else None
    except Exception:
        embedding = None
    if embedding is None:
        return None
    return payload, embedding


def people(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        status = _list_status(request.POST.get("list_status") or "") or Person.WHITELIST
        uploads = request.FILES.getlist("photos")
        if not name or not uploads:
            messages.error(request, "Enter a name and choose at least one face photo.")
        else:
            person = None
            for upload in uploads:
                found = _embed_upload(upload)
                if found is None:
                    continue
                payload, embedding = found
                if person is None:
                    person = Person(name=name, embedding=to_bytes(embedding), list_status=status)
                    person.photo.save(upload.name or "face.jpg", ContentFile(payload), save=False)
                    person.save()
                add_sample(person, payload, embedding, upload.name or f"{person.pk}.jpg")
            if person is None:
                messages.error(request, "No face found. Use front-facing photos.")
            else:
                messages.success(request, f"Enrolled {name} on the {status}.")
                return redirect("people")
    enrolled = list(Person.objects.prefetch_related("samples").order_by("-created_at"))
    for person in enrolled:
        person.needs_new_photo = load_person_vector(person) is None
    return render(request, "camera/people.html", {"people": enrolled})


@require_POST
def person_delete(request, pk):
    get_object_or_404(Person, pk=pk).delete()
    return redirect("people")


@require_POST
def person_list(request, pk):
    person = get_object_or_404(Person, pk=pk)
    status = _list_status(request.POST.get("list_status") or "")
    if not status:
        messages.error(request, "Choose whitelist or blacklist.")
    else:
        person.list_status = status
        person.save(update_fields=["list_status"])
        engine._gallery_at = 0.0
        messages.success(request, f"{person.name} is on the {status}.")
    return redirect("people")


@require_POST
def person_samples(request, pk):
    person = get_object_or_404(Person, pk=pk)
    added = 0
    for upload in request.FILES.getlist("photos"):
        found = _embed_upload(upload)
        if found is None:
            continue
        payload, embedding = found
        if add_sample(person, payload, embedding, upload.name or f"{person.pk}.jpg"):
            added += 1
    if added:
        engine._gallery_at = 0.0
        messages.success(request, f"Added {added} photo{'s' if added != 1 else ''} to {person.name}.")
    else:
        messages.error(request, "No new face photo was added.")
    return redirect("people")


@require_POST
def sample_delete(request, pk, sample_pk):
    person = get_object_or_404(Person, pk=pk)
    sample = get_object_or_404(PersonSample, pk=sample_pk, person=person)
    if person.samples.count() <= 1:
        messages.error(request, "Keep at least one sample photo.")
        return redirect("people")
    sample.delete()
    remaining = person.samples.order_by("created_at").first()
    if remaining is not None:
        payload = _read_field(remaining.photo)
        if payload:
            person.photo.save(f"cover-{person.pk}.jpg", ContentFile(payload), save=False)
        person.embedding = bytes(remaining.embedding)
        person.save()
    engine._gallery_at = 0.0
    return redirect("people")


def settings(request):
    nvr = Nvr.objects.order_by("pk").first()
    if nvr is not None:
        host, username = nvr.host, nvr.username
        password_value = nvr.password
    else:
        host, username = env_nvr_fields()
        password_value = ""
    if request.method == "POST":
        host = request.POST.get("host", "").strip()
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        if password != "":
            password_value = password
        ip = _ipv4(host)
        if not ip:
            messages.error(request, "Enter a valid IPv4 address.")
        elif not username:
            messages.error(request, "Username is required.")
        elif nvr is None and password == "":
            messages.error(request, "Password is required.")
        else:
            if nvr is None:
                Nvr.objects.create(host=ip, username=username, password=password)
            else:
                nvr.host = ip
                nvr.username = username
                if password != "":
                    nvr.password = password
                nvr.save()
                Nvr.objects.exclude(pk=nvr.pk).delete()
            was_running = engine.running or engine.starting
            resume_camera = engine.camera_number
            resume_stream = engine.stream
            if was_running:
                engine.stop()
                thread = engine._thread
                if thread is not None and thread.is_alive():
                    thread.join(timeout=10)
            try:
                camera, stream_name = configured_camera()
                probe_url, _channel = stream_url(camera, stream_name)
                reason = probe_stream(probe_url)
            except Exception:
                reason = "Could not read the camera."
            if reason:
                messages.error(request, f"NVR saved, but the camera did not open. {reason}")
            else:
                messages.success(request, "NVR saved.")
            if was_running and not reason:
                engine.start(resume_camera, resume_stream)
            else:
                with engine._lock:
                    engine.error = reason
            return redirect("settings")
    return render(
        request,
        "camera/settings.html",
        {
            "host": host,
            "username": username,
            "password": password_value,
        },
    )


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
                for person in Person.objects.prefetch_related("samples"):
                    stored_vectors = load_person_vectors(person)
                    if not stored_vectors:
                        continue
                    score = max(similarity(embedding, stored) for stored in stored_vectors)
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
