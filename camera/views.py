import ipaddress
import json
import time
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout, update_session_auth_hash
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.db import IntegrityError, close_old_connections
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from .access import (
    FEATURES,
    allowed_features,
    apply_avatar,
    clean_avatar,
    clean_username,
    initial_for,
    is_last_active_staff,
    public_media_url,
    password_problem,
    profile_for,
    replace_grants,
    require_feature,
    require_login,
    require_staff,
)
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


@require_feature("live")
def live(request):
    engine.ensure_defaults()
    return render(
        request,
        "camera/live.html",
        {"camera": engine.camera_number, "stream": engine.stream},
    )


@require_feature("live")
def stream(request):
    def frames():
        close_old_connections()
        while True:
            jpeg = engine.current_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    response = StreamingHttpResponse(frames(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-store"
    return response


@require_feature("live", json=True)
def status(request):
    engine.ensure_defaults()
    return JsonResponse(engine.status())


@require_feature("live")
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


@require_feature("live", json=True)
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
    "object_removed": "Click the area corners. Choose a picture change, or an alarm only after the whole object has left the area.",
    "people_count": "Click the area corners. The live view shows how many people are standing in the shape.",
    "crowd": "Click the area corners. An alarm fires when this many people stay in the area for the wait.",
}


@require_feature("analytics")
def zone(request):
    camera = request.GET.get("camera") or request.POST.get("camera") or ""
    stream_name = request.GET.get("stream") or request.POST.get("stream") or ""
    query = "kind=zone"
    if camera:
        query += f"&camera={camera}"
    if stream_name in ("sub", "main"):
        query += f"&stream={stream_name}"
    return redirect(f"/analytics/?{query}")


@require_feature("analytics")
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
            if kind == AnalyticsRule.OBJECT_REMOVED:
                if coverage not in dict(AnalyticsRule.REMOVE_COVERAGE):
                    coverage = AnalyticsRule.TOUCH
            elif kind == AnalyticsRule.OBJECT_IN:
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
            "needs_remove_coverage": kind == AnalyticsRule.OBJECT_REMOVED,
            "coverage": saved.coverage if saved else AnalyticsRule.TOUCH,
            "coverages": [{"id": key, "label": label} for key, label in AnalyticsRule.COVERAGE],
            "remove_coverages": [{"id": key, "label": label} for key, label in AnalyticsRule.REMOVE_COVERAGE],
            "direction": saved.direction if saved else AnalyticsRule.ANY,
            "directions": [{"id": key, "label": label} for key, label in AnalyticsRule.DIRECTIONS],
            "max_people": saved.max_people if saved else 5,
            "hint": _ANALYTICS_HINTS[kind],
            "kind_label": dict(AnalyticsRule.KINDS)[kind],
        },
    )


@require_feature("analytics")
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


@require_feature("alarms")
def alarms(request):
    alarms, page_numbers = _page(request, Alarm.objects.defer("face_embedding"), ALARM_PAGE_SIZE)
    return render(request, "camera/alarms.html", {"alarms": alarms, "page_numbers": page_numbers})


@require_feature("recognition")
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


@require_feature("recognition")
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


@require_feature("recognition")
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


@require_feature("people")
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


@require_feature("people")
@require_POST
def person_delete(request, pk):
    get_object_or_404(Person, pk=pk).delete()
    return redirect("people")


@require_feature("people")
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


@require_feature("people")
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


@require_feature("people")
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


@require_feature("settings")
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


@require_feature("search")
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


def _next_url(request) -> str:
    candidate = request.POST.get("next") or request.GET.get("next") or ""
    if url_has_allowed_host_and_scheme(candidate, allowed_hosts={request.get_host()}):
        return candidate
    return reverse("portal")


def login_view(request):
    if request.user.is_authenticated and request.user.is_active:
        return redirect("portal")
    error = ""
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        account = authenticate(request, username=username, password=password)
        if account is None:
            error = "Those details were not recognised."
        else:
            login(request, account)
            return redirect(_next_url(request))
    return render(
        request,
        "camera/login.html",
        {"error": error, "next": request.GET.get("next", "")},
    )


@require_POST
def logout_view(request):
    logout(request)
    return redirect("login")


@require_login
def portal(request):
    features = allowed_features(request.user)
    recent_alarms = None
    if any(item["code"] == "alarms" for item in features):
        since = timezone.now() - timedelta(hours=24)
        recent_alarms = Alarm.objects.filter(created_at__gte=since).count()
    return render(
        request,
        "camera/portal.html",
        {"features": features, "recent_alarms": recent_alarms},
    )


def _user_rows():
    rows = []
    accounts = get_user_model().objects.order_by("username").prefetch_related("feature_grants")
    for account in accounts:
        profile = profile_for(account)
        rows.append(
            {
                "account": account,
                "profile": profile,
                "avatar_url": public_media_url(profile.avatar),
                "initial": initial_for(account),
                "granted": {grant.feature for grant in account.feature_grants.all()},
                "last_staff": is_last_active_staff(account),
            }
        )
    return rows


def _create_account(request):
    User = get_user_model()
    username, problem = clean_username(request.POST.get("username", ""))
    if problem:
        messages.error(request, problem)
        return
    if User.objects.filter(username=username).exists():
        messages.error(request, "That username is already in use.")
        return
    password = request.POST.get("password", "")
    problem = password_problem(password, User(username=username))
    if problem:
        messages.error(request, problem)
        return
    upload = request.FILES.get("avatar")
    if upload is not None:
        problem = clean_avatar(upload)
        if problem:
            messages.error(request, problem)
            return
    try:
        account = User.objects.create_user(
            username=username,
            password=password,
            is_staff=request.POST.get("is_staff") == "on",
            is_active=True,
        )
    except IntegrityError:
        messages.error(request, "That username is already in use.")
        return
    replace_grants(account, request.POST.getlist("features"))
    if upload is not None:
        apply_avatar(profile_for(account), upload=upload)
    messages.success(request, f"Created {username}.")


def _save_account(request):
    User = get_user_model()
    raw = request.POST.get("user_id")
    try:
        account = User.objects.get(pk=int(raw))
    except (TypeError, ValueError, User.DoesNotExist):
        messages.error(request, "That account was not found.")
        return
    wants_staff = request.POST.get("is_staff") == "on"
    wants_active = request.POST.get("is_active") == "on"
    if is_last_active_staff(account) and (not wants_staff or not wants_active):
        messages.error(request, "The last admin has to stay an active admin.")
        return
    password = request.POST.get("password", "")
    if password:
        problem = password_problem(password, account)
        if problem:
            messages.error(request, problem)
            return
    upload = request.FILES.get("avatar")
    remove = request.POST.get("remove_avatar") == "on"
    if upload is not None:
        problem = clean_avatar(upload)
        if problem:
            messages.error(request, problem)
            return
    account.is_staff = wants_staff
    account.is_active = wants_active
    if password:
        account.set_password(password)
    account.save()
    replace_grants(account, request.POST.getlist("features"))
    profile = profile_for(account)
    if upload is not None:
        apply_avatar(profile, upload=upload)
    elif remove:
        apply_avatar(profile, remove=True)
    messages.success(request, f"Saved {account.username}.")


@require_staff
def users(request):
    if request.method == "POST":
        action = request.POST.get("action", "")
        if action == "create":
            _create_account(request)
        elif action == "save":
            _save_account(request)
        return redirect("users")
    return render(request, "camera/users.html", {"rows": _user_rows(), "features": FEATURES})


@require_login
def account(request):
    profile = profile_for(request.user)
    if request.method == "POST":
        action = request.POST.get("action", "")
        if action == "avatar":
            upload = request.FILES.get("avatar")
            remove = request.POST.get("remove_avatar") == "on"
            if upload is None and not remove:
                messages.error(request, "Choose a picture first.")
            else:
                problem = apply_avatar(profile, upload=upload, remove=remove)
                if problem:
                    messages.error(request, problem)
                elif upload is not None:
                    messages.success(request, "Picture updated.")
                else:
                    messages.success(request, "Picture removed.")
        elif action == "password":
            current = request.POST.get("current_password", "")
            new = request.POST.get("new_password", "")
            again = request.POST.get("confirm_password", "")
            if not request.user.check_password(current):
                messages.error(request, "Current password is wrong.")
            elif new != again:
                messages.error(request, "New passwords do not match.")
            else:
                problem = password_problem(new, request.user)
                if problem:
                    messages.error(request, problem)
                else:
                    request.user.set_password(new)
                    request.user.save()
                    update_session_auth_hash(request, request.user)
                    messages.success(request, "Password updated.")
        return redirect("account")
    return render(
        request,
        "camera/account.html",
        {
            "profile": profile,
            "avatar_url": public_media_url(profile.avatar),
            "initial": initial_for(request.user),
        },
    )
