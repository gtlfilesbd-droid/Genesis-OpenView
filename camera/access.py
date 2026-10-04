from functools import wraps

from django.conf import settings
from django.contrib.auth import get_user_model, logout
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import redirect

from .models import FeatureGrant, UserProfile

FEATURES = (
    {
        "code": "live",
        "label": "Live",
        "url_name": "live",
        "view_names": ("live", "stream", "status", "control", "objects"),
        "summary": "Watch the camera and choose what it detects.",
    },
    {
        "code": "analytics",
        "label": "Analytics",
        "url_name": "analytics",
        "view_names": ("analytics", "zone", "zone_frame"),
        "summary": "Draw zones, lines, counts, and crowd rules.",
    },
    {
        "code": "alarms",
        "label": "Alarms",
        "url_name": "alarms",
        "view_names": ("alarms",),
        "summary": "Review the events the cameras have raised.",
    },
    {
        "code": "recognition",
        "label": "Recognition",
        "url_name": "recognition",
        "view_names": ("recognition", "capture_classify", "capture_sample"),
        "summary": "Match captured faces with people you know.",
    },
    {
        "code": "people",
        "label": "People",
        "url_name": "people",
        "view_names": ("people", "person_delete", "person_list", "person_samples", "sample_delete"),
        "summary": "Keep the whitelist and the blacklist.",
    },
    {
        "code": "search",
        "label": "Search",
        "url_name": "search",
        "view_names": ("search",),
        "summary": "Find a person from a photograph.",
    },
    {
        "code": "settings",
        "label": "Settings",
        "url_name": "settings",
        "view_names": ("settings",),
        "summary": "Save the NVR address and its login.",
    },
)

FEATURE_CODES = {item["code"] for item in FEATURES}
AVATAR_MAX_BYTES = 2 * 1024 * 1024
AVATAR_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp"}
AVATAR_FORMATS = {"JPEG", "PNG", "WEBP"}
_username_validator = UnicodeUsernameValidator()


def allowed_features(user):
    if not getattr(user, "is_authenticated", False) or not user.is_active:
        return []
    if user.is_staff:
        return list(FEATURES)
    codes = set(user.feature_grants.values_list("feature", flat=True))
    return [item for item in FEATURES if item["code"] in codes]


def can(user, code: str) -> bool:
    if not getattr(user, "is_authenticated", False) or not user.is_active:
        return False
    if code not in FEATURE_CODES:
        return False
    if user.is_staff:
        return True
    return user.feature_grants.filter(feature=code).exists()


def profile_for(user):
    profile, _created = UserProfile.objects.get_or_create(user=user)
    return profile


def public_media_url(field) -> str:
    name = getattr(field, "name", "") if field else ""
    if not name:
        return ""
    prefix = settings.MEDIA_URL or "/media/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + name.lstrip("/")


def initial_for(user) -> str:
    name = (getattr(user, "username", "") or "?").strip()
    return name[:1].upper() or "?"


def is_last_active_staff(user) -> bool:
    if not user.is_staff or not user.is_active:
        return False
    return get_user_model().objects.filter(is_staff=True, is_active=True).count() == 1


def replace_grants(user, codes) -> None:
    keep = [code for code in codes if code in FEATURE_CODES]
    FeatureGrant.objects.filter(user=user).exclude(feature__in=keep).delete()
    have = set(FeatureGrant.objects.filter(user=user).values_list("feature", flat=True))
    FeatureGrant.objects.bulk_create(
        [FeatureGrant(user=user, feature=code) for code in keep if code not in have]
    )


def clean_username(value: str) -> tuple[str, str]:
    username = (value or "").strip()
    if not username:
        return "", "Username is required."
    if len(username) > 150:
        return "", "Username is too long."
    try:
        _username_validator(username)
    except ValidationError:
        return "", "Use letters, numbers, and @ . + - _ only."
    return username, ""


def password_problem(password: str, user=None) -> str:
    try:
        validate_password(password, user)
    except ValidationError as exc:
        return " ".join(exc.messages)
    return ""


def clean_avatar(upload) -> str:
    if upload.size > AVATAR_MAX_BYTES:
        return "Picture must be 2 MB or smaller."
    content_type = (getattr(upload, "content_type", "") or "").split(";")[0].strip().lower()
    if content_type not in AVATAR_CONTENT_TYPES:
        return "Use a JPEG, PNG, or WEBP picture."
    from PIL import Image

    kind = ""
    try:
        upload.seek(0)
        image = Image.open(upload)
        kind = (image.format or "").upper()
        image.verify()
    except Exception:
        return "That file is not a readable picture."
    finally:
        upload.seek(0)
    if kind not in AVATAR_FORMATS:
        return "Use a JPEG, PNG, or WEBP picture."
    return ""


def apply_avatar(profile, upload=None, remove: bool = False) -> str:
    if upload is not None:
        problem = clean_avatar(upload)
        if problem:
            return problem
        if profile.avatar:
            profile.avatar.delete(save=False)
        profile.avatar = upload
        profile.save()
        return ""
    if remove and profile.avatar:
        profile.avatar.delete(save=True)
    return ""


def _refuse_anonymous(request, json: bool):
    user = request.user
    if user.is_authenticated and not user.is_active:
        logout(request)
    if not request.user.is_authenticated:
        if json:
            return JsonResponse({"ok": False}, status=401)
        return redirect_to_login(request.get_full_path(), settings.LOGIN_URL)
    return None


def require_login(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        refused = _refuse_anonymous(request, json=False)
        if refused is not None:
            return refused
        return view(request, *args, **kwargs)

    return wrapper


def require_staff(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        refused = _refuse_anonymous(request, json=False)
        if refused is not None:
            return refused
        if not request.user.is_staff:
            from django.contrib import messages

            messages.error(request, "Only an admin can open that page.")
            return redirect("portal")
        return view(request, *args, **kwargs)

    return wrapper


def require_feature(code: str, *, json: bool = False):
    def decorator(view):
        @wraps(view)
        def wrapper(request, *args, **kwargs):
            refused = _refuse_anonymous(request, json=json)
            if refused is not None:
                return refused
            if not can(request.user, code):
                if json:
                    return JsonResponse({"ok": False}, status=403)
                from django.contrib import messages

                messages.error(request, "You do not have access to that page.")
                return redirect("portal")
            return view(request, *args, **kwargs)

        return wrapper

    return decorator


def console(request):
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated or not user.is_active:
        return {
            "nav_features": [],
            "is_console_admin": False,
            "account_initial": "",
            "account_avatar_url": "",
        }
    avatar_url = ""
    try:
        avatar_url = public_media_url(user.profile.avatar)
    except UserProfile.DoesNotExist:
        avatar_url = ""
    return {
        "nav_features": allowed_features(user),
        "is_console_admin": user.is_staff,
        "account_initial": initial_for(user),
        "account_avatar_url": avatar_url,
    }
