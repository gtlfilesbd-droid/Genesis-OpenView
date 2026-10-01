import hashlib
import os
import re
import socket
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, urlunparse

ROOT = Path(__file__).resolve().parents[2]
_CHANNEL = re.compile(r"/Streaming/Channels/\d+")


def load_env_value(key: str) -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    return os.environ.get(key, "").strip()


def build_rtsp_url(host: str, username: str, password: str) -> str:
    user = quote(username, safe="")
    secret = quote(password, safe="")
    return f"rtsp://{user}:{secret}@{host}:554/Streaming/Channels/102"


def saved_nvr_url() -> str:
    try:
        from camera.models import Nvr
    except Exception:
        return ""
    try:
        nvr = Nvr.objects.order_by("pk").first()
    except Exception:
        return ""
    if nvr is None or not nvr.host or not nvr.username:
        return ""
    return build_rtsp_url(nvr.host, nvr.username, nvr.password)


def env_nvr_fields() -> tuple[str, str]:
    url = load_env_value("RTSP_URL")
    if not url or "user:password@host" in url:
        return "", ""
    parsed = urlparse(url)
    return parsed.hostname or "", unquote(parsed.username or "")


def base_rtsp_url() -> str:
    stored = saved_nvr_url()
    if stored:
        return stored
    url = load_env_value("RTSP_URL")
    if not url or "user:password@host" in url:
        raise RuntimeError("Set the NVR in Settings")
    return url


def parse_channel(url: str) -> tuple[int, str]:
    match = _CHANNEL.search(url)
    if not match:
        return 1, "sub"
    code = match.group(0).rsplit("/", 1)[-1]
    stream = "main" if code.endswith("01") else "sub"
    camera = int(code[:-2] or "1")
    return camera, stream


def configured_camera() -> tuple[int, str]:
    """Camera and stream from RTSP_URL, or camera 1 sub when that URL has no channel."""
    url = load_env_value("RTSP_URL")
    if not url or "user:password@host" in url or not _CHANNEL.search(url):
        return 1, "sub"
    return parse_channel(url)


def _auth_field(challenge: str, name: str) -> str:
    match = re.search(rf'{name}="([^"]*)"', challenge, re.IGNORECASE)
    return match.group(1) if match else ""


def _digest_authorization(user: str, password: str, uri: str, challenge: str) -> str:
    realm = _auth_field(challenge, "realm")
    nonce = _auth_field(challenge, "nonce")
    qop = _auth_field(challenge, "qop").split(",")[0].strip()
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode()).hexdigest()
    if qop:
        nc, cnonce = "00000001", "a1b2c3d4"
        response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
        return (
            f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", '
            f'response="{response}", qop={qop}, nc={nc}, cnonce="{cnonce}"'
        )
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}"'


def _rtsp_exchange(sock: socket.socket, request: str) -> tuple[int, str]:
    sock.sendall(request.encode())
    chunks: list[bytes] = []
    while b"\r\n\r\n" not in b"".join(chunks):
        data = sock.recv(4096)
        if not data:
            break
        chunks.append(data)
    head = b"".join(chunks).split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace")
    status_line = head.split("\r\n", 1)[0]
    try:
        status = int(status_line.split()[1])
    except (IndexError, ValueError):
        status = 0
    challenge = ""
    for line in head.split("\r\n")[1:]:
        if line.lower().startswith("www-authenticate:"):
            challenge = line.split(":", 1)[1].strip()
            break
    return status, challenge


def describe_rtsp(url: str) -> str:
    """Return '' when the NVR accepts the login, otherwise a short public error."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 554
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    uri = urlunparse((parsed.scheme or "rtsp", f"{host}:{port}", parsed.path or "/", "", "", ""))
    try:
        sock = socket.create_connection((host, port), timeout=5)
    except socket.timeout:
        return "The NVR did not answer."
    except OSError:
        return "Could not reach the NVR."
    sock.settimeout(5)
    try:
        request = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\nUser-Agent: AICamera\r\n\r\n"
        status, challenge = _rtsp_exchange(sock, request)
        if status == 200:
            return ""
        if status != 401 or "digest" not in challenge.lower():
            return "NVR login was rejected." if status == 401 else "Could not read the camera."
        authorization = _digest_authorization(user, password, uri, challenge)
        authed = (
            f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n"
            f"Authorization: {authorization}\r\nUser-Agent: AICamera\r\n\r\n"
        )
        try:
            status, _challenge = _rtsp_exchange(sock, authed)
        except OSError:
            sock.close()
            sock = socket.create_connection((host, port), timeout=5)
            sock.settimeout(5)
            status, _challenge = _rtsp_exchange(sock, authed)
        if status == 200:
            return ""
        if status == 401:
            return "NVR login was rejected."
        return "Could not read the camera."
    except socket.timeout:
        return "The NVR did not answer."
    except OSError:
        return "Could not reach the NVR."
    finally:
        sock.close()


def stream_url(camera_number: int, stream: str) -> tuple[str, str]:
    """Hikvision channel: camera 1 sub is 102, camera 12 sub is 1202."""
    suffix = "01" if stream == "main" else "02"
    channel = f"{int(camera_number)}{suffix}"
    url = _CHANNEL.sub(f"/Streaming/Channels/{channel}", base_rtsp_url(), count=1)
    if url == base_rtsp_url() and f"/Streaming/Channels/{channel}" not in url:
        raise RuntimeError("RTSP_URL has no /Streaming/Channels/ path")
    return url, channel
