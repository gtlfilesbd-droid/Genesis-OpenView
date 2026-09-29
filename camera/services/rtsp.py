import os
import re
from pathlib import Path

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


def base_rtsp_url() -> str:
    url = load_env_value("RTSP_URL")
    if not url or "user:password@host" in url:
        raise RuntimeError("Set RTSP_URL in .env")
    return url


def parse_channel(url: str) -> tuple[int, str]:
    match = _CHANNEL.search(url)
    if not match:
        return 1, "sub"
    code = match.group(0).rsplit("/", 1)[-1]
    stream = "main" if code.endswith("01") else "sub"
    camera = int(code[:-2] or "1")
    return camera, stream


def stream_url(camera_number: int, stream: str) -> tuple[str, str]:
    """Hikvision channel: camera 1 sub is 102, camera 12 sub is 1202."""
    suffix = "01" if stream == "main" else "02"
    channel = f"{int(camera_number)}{suffix}"
    url = _CHANNEL.sub(f"/Streaming/Channels/{channel}", base_rtsp_url(), count=1)
    if url == base_rtsp_url() and f"/Streaming/Channels/{channel}" not in url:
        raise RuntimeError("RTSP_URL has no /Streaming/Channels/ path")
    return url, channel
