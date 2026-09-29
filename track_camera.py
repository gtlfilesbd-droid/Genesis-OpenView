import os
from pathlib import Path

import torch
from ultralytics import YOLO


def load_rtsp_url() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "RTSP_URL":
                return value.strip().strip('"').strip("'")
    return os.environ.get("RTSP_URL", "").strip()


def main() -> None:
    rtsp_url = load_rtsp_url()
    placeholder = "rtsp://user:password@host:554/..."
    if not rtsp_url or "user:password@host" in rtsp_url or rtsp_url == placeholder:
        raise SystemExit("Set RTSP_URL in .env (see .env.example).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolo11n.pt")
    results = model.track(
        source=rtsp_url,
        show=True,
        stream=True,
        persist=True,
        classes=[0],  # person only
        imgsz=640,
        device=device,
    )
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            print(f"persons: {len(ids)}  ids: {ids}")


if __name__ == "__main__":
    main()
