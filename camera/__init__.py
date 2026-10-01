import os

# OpenCV reads this while importing cv2. It has to be set before that import.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;8000000",
)
