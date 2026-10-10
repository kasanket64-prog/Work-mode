"""Bounded, synchronous OpenCV frame source with explicit timestamp provenance."""

import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_seconds: float
    timestamp_source: str
    image: Image.Image


class FrameSource:
    """No application frame queue: decode on demand and retain only the current frame."""

    def __init__(self, source: str | int, stride: int = 1, *, network=False):
        if type(stride) is not int or stride < 1:
            raise ValueError("frame stride must be a positive integer")
        if isinstance(source, int) and source < 0:
            raise ValueError("camera index must be nonnegative")
        if network:
            parsed = urlsplit(source)
            if (
                parsed.scheme not in {"https", "http", "rtsp"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
            ):
                raise ValueError(
                    "stream requires an HTTP(S) or RTSP URL without embedded credentials"
                )
        elif isinstance(source, str) and not Path(source).is_file():
            raise ValueError(f"video file does not exist: {source}")
        self.network = network
        self.source = source
        self.stride = stride
        self.frames_read = 0
        self.frames_skipped = 0
        self.end_reason = None
        self.capture = None

    def __enter__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Install video dependencies: pip install -e ".[video]"') from exc
        self.cv2 = cv2
        self.capture = (
            cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    10000,
                ],
            )
            if self.network
            else cv2.VideoCapture(self.source)
        )
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "cannot open stream" if self.network else f"cannot open video/camera: {self.source}"
            )
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.started = time.monotonic()
        self.previous_timestamp = -1.0
        return self

    def __exit__(self, *args):
        if self.capture is not None:
            self.capture.release()

    def __iter__(self):
        while True:
            ok, bgr = self.capture.read()
            if not ok:
                if self.network:
                    self.end_reason = "stream_read_failure"
                    raise RuntimeError("stream ended, timed out, or stopped returning frames")
                if isinstance(self.source, int):
                    self.end_reason = "camera_read_failure"
                    raise RuntimeError("camera stopped returning frames")
                # OpenCV cannot reliably distinguish normal EOF from a decoder failure.
                self.end_reason = "video_eof_or_decode_failure"
                if self.frames_read == 0:
                    raise RuntimeError("video contains no decodable frames")
                return
            index = self.frames_read
            self.frames_read += 1
            if index % self.stride:
                self.frames_skipped += 1
                continue
            if isinstance(self.source, int) or self.network:
                timestamp = time.monotonic() - self.started
                clock = "host_monotonic_after_read"
            else:
                timestamp = float(self.capture.get(self.cv2.CAP_PROP_POS_MSEC)) / 1000
                clock = "video_position_msec"
                if not isfinite(timestamp) or timestamp < 0 or timestamp <= self.previous_timestamp:
                    if not isfinite(self.fps) or self.fps <= 0:
                        raise RuntimeError("video timestamps are unavailable and FPS is invalid")
                    timestamp = max(
                        index / self.fps, self.previous_timestamp + self.stride / self.fps
                    )
                    clock = "estimated_from_fps"
            self.previous_timestamp = timestamp
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            yield Frame(index, timestamp, clock, Image.fromarray(rgb))
