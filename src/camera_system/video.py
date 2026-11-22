"""Bounded OpenCV frame source with explicit timestamp provenance, live buffering, and reconnects."""

import os
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import quote, urlsplit, urlunsplit

from PIL import Image


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_seconds: float
    timestamp_source: str
    image: Image.Image


class FrameSource:
    """Decode frames with optional latest-frame live buffering and reconnect handling."""

    def __init__(
        self,
        source: str | int,
        stride: int = 1,
        *,
        network: bool = False,
        credentials_env: str | None = None,
        threaded: bool = False,
        max_reconnects: int = 0,
        reconnect_delay: float = 1.0,
    ):
        if type(stride) is not int or stride < 1:
            raise ValueError("frame stride must be a positive integer")
        if type(source) is bool or not isinstance(source, (str, int)):
            raise ValueError("video source must be a file path, URL, or nonnegative camera index")
        if isinstance(source, int) and source < 0:
            raise ValueError("camera index must be nonnegative")
        if type(max_reconnects) is not int or max_reconnects < 0:
            raise ValueError("max_reconnects must be a nonnegative integer")
        if (
            type(reconnect_delay) not in (int, float)
            or not isfinite(reconnect_delay)
            or reconnect_delay <= 0
        ):
            raise ValueError("reconnect_delay must be positive and finite")
        open_source: str | int = source
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
            if credentials_env is not None:
                if not isinstance(credentials_env, str) or not credentials_env.strip():
                    raise ValueError("credentials_env must be a nonempty environment variable name")
                secret = os.environ.get(credentials_env, "")
                if not secret or ":" not in secret:
                    raise ValueError(
                        f"Environment variable {credentials_env} must be set as username:password"
                    )
                user, password = secret.split(":", 1)
                if not user:
                    raise ValueError(
                        f"Environment variable {credentials_env} must include a non-empty username"
                    )
                netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parsed.netloc}"
                open_source = urlunsplit(
                    (parsed.scheme, netloc, parsed.path, parsed.query, "")
                )
        else:
            if credentials_env is not None:
                raise ValueError("stream credentials require a network --stream URL")
            if isinstance(source, str) and not Path(source).is_file():
                raise ValueError(f"video file does not exist: {source}")

        self.network = network
        self.source = source
        self._open_source = open_source
        self.stride = stride
        self.threaded = bool(threaded)
        self.max_reconnects = max_reconnects
        self.reconnect_delay = float(reconnect_delay)
        self.frames_read = 0
        self.frames_skipped = 0
        self.dropped_frames = 0 if self.threaded else None
        self.reconnects = 0
        self.end_reason = None
        self.capture = None
        self._stop_event = Event()
        self._buf_lock = Lock()
        self._buf_cond = Condition(self._buf_lock)
        self._latest_frame = None
        self._reader_error = None
        self._reader_eof = False
        self._reader_thread: Thread | None = None

    def _open_capture(self):
        capture = (
            self.cv2.VideoCapture(
                self._open_source,
                self.cv2.CAP_FFMPEG,
                [
                    self.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    10000,
                    self.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    10000,
                ],
            )
            if self.network
            else self.cv2.VideoCapture(self._open_source)
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "cannot open stream" if self.network else f"cannot open video/camera: {self.source}"
            )
        return capture

    def _reconnect(self) -> bool:
        if not (self.network or isinstance(self.source, int)):
            return False
        while self.reconnects < self.max_reconnects and not self._stop_event.is_set():
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            delay = self.reconnect_delay * (2**self.reconnects)
            self.reconnects += 1
            if self._stop_event.wait(delay):
                return False
            try:
                self.capture = self._open_capture()
                return True
            except RuntimeError:
                continue
        return False

    def _extract_timestamp(self, index: int) -> tuple[float, str]:
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
        return timestamp, clock

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.capture is None:
                break
            ok, bgr = self.capture.read()
            if not ok:
                if self._reconnect():
                    continue
                with self._buf_cond:
                    if self.network:
                        self.end_reason = "stream_read_failure"
                        self._reader_error = RuntimeError(
                            "stream ended, timed out, or stopped returning frames"
                        )
                    elif isinstance(self.source, int):
                        self.end_reason = "camera_read_failure"
                        self._reader_error = RuntimeError("camera stopped returning frames")
                    else:
                        self.end_reason = "video_eof_or_decode_failure"
                        if self.frames_read == 0:
                            self._reader_error = RuntimeError("video contains no decodable frames")
                        else:
                            self._reader_eof = True
                    self._buf_cond.notify_all()
                break
            index = self.frames_read
            self.frames_read += 1
            if index % self.stride:
                self.frames_skipped += 1
                continue
            try:
                timestamp, clock = self._extract_timestamp(index)
            except RuntimeError as exc:
                with self._buf_cond:
                    self._reader_error = exc
                    self._buf_cond.notify_all()
                break
            with self._buf_cond:
                if self._latest_frame is not None:
                    self.dropped_frames = (self.dropped_frames or 0) + 1
                self._latest_frame = (index, timestamp, clock, bgr)
                self._buf_cond.notify_all()

    def __enter__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Install video dependencies: pip install -e ".[video]"') from exc
        self.cv2 = cv2
        self.capture = self._open_capture()
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.started = time.monotonic()
        self.previous_timestamp = -1.0
        if self.threaded:
            self._stop_event.clear()
            self._reader_thread = Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        with self._buf_cond:
            self._buf_cond.notify_all()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __iter__(self):
        if self.threaded:
            while True:
                with self._buf_cond:
                    while (
                        self._latest_frame is None
                        and self._reader_error is None
                        and not self._reader_eof
                        and not self._stop_event.is_set()
                    ):
                        self._buf_cond.wait(timeout=0.5)
                    if self._latest_frame is not None:
                        item = self._latest_frame
                        self._latest_frame = None
                    elif self._reader_error is not None:
                        raise self._reader_error
                    else:
                        return
                index, timestamp, clock, bgr = item
                rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                yield Frame(index, timestamp, clock, Image.fromarray(rgb))
        while True:
            ok, bgr = self.capture.read()
            if not ok:
                if self._reconnect():
                    continue
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
            timestamp, clock = self._extract_timestamp(index)
            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            yield Frame(index, timestamp, clock, Image.fromarray(rgb))
