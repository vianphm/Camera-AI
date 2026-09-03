"""RTSP IP camera capture stream with auto-reconnection logic."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np
from src.camera.stream import CameraStream


class RTSPStream(CameraStream):
    """Captures video frames from an RTSP stream with resilience to network drops."""

    def __init__(
        self,
        url: str,
        reconnect_delay_seconds: float = 2.0,
        max_reconnect_attempts: int = 10,
        max_queue_size: int = 2,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.url = url
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_reconnect_attempts = max_reconnect_attempts
        self._source_name = "rtsp_stream"
        self._cap: Optional[cv2.VideoCapture] = None

    def _open_capture(self) -> bool:
        # Avoid buffering frames inside OpenCV backend
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            # Attempt reconnect
            time.sleep(self.reconnect_delay_seconds)
            if not self._open_capture():
                return False, None

        success, frame = self._cap.read()
        if not success:
            # Stream drop, release and let next loop attempt reconnect
            self._close_capture()
            return False, None
        return True, frame

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if fps > 0:
                return fps
        return 25.0

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (1920, 1080)
