"""Webcam and USB camera capture stream implementation."""

from typing import Optional, Tuple
import cv2
import numpy as np
from src.camera.stream import CameraStream


class WebcamStream(CameraStream):
    """Captures video frames from physical webcam or USB camera."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        api_preference: str = "default",
        max_queue_size: int = 2,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.api_preference = api_preference
        self._source_name = f"webcam_{device_index}"
        self._cap: Optional[cv2.VideoCapture] = None

    def _open_capture(self) -> bool:
        import sys
        backend = cv2.CAP_ANY
        pref = self.api_preference.lower()
        if pref == "dshow" or (pref == "default" and sys.platform == "win32"):
            backend = cv2.CAP_DSHOW
        elif pref == "v4l2":
            backend = cv2.CAP_V4L2

        self._cap = cv2.VideoCapture(self.device_index, backend)
        # Fallback to ANY if DSHOW fails to open
        if (not self._cap or not self._cap.isOpened()) and backend == cv2.CAP_DSHOW:
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)

        if not self._cap or not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return True

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return float(self._cap.get(cv2.CAP_PROP_FPS)) or float(self.target_fps)
        return float(self.target_fps)

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (self.target_width, self.target_height)
