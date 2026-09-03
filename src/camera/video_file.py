"""Local video file stream implementation for evaluation and testing."""

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from src.camera.stream import CameraStream


class VideoFileStream(CameraStream):
    """Streams frames from a recorded video file (e.g. MP4, AVI)."""

    def __init__(
        self,
        video_path: str | Path,
        loop: bool = False,
        simulate_realtime: bool = True,
        max_queue_size: int = 5,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.video_path = str(video_path)
        self.loop = loop
        self.simulate_realtime = simulate_realtime
        self._source_name = Path(self.video_path).name
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._last_read_time = 0.0

    def _open_capture(self) -> bool:
        if not Path(self.video_path).exists():
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            return False
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps > 0:
            self._fps = fps
        return True

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        if self.simulate_realtime and self._last_read_time > 0:
            target_dt = 1.0 / max(1.0, self._fps)
            elapsed = time.time() - self._last_read_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
        self._last_read_time = time.time()

        success, frame = self._cap.read()
        if not success:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self._cap.read()
            return False, None

        return True, frame

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return (w, h)
        return (1280, 720)
