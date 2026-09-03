"""Abstract base class and threaded frame queue for camera streams."""

import time
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FramePacket:
    """Represents an acquired video frame along with metadata."""
    frame: np.ndarray
    frame_idx: int
    timestamp: float
    source_name: str


class CameraStream(ABC):
    """Abstract interface for all video and camera input streams."""

    def __init__(self, max_queue_size: int = 2) -> None:
        self.max_queue_size = max_queue_size
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._source_name = "unknown"

    @abstractmethod
    def _open_capture(self) -> bool:
        """Open the physical or network video capture device."""
        pass

    @abstractmethod
    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single raw frame from the capture device."""
        pass

    @abstractmethod
    def _close_capture(self) -> None:
        """Release capture device resources."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get native frame rate of stream."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get native (width, height) of stream."""
        pass

    def start(self) -> bool:
        """Start the background thread to continually ingest frames."""
        if self._running:
            return True

        if not self._open_capture():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        """Background thread loop that reads frames and puts freshest into queue."""
        while self._running:
            success, frame = self._read_raw_frame()
            if not success or frame is None:
                # Video file ended or temporary RTSP drop
                time.sleep(0.01)
                continue

            self._frame_count += 1
            packet = FramePacket(
                frame=frame,
                frame_idx=self._frame_count,
                timestamp=time.time(),
                source_name=self._source_name,
            )

            # Drop oldest frame if queue full to prevent lag accumulation
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Read the next available frame packet."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_opened(self) -> bool:
        """Check if stream is actively running."""
        return self._running

    def release(self) -> None:
        """Stop background worker and release camera resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._close_capture()
        # Empty queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
