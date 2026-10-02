"""Event Video Recorder with Pre-Buffer Ring and Asynchronous MP4 Export."""

import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import cv2
import numpy as np


class EventVideoRecorder:
    """
    Ghi hình thông minh theo sự kiện:
    - Lưu sẵn 5 giây video trong RAM trước khi sự kiện xảy ra (Pre-Buffer).
    - Khi có báo động -> Ghi tiếp 20 giây sau đó (Post-Buffer).
    - Xuất video MP4 và gọi hàm callback đẩy lên Google Drive & gửi Telegram.
    """

    def __init__(
        self,
        fps: float = 20.0,
        pre_buffer_seconds: float = 5.0,
        post_buffer_seconds: float = 20.0,
        output_dir: str = "data/alarm_videos",
        on_video_completed: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.fps = fps
        self.pre_buffer_maxlen = max(10, int(fps * pre_buffer_seconds))
        self.post_buffer_frames_target = int(fps * post_buffer_seconds)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_video_completed = on_video_completed

        self._pre_buffer: deque = deque(maxlen=self.pre_buffer_maxlen)
        self._is_recording: bool = False
        self._recorded_frames: List[np.ndarray] = []
        self._remaining_post_frames: int = 0
        self._current_event_metadata: dict = {}
        self._lock = threading.Lock()

    def push_frame(self, frame: np.ndarray) -> None:
        """Nhận từng frame từ camera stream và quản lý bộ nhớ đệm."""
        with self._lock:
            if not self._is_recording:
                # Luôn lưu giữ các khung hình gần nhất trong bộ nhớ RAM
                self._pre_buffer.append(frame.copy())
            else:
                self._recorded_frames.append(frame.copy())
                self._remaining_post_frames -= 1

                if self._remaining_post_frames <= 0:
                    self._finalize_and_export_async()

    def trigger_event(self, metadata: dict) -> bool:
        """Kích hoạt ghi hình sự kiện khi AI phát hiện trộm."""
        with self._lock:
            if self._is_recording:
                # Đang trong quá trình ghi hình sự kiện trước đó, gia hạn thêm thời gian
                self._remaining_post_frames = self.post_buffer_frames_target
                return False

            self._is_recording = True
            # Gom toàn bộ 5 giây trước sự kiện
            self._recorded_frames = list(self._pre_buffer)
            self._remaining_post_frames = self.post_buffer_frames_target
            self._current_event_metadata = metadata
            print(f"🔴 [REC] Bắt đầu ghi hình sự kiện đột nhập (Pre-buffer {len(self._recorded_frames)} frames)...")
            return True

    def _finalize_and_export_async(self) -> None:
        """Kết thúc ghi hình và chuyển sang luồng phụ để ghi file MP4 không làm đơ camera."""
        frames_to_write = list(self._recorded_frames)
        metadata = dict(self._current_event_metadata)

        # Reset trạng thái
        self._is_recording = False
        self._recorded_frames.clear()

        # Xuất file trong background thread
        thread = threading.Thread(
            target=self._write_video_worker,
            args=(frames_to_write, metadata),
            daemon=True,
        )
        thread.start()

    def _write_video_worker(self, frames: List[np.ndarray], metadata: dict) -> None:
        """Worker thread ghi file MP4 và gọi upload Google Drive."""
        if not frames:
            return

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        video_filename = self.output_dir / f"Intruder_{timestamp_str}.mp4"
        snapshot_filename = self.output_dir / f"Intruder_{timestamp_str}.jpg"

        h, w = frames[0].shape[:2]

        # Lưu ảnh snapshot ở thời điểm trộm vào rõ nhất (giữa clip)
        mid_idx = min(len(frames) - 1, self.pre_buffer_maxlen + 5)
        cv2.imwrite(str(snapshot_filename), frames[mid_idx])

        # Sử dụng fourcc mp4v phổ biến trên Windows
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_filename), fourcc, self.fps, (w, h))

        for f in frames:
            writer.write(f)
        writer.release()

        print(f"💾 [SAVED] Đã lưu video bằng chứng tại: {video_filename} ({len(frames)} frames)")
        metadata["video_path"] = str(video_filename)
        metadata["snapshot_path"] = str(snapshot_filename)

        # Gọi hàm callback (Google Drive Upload + Telegram)
        if self.on_video_completed:
            try:
                self.on_video_completed(str(video_filename), metadata)
            except Exception as e:
                print(f"❌ [RECORDER CALLBACK ERROR]: {e}")
