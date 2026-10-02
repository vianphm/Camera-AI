"""24/7 Continuous Video Recorder with Segment Splitting and Rolling Storage Retention."""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import cv2
import numpy as np


class Continuous247Recorder:
    """
    Ghi hình liên tục 24/7:
    - Chia nhỏ video thành từng file (mặc định 15 phút / file).
    - Tự động dọn dẹp các file cũ nhất khi ổ đĩa đạt giới hạn (FIFO Rolling Retention).
    - Đóng dấu thời gian (Timestamp) lên từng khung hình.
    - Chạy trên luồng nền độc lập, không làm chậm quá trình AI phân tích.
    - Hỗ trợ tự động đẩy từng đoạn video lên Google Drive (nếu cấu hình).
    """

    def __init__(
        self,
        enabled: bool = True,
        output_dir: str = "data/records_24_7",
        segment_duration_minutes: float = 15.0,
        max_storage_gb: float = 30.0,
        retention_days: int = 7,
        fps: float = 20.0,
        on_segment_completed: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_duration_seconds = segment_duration_minutes * 60.0
        self.max_storage_bytes = int(max_storage_gb * 1024 * 1024 * 1024)
        self.retention_seconds = retention_days * 86400
        self.fps = fps
        self.on_segment_completed = on_segment_completed

        self._current_writer: Optional[cv2.VideoWriter] = None
        self._current_file_path: Optional[Path] = None
        self._segment_start_time: float = 0.0
        self._frame_size: Optional[tuple] = None
        self._lock = threading.Lock()
        self._frame_queue = []

        if self.enabled:
            print(f"📼 [REC 24/7] Kích hoạt ghi hình liên tục (Mỗi đoạn: {segment_duration_minutes} phút | Giới hạn ổ cứng: {max_storage_gb} GB)")

    def push_frame(self, frame: np.ndarray) -> None:
        """Nhận từng khung hình và ghi vào file video liên tục."""
        if not self.enabled or frame is None:
            return

        now = time.time()
        h, w = frame.shape[:2]

        with self._lock:
            # Kiểm tra xem cần tạo file mới không (lần đầu hoặc hết thời lượng 1 đoạn)
            if self._current_writer is None or (now - self._segment_start_time) >= self.segment_duration_seconds:
                self._rotate_segment(now, (w, h))

            if self._current_writer:
                # Đóng dấu thời gian ngày giờ lên góc dưới khung hình
                frame_to_write = frame.copy()
                t_str = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    frame_to_write,
                    t_str,
                    (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self._current_writer.write(frame_to_write)

    def _rotate_segment(self, current_time: float, frame_size: tuple) -> None:
        """Đóng file video hiện tại và mở file mới."""
        old_file = None
        if self._current_writer is not None:
            self._current_writer.release()
            self._current_writer = None
            old_file = str(self._current_file_path)
            print(f"📼 [REC 24/7] Đã hoàn tất đoạn video: {old_file}")

        # Tự động dọn dẹp dung lượng ổ cứng
        self._cleanup_old_records()

        # Tạo file video mới
        time_str = time.strftime("%Y%m%d_%H%M%S")
        self._current_file_path = self.output_dir / f"Record_24_7_{time_str}.mp4"
        self._segment_start_time = current_time
        self._frame_size = frame_size

        w, h = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._current_writer = cv2.VideoWriter(str(self._current_file_path), fourcc, self.fps, (w, h))
        print(f"📼 [REC 24/7] Đang ghi đoạn mới: {self._current_file_path.name}")

        # Gọi callback (nếu muốn upload đoạn video vừa xong lên Google Drive)
        if old_file and self.on_segment_completed:
            threading.Thread(target=self.on_segment_completed, args=(old_file,), daemon=True).start()

    def _cleanup_old_records(self) -> None:
        """Xóa các file cũ nhất nếu vượt quá giới hạn ổ đĩa (FIFO Rolling Storage)."""
        try:
            files = sorted(self.output_dir.glob("Record_24_7_*.mp4"), key=lambda f: f.stat().st_mtime)
            total_size = sum(f.stat().st_size for f in files)
            now = time.time()

            for f in files:
                file_age = now - f.stat().st_mtime
                # Xóa nếu quá số ngày lưu trữ HOẶC tổng dung lượng vượt quá giới hạn
                if file_age > self.retention_seconds or total_size > self.max_storage_bytes:
                    size = f.stat().st_size
                    f.unlink(missing_ok=True)
                    total_size -= size
                    print(f"♻️ [STORAGE CLEANUP] Đã tự động xóa file cũ: {f.name} để giải phóng ổ cứng")
        except Exception as e:
            print(f"[CLEANUP WARNING]: {e}")

    def close(self) -> None:
        """Đóng an toàn khi dừng chương trình."""
        with self._lock:
            if self._current_writer is not None:
                self._current_writer.release()
                self._current_writer = None
                print("📼 [REC 24/7] Đã lưu file cuối cùng an toàn.")
