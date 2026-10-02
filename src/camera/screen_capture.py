"""Screen and Window Capture Camera Stream using mss and Win32 API."""

import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from src.camera.stream import CameraStream

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import win32gui
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


class ScreenCaptureStream(CameraStream):
    """
    Bắt hình trực tiếp từ cửa sổ ứng dụng EZVIZ Studio hoặc vùng màn hình:
    - Tự động dò tìm tọa độ cửa sổ EZVIZ Studio đang mở trên desktop.
    - Chụp khung hình tốc độ cao (30 FPS) với độ trễ cực thấp (< 2ms).
    """

    def __init__(
        self,
        target_title: str = "ezviz",
        custom_bbox: Optional[Dict[str, int]] = None,
        fps: float = 25.0,
        max_queue_size: int = 2,
    ) -> None:
        super().__init__(max_queue_size=max_queue_size)
        self.target_title = target_title.lower()
        self.custom_bbox = custom_bbox
        self.fps = fps
        self._source_name = "screen_capture_ezviz"
        self._sct = None
        self._capture_box: Dict[str, int] = {"top": 0, "left": 0, "width": 1280, "height": 720}
        self._hwnd = None

    def _find_window_rect(self) -> Optional[Tuple[int, Dict[str, int]]]:
        """Dò tìm cửa sổ EZVIZ Studio trên Windows."""
        if not WIN32_AVAILABLE:
            return None

        found_windows = []

        def enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 300 and h > 200:
                    found_windows.append((hwnd, title, rect, w, h))

        win32gui.EnumWindows(enum_cb, None)

        # 1. Tìm theo tên có chữ ezviz hoặc studio
        for hwnd, title, rect, w, h in found_windows:
            t_lower = title.lower()
            if self.target_title in t_lower or "studio" in t_lower or "cam" in t_lower:
                return hwnd, {"top": max(0, rect[1]), "left": max(0, rect[0]), "width": w, "height": h}

        # 2. Nếu không có tên cụ thể, tìm cửa sổ EZVIZ theo process name
        try:
            import psutil
            for p in psutil.process_iter(["pid", "name"]):
                if "ezviz" in (p.info["name"] or "").lower():
                    for hwnd, title, rect, w, h in found_windows:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == p.info["pid"]:
                            return hwnd, {"top": max(0, rect[1]), "left": max(0, rect[0]), "width": w, "height": h}
        except Exception:
            pass

        return None

    def _open_capture(self) -> bool:
        if not MSS_AVAILABLE:
            print("❌ Lỗi: Chưa cài đặt thư viện 'mss'. Hãy chạy: pip install mss")
            return False

        self._sct = mss.mss()

        if self.custom_bbox:
            self._capture_box = self.custom_bbox
            print(f"[*] Sử dụng vùng chụp tùy chỉnh: {self._capture_box}")
            return True

        win_info = self._find_window_rect()
        if win_info:
            self._hwnd, box = win_info
            self._capture_box = box
            title = win32gui.GetWindowText(self._hwnd) if WIN32_AVAILABLE else "EZVIZ Studio"
            print(f"[*] Đã tự động nhận diện cửa sổ EZVIZ: \"{title}\" (Tọa độ: {box})")
        else:
            # Fallback: Chụp màn hình chính
            monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
            self._capture_box = {
                "top": monitor["top"],
                "left": monitor["left"],
                "width": monitor["width"],
                "height": monitor["height"],
            }
            print(f"[*] Chưa tìm thấy cửa sổ EZVIZ riêng biệt, chụp toàn màn hình: {self._capture_box}")

        return True

    def _read_raw_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._sct is None:
            return False, None

        try:
            # Cập nhật tọa độ nếu cửa sổ di chuyển
            if self._hwnd and WIN32_AVAILABLE and win32gui.IsWindow(self._hwnd):
                rect = win32gui.GetWindowRect(self._hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 100 and h > 100:
                    self._capture_box = {"top": max(0, rect[1]), "left": max(0, rect[0]), "width": w, "height": h}

            sct_img = self._sct.grab(self._capture_box)
            # Chuyển từ BGRA sang BGR
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception as e:
            time.sleep(0.05)
            return False, None

    def _close_capture(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None

    def get_fps(self) -> float:
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        return (self._capture_box.get("width", 1280), self._capture_box.get("height", 720))
