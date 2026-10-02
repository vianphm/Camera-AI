"""Audio Siren module for triggering urgent sound alerts upon intrusion."""

import threading
import sys
import time


class SirenPlayer:
    """Plays an alternating frequency siren alarm on Windows without blocking the AI loop."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._is_playing = False
        self._lock = threading.Lock()

    def trigger_siren(self, repeat_count: int = 4) -> None:
        """Kích hoạt còi hú cảnh báo."""
        if not self.enabled:
            return

        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True

        thread = threading.Thread(target=self._play_worker, args=(repeat_count,), daemon=True)
        thread.start()

    def _play_worker(self, repeat_count: int) -> None:
        try:
            if sys.platform == "win32":
                import winsound
                for _ in range(repeat_count):
                    # Tần số cao thấp đan xen mô phỏng còi báo động khẩn cấp
                    winsound.Beep(2400, 350)
                    winsound.Beep(1600, 250)
            else:
                # Fallback trên Linux/macOS
                print("\a", end="", flush=True)
        except Exception as e:
            print(f"[SIREN WARNING] Không thể phát âm thanh: {e}")
        finally:
            with self._lock:
                self._is_playing = False
