"""Time Guard for managing late-night arming schedules and curfew enforcement."""

from datetime import datetime, time
from typing import Dict, Any


class TimeGuard:
    """Manages armed/disarmed state based on real-time schedule or manual override."""

    def __init__(
        self,
        enabled: bool = True,
        start_time_str: str = "22:30",
        end_time_str: str = "05:30",
        always_armed: bool = False,
    ) -> None:
        self.enabled = enabled
        self.start_time_str = start_time_str
        self.end_time_str = end_time_str
        self.always_armed = always_armed

    @classmethod
    def from_config(cls, schedule_cfg: Dict[str, Any]) -> "TimeGuard":
        return cls(
            enabled=schedule_cfg.get("enabled", True),
            start_time_str=schedule_cfg.get("start_time", "22:30"),
            end_time_str=schedule_cfg.get("end_time", "05:30"),
            always_armed=schedule_cfg.get("always_armed", False),
        )

    def is_armed(self) -> bool:
        """Kiểm tra thời điểm hiện tại hệ thống có đang BẬT chế độ chống trộm không."""
        if not self.enabled:
            return False

        if self.always_armed:
            return True

        now = datetime.now().time()
        start = datetime.strptime(self.start_time_str, "%H:%M").time()
        end = datetime.strptime(self.end_time_str, "%H:%M").time()

        if start > end:
            # Qua đêm: ví dụ từ 22:30 tối nay đến 05:30 sáng hôm sau
            return now >= start or now <= end
        else:
            # Cùng ngày: ví dụ từ 13:00 đến 17:00
            return start <= now <= end

    def get_status_str(self) -> str:
        """Trả về chuỗi hiển thị trạng thái trên giao diện HUD."""
        if self.always_armed:
            return "ARMED 24/7 (ĐANG BẬT BÁO ĐỘNG)"
        elif self.is_armed():
            return f"ARMED NIGHT ({self.start_time_str} - {self.end_time_str})"
        else:
            return f"DISARMED (Chế độ giám sát ngày, tự bật lúc {self.start_time_str})"
