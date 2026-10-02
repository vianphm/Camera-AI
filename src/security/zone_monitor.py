"""Zone & Perimeter Intrusion Monitor with Polygon ROI & Loitering Analysis."""

import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


@dataclass
class IntrusionEvent:
    track_id: int
    zone_name: str
    reason: str
    dwell_time: float
    timestamp: float
    bbox: List[float]


class ZoneMonitor:
    """Monitors restricted zones, tripwires, and flags loitering or intrusions."""

    def __init__(self, zones_cfg: List[Dict[str, Any]], frame_size: Tuple[int, int] = (1280, 720)) -> None:
        self.frame_width, self.frame_height = frame_size
        self.zones = []
        self._track_enter_timestamps: Dict[Tuple[int, str], float] = {}
        self._last_alert_timestamps: Dict[int, float] = {}

        self.update_zones(zones_cfg, frame_size)

    def update_zones(self, zones_cfg: List[Dict[str, Any]], frame_size: Tuple[int, int]) -> None:
        """Cập nhật tọa độ các vùng đa giác ROI theo độ phân giải frame thực tế."""
        self.frame_width, self.frame_height = frame_size
        self.zones = []

        for z in zones_cfg:
            if not z.get("enabled", True):
                continue
            name = z.get("name", "Zone")
            poly_norm = z.get("polygon", [])
            pts = np.array(
                [[int(p[0] * self.frame_width), int(p[1] * self.frame_height)] for p in poly_norm],
                dtype=np.int32,
            )
            self.zones.append({
                "name": name,
                "pts": pts,
                "alert_immediately": z.get("alert_immediately", False),
                "loitering_limit": float(z.get("loitering_limit_seconds", 0.0)),
            })

    def check_person(self, track_id: int, bbox: List[float], current_time: Optional[float] = None) -> Optional[IntrusionEvent]:
        """
        Kiểm tra một người (track_id, bbox=[x1, y1, x2, y2]) có vi phạm an ninh không.
        Sử dụng điểm chân người (tọa độ đáy bbox) để tránh bị kích hoạt giả bởi bóng đèn.
        """
        now = current_time if current_time is not None else time.time()
        x1, y1, x2, y2 = bbox
        foot_x = int((x1 + x2) / 2.0)
        foot_y = int(y2)
        foot_point = (foot_x, foot_y)

        # Tránh spam báo động liên tục cho cùng một track_id (cooldown 15 giây)
        last_alert = self._last_alert_timestamps.get(track_id, 0.0)
        in_cooldown = (now - last_alert) < 15.0

        for zone in self.zones:
            # cv2.pointPolygonTest trả về >= 0 nếu điểm nằm trong hoặc trên cạnh đa giác
            dist = cv2.pointPolygonTest(zone["pts"], foot_point, False)
            if dist >= 0:
                key = (track_id, zone["name"])
                first_seen = self._track_enter_timestamps.setdefault(key, now)
                dwell_time = now - first_seen

                # 1. Trường hợp Vùng cấm tuyệt đối (bước vào là báo ngay)
                if zone["alert_immediately"]:
                    if not in_cooldown:
                        self._last_alert_timestamps[track_id] = now
                        return IntrusionEvent(
                            track_id=track_id,
                            zone_name=zone["name"],
                            reason=f"Đột nhập trái phép vào vùng cấm [{zone['name']}]",
                            dwell_time=dwell_time,
                            timestamp=now,
                            bbox=bbox,
                        )

                # 2. Trường hợp Lảng vảng (đứng quá thời gian quy định)
                elif dwell_time >= zone["loitering_limit"]:
                    if not in_cooldown:
                        self._last_alert_timestamps[track_id] = now
                        return IntrusionEvent(
                            track_id=track_id,
                            zone_name=zone["name"],
                            reason=f"Phát hiện lảng vảng bất thường ({dwell_time:.1f}s) tại [{zone['name']}]",
                            dwell_time=dwell_time,
                            timestamp=now,
                            bbox=bbox,
                        )

        return None

    def cleanup_tracks(self, active_track_ids: List[int]) -> None:
        """Dọn dẹp bộ nhớ theo dõi cho các track_id đã rời khỏi khung hình."""
        keys_to_del = [k for k in self._track_enter_timestamps if k[0] not in active_track_ids]
        for k in keys_to_del:
            self._track_enter_timestamps.pop(k, None)

    def draw_zones(self, frame: np.ndarray, is_armed: bool = True) -> np.ndarray:
        """Vẽ trực quan hóa các vùng cấm lên khung hình camera."""
        color = (0, 0, 255) if is_armed else (0, 255, 255) # Đỏ khi bật báo động, Vàng khi giám sát thường
        overlay = frame.copy()

        for zone in self.zones:
            pts = zone["pts"].reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [pts], color=color)

            # Ghi tên vùng
            first_pt = zone["pts"][0]
            label = f"{zone['name']} ({'CẤM' if zone['alert_immediately'] else 'Lảng vảng'})"
            cv2.putText(
                frame,
                label,
                (first_pt[0], max(20, first_pt[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Hiệu ứng mờ bán trong suốt cho vùng cảnh báo
        alpha = 0.15 if is_armed else 0.08
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
