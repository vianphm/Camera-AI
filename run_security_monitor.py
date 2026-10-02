"""RUN SECURITY MONITOR — AI CẢNH BÁO TRỘM ĐÊM KHUYA & ĐẨY CLOUD GOOGLE DRIVE
================================================================================
Hệ thống AI Giám sát An ninh Đêm khuya cho Camera EZVIZ / RTSP / Webcam:
- Tự động nhận diện khung giờ giới nghiêm ban đêm (mặc định 22h30 - 05h30).
- Giám sát vùng cấm (Polygon ROI) và hàng rào ảo (Tripwire).
- Bỏ qua chuyển động tĩnh (Tier-0 Motion Gating) để tiết kiệm CPU/GPU.
- Phân biệt chính xác người (YOLOv8 + ByteTrack), triệt tiêu báo động giả.
- Hú còi cảnh báo khẩn cấp (Siren Sound Alarm).
- Tự động cắt clip MP4 chất lượng cao (quay trước 5 giây + ghi tiếp 20 giây).
- Tự động đẩy clip bằng chứng lên Google Drive (Google One) và gửi về Telegram.

Cách chạy:
    python run_security_monitor.py --source rtsp
    python run_security_monitor.py --source webcam
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo đường dẫn gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.camera.stream import CameraStream
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream
from src.camera.screen_capture import ScreenCaptureStream
from src.detection.motion_gater import MotionGater
from src.detection.person_detector import YOLOv8PersonDetector
from src.tracking.track_manager import ByteTrackManager
from src.security.time_guard import TimeGuard
from src.security.zone_monitor import ZoneMonitor, IntrusionEvent
from src.security.event_recorder import EventVideoRecorder
from src.security.continuous_recorder import Continuous247Recorder
from src.security.siren import SirenPlayer
from src.alerts.gdrive_uploader import GoogleDriveUploader
from src.alerts.telegram import TelegramClient, load_telegram_settings


def load_security_configs() -> Dict[str, Any]:
    """Tải cấu hình an ninh, camera, thông báo."""
    try:
        sec_cfg = load_config("security.yaml").get("security", {})
    except Exception:
        sec_cfg = {}

    try:
        cam_cfg = load_config("camera.yaml")
    except Exception:
        cam_cfg = {}

    try:
        notif_cfg = load_config("notifications.yaml")
    except Exception:
        notif_cfg = {}

    return {
        "security": sec_cfg,
        "camera": cam_cfg,
        "notifications": notif_cfg,
    }


def init_camera(source: str, rtsp_url: Optional[str], device_index: int, cam_cfg: dict) -> CameraStream:
    """Khởi tạo luồng camera phù hợp."""
    if source == "screen":
        print("[*] Đang khởi động chế độ BẮT HÌNH TRỰC TIẾP TỪ CỬA SỔ EZVIZ STUDIO...")
        stream = ScreenCaptureStream()
    elif source == "rtsp":
        url = rtsp_url or cam_cfg.get("rtsp", {}).get("url")
        if not url or "VERIFICATION_CODE" in url:
            print("\n❌ LƯU Ý QUAN TRỌNG VỀ CAMERA EZVIZ:")
            print("  Vui lòng mở file configs/camera.yaml và điền thông tin camera EZVIZ:")
            print("  - Thay VERIFICATION_CODE bằng 6 chữ cái in hoa dưới đáy camera.")
            print("  - Thay IP_CAMERA bằng địa chỉ IP của camera.")
            print("  Ví dụ: rtsp://admin:ABCDEF@192.168.1.50:554/Streaming/Channels/101\n")
        print(f"[*] Đang kết nối luồng RTSP EZVIZ: {url}")
        stream = RTSPStream(url=url)
    elif source == "video":
        v_path = cam_cfg.get("video", {}).get("path", "data/videos/sample.mp4")
        print(f"[*] Đang đọc tệp video thử nghiệm: {v_path}")
        stream = VideoFileStream(path=v_path)
    else:
        print(f"[*] Đang kết nối Webcam (Thiết bị #{device_index})...")
        stream = WebcamStream(device_index=device_index)

    if not stream.start():
        print(f"❌ Không thể mở nguồn camera: {source}. Vui lòng kiểm tra lại kết nối!")
        sys.exit(1)
    return stream


class SecurityMonitorApp:
    def __init__(self, configs: dict, stream: CameraStream, always_armed: bool = False):
        self.configs = configs
        self.stream = stream
        self.sec_cfg = configs["security"]

        # 1. Khởi tạo TimeGuard (Lịch trực đêm)
        schedule_cfg = self.sec_cfg.get("schedule", {})
        if always_armed:
            schedule_cfg["always_armed"] = True
        self.time_guard = TimeGuard.from_config(schedule_cfg)

        # 2. Khởi tạo Vùng cấm ZoneMonitor
        w, h = stream.get_resolution()
        if w <= 0 or h <= 0:
            w, h = 1280, 720
        self.zone_monitor = ZoneMonitor(self.sec_cfg.get("zones", []), frame_size=(w, h))

        # 3. Khởi tạo Còi hú Siren
        self.siren = SirenPlayer(enabled=self.sec_cfg.get("actions", {}).get("play_siren", True))

        # 4. Khởi tạo Google Drive Uploader
        gdrive_folder = self.sec_cfg.get("actions", {}).get("gdrive_folder_name", "EZVIZ_Security_Alerts")
        self.gdrive_uploader = GoogleDriveUploader(folder_name=gdrive_folder)

        # 5. Khởi tạo Telegram Client
        self.telegram_settings = load_telegram_settings()
        self.telegram_client = None
        if self.telegram_settings.is_ready:
            self.telegram_client = TelegramClient(bot_token=self.telegram_settings.bot_token)
            print("  • Telegram Bot Alerts     : [KÍCH HOẠT]")
        else:
            print("  • Telegram Bot Alerts     : [CHƯA CẤU HÌNH] (Có thể cấu hình trong Web Dashboard hoặc App)")

        # 6. Khởi tạo Event Video Recorder (Khi có trộm)
        rec_cfg = self.sec_cfg.get("recording", {})
        self.recorder = EventVideoRecorder(
            fps=float(rec_cfg.get("video_fps", 20.0)),
            pre_buffer_seconds=float(rec_cfg.get("pre_buffer_seconds", 5.0)),
            post_buffer_seconds=float(rec_cfg.get("post_buffer_seconds", 20.0)),
            output_dir=rec_cfg.get("output_dir", "data/alarm_videos"),
            on_video_completed=self._handle_recorded_video,
        )

        # 6b. Khởi tạo Continuous 24/7 Recorder (Lưu trữ liên tục 24/7)
        c_rec_cfg = self.sec_cfg.get("continuous_recording", {})
        self.continuous_recorder = Continuous247Recorder(
            enabled=c_rec_cfg.get("enabled", True),
            output_dir=c_rec_cfg.get("output_dir", "data/records_24_7"),
            segment_duration_minutes=float(c_rec_cfg.get("segment_duration_minutes", 15.0)),
            max_storage_gb=float(c_rec_cfg.get("max_storage_gb", 40.0)),
            retention_days=int(c_rec_cfg.get("retention_days", 7)),
            fps=float(c_rec_cfg.get("video_fps", 20.0)),
            on_segment_completed=self._handle_continuous_segment if c_rec_cfg.get("sync_to_gdrive", False) else None,
        )

        # 7. AI Detection & Tracking
        min_conf = float(self.sec_cfg.get("behavior", {}).get("min_person_confidence", 0.45))
        self.detector = YOLOv8PersonDetector(model_path="yolov8n.pt", conf_threshold=min_conf)
        self.tracker = ByteTrackManager()
        self.motion_gater = MotionGater(enabled=True, min_motion_ratio=0.001)

    def _handle_continuous_segment(self, segment_path: str) -> None:
        """Đẩy đoạn ghi 24/7 lên Google Drive (nếu bật)."""
        if self.gdrive_uploader.is_ready:
            print(f"☁️ [GDRIVE 24/7] Đang đồng bộ đoạn video 24/7: {Path(segment_path).name}...")
            self.gdrive_uploader.upload_file(segment_path, mime_type="video/mp4")

    def _handle_recorded_video(self, video_path: str, metadata: dict) -> None:
        """Callback khi video sự kiện ghi xong -> Đẩy lên Google Drive & Gửi Telegram."""
        print(f"\n🚀 [UPLOAD DISPATCH] Đang xử lý video bằng chứng: {video_path}")
        gdrive_url = None

        # 1. Tải lên Google Drive
        if self.sec_cfg.get("actions", {}).get("upload_gdrive", True) and self.gdrive_uploader.is_ready:
            gdrive_url = self.gdrive_uploader.upload_file(video_path, mime_type="video/mp4")

        # 2. Gửi thông báo khẩn cấp Telegram
        if self.sec_cfg.get("actions", {}).get("send_telegram", True) and self.telegram_client:
            snapshot_path = metadata.get("snapshot_path")
            zone_name = metadata.get("zone", "Vùng an ninh")
            reason = metadata.get("reason", "Phát hiện đối tượng xâm nhập")
            t_str = time.strftime("%H:%M:%S - %d/%m/%Y")

            caption_lines = [
                "🚨 [CẢNH BÁO ĐỘT NHẬP BAN ĐÊM] 🚨",
                f"⏰ Thời gian : {t_str}",
                f"📍 Vị trí    : {zone_name} (Camera EZVIZ)",
                f"⚠️ Sự kiện   : {reason}",
            ]
            if gdrive_url:
                caption_lines.append(f"☁️ Xem video Cloud: {gdrive_url}")
            else:
                caption_lines.append("💾 Video đã được lưu tại ổ cứng máy tính.")

            caption = "\n".join(caption_lines)

            try:
                if snapshot_path and Path(snapshot_path).exists():
                    self.telegram_client.send_photo(
                        chat_id=self.telegram_settings.chat_id,
                        photo_path=Path(snapshot_path),
                        caption=caption,
                    )
                else:
                    self.telegram_client.send_message(
                        chat_id=self.telegram_settings.chat_id,
                        text=caption,
                    )
                print("📱 [TELEGRAM] Đã gửi thông báo cảnh báo thành công!")
            except Exception as tg_err:
                print(f"❌ [TELEGRAM ERROR]: {tg_err}")

    def run(self, show_gui: bool = True) -> None:
        """Vòng lặp chính xử lý từng frame hình."""
        print("\n" + "=" * 70)
        print("  HỆ THỐNG AI CẢNH BÁO TRỘM ĐÊM KHUYA ĐANG HOẠT ĐỘNG")
        print(f"  • Trạng thái lịch trực : {self.time_guard.get_status_str()}")
        print(f"  • Google Drive Cloud   : [{'SẴN SÀNG' if self.gdrive_uploader.is_ready else 'CHƯA ĐĂNG NHẬP'}]")
        print("  • Bấm phím [Q] hoặc [ESC] trên cửa sổ hình ảnh để dừng hệ thống.")
        print("=" * 70 + "\n")

        fps_counter = 0
        fps_start = time.time()
        fps_display = 0.0

        try:
            while True:
                packet = self.stream.read()
                if packet is None:
                    time.sleep(0.01)
                    continue

                frame = packet.frame
                h, w = frame.shape[:2]
                is_armed = self.time_guard.is_armed()

                # Cập nhật kích thước vùng nếu thay đổi
                if (w, h) != (self.zone_monitor.frame_width, self.zone_monitor.frame_height):
                    self.zone_monitor.update_zones(self.sec_cfg.get("zones", []), (w, h))

                # Đẩy frame vào bộ nhớ đệm video sự kiện (Pre-buffer)
                self.recorder.push_frame(frame)

                # Ghi hình lưu trữ liên tục 24/7
                self.continuous_recorder.push_frame(frame)

                # Tier-0 Motion Gating: Tiết kiệm tài nguyên nếu cảnh tĩnh ban đêm
                gate = self.motion_gater.evaluate(frame)
                detections = []
                tracks = []

                if gate.should_run_ai:
                    # 1. Phát hiện người bằng YOLOv8
                    detections = self.detector.detect(frame)
                    # 2. Định danh và theo dõi bằng ByteTrack
                    tracks = self.tracker.update(detections, frame)

                # Làm sạch dữ liệu theo dõi cũ
                active_ids = [t.track_id for t in tracks]
                self.zone_monitor.cleanup_tracks(active_ids)

                # 3. Phân tích vùng cấm & lảng vảng cho từng người
                alarm_triggered_this_frame = False
                alarm_reason = ""
                alarm_zone = ""

                for t in tracks:
                    event = self.zone_monitor.check_person(t.track_id, list(t.bbox))
                    if event and is_armed:
                        alarm_triggered_this_frame = True
                        alarm_reason = event.reason
                        alarm_zone = event.zone_name
                        print(f"🚨 [ALARM TRIGGERED] {event.reason} (Track #{event.track_id})")

                        # Hú còi báo động khẩn cấp
                        self.siren.trigger_siren()

                        # Kích hoạt ghi hình sự kiện
                        self.recorder.trigger_event({
                            "track_id": event.track_id,
                            "zone": event.zone_name,
                            "reason": event.reason,
                            "timestamp": event.timestamp,
                        })

                # Trực quan hóa HUD lên khung hình
                if show_gui:
                    vis_frame = frame.copy()
                    # Vẽ các vùng cấm
                    vis_frame = self.zone_monitor.draw_zones(vis_frame, is_armed=is_armed)

                    # Vẽ Bounding Box người
                    for t in tracks:
                        x1, y1, x2, y2 = map(int, t.bbox)
                        box_color = (0, 0, 255) if is_armed else (0, 255, 0)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            vis_frame,
                            f"ID #{t.track_id}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            box_color,
                            2,
                        )

                    # Vẽ Banner trạng thái trên cùng
                    hud_color = (0, 0, 180) if is_armed else (50, 100, 50)
                    cv2.rectangle(vis_frame, (0, 0), (w, 40), hud_color, -1)
                    rec_status = " | REC 24/7" if self.continuous_recorder.enabled else ""
                    status_text = f"AI CAMERA AN NINH | {self.time_guard.get_status_str()}{rec_status} | FPS: {fps_display:.1f}"
                    cv2.putText(vis_frame, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    # Nếu đang có báo động, vẽ viền nhấp nháy đỏ quanh màn hình
                    if self.recorder._is_recording:
                        cv2.rectangle(vis_frame, (0, 0), (w, h), (0, 0, 255), 8)
                        cv2.putText(
                            vis_frame,
                            "🔴 REC - CANH BAO XAM NHAP DANG GHI HINH",
                            (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                    cv2.imshow("AI Canh Bao Trom Dem Khuya (EZVIZ)", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        print("\n[*] Người dùng đã bấm dừng hệ thống.")
                        break

                # Tính toán FPS hiển thị
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    fps_display = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

        finally:
            self.continuous_recorder.close()
            self.stream.stop()
            if show_gui:
                cv2.destroyAllWindows()
            print("[*] Đã đóng luồng camera và giải phóng tài nguyên thành công.")


def main():
    parser = argparse.ArgumentParser(description="AI Cảnh Báo Trộm Đêm Khuya & Lưu Cloud Google Drive")
    parser.add_argument("--source", type=str, choices=["screen", "rtsp", "webcam", "video"], default="screen", help="Nguồn camera (mặc định: screen để bắt hình từ EZVIZ Studio, hoặc rtsp, webcam, video)")
    parser.add_argument("--rtsp-url", type=str, default=None, help="Link RTSP nếu không dùng trong config")
    parser.add_argument("--device-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--always-armed", action="store_true", help="Bật chế độ báo động 24/7 (bỏ qua khung giờ đêm)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy chế độ ngầm không mở cửa sổ hiển thị")
    args = parser.parse_args()

    configs = load_security_configs()
    stream = init_camera(args.source, args.rtsp_url, args.device_index, configs["camera"])

    app = SecurityMonitorApp(configs, stream, always_armed=args.always_armed)
    app.run(show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
