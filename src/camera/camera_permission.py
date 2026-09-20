"""Camera permission and accessibility diagnostic helper for Windows."""

import sys
import subprocess
from typing import Tuple, List, Optional
import cv2


def test_camera_access(device_index: int = 0) -> Tuple[bool, str, List[int]]:
    """Kiểm tra quyền và khả năng mở camera trên Windows.
    
    Returns:
        (success: bool, message: str, working_indices: List[int])
    """
    working_devices = []
    
    # Quét thử các cổng từ 0 đến 3
    for idx in range(4):
        # Thử DirectShow trước
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Thử Media Foundation (Windows native)
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                working_devices.append(idx)
                
    if device_index in working_devices:
        return True, f"Camera {device_index} đang hoạt động bình thường.", working_devices
        
    if len(working_devices) > 0:
        return True, f"Camera {device_index} không khả dụng, nhưng tìm thấy Camera {working_devices[0]} đang hoạt động.", working_devices
        
    # Không mở được bất kỳ camera nào
    msg = (
        "Không thể mở Camera! Có thể do:\n"
        "1. Quyền truy cập Camera trong Windows đang bị TẮT (Camera Privacy Settings).\n"
        "2. Đang có ứng dụng khác (Zoom, Teams, Google Meet, Zalo...) chiếm giữ Camera.\n"
        "3. Chưa cắm hoặc chưa nhận diện được thiết bị Camera USB."
    )
    return False, msg, working_devices


def request_windows_camera_permission_dialog() -> bool:
    """Hiển thị hộp thoại hỏi người dùng cấp quyền camera và mở Windows Settings nếu cần."""
    if sys.platform != "win32":
        return False
        
    try:
        import ctypes
        MB_YESNO = 0x00000004
        MB_ICONWARNING = 0x00000030
        MB_TOPMOST = 0x00040000
        IDYES = 6
        
        title = "Fall & Stroke Warning System — Yêu Cầu Quyền Camera"
        message = (
            "Hệ thống phát hiện Camera chưa được cấp quyền hoặc đang bị ứng dụng khác chiếm giữ!\n\n"
            "Để hệ thống có thể nhận diện và cảnh báo té ngã / đột quỵ, vui lòng đảm bảo:\n"
            "• Đã BẬT quyền 'Allow apps to access your camera' trong Windows.\n"
            "• Tắt các ứng dụng đang dùng camera (Zoom, Teams, Zalo...).\n\n"
            "Bạn có muốn MỞ CÀI ĐẶT QUYỀN CAMERA (Windows Settings) ngay bây giờ không?"
        )
        
        response = ctypes.windll.user32.MessageBoxW(
            0, message, title, MB_YESNO | MB_ICONWARNING | MB_TOPMOST
        )
        
        if response == IDYES:
            subprocess.Popen(["cmd", "/c", "start", "ms-settings:privacy-webcam"], shell=True)
            return True
            
    except Exception as e:
        print(f"[Warning] Không thể hiển thị hộp thoại cấp quyền: {e}")
        
    return False


if __name__ == "__main__":
    ok, msg, cams = test_camera_access(0)
    print(f"Status: {ok}")
    print(f"Message: {msg}")
    print(f"Working cameras: {cams}")
    if not ok:
        request_windows_camera_permission_dialog()
