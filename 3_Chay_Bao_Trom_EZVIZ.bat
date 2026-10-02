@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AI Cảnh Báo Trộm Đêm Khuya — Camera EZVIZ & Cloud Google Drive
cls

:: Kiểm tra nếu chưa cài đặt môi trường ảo .venv thì tự động chạy cài đặt
if not exist ".venv\Scripts\python.exe" (
    echo [*] Phát hiện hệ thống chưa được cài đặt lần đầu.
    echo [*] Đang tự động chuyển sang trình cài đặt 1-Click...
    echo.
    call "1_Cai_Dat_Tu_Dong.bat"
    if not exist ".venv\Scripts\python.exe" (
        echo [Lỗi] Quá trình cài đặt chưa thành công. Không thể khởi động.
        pause
        exit /b 1
    )
    cls
)

echo ===============================================================================
echo     AI CẢNH BÁO TRỘM ĐÊM KHUYA — CAMERA EZVIZ & GOOGLE DRIVE CLOUD
echo ===============================================================================
echo  [*] Lịch trực an ninh : Tự động kích hoạt từ 22h30 đêm đến 05h30 sáng
echo  [*] Chống trộm bằng AI: YOLOv8 + ByteTrack + Vùng cấm đa giác (ROI)
echo  [*] Lưu trữ bằng chứng: Cắt clip MP4 tự động và đẩy lên Google Drive
echo  [*] Thông báo khẩn cấp: Telegram Bot kèm ảnh chụp và link xem video
echo ===============================================================================
echo.
echo  [Mẹo] Để dừng hệ thống, nhấn phím [Q] trên cửa sổ camera hoặc đóng cửa sổ này.
echo.

:: Khởi chạy bộ giám sát an ninh
"%~dp0.venv\Scripts\python.exe" "%~dp0run_security_monitor.py" --source rtsp

echo.
echo [Hệ thống đã dừng an toàn]
pause
