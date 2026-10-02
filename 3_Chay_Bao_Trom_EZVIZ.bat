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
echo     AI CẢNH BÁO TRỘM ĐÊM KHUYA & GHI HÌNH 24/7 — EZVIZ STUDIO & CLOUD
echo ===============================================================================
echo  [*] Nguồn hình ảnh    : Bắt hình trực tiếp từ cửa sổ EZVIZ Studio
echo  [*] Ghi hình liên tục : Lưu trữ 24/7 vào thư mục data/records_24_7/ (đoạn 15p)
echo  [*] Tự động dọn dẹp   : Xóa file cũ nhất khi đầy ổ cứng (FIFO Rolling Storage)
echo  [*] Cảnh báo trộm đêm : Nhận diện người, hú còi và cắt clip đẩy Google Drive
echo ===============================================================================
echo.
echo  [Lưu ý] Bạn hãy mở sẵn phần mềm EZVIZ Studio để xem hình camera trước!
echo  [Mẹo] Để dừng hệ thống, nhấn phím [Q] trên cửa sổ camera hoặc đóng cửa sổ này.
echo.

:: Khởi chạy bộ giám sát an ninh và ghi hình 24/7
"%~dp0.venv\Scripts\python.exe" "%~dp0run_security_monitor.py" --source screen

echo.
echo [Hệ thống đã dừng an toàn]
pause
