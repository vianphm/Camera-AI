@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Trình Đóng Gói Ứng Dụng Độc Lập
cls

echo ===============================================================================
echo     FALL AND STROKE WARNING SYSTEM — TIẾN TRÌNH ĐÓNG GÓI PYINSTALLER
echo ===============================================================================
echo.
echo  [*] Trạng thái: Đang chuẩn bị môi trường đóng gói...
echo  [*] Mục tiêu: Tạo ứng dụng độc lập (.exe) không cần cài Python
echo  [*] Cấu hình giao diện: No-Console (Ẩn cửa sổ đen) + Native App Window
echo ===============================================================================
echo.

:: Kiểm tra môi trường ảo
if not exist ".venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy môi trường ảo .venv!
    echo Vui lòng đảm bảo đã khởi tạo .venv trước khi đóng gói.
    pause
    exit /b 1
)

:: Kiểm tra PyInstaller
if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [*] Đang cài đặt PyInstaller vào .venv...
    .\.venv\Scripts\python.exe -m pip install pyinstaller
)

echo [*] Bắt đầu biên dịch ứng dụng bằng PyInstaller (Quá trình này mất khoảng 1 - 2 phút)...
echo.

.\.venv\Scripts\pyinstaller.exe --clean "fall_and_stroke_monitor.spec" -y

if %errorlevel% neq 0 (
    echo.
    echo ===============================================================================
    echo  [THẤT BẠI] Đã xảy ra lỗi trong quá trình biên dịch PyInstaller!
    echo ===============================================================================
    pause
    exit /b %errorlevel%
)

:: Tạo sẵn thư mục lưu trữ sự kiện snapshot và copy cấu hình
if not exist "dist\Fall_and_Stroke_Warning_System\data\processed\events" (
    mkdir "dist\Fall_and_Stroke_Warning_System\data\processed\events" 2>nul
)
if not exist "dist\Fall_and_Stroke_Warning_System\configs" (
    xcopy /E /I /Y "configs" "dist\Fall_and_Stroke_Warning_System\configs" >nul
)
if not exist "dist\Fall_and_Stroke_Warning_System\frontend" (
    xcopy /E /I /Y "frontend" "dist\Fall_and_Stroke_Warning_System\frontend" >nul
)
if not exist "dist\Fall_and_Stroke_Warning_System\models" (
    xcopy /E /I /Y "models" "dist\Fall_and_Stroke_Warning_System\models" >nul
)
copy /Y "HUONG_DAN_SU_DUNG.txt" "dist\Fall_and_Stroke_Warning_System\HUONG_DAN_SU_DUNG.txt" >nul

echo.
echo ===============================================================================
echo  [THÀNH CÔNG] ĐÃ ĐÓNG GÓI ỨNG DỤNG HOÀN CHỈNH!
echo ===============================================================================
echo  • Thư mục ứng dụng: dist\Fall_and_Stroke_Warning_System\
echo  • File chạy chính : dist\Fall_and_Stroke_Warning_System\Fall_and_Stroke_Warning_System.exe
echo.
echo  Người dùng chỉ cần nhấp đúp vào:
echo  👉 Fall_and_Stroke_Warning_System.exe
echo  để sử dụng ngay lập tức mà KHÔNG CẦN cài thêm bất kỳ phần mềm nào khác!
echo ===============================================================================
echo.
pause
