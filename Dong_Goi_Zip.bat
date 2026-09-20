@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Đóng Gói File ZIP
cls

echo ===============================================================================
echo     FALL AND STROKE WARNING SYSTEM — TỰ ĐỘNG ĐÓNG GÓI GỬI CHO NGƯỜI DÙNG
echo ===============================================================================
echo.
echo [*] Đang thực hiện đóng gói dự án sạch sẽ thành file ZIP...
echo.

.\.venv\Scripts\python.exe dong_goi_du_an.py

echo.
pause
