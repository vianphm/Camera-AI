# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
project_root = Path(SPECPATH).resolve()

# Tự động thu thập thư viện động, file dữ liệu của ONNX Runtime và OpenCV
ort_datas, ort_binaries, ort_hidden = collect_all('onnxruntime')
cv2_datas, cv2_binaries, cv2_hidden = collect_all('cv2')

# Các thư mục cấu hình, mô hình AI, và giao diện web tĩnh
datas = [
    (str(project_root / 'configs'), 'configs'),
    (str(project_root / 'models'), 'models'),
    (str(project_root / 'frontend'), 'frontend'),
    (str(project_root / 'app_icon.ico'), '.'),
    (str(project_root / 'app_icon.png'), '.'),
] + ort_datas + cv2_datas

binaries = ort_binaries + cv2_binaries

# Danh sách đầy đủ các hidden imports cho web server và AI engine
hiddenimports = [
    'src',
    'src.api',
    'src.api.server',
    'src.pipeline',
    'src.pipeline.realtime_pipeline',
    'src.pipeline.decoupled_pipeline',
    'src.camera',
    'src.camera.stream',
    'src.camera.webcam',
    'src.camera.rtsp',
    'src.camera.video_file',
    'src.camera.synthetic',
    'src.camera.camera_permission',
    'src.camera.multi_camera',
    'src.pose',
    'src.pose.pose_estimator',
    'src.pose.rtmo_estimator',
    'src.pose.keypoints',
    'src.detection',
    'src.detection.detector',
    'src.detection.person_detector',
    'src.detection.detector_factory',
    'src.tracking',
    'src.tracking.tracker',
    'src.tracking.bytetrack',
    'src.tracking.track_manager',
    'src.temporal',
    'src.temporal.temporal_model',
    'src.temporal.temporal_factory',
    'src.anomaly',
    'src.anomaly.anomaly_score',
    'src.risk',
    'src.risk.risk_engine',
    'src.risk.state_machine',
    'src.risk.threshold_manager',
    'src.alerts',
    'src.alerts.alert_manager',
    'src.alerts.event_logger',
    'src.alerts.notification',
    'src.utils',
    'src.utils.config',
    'src.utils.profiler',
    'main',
    'yaml',
    'dotenv',
    'pydantic',
    'pydantic_core',
    'filterpy',
    'filterpy.kalman',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.distance',
    'websockets',
    'websockets.legacy',
    'websockets.legacy.server',
    'websockets.asyncio',
] + ort_hidden + cv2_hidden + collect_submodules('uvicorn') + collect_submodules('fastapi') + collect_submodules('starlette')

# Loại bỏ các module trùng lặp
hiddenimports = sorted(list(set(hiddenimports)))

a = Analysis(
    ['app_runner.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL.ImageTk', 'tests', 'archive'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fall_and_Stroke_Warning_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # ẨN HOÀN TOÀN CỬA SỔ CONSOLE THEO YÊU CẦU CỦA NGƯỜI DÙNG!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Fall_and_Stroke_Warning_System',
)
