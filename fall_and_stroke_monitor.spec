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

# Các thư mục cấu hình, giao diện web tĩnh và CHỈ các mô hình AI mà pipeline thực sự dùng
# (RTMO-s pose + ST-Transformer + Pose Autoencoder). Các file .pt / yolov8 / int8 thử nghiệm
# không được đóng gói để giảm dung lượng bộ cài.
REQUIRED_MODELS = [
    'models/pose/rtmo-s.onnx',
    'models/checkpoints/best_st_transformer.onnx',
    'models/checkpoints/best_pose_autoencoder.onnx',
]
for rel in REQUIRED_MODELS:
    if not (project_root / rel).exists():
        raise SystemExit(f'[BUILD] Thiếu mô hình bắt buộc: {rel}')

datas = [
    (str(project_root / 'configs'), 'configs'),
    (str(project_root / 'frontend'), 'frontend'),
    (str(project_root / 'app_icon.ico'), '.'),
    (str(project_root / 'app_icon.png'), '.'),
] + [(str(project_root / rel), str(Path(rel).parent)) for rel in REQUIRED_MODELS] + ort_datas + cv2_datas

# Ứng dụng chạy GPU qua DirectML. Các provider CUDA/TensorRT còn sót lại trong .venv
# (từ onnxruntime-gpu) không dùng được với bản DirectML và nặng ~165 MB -> loại bỏ.
_EXCLUDED_ORT_DLLS = ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_tensorrt.dll')
binaries = [b for b in ort_binaries if Path(b[0]).name.lower() not in _EXCLUDED_ORT_DLLS] + cv2_binaries

# Danh sách đầy đủ các hidden imports cho web server và AI engine
hiddenimports = collect_submodules('src') + [
    'main',
    'yaml',
    'dotenv',
    'pydantic',
    'pydantic_core',
    'scipy',
    'scipy.optimize',
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
    excludes=['tkinter', 'matplotlib', 'PIL.ImageTk', 'tests', 'archive', 'torch', 'torchvision',
              'ultralytics', 'tensorrt', 'cuda', 'PIL', 'pytest', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Lọc lần nữa sau Analysis (hook onnxruntime có thể tự thêm lại các DLL này)
a.binaries = [b for b in a.binaries if Path(b[0]).name.lower() not in _EXCLUDED_ORT_DLLS]

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
