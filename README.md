# Fall and Stroke Warning System: Real-Time Camera Monitoring & Abnormal Behavior Detection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2012)-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Pose%20%2B%20Detect-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, real-time computer vision system that monitors video streams from security cameras, webcams, or RTSP feeds to detect falls, sudden collapses, abnormal gait, and prolonged immobility in elderly individuals.

> [!IMPORTANT]
> **Clinical Safety & Non-Diagnostic Principle**:
> This system is designed for **safety alerting and risk escalation**, NOT clinical medical diagnosis. It does not claim a person has had a stroke or fracture. All high-risk events dispatch an alert for human verification:
> *"Possible medical emergency / abnormal behavior detected. Please check the person."*

---

## Key Features

- **Multi-Source Video Ingestion**: Threaded reader supporting USB Webcams, RTSP IP Cameras (with automatic reconnection backoff), and local MP4/AVI videos.
- **Robust Perception & Multi-Object Tracking**: YOLOv8s Person Detection coupled with ByteTrack for continuous identity preservation even during partial floor occlusions.
- **Accurate Pose Extraction**: 17 COCO keypoints normalized around the torso root, invariant to camera distance and perspective.
- **Deep Temporal Behavior Modeling**: Spatial-Temporal Transformer and TCN over sliding sequence windows ($T=30$ frames / 2.0s) to recognize complex multi-stage actions (`walking -> stumbling -> fall -> immobile`).
- **Unsupervised Anomaly Scoring**: Deep Pose Autoencoder trained on normal activities of daily living (ADL) to flag novel, unseen anomalous dynamics.
- **Clinical Risk Reasoner & Debounced State Machine**: Multi-signal fusion ($w_{act} + w_{anom} + w_{kin} + w_{immob}$) evaluated through an anti-flapping finite state machine (`NORMAL -> SUSPICIOUS -> ABNORMAL -> HIGH_RISK -> ALERT_SENT`).
- **Privacy-by-Design**: Local edge execution, real-time face blurring, and zero raw cloud streaming.
- **Real-Time Web API & Dashboard**: FastAPI backend with WebSocket live streams and instant alert dispatch.

---

## System Architecture

```text
Camera Stream (RTSP / Webcam / File)
   ↓
Frame Preprocessing & Zero-Lag Buffer
   ↓
YOLOv8 Person Detection (Class: Person)
   ↓
ByteTrack Multi-Object Tracker (Track ID & Trajectory)
   ↓
YOLOv8-Pose (17 Keypoints Normalization)
   ↓
Temporal Sliding Window Ring Buffer (T=30 frames)
   ↓
┌──────────────────────────────────────────────┐
│        Dual-Stream Intelligence Engine       │
│  1. Supervised Spatial-Temporal Transformer  │
│  2. Unsupervised Pose Sequence Autoencoder   │
│  3. Kinematic Drop Velocity & Aspect Ratio   │
└──────────────────────────────────────────────┘
   ↓
Multi-Signal Risk Fusion Engine
   ↓
Debounced Event State Machine (Hysteresis Filter)
   ↓
Alert Manager & Dispatcher (WebSocket / Webhook / Telegram)
```

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.11
- NVIDIA GPU with CUDA 12.x support (e.g. RTX 3050 Laptop GPU 6GB)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/QuocCuong66/Project-monitoring-model.git
cd Project-monitoring-model

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch with CUDA support (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Running Real-Time Inference (Chỉ cần chạy main.py)

Toàn bộ hệ thống có thể khởi chạy chỉ với **một lệnh duy nhất**:
```powershell
.\.venv\Scripts\python.exe main.py
```
> Hệ thống sẽ tự động kiểm tra phần cứng, tải các mô hình AI, kết nối camera/webcam, khởi chạy Web API dashboard tại `http://localhost:8000`, và mở giao diện HUD trực quan hóa thời gian thực.

**Các tùy chọn nâng cao qua cờ dòng lệnh:**
```powershell
# Chạy với camera RTSP (IP Camera):
.\.venv\Scripts\python.exe main.py --source rtsp --rtsp-url "rtsp://username:password@192.168.1.100:554/stream"

# Chạy với tệp video MP4:
.\.venv\Scripts\python.exe main.py --source video --video-path "data/videos/test_fall.mp4"

# Chạy ở chế độ không màn hình (Headless server mode):
.\.venv\Scripts\python.exe main.py --no-gui
```

### 4. Running the Web API Server
```powershell
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
Open browser at: `http://localhost:8000/docs` for interactive Swagger API.

---

## Documentation Links

- [**`ARCHITECTURE.md`**](ARCHITECTURE.md): Comprehensive system architecture, data flows, and state machine.
- [**`MODEL_SELECTION.md`**](MODEL_SELECTION.md): Benchmark matrix and hardware resource allocation rationale.
- [**`DATASET.md`**](DATASET.md): Action taxonomy, public benchmarks, hard negatives, and leak-free split.
- [**`TRAINING.md`**](TRAINING.md): Training pipeline, augmentations, loss functions, and ONNX/TensorRT export.
- [**`EVALUATION.md`**](EVALUATION.md): Accuracy metrics, latency profiler, and real-time benchmark reports.
- [**`IMPLEMENTATION_PLAN.md`**](IMPLEMENTATION_PLAN.md): 13-phase roadmap and directory tree details.
