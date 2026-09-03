# KẾ HOẠCH TRIỂN KHAI HỆ THỐNG (IMPLEMENTATION PLAN)
## AI Camera Monitoring & Abnormal Behavior Detection for Elderly Care

---

## 1. KẾT QUẢ AUDIT MÔI TRƯỜNG & PHẦN CỨNG (SYSTEM AUDIT SUMMARY)

Qua việc kiểm tra hệ thống thực tế tại workspace `d:\Project-monitoring-model`:

| Thành phần | Thông số kiểm tra thực tế | Nhận định kỹ thuật |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows 11 64-bit | Shell: PowerShell |
| **CPU** | 12th Gen Intel(R) Core(TM) i5-12450HX (8 cores, 12 threads) | Đủ năng lực xử lý đa luồng I/O, giải mã RTSP và Kalman Filter |
| **RAM** | 16 GB | Dư dả nạp mô hình và sliding window sequence buffers |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) | Driver: 577.05, hỗ trợ CUDA 12.x |
| **Python** | Python 3.11.9 (`C:\Users\QC\AppData\Local\Programs\Python\Python311\python.exe`) | Đã cài đặt chuẩn xác theo yêu cầu người dùng |
| **Hiện trạng PyTorch** | PyTorch 2.13.0+cpu (Bản CPU) đang ở global Python 3.11 | **Hành động bắt buộc**: Tạo virtual environment `.venv` riêng với `py -3.11 -m venv .venv` và cài đặt `torch torchvision` hỗ trợ CUDA 12.x để khai thác GPU RTX 3050 |
| **Repository** | `d:\Project-monitoring-model` | Repository Git mới, sạch sẽ (chưa có code cũ bị xung đột) |

---

## 2. CẤU TRÚC THƯ MỤC CHUẨN PRODUCTION-READY

```text
d:\Project-monitoring-model/
│
├── README.md                          # Hướng dẫn tổng thể, setup, hướng dẫn chạy
├── AGENTS.md                          # Hướng dẫn cho AI agent và conventions
├── ARCHITECTURE.md                    # Kiến trúc kỹ thuật chi tiết
├── MODEL_SELECTION.md                 # Bảng benchmark và cơ sở chọn model
├── DATASET.md                         # Schema dữ liệu, benchmarks và hard negatives
├── TRAINING.md                        # Chiến lược training, losses, export ONNX/TensorRT
├── EVALUATION.md                      # Chỉ số kiểm thử, benchmark realtime và profile trễ
├── IMPLEMENTATION_PLAN.md             # Kế hoạch triển khai chi tiết từng phase
├── pyproject.toml                     # Metadata dự án và cấu hình build
├── requirements.txt                   # Danh mục thư viện phụ thuộc (Python 3.11 + CUDA)
├── .env.example                       # Biến môi trường mẫu (Camera URL, Telegram token, ...)
├── .gitignore                         # Bỏ qua models nặng, venv, cache, data tạm
├── docker-compose.yml                 # Triển khai Docker container
│
├── configs/                           # Cấu hình độc lập (YAML-driven)
│   ├── model.yaml                     # Cấu hình detector, tracker, pose, temporal, anomaly
│   ├── inference.yaml                 # Cấu hình batch size, device, fp16, display overlay
│   ├── training.yaml                  # Cấu hình learning rate, epochs, losses, augmentations
│   ├── thresholds.yaml                # Ngưỡng risk, debounce duration, cooldown, weights
│   └── camera.yaml                    # URL camera, RTSP, webcam index, resolution, FPS
│
├── data/                              # Lưu trữ dữ liệu thử nghiệm & artifacts
│   ├── raw/                           # Video thô
│   ├── videos/                        # Video mẫu phục vụ test
│   ├── frames/                        # Frame trích xuất tạm thời
│   ├── poses/                         # Mảng skeleton trích xuất (.npz)
│   ├── annotations/                   # JSONL metadata gán nhãn
│   ├── processed/                     # Sequence datasets sẵn sàng nạp training
│   └── splits/                        # Split file train.jsonl, val.jsonl, test.jsonl
│
├── datasets/                          # Thư mục chứa sample clips theo phân loại
│   ├── normal/
│   ├── fall/
│   ├── abnormal/
│   ├── walking/
│   ├── sitting/
│   ├── standing/
│   ├── lying/
│   └── emergency/
│
├── models/                            # Thư mục lưu trữ trọng số và export
│   ├── checkpoints/                   # Checkpoints tốt nhất trong training
│   ├── detector/                      # Weights YOLOv8s/v11s person detector
│   ├── tracker/                       # Cấu hình ByteTrack
│   ├── pose/                          # Weights YOLOv8s-Pose
│   ├── action/                        # Weights Temporal Action Recognizer
│   ├── temporal/                      # ONNX / TensorRT engine
│   └── anomaly/                       # Weights Anomaly Autoencoder
│
├── src/                               # Toàn bộ mã nguồn cốt lõi (Modular, Typed)
│   ├── camera/                        # Ingestion layer (Webcam, RTSP, Video File)
│   │   ├── __init__.py
│   │   ├── stream.py                  # Abstract Base Class CameraStream & Threaded Reader
│   │   ├── rtsp.py                    # RTSP Stream với Auto-reconnect
│   │   ├── webcam.py                  # USB / Webcam Stream
│   │   └── video_file.py              # File-based Stream
│   │
│   ├── detection/                     # Person Detection layer
│   │   ├── __init__.py
│   │   ├── detector.py                # Abstract Base Class PersonDetector & Detection dataclass
│   │   └── person_detector.py         # Triển khai YOLOv8/v11 Person Detector
│   │
│   ├── tracking/                      # Multi-Object Tracking layer
│   │   ├── __init__.py
│   │   ├── tracker.py                 # Abstract Base Class Tracker & Track dataclass
│   │   └── track_manager.py           # Triển khai ByteTrack & Quản lý lịch sử quỹ đạo
│   │
│   ├── pose/                          # Pose Estimation layer
│   │   ├── __init__.py
│   │   ├── pose_estimator.py          # Abstract Base Class PoseEstimator
│   │   ├── yolo_pose.py               # Triển khai YOLOv8-Pose trích xuất 17 keypoints
│   │   └── keypoints.py               # Chuẩn hóa tọa độ, tính góc khớp, khoảng cách giải phẫu
│   │
│   ├── temporal/                      # Temporal Modeling layer (Trọng tâm)
│   │   ├── __init__.py
│   │   ├── sequence_buffer.py         # Ring Buffer (deque) quản lý sliding window theo track_id
│   │   ├── temporal_features.py       # Trích xuất đặc trưng động học (V_y, A_y, aspect ratio)
│   │   └── temporal_model.py          # Spatial-Temporal Transformer / TCN Network
│   │
│   ├── action/                        # Action Classification layer
│   │   ├── __init__.py
│   │   ├── action_classifier.py       # Supervised Action Classifier Wrapper
│   │   └── behavior_classifier.py     # Phân tích chuỗi hành vi phức tạp
│   │
│   ├── anomaly/                       # Unsupervised Anomaly Detection layer
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py        # Abstract Base Class AnomalyDetector
│   │   └── anomaly_score.py           # Pose Sequence Autoencoder / VAE Reconstruction Scorer
│   │
│   ├── risk/                          # Clinical Safety & Risk Reasoning layer
│   │   ├── __init__.py
│   │   ├── risk_engine.py             # Động cơ hợp nhất đa tín hiệu & chấm điểm nguy cơ
│   │   ├── state_machine.py           # Finite State Machine với cơ chế lọc trễ Debounce/Hysteresis
│   │   └── threshold_manager.py       # Quản lý và nạp ngưỡng động từ YAML
│   │
│   ├── alerts/                        # Alert & Notification layer
│   │   ├── __init__.py
│   │   ├── alert_manager.py           # Điều phối cảnh báo, quản lý Cooldown & Deduplication
│   │   ├── notification.py            # Dispatcher đa kênh (WebSocket, REST, Telegram, Webhook)
│   │   └── event_logger.py            # Ghi log sự kiện mã hóa, bảo vệ quyền riêng tư
│   │
│   ├── pipeline/                      # End-to-End Orchestrator
│   │   ├── __init__.py
│   │   └── realtime_pipeline.py       # Pipeline thời gian thực kết nối toàn bộ luồng
│   │
│   ├── api/                           # Web API & Dashboard backend
│   │   ├── __init__.py
│   │   └── server.py                  # FastAPI REST API + WebSockets cho Live Video / Alerts
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── config.py                  # Loader cấu hình YAML an toàn
│       ├── visualizer.py              # Vẽ BBox, Skeleton, Trạng thái, Risk Score lên frame
│       ├── profiler.py                # Đo đạc FPS, Latency từng mắt xích và tài nguyên VRAM
│       └── privacy.py                 # Làm mờ khuôn mặt (Face blurring / anonymization)
│
├── training/                          # Kịch bản huấn luyện mô hình
│   ├── train_detector.py              # Fine-tune Person Detector
│   ├── train_action.py                # Huấn luyện ST-Transformer / TCN
│   ├── train_temporal.py              # Huấn luyện mô hình chuỗi thời gian
│   ├── train_anomaly.py               # Huấn luyện Unsupervised Pose Autoencoder
│   └── evaluate.py                    # Đánh giá Precision, Recall, F1, Confusion Matrix
│
├── inference/                         # Kịch bản chạy suy luận
│   ├── realtime.py                    # Chạy suy luận camera thời gian thực với UI overlay
│   └── batch.py                       # Chạy suy luận hàng loạt trên danh sách video
│
├── scripts/                           # Công cụ hỗ trợ và benchmark
│   ├── extract_frames.py              # Trích xuất frames từ video
│   ├── prepare_dataset.py             # Tiền xử lý và tạo dataset splits
│   ├── generate_poses.py              # Trích xuất và serialize poses thành .npz
│   ├── export_onnx.py                 # Xuất mô hình PyTorch sang ONNX / TensorRT
│   └── benchmark.py                   # Benchmark FPS, Latency và VRAM
│
└── tests/                             # Unit tests & Integration tests
    ├── test_camera.py
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_pose.py
    ├── test_temporal.py
    ├── test_anomaly.py
    ├── test_risk_engine.py
    ├── test_state_machine.py
    └── test_pipeline.py
```

---

## 3. LỘ TRÌNH TRIỂN KHAI TỪNG GIAI ĐOẠN (PHASE-BY-PHASE ROADMAP)

### Phase 0: Môi trường & Tài liệu Kiến trúc (Hoàn tất)
* Khởi tạo virtual environment Python 3.11 (`.venv`).
* Soạn thảo 6 tài liệu thiết kế cốt lõi: `ARCHITECTURE.md`, `MODEL_SELECTION.md`, `DATASET.md`, `TRAINING.md`, `EVALUATION.md`, `IMPLEMENTATION_PLAN.md`.
* Chuẩn bị file quản lý dependencies: `requirements.txt`, `pyproject.toml`, `.env.example`, `.gitignore`.

### Phase 1: Camera Ingestion Layer (`src/camera/`)
* Xây dựng `CameraStream` interface với luồng đọc tách biệt (Threaded capture loop) và Ring Queue chống tích lũy trễ (Zero-lag queue).
* Hỗ trợ webcam USB, IP camera / RTSP (với auto-reconnect backoff), và MP4 video file.
* Viết unit test kiểm tra khả năng đọc frame mượt mà tại target 30 FPS.

### Phase 2: Person Detection & Multi-Object Tracking (`src/detection/`, `src/tracking/`)
* Xây dựng `PersonDetector` interface và lớp triển khai `YOLOv8PersonDetector`.
* Xây dựng `Tracker` interface và lớp triển khai `ByteTrackManager`.
* Quản lý lịch sử quỹ đạo trọng tâm `PersonTrack`, tính toán vận tốc di chuyển mặt phẳng sàn.
* Unit test: Kiểm tra khả năng gán `track_id` liên tục khi người di chuyển và thay đổi tư thế.

### Phase 3: Human Pose Estimation & Keypoints (`src/pose/`)
* Xây dựng `PoseEstimator` interface và triển khai `YOLOPoseEstimator` trích xuất 17 điểm COCO keypoints.
* Xây dựng `keypoints.py`: Chuẩn hóa tọa độ dựa trên gốc tọa độ hông (Hip-center) và chiều dài thân người (Torso-scaling).
* Viết module trực quan hóa khung xương `visualizer.py` phục vụ debug hiển thị.

### Phase 4: Baseline Action Recognition & Temporal Sequence Buffer (`src/temporal/`, `src/action/`)
* Xây dựng `SequenceBuffer`: Quản lý bộ đệm trượt $T=30$ frames theo từng `track_id`.
* Trích xuất đặc trưng động học vật lý: Tốc độ rơi thẳng đứng $V_y$, gia tốc rơi $A_y$, tỷ lệ khung bao $W/H$, góc nghiêng cơ thể.
* Xây dựng baseline classifier phân loại các hành vi cơ bản (`walking`, `sitting`, `standing`, `lying`).

### Phase 5: Deep Temporal Modeling (`src/temporal/`)
* Triển khai kiến trúc **Spatial-Temporal Transformer (ST-Transformer)** và **TCN (Temporal Convolutional Network)** trong PyTorch.
* Xử lý chuỗi xương đa chiều $T \times 17 \times 3$, phân loại 10 lớp hành vi, đặc biệt nhận diện hành động quá độ: `falling`, `stumbling`, `getting_up`, `immobile`.
* Unit test: Kiểm thử forward pass với tensor giả lập, kiểm tra trễ $\le 5\text{ ms}$.

### Phase 6: Unsupervised Anomaly Detection (`src/anomaly/`)
* Xây dựng `PoseSequenceAutoencoder` (Conv1D/GRU based) học phân phối chuỗi cử động bình thường.
* Tính toán chỉ số dị biệt `AnomalyScore` $\in [0, 1]$ từ sai số tái tạo ($MSE / L_1$).
* Kiểm thử khả năng phát hiện các chuyển động bất thường ngoài các nhãn được định nghĩa sẵn.

### Phase 7: Multi-Signal Fusion Engine (`src/risk/`)
* Xây dựng `MultiSignalFusionEngine`: Hợp nhất xác suất hành động khẩn cấp, điểm dị biệt, gia tốc rơi, và chỉ số bất động theo công thức đa biến có trọng số cấu hình động.
* Đảm bảo tính minh bạch trong quá trình ra quyết định (Explainable Evidence).

### Phase 8: Risk Engine & Finite State Machine (`src/risk/`)
* Xây dựng `EventStateMachine`:
  `NORMAL` $\rightarrow$ `SUSPICIOUS` $\rightarrow$ `ABNORMAL` $\rightarrow$ `HIGH_RISK` $\rightarrow$ `ALERT_SENT` $\rightarrow$ `WAITING_FOR_CONFIRMATION`.
* Tích hợp bộ lọc Hysteresis và Debounce time để triệt tiêu báo động giả khi đối tượng cúi nhặt đồ hoặc ngồi nhanh.
* Tích hợp logic tự giải tỏa (`EVENT_RECOVERED`) khi đối tượng tự đứng dậy.

### Phase 9: Alert Dispatcher & Notification Pipeline (`src/alerts/`)
* Xây dựng `AlertManager`: Quản lý thời gian Cooldown giữa các lần báo động (tránh spam cảnh báo).
* Xây dựng `NotificationDispatcher`:
  * WebSocket broadcasting cho giao diện giám sát realtime.
  * Webhook / REST POST cho hệ thống quản lý trung tâm.
  * Telegram Bot notification (gửi kèm snapshot frame làm mờ mặt).
* Tuân thủ quy tắc an toàn: Không đưa ra chẩn đoán y tế, định dạng thông báo:
  > *"Possible medical emergency / abnormal behavior detected. Please check the person."*

### Phase 10: Real-Time Pipeline Orchestration (`src/pipeline/`, `inference/`)
* Tích hợp toàn bộ các tầng vào lớp `RealtimePipeline`.
* Tối ưu hóa đa luồng: I/O Thread, Inference Worker Thread, Rendering/Visualizer Thread.
* Cung cấp kịch bản chạy dòng lệnh trực tiếp: `inference/realtime.py` (hỗ trợ hiển thị UI hoặc headless mode).

### Phase 11: Web API Server & Dashboard (`src/api/`)
* Xây dựng FastAPI server với các endpoint:
  * `GET /health`, `GET /status`
  * `POST /camera/start`, `POST /camera/stop`
  * `GET /events`, `GET /events/{id}`
  * `WS /ws/stream` (Stream video mjpeg hoặc metadata overlays)
  * `WS /ws/alerts` (Stream sự kiện khẩn cấp tức thời)

### Phase 12: Training, Benchmark & Production Packaging
* Hoàn thiện các scripts trong `training/` và `scripts/`.
* Chạy benchmark đo đạc FPS, Latency từng tầng, VRAM và CPU utilization trên máy thực tế.
* Chuẩn bị Dockerfile và `docker-compose.yml` cho triển khai production.

---

## 4. BẢO ĐẢM TÍNH KHẢ THI TRÊN PHẦN CỨNG THỰC TẾ

* **VRAM RTX 3050 (6GB)**:
  Kiến trúc kết hợp `YOLOv8s-Pose` (1 pass trích xuất cả người và xương) + `ST-Transformer` chạy trên chuỗi xương nhẹ (không chạy 3D-CNN cồng kềnh trên frame video) tiêu thụ **dưới 3.5 GB VRAM**, cho phép duy trì tốc độ **$\ge 35-45\text{ FPS}$** liên tục mà không gây quá nhiệt hay cạn kiệt bộ nhớ đồ họa.
