# Repository QA Audit Report

## 1. Executive Summary
This document provides a comprehensive technical audit of the complete codebase for the **Real-Time AI Camera Monitoring & Abnormal Behavior Detection System**. Every major component, class, method, dependency, and potential operational risk is systematically cataloged.

## 2. Component Inventory & Audit Matrix

| Component | File | Class | Key Function(s) | Purpose | Dependencies | Current Status | Potential Problem / Risk |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Camera Ingestion** | `src/camera/stream.py` | `CameraStream` | `start()`, `read()`, `_capture_loop()`, `release()` | Threaded frame acquisition with zero-lag drop-oldest queue | `queue`, `threading`, `numpy` | **VERIFIED (PASS)** | If consumer thread hangs, camera memory must not leak |
| **Webcam Stream** | `src/camera/webcam.py` | `WebcamStream` | `_open_capture()`, `_read_raw_frame()` | OpenCV DirectShow/V4L2 capture | `cv2` | **VERIFIED (PASS)** | Camera index unavailable if another app locks it |
| **RTSP Stream** | `src/camera/rtsp.py` | `RTSPStream` | `_open_capture()`, `_reconnect_with_backoff()` | Resilient IP camera ingestion with exponential reconnect | `cv2`, `time` | **VERIFIED (PASS)** | Heavy network jitter or packet loss on RTSP stream |
| **Multi-Camera** | `src/camera/multi_camera.py` | `MultiCameraManager` | `add_camera()`, `read_all()`, `create_grid_display()` | Scalable multi-stream orchestration and grid visualization | `cv2`, `threading` | **VERIFIED (PASS)** | Scalability bounded by CPU decode when running >4 1080p RTSP feeds concurrently |
| **Person Detection** | `src/detection/person_detector.py` | `YOLOv8PersonDetector` | `detect()` | Person bounding box localization (COCO Class 0) | `ultralytics`, `torch` | **VERIFIED (PASS)** | Extreme crowd occlusion or persons lying down partially off-screen |
| **Detector Factory**| `src/detection/detector_factory.py` | Factory Registry | `create_detector()`, `register_detector()` | Model decoupling without code alteration | Dynamic registry | **VERIFIED (PASS)** | Non-existent model weight path provided in YAML |
| **Object Tracking** | `src/tracking/track_manager.py` | `ByteTrackManager` | `update()`, `_associate()` | Two-stage high/low score Kalman association | `filterpy`, `numpy` | **VERIFIED (PASS)** | ID switches during long occlusions (>4s) without visual re-ID |
| **Pose Estimation** | `src/pose/pose_estimator.py` | `YOLOv8PoseEstimator` | `estimate()`, `_extract_keypoints()` | 17 COCO skeletal keypoints estimation | `ultralytics`, `torch` | **VERIFIED (PASS)** | Low lighting causing noisy keypoint jitter |
| **Keypoint Normalization** | `src/pose/keypoints.py` | Standalone utils | `normalize_keypoints()`, `compute_torso_angle()` | Hip-centered torso-scale invariance & torso angle | `numpy` | **VERIFIED (PASS)** | Missing hip joints defaults to bounding box center |
| **Sequence Buffer** | `src/temporal/sequence_buffer.py` | `SequenceBuffer` | `add_snapshot()`, `get_normalized_sequence()` | Per-track sliding window of $T=30$ poses | `collections.deque` | **VERIFIED (PASS)** | Memory growth if inactive tracks are never pruned |
| **Kinematic Features** | `src/temporal/temporal_features.py` | `TemporalFeatureExtractor` | `extract()` | Physics-based $V_y, A_y$, drop severity, immobility | `numpy` | **VERIFIED (FIXED BUG-002)** | Was only calculating instantaneous 2-frame delta; now fixed to scan window peak |
| **Action Classifier** | `src/action/action_classifier.py` | `ActionClassifier` | `predict()` | Supervised 10-class action categorization | `torch`, `SpatialTemporalTransformer` | **VERIFIED (PASS)** | Fallback heuristic activates if `.pt` weights not trained |
| **Behavior Analyzer** | `src/action/behavior_classifier.py` | `BehaviorSequenceAnalyzer`| `update()`, `detect_fall_and_collapse_pattern()` | Multi-stage temporal sequence pattern recognition | `collections.deque` | **VERIFIED (PASS)** | Short sequences (<8 frames) cannot confirm fall pattern |
| **Anomaly Detection** | `src/anomaly/anomaly_score.py` | `ReconstructionAnomalyScorer` | `score()` | Unsupervised Conv1D Autoencoder reconstruction error | `torch`, `PoseSequenceAutoencoder` | **VERIFIED (PASS)** | Threshold sensitivity under unusual physical exercises |
| **Risk Engine** | `src/risk/risk_engine.py` | `RiskEngine` | `assess()`, `cleanup_inactive()` | Multi-signal linear fusion & state machine orchestration | `numpy`, `EventStateMachine` | **VERIFIED (PASS)** | Track state machines must be evicted upon person exit |
| **State Machine** | `src/risk/state_machine.py` | `EventStateMachine` | `update()` | Debounced Hysteresis FSM with autonomous recovery | `enum` | **VERIFIED (FIXED BUG-001)** | Fixed SUSPICIOUS early-return lock bug |
| **Event Logger** | `src/alerts/event_logger.py` | `EventLogger` | `log_event()` | Structured JSONL logging & anonymized snapshots | `json`, `cv2`, `pathlib` | **VERIFIED (PASS)** | Disk space management if snapshots not purged after 7 days |
| **Notification** | `src/alerts/notification.py` | `NotificationDispatcher` | `dispatch()`, `subscribe_websocket()` | Multi-channel broadcast (Console, WS, Webhook, Telegram)| `requests` | **VERIFIED (PASS)** | Slow webhook endpoint could block if not asynchronous |
| **Alert Manager** | `src/alerts/alert_manager.py` | `AlertManager` | `process_assessment()` | Rate-limiting & cooldown deduplication | `ThresholdManager` | **VERIFIED (PASS)** | Re-arming after recovery requires strict cooldown expiration |
| **Master Pipeline**| `src/pipeline/realtime_pipeline.py`| `RealtimePipeline` | `process_frame()` | End-to-end multi-stage pipeline coordinator | All above modules | **VERIFIED (PASS)** | Latency budget discipline when executing all stages sequentially |
| **Web API Server** | `src/api/server.py` | FastAPI App | `/health`, `/status`, `/stream/mjpeg`, `/ws/alerts` | Web dashboard backend and live video feed | `fastapi`, `uvicorn`, `threading`| **VERIFIED (PASS)** | Multiple simultaneous MJPEG clients increase CPU encoding load |

## 3. Findings Summary
1. **BUG-001 (Resolved)**: In `EventStateMachine`, transient glitch caused premature lock in recovery duration. Fixed by permitting instant de-escalation from `SUSPICIOUS` back to `NORMAL`.
2. **BUG-002 (Resolved)**: In `TemporalFeatureExtractor`, downward velocity was computed strictly between the last 2 frames, causing fallen recumbent persons to register $V_y = 0.0$ and allowing intentional lying to inherit a false drop severity of 0.40. Fixed by computing window peak metrics and gating drop severity by dynamic transition evidence.
