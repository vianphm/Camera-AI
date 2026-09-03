# QA Function Inventory

## 1. Inventory Specifications
This document provides an exhaustive inventory of the critical functions in the codebase, detailing their signatures, inputs, outputs, side-effects, and verification test status.

## 2. Function Matrix

### Ingestion Layer (`src/camera/`)
* **`CameraStream.start()`**
  * File: `src/camera/stream.py`
  * Class: `CameraStream`
  * Input: None
  * Output: `bool` (True if capture thread successfully launched)
  * Side effects: Starts background daemon thread `CameraWorker`, spawns unbounded capture loop.
  * Expected behavior: Idempotent (returns True immediately if already running).
  * Test status: **VERIFIED (`tests/test_camera.py`, `tests/unit/test_realtime_freshness.py`)**

* **`CameraStream.read(timeout=1.0)`**
  * File: `src/camera/stream.py`
  * Class: `CameraStream`
  * Input: `timeout: float`
  * Output: `Optional[FramePacket]`
  * Side effects: Dequeues freshest frame from thread-safe bounded queue.
  * Expected behavior: Returns newest `FramePacket` with accurate `timestamp` and monotonically increasing `frame_idx`. Returns None on timeout.
  * Test status: **VERIFIED (`tests/test_camera.py`, `tests/unit/test_realtime_freshness.py`)**

* **`MultiCameraManager.read_all(timeout=0.5)`**
  * File: `src/camera/multi_camera.py`
  * Class: `MultiCameraManager`
  * Input: `timeout: float`
  * Output: `Dict[str, FramePacket]`
  * Side effects: Queries all active camera streams concurrently.
  * Expected behavior: Returns dictionary mapping `camera_id` to its latest packet.
  * Test status: **VERIFIED (`main.py --source multi`)**

---

### Detection & Tracking Layer (`src/detection/` & `src/tracking/`)
* **`YOLOv8PersonDetector.detect(frame)`**
  * File: `src/detection/person_detector.py`
  * Class: `YOLOv8PersonDetector`
  * Input: `frame: np.ndarray` (BGR image)
  * Output: `List[Detection]`
  * Side effects: Executes GPU/CPU neural forward pass.
  * Expected behavior: Returns bounding boxes and confidences strictly for COCO class 0 (person).
  * Test status: **VERIFIED (`tests/test_detection.py`, `tests/test_pipeline.py`)**

* **`ByteTrackManager.update(detections, frame)`**
  * File: `src/tracking/track_manager.py`
  * Class: `ByteTrackManager`
  * Input: `detections: List[Detection]`, `frame: np.ndarray`
  * Output: `List[Track]`
  * Side effects: Updates Kalman Filter states, updates trajectory history, marks lost tracks.
  * Expected behavior: Preserves track identities across frames; recovers low-confidence boxes via 2nd-stage association.
  * Test status: **VERIFIED (`tests/test_tracking.py`)**

---

### Pose Estimation & Normalization (`src/pose/`)
* **`YOLOv8PoseEstimator.estimate(frame, detections)`**
  * File: `src/pose/pose_estimator.py`
  * Class: `YOLOv8PoseEstimator`
  * Input: `frame: np.ndarray`, `detections: List[Detection]`
  * Output: `List[PoseResult]`
  * Side effects: Executes YOLO-Pose inference.
  * Expected behavior: Extracts 17 COCO keypoints `(x, y, conf)`, normalizes them (Mid-Hip root + Torso scale), and computes torso angle in degrees $[0^\circ, 90^\circ]$.
  * Test status: **VERIFIED (`tests/test_pose.py`)**

* **`normalize_keypoints(keypoints)`**
  * File: `src/pose/keypoints.py`
  * Function: Standalone
  * Input: `keypoints: np.ndarray` shape `(17, 3)`
  * Output: `np.ndarray` shape `(17, 3)`
  * Side effects: None (pure function).
  * Expected behavior: Translates keypoints so Mid-Hip is at origin `(0, 0)`; scales by torso distance (neck-to-hip) to achieve scale and distance invariance.
  * Test status: **VERIFIED (`tests/test_pose.py`)**

---

### Temporal Reasoning & Kinematics (`src/temporal/`)
* **`SequenceBuffer.add_snapshot(track_id, snapshot)`**
  * File: `src/temporal/sequence_buffer.py`
  * Class: `SequenceBuffer`
  * Input: `track_id: int`, `snapshot: TrackSnapshot`
  * Output: `None`
  * Side effects: Appends snapshot to per-track `deque(maxlen=window_size)`.
  * Expected behavior: Bounded memory usage per track, automatically drops oldest snapshot beyond window size ($T=30$).
  * Test status: **VERIFIED (`tests/test_temporal.py`, `tests/unit/test_temporal_reasoning.py`)**

* **`TemporalFeatureExtractor.extract(snapshots)`**
  * File: `src/temporal/temporal_features.py`
  * Class: `TemporalFeatureExtractor`
  * Input: `snapshots: List[TrackSnapshot]`
  * Output: `KinematicFeatures`
  * Side effects: None (pure computation).
  * Expected behavior: Calculates peak downward velocity, peak acceleration, current aspect ratio, aspect ratio change, current torso angle, angular velocity, immobility index, and gated drop severity score.
  * Test status: **VERIFIED (`tests/test_temporal.py`, `tests/unit/test_temporal_reasoning.py`)**

---

### Action & Anomaly Analysis (`src/action/` & `src/anomaly/`)
* **`ActionClassifier.predict(sequence)`**
  * File: `src/action/action_classifier.py`
  * Class: `ActionClassifier`
  * Input: `sequence: np.ndarray` shape `(T, 17, 3)`
  * Output: `ActionPrediction`
  * Side effects: PyTorch forward pass.
  * Expected behavior: Outputs predicted action class, confidence, class probability dictionary, emergency probability, and severity score.
  * Test status: **VERIFIED (`tests/test_temporal.py`)**

* **`ReconstructionAnomalyScorer.score(sequence)`**
  * File: `src/anomaly/anomaly_score.py`
  * Class: `ReconstructionAnomalyScorer`
  * Input: `sequence: np.ndarray` shape `(T, 17, 3)`
  * Output: `AnomalyResult`
  * Side effects: PyTorch Autoencoder forward pass.
  * Expected behavior: Computes sequence reconstruction loss and maps it through a sigmoid transfer function to normalized anomaly score $[0.0, 1.0]$.
  * Test status: **VERIFIED (`tests/test_anomaly.py`)**

---

### Risk Engine & State Machine (`src/risk/`)
* **`RiskEngine.assess(track_id, action_pred, anomaly_res, kinematics, timestamp)`**
  * File: `src/risk/risk_engine.py`
  * Class: `RiskEngine`
  * Input: `track_id: int`, prediction results, kinematics, current timestamp
  * Output: `RiskAssessment`
  * Side effects: Instantiates or updates per-person `EventStateMachine`.
  * Expected behavior: Fuses multi-modal signals into $[0.0, 1.0]$ score and queries FSM for state transition and alert dispatch trigger.
  * Test status: **VERIFIED (`tests/test_risk_engine.py`, `tests/unit/test_alert_dedup_and_isolation.py`)**

* **`EventStateMachine.update(risk_score, action, timestamp)`**
  * File: `src/risk/state_machine.py`
  * Class: `EventStateMachine`
  * Input: `risk_score: float`, `action: str`, `timestamp: float`
  * Output: `Tuple[MonitorState, bool]`
  * Side effects: Updates internal state, state entry time, recovery timers.
  * Expected behavior: Enforces hysteresis debounce durations; permits autonomous recovery; prevents false positive state latching.
  * Test status: **VERIFIED (`tests/test_state_machine.py`, `tests/unit/test_temporal_reasoning.py`)**

---

### Alerts & Notifications (`src/alerts/`)
* **`AlertManager.process_assessment(assessment, frame, timestamp)`**
  * File: `src/alerts/alert_manager.py`
  * Class: `AlertManager`
  * Input: `assessment: RiskAssessment`, optional BGR frame, timestamp
  * Output: `Optional[AlertEvent]`
  * Side effects: Logs event to JSONL, saves anonymized snapshot, dispatches over WebSocket / Webhook / Telegram.
  * Expected behavior: Suppresses duplicate alerts within the cooldown window (e.g. 60 seconds).
  * Test status: **VERIFIED (`tests/unit/test_alert_dedup_and_isolation.py`)**
