# Ground Truth Audit

## 1. Available Evaluation Data
* **Repository Scan Results**:
  * `datasets/`: Contains directory structure (`abnormal/`, `emergency/`, `fall/`, `lying/`, `normal/`, `sitting/`, `standing/`, `walking/`) with `.gitkeep` files only. No raw video files or public benchmark datasets (e.g., UR Fall Detection, Le2i, UP-Fall, NTU RGB+D) are currently mounted in the repository.
  * `data/annotations/`: Contains `.gitkeep` only. No pre-annotated bounding boxes or frame-level temporal labels found.
  * `data/videos/`: Contains `.gitkeep` only.
  * `data/splits/`: Contains `.gitkeep` only.
* **Synthetic & Programmatic Test Suites**:
  * `tests/unit/`: Programmatic skeletal sequence generators with precisely controlled timestamps, joint trajectories, keypoint drop rates, and synthetic kinematic profiles.

## 2. Annotation Schema
* The architecture defines the standard metadata schema in `DATASET.md`:
  * `video_id`: string
  * `person_id`: integer
  * `start_timestamp`: float (seconds)
  * `end_timestamp`: float (seconds)
  * `event_label`: string (e.g., `fall`, `adl`, `stumble`)
  * `behavior_label`: string (1 of 10 canonical action classes)
  * `event_onset_timestamp`: float (seconds)
  * `event_confirmation_timestamp`: float (seconds)
* Currently, no external annotated JSON/CSV files conforming to this schema exist in the repository.

## 3. Verified Labels
* **In-Repository Labeled Data**: 0 external physical videos.
* **Unit & Integration Test Ground Truth**: Programmatically constructed test fixtures with deterministic ground truth labels:
  * Scenario A1: Fall-related sequence (`walking -> stumble -> fall -> lying -> immobile`)
  * Scenario A2: Normal lying sequence (`standing -> sitting -> lying -> lying`)
  * Scenario B: Rapid downward motion leading to floor impact (`fall`)
  * Scenario C: Gentle descent to bed/sofa (`normal_lying`)
  * Scenario D: Fall followed by self-recovery (`recovery`)
  * Scenario E: Single-frame noise injection (`glitch`)

## 4. Metrics That Can Be Computed
* **Unit & Component Behavioral Correctness**: State machine transition correctness, debounce hysteresis pass/fail, alert deduplication pass/fail, mathematical invariant checks (aspect ratio, $V_y$, $A_y$, cosine torso angle).
* **Controlled Scenario Pass/Fail Rates**: Percentage of deterministic scenario test cases passing behavioral contracts.
* **Engineering Performance & Latency**: `capture_latency_ms`, `detection_latency_ms`, `tracking_latency_ms`, `pose_latency_ms`, `temporal_latency_ms`, `risk_latency_ms`, `alert_latency_ms`, `frame_age_ms`, `queue_depth`, RAM/VRAM usage.

## 5. Metrics NOT AVAILABLE
* **Metric**: Real-world Action Precision / Recall / F1 / Specificity / PR-AUC / ROC-AUC on public benchmark datasets.
  * **Reason**: No verified real-world benchmark dataset (UR Fall, Le2i, UP-Fall) is mounted in `datasets/`.
* **Metric**: Real-world False Alarm Rate per 24 hours on live clinical footage.
  * **Reason**: Continuous 24-hour physical elderly room recording is not available in local test environment.
* **Metric**: Physical Event Onset to Alert Latency on unannotated live webcam feed.
  * **Reason**: Live human camera action does not possess frame-accurate microsecond ground-truth onset markers unless manually stamped in controlled scenario tests.

## 6. Data Leakage Risk
* **Status**: PASS (Low Risk)
* **Evidence**: Current synthetic training scripts (`training/train_action.py`, `training/train_anomaly.py`) generate disjoint train/val splits by subject ID / sequence seed. No contamination between training and unit test fixtures.

## 7. Evaluation Validity
* **Status**: PARTIALLY VALID
* **Assessment**: All algorithmic, kinematic, temporal logic, latency, throughput, memory stability, and resilience tests are FULLY VALID and verifiable. Real-world dataset benchmark accuracy metrics are deferred until a verified physical dataset is mounted.
