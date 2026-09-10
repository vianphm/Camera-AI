# Test Execution Evidence Log

## 1. Test Environment Baseline
* **Platform**: Windows 11 (AMD64)
* **Python Runtime**: Python 3.11.9 (`d:\Project-monitoring-model\.venv\Scripts\python.exe`)
* **Host CPU**: 12th Gen Intel Core i5-12450HX (8 Cores, 12 Threads)
* **Host RAM**: 16 GB DDR4/DDR5
* **Host GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, Driver 577.05)
* **PyTorch Version**: 2.13.0+cpu
* **OpenCV Version**: 4.10.0
* **Ultralytics Version**: 8.4.19

---

## 2. Comprehensive Test Execution Records

| Test Target / Suite | Execution Command | Input / Scenario Description | Expected Outcome | Actual Outcome | Status | Duration |
|:---|:---|:---|:---|:---|:---:|:---:|
| **Unit Test Suite (31 tests)** | `pytest tests/ -v` | Full test suite across camera, detection, tracking, pose, temporal, anomaly, risk, state machine, alerts | 100% passing tests | 31/31 passed | **PASS** | 5.96s |
| **Temporal Reasoning Suite** | `pytest tests/unit/test_temporal_reasoning.py -v` | Scenarios A1 (fall) vs A2 (lying), B (timeline), C (sofa), D (recovery), E (glitch) | Distinguish history; recover autonomously; suppress noise | 5/5 passed | **PASS** | 1.46s |
| **False Positive Suite (14 ADL)** | `pytest tests/scenarios/false_positive/ -v` | Intentional lying, sleeping, fast sitting, bending, exercises, stretching, camera shake | Zero emergency alarms emitted | 1/1 passed (14 scenarios) | **PASS** | 1.67s |
| **Realtime Freshness & Backpressure** | `pytest tests/unit/test_realtime_freshness.py -v` | Camera 30 FPS vs Slow AI Consumer 16 FPS | Queue $\le 1$, $P_{95}\text{ age} < 100\text{ms}$ | Queue $\le 1$, $P_{95} = 68.5\text{ms}$ | **PASS** | 1.87s |
| **Alert Deduplication & Isolation** | `pytest tests/unit/test_alert_dedup_and_isolation.py -v` | Sustained fall 300 frames (30s); 3 people in room | 1 alert only; no cross-person contamination | 3/3 passed | **PASS** | 1.91s |
| **End-to-End Timeline Benchmark** | `pytest tests/unit/test_event_timeline.py -v` | Microsecond clock $T_0 \rightarrow T_7$ event propagation | Total $T_0 \rightarrow T_7 \le 2.5\text{s}$ | Total $= 1.32\text{s}$ | **PASS** | 1.72s |
| **Pipeline Latency Benchmark** | `python scripts/benchmark.py --frames 30` | 30 full inference frames on pipeline | Average frame latency profile | $P_{50} = 118.8\text{ms}$, FPS $= 8.1$ | **PASS** | 3.69s |
| **Master Entrypoint Verification** | `python main.py --help` | Verify CLI arguments, multi-cam, unblurred display | Clean exit 0, arguments parsed | Verified exit code 0 | **PASS** | 0.82s |

---

## 3. Bug Fix Verification History

### BUG-001: Premature Recovery Lock in `EventStateMachine`
* **Severity**: MEDIUM
* **Reproduction**: A single transient glitch frame transitioned the state from `NORMAL` to `SUSPICIOUS`, which then required waiting for `min_recovery_duration` (1.0s) before returning to `NORMAL`.
* **Root Cause**: Early return in recovery check intercepted de-escalation for low-severity `SUSPICIOUS` states.
* **Fix**: Implemented instant drop back to `NORMAL` if person was only in `SUSPICIOUS` when normal low-risk actions resume.
* **Regression Test**: `tests/unit/test_temporal_reasoning.py::test_scenario_e_single_frame_noise_tolerance` (PASSED).

### BUG-002: Instantaneous 2-Frame Velocity Blindspot in `TemporalFeatureExtractor`
* **Severity**: HIGH
* **Reproduction**: When a person completed a fall and was currently lying on the floor, `vertical_velocity` was $0.0$ because it was computed solely on the last 2 frames. Simultaneously, an intentionally resting person received a false drop severity score of $0.40$ due to horizontal posture alone.
* **Root Cause**: Lack of window peak velocity scanning and lack of dynamic transition gating.
* **Fix**: Scanned peak downward velocity and peak downward acceleration across the entire temporal window; gated `drop_severity_score` by transition evidence.
* **Regression Test**: `tests/unit/test_temporal_reasoning.py::test_scenario_a_same_current_frame_different_history` (PASSED).
