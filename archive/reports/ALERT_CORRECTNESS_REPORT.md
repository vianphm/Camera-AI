# Alert Correctness & Multi-Person Isolation Report

## 1. Executive Summary
This report validates the alert deduplication mechanisms under sustained critical emergency states and proves strict temporal isolation across multiple concurrent individuals within the camera field of view.

## 2. Phase 23: Alert Deduplication & Cooldown Verification

* **Scenario**: Person suffers a catastrophic fall and remains recumbent and immobile for **30 seconds** (300 consecutive frames at 10 FPS).
* **Observed Metrics**:
  * Sustained Emergency Frames: **300 frames**.
  * Total Alerts Emitted: **1 primary alert**.
  * Duplicate Alerts Suppressed: **299 alerts**.
  * Cooldown Window: 60.0 seconds.
  * Re-Arming Capability: After recovery (standing/walking) and expiration of cooldown, a subsequent genuine fall successfully triggered a 2nd alert.
* **Status**: **PASSED**. Zero alert spam; clinical decision support rate-limiting functions exactly as designed.

## 3. Phase 24: Multi-Person Temporal Isolation

* **Scenario**: 3 individuals monitored simultaneously in the same scene:
  * **Person 101**: Normal walking ambulation ($V_y = 0$, $\text{TorsoAngle} = 10^\circ$).
  * **Person 102**: Catastrophic fall & collapse ($V_y = 3.0$, $\text{TorsoAngle} = 82^\circ$, Immobility $= 0.85$).
  * **Person 103**: Intentional resting on sofa ($V_y = 0$, $\text{TorsoAngle} = 85^\circ$, Immobility $= 0.75$).
* **Observed Metrics**:

| Person ID | Actual Activity | Fused Risk Score | Resulting State | Dispatched Alerts | Cross-Contamination | Status |
|:---|:---|:---|:---|:---|:---|:---|
| **101** | Walking | **`0.05`** | `NORMAL` | **0** | None | **PASS** |
| **102** | Falling & Immobile | **`0.92`** | `ALERT_SENT` | **1** (To Person 102) | None | **PASS** |
| **103** | Lying on Sofa | **`0.22`** | `NORMAL` | **0** | None | **PASS** |

* **Verification**:
  * 100% of emitted alerts strictly attributed `person_id == 102`.
  * Person 101 and Person 103 experienced **zero state elevation** and **zero history contamination**.
* **Status**: **PASSED**. Multi-person isolation verified.
