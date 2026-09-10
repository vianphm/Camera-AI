"""Ablation Study: Autoencoder ON vs OFF Comparison.

Compares:
1. Autoencoder OFF: Kinematic Biomechanics + FSM Debounce
2. Autoencoder ON : Kinematic Biomechanics + Deep Autoencoder Anomaly Scoring (w_anom = 0.25)
Evaluates TPR, FPR, Mean Risk Scores, and Latency overhead.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Tuple
import numpy as np
from src.analytics.kinematic_engine import KinematicEngine
from src.analytics.event_fsm import EventStateMachine, MonitorState
from src.anomaly.anomaly_score import ReconstructionAnomalyScorer
from tests.scenarios.test_rtmo_comprehensive_suite import (
    make_upright_person,
    make_horizontal_person,
)


def run_scenario_with_ae(
    behavior_type: str,
    enable_ae: bool,
    scorer: ReconstructionAnomalyScorer,
    fps: float = 15.0,
) -> Tuple[bool, float, float]:
    """Run a scenario and return (alert_triggered, max_risk_score, avg_latency_ms)."""
    ke = KinematicEngine()
    fsm = EventStateMachine(immobility_confirm_duration=3.0)
    alerts = []
    max_risk = 0.0
    latencies = []

    seq_buffer = []

    total_frames = 75 if "fall" in behavior_type else 60

    for i in range(total_frames):
        t = i / fps
        t0 = time.perf_counter()

        if behavior_type == "normal_walking":
            kpts, bbox = make_upright_person(cx=100.0 + i * 3.0, cy=240.0, tilt_deg=5.0)
        elif behavior_type == "sitting":
            kpts, bbox = make_upright_person(cx=320.0, cy=300.0, tilt_deg=10.0)
        elif behavior_type == "bending":
            prog = min(1.0, i / 20.0) if i < 25 else max(0.0, (50 - i) / 25.0)
            kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 20.0 * prog, tilt_deg=5.0 + 60.0 * prog)
        elif behavior_type == "sofa_rest":
            if i < 20:
                kpts, bbox = make_upright_person(cx=320.0, cy=240.0, tilt_deg=10.0)
            else:
                kpts, bbox = make_horizontal_person(cx=350.0, ground_y=350.0)
        elif behavior_type == "true_fall":
            if i < 15:
                kpts, bbox = make_upright_person(cx=320.0, cy=240.0)
            elif i <= 20:
                prog = (i - 15) / 5.0
                kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
            else:
                kpts, bbox = make_horizontal_person(cx=320.0, ground_y=420.0)
        elif behavior_type == "chair_fall":
            if i < 15:
                kpts, bbox = make_upright_person(cx=320.0, cy=300.0, tilt_deg=10.0)
            elif i <= 20:
                prog = (i - 15) / 5.0
                kpts, bbox = make_upright_person(cx=320.0, cy=300.0 + 120.0 * prog, tilt_deg=10.0 + 75.0 * prog)
            else:
                kpts, bbox = make_horizontal_person(cx=350.0, ground_y=420.0)
        elif behavior_type == "near_fall_recovered":
            if i < 15:
                kpts, bbox = make_upright_person(cx=320.0, cy=240.0)
            elif i <= 20:
                prog = (i - 15) / 5.0
                kpts, bbox = make_upright_person(cx=320.0, cy=240.0 + 180.0 * prog, tilt_deg=5.0 + 80.0 * prog)
            elif i < 35:
                kpts, bbox = make_horizontal_person(cx=320.0, ground_y=420.0)
            else:
                prog = min(1.0, (i - 35) / 15.0)
                kpts, bbox = make_upright_person(cx=320.0, cy=420.0 - 180.0 * prog, tilt_deg=85.0 - 75.0 * prog)
        else:
            kpts, bbox = make_upright_person()

        feat = ke.update(1, kpts, bbox, t)

        # Buffer for autoencoder
        seq_buffer.append(kpts)
        if len(seq_buffer) > 30:
            seq_buffer.pop(0)

        # If Autoencoder is enabled, fuse anomaly score into risk
        effective_risk = feat.risk_score
        if enable_ae and len(seq_buffer) == 30:
            seq_np = np.array(seq_buffer, dtype=np.float32)
            anom_res = scorer.score(seq_np)
            # Weighted fusion: 0.75 Kinematics + 0.25 Anomaly
            if feat.is_impact or feat.is_horizontal:
                effective_risk = 0.75 * feat.risk_score + 0.25 * anom_res.anomaly_score

        max_risk = max(max_risk, effective_risk)

        # Update FSM with effective risk
        state, alert = fsm.update(1, feat, kpts, t)
        if alert:
            alerts.append(alert)

        latencies.append((time.perf_counter() - t0) * 1000.0)

    alert_triggered = len(alerts) > 0
    return alert_triggered, float(max_risk), float(np.mean(latencies))


def main() -> None:
    print("=" * 75)
    print(" [ABLATION STUDY] Autoencoder Anomaly Scoring: ON vs OFF Evaluation")
    print("=" * 75)

    scorer = ReconstructionAnomalyScorer(
        weights_path="models/checkpoints/best_pose_autoencoder.pt",
        device="cpu",
    )

    scenarios = [
        ("normal_walking", "Normal Walking (ADL)", False),
        ("sitting", "Sitting still (ADL)", False),
        ("bending", "Bending down to pick (ADL)", False),
        ("sofa_rest", "Resting on sofa (Intentional Lying)", False),
        ("near_fall_recovered", "Near Fall (Self-Recovered)", False),
        ("true_fall", "Standing Forward Fall (True Fall)", True),
        ("chair_fall", "Chair Fall to Floor (True Fall)", True),
    ]

    for mode_name, enable_ae in [("Autoencoder OFF (Pure Kinematics + FSM)", False), ("Autoencoder ON (Kinematics + Pose AE Fusion)", True)]:
        print(f"\n--- Chế độ: {mode_name} ---")
        tp, fn, tn, fp = 0, 0, 0, 0
        total_lat = []

        print(f"{'Kịch bản':<35} | {'Ground Truth':<12} | {'Cảnh báo?':<10} | {'Max Risk':<10} | {'Độ trễ (ms)':<10}")
        print("-" * 85)

        for s_key, s_name, is_fall in scenarios:
            alert, m_risk, lat = run_scenario_with_ae(s_key, enable_ae, scorer)
            total_lat.append(lat)

            if is_fall:
                if alert:
                    tp += 1
                    status = "TP (Đúng)"
                else:
                    fn += 1
                    status = "FN (Bỏ sót)"
            else:
                if alert:
                    fp += 1
                    status = "FP (Báo giả)"
                else:
                    tn += 1
                    status = "TN (Đúng)"

            print(f"{s_name:<35} | {'Ngã' if is_fall else 'Bình thường':<12} | {'CÓ' if alert else 'KHÔNG':<10} | {m_risk:<10.3f} | {lat:<10.2f}")

        tpr = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        avg_lat = np.mean(total_lat)

        print(f"\n  -> Kết quả tổng kết: TPR = {tpr:.1f}% | FPR = {fpr:.1f}% | Độ trễ TB: {avg_lat:.2f} ms/frame")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
