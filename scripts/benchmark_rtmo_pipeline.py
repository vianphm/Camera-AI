"""Benchmark RTMO-s End-to-End Pipeline Performance.

Measures latency breakdown across stages:
- Preprocess (Letterbox & NCHW)
- RTMO-s Inference (Single-Pass Bbox + 17 Keypoints)
- Vectorized Unscale (Postprocess)
- Multi-Object Tracking (ByteTrack)
- Biomechanical Kinematics & Debounced FSM
Evaluates on 720p (1280x720) and 1080p (1920x1080) streams with P50, P95, P99 latency and FPS.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np

from src.pose.rtmo_estimator import RTMOPoseEstimator
from src.tracking.track_manager import ByteTrackManager
from src.analytics.kinematic_engine import KinematicEngine
from src.analytics.event_fsm import EventStateMachine


def run_benchmark(
    resolution: Tuple[int, int],
    num_frames: int = 100,
    warmup_frames: int = 10,
) -> Dict[str, Any]:
    """Run stage-wise latency benchmark for RTMO-s pipeline."""
    w, h = resolution
    estimator = RTMOPoseEstimator(conf_threshold=0.30)
    tracker = ByteTrackManager(track_high_thresh=0.5, track_low_thresh=0.15)
    ke = KinematicEngine()
    fsm = EventStateMachine()

    # Create realistic test frame from bus.jpg or synthetic person
    bus_path = ".venv/Lib/site-packages/ultralytics/assets/bus.jpg"
    base_img = cv2.imread(bus_path)
    if base_img is not None:
        frame = cv2.resize(base_img, (w, h))
    else:
        frame = np.ones((h, w, 3), dtype=np.uint8) * 120

    timings = {
        "preprocess": [],
        "inference": [],
        "unscale": [],
        "tracking": [],
        "kinematics_fsm": [],
        "total": [],
    }

    # Warmup
    for _ in range(warmup_frames):
        tensor, scale_ratio, pad_w, pad_h = estimator.preprocess(frame)
        if estimator.session is not None:
            outs = estimator.session.run(None, {estimator.input_name: tensor})
            dets, items = estimator.postprocess(outs[0], outs[1], scale_ratio, pad_w, pad_h, w, h)
            tracks = tracker.update(dets, time.time())
            for t in tracks:
                ke.update(t.track_id, np.zeros((17, 3), dtype=np.float32), t.bbox, time.time())

    # Benchmark iterations
    for f_idx in range(num_frames):
        t0 = time.perf_counter()

        # 1. Preprocess
        t_pre_start = time.perf_counter()
        tensor, scale_ratio, pad_w, pad_h = estimator.preprocess(frame)
        t_pre = (time.perf_counter() - t_pre_start) * 1000.0

        # 2. RTMO-s Inference
        t_inf_start = time.perf_counter()
        outs = estimator.session.run(None, {estimator.input_name: tensor}) if estimator.session else (np.zeros((1, 1, 5)), np.zeros((1, 1, 17, 3)))
        t_inf = (time.perf_counter() - t_inf_start) * 1000.0

        # 3. Vectorized Unscale
        t_uns_start = time.perf_counter()
        dets, items = estimator.postprocess(outs[0], outs[1], scale_ratio, pad_w, pad_h, w, h)
        t_uns = (time.perf_counter() - t_uns_start) * 1000.0

        # 4. ByteTrack
        t_trk_start = time.perf_counter()
        cur_time = f_idx / 15.0
        tracks = tracker.update(dets, cur_time)
        t_trk = (time.perf_counter() - t_trk_start) * 1000.0

        # 5. Kinematics & FSM
        t_kin_start = time.perf_counter()
        matched = estimator.match_tracks_to_keypoints(tracks, items, iou_threshold=0.70)
        for trk in tracks:
            kp = matched.get(trk.track_id, np.zeros((17, 3), dtype=np.float32))
            feat = ke.update(trk.track_id, kp, trk.bbox, cur_time)
            fsm.update(trk.track_id, feat, kp, cur_time)
        t_kin = (time.perf_counter() - t_kin_start) * 1000.0

        t_tot = (time.perf_counter() - t0) * 1000.0

        timings["preprocess"].append(t_pre)
        timings["inference"].append(t_inf)
        timings["unscale"].append(t_uns)
        timings["tracking"].append(t_trk)
        timings["kinematics_fsm"].append(t_kin)
        timings["total"].append(t_tot)

    summary = {
        "resolution": f"{w}x{h}",
        "active_provider": estimator.active_provider,
        "mean_preprocess_ms": float(np.mean(timings["preprocess"])),
        "mean_inference_ms": float(np.mean(timings["inference"])),
        "mean_unscale_ms": float(np.mean(timings["unscale"])),
        "mean_tracking_ms": float(np.mean(timings["tracking"])),
        "mean_kinematics_ms": float(np.mean(timings["kinematics_fsm"])),
        "mean_total_ms": float(np.mean(timings["total"])),
        "p50_total_ms": float(np.percentile(timings["total"], 50)),
        "p95_total_ms": float(np.percentile(timings["total"], 95)),
        "p99_total_ms": float(np.percentile(timings["total"], 99)),
        "fps": float(1000.0 / max(np.mean(timings["total"]), 1e-4)),
    }
    return summary


def main() -> None:
    print("=" * 80)
    print(" [BENCHMARK] RTMO-s End-to-End Pipeline Performance Evaluation")
    print("=" * 80)

    res_720p = run_benchmark((1280, 720), num_frames=50)
    res_1080p = run_benchmark((1920, 1080), num_frames=50)

    print("\n### BẢNG KẾT QUẢ ĐO HIỆU NĂNG LATENCY & THROUGHPUT (FPS)\n")
    print("| Độ phân giải | Provider | Tiền xử lý (ms) | RTMO-s Inference (ms) | Unscale (ms) | ByteTrack (ms) | Kinematics & FSM (ms) | Tổng trễ (ms) | P95 (ms) | FPS Trung bình |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in [res_720p, res_1080p]:
        print(
            f"| {r['resolution']} "
            f"| {r['active_provider']} "
            f"| {r['mean_preprocess_ms']:.2f} "
            f"| {r['mean_inference_ms']:.2f} "
            f"| {r['mean_unscale_ms']:.2f} "
            f"| {r['mean_tracking_ms']:.2f} "
            f"| {r['mean_kinematics_ms']:.2f} "
            f"| {r['mean_total_ms']:.2f} "
            f"| {r['p95_total_ms']:.2f} "
            f"| **{r['fps']:.1f} FPS** |"
        )
    print("\n" + "=" * 80)


if __name__ == "__main__":
    from typing import Any
    main()
