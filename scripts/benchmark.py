"""End-to-End Performance and Latency Benchmark Script."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.realtime_pipeline import RealtimePipeline


def run_benchmark(num_frames: int = 150, warmup: int = 20, device: str = "cuda") -> None:
    print("\n" + "=" * 65)
    print("ELDERLY AI MONITOR: END-TO-END PIPELINE BENCHMARK")
    print("=" * 65)

    print(f"[Info] Initializing pipeline on device: {device}...")
    pipeline = RealtimePipeline()

    # Create dummy 720p frame (1280x720x3)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add a synthetic white rectangle resembling a person to trigger detection/pose
    dummy_frame[150:550, 500:700] = 200

    print(f"[Info] Running {warmup} warmup frames...")
    for i in range(warmup):
        pipeline.process_frame(dummy_frame, timestamp=time.time())

    print(f"[Info] Running benchmark across {num_frames} frames...")
    t0 = time.perf_counter()

    for i in range(num_frames):
        ts = time.time()
        pipeline.process_frame(dummy_frame, timestamp=ts)

    total_time = time.perf_counter() - t0
    avg_fps = num_frames / total_time
    latency_ms = (total_time / num_frames) * 1000.0

    summary = pipeline.profiler.get_summary()
    telemetry = pipeline.resource_monitor.get_telemetry()

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Frames Processed: {num_frames}")
    print(f"Total Elapsed Time:     {total_time:.2f} s")
    print(f"Overall Pipeline FPS:   {avg_fps:.1f} FPS")
    print(f"Average Latency/Frame:  {latency_ms:.2f} ms\n")

    print("Stage Latency Breakdown (Average):")
    print("-" * 65)
    for k, v in summary.items():
        if k.endswith("_ms"):
            print(f"  • {k:<25}: {v:>6.2f} ms")

    print("\nHardware Telemetry:")
    print("-" * 65)
    print(f"  • CPU Utilization     : {telemetry.get('cpu_percent', 0.0):.1f}%")
    print(f"  • System RAM Used     : {telemetry.get('ram_used_gb', 0.0)} GB ({telemetry.get('system_ram_percent', 0.0)}%)")
    if telemetry.get("gpu_available"):
        print(f"  • GPU Device          : {telemetry.get('gpu_device_name', 'NVIDIA GPU')}")
        print(f"  • GPU VRAM Allocated  : {telemetry.get('gpu_vram_allocated_mb', 0.0)} MB")
        print(f"  • GPU VRAM Reserved   : {telemetry.get('gpu_vram_reserved_mb', 0.0)} MB")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Benchmark Runner")
    parser.add_argument("--frames", type=int, default=150, help="Number of benchmark frames")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup frames")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()

    run_benchmark(num_frames=args.frames, warmup=args.warmup, device=args.device)
