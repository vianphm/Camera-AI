"""Batch video processing script for offline evaluation and benchmark analysis."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.realtime_pipeline import RealtimePipeline


def process_video(pipeline: RealtimePipeline, video_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run pipeline on a single video file and record event logs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "status": "error_cannot_open"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detected_alerts: List[Dict[str, Any]] = []
    frame_idx = 0
    t0 = time.time()

    with tqdm(total=total_frames, desc=video_path.name, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps

            res = pipeline.process_frame(frame, timestamp)
            for alert in res.alerts:
                detected_alerts.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "person_id": alert.person_id,
                    "risk_score": alert.risk_score,
                    "evidence": alert.evidence,
                })
            pbar.update(1)

    cap.release()
    elapsed = time.time() - t0
    avg_fps = frame_idx / max(1e-4, elapsed)

    report = {
        "video_name": video_path.name,
        "total_frames": frame_idx,
        "duration_seconds": frame_idx / fps,
        "processing_time_seconds": round(elapsed, 2),
        "average_fps": round(avg_fps, 1),
        "total_alerts": len(detected_alerts),
        "alerts": detected_alerts,
    }

    report_path = output_dir / f"{video_path.stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch video evaluation pipeline")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing MP4/AVI videos")
    parser.add_argument("--output-dir", type=str, default="results/batch_reports", help="Directory to save JSON reports")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
    if not videos:
        print(f"[Warning] No video files found in {input_path}")
        return

    print(f"[Info] Found {len(videos)} videos to evaluate.")
    pipeline = RealtimePipeline()

    reports = []
    for vid in videos:
        rep = process_video(pipeline, vid, output_path)
        reports.append(rep)

    summary_file = output_path / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"[Info] Batch processing complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
