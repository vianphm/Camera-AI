"""Extract and serialize normalized pose sequences to .npz files."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.person_detector import YOLOv8PersonDetector
from src.tracking.track_manager import ByteTrackManager
from src.pose.pose_estimator import YOLOv8PoseEstimator


def generate_pose_sequences(video_path: Path, output_file: Path, window_size: int = 30) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return 0

    detector = YOLOv8PersonDetector()
    tracker = ByteTrackManager()
    pose_estimator = YOLOv8PoseEstimator()

    track_buffers = {}
    extracted_sequences = []

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps

        dets = detector.detect(frame)
        active_tracks = tracker.update(dets, ts)
        poses = pose_estimator.estimate(frame, active_tracks)
        pose_dict = {p.track_id: p for p in poses}

        for trk in active_tracks:
            tid = trk.track_id
            if tid in pose_dict:
                if tid not in track_buffers:
                    track_buffers[tid] = []
                track_buffers[tid].append(pose_dict[tid].normalized_keypoints)

                if len(track_buffers[tid]) == window_size:
                    seq_arr = np.array(track_buffers[tid], dtype=np.float32)  # (30, 17, 3)
                    extracted_sequences.append(seq_arr)
                    # Slide window by stride (e.g. 5 frames)
                    track_buffers[tid] = track_buffers[tid][5:]

    cap.release()

    if extracted_sequences:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, sequences=np.array(extracted_sequences))
        print(f"[Success] Saved {len(extracted_sequences)} sequences of shape ({window_size}, 17, 3) to {output_file}")
        return len(extracted_sequences)
    else:
        print("[Warning] No full pose sequences extracted.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose sequences to .npz")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="data/poses/sequences.npz", help="Output .npz file")
    parser.add_argument("--window", type=int, default=30, help="Window size in frames")
    args = parser.parse_args()

    generate_pose_sequences(Path(args.video), Path(args.output), window_size=args.window)


if __name__ == "__main__":
    main()
