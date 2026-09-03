"""Extract video frames at target FPS for training and annotation."""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, target_fps: float = 15.0) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open {video_path}")
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / target_fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_file = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames at target frame rate")
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, default="data/frames", help="Output directory")
    parser.add_argument("--fps", type=float, default=15.0, help="Target frames per second")
    args = parser.parse_args()

    vpath = Path(args.video)
    out_dir = Path(args.output) / vpath.stem
    saved = extract_frames(vpath, out_dir, target_fps=args.fps)
    print(f"[Success] Extracted {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
