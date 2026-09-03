"""Pose estimation interface and YOLOv8-Pose implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

from src.tracking.tracker import Track
from src.pose.keypoints import normalize_keypoints, compute_torso_angle

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@dataclass
class PoseResult:
    """Represents the skeletal pose of a tracked individual."""
    track_id: int
    keypoints: np.ndarray             # Raw image coordinates: shape (17, 3) -> [x, y, conf]
    normalized_keypoints: np.ndarray  # Root-centered & scale-invariant: shape (17, 3)
    torso_angle: float               # Angle relative to vertical in degrees [0..90]
    confidence: float


class PoseEstimator(ABC):
    """Abstract interface for human pose estimators."""

    @abstractmethod
    def estimate(self, frame: np.ndarray, tracks: List[Track]) -> List[PoseResult]:
        """Estimate human poses for active tracks in the given frame.

        Args:
            frame: Input image (H, W, 3) in BGR.
            tracks: List of confirmed person tracks.

        Returns:
            List of PoseResult objects matching tracks.
        """
        pass


class YOLOv8PoseEstimator(PoseEstimator):
    """YOLOv8-Pose implementation extracting 17 COCO skeletal keypoints."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8s-pose.pt",
        conf_threshold: float = 0.35,
        device: str = "cuda",
        half: bool = True,
        img_size: int = 640,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.half = half and (device != "cpu")
        self.img_size = img_size
        self.model = None

        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"[Warning] Could not initialize YOLO-Pose from {model_path}: {e}")

    def estimate(self, frame: np.ndarray, tracks: List[Track]) -> List[PoseResult]:
        if not tracks or self.model is None or not ULTRALYTICS_AVAILABLE:
            return []

        predict_kwargs = {
            "source": frame,
            "conf": self.conf_threshold,
            "device": self.device,
            "imgsz": self.img_size,
            "verbose": False,
        }
        if self.half and self.device != "cpu":
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        if not results or results[0].keypoints is None:
            return []

        res = results[0]
        detected_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.array([])
        keypoints_data = res.keypoints.data.cpu().numpy()  # shape: (N, 17, 3)

        if len(detected_boxes) == 0 or len(keypoints_data) == 0:
            return []

        pose_results: List[PoseResult] = []

        # Match each Track with the closest detected pose bbox using center distance or IoU
        for track in tracks:
            t_cx = (track.bbox[0] + track.bbox[2]) / 2.0
            t_cy = (track.bbox[1] + track.bbox[3]) / 2.0

            best_idx = -1
            min_dist = float("inf")

            for i, d_box in enumerate(detected_boxes):
                d_cx = (d_box[0] + d_box[2]) / 2.0
                d_cy = (d_box[1] + d_box[3]) / 2.0
                dist = np.hypot(t_cx - d_cx, t_cy - d_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            # Threshold distance to prevent matching an unrelated person (e.g. < max(track.width, track.height))
            max_dist_allowed = max(track.width, track.height) * 0.8
            if best_idx >= 0 and min_dist <= max_dist_allowed:
                raw_kp = keypoints_data[best_idx]  # (17, 3)
                norm_kp = normalize_keypoints(raw_kp, bbox_fallback=track.bbox)
                torso_angle = compute_torso_angle(raw_kp)
                mean_conf = float(np.mean(raw_kp[:, 2]))

                pose_results.append(
                    PoseResult(
                        track_id=track.track_id,
                        keypoints=raw_kp,
                        normalized_keypoints=norm_kp,
                        torso_angle=torso_angle,
                        confidence=mean_conf,
                    )
                )

        return pose_results
