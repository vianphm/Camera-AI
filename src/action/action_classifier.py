"""Supervised action classification for 10 elderly behavior categories."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from src.temporal.temporal_model import SpatialTemporalTransformer, TCNSequenceClassifier

ACTION_CLASSES: List[str] = [
    "walking",            # 0
    "standing",           # 1
    "sitting",            # 2
    "lying",              # 3
    "bending",            # 4
    "falling",            # 5
    "getting_up",         # 6
    "stumbling",          # 7
    "abnormal_movement",  # 8
    "immobile",           # 9
]

ACTION_SEVERITY: Dict[str, float] = {
    "walking": 0.0,
    "standing": 0.0,
    "sitting": 0.0,
    "getting_up": 0.0,
    "bending": 0.15,
    "lying": 0.20,
    "stumbling": 0.65,
    "abnormal_movement": 0.80,
    "falling": 0.95,
    "immobile": 0.98,
}


@dataclass
class ActionPrediction:
    """Predicted action probabilities and severity."""
    primary_action: str
    confidence: float
    probabilities: Dict[str, float]
    emergency_prob: float  # Sum of falling + immobile + abnormal_movement probabilities
    severity_score: float


class ActionClassifier:
    """Classifies temporal skeleton sequences into one of 10 human action classes."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        num_classes: int = 10,
        architecture: str = "tcn",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.num_classes = num_classes
        self.architecture = architecture.lower()

        if self.architecture == "tcn":
            self.model = TCNSequenceClassifier(input_dim=51, num_classes=num_classes)
        else:
            self.model = SpatialTemporalTransformer(input_dim=51, num_classes=num_classes)

        self.has_weights = False

        if weights_path and Path(weights_path).exists():
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                self.has_weights = True
            except Exception as e:
                print(f"[Warning] Failed to load action weights from {weights_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> ActionPrediction:
        """Predict action for a normalized sequence of shape (T, 17, 3).

        Args:
            sequence: Array of shape (T, 17, 3) where each row is (x, y, conf).

        Returns:
            ActionPrediction instance.
        """
        # If model weights exist, run forward pass
        if self.has_weights:
            with torch.no_grad():
                tensor_seq = torch.from_numpy(sequence).unsqueeze(0).to(self.device)  # (1, T, 17, 3)
                logits = self.model(tensor_seq)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        else:
            # Fallback heuristic classifier when deep weights have not been trained yet
            probs = self._heuristic_predict(sequence)

        top_idx = int(np.argmax(probs))
        primary = ACTION_CLASSES[top_idx]
        conf = float(probs[top_idx])

        prob_dict = {ACTION_CLASSES[i]: float(probs[i]) for i in range(len(ACTION_CLASSES))}
        emergency_prob = float(prob_dict.get("falling", 0.0) + prob_dict.get("immobile", 0.0) + prob_dict.get("abnormal_movement", 0.0) * 0.7)
        severity = float(ACTION_SEVERITY.get(primary, 0.0))

        return ActionPrediction(
            primary_action=primary,
            confidence=conf,
            probabilities=prob_dict,
            emergency_prob=min(1.0, emergency_prob),
            severity_score=severity,
        )

    def _heuristic_predict(self, sequence: np.ndarray) -> np.ndarray:
        """Physical heuristic classifier based on skeleton posture & vertical movement."""
        probs = np.zeros(self.num_classes, dtype=np.float32)

        # Baseline: normal actions
        probs[0] = 0.2  # walking
        probs[1] = 0.5  # standing
        probs[2] = 0.1  # sitting

        # Inspect last frame posture
        last_frame = sequence[-1]
        coords = last_frame[:, :2]
        confs = last_frame[:, 2]

        # Calculate height and aspect ratio in normalized frame
        valid = confs > 0.2
        if np.any(valid):
            pts = coords[valid]
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            h = max(0.1, max_y - min_y)
            w = max(0.1, max_x - min_x)
            ar = w / h
        else:
            ar = 0.5

        # Check downward movement of hip
        y_displacement = sequence[-1, 11, 1] - sequence[0, 11, 1] if len(sequence) > 1 else 0.0

        # Motion variance
        motion_var = float(np.mean(np.var(sequence[:, :, :2], axis=0)))

        if ar > 1.3:
            # Person is horizontal
            if motion_var < 0.005:
                # Immobile on floor
                probs = np.zeros(self.num_classes, dtype=np.float32)
                probs[9] = 0.85  # immobile
                probs[3] = 0.10  # lying
            else:
                probs = np.zeros(self.num_classes, dtype=np.float32)
                probs[3] = 0.60  # lying
                probs[9] = 0.25  # immobile
        elif y_displacement > 0.8:
            # Fast downward descent
            probs = np.zeros(self.num_classes, dtype=np.float32)
            probs[5] = 0.88  # falling
            probs[7] = 0.10  # stumbling
        elif y_displacement < -0.6 and ar < 0.8:
            probs = np.zeros(self.num_classes, dtype=np.float32)
            probs[6] = 0.80  # getting_up
        elif 0.8 < ar <= 1.3:
            if motion_var > 0.08:
                probs[7] = 0.70  # stumbling
            else:
                probs[4] = 0.60  # bending
                probs[2] = 0.30  # sitting

        # Normalize
        s = np.sum(probs)
        return probs / s if s > 0 else np.ones(self.num_classes, dtype=np.float32) / self.num_classes
