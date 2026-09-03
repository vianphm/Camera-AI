"""Deep autoencoder and reconstruction error scoring for unsupervised anomaly detection."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
from src.anomaly.anomaly_detector import AnomalyDetector, AnomalyResult


class PoseSequenceAutoencoder(nn.Module):
    """Conv1D temporal autoencoder for skeletal sequence reconstruction."""

    def __init__(self, input_dim: int = 51, latent_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: compresses (B, 51, T) -> (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T -> T/2

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # T/2 -> 4

            nn.Flatten(),
            nn.Linear(128 * 4, latent_dim),
            nn.ReLU(),
        )

        # Decoder: reconstructs (B, latent_dim) -> (B, 51, T)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),   # 8 -> 16
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, 17, 3) or (B, T, 51) or (B, 51, T).

        Returns:
            Reconstructed sequence tensor matching input shape.
        """
        orig_shape = x.shape
        if x.dim() == 4:
            b, t, k, c = x.shape
            x = x.view(b, t, k * c).transpose(1, 2)  # (B, 51, T)
        elif x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # (B, 51, T)

        target_len = x.shape[2]

        z = self.encoder(x)  # (B, latent_dim)
        dec_fc = self.decoder_fc(z).view(-1, 128, 4)  # (B, 128, 4)
        reconstructed = self.decoder_conv(dec_fc)       # (B, 51, 32)

        # Interpolate to exact original length T
        if reconstructed.shape[2] != target_len:
            reconstructed = nn.functional.interpolate(reconstructed, size=target_len, mode="linear", align_corners=False)

        if len(orig_shape) == 4:
            # Reshape back to (B, T, 17, 3)
            reconstructed = reconstructed.transpose(1, 2).view(orig_shape)

        return reconstructed


class ReconstructionAnomalyScorer(AnomalyDetector):
    """Computes anomaly score based on autoencoder reconstruction discrepancy."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.22,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.threshold = threshold
        self.model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64)
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
                print(f"[Warning] Failed to load anomaly weights from {weights_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

    def score(self, sequence: np.ndarray) -> AnomalyResult:
        """Compute reconstruction error and map to [0, 1] anomaly score."""
        if self.has_weights:
            with torch.no_grad():
                tensor_seq = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
                reconstructed = self.model(tensor_seq)
                rec_np = reconstructed.cpu().numpy()[0]
                error = float(np.mean(np.abs(sequence[:, :, :2] - rec_np[:, :, :2])))
        else:
            # Heuristic calculation based on extreme body deformation & sudden acceleration
            error = self._heuristic_anomaly_error(sequence)

        # Sigmoid scaling around baseline threshold
        k = 15.0  # steepness
        anomaly_score = float(1.0 / (1.0 + np.exp(-k * (error - self.threshold))))
        is_anomaly = anomaly_score > 0.65

        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            reconstruction_error=error,
            baseline_threshold=self.threshold,
        )

    def _heuristic_anomaly_error(self, sequence: np.ndarray) -> float:
        """Physical anomaly heuristic based on abnormal kinematic variance."""
        pts = sequence[:, :, :2]  # (T, 17, 2)
        # Check acceleration spikes
        if len(pts) >= 3:
            vel = np.diff(pts, axis=0)
            acc = np.diff(vel, axis=0)
            max_acc = float(np.max(np.abs(acc)))
            # Check horizontal distortion
            last_frame = pts[-1]
            height = max(0.1, np.max(last_frame[:, 1]) - np.min(last_frame[:, 1]))
            width = max(0.1, np.max(last_frame[:, 0]) - np.min(last_frame[:, 0]))
            ar = width / height
            error = 0.10 + 0.15 * min(1.0, max_acc / 0.5) + (0.12 if ar > 1.2 else 0.0)
            return float(min(1.0, error))
        return 0.10
