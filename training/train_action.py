"""Training script for Spatial-Temporal Transformer on skeletal sequences with Focal Loss."""

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import SpatialTemporalTransformer
from src.action.action_classifier import ACTION_CLASSES
from src.utils.config import load_config


class SkeletalSequenceDataset(Dataset):
    """Dataset for skeletal sequence arrays (N, T, 17, 3)."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()

        if self.augment:
            seq = self._apply_augmentation(seq)

        return torch.from_numpy(seq), torch.tensor(self.labels[idx])

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        # 1. Coordinate jitter
        noise = np.random.normal(0, 0.01, size=seq.shape).astype(np.float32)
        seq[:, :, :2] += noise[:, :, :2]

        # 2. Random 2D planar rotation (-15 to +15 deg)
        angle_deg = np.random.uniform(-15.0, 15.0)
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        seq[:, :, :2] = np.dot(seq[:, :, :2], rot_mat)

        # 3. Keypoint occlusion dropout (drop 1-2 keypoints)
        if np.random.rand() > 0.5:
            drop_idx = np.random.randint(0, 17)
            seq[:, drop_idx, :2] = 0.0
            seq[:, drop_idx, 2] = 0.0

        return seq


class MultiClassFocalLoss(nn.Module):
    """Focal Loss to handle class imbalance (rare fall events vs frequent walking)."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_action_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cuda",
    save_path: str = "models/checkpoints/best_st_transformer.pt",
) -> None:
    """Train SpatialTemporalTransformer model."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Info] Training on device: {dev}")

    model = SpatialTemporalTransformer(input_dim=51, num_classes=num_classes).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MultiClassFocalLoss(gamma=2.0)

    best_val_acc = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += len(y)

        val_acc = v_correct / max(1, v_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {total_loss/total:.4f} - Train Acc: {train_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {val_acc*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Temporal Action Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Generate synthetic training samples if no pre-extracted dataset exists
    print("[Info] Preparing training datasets...")
    N = 200
    T = 30
    synth_x = np.random.randn(N, T, 17, 3).astype(np.float32)
    synth_y = np.random.randint(0, len(ACTION_CLASSES), size=(N,)).astype(np.int64)

    train_ds = SkeletalSequenceDataset(synth_x[:160], synth_y[:160], augment=True)
    val_ds = SkeletalSequenceDataset(synth_x[160:], synth_y[160:], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    train_action_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(ACTION_CLASSES),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
