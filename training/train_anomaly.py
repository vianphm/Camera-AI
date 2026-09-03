"""Training script for Pose Sequence Autoencoder on normal skeletal activity sequences."""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly.anomaly_score import PoseSequenceAutoencoder


class NormalPoseDataset(Dataset):
    """Dataset containing normal ADL activities for reconstruction training."""

    def __init__(self, sequences: np.ndarray) -> None:
        self.sequences = sequences.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx].copy()
        # Shape: (T, 17, 3)
        return torch.from_numpy(seq)


def train_anomaly_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    lr: float = 0.0005,
    device: str = "cuda",
    save_path: str = "models/checkpoints/best_pose_autoencoder.pt",
) -> None:
    """Train unsupervised autoencoder minimizing reconstruction loss."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Info] Training Anomaly Autoencoder on {dev}")

    model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0

        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            count += len(batch)

        # Validation
        model.eval()
        v_loss, v_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                rec = model(batch)
                loss = criterion(rec, batch)
                v_loss += loss.item() * len(batch)
                v_count += len(batch)

        train_l = total_loss / max(1, count)
        val_l = v_loss / max(1, v_count)
        print(f"Epoch {epoch:02d}/{epochs:02d} - Train Recon Loss: {train_l:.5f} - Val Recon Loss: {val_l:.5f}")

        if val_l < best_val_loss:
            best_val_loss = val_l
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_l}, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {val_l:.5f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pose Sequence Autoencoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Synthetic normal samples (standing/walking/sitting)
    N = 200
    T = 30
    synth_x = np.random.randn(N, T, 17, 3).astype(np.float32)

    train_ds = NormalPoseDataset(synth_x[:160])
    val_ds = NormalPoseDataset(synth_x[160:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    train_anomaly_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
