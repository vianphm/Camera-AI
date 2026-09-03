"""Export PyTorch temporal models to ONNX format with dynamic batching."""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import SpatialTemporalTransformer
from src.anomaly.anomaly_score import PoseSequenceAutoencoder


def export_action_model(weights_path: str, output_path: str) -> None:
    print(f"[Info] Exporting Action Model to ONNX: {output_path}...")
    model = SpatialTemporalTransformer(input_dim=51, num_classes=10)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.eval()
    dummy_input = torch.randn(1, 30, 17, 3, dtype=torch.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["skeletal_sequence"],
        output_names=["action_logits"],
        dynamic_axes={
            "skeletal_sequence": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )
    print(f"[Success] Action model exported successfully to {output_path}")


def export_anomaly_model(weights_path: str, output_path: str) -> None:
    print(f"[Info] Exporting Anomaly Model to ONNX: {output_path}...")
    model = PoseSequenceAutoencoder(input_dim=51, latent_dim=64)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.eval()
    dummy_input = torch.randn(1, 30, 17, 3, dtype=torch.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["skeletal_sequence"],
        output_names=["reconstructed_sequence"],
        dynamic_axes={
            "skeletal_sequence": {0: "batch_size"},
            "reconstructed_sequence": {0: "batch_size"},
        },
    )
    print(f"[Success] Anomaly model exported successfully to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--model-type", type=str, choices=["action", "anomaly", "all"], default="all")
    parser.add_argument("--action-weights", type=str, default="models/checkpoints/best_st_transformer.pt")
    parser.add_argument("--anomaly-weights", type=str, default="models/checkpoints/best_pose_autoencoder.pt")
    args = parser.parse_args()

    if args.model_type in ["action", "all"]:
        export_action_model(args.action_weights, "models/temporal/st_transformer.onnx")
    if args.model_type in ["anomaly", "all"]:
        export_anomaly_model(args.anomaly_weights, "models/anomaly/pose_autoencoder.onnx")


if __name__ == "__main__":
    main()
