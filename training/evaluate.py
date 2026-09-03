"""Model evaluation script computing Precision, Recall, F1, and Confusion Matrix."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.temporal.temporal_model import SpatialTemporalTransformer
from src.action.action_classifier import ACTION_CLASSES
from training.train_action import SkeletalSequenceDataset


def evaluate_model(
    model: SpatialTemporalTransformer,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    num_classes = len(ACTION_CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds_np, targets_np):
        cm[t, p] += 1

    metrics_per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0

    for i, cls_name in enumerate(ACTION_CLASSES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)

        metrics_per_class[cls_name] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "support": int(np.sum(cm[i, :])),
        }
        macro_precision += prec
        macro_recall += rec

    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 = 2 * (macro_precision * macro_recall) / max(1e-6, macro_precision + macro_recall)

    overall_acc = float(np.sum(preds_np == targets_np) / len(targets_np))

    summary = {
        "overall_accuracy": round(overall_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": metrics_per_class,
        "confusion_matrix": cm.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"Macro Precision:  {macro_precision*100:.2f}%")
    print(f"Macro Recall:     {macro_recall*100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1*100:.2f}%\n")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 60)
    for cls_name, m in metrics_per_class.items():
        print(f"{cls_name:<20} | {m['precision']*100:>8.1f}% | {m['recall']*100:>8.1f}% | {m['f1']*100:>8.1f}%")
    print("=" * 60)
    print(f"[Info] Full metrics and confusion matrix saved to {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Action Model")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best_st_transformer.pt")
    parser.add_argument("--output-dir", type=str, default="results/evaluation_reports")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = SpatialTemporalTransformer(input_dim=51, num_classes=len(ACTION_CLASSES))

    if Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=dev)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[Info] Loaded weights from {args.weights}")
    else:
        print(f"[Warning] Weights file {args.weights} not found. Running benchmark with uninitialized weights.")

    model.to(dev)

    # Test dataset
    N, T = 100, 30
    synth_x = np.random.randn(N, T, 17, 3).astype(np.float32)
    synth_y = np.random.randint(0, len(ACTION_CLASSES), size=(N,)).astype(np.int64)

    test_ds = SkeletalSequenceDataset(synth_x, synth_y)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    evaluate_model(model, test_loader, dev, Path(args.output_dir))


if __name__ == "__main__":
    main()
