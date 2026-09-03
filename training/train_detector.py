"""Fine-tuning script for YOLOv8 person detector."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO Person Detector")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml dataset definition")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained weights to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0", help="GPU index or 'cpu'")
    args = parser.parse_args()

    print(f"[Info] Starting YOLO fine-tuning with {args.model} on {args.data}...")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        classes=[0],  # Person class only
        project="models/detector",
        name="fine_tuned_person_detector",
    )
    print("[Info] Training complete. Exporting best weights...")


if __name__ == "__main__":
    main()
