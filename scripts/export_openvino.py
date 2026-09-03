"""Export models to Intel OpenVINO (FP16 / INT8) for CPU/iGPU acceleration.

Optimized for Intel 12th Gen Core i5-12450HX CPU using AVX2 and VNNI instructions.
Allows running elderly AI monitoring with ultra-low latency on systems without dedicated GPUs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_yolo_openvino(
    model_name: str = "yolov8n-pose.pt",
    half: bool = True,
    int8: bool = False,
    img_size: int = 640,
) -> None:
    """Export YOLOv8-Pose to OpenVINO using Ultralytics exporter."""
    print(f"[Info] Exporting YOLO-Pose ({model_name}) to OpenVINO...")
    print(f"       Configuration: half={half}, int8={int8}, imgsz={img_size}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        export_kwargs = {
            "format": "openvino",
            "half": half,
            "int8": int8,
            "imgsz": img_size,
        }
        exported_path = model.export(**export_kwargs)
        print(f"[Success] OpenVINO model generated successfully at: {exported_path}")
        print("          You can now pass this directory directly to YOLOv8PoseEstimator(model_path=...)")
    except Exception as e:
        print(f"[Error] Failed to export YOLO to OpenVINO: {e}")
        print("        Ensure 'openvino' is installed: pip install openvino")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to Intel OpenVINO (FP16/INT8)")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Path to YOLOv8 pose model")
    parser.add_argument("--half", action="store_true", default=True, help="Export OpenVINO FP16")
    parser.add_argument("--int8", action="store_true", default=False, help="Export OpenVINO INT8 quantization")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    export_yolo_openvino(
        model_name=args.model,
        half=args.half,
        int8=args.int8,
        img_size=args.imgsz,
    )


if __name__ == "__main__":
    main()
