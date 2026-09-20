import os
import shutil
from pathlib import Path

def reorganize():
    root = Path(__file__).resolve().parent.parent
    print(f"[*] Reorganizing root directory: {root}")

    # 1. Tạo thư mục docs/ và di chuyển các file tài liệu thiết kế vào docs/
    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)
    doc_files = [
        "ARCHITECTURE.md",
        "BEHAVIOR_CLASS_CONTRACT.md",
        "DATASET.md",
        "EVALUATION.md",
        "MODEL_SELECTION.md",
        "TRAINING.md",
    ]
    for doc in doc_files:
        src_f = root / doc
        if src_f.exists():
            shutil.move(str(src_f), str(docs_dir / doc))
            print(f" [+] Moved to docs/: {doc}")

    # 2. Di chuyển yolov8n-pose.onnx vào models/pose/
    pose_onnx = root / "yolov8n-pose.onnx"
    if pose_onnx.exists():
        dest = root / "models" / "pose" / "yolov8n-pose.onnx"
        shutil.move(str(pose_onnx), str(dest))
        print(" [+] Moved yolov8n-pose.onnx to models/pose/")

    # 3. Xóa các file thừa, trùng lặp không cần thiết
    obsolete_files = [
        "yolov8n-pose.pt",            # PyTorch weights cũ, không dùng
        "setup.bat",                   # Trùng lặp với 1_Cai_Dat_Tu_Dong.bat
        "Run_Elderly_Monitor.bat",     # File cũ thời tên cũ
        "Run_Fall_Stroke_Warning.bat", # Trùng lặp với 2_Chay_He_Thong.bat
    ]
    for obs in obsolete_files:
        p = root / obs
        if p.exists():
            p.unlink()
            print(f" [-] Deleted obsolete file: {obs}")

    # 4. Di chuyển các file script phụ trợ vào thư mục scripts/ (vì đã có trên UI Settings)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    util_scripts = [
        "Bat_Khoi_Dong_Cung_Windows.bat",
        "Tat_Khoi_Dong_Cung_Windows.bat",
        "Kiem_Tra_Quyen_Camera.bat",
        "Dung_He_Thong.bat",
        "Chay_Ngam_He_Thong.vbs"
    ]
    for s in util_scripts:
        src_s = root / s
        if src_s.exists():
            dest_s = scripts_dir / s
            if dest_s.exists():
                dest_s.unlink()
            shutil.move(str(src_s), str(dest_s))
            print(f" [+] Moved to scripts/: {s}")

    # 5. Dọn dẹp __pycache__ ở root
    pycache = root / "__pycache__"
    if pycache.exists():
        shutil.rmtree(str(pycache), ignore_errors=True)
        print(" [-] Cleaned root __pycache__")

    print("\n[SUCCESS] Hoàn tất tổ chức lại thư mục dự án sạch đẹp, gọn gàng!")

if __name__ == "__main__":
    reorganize()
