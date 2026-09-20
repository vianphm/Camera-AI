import os
import sys
import zipfile
import shutil
from pathlib import Path

def package_project():
    root_dir = Path(__file__).resolve().parent
    dist_dir = root_dir / "dist" / "Fall_and_Stroke_Warning_System"
    output_zip = root_dir / "Fall_and_Stroke_Warning_System_Portable.zip"
    
    print("=" * 70)
    print("    FALL AND STROKE WARNING SYSTEM — CÔNG CỤ TỰ ĐỘNG ĐÓNG GÓI PORTABLE")
    print("=" * 70)
    print(f"[*] Thư mục dự án : {root_dir}")
    print(f"[*] Tệp nén đầu ra: {output_zip.name}\n")
    
    if dist_dir.exists() and (dist_dir / "Fall_and_Stroke_Warning_System.exe").exists():
        print(f"[+] Tìm thấy bản build độc lập PyInstaller tại: dist/Fall_and_Stroke_Warning_System")
        print(f"[*] Đang đóng gói toàn bộ ứng dụng độc lập (Standalone App)...")
        
        # Đảm bảo có hướng dẫn sử dụng trong thư mục dist
        huong_dan_src = root_dir / "HUONG_DAN_SU_DUNG.txt"
        if huong_dan_src.exists():
            shutil.copy2(huong_dan_src, dist_dir / "HUONG_DAN_SU_DUNG.txt")
            
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for dirpath, _, filenames in os.walk(dist_dir):
                for filename in filenames:
                    full_path = Path(dirpath) / filename
                    rel_path = full_path.relative_to(dist_dir)
                    zipf.write(full_path, arcname=f"Fall_and_Stroke_Warning_System/{str(rel_path)}")
    else:
        print("[!] Chưa tìm thấy thư mục dist/Fall_and_Stroke_Warning_System.")
        print("[*] Đang đóng gói phiên bản mã nguồn dự phòng...")
        include_dirs = ["configs", "frontend", "models", "src", "scripts", "docs"]
        include_files = [
            "Fall_and_Stroke_Warning_System.exe",
            "Build_App_PyInstaller.bat",
            "HUONG_DAN_SU_DUNG.txt",
            "README.md",
            "app_icon.ico",
            "app_icon.png",
            "app_runner.py",
            "main.py",
            "requirements.txt",
        ]
        exclude_extensions = {".pyc", ".pyo", ".tmp", ".log"}
        exclude_dirs = {"__pycache__", ".git", ".claude", ".gemini", "tests", "data", "build", "dist"}

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for fname in include_files:
                fpath = root_dir / fname
                if fpath.is_file():
                    zipf.write(fpath, arcname=f"Fall_and_Stroke_Warning_System/{fname}")
            
            for folder_name in include_dirs:
                folder_path = root_dir / folder_name
                if not folder_path.is_dir():
                    continue
                for dirpath, dirnames, filenames in os.walk(folder_path):
                    dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
                    for filename in filenames:
                        ext = Path(filename).suffix.lower()
                        if ext in exclude_extensions:
                            continue
                        full_path = Path(dirpath) / filename
                        rel_path = full_path.relative_to(root_dir)
                        zipf.write(full_path, arcname=f"Fall_and_Stroke_Warning_System/{str(rel_path)}")

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f" [THÀNH CÔNG] Đã tạo gói nén hoàn chỉnh: {output_zip.name}")
    print(f" [DUNG LƯỢNG]: {zip_size_mb:.2f} MB")
    print(f" [VỊ TRÍ FILE]: {output_zip}")
    print(" Người nhận chỉ cần giải nén file ZIP và bấm vào file:")
    print(" 👉 Fall_and_Stroke_Warning_System.exe")
    print(" là sử dụng được ngay, KHÔNG CẦN CÀI THÊM BẤT KỲ CÁI GÌ!")
    print("=" * 70)

if __name__ == "__main__":
    package_project()
