"""HƯỚNG DẪN & XÁC THỰC GOOGLE DRIVE (GOOGLE ONE) TỰ ĐỘNG
================================================================================
Chạy file này để cấp quyền tải video camera lên Google Drive của bạn một lần duy nhất.
Sau khi xong, hệ thống sẽ lưu token và chạy tự động vĩnh viễn không cần đăng nhập lại.

Cách chạy:
    python scripts/setup_gdrive.py
================================================================================
"""

import sys
from pathlib import Path

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo đường dẫn gốc của project có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CREDENTIALS_FILE = PROJECT_ROOT / "configs" / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "configs" / "gdrive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def guide_user_how_to_get_credentials():
    print("""
================================================================================
 ⚠️  CHƯA TÌM THẤY FILE: configs/credentials.json
================================================================================
Để kết nối Google Drive của bạn, bạn chỉ cần lấy file credentials.json miễn phí:

BƯỚC 1: Truy cập Google Cloud Console:
        👉 https://console.cloud.google.com/

BƯỚC 2: Tạo một Project mới (ví dụ đặt tên: 'Camera-AI-Storage').

BƯỚC 3: Bật Google Drive API:
        • Vào thanh tìm kiếm trên cùng gõ: 'Google Drive API' -> Bấm [Enable] (Bật).

BƯỚC 4: Tạo thông tin xác thực (Credentials):
        • Vào menu trái chọn 'APIs & Services' -> 'Credentials'.
        • Bấm [+ CREATE CREDENTIALS] trên đầu trang -> Chọn 'OAuth client ID'.
        • Trong phần Application type: Chọn 'Desktop app' (Ứng dụng cho máy tính).
        • Đặt tên (ví dụ: 'Camera Uploader') -> Bấm [Create].

BƯỚC 5: Tải file về máy tính:
        • Bấm nút [DOWNLOAD JSON] (hoặc biểu tượng tải xuống).
        • Đổi tên file vừa tải thành: credentials.json
        • Copy/Paste file đó vào thư mục: configs/ của dự án này!
          (Đường dẫn chính xác: configs/credentials.json)

Sau khi dán file vào configs/, hãy chạy lại lệnh này:
    python scripts/setup_gdrive.py
================================================================================
""")


def authenticate_google_drive():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("\n❌ LỖI: Thiếu thư viện xác thực Google.")
        print("Vui lòng chạy lệnh cài đặt:")
        print("    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib\n")
        return

    if not CREDENTIALS_FILE.exists():
        guide_user_how_to_get_credentials()
        return

    print("\n[*] Đang mở trình duyệt để bạn đăng nhập và cấp quyền Google Drive...")
    print("[*] Một tab trình duyệt sẽ tự động bật lên, bạn chỉ cần chọn tài khoản Google và bấm [Allow / Cho phép].")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)

        # Lưu token lại
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

        print("\n================================================================================")
        print(" 🎉 [THÀNH CÔNG RỰC RỠ] Đã kết nối thành công với tài khoản Google Drive của bạn!")
        print(f" • Token đã được lưu an toàn tại: {TOKEN_FILE}")
        print(" • Từ bây giờ, mỗi khi phát hiện trộm đêm khuya, video clip sẽ tự động bay lên Google Drive!")
        print("================================================================================\n")
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra trong quá trình xác thực: {e}")


if __name__ == "__main__":
    authenticate_google_drive()
