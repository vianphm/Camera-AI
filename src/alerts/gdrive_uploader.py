"""Google Drive Uploader for automatically pushing security alarm clips to the Cloud."""

import os
from pathlib import Path
from typing import Optional, Dict, Any


class GoogleDriveUploader:
    """Tải video và hình ảnh bằng chứng lên Google Drive (Google One) tự động."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(
        self,
        token_path: str = "configs/gdrive_token.json",
        credentials_path: str = "configs/credentials.json",
        folder_name: str = "EZVIZ_Security_Alerts",
    ) -> None:
        self.token_path = Path(token_path)
        self.credentials_path = Path(credentials_path)
        self.folder_name = folder_name
        self.service = None
        self._folder_id: Optional[str] = None

        self._init_service()

    def _init_service(self) -> bool:
        """Khởi tạo service Google Drive từ token đã lưu."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            if self.token_path.exists():
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
                self.service = build("drive", "v3", credentials=creds)
                return True
        except ImportError:
            print("⚠️ [GDRIVE] Chưa cài thư viện Google Drive API. Chạy: pip install google-api-python-client google-auth-oauthlib")
        except Exception as e:
            print(f"⚠️ [GDRIVE AUTH ERROR]: {e}")

        return False

    @property
    def is_ready(self) -> bool:
        return self.service is not None

    def _get_or_create_folder(self) -> Optional[str]:
        """Tìm hoặc tạo thư mục lưu trữ riêng trên Google Drive."""
        if self._folder_id:
            return self._folder_id

        if not self.service:
            return None

        try:
            # Tìm thư mục đã tồn tại
            query = f"name = '{self.folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            response = self.service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
            files = response.get("files", [])

            if files:
                self._folder_id = files[0]["id"]
                return self._folder_id

            # Chưa có thì tạo mới
            folder_metadata = {
                "name": self.folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = self.service.files().create(body=folder_metadata, fields="id").execute()
            self._folder_id = folder.get("id")
            print(f"📁 [GDRIVE] Đã tạo thư mục lưu trữ mới: '{self.folder_name}' (ID: {self._folder_id})")
            return self._folder_id
        except Exception as e:
            print(f"❌ [GDRIVE FOLDER ERROR]: {e}")
            return None

    def upload_file(self, file_path: str, mime_type: str = "video/mp4", share_link: bool = True) -> Optional[str]:
        """
        Tải một file (MP4 hoặc JPG) lên Google Drive.
        Trả về đường link xem online (webViewLink).
        """
        if not self.service:
            print("⚠️ [GDRIVE] Chưa cấu hình Google Drive, bỏ qua tải lên Cloud.")
            return None

        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"❌ [GDRIVE] Tệp không tồn tại: {file_path}")
            return None

        try:
            from googleapiclient.http import MediaFileUpload

            folder_id = self._get_or_create_folder()
            file_metadata = {"name": path_obj.name}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            media = MediaFileUpload(str(path_obj), mimetype=mime_type, resumable=True)
            print(f"☁️ [GDRIVE UPLOADING] Đang tải {path_obj.name} lên Google Drive...")

            uploaded_file = (
                self.service.files()
                .create(body=file_metadata, media_body=media, fields="id, webViewLink, webContentLink")
                .execute()
            )

            file_id = uploaded_file.get("id")
            web_link = uploaded_file.get("webViewLink")

            # Mở quyền xem qua link để gửi vào Telegram cho chủ nhà bấm xem ngay được
            if share_link and file_id:
                try:
                    permission = {"type": "anyone", "role": "reader"}
                    self.service.permissions().create(fileId=file_id, body=permission).execute()
                except Exception as perm_err:
                    print(f"⚠️ [GDRIVE PERMISSION NOTICE]: {perm_err}")

            print(f"✅ [GDRIVE SUCCESS] Tải lên thành công! Link xem video: {web_link}")
            return web_link
        except Exception as e:
            print(f"❌ [GDRIVE UPLOAD ERROR]: {e}")
            return None
