"""Camera and video stream ingestion module."""

from src.camera.stream import CameraStream, FramePacket
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream
from src.camera.multi_camera import MultiCameraManager, CameraConfig

__all__ = [
    "CameraStream",
    "FramePacket",
    "WebcamStream",
    "RTSPStream",
    "VideoFileStream",
    "MultiCameraManager",
    "CameraConfig",
]
