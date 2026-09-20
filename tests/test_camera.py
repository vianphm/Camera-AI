"""Unit tests for camera ingestion modules."""

import numpy as np
import pytest
from src.camera.stream import FramePacket, CameraStream
from src.camera.video_file import VideoFileStream


def test_frame_packet_creation():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    packet = FramePacket(frame=dummy, frame_idx=1, timestamp=12345.67, source_name="test_cam")
    assert packet.frame.shape == (100, 100, 3)
    assert packet.frame_idx == 1
    assert packet.source_name == "test_cam"


def test_video_file_stream_nonexistent():
    stream = VideoFileStream(video_path="nonexistent_video.mp4")
    assert not stream.start()
    assert not stream.is_opened()
    stream.release()


def test_camera_api_endpoints():
    from fastapi.testclient import TestClient
    from src.api.server import app

    client = TestClient(app)

    # Test /camera/info
    res_info = client.get("/camera/info")
    assert res_info.status_code == 200
    data_info = res_info.json()
    assert "source_type" in data_info
    assert "resolution" in data_info

    # Test /camera/devices
    res_devices = client.get("/camera/devices")
    assert res_devices.status_code == 200
    data_devices = res_devices.json()
    assert isinstance(data_devices, list)
    assert len(data_devices) > 0
    assert "device_index" in data_devices[0]

    # Test /camera/switch validation
    res_invalid = client.post("/camera/switch", json={"source_type": "rtsp", "rtsp_url": ""})
    assert res_invalid.status_code == 400

