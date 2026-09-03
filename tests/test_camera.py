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
