"""FastAPI Web Server exposing REST APIs and WebSockets for monitoring and alerts."""

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.pipeline.realtime_pipeline import RealtimePipeline
from src.camera.webcam import WebcamStream
from src.camera.rtsp import RTSPStream
from src.camera.video_file import VideoFileStream
from src.alerts.event_logger import EventLogger, AlertEvent
from src.utils.config import get_project_root, load_config
from src.utils.profiler import ResourceMonitor

app = FastAPI(
    title="Elderly AI Monitor API",
    description="Real-time Computer Vision & Temporal Behavioral Anomaly Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
state_lock = threading.Lock()
pipeline_instance: Optional[RealtimePipeline] = None
camera_stream: Optional[Any] = None
is_running = False
latest_annotated_frame: Optional[bytes] = None
active_ws_clients: List[WebSocket] = []
event_logger = EventLogger()
resource_monitor = ResourceMonitor()


class CameraStartRequest(BaseModel):
    source_type: str = "webcam"  # "webcam", "rtsp", "video"
    device_index: int = 0
    rtsp_url: Optional[str] = None
    video_path: Optional[str] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Check API server health and GPU telemetry."""
    telemetry = resource_monitor.get_telemetry()
    return {
        "status": "healthy",
        "service": "elderly-ai-monitor",
        "timestamp": time.time(),
        "telemetry": telemetry,
    }


@app.get("/status")
def system_status() -> Dict[str, Any]:
    """Retrieve operational state of camera and AI inference pipeline."""
    with state_lock:
        running = is_running
        fps = pipeline_instance.profiler.fps if pipeline_instance else 0.0
        tracks_count = len(pipeline_instance.tracker._tracks) if (pipeline_instance and hasattr(pipeline_instance.tracker, "_tracks")) else 0

    return {
        "camera_active": running,
        "current_fps": round(fps, 1),
        "tracked_persons_count": tracks_count,
        "timestamp": time.time(),
    }


@app.post("/camera/start")
def start_camera(req: CameraStartRequest) -> Dict[str, Any]:
    """Start the video ingestion and AI inference pipeline."""
    global camera_stream, pipeline_instance, is_running

    with state_lock:
        if is_running:
            return {"status": "already_running", "message": "Camera pipeline is already active."}

        # Initialize Camera
        if req.source_type == "webcam":
            camera_stream = WebcamStream(device_index=req.device_index)
        elif req.source_type == "rtsp":
            if not req.rtsp_url:
                raise HTTPException(status_code=400, detail="rtsp_url must be provided for rtsp source_type.")
            camera_stream = RTSPStream(url=req.rtsp_url)
        elif req.source_type == "video":
            if not req.video_path:
                raise HTTPException(status_code=400, detail="video_path must be provided for video source_type.")
            camera_stream = VideoFileStream(video_path=req.video_path, loop=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source_type: {req.source_type}")

        if not camera_stream.start():
            raise HTTPException(status_code=500, detail="Failed to initialize camera capture stream.")

        # Initialize Pipeline
        pipeline_instance = RealtimePipeline()

        # Connect alert dispatcher callback to broadcast over WebSocket
        def on_alert_dispatched(event: AlertEvent):
            broadcast_alert_event(event)

        pipeline_instance.alert_manager.dispatcher.subscribe_websocket(on_alert_dispatched)

        is_running = True
        threading.Thread(target=_pipeline_worker_loop, daemon=True, name="PipelineWorker").start()

    return {"status": "started", "source": req.source_type}


@app.post("/camera/stop")
def stop_camera() -> Dict[str, Any]:
    """Stop the video pipeline."""
    global camera_stream, is_running

    with state_lock:
        if not is_running:
            return {"status": "not_running", "message": "Pipeline is not running."}

        is_running = False
        if camera_stream:
            camera_stream.release()
            camera_stream = None

    return {"status": "stopped"}


def _pipeline_worker_loop() -> None:
    """Continuous processing loop running on a dedicated thread."""
    global latest_annotated_frame, is_running

    while is_running and camera_stream and camera_stream.is_opened():
        packet = camera_stream.read(timeout=0.5)
        if packet is None or packet.frame is None:
            time.sleep(0.01)
            continue

        result = pipeline_instance.process_frame(packet.frame, packet.timestamp)

        # Encode frame to JPEG for MJPEG browser stream
        ret, jpeg = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with state_lock:
                latest_annotated_frame = jpeg.tobytes()


@app.get("/events")
def list_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent historical emergency alerts."""
    return event_logger.get_recent_events(limit=limit)


@app.get("/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    """Fetch details of a specific event."""
    events = event_logger.get_recent_events(limit=200)
    for evt in events:
        if evt.get("event_id") == event_id:
            return evt
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")


def mjpeg_generator():
    """Generator yielding multipart MJPEG video frames."""
    while is_running:
        with state_lock:
            frame_bytes = latest_annotated_frame

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS max for MJPEG web feed


@app.get("/stream/mjpeg")
def video_mjpeg_feed():
    """MJPEG Live Video Stream for HTML <img> embedding."""
    if not is_running:
        raise HTTPException(status_code=400, detail="Camera pipeline is not running. Start via POST /camera/start.")
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def broadcast_alert_event(event: AlertEvent) -> None:
    """Send alert JSON to all connected WebSocket clients."""
    from dataclasses import asdict
    payload = json.dumps(asdict(event))

    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        pass

    for ws in list(active_ws_clients):
        try:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send_text(payload), loop)
        except Exception:
            pass


@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """WebSocket stream pushing real-time emergency events."""
    await websocket.accept()
    active_ws_clients.append(websocket)
    try:
        while True:
            # Keep-alive ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_ws_clients:
            active_ws_clients.remove(websocket)
