"""Master real-time video intelligence pipeline orchestrating all modules."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from src.detection.detector import PersonDetector
from src.detection.person_detector import YOLOv8PersonDetector
from src.detection.detector_factory import create_detector
from src.tracking.tracker import Tracker, Track
from src.tracking.track_manager import ByteTrackManager
from src.pose.pose_estimator import PoseEstimator, PoseResult, YOLOv8PoseEstimator
from src.pose.pose_factory import create_pose_estimator
from src.temporal.sequence_buffer import SequenceBuffer, TrackSnapshot
from src.temporal.temporal_features import TemporalFeatureExtractor
from src.action.action_classifier import ActionClassifier
from src.action.behavior_classifier import BehaviorSequenceAnalyzer
from src.anomaly.anomaly_detector import AnomalyDetector
from src.anomaly.anomaly_score import ReconstructionAnomalyScorer
from src.risk.risk_engine import RiskEngine, RiskAssessment
from src.risk.threshold_manager import ThresholdManager
from src.alerts.alert_manager import AlertManager, AlertEvent
from src.utils.visualizer import PipelineVisualizer
from src.utils.privacy import FaceBlurrer
from src.utils.profiler import LatencyProfiler, ResourceMonitor
from src.utils.config import load_config


@dataclass
class PipelineFrameResult:
    """Comprehensive output produced after processing a single video frame."""
    frame_idx: int
    timestamp: float
    annotated_frame: np.ndarray
    tracks: List[Track]
    poses: List[PoseResult]
    assessments: List[RiskAssessment]
    alerts: List[AlertEvent]
    telemetry: Dict[str, Any]


class RealtimePipeline:
    """End-to-End multi-stage video understanding pipeline for elderly monitoring."""

    def __init__(
        self,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[Tracker] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        action_classifier: Optional[ActionClassifier] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        risk_engine: Optional[RiskEngine] = None,
        alert_manager: Optional[AlertManager] = None,
        visualizer: Optional[PipelineVisualizer] = None,
        blur_faces: bool = False,
    ) -> None:
        # Load configs
        try:
            model_cfg = load_config("model.yaml")
        except Exception:
            model_cfg = {}

        try:
            infer_cfg = load_config("inference.yaml")
        except Exception:
            infer_cfg = {}

        import torch
        dev = infer_cfg.get("runtime", {}).get("device", "cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.use_fp16 = infer_cfg.get("runtime", {}).get("use_fp16", True) and (dev != "cpu")
        # Real-time stream: no face blurring
        self.blur_faces = infer_cfg.get("privacy", {}).get("blur_faces", False) and blur_faces

        # 1. Detection (Config-driven Pluggable Architecture)
        det_cfg = model_cfg.get("detector", {})
        det_arch = det_cfg.get("architecture", "yolov8")
        self.detector = detector or create_detector(
            architecture=det_arch,
            model_path=det_cfg.get("model_name", "yolov8s.pt"),
            conf_threshold=det_cfg.get("conf_threshold", 0.40),
            device=self.device,
            half=self.use_fp16,
        )

        # 2. Tracking (ByteTrack)
        trk_cfg = model_cfg.get("tracker", {})
        self.tracker = tracker or ByteTrackManager(
            track_high_thresh=trk_cfg.get("track_high_thresh", 0.50),
            track_low_thresh=trk_cfg.get("track_low_thresh", 0.15),
            new_track_thresh=trk_cfg.get("new_track_thresh", 0.60),
            match_thresh=trk_cfg.get("match_thresh", 0.70),
        )

        # 3. Pose (Config-driven Pluggable Architecture)
        pose_cfg = model_cfg.get("pose", {})
        pose_arch = pose_cfg.get("architecture", "yolov8_pose")
        self.pose_estimator = pose_estimator or create_pose_estimator(
            architecture=pose_arch,
            model_path=pose_cfg.get("model_name", "yolov8s-pose.pt"),
            conf_threshold=pose_cfg.get("conf_threshold", 0.35),
            device=self.device,
            half=self.use_fp16,
        )

        # 4. Temporal Sequence Buffer & Kinematics
        temp_cfg = model_cfg.get("temporal", {})
        window_size = temp_cfg.get("window_size", 30)
        self.sequence_buffer = SequenceBuffer(window_size=window_size)
        self.feature_extractor = TemporalFeatureExtractor(fps=15.0)

        # 5. Dual-Stream AI Models
        self.action_classifier = action_classifier or ActionClassifier(
            weights_path=temp_cfg.get("weights_path", None),
            device=self.device,
        )
        self.anomaly_detector = anomaly_detector or ReconstructionAnomalyScorer(
            weights_path=model_cfg.get("anomaly", {}).get("weights_path", None),
            device=self.device,
        )
        self.behavior_analyzer = BehaviorSequenceAnalyzer()

        # 6. Risk Engine & State Machine
        self.thresholds = ThresholdManager()
        self.risk_engine = risk_engine or RiskEngine(self.thresholds)

        # 7. Alert Manager
        self.alert_manager = alert_manager or AlertManager(self.thresholds)

        # 8. Visualizer & Utilities
        self.visualizer = visualizer or PipelineVisualizer(infer_cfg.get("display", {}))
        self.face_blurrer = FaceBlurrer()
        self.profiler = LatencyProfiler()
        self.resource_monitor = ResourceMonitor()

        self._frame_count = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> PipelineFrameResult:
        """Execute end-to-end multi-stage pipeline on a single frame."""
        self.profiler.start_frame()
        self._frame_count += 1
        annotated_frame = frame.copy()

        # STAGE 1: Person Detection
        self.profiler.start("detection")
        detections = self.detector.detect(frame)
        self.profiler.stop("detection")

        # STAGE 2: Multi-Object Tracking
        self.profiler.start("tracking")
        active_tracks = self.tracker.update(detections, timestamp)
        active_track_ids = {t.track_id for t in active_tracks}
        self.sequence_buffer.update_activity(active_track_ids)
        self.risk_engine.cleanup_inactive(active_track_ids)
        self.profiler.stop("tracking")

        # STAGE 3: Human Pose Estimation
        self.profiler.start("pose")
        poses = self.pose_estimator.estimate(frame, active_tracks)
        pose_dict = {p.track_id: p for p in poses}
        self.profiler.stop("pose")

        # STAGE 4 & 5: Temporal Sequence Buffer & AI Reasoning
        self.profiler.start("temporal_reasoning")
        assessments: List[RiskAssessment] = []
        alerts_raised: List[AlertEvent] = []

        for track in active_tracks:
            tid = track.track_id
            pose = pose_dict.get(tid)

            # Record snapshot
            if pose is not None:
                snapshot = TrackSnapshot(
                    timestamp=timestamp,
                    bbox=track.bbox,
                    keypoints=pose.keypoints,
                    normalized_keypoints=pose.normalized_keypoints,
                    torso_angle=pose.torso_angle,
                )
                self.sequence_buffer.add_snapshot(tid, snapshot)

            # Retrieve temporal sequence
            seq = self.sequence_buffer.get_normalized_sequence(tid)
            if seq is not None and self.sequence_buffer.is_ready(tid):
                # Supervised Action Classification
                action_pred = self.action_classifier.predict(seq)
                self.behavior_analyzer.update(tid, action_pred.primary_action, timestamp)

                # Unsupervised Anomaly Scoring
                anomaly_res = self.anomaly_detector.score(seq)

                # Kinematic Feature Extraction
                snapshots_hist = self.sequence_buffer.get_snapshots(tid)
                kinematics = self.feature_extractor.extract(snapshots_hist)

                # Multi-Signal Risk Assessment
                assessment = self.risk_engine.assess(
                    track_id=tid,
                    action_pred=action_pred,
                    anomaly_res=anomaly_res,
                    kinematics=kinematics,
                    timestamp=timestamp,
                )
                assessments.append(assessment)

                # STAGE 6: Alert Dispatching
                # If high-risk confirmed, prepare anonymized snapshot
                snapshot_frame = None
                if assessment.should_alert:
                    snapshot_frame = frame.copy()
                    if self.blur_faces and pose is not None:
                        snapshot_frame = self.face_blurrer.blur_keypoints_face(snapshot_frame, pose.keypoints)
                    elif self.blur_faces:
                        snapshot_frame = self.face_blurrer.blur_bbox_head(snapshot_frame, track.bbox)

                alert_evt = self.alert_manager.process_assessment(
                    assessment=assessment,
                    frame=snapshot_frame,
                    timestamp=timestamp,
                )
                if alert_evt:
                    alerts_raised.append(alert_evt)

                # Render Visual Overlays
                if pose is not None and self.visualizer.show_skeleton:
                    color = self.visualizer.get_state_color(assessment.state)
                    self.visualizer.draw_skeleton(annotated_frame, pose.keypoints, color=color)

                self.visualizer.draw_person_card(
                    frame=annotated_frame,
                    track_id=tid,
                    bbox=track.bbox,
                    state=assessment.state,
                    action=action_pred.primary_action,
                    risk_score=assessment.risk_score,
                    evidence=assessment.evidence,
                )
            else:
                # Buffer warming up, draw minimal bounding box
                self.visualizer.draw_person_card(
                    frame=annotated_frame,
                    track_id=tid,
                    bbox=track.bbox,
                    state="NORMAL",
                    action="calibrating",
                    risk_score=0.0,
                )

        self.profiler.stop("temporal_reasoning")

        # STAGE 7: Privacy Face Blurring on Live Output (if enabled)
        if self.blur_faces:
            for p in poses:
                self.face_blurrer.blur_keypoints_face(annotated_frame, p.keypoints)

        # STAGE 8: HUD & Telemetry
        telemetry = {
            **self.profiler.get_summary(),
            **self.resource_monitor.get_telemetry(),
            "active_tracks_count": len(active_tracks),
        }
        self.visualizer.draw_hud(
            frame=annotated_frame,
            fps=self.profiler.fps,
            active_tracks=len(active_tracks),
            telemetry=telemetry,
        )

        return PipelineFrameResult(
            frame_idx=self._frame_count,
            timestamp=timestamp,
            annotated_frame=annotated_frame,
            tracks=active_tracks,
            poses=poses,
            assessments=assessments,
            alerts=alerts_raised,
            telemetry=telemetry,
        )
