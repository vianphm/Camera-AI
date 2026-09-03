"""Keypoint constants, normalization, and geometric kinematic calculations."""

from typing import List, Tuple, Optional
import numpy as np

# Standard 17 COCO Keypoints
KEYPOINT_NAMES: List[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

SKELETON_EDGES: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Lower body & hips
    (5, 11), (6, 12), (5, 6),                           # Torso rectangle
    (5, 7), (7, 9), (6, 8), (8, 10),                    # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6) # Head
]


def normalize_keypoints(keypoints: np.ndarray, bbox_fallback: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Normalize 17 keypoints to be root-centered and scale-invariant.

    Root point: Mid-Hip (average of left hip [11] and right hip [12]).
    Scale factor: Torso length (distance between Mid-Shoulder and Mid-Hip).

    Args:
        keypoints: Array of shape (17, 3) where each row is (x, y, conf).
        bbox_fallback: Optional (x1, y1, x2, y2) used if hip/shoulder keypoints have low confidence.

    Returns:
        Normalized array of shape (17, 3) where x, y are normalized and conf is preserved.
    """
    if len(keypoints) < 17:
        return np.zeros((17, 3), dtype=np.float32)

    coords = keypoints[:, :2].copy()
    confs = keypoints[:, 2:3].copy()

    l_hip, r_hip = coords[11], coords[12]
    c_l_hip, c_r_hip = confs[11, 0], confs[12, 0]

    # Compute root (Mid-Hip)
    if c_l_hip > 0.2 and c_r_hip > 0.2:
        root = (l_hip + r_hip) / 2.0
    elif c_l_hip > 0.2:
        root = l_hip
    elif c_r_hip > 0.2:
        root = r_hip
    else:
        # Fallback to center of bounding box or mean of valid points
        valid = confs[:, 0] > 0.2
        if np.any(valid):
            root = np.mean(coords[valid], axis=0)
        elif bbox_fallback is not None:
            root = np.array([(bbox_fallback[0] + bbox_fallback[2]) / 2.0, (bbox_fallback[1] + bbox_fallback[3]) / 2.0])
        else:
            root = np.array([0.0, 0.0])

    # Compute scale (Torso length: distance between Mid-Shoulder and Mid-Hip)
    l_sh, r_sh = coords[5], coords[6]
    c_l_sh, c_r_sh = confs[5, 0], confs[6, 0]

    if (c_l_sh > 0.2 or c_r_sh > 0.2) and (c_l_hip > 0.2 or c_r_hip > 0.2):
        if c_l_sh > 0.2 and c_r_sh > 0.2:
            mid_sh = (l_sh + r_sh) / 2.0
        elif c_l_sh > 0.2:
            mid_sh = l_sh
        else:
            mid_sh = r_sh
        scale = float(np.linalg.norm(mid_sh - root))
    else:
        scale = 0.0

    # Fallback scale: use bounding box height if torso scale is too small or occluded
    if scale < 15.0:
        if bbox_fallback is not None:
            scale = max(20.0, (bbox_fallback[3] - bbox_fallback[1]) * 0.4)
        else:
            scale = 100.0

    # Center and scale
    norm_coords = (coords - root) / max(1.0, scale)

    # Return normalized (17, 3)
    return np.hstack([norm_coords, confs]).astype(np.float32)


def compute_torso_angle(keypoints: np.ndarray) -> float:
    """Compute angle (degrees) between torso axis (Mid-Hip to Mid-Shoulder) and vertical vector.

    Returns:
        Angle in degrees:
        ~0 degrees   -> Straight Upright (Standing / Walking)
        ~45 degrees  -> Bending forward / Stumbling
        ~90 degrees  -> Completely Horizontal (Lying on Floor / Collapsed)
    """
    coords = keypoints[:, :2]
    confs = keypoints[:, 2]

    # Mid-hip
    if confs[11] > 0.2 and confs[12] > 0.2:
        mid_hip = (coords[11] + coords[12]) / 2.0
    elif confs[11] > 0.2:
        mid_hip = coords[11]
    elif confs[12] > 0.2:
        mid_hip = coords[12]
    else:
        return 0.0

    # Mid-shoulder
    if confs[5] > 0.2 and confs[6] > 0.2:
        mid_sh = (coords[5] + coords[6]) / 2.0
    elif confs[5] > 0.2:
        mid_sh = coords[5]
    elif confs[6] > 0.2:
        mid_sh = coords[6]
    else:
        return 0.0

    torso_vec = mid_sh - mid_hip  # Points upwards when standing: dy < 0
    dx, dy = torso_vec[0], torso_vec[1]
    norm = np.hypot(dx, dy)
    if norm < 1e-4:
        return 0.0

    # Vertical upward vector in image coordinates is (0, -1)
    cos_angle = (-dy) / norm
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def compute_motion_energy(kp_curr: np.ndarray, kp_prev: np.ndarray) -> float:
    """Compute average displacement of keypoints between two consecutive frames."""
    valid = (kp_curr[:, 2] > 0.3) & (kp_prev[:, 2] > 0.3)
    if not np.any(valid):
        return 0.0
    displacements = np.linalg.norm(kp_curr[valid, :2] - kp_prev[valid, :2], axis=1)
    return float(np.mean(displacements))
