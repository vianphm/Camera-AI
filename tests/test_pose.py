"""Unit tests for pose keypoints normalization and geometry."""

import numpy as np
import pytest
from src.pose.keypoints import normalize_keypoints, compute_torso_angle, compute_motion_energy


def test_normalize_keypoints_shape():
    dummy_kp = np.zeros((17, 3), dtype=np.float32)
    # Set hips and shoulders
    dummy_kp[5] = [90.0, 100.0, 0.9]   # L Shoulder
    dummy_kp[6] = [110.0, 100.0, 0.9]  # R Shoulder
    dummy_kp[11] = [95.0, 200.0, 0.9]  # L Hip
    dummy_kp[12] = [105.0, 200.0, 0.9] # R Hip

    norm_kp = normalize_keypoints(dummy_kp)
    assert norm_kp.shape == (17, 3)
    # Hip root should be near (0, 0)
    mid_hip = (norm_kp[11, :2] + norm_kp[12, :2]) / 2.0
    assert pytest.approx(mid_hip[0], 0.01) == 0.0
    assert pytest.approx(mid_hip[1], 0.01) == 0.0


def test_compute_torso_angle_standing_vs_lying():
    # Standing pose: Shoulder above Hip (dy < 0 in image coords)
    kp_standing = np.zeros((17, 3), dtype=np.float32)
    kp_standing[5] = [100.0, 100.0, 0.9]  # Shoulder
    kp_standing[6] = [100.0, 100.0, 0.9]
    kp_standing[11] = [100.0, 200.0, 0.9] # Hip
    kp_standing[12] = [100.0, 200.0, 0.9]

    angle_standing = compute_torso_angle(kp_standing)
    assert pytest.approx(angle_standing, abs=2.0) == 0.0

    # Lying pose: Shoulder and Hip at same height, separated horizontally (dy = 0)
    kp_lying = np.zeros((17, 3), dtype=np.float32)
    kp_lying[5] = [200.0, 200.0, 0.9]  # Shoulder to right
    kp_lying[6] = [200.0, 200.0, 0.9]
    kp_lying[11] = [100.0, 200.0, 0.9] # Hip to left
    kp_lying[12] = [100.0, 200.0, 0.9]

    angle_lying = compute_torso_angle(kp_lying)
    assert pytest.approx(angle_lying, abs=2.0) == 90.0


def test_compute_motion_energy():
    kp1 = np.ones((17, 3), dtype=np.float32)
    kp2 = np.ones((17, 3), dtype=np.float32)
    # Identical poses -> 0 motion energy
    assert compute_motion_energy(kp1, kp2) == 0.0

    # Displaced poses
    kp2[:, :2] += 5.0
    energy = compute_motion_energy(kp2, kp1)
    assert pytest.approx(energy, 0.01) == np.hypot(5.0, 5.0)
