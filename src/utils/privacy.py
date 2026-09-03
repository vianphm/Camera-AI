"""Privacy preservation and face anonymization utility."""

from typing import Optional, List, Tuple
import cv2
import numpy as np


class FaceBlurrer:
    """Anonymizes faces in video frames using detected keypoints or bounding boxes."""

    def __init__(self, kernel_size: int = 31, sigma: float = 15.0) -> None:
        """Initialize FaceBlurrer.

        Args:
            kernel_size: Gaussian blur kernel size (must be odd positive integer).
            sigma: Gaussian standard deviation.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma = sigma

    def blur_keypoints_face(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        head_kp_indices: Optional[List[int]] = None,
        padding_ratio: float = 0.4,
    ) -> np.ndarray:
        """Blur face region inferred from facial keypoints (Nose, Eyes, Ears: 0..4 in COCO).

        Args:
            frame: Input BGR image (H, W, 3).
            keypoints: Array of shape (17, 3) where each row is (x, y, conf).
            head_kp_indices: Indices for head keypoints. Defaults to [0, 1, 2, 3, 4].
            padding_ratio: Expansion around head keypoints bounding box.

        Returns:
            Frame with blurred face region.
        """
        if head_kp_indices is None:
            head_kp_indices = [0, 1, 2, 3, 4]  # Nose, L-Eye, R-Eye, L-Ear, R-Ear

        h, w = frame.shape[:2]
        valid_pts = []
        for idx in head_kp_indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > 0.3 and 0 <= x < w and 0 <= y < h:
                    valid_pts.append((x, y))

        if not valid_pts:
            return frame

        pts_np = np.array(valid_pts, dtype=np.float32)
        min_x, min_y = np.min(pts_np, axis=0)
        max_x, max_y = np.max(pts_np, axis=0)

        # Apply padding
        box_w = max_x - min_x
        box_h = max_y - min_y
        pad_x = max(box_w * padding_ratio, 15.0)
        pad_y = max(box_h * padding_ratio, 15.0)

        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(w, int(max_x + pad_x))
        y2 = min(h, int(max_y + pad_y))

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), self.sigma)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame

    def blur_bbox_head(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        head_height_ratio: float = 0.22,
    ) -> np.ndarray:
        """Fallback blurring when keypoints are occluded: blurs top portion of person bbox."""
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        x1 = max(0, int(bx1))
        y1 = max(0, int(by1))
        x2 = min(w, int(bx2))
        head_h = (by2 - by1) * head_height_ratio
        y2 = min(h, int(by1 + head_h))

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (self.kernel_size, self.kernel_size), self.sigma)
            frame[y1:y2, x1:x2] = blurred_roi

        return frame
