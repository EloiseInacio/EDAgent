import numpy as np

from config import SKELETON_CONF_THRESHOLD, SKELETON_MIN_JOINTS


def load_skeleton_npz(npz_path: str) -> dict:
    """
    Load skeleton data from a .npz file.

    Expected keys:
      - keypoints : (n_frames, 17, 3)  — x, y, confidence per joint
      - boxes     : (n_frames, 4)       — bounding box per frame [x1, y1, x2, y2]
      - fps       : scalar float
      - width     : scalar int
      - height    : scalar int
    """
    data = np.load(npz_path, allow_pickle=True)
    return data


def compute_duration(keypoints: np.ndarray, fps: float) -> float:
    """
    Duration of the skeleton sequence in seconds.

    Args:
        keypoints: (n_frames, 17, 3)
        fps: frames per second

    Returns:
        duration_seconds: float
    """
    n_frames = keypoints.shape[0]
    return n_frames / fps


def compute_visibility(keypoints: np.ndarray, conf_threshold: float = 0.3) -> float:
    """
    Mean number of visible joints per frame.

    A joint is considered visible when its confidence score >= conf_threshold.

    Args:
        keypoints: (n_frames, 17, 3)  — last channel is confidence
        conf_threshold: minimum confidence to count a joint as visible

    Returns:
        mean_visible_joints: float
    """
    confidences = keypoints[..., 2]          # (n_frames, 17)
    visible_per_frame = (confidences >= conf_threshold).sum(axis=1)  # (n_frames,)
    return float(visible_per_frame.mean())


def compute_trace_length(
    keypoints: np.ndarray,
    conf_threshold: float = SKELETON_CONF_THRESHOLD,
    min_joints: int = SKELETON_MIN_JOINTS,
) -> int:
    """
    Max number of consecutive frames where at least `min_joints` keypoints
    are above `conf_threshold`.

    Args:
        keypoints: (n_frames, 17, 3)
        conf_threshold: minimum confidence to count a joint as visible
        min_joints: minimum number of visible joints required to qualify a frame

    Returns:
        trace_length: int — length of the longest qualifying run
    """
    confidences = keypoints[..., 2]                          # (n_frames, 17)
    qualified = (confidences >= conf_threshold).sum(axis=1) >= min_joints  # (n_frames,) bool

    max_run = 0
    current_run = 0
    for q in qualified:
        if q:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return max_run


def compute_avg_keypoint_confidence(keypoints: np.ndarray) -> float:
    """
    Average confidence score across all joints and all frames.

    Args:
        keypoints: (n_frames, 17, 3)

    Returns:
        avg_confidence: float
    """
    return float(keypoints[..., 2].mean())


def compute_confidence_variance(keypoints: np.ndarray) -> float:
    """
    Variance of confidence scores across all joints and all frames.

    Args:
        keypoints: (n_frames, 17, 3)

    Returns:
        conf_variance: float
    """
    return float(keypoints[..., 2].var())


# COCO 17-joint limb definitions
_LIMBS = {
    "left_shoulder_elbow":  (5, 7),
    "right_shoulder_elbow": (6, 8),
    "left_hip_knee":        (11, 13),
    "right_hip_knee":       (12, 14),
}


def compute_joint_velocity_stats(
    keypoints: np.ndarray,
    conf_threshold: float = SKELETON_CONF_THRESHOLD,
) -> dict:
    """
    Mean, max, and variance of joint velocity (px/frame) across all frames and joints.

    Velocity for joint j at frame t = Euclidean distance between (x_t, y_t) and
    (x_{t-1}, y_{t-1}). Only included when the joint confidence is >= conf_threshold
    in both consecutive frames to avoid spurious displacements from occluded joints.

    Args:
        keypoints: (n_frames, 17, 3)
        conf_threshold: minimum confidence in both frames to include a joint

    Returns:
        dict with keys: velocity_mean, velocity_max, velocity_variance
        Returns NaN for all values when no valid joint transitions exist.
    """
    xy   = keypoints[..., :2]   # (n_frames, 17, 2)
    conf = keypoints[..., 2]    # (n_frames, 17)

    # Both frames in a transition must be above threshold
    both_visible = (conf[:-1] >= conf_threshold) & (conf[1:] >= conf_threshold)  # (n_frames-1, 17)

    delta = xy[1:] - xy[:-1]                                    # (n_frames-1, 17, 2)
    speed = np.linalg.norm(delta, axis=-1)                      # (n_frames-1, 17)

    valid_speeds = speed[both_visible]

    if valid_speeds.size == 0:
        nan = float("nan")
        return {"velocity_mean": nan, "velocity_max": nan, "velocity_variance": nan}

    return {
        "velocity_mean":     float(valid_speeds.mean()),
        "velocity_max":      float(valid_speeds.max()),
        "velocity_variance": float(valid_speeds.var()),
    }


def compute_bbox_area_ratio(
    bboxes: np.ndarray,
    width: int,
    height: int,
) -> float:
    """
    Mean ratio of bounding-box area to frame area across all frames.

    Assumes bboxes are in [x1, y1, x2, y2] format (absolute pixel coords).

    Args:
        bboxes: (n_frames, 4) — [x1, y1, x2, y2] in absolute pixel coords
        width:  frame width in pixels
        height: frame height in pixels

    Returns:
        mean_bbox_area_ratio: float in [0, 1]
    """
    frame_area = width * height
    if frame_area == 0:
        return float("nan")

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bbox_areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    return float((bbox_areas / frame_area).mean())


def compute_limb_length_variance(
    keypoints: np.ndarray,
    conf_threshold: float = SKELETON_CONF_THRESHOLD,
) -> dict:
    """
    Variance of limb lengths (in pixels) across frames for four limbs:
      left_shoulder_elbow, right_shoulder_elbow, left_hip_knee, right_hip_knee.

    A frame is included for a given limb only when both endpoint joints have
    confidence >= conf_threshold.

    Args:
        keypoints: (n_frames, 17, 3)
        conf_threshold: minimum confidence for both endpoints

    Returns:
        dict mapping limb name -> variance (float). NaN when fewer than 2
        valid frames exist for that limb.
    """
    xy   = keypoints[..., :2]   # (n_frames, 17, 2)
    conf = keypoints[..., 2]    # (n_frames, 17)

    result = {}
    for name, (j_a, j_b) in _LIMBS.items():
        both_visible = (conf[:, j_a] >= conf_threshold) & (conf[:, j_b] >= conf_threshold)
        if both_visible.sum() < 2:
            result[name] = float("nan")
            continue
        lengths = np.linalg.norm(xy[both_visible, j_a] - xy[both_visible, j_b], axis=-1)
        result[name] = float(lengths.var())

    return result
