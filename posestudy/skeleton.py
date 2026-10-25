"""COCO-17 keypoint definitions and joint-angle helpers.

Only depends on NumPy, so it can be used for analysis without MMPose installed.
"""

import numpy as np

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KPT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Pairs of keypoint names joined by a bone when drawing the skeleton.
SKELETON = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"), ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
]
SKELETON_IDX = [(KPT_INDEX[a], KPT_INDEX[b]) for a, b in SKELETON]

# Joint angle = interior angle at the middle keypoint (degrees, 0-180).
JOINT_ANGLES = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

# BGR colours: left side orange, right side blue, centre green.
def kpt_color(name):
    if name.startswith("left"):
        return (0, 140, 255)
    if name.startswith("right"):
        return (255, 120, 0)
    return (0, 200, 0)


def angle_between(a, b, c):
    """Interior angle ABC in degrees. a, b, c are (..., 2) arrays."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    v1, v2 = a - b, c - b
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_joint_angles(keypoints, angle_defs=JOINT_ANGLES):
    """keypoints: (T, 17, 2) -> dict angle_name -> (T,) array of degrees."""
    kp = np.asarray(keypoints, dtype=float)
    return {
        name: angle_between(kp[..., KPT_INDEX[a], :], kp[..., KPT_INDEX[b], :],
                            kp[..., KPT_INDEX[c], :])
        for name, (a, b, c) in angle_defs.items()
    }
