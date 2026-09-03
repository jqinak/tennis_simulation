"""TaskNPoint user-stroke motion goals and deployment constants.

Values extracted from the read-only tasknpoint repo:
- motion_lib.py entries user_forehand / user_volley / user_backhand
  (goal means verified numerically against the shipped npz probe frames)
- ONNX metadata (assets/tnp/tnp_metadata.json): joint order, PD gains,
  action scales, default pose, target phases.

Frames: goal means are expressed in the reference motion's initial pelvis
frame (= the anchor frame used by the trained command sampler).
"""
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(_ROOT, "assets", "tnp")
ONNX_PATH = os.path.join(ASSET_DIR, "policy.onnx")
METADATA_PATH = os.path.join(ASSET_DIR, "tnp_metadata.json")

MOTION_ORDER = ["user_forehand", "user_volley", "user_backhand"]

# motion_lib.py user_strokes entries (probe contact phase, goal means)
MOTION_GOALS = {
    "user_forehand": {
        "contact_phase": 0.514,
        "pos_mean": np.array([-0.0138, -1.3894, 0.0955]),
        "vel_mean": np.array([1.0, 0.0, 0.1]),
        "ori_mean_rpy": np.array([1.0038, 0.4774, -1.1247]),
    },
    "user_volley": {
        "contact_phase": 0.455,
        "pos_mean": np.array([0.2334, 1.5483, 0.6353]),
        "vel_mean": np.array([1.0, 0.0, 0.1]),
        "ori_mean_rpy": np.array([-0.6375, 0.2098, 2.0967]),
    },
    "user_backhand": {
        "contact_phase": 0.476,
        "pos_mean": np.array([0.7557, 0.8593, 0.0323]),
        "vel_mean": np.array([1.0, 0.0, 0.1]),
        "ori_mean_rpy": np.array([1.2601, 0.7908, 1.9373]),
    },
}

# Deploy-time constant (deploy/simulation: contact_duration default 0.3)
CONTACT_DURATION_PHASE = 0.3
# Deploy sim node: closest-approach cutoff (m^2) for a valid ball target
BALL_TARGET_CUTOFF_SQ = 0.5

# Unitree G1 effort limits per joint (order = ONNX joint_names), N*m
EFFORT_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,          # left leg
    88, 139, 88, 139, 50, 50,          # right leg
    88, 50, 50,                        # waist yaw/roll/pitch
    25, 25, 25, 25, 25, 5, 5,          # left arm
    25, 25, 25, 25, 25, 5, 5,          # right arm
], dtype=float)

# Training-time waist roll/pitch were clamped to +-0.001 rad
WAIST_ROLL_PITCH_LIMIT = 0.001


def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)["onnx"]


def load_motions():
    """Load the three reference motions: dict name -> arrays."""
    motions = {}
    for name in MOTION_ORDER:
        d = np.load(os.path.join(ASSET_DIR, "motions", name + ".npz"))
        motions[name] = {
            "fps": float(d["fps"].item()),
            "joint_pos": d["joint_pos"].astype(np.float64),
            "joint_vel": d["joint_vel"].astype(np.float64),
            "body_quat_w": d["body_quat_w"].astype(np.float64),
            "num_frames": int(d["joint_pos"].shape[0]),
        }
    return motions
