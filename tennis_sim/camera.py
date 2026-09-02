import numpy as np


def set_free_cam(cam, pos, lookat, up_hint=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=float)
    lookat = np.asarray(lookat, dtype=float)
    offset = pos - lookat
    dist = float(np.linalg.norm(offset))
    az = float(np.degrees(np.arctan2(offset[1], offset[0]))) + 180.0
    el = -float(np.degrees(np.arcsin(offset[2] / max(dist, 1e-9))))
    cam.type = 0
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    return cam


CAMERA_PRESETS = {
    "wide": {"pos": (-15.5, -17.5, 9.0), "lookat": (0.0, 0.0, 0.9)},
    "side": {"pos": (-11.0, 14.5, 2.4), "lookat": (0.0, 0.0, 1.0)},
    "behind_machine": {"pos": (-17.0, 0.0, 2.8), "lookat": (3.0, 0.0, 1.0)},
    "behind_robot": {"pos": (17.5, 0.0, 3.2), "lookat": (-3.0, 0.0, 1.0)},
    "robot_close": {"pos": (13.5, -4.5, 1.9), "lookat": (10.8, -0.4, 1.1)},
    "net_front": {"pos": (-1.2, -6.5, 1.7), "lookat": (9.0, 0.0, 1.0)},
    "rally": {"pos": (2.5, -9.0, 2.2), "lookat": (6.8, 0.3, 1.0)},
}
