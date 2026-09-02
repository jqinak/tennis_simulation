import os
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio

import numpy as np

import mujoco

from tennis_sim.camera import set_free_cam

# Comet tail (render-only, zero physics impact). Distinct colors for the
# outbound feed ("going", toward the robot at +x) and the return after the
# robot's hit (back toward the machine at -x).
TRAIL_COLOR_GOING = (0.20, 0.75, 1.00)
TRAIL_COLOR_RETURN = (1.00, 0.45, 0.05)
TRAIL_MAX_AGE = 1.5
TRAIL_MAX_LEN = 5.0
TRAIL_MIN_SPEED = 1.2
TRAIL_WIDTH_HEAD = 0.036
TRAIL_WIDTH_TAIL = 0.007
TRAIL_ALPHA_HEAD = 0.80
TRAIL_ALPHA_TAIL = 0.05
TRAIL_HEAD_RADIUS = 0.050
TRAIL_HEAD_ALPHA = 0.25


class BallTrail:
    """Stores recent ball samples and injects fading comet-tail geoms into the
    MuJoCo render scene. Purely a visualization overlay."""

    def __init__(self):
        self.samples = deque()  # (t, pos(3,), kind)
        self.kind = 0
        self._last_pos = None

    def reset(self):
        self.samples.clear()
        self._last_pos = None

    def add(self, t, pos, vel=None):
        pos = np.asarray(pos, dtype=float)
        if self._last_pos is not None and float(np.linalg.norm(pos - self._last_pos)) > 3.0:
            self.reset()
        if vel is not None:
            vx = float(vel[0])
            if vx > 0.15:
                self.kind = 0
            elif vx < -0.15:
                self.kind = 1
        elif self._last_pos is not None:
            step = pos - self._last_pos
            if float(np.linalg.norm(step)) > 0.05:
                self.kind = 0 if step[0] > 0.0 else 1
        self._last_pos = pos.copy()
        speed = float(np.linalg.norm(vel)) if vel is not None else 0.0
        if speed >= TRAIL_MIN_SPEED:
            self.samples.append((float(t), pos.copy(), self.kind))
        while self.samples and (
                t - self.samples[0][0] > TRAIL_MAX_AGE
                or float(np.linalg.norm(self.samples[0][1] - pos)) > TRAIL_MAX_LEN):
            self.samples.popleft()

    def geoms(self):
        """(p0, p1, width, rgba) capsule segments, oldest first."""
        n = len(self.samples)
        if n < 2:
            return
        pts = list(self.samples)
        for i in range(n - 1):
            t0, p0, k0 = pts[i]
            _, p1, _ = pts[i + 1]
            u = i / max(1, n - 1)
            u = u ** 1.2
            width = TRAIL_WIDTH_TAIL + (TRAIL_WIDTH_HEAD - TRAIL_WIDTH_TAIL) * u
            alpha = TRAIL_ALPHA_TAIL + (TRAIL_ALPHA_HEAD - TRAIL_ALPHA_TAIL) * u
            base = TRAIL_COLOR_GOING if k0 == 0 else TRAIL_COLOR_RETURN
            yield p0, p1, width, (base[0], base[1], base[2], alpha)
        t_last, p_last, k_last = pts[-1]
        base = TRAIL_COLOR_GOING if k_last == 0 else TRAIL_COLOR_RETURN
        yield p_last, p_last, TRAIL_HEAD_RADIUS, (base[0], base[1], base[2], TRAIL_HEAD_ALPHA)


class Recorder:
    def __init__(self, model, path, fps=50, width=1920, height=1080, camera="broadcast",
                 trail=True):
        self.model = model
        self.path = path
        self.fps = fps
        self.camera_mode = camera
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_params=["-profile:v", "high", "-movflags", "+faststart"])
        self.frames = 0
        self.trail = BallTrail() if trail else None
        self._last_t = None

    def _update_cam(self, data, ball_pos=None, robot_pos=None):
        mode = self.camera_mode
        if mode == "broadcast":
            # same view direction as the original broadcast angle, pulled back
            # so the whole court (all four baselines/sidelines) stays in frame
            set_free_cam(self.cam, (-14.45, 19.5, 6.85), (0.5, 0.0, 1.0))
        elif mode == "robot_front":
            # follow camera right in front of the robot, keeping the strokes
            # large and clear in frame
            base = robot_pos if robot_pos is not None else (10.9, 0.0)
            set_free_cam(self.cam, (base[0] - 3.4, base[1] - 0.4, 1.6),
                         (base[0] - 0.1, base[1] + 0.1, 0.95))
        elif mode == "side":
            set_free_cam(self.cam, (-9.0, 14.5, 2.4), (0.0, 0.0, 1.0))
        elif mode == "behind_robot":
            set_free_cam(self.cam, (17.0, 0.0, 3.0), (-3.0, 0.0, 1.0))
        elif mode == "robot_close":
            base = robot_pos if robot_pos is not None else (10.9, 0.0)
            set_free_cam(self.cam, (base[0] + 2.6, base[1] - 4.2, 2.0),
                         (base[0], base[1], 1.0))
        elif mode == "rally":
            base = robot_pos if robot_pos is not None else (6.5, 0.0)
            set_free_cam(self.cam, (base[0] - 4.0, base[1] - 9.0, 2.2),
                         (base[0] + 0.3, base[1] + 0.3, 1.0))
        elif mode == "track_ball" and ball_pos is not None:
            set_free_cam(self.cam, (-4.0, -13.0, 5.0),
                         (0.6 * ball_pos[0], 0.6 * ball_pos[1], ball_pos[2]))
        else:
            set_free_cam(self.cam, (-11.0, 15.0, 5.5), (0.5, 0.0, 1.0))

    def _inject_trail(self):
        if self.trail is None:
            return
        scn = self.renderer.scene
        for p0, p1, width, rgba in self.trail.geoms():
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            if np.linalg.norm(np.asarray(p1) - np.asarray(p0)) < 1e-6:
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([width, 0.0, 0.0]), np.asarray(p0, dtype=float),
                                    np.zeros(9), np.asarray(rgba, dtype=float))
            else:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(width),
                                     np.asarray(p0, dtype=float), np.asarray(p1, dtype=float))
                g.rgba[:] = rgba
            scn.ngeom += 1
    def add_frame(self, data, ball_pos=None, ball_vel=None, robot_pos=None):
        self._update_cam(data, ball_pos, robot_pos)
        self.renderer.update_scene(data, camera=self.cam)
        if self.trail is not None and ball_pos is not None:
            if self._last_t is not None and data.time < self._last_t:
                self.trail.reset()
            self._last_t = data.time
            if not self.trail.samples or self.trail.samples[-1][0] != data.time:
                self.trail.add(data.time, ball_pos, ball_vel)
            self._inject_trail()
        img = self.renderer.render()
        self.writer.append_data(img)
        self.frames += 1

    def close(self):
        self.writer.close()
        self.renderer.close()
        return self.path, self.frames


class EpisodeRecorder:
    def __init__(self, env, path, fps=50, width=1920, height=1080, camera="broadcast",
                 trail=True):
        self.env = env
        self.recorder = Recorder(env.model, path, fps=fps, width=width, height=height,
                                 camera=camera, trail=trail)
        self.frame_every = int(round(1.0 / fps / env.dt))
        self.k = 0

    def on_step(self, env, obs, info):
        if self.trail_sample():
            self.recorder.trail.add(env.data.time, obs["ball_pos"], obs["ball_vel"])
        if self.k % self.frame_every == 0:
            self.recorder.add_frame(env.data, ball_pos=obs["ball_pos"],
                                    ball_vel=obs["ball_vel"],
                                    robot_pos=obs["pelvis_pos"][:2])
        self.k += 1

    def trail_sample(self):
        return self.recorder.trail is not None

    def close(self):
        return self.recorder.close()


def record_scenes(env, path_prefix, seconds_per_scene=3.0, camera="broadcast", fps=50,
                  action=None, serve_schedule=None):
    rec = EpisodeRecorder(env, path_prefix, fps=fps, camera=camera)
    out = env.run_episode(seconds_per_scene, policy=(action if action is not None else
                                                     (lambda obs: obs["qpos"])),
                          serve_schedule=serve_schedule, on_step=rec.on_step)
    info = rec.close()
    return info
