import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio

import numpy as np

import mujoco

from tennis_sim.camera import set_free_cam


class Recorder:
    def __init__(self, model, path, fps=50, width=1920, height=1080, camera="broadcast"):
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

    def _update_cam(self, data, ball_pos=None, robot_pos=None):
        mode = self.camera_mode
        if mode == "broadcast":
            set_free_cam(self.cam, (-11.0, 15.0, 5.5), (0.5, 0.0, 1.0))
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

    def add_frame(self, data, ball_pos=None, robot_pos=None):
        self._update_cam(data, ball_pos, robot_pos)
        self.renderer.update_scene(data, camera=self.cam)
        img = self.renderer.render()
        self.writer.append_data(img)
        self.frames += 1

    def close(self):
        self.writer.close()
        self.renderer.close()
        return self.path, self.frames


class EpisodeRecorder:
    def __init__(self, env, path, fps=50, width=1920, height=1080, camera="broadcast"):
        self.env = env
        self.recorder = Recorder(env.model, path, fps=fps, width=width, height=height,
                                 camera=camera)
        self.frame_every = int(round(1.0 / fps / env.dt))
        self.k = 0

    def on_step(self, env, obs, info):
        if self.k % self.frame_every == 0:
            self.recorder.add_frame(env.data, ball_pos=obs["ball_pos"],
                                    robot_pos=obs["pelvis_pos"][:2])
        self.k += 1

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
