import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tennis_sim import constants as C
from tennis_sim.env import TennisEnv
from tennis_sim.render import EpisodeRecorder
from tennis_sim.scripted_forehand import ScriptedForehand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/videos/smoke_demo.mp4")
    parser.add_argument("--camera", default="rally",
                        choices=["broadcast", "robot_front", "side", "behind_robot",
                                 "robot_close", "track_ball", "rally"])
    parser.add_argument("--seconds", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    env = TennisEnv(robot_pos=(6.5, 0.0, 0.79))
    env.max_time = args.seconds + 8.0
    env.reset(machine_pos=(-12.7, 0.4, 0.0), machine_yaw=0.0)
    policy = ScriptedForehand(env, serve_t=0.8, feed_speed=26.0)
    fired = {"done": False}

    def schedule(t, env):
        if t >= 0.8 and not fired["done"]:
            env.serve_ball(plan=policy.plan)
            fired["done"] = True

    rec = EpisodeRecorder(env, args.out, fps=50, width=args.width, height=args.height,
                          camera=args.camera)
    # Record until the ball has fully stopped after the robot's return (the
    # feed bounces once before the stroke, per tennis rules).
    obs = env.get_obs()
    hit = False
    rest_n = 0
    rest_xy = None
    max_steps = int((args.seconds + 8.0) / env.dt)
    for k in range(max_steps):
        t = k * env.dt
        schedule(t, env)
        obs, _, term, info = env.step(policy.act(obs))
        rec.on_step(env, obs, info)
        if any(e["type"] == "racket_hit" for e in info["events"]):
            hit = True
        if hit:
            v = float(np.linalg.norm(obs["ball_vel"]))
            w = float(np.linalg.norm(obs["ball_omega"]))
            grounded = obs["ball_pos"][2] < C.BALL_RADIUS * 2.5
            if v < 0.15 and w < 2.0 and grounded:
                rest_n += 1
            else:
                rest_n = 0
            if rest_n >= 40:
                break
        if term:
            break
    path, frames = rec.close()
    hits = [e for e in env.events if e["type"] == "racket_hit"]
    print("video: %s (%d frames) hits=%d" % (path, frames, len(hits)))


if __name__ == "__main__":
    main()
