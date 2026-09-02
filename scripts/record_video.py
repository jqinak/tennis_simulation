import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_sim.env import TennisEnv
from tennis_sim.render import EpisodeRecorder
from tennis_sim.scripted_forehand import ScriptedForehand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/videos/smoke_demo.mp4")
    parser.add_argument("--camera", default="rally",
                        choices=["broadcast", "side", "behind_robot", "robot_close",
                                 "track_ball", "rally"])
    parser.add_argument("--seconds", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    env = TennisEnv(robot_pos=(6.5, 0.0, 0.79))
    env.max_time = args.seconds + 2.0
    env.reset(machine_pos=(-12.7, 0.4, 0.0), machine_yaw=0.0)
    policy = ScriptedForehand(env, serve_t=0.8, feed_speed=26.0)
    fired = {"done": False}

    def schedule(t, env):
        if t >= 0.8 and not fired["done"]:
            env.serve_ball(plan=policy.plan)
            fired["done"] = True

    rec = EpisodeRecorder(env, args.out, fps=50, width=args.width, height=args.height,
                          camera=args.camera)
    out = env.run_episode(args.seconds, policy=policy, serve_schedule=schedule,
                          on_step=rec.on_step)
    path, frames = rec.close()
    hits = [e for e in env.events if e["type"] == "racket_hit"]
    print("video: %s (%d frames) hits=%d" % (path, frames, len(hits)))


if __name__ == "__main__":
    main()
