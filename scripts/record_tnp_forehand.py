#!/usr/bin/env python
"""Re-record the successful TaskNPoint forehand rally.

Exact reconstruction of the winning configuration found during the sweep:
old racket mount (the project's natural 25-degree wrist-offset mount, no
training-mount patch / no roll), FixedGoalPolicy with the deterministic
force trigger, rise feed through the measured racket contact point,
frame 68 / speed 20 / margin 0.05, robot at the baseline.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tennis_sim.env import TennisEnv
from tennis_sim.render import EpisodeRecorder
from tennis_sim.scripted_forehand import plan_feed_for_contact, predict_ball_contact
from tennis_sim.tnp.policy import TnpPolicy, STATE_SWING, STATE_READY
from tennis_sim.tnp.robot_patch import patch_robot_to_tasknpoint

MACHINE_POS = (-12.7, -1.2, 0.0)


class FixedGoalPolicy(TnpPolicy):
    trigger_targets = None

    def _update_estimator(self, obs):
        from tennis_sim.tnp import ball_estimator as BE

        if not self.ball_launched:
            self.est_target_time = None
            self.est_target_pos = None
            return
        pos = obs["ball_pos"]
        vel = obs["ball_vel"]
        if self.physics == "sim":
            t, p = BE.predict_trajectory_sim(pos, vel, omega=obs.get("ball_omega"))
        else:
            t, p = BE.predict_trajectory_tasknpoint(pos, vel)
        if len(t) == 0:
            self.est_target_time = None
            self.est_target_pos = None
            return
        m = self.motion_idx
        tgt = (self.trigger_targets or {}).get(m, self.goal_pos_w[m])
        hit = BE.closest_approach(t, p, tgt)
        if hit is None:
            self.est_target_time = None
            self.est_target_pos = None
            return
        self.est_target_time = hit[0]
        self.est_target_pos = hit[1]

    def _maybe_trigger(self, obs):
        if self.state != STATE_READY or not self.ball_launched:
            return
        ttc = self.est_target_time
        if ttc is None:
            return
        threshold = self.time_to_contact[self.motion_idx] + self.trigger_margin
        if 0.0 <= ttc < threshold:
            self.state = STATE_SWING
            self.frame = 0
            self.swing_count += 1
            self.swing_goal_w = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/videos/tnp_forehand.mp4")
    ap.add_argument("--camera", default="robot_front")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    from tennis_sim import world as W

    scene_xml = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "assets", "scene_forehand.xml")
    W.write_scene(path=scene_xml)
    env = TennisEnv(scene_path=scene_xml)
    patch_robot_to_tasknpoint(env, verbose=False)
    policy = FixedGoalPolicy(env, trigger_margin=0.05, verbose=False)
    env.max_time = args.seconds + 6.0
    m_idx = 0

    env.reset(machine_pos=MACHINE_POS, machine_yaw=0.0)
    # airswing capture of the reference forehand (racket path, frame 68)
    obs = env.get_obs()
    policy.reset(obs)
    for _ in range(int(1.6 / env.dt)):
        obs, _, _, _ = env.step(policy.act(obs))
    policy.motion_idx = m_idx
    policy.force_trigger(m_idx)
    mo = policy.motions[policy.motion_names[m_idx]]
    pid = None
    import mujoco

    pid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    P = []
    for k in range(mo["num_frames"] + 5):
        sid = env.ball_ctrl.racket_site
        P.append(env.data.site_xpos[sid].copy())
        obs, _, term, _ = env.step(policy.act(obs))
        if term:
            break
    P = np.array(P)
    f = 68
    contact = P[min(f, len(P) - 1)].copy() + np.array([-0.08, 0.0, -0.02])
    print("[fh] frame 68 racket contact target:", np.round(contact, 3))

    # deterministic force-trigger time
    ok = False
    info_evt = {"all_events": []}
    for attempt in range(4):
        try:
            best = None
            plan = None
            c = None
            lo, hi = 9.0, 30.0
            for i in range(8):
                spd = 0.5 * (lo + hi) if i else 20.0
                try:
                    p_i = plan_feed_for_contact(env, contact, speed=spd, spin=2000.0)
                    c_i = predict_ball_contact(p_i, target_z=contact[2])
                except ValueError:
                    lo = spd
                    continue
                if c_i is None:
                    lo = spd
                    continue
                err = c_i["t"] - 1.35
                if best is None or abs(err) < abs(best[0]):
                    best, plan, c = (err, p_i, c_i), p_i, c_i
                if abs(err) < 0.05:
                    break
                if err < 0:
                    hi = spd
                else:
                    lo = spd
            if plan is None:
                raise RuntimeError("no feed")
        except RuntimeError as exc:
            print("  feed failed:", exc)
            contact[2] += 0.1
            continue

        def episode(record):
            nonlocal ok, info_evt, racket_at_f, ball_at_f
            env.reset(machine_pos=MACHINE_POS, machine_yaw=0.0)
            obs = env.get_obs()
            policy.reset(obs)
            policy.motion_idx = m_idx
            policy.time_to_contact[m_idx] = f / 50.0
            policy.trigger_targets = {m_idx: contact.copy()}
            rec = EpisodeRecorder(env, args.out, fps=50, width=1920, height=1080,
                                  camera=args.camera) if record else None
            served = {"done": False}
            rest_n = 0
            hit = False
            racket_at_f = None
            ball_at_f = None
            for k in range(int((args.seconds + 6.0) / env.dt)):
                if not served["done"] and k * env.dt >= 2.2:
                    served["done"] = True
                    env.serve_ball(plan=plan)
                    policy.notify_launch()
                    print("  serve t=%.2f speed=%.1f" % (k * env.dt, plan["speed"]))
                pre_frame = policy.frame if policy.state == STATE_SWING else None
                obs, _, term, info_evt = env.step(policy.act(obs))
                if rec is not None:
                    rec.on_step(env, obs, info_evt)
                if policy.state == STATE_SWING and pre_frame == f and racket_at_f is None:
                    sid = env.ball_ctrl.racket_site
                    racket_at_f = env.data.site_xpos[sid].copy()
                    ball_at_f = obs["ball_pos"].copy()
                if any(e["type"] == "racket_hit" for e in info_evt["events"]):
                    hit = True
                if hit:
                    v = float(np.linalg.norm(obs["ball_vel"]))
                    w = float(np.linalg.norm(obs["ball_omega"]))
                    grounded = obs["ball_pos"][2] < 0.09
                    rest_n = rest_n + 1 if (v < 0.15 and w < 2.0 and grounded) else 0
                    if rest_n >= 40:
                        break
                if term:
                    break
            if rec is not None:
                rec.close()
            all_events = info_evt["all_events"]
            hits = [e for e in all_events if e["type"] == "racket_hit"]
            bounces = [e for e in all_events if e["type"] == "bounce"]
            far_in = [b for b in bounces
                      if b["zone"]["side"] == "far" and b["zone"]["in"]]
            ok = bool(hits) and len(far_in) >= 1
            print("[fh] attempt: hit=%s hits=%d far_in=%d" % (
                bool(hits), len(hits), len(far_in)))
            for e in hits:
                print("  racket_hit t=%.2f v_in=%.1f v_out=%.1f racket=%.1f" % (
                    e["t"], e["ball_speed_in"], e["ball_speed_out"], e["racket_speed"]))

        racket_at_f = None
        ball_at_f = None
        episode(record=False)
        if ok:
            episode(record=True)
            print("[fh] SUCCESS -- video:", args.out)
            return
        if racket_at_f is None or ball_at_f is None:
            break
        miss = ball_at_f - racket_at_f
        print("  miss:", np.round(miss, 3))
        contact = contact - 0.8 * miss
    print("[fh] no full rally; best attempt kept in", args.out)


if __name__ == "__main__":
    main()
