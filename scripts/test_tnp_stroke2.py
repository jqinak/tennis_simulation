#!/usr/bin/env python
"""TaskNPoint stroke deployment, fork-success recipe + tennis-legal bounce feed.

Uses the FULL TnpPolicy (estimator trigger on the nominal contact goal +
frozen-intercept tracking, exactly like the successful forehand run in
tennis_simulation_policy), with:
  - the racket mount aligned to the TNP training pose at runtime
  - the robot actuators patched to the training gains
  - the machine feed planned to BOUNCE ONCE on the robot side and rise
    through the (offset) contact goal -- the receiver never volleys
A grid over the goal offset / margin is searched per stroke until the return
lands in the far court.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tennis_sim.env import TennisEnv
from tennis_sim.render import EpisodeRecorder
from tennis_sim.scripted_forehand import plan_feed_for_contact, predict_ball_contact
from tennis_sim.tnp.policy import TnpPolicy
from tennis_sim.tnp.robot_patch import (patch_robot_to_tasknpoint,
                                        patch_racket_to_tasknpoint,
                                        patch_feet_to_tasknpoint)
from scripts.test_tnp_stroke import solve_feed_apex

MACHINE_POS = (-12.7, -1.2, 0.0)
STROKE_IDX = {"forehand": 0, "volley": 1, "backhand": 2}


def run_once_rest(args, env, policy, m_idx, plan, t_arrive, record=False):
    env.reset(machine_pos=MACHINE_POS, machine_yaw=0.0)
    obs = env.get_obs()
    policy.reset(obs)
    policy.motion_idx = m_idx
    contact = policy.goal_pos_w[m_idx].copy()

    rec = EpisodeRecorder(env, args.out, fps=50, width=1920, height=1080,
                          camera=args.camera) if record else None
    served = {"done": False}
    rest_n = 0
    hit = False
    info_evt = {"all_events": []}
    for k in range(int((args.seconds + 6.0) / env.dt)):
        t = k * env.dt
        if not served["done"] and t >= args.serve_t:
            served["done"] = True
            env.serve_ball(plan=plan)
            policy.notify_launch()
        obs, _, term, info_evt = env.step(policy.act(obs))
        if rec is not None:
            rec.on_step(env, obs, info_evt)
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
    net = [e for e in all_events if e["type"] == "net_hit"]
    far_in = [b for b in bounces if b["zone"]["side"] == "far" and b["zone"]["in"]]
    info = {"hits": len(hits), "far_in": len(far_in), "net": len(net)}
    ok = bool(hits) and len(far_in) >= 1 and not net
    print("  -> hit=%s hits=%d far_in=%d net=%d" % (
        bool(hits), len(hits), len(far_in), len(net)))
    for e in hits:
        print("     racket_hit t=%.2f v_in=%.1f v_out=%.1f racket=%.1f" % (
            e["t"], e["ball_speed_in"], e["ball_speed_out"], e["racket_speed"]))
    for b in bounces[:4]:
        print("     bounce t=%.2f at %s %s" % (b["t"], np.round(b["pos"], 2), b["zone"]))
    return ok, info


def run_once(args, env, policy, m_idx, contact, record=False):
    env.reset(machine_pos=(MACHINE_POS[0], args.machine_y, 0.0), machine_yaw=0.0)
    try:
        if args.pass_phase == "direct":
            # fork-proven bisect: feed speed so the flight time hits
            # args.arrival_t (the successful run used 1.45 s)
            best = None
            plan = None
            lo, hi = 13.0, 30.0
            for i in range(8):
                spd = 0.5 * (lo + hi) if i else args.speed
                try:
                    p_i = env.machine.serve_through_point(
                        contact, speed=spd, spin_rpm=args.spin, spin_topspin=True)
                except ValueError:
                    lo = spd
                    continue
                err = p_i["flight_time"] - args.arrival_t
                if best is None or abs(err) < abs(best[0]):
                    best = (err, p_i)
                if abs(err) < 0.03:
                    break
                if err < 0:
                    lo = spd
                else:
                    hi = spd
            if best is None:
                print("  feed failed: no through-point solution")
                return False, {"hits": 0, "far_in": 0, "net": 0}
            plan = best[1]
            t_arrive = args.serve_t + plan["flight_time"]
            env.serve_ball(plan=plan)  # position the ball (re-reset below)
            return run_once_rest(args, env, policy, m_idx, plan, t_arrive, record)
        if args.pass_phase == "fall":
            hi = contact.copy()
            hi[2] += 0.15
            plan, c = solve_feed_apex(env, hi, spin=args.spin,
                                      speed0=args.speed, arrival_t=args.arrival_t,
                                      verbose=False)
            # ball falls back through the contact height ~0.17 s after apex
            c = dict(c)
            c["t"] = c["t"] + 0.17
            c["pos"] = contact.copy()
        else:
            try:
                plan, c = solve_feed_apex(env, contact, spin=args.spin,
                                          speed0=args.speed,
                                          arrival_t=args.arrival_t, verbose=False)
            except RuntimeError:
                # no flat solution (e.g. high contact): rise through it instead
                plan = plan_feed_for_contact(env, contact, speed=args.speed,
                                             spin=args.spin)
                c = predict_ball_contact(plan, target_z=contact[2])
                if c is None:
                    raise RuntimeError("no rise feed solution")
    except RuntimeError as exc:
        print("  feed failed: %s" % exc)
        return False, {"hits": 0, "far_in": 0, "net": 0}
    t_arrive = c["t"]

    env.reset(machine_pos=(MACHINE_POS[0], args.machine_y, 0.0), machine_yaw=0.0)
    obs = env.get_obs()
    policy.reset(obs)
    policy.motion_idx = m_idx
    policy.goal_pos_w[m_idx] = contact.copy()

    rec = EpisodeRecorder(env, args.out, fps=50, width=1920, height=1080,
                          camera=args.camera) if record else None
    served = {"done": False}
    rest_n = 0
    hit = False
    for k in range(int((args.seconds + 6.0) / env.dt)):
        t = k * env.dt
        if not served["done"] and t >= args.serve_t:
            served["done"] = True
            env.serve_ball(plan=plan)
            policy.notify_launch()
        obs, _, term, info_evt = env.step(policy.act(obs))
        if rec is not None:
            rec.on_step(env, obs, info_evt)
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
    net = [e for e in all_events if e["type"] == "net_hit"]
    far_in = [b for b in bounces if b["zone"]["side"] == "far" and b["zone"]["in"]]
    info = {"hits": len(hits), "far_in": len(far_in), "net": len(net)}
    ok = bool(hits) and len(far_in) >= 1 and not net
    print("  -> hit=%s hits=%d far_in=%d net=%d" % (
        bool(hits), len(hits), len(far_in), len(net)))
    for e in hits:
        print("     racket_hit t=%.2f v_in=%.1f v_out=%.1f racket=%.1f" % (
            e["t"], e["ball_speed_in"], e["ball_speed_out"], e["racket_speed"]))
    for b in bounces[:4]:
        print("     bounce t=%.2f at %s %s" % (b["t"], np.round(b["pos"], 2),
                                               b["zone"]))
    for e in net:
        print("     NET_HIT t=%.2f" % e["t"])
    return ok, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", required=True, choices=list(STROKE_IDX))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--serve-t", type=float, default=2.2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--spin", type=float, default=2000.0)
    ap.add_argument("--arrival-t", type=float, default=1.5)
    ap.add_argument("--pass-phase", default="apex", choices=["apex", "fall", "direct"])
    ap.add_argument("--robot-x", type=float, default=10.9)
    ap.add_argument("--machine-y", type=float, default=-1.2)
    ap.add_argument("--offsets", default=None,
                    help="semicolon list of dx,dy,dz triples")
    ap.add_argument("--camera", default="robot_front")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = "outputs/videos/tnp_%s.mp4" % args.stroke
    m_idx = STROKE_IDX[args.stroke]

    from tennis_sim import world as W

    scene_xml = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "assets", "scene_%s.xml" % args.stroke)
    W.write_scene(path=scene_xml, robot_pos=(args.robot_x, 0.0, 0.79))
    env = TennisEnv(scene_path=scene_xml)
    patch_robot_to_tasknpoint(env, verbose=False)
    patch_racket_to_tasknpoint(env, verbose=False)
    policy = TnpPolicy(env, trigger_margin=args.margin, verbose=False)
    env.max_time = args.seconds + 6.0

    env.reset(machine_pos=MACHINE_POS, machine_yaw=0.0)
    obs = env.get_obs()
    policy.reset(obs)
    policy.motion_idx = m_idx
    nominal = policy.goal_pos_w[m_idx].copy()
    print("[%s] nominal goal %s" % (args.stroke, np.round(nominal, 3)))

    if args.offsets:
        offs = []
        for triple in args.offsets.split(";"):
            vals = [float(x) for x in triple.split(",")]
            while len(vals) < 3:
                vals.insert(0, 0.0)
            offs.append(tuple(vals))
    else:
        offs = [(0.0, 0.0, 0.0), (0.0, 0.2, -0.1), (0.0, -0.2, -0.1),
                (0.0, 0.2, 0.0), (0.0, -0.2, 0.0)]

    best = None
    for dy, _, dz in offs:
        contact = nominal + np.array([0.0, dy, dz])
        args.margin_serve = None
        for margin in [args.margin, 0.0, -0.05]:
            policy.trigger_margin = margin
            print("[%s] try contact %s margin %+.2f speed %.0f" % (
                args.stroke, np.round(contact, 3), margin, args.speed))
            ok, info = run_once(args, env, policy, m_idx, contact, record=False)
            score = (2 if ok else 0) + info["far_in"] - 2 * info["net"]
            if best is None or score > best[0]:
                best = (score, dict(contact=contact.tolist(), margin=margin))
            if ok:
                print("[%s] SUCCESS contact=%s margin=%+.2f" % (
                    args.stroke, np.round(contact, 3), margin))
                live_env = env
                live_env.max_time = args.seconds + 6.0
                run_once(args, live_env, policy, m_idx, contact, record=True)
                print("SWEEP_SUCCESS " + str(best[1]))
                return
    if best is not None and not getattr(args, 'no_video', False):
        print("[final] recording the best config")
        contact = np.array(best[1]["contact"])
        policy.trigger_margin = best[1]["margin"]
        run_once(args, env, policy, m_idx, contact, record=True)
    print("SWEEP_BEST " + str(best[1] if best else None))


if __name__ == "__main__":
    main()
