#!/usr/bin/env python
"""Deploy the TaskNPoint user-strokes policy (forehand / backhand / backhand
volley) in this simulator with tennis-legal feeds.

Per stroke:
  1. runtime patches: TNP actuator gains + racket mount aligned to training
  2. ONE closed-loop airswing capture of the reference motion, giving this
     project's racket-head path / face normals / speeds
  3. frames ranked by how netward the face points times racket speed; for each
     candidate frame a bounce-first feed (one bounce on the robot side) is
     aimed at the measured racket position and refined by the observed
     ball-vs-racket miss until the stroke returns the ball into the far court
  4. the winning config is re-run under the recorder
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
from tennis_sim.tnp.robot_patch import (patch_robot_to_tasknpoint,
                                        patch_racket_to_tasknpoint,
                                        patch_racket_roll)

MACHINE_POS = (-12.7, -1.2, 0.0)
STROKE_IDX = {"forehand": 0, "volley": 1, "backhand": 2}


class FixedGoalPolicy(TnpPolicy):
    """Keeps the NOMINAL position goal during the swing (no intercept
    override), so the closed-loop swing reproduces the airswing racket path
    exactly. Trigger semantics unchanged, but the trigger/estimate target can
    be pointed at the actual feed pass point (trigger_targets[m])."""

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
            if self.verbose:
                print("[tnp] TRIGGER swing #%d motion=%s ttc=%.3f (thr %.3f)"
                      % (self.swing_count, self.motion_names[self.motion_idx],
                         ttc, threshold))


def pelvis_id(env):
    import mujoco

    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")


def airswing_capture(env, policy, motion_idx):
    obs = env.get_obs()
    policy.reset(obs)
    for _ in range(int(1.6 / env.dt)):
        obs, _, _, _ = env.step(policy.act(obs))
    policy.motion_idx = motion_idx
    policy.force_trigger(motion_idx)
    mo = policy.motions[policy.motion_names[motion_idx]]
    pid = pelvis_id(env)
    P, N, A, Q = [], [], [], []
    for k in range(mo["num_frames"] + 5):
        sid = env.ball_ctrl.racket_site
        Rf = env.data.site_xmat[sid].reshape(3, 3)
        P.append(env.data.site_xpos[sid].copy())
        N.append(Rf[:, 2].copy())
        A.append(Rf[:, 1].copy())
        Q.append(env.data.xpos[pid].copy())
        obs, _, term, _ = env.step(policy.act(obs))
        if term:
            break
    P, N, A, Q = np.array(P), np.array(N), np.array(A), np.array(Q)
    V = np.zeros_like(P)
    V[1:-1] = (P[2:] - P[:-2]) * mo["fps"] / 2.0
    return {"P": P, "N": N, "A": A, "V": V, "pelvis": Q, "fps": mo["fps"],
            "ttc": policy.time_to_contact[motion_idx]}


def solve_feed_apex(env, contact_point, spin=2000.0, arrival_t=1.8,
                    verbose=False, speed0=20.0):
    """Bounce feed with the ball arriving at the contact point near its
    post-bounce APEX (flat ball, vz~0): scan the landing distance and pick the
    plan whose predicted pass velocity at the contact point is flattest."""
    import numpy as _np

    goal = _np.asarray(contact_point, float)
    machine_pos = env.machine.launch_site_pos()[:2]
    toward = goal[:2] - machine_pos
    toward = toward / max(_np.linalg.norm(toward), 1e-9)
    best = None
    for d in (0.7, 1.0, 1.3, 1.6, 2.0, 2.4, 2.9, 3.4):
        target_xy = goal[:2] - toward * d
        try:
            plan = env.machine.plan(target_xy, speed0, spin, True)
            c = predict_ball_contact(plan, target_z=goal[2])
        except ValueError:
            continue
        if c is None:
            continue
        pos_err = float(_np.linalg.norm(c["pos"][:2] - goal[:2]))
        if pos_err > 0.25:
            continue
        flat = abs(float(c["vel"][2]))
        t_ok = abs(c["t"] - arrival_t)
        score = flat + 4.0 * pos_err + 3.0 * t_ok
        if best is None or score < best[0]:
            best = (score, plan, c, d)
    if best is None:
        raise RuntimeError("no apex feed solution")
    if verbose:
        print("  apex feed: d=%.1f t=%.3f pos=%s vel=%s" % (
            best[3], best[2]["t"], _np.round(best[2]["pos"], 3),
            _np.round(best[2]["vel"], 2)))
    return best[1], best[2]


def solve_bounce_feed(env, contact_point, spin=2000.0, speed0=20.0,
                      lo=9.0, hi=30.0, iters=8, verbose=False, arrival_t=1.8):
    best = None
    plan = None
    contact = None
    for i in range(iters):
        speed = float(0.5 * (lo + hi)) if i else speed0
        try:
            plan_i = plan_feed_for_contact(env, contact_point, speed=speed, spin=spin)
            c = predict_ball_contact(plan_i, target_z=contact_point[2])
        except ValueError:
            lo = speed
            continue
        if c is None:
            lo = speed
            continue
        err = c["t"] - arrival_t
        if best is None or abs(err) < abs(best[0]):
            best = (err, plan_i, speed, c)
            plan, contact = plan_i, c
        if abs(err) < 0.05:
            break
        if err < 0:
            lo = speed
        else:
            hi = speed
    if plan is None:
        raise RuntimeError("no bounce feed solution")
    if verbose:
        print("  feed: speed=%.1f t=%.3f pos=%s" % (
            best[2], contact["t"], np.round(contact["pos"], 3)))
    return plan, contact


def live_run(env, policy, m_idx, sw, f, contact, args, record=False,
             machine_y=MACHINE_POS[1]):
    """One live episode. Returns (ok, info, miss_vector_or_None)."""
    try:
        plan, pred = solve_feed_apex(env, contact, spin=args.spin,
                                     arrival_t=args.arrival_t)
    except RuntimeError as exc:
        print("  feed failed: %s" % exc)
        return False, {"hits": 0, "far_in": 0, "net": 0}, None

    env.reset(machine_pos=(MACHINE_POS[0], machine_y, 0.0), machine_yaw=0.0)
    obs = env.get_obs()
    policy.reset(obs)
    policy.motion_idx = m_idx
    policy.time_to_contact[m_idx] = f / sw["fps"]
    # deterministic trigger: swing starts so that frame f coincides with the
    # planned ball arrival at the contact point
    t_trigger = args.serve_t + pred["t"] - f / sw["fps"] + args.margin
    print("  t_trigger=%.3f (serve %.2f + pass %.3f - f/fps %.3f + margin %.2f)"
          % (t_trigger, args.serve_t, pred["t"], f / sw["fps"], args.margin))
    policy.trigger_targets = {m_idx: contact.copy()}

    rec = EpisodeRecorder(env, args.out, fps=50, width=1920, height=1080,
                          camera=args.camera) if record else None
    served = {"done": False}
    rest_n = 0
    hit = False
    hit_dbg = [False]
    frames = 0
    racket_at_f = None
    ball_at_f = None
    prev_vel = obs["ball_vel"].copy()
    for k in range(int((args.seconds + 6.0) / env.dt)):
        t = k * env.dt
        if not served["done"] and t >= args.serve_t:
            served["done"] = True
            env.serve_ball(plan=plan)
            policy.notify_launch()
        if k % 25 == 0:
            print("  [dbg] t=%.2f state=%s frame=%d" % (t, policy.state, policy.frame))
        if policy.state == STATE_READY and t >= t_trigger:
            policy.force_trigger(m_idx)
            print("  force trigger at t=%.2f (planned arrival %.2f)" % (t, args.serve_t + pred["t"]))
        pre_frame = policy.frame if policy.state == STATE_SWING else None
        obs, _, term, info_evt = env.step(policy.act(obs))
        frames += 1
        if rec is not None:
            rec.on_step(env, obs, info_evt)
        if policy.state == STATE_SWING and pre_frame == f and racket_at_f is None:
            sid = env.ball_ctrl.racket_site
            racket_at_f = env.data.site_xpos[sid].copy()
            ball_at_f = obs["ball_pos"].copy()
        if any(e["type"] == "racket_hit" for e in info_evt["events"]):
            hit = True
            if not hit_dbg[0]:
                sid = env.ball_ctrl.racket_site
                hit_dbg[0] = True
                print("     AT-HIT face=%s racket_vel=%s racket=%s" % (
                    np.round(env.data.site_xmat[sid].reshape(3, 3)[:, 2], 2),
                    np.round([v for v in np.zeros(3)], 2), ""))
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
        if "v_out" in e:
            print("     v_out_vec=%s" % np.round(e["v_out"], 2))
    for b in bounces[:4]:
        print("     bounce t=%.2f at %s %s" % (b["t"], np.round(b["pos"], 2),
                                               b["zone"]))
    for e in net:
        print("     NET_HIT t=%.2f" % e["t"])
    miss = None
    if racket_at_f is not None and ball_at_f is not None and not ok:
        miss = ball_at_f - racket_at_f
        print("     miss(ball-racket@f%d)=%s" % (f, np.round(miss, 3)))
    at_hit = None
    if hit_dbg[0]:
        sid = env.ball_ctrl.racket_site
        Rf = env.data.site_xmat[sid].reshape(3, 3)
        at_hit = (Rf[:, 2].copy(), Rf[:, 0].copy())
    return ok, info, (miss, at_hit)


N_DES_TABLE = {
    "forehand": np.array([-0.95, 0.0, 0.31]),
    "backhand": np.array([-0.95, 0.0, 0.31]),
    # the volley swipe descends: aim the face more open to lift the return
    "volley": np.array([-0.80, 0.0, 0.60]),
}
N_DES_TABLE = {k: v / np.linalg.norm(v) for k, v in N_DES_TABLE.items()}


def racket_axes(env, f):
    sid = env.ball_ctrl.racket_site
    R = env.data.site_xmat[sid].reshape(3, 3)
    return R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()


def run_stroke(args):
    from tennis_sim import world as W

    scene_xml = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "assets", "scene_%s.xml" % args.stroke)
    W.write_scene(path=scene_xml)
    env = TennisEnv(scene_path=scene_xml,
                    robot_pos=(args.robot_x, 0.0, 0.79))
    patch_robot_to_tasknpoint(env, verbose=False)
    patch_racket_to_tasknpoint(env, verbose=False)
    policy = FixedGoalPolicy(env, trigger_margin=args.margin, verbose=False)
    env.max_time = args.seconds + 6.0
    m_idx = STROKE_IDX[args.stroke]

    N_DES = N_DES_TABLE[args.stroke]
    env.reset(machine_pos=MACHINE_POS, machine_yaw=0.0)
    sw = airswing_capture(env, policy, m_idx)
    print("[%s] airswing %d frames, ttc=%.3f" % (args.stroke, len(sw["P"]), sw["ttc"]))
    # the estimator/trigger must key on the actual feed pass point
    policy.trigger_targets = {m_idx: None}

    frames = args.frames
    if frames:
        cand = [int(x) for x in frames.split(",")]
    else:
        # rank by the POST-ROLL face (the roll can aim any face netward) times
        # the racket speed at that frame
        window = sw["P"].shape[0] - 2
        cands = []
        for f in range(5, window):
            n_cur, a = sw["N"][f], sw["A"][f]
            A2 = np.stack([n_cur, -a], axis=1)
            coef, *_ = np.linalg.lstsq(A2, N_DES, rcond=None)
            roll = float(np.degrees(np.arctan2(coef[1], coef[0])))
            n_after = (np.cos(np.radians(roll)) * n_cur -
                       np.sin(np.radians(roll)) * a)
            vr = min(np.linalg.norm(sw["V"][f]), 8.0)
            score = float(n_after @ N_DES) * (0.35 + 0.65 * vr / 8.0)
            cands.append((score, f, roll))
        cands.sort(reverse=True)
        cand = [f for _, f, _ in cands[:args.top]]
        print("[%s] ranked frames (post-roll): %s" % (
            args.stroke, [(f, round(sc, 2)) for sc, f, _ in cands]))

    base_contact = None
    best = None
    machine_ys = [float(x) for x in args.machine_ys.split(",")]
    for f in cand:
        f = min(f, len(sw["P"]) - 1)
        # roll the racket about its shaft so the face aims at the net from
        # this frame's arm pose (closed form; path unchanged)
        n_cur, a = sw["N"][f].copy(), sw["A"][f].copy()
        A = np.stack([n_cur, -a], axis=1)
        coef, *_ = np.linalg.lstsq(A, N_DES, rcond=None)
        roll = float(np.degrees(np.arctan2(coef[1], coef[0])))
        patch_racket_roll(env, roll)
        n_new = (np.cos(np.radians(roll)) * n_cur -
                 np.sin(np.radians(roll)) * a)
        print("[%s] f=%d roll=%+.1f deg -> face %s (score %.2f)" % (
            args.stroke, f, roll, np.round(n_new, 2), float(n_new @ N_DES)))
        contact = sw["P"][f].copy()
        if args.contact_xy is not None:
            contact[0] += args.contact_xy[0]
            contact[1] += args.contact_xy[1]
        if args.contact_z is not None:
            contact[2] += args.contact_z
        print("[%s] try frame=%d racket=%s n=%s |v_r|=%.1f" % (
            args.stroke, f, np.round(sw["P"][f], 3), np.round(sw["N"][f], 2),
            np.linalg.norm(sw["V"][f])))
        for machine_y in machine_ys:
            for speed in [args.speed]:
                policy.trigger_margin = args.margin
                tgt = contact.copy()
                roll_total = 0.0
                for it in range(args.refine):
                    print("[%s] f=%d my=%+.1f v=%.0f refine %d target %s" % (
                        args.stroke, f, machine_y, speed, it, np.round(tgt, 3)))
                    # -- face closed loop: roll until the AT-HIT face points
                    # netward (the live wrist pose drifts from the airswing) --
                    miss = None
                    at_hit = None
                    for face_it in range(3):
                        ok, info, (miss, at_hit) = live_run(
                            env, policy, m_idx, sw, f, tgt, args,
                            record=(not args.no_video and
                                    it == 0 and face_it == 0),
                            machine_y=machine_y)
                        if ok:
                            print("[%s] SUCCESS config: frame=%d machine_y=%+.1f "
                                  "speed=%.0f target=%s" % (
                                      args.stroke, f, machine_y, speed,
                                      np.round(tgt, 3)))
                            if not args.no_video:
                                live_run(env, policy, m_idx, sw, f, tgt, args,
                                         record=True, machine_y=machine_y)
                            print("SWEEP_SUCCESS " + str(
                                dict(frame=f, machine_y=machine_y, speed=speed,
                                     target=tgt.tolist(), roll_total=roll_total)))
                            return
                        if at_hit is None or miss is None:
                            break
                        n_hit, shaft = at_hit
                        n_hit_p = n_hit - shaft * float(n_hit @ shaft)
                        des_p = N_DES - shaft * float(N_DES @ shaft)
                        if (np.linalg.norm(n_hit_p) < 1e-6
                                or np.linalg.norm(des_p) < 1e-6):
                            break
                        n_hit_p /= np.linalg.norm(n_hit_p)
                        des_p /= np.linalg.norm(des_p)
                        err = float(np.degrees(np.arctan2(
                            float(np.cross(n_hit_p, des_p) @ shaft),
                            float(n_hit_p @ des_p))))
                        print("  face_it %d: at-hit face %s err=%+.1f deg" % (
                            face_it, np.round(n_hit, 2), err))
                        if abs(err) < 12.0:
                            break
                        roll_total += err
                        patch_racket_roll(env, roll_total)
                    # -- feed refinement (ball side only; deterministic racket) --
                    if miss is None:
                        break
                    tgt = tgt - 0.8 * miss
    print("SWEEP_BEST " + str(best[1] if best else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke", required=True, choices=list(STROKE_IDX))
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--serve-t", type=float, default=2.2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--spin", type=float, default=2000.0)
    ap.add_argument("--arrival-t", type=float, default=1.8)
    ap.add_argument("--frames", default=None, help="comma-separated frame list")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--refine", type=int, default=3)
    ap.add_argument("--contact-xy", type=float, nargs=2, default=None)
    ap.add_argument("--contact-z", type=float, default=None)
    ap.add_argument("--camera", default="robot_front")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--machine-ys", default="-1.2")
    ap.add_argument("--robot-x", type=float, default=10.9)
    args = ap.parse_args()
    if args.out is None:
        args.out = "outputs/videos/tnp_%s.mp4" % args.stroke
    args.no_video = bool(args.no_video)
    run_stroke(args)


if __name__ == "__main__":
    main()
