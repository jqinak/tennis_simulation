import argparse
import os
import re
import sys

import numpy as np

import mujoco

from tennis_sim import aero
from tennis_sim import ball_machine
from tennis_sim import bounce as B
from tennis_sim import constants as C
from tennis_sim import world as W

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    print("[%s] %s %s" % (status, name, detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


class BallSim:
    def __init__(self, model, data):
        self.m = model
        self.d = data
        self.ctrl = B.BallController(model)

    def set_ball(self, pos, vel=np.zeros(3), omega=np.zeros(3)):
        self.ctrl.reset()
        self.d.qpos[self.ctrl.qadr:self.ctrl.qadr + 3] = pos
        self.d.qpos[self.ctrl.qadr + 3:self.ctrl.qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[self.ctrl.vadr:self.ctrl.vadr + 3] = vel
        self.d.qvel[self.ctrl.vadr + 3:self.ctrl.vadr + 6] = omega
        self.d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        pos, v, w, _ = self.ctrl.ball_states(self.d)
        return pos, v, w

    def run(self, tmax, stop_fn=None):
        n = int(tmax / self.m.opt.timestep)
        for _ in range(n):
            self.ctrl.step(self.d)
            if stop_fn is not None and stop_fn(self):
                return False
        return True


def build_ball_only_scene():
    path = W.write_scene(robot=False, machine=False)
    model, data = W.load_scene(path)
    return BallSim(model, data)


def measure_rebound(sim, drop_bottom_height, x=5.0, v0z=0.0):
    sim.set_ball([x, 0.0, drop_bottom_height + C.BALL_RADIUS], vel=[0.0, 0.0, v0z])
    seen = [False]
    ended = [False]
    max_z = [0.0]

    def after_contact(s):
        pos, vel, _ = s.state()
        g = s.ctrl.was_ground
        if not seen[0]:
            if g:
                seen[0] = True
            return False
        if not ended[0]:
            if not g:
                ended[0] = True
                max_z[0] = pos[2]
            return False
        if g:
            return True
        max_z[0] = max(max_z[0], pos[2])
        return vel[2] < -0.4

    sim.run(5.0, after_contact)
    return max_z[0] - C.BALL_RADIUS


def calibrate_restitution():
    print("== restitution calibration (ITF 2.54m -> 1.41m) ==")
    sim = build_ball_only_scene()
    target = 1.41
    for it in range(4):
        C.RESTITUTION_INTERCEPT = round(C.RESTITUTION_INTERCEPT, 5)
        h = measure_rebound(sim, 2.54)
        e_cur = float(np.clip(C.RESTITUTION_INTERCEPT + C.RESTITUTION_SLOPE * 6.9,
                              C.RESTITUTION_MIN, C.RESTITUTION_MAX))
        e_new = float(np.clip(e_cur * np.sqrt(target / max(h, 0.05)), 0.70, 0.78))
        print("iter %d: intercept=%.4f rebound=%.3f -> e %.4f" % (it, C.RESTITUTION_INTERCEPT, h,
                                                                  e_new))
        if abs(h - target) < 0.01:
            break
        C.RESTITUTION_INTERCEPT = round(C.RESTITUTION_INTERCEPT + (e_new - e_cur), 5)
    h = measure_rebound(sim, 2.54)
    const_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants.py")
    with open(const_path) as f:
        text = f.read()
    text = re.sub(r'RESTITUTION_INTERCEPT = [0-9.]+',
                  'RESTITUTION_INTERCEPT = %s' % C.RESTITUTION_INTERCEPT, text)
    with open(const_path, "w") as f:
        f.write(text)
    print("constants.py updated, intercept=%.5f, final rebound=%.3f m" % (
        C.RESTITUTION_INTERCEPT, h))
    return h


def check_rebound_band():
    sim = build_ball_only_scene()
    h = measure_rebound(sim, 2.54)
    report("ITF drop 2.54m rebound", 1.35 <= h <= 1.47, "rebound=%.3f m (spec 1.35-1.47)" % h)
    ok_all = True
    for drop_h in (1.0, 1.8, 3.2):
        h2 = measure_rebound(sim, drop_h)
        e = np.sqrt(max(h2, 1e-6) / drop_h)
        good = 0.66 <= e <= 0.80
        ok_all = ok_all and good
        report("rebound ratio @ drop %.1fm" % drop_h, good, "e=%.3f" % e)
    return ok_all


def check_high_speed_restitution():
    sim = build_ball_only_scene()
    sim.set_ball([5.0, 0.0, 0.5], vel=[0.0, 0.0, -30.0])
    post = [None]

    def watch(s):
        if s.ctrl.last_bounce is not None:
            post[0] = s.ctrl.last_bounce
            return True
        return False

    sim.run(1.0, watch)
    if post[0] is None:
        report("high-speed restitution", False, "no bounce event")
        return False
    e = post[0]["restitution"]
    v_out = post[0]["v_post"][2]
    ok = 0.70 <= e <= 0.72 and 20.0 <= v_out <= 22.6
    report("high-speed restitution @30m/s", ok, "e=%.3f v_out=%.2f m/s" % (e, v_out))
    return ok


def check_penetration():
    sim = build_ball_only_scene()
    sim.set_ball([5.0, 0.0, 0.5], vel=[0.0, 0.0, -50.0])
    min_z = [9.9]

    def watch(s):
        pos, _, _ = s.state()
        if pos[2] < 0.6:
            min_z[0] = min(min_z[0], pos[2])
        if s.ctrl.last_bounce is not None:
            return True
        return False

    sim.run(0.6, watch)
    pen = C.BALL_RADIUS - min_z[0]
    ok = 0.0 <= pen < 0.025
    report("penetration @50m/s impact", ok, "pen=%.4f m" % pen)
    return ok


def check_friction():
    sim = build_ball_only_scene()
    sim.set_ball([-5.0, 0.0, 1.0], vel=[6.0, 0.0, -2.0])

    def watch(s):
        return s.ctrl.last_bounce is not None

    sim.run(2.5, watch)
    if sim.ctrl.last_bounce is None:
        report("friction bounce horizontal loss", False, "no bounce event")
        return False
    pre_v = float(np.linalg.norm(sim.ctrl.last_bounce["v_pre"][:2]))
    post_v = float(np.linalg.norm(sim.ctrl.last_bounce["v_post"][:2]))
    ok = 0.55 <= post_v / pre_v <= 0.95
    report("friction bounce horizontal loss", ok, "pre=%.2f post=%.2f m/s" % (pre_v, post_v))
    return ok


def check_drag():
    sim = build_ball_only_scene()
    p0 = np.array([0.0, 0.0, 2.0])
    v0 = np.array([40.0, 0.0, 0.0])
    w0 = np.zeros(3)
    sim.set_ball(p0, v0, w0)
    traj_ref, _, _ = aero.rk4_trajectory(p0, v0, w0, 0.5, dt=0.001)
    max_err = 0.0
    t = 0.0

    def sample(s):
        nonlocal max_err, t
        t += s.m.opt.timestep
        idx = int(round(t / 0.001))
        if idx < len(traj_ref):
            pos, _, _ = s.state()
            max_err = max(max_err, float(np.linalg.norm(pos - traj_ref[idx][0])))
        return False

    sim.run(0.5, sample)
    ok = max_err < 0.05
    report("drag trajectory vs analytic RK4", ok, "max_err=%.4f m over 0.5s @40m/s" % max_err)
    return ok


def check_magnus():
    sim = build_ball_only_scene()
    p0 = np.array([-2.0, 0.0, 1.5])
    v0 = np.array([30.0, 0.0, 0.0])
    w_top = np.array([0.0, 3000 * 2 * np.pi / 60, 0.0])
    sim.set_ball(p0, v0, w_top)
    traj_ref, _, _ = aero.rk4_trajectory(p0, v0, w_top, 0.4, dt=0.001)
    max_err = [0.0]
    t = [0.0]

    def sample(s):
        t[0] += s.m.opt.timestep
        idx = int(round(t[0] / 0.001))
        if idx < len(traj_ref):
            pos, _, _ = s.state()
            max_err[0] = max(max_err[0], float(np.linalg.norm(pos - traj_ref[idx][0])))
        return False

    sim.run(0.4, sample)
    ok = max_err[0] < 0.05
    report("magnus trajectory vs analytic RK4", ok, "max_err=%.4f m @3000rpm topspin" % max_err[0])

    def landing_x(omega):
        sim.set_ball(p0, v0, omega)

        def stop(s):
            return s.ctrl.was_ground

        sim.run(1.5, stop)
        return sim.state()[0][0]

    x_flat = landing_x(np.zeros(3))
    x_top = landing_x(w_top)
    x_back = landing_x(-w_top)
    ok2 = x_top < x_flat < x_back
    report("magnus directional dip/lift", ok2,
           "x_back=%.2f x_flat=%.2f x_top=%.2f" % (x_back, x_flat, x_top))
    return ok and ok2


def check_spin_decay():
    sim = build_ball_only_scene()
    sim.set_ball([-3.0, 0.0, 2.5], vel=[10.0, 0.0, 6.0], omega=[0.0, 300.0, 0.0])
    sim.run(1.0)
    _, _, w_now = sim.state()
    expected = 300.0 * np.exp(-C.SPIN_DECAY_RATE * 1.0)
    ok = abs(np.linalg.norm(w_now) - expected) < 0.02 * expected
    report("spin decay (free flight)", ok, "|w|=%.1f expected %.1f rad/s" % (
        np.linalg.norm(w_now), expected))
    return ok


def check_machine():
    path = W.write_scene(robot=False, machine=True)
    model, data = W.load_scene(path)
    sim = BallSim(model, data)
    machine = ball_machine.BallMachine(model, data)
    machine.set_position(pos=(-12.7, -1.2, 0.0), yaw=0.0)
    rng = np.random.default_rng(7)
    machine_rng = np.random.default_rng(11)

    import tennis_sim.ball_machine as _bm

    _orig_rng_method = machine.serve

    def serve_seeded(target_xy, mode="flat", speed=None, spin_rpm=None):
        state = np.random.get_state()
        np.random.seed(int(machine_rng.integers(0, 2**31 - 1)))
        try:
            return _orig_rng_method(target_xy, mode=mode, speed=speed, spin_rpm=spin_rpm)
        finally:
            np.random.set_state(state)
    targets = [
        (8.0, -1.0, "flat"), (9.0, 1.5, "topspin"), (5.0, -2.5, "slice"),
        (10.0, 0.0, "flat"), (3.0, 2.0, "topspin"), (7.0, -3.0, "slice"),
        (6.4, 1.0, "flat"), (9.5, -1.8, "topspin"), (2.0, -1.5, "slice"), (8.5, 2.5, "flat"),
    ]
    all_ok = True
    max_err = 0.0
    for tx, ty, mode in targets:
        target = np.array([tx + rng.uniform(-0.3, 0.3), ty + rng.uniform(-0.3, 0.3)])
        plan = machine.serve(target, mode=mode)
        sim.ctrl.reset()

        def stop(s):
            return s.ctrl.was_ground

        sim.run(2.5, stop)
        land, _, _ = sim.state()
        err = float(np.linalg.norm(land[:2] - target))
        max_err = max(max_err, err)
        ok = err < 0.45
        all_ok = all_ok and ok
        report("machine serve (%s -> %.1f,%.1f)" % (mode, tx, ty), ok,
               "err=%.2fm T=%.2fs v=%.1fm/s" % (err, plan["flight_time"], plan["speed"]))
    report("machine overall landing accuracy", max_err < 0.45, "max_err=%.3f m" % max_err)

    deep = machine.serve(np.array([10.5, 0.0]), mode="flat", speed=33.0)
    t_ok = 0.60 <= deep["flight_time"] <= 1.35
    report("flat serve flight time realistic", t_ok, "T=%.2fs for full court" % deep["flight_time"])
    all_ok = all_ok and t_ok

    blocked_all = True
    for k in range(5):
        machine.serve_direct(np.array([0.0, -0.3 + 0.15 * k, 0.55]), speed=40.0 + 2.0 * k,
                             spin_rpm=800.0)
        sim.ctrl.reset()
        crossed = [False]

        def watch(s, crossed=crossed):
            pos, _, _ = s.state()
            if pos[0] > 0.6:
                crossed[0] = True
                return True
            return False

        sim.run(2.0, watch)
        ok = not crossed[0]
        blocked_all = blocked_all and ok
        report("net blocks 40+ m/s ball #%d" % k, ok)
    all_ok = all_ok and blocked_all
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--skip-machine", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_restitution()
        W.write_scene()

    print("== physics verification ==")
    check_rebound_band()
    check_high_speed_restitution()
    check_penetration()
    check_friction()
    check_drag()
    check_magnus()
    check_spin_decay()
    if not args.skip_machine:
        check_machine()

    print("== summary: %d passed, %d failed ==" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
