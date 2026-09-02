import numpy as np

from tennis_sim import aero
from tennis_sim import constants as C

UP = np.array([0.0, 0.0, 1.0])

SERVE_MODES = {
    "flat": {"speed": (24.0, 33.0), "spin_rpm": (600.0, 1000.0), "topspin": True},
    "topspin": {"speed": (16.0, 27.0), "spin_rpm": (2200.0, 3200.0), "topspin": True},
    "slice": {"speed": (19.0, 29.0), "spin_rpm": (1200.0, 2000.0), "topspin": False},
    "lob": {"speed": (8.0, 13.0), "spin_rpm": (500.0, 900.0), "topspin": True},
}


def spin_axis_for(vel_dir):
    axis = np.cross(UP, vel_dir)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])
        n = 1.0
    return axis / n


def quat_from_two_vectors(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    half_sin = np.linalg.norm(v)
    w = 1.0 + c
    q = np.concatenate(([w], v))
    q /= np.linalg.norm(q)
    return q


def quat_rotate(q, v):
    w = q[0]
    u = q[1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def aim_quat(dir_vec):
    return quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), dir_vec)


def net_height_at(y):
    t = min(abs(y) / C.NET_POST_Y, 1.0)
    sag = 1.0 - (2.0 * t - 1.0) ** 2
    return C.NET_HEIGHT_POST - (C.NET_HEIGHT_POST - C.NET_HEIGHT_CENTER) * sag


def trajectory_net_clearance(traj, margin=0.12):
    prev = traj[0][0]
    for pos, _ in traj[1:]:
        if prev[0] < 0.0 <= pos[0] or pos[0] < 0.0 <= prev[0]:
            alpha = (0.0 - prev[0]) / (pos[0] - prev[0]) if abs(pos[0] - prev[0]) > 1e-9 else 0.0
            z = prev[2] + alpha * (pos[2] - prev[2])
            y = prev[1] + alpha * (pos[1] - prev[1])
            if z < net_height_at(y) + margin:
                return False
        prev = pos
    return True


def solve_launch(exit_pos, target_xy, speed, omega, net_margin=0.10):
    target_xy = np.asarray(target_xy, float)
    exit_pos = np.asarray(exit_pos, float)
    rel = target_xy - exit_pos[:2]
    base_yaw = np.arctan2(rel[1], rel[0])

    def land(theta, phi):
        v = speed * np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                              np.sin(theta)])
        traj, pos, _ = aero.rk4_trajectory(exit_pos, v, omega, 2.5, dt=0.004)
        return pos[:2], v, traj

    thetas = np.linspace(0.005, 1.10, 12)
    dxs = []
    for th in thetas:
        xy, _, _ = land(th, base_yaw)
        dxs.append(xy[0] - target_xy[0])
    candidates = []
    for i in range(len(thetas) - 1):
        if dxs[i] == 0.0 or dxs[i] * dxs[i + 1] <= 0:
            lo, hi = thetas[i], thetas[i + 1]
            x_lo = dxs[i]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                x_mid, _, _ = land(mid, base_yaw)
                dx_mid = x_mid[0] - target_xy[0]
                if dx_mid == 0.0 or x_lo * dx_mid <= 0:
                    hi = mid
                else:
                    lo, x_lo = mid, dx_mid
                if hi - lo < 1e-4:
                    break
            candidates.append(0.5 * (lo + hi))
    if not candidates:
        overshoot = bool(dxs and min(dxs) > 0.0)
        return None, False, False, False, False, overshoot
    best = None
    cleared_any = False
    ranged_any = False
    for theta in candidates:
        phi = base_yaw
        xy, v, traj = land(theta, phi)
        for _ in range(16):
            err_y = xy[1] - target_xy[1]
            if abs(err_y) < 0.005:
                break
            dphi = 1e-4
            xy2, _, _ = land(theta, phi + dphi)
            deriv = (xy2[1] - xy[1]) / dphi
            if abs(deriv) < 1e-9:
                break
            phi -= float(np.clip(err_y / deriv, -0.05, 0.05))
            xy, v, traj = land(theta, phi)
        err = float(np.linalg.norm(xy - target_xy))
        flight_t = (len(traj) - 1) * 0.002
        cleared = trajectory_net_clearance(traj, net_margin)
        if cleared:
            cleared_any = True
        if err < 0.08:
            ranged_any = True
        toolong = flight_t > 1.8
        score = err + (0.0 if cleared else 100.0) + (50.0 if toolong else 0.0)
        if best is None or score < best[0]:
            best = (score, v, cleared, err, flight_t, toolong)
        if cleared and err < 0.08 and flight_t < 1.8:
            return v, True, True, True, False, False
    if best is None:
        return None, False, False, False, False, False
    return (best[1], best[2] and best[3] < 0.35 and not best[5], best[2], ranged_any, best[5],
            False)


def solve_feed_through_point(exit_pos, point, speed, omega, tol=0.10):
    point = np.asarray(point, float)
    exit_pos = np.asarray(exit_pos, float)
    rel = point - exit_pos
    base_yaw = np.arctan2(rel[1], rel[0])

    def traj_of(theta, phi):
        v = speed * np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                              np.sin(theta)])
        traj, _, _ = aero.rk4_trajectory(exit_pos, v, omega, 2.0, dt=0.006)
        return traj, v

    def miss(theta, phi):
        traj, _ = traj_of(theta, phi)
        dmin = 1e9
        for p, _ in traj:
            d = np.linalg.norm(p - point)
            if d < dmin:
                dmin = d
        return dmin

    thetas = np.linspace(-0.05, 0.85, 26)
    misses = [miss(th, base_yaw) for th in thetas]
    i0 = int(np.argmin(misses))
    if misses[i0] > 0.5:
        return None, None, None, False
    lo = thetas[max(i0 - 1, 0)]
    hi = thetas[min(i0 + 1, len(thetas) - 1)]
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if miss(lo, base_yaw) < miss(hi, base_yaw):
            hi = mid
        else:
            lo = mid
    theta = 0.5 * (lo + hi)
    phi = base_yaw
    for _ in range(16):
        traj, _ = traj_of(theta, phi)
        idx = int(np.argmin([np.linalg.norm(p - point) for p, _ in traj]))
        p_min = traj[idx][0]
        err_lat = p_min[1] - point[1]
        dphi = 1e-4
        traj2, _ = traj_of(theta, phi + dphi)
        idx2 = int(np.argmin([np.linalg.norm(p - point) for p, _ in traj2]))
        deriv = (traj2[idx2][0][1] - p_min[1]) / dphi
        if abs(deriv) < 1e-9:
            break
        phi -= float(np.clip(err_lat / deriv, -0.03, 0.03))
    traj, v = traj_of(theta, phi)
    dists = [np.linalg.norm(p - point) for p, _ in traj]
    idx = int(np.argmin(dists))
    ok = dists[idx] < tol
    return v, ok, idx * 0.006, traj


class BallMachine:
    def __init__(self, model, data):
        import mujoco

        self.m = model
        self.d = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_machine")
        self.mocap_id = int(model.body_mocapid[self.body_id])
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "launch_site")
        self.ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_adr = model.jnt_qposadr[self.ball_jnt]
        self.ball_dof_adr = model.jnt_dofadr[self.ball_jnt]

    def set_position(self, pos=None, yaw=None):
        if pos is not None:
            self.d.mocap_pos[self.mocap_id] = np.asarray(pos, float)
        if yaw is not None:
            q = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
            self.d.mocap_quat[self.mocap_id] = q
        import mujoco

        mujoco.mj_kinematics(self.m, self.d)

    def launch_site_pos(self):
        return self.d.site_xpos[self.site_id].copy()

    def plan(self, target_xy, speed, spin_rpm=0.0, spin_topspin=True):
        target_xy = np.asarray(target_xy, float)[:2]
        rel = target_xy - self.launch_site_pos()[:2]
        if np.linalg.norm(rel) < 0.5:
            raise ValueError("target too close to machine")
        yaw_guess = np.arctan2(rel[1], rel[0])
        self.set_position(yaw=yaw_guess)
        import mujoco

        sign = 1.0 if spin_topspin else -1.0
        w_mag = sign * abs(spin_rpm) * 2.0 * np.pi / 60.0
        omega = np.zeros(3)

        def attempt(sp):
            nonlocal omega
            v = None
            res = (False, False, False, False, False)
            for _ in range(2):
                exit_pos = self.launch_site_pos()
                vr, ok, cleared, ranged, toolong, overshoot = solve_launch(
                    exit_pos, target_xy, sp, omega)
                res = (ok, cleared, ranged, toolong, overshoot)
                if not ok:
                    return vr, res
                v = vr
                dir_w = v / np.linalg.norm(v)
                self.d.mocap_quat[self.mocap_id] = aim_quat(dir_w)
                mujoco.mj_kinematics(self.m, self.d)
                omega = w_mag * spin_axis_for(dir_w)
            exit_pos = self.launch_site_pos()
            vr, ok, cleared, ranged, toolong, overshoot = solve_launch(
                exit_pos, target_xy, sp, omega)
            res = (ok, cleared, ranged, toolong, overshoot)
            if ok:
                v = vr
            return v, res

        seq = [float(speed)]
        lo, hi = 6.0, max(1.3 * float(speed), 20.0)
        v = None
        ok = False
        best_v = None
        best_cleared = False
        best_toolong = True
        for _ in range(8):
            sp = seq[-1]
            v, (ok, cleared, ranged, toolong, overshoot) = attempt(sp)
            if ok:
                break
            if v is not None:
                if cleared and not toolong and not best_cleared:
                    best_v, best_cleared, best_toolong = v, True, False
                elif not best_cleared and cleared and best_toolong:
                    best_v, best_cleared, best_toolong = v, True, toolong
                elif best_v is None:
                    best_v = v
            if overshoot or (ranged and (not cleared)):
                hi = sp
            elif ranged and toolong:
                lo = sp
            elif ranged:
                hi = sp
            else:
                lo = sp
            seq.append(0.5 * (lo + hi))
        if not ok:
            if best_v is not None:
                v = best_v
                import sys

                print("[ball_machine] warning: best-effort serve, target %.1fm speed %.1f m/s"
                      % (np.linalg.norm(rel), speed), file=sys.stderr)
        if v is None:
            raise ValueError("no feasible trajectory to target %.1f m" % np.linalg.norm(rel))
        dir_w = v / np.linalg.norm(v)
        self.d.mocap_quat[self.mocap_id] = aim_quat(dir_w)
        mujoco.mj_kinematics(self.m, self.d)
        exit_pos = self.launch_site_pos()
        traj, land_pos, _ = aero.rk4_trajectory(exit_pos, v, omega, 4.0)
        flight_time = (len(traj) - 1) * 0.002
        plan = {
            "exit_pos": exit_pos,
            "velocity": v,
            "omega": omega,
            "speed": float(np.linalg.norm(v)),
            "spin_rpm": float(spin_rpm),
            "predicted_landing": land_pos[:2],
            "flight_time": float(flight_time),
            "dir": dir_w,
        }
        return plan

    def execute_serve(self, plan):
        import mujoco

        qadr = self.ball_qpos_adr
        vadr = self.ball_dof_adr
        self.d.mocap_quat[self.mocap_id] = aim_quat(plan["dir"])
        mujoco.mj_kinematics(self.m, self.d)
        self.d.qpos[qadr:qadr + 3] = plan["exit_pos"] - plan["dir"] * 0.05
        self.d.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[vadr:vadr + 3] = plan["velocity"]
        self.d.qvel[vadr + 3:vadr + 6] = plan["omega"]
        self.d.xfrc_applied[self.ball_body] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def serve_direct(self, aim_point, speed=30.0, spin_rpm=0.0, spin_topspin=True):
        import mujoco

        aim_point = np.asarray(aim_point, float)
        exit_pos = self.launch_site_pos()
        dir_w = aim_point - exit_pos
        dir_w = dir_w / np.linalg.norm(dir_w)
        self.d.mocap_quat[self.mocap_id] = aim_quat(dir_w)
        mujoco.mj_kinematics(self.m, self.d)
        exit_pos = self.launch_site_pos()
        v = speed * dir_w
        axis = spin_axis_for(dir_w)
        sign = 1.0 if spin_topspin else -1.0
        omega = sign * abs(spin_rpm) * 2.0 * np.pi / 60.0 * axis
        plan = {
            "exit_pos": exit_pos,
            "velocity": v,
            "omega": omega,
            "speed": float(speed),
            "spin_rpm": float(spin_rpm),
            "predicted_landing": np.array([np.nan, np.nan]),
            "flight_time": float("nan"),
            "dir": dir_w,
        }
        self.execute_serve(plan)
        return plan

    def serve_through_point(self, point, speed=19.0, spin_rpm=800.0, spin_topspin=True):
        import mujoco

        exit_pos = self.launch_site_pos()
        rel = np.asarray(point)[:2] - exit_pos[:2]
        axis_ref = spin_axis_for(rel / np.linalg.norm(rel))
        sign = 1.0 if spin_topspin else -1.0
        omega = sign * abs(spin_rpm) * 2.0 * np.pi / 60.0 * axis_ref
        yaw_guess = np.arctan2(rel[1], rel[0])
        self.set_position(yaw=yaw_guess)
        v, ok, t_pass, traj = solve_feed_through_point(self.launch_site_pos(), point, speed,
                                                       omega)
        if v is None:
            raise ValueError("cannot feed through point %s" % np.round(point, 2))
        dir_w = v / np.linalg.norm(v)
        self.d.mocap_quat[self.mocap_id] = aim_quat(dir_w)
        mujoco.mj_kinematics(self.m, self.d)
        exit_pos = self.launch_site_pos()
        v2, ok2, t_pass2, traj2 = solve_feed_through_point(exit_pos, point, speed, omega)
        if ok2:
            v, t_pass = v2, t_pass2
            dir_w = v / np.linalg.norm(v)
        plan = {
            "exit_pos": exit_pos - dir_w * 0.05,
            "velocity": v,
            "omega": omega,
            "speed": float(speed),
            "spin_rpm": float(spin_rpm),
            "predicted_landing": np.array([np.nan, np.nan]),
            "flight_time": float(t_pass),
            "dir": dir_w,
        }
        self.execute_serve(plan)
        return plan

    def serve(self, target_xy, mode="flat", speed=None, spin_rpm=None):
        cfg = SERVE_MODES[mode]
        rng = np.random.default_rng()
        if speed is None:
            speed = rng.uniform(*cfg["speed"])
        if spin_rpm is None:
            spin_rpm = rng.uniform(*cfg["spin_rpm"])
        plan = self.plan(target_xy, speed, spin_rpm, cfg["topspin"])
        self.execute_serve(plan)
        return plan
