import numpy as np

import mujoco

from tennis_sim import aero
from tennis_sim import bounce as B
from tennis_sim import constants as C
from tennis_sim.env import RIGHT_ARM_JOINTS, WAIST_JOINTS


def predict_ball_contact(plan, target_z=0.92, max_t=3.0):
    pos = np.array(plan["exit_pos"], dtype=float) - plan["dir"] * 0.05
    vel = np.array(plan["velocity"], dtype=float)
    omega = np.array(plan["omega"], dtype=float)
    t_total = 0.0
    for phase in range(2):
        traj, end_pos, end_vel = aero.rk4_trajectory(pos, vel, omega, max_t, dt=0.001)
        if phase == 0:
            ground_hit = None
            for i, (p, v) in enumerate(traj):
                if p[2] <= C.BALL_RADIUS and i > 0:
                    ground_hit = i
                    break
            if ground_hit is None:
                return None
            p_hit, v_hit = traj[ground_hit]
            w_hit = omega * np.exp(-C.SPIN_DECAY_RATE * (len(traj) - 1) * 0.001)
            vel, omega = B.ground_bounce_velocity(v_hit, w_hit)
            pos = np.array([p_hit[0], p_hit[1], max(p_hit[2], C.BALL_RADIUS)])
            t_total += ground_hit * 0.001
        else:
            for i, (p, v) in enumerate(traj):
                if p[2] >= target_z and i > 0 and v[2] > 0:
                    return {"pos": p, "vel": v, "t": t_total + i * 0.001}
                if i > 0 and v[2] < 0 and p[2] < target_z:
                    best = min(traj[:i + 1], key=lambda pv: abs(pv[0][2] - target_z))
                    return {"pos": best[0], "vel": best[1], "t": t_total + i * 0.001}
            return None
    return None


def solve_ik(model, base_qpos, site_id, target_pos, qpos_adr, dof_adr, q_init, iters=200,
             tol=0.002, desired_normal=None, q_ref=None, posture_k=0.04):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = base_qpos
    q = np.array(q_init, dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        for k, adr in enumerate(qpos_adr):
            scratch.qpos[adr] = q[k]
        mujoco.mj_forward(model, scratch)
        err = np.asarray(target_pos, dtype=float) - scratch.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        J = jacp[:, dof_adr]
        lam = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        step = np.linalg.norm(dq)
        if step > 0.35:
            dq *= 0.35 / step
        if q_ref is not None:
            dq = dq + posture_k * (np.asarray(q_ref) - q)
        q = np.clip(q + dq, -3.14, 3.14)
    if desired_normal is not None:
        desired_normal = np.asarray(desired_normal, dtype=float)
        wrist_start = 4
        for it in range(120):
            for k, adr in enumerate(qpos_adr):
                scratch.qpos[adr] = q[k]
            mujoco.mj_forward(model, scratch)
            xmat = scratch.site_xmat[site_id].reshape(3, 3)
            cur_n = xmat[:, 2]
            e_rot = np.cross(cur_n, desired_normal)
            if np.linalg.norm(e_rot) < 0.08:
                break
            err_p = np.asarray(target_pos, dtype=float) - scratch.site_xpos[site_id]
            mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
            Jw = jacr[:, dof_adr[wrist_start:]]
            dq_w = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-4 * np.eye(3), e_rot)
            step = np.linalg.norm(dq_w)
            if step > 0.2:
                dq_w *= 0.2 / step
            for k in range(wrist_start, len(qpos_adr)):
                q[k] += dq_w[k - wrist_start]
            for _ in range(3):
                for k, adr in enumerate(qpos_adr):
                    scratch.qpos[adr] = q[k]
                mujoco.mj_forward(model, scratch)
                err_p = np.asarray(target_pos, dtype=float) - scratch.site_xpos[site_id]
                mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
                J = jacp[:, dof_adr]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err_p)
                step = np.linalg.norm(dq)
                if step > 0.2:
                    dq *= 0.2 / step
                q += dq
    return q


def solve_ik6d(model, base_qpos, site_id, target_pos, qpos_adr, dof_adr, q_init,
               joint_names, desired_normal, restarts=10, seed=0, q_ref=None,
               posture_k=0.03):
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    lo = model.jnt_range[jid][:, 0].copy()
    hi = model.jnt_range[jid][:, 1].copy()
    rng = np.random.default_rng(seed)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = base_qpos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    n_des = np.asarray(desired_normal, dtype=float)

    def run(q0, iters=320):
        q = np.clip(np.array(q0, dtype=float), lo, hi)
        best = (1e9, q.copy())
        for _ in range(iters):
            for k, adr in enumerate(qpos_adr):
                scratch.qpos[adr] = q[k]
            mujoco.mj_forward(model, scratch)
            ep = np.asarray(target_pos, dtype=float) - scratch.site_xpos[site_id]
            cur = scratch.site_xmat[site_id].reshape(3, 3)[:, 2]
            er = np.cross(cur, n_des)
            cost = np.linalg.norm(ep) + 0.45 * np.linalg.norm(er)
            if cost < best[0]:
                best = (cost, q.copy())
            if np.linalg.norm(ep) < 0.03 and np.linalg.norm(er) < 0.12:
                break
            mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
            J = np.vstack([jacp[:, dof_adr], 0.5 * jacr[:, dof_adr]])
            e6 = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e6)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            if q_ref is not None:
                dq = dq + posture_k * (np.asarray(q_ref, dtype=float) - q)
            q = np.clip(q + dq, lo, hi)
        return best

    inits = [q_init, np.clip(np.array(q_init) + np.array([0.9, 0.1, -0.6, 0.4, 1.2, 0.8, 0.0]),
                             lo, hi)]
    for _ in range(restarts):
        inits.append(np.clip(np.array(q_init) + rng.uniform(-0.9, 0.9, len(q_init)), lo, hi))
    best = (1e9, None)
    for q0 in inits:
        cost, q = run(q0)
        if cost < best[0]:
            best = (cost, q)
    return best[1]


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def plan_feed_for_contact(env, contact_point, speed=15.0, spin=2400.0, iters=2):
    contact_point = np.asarray(contact_point, dtype=float)
    target = contact_point[:2] - np.array([2.4, 0.05])
    plan = None
    for _ in range(iters):
        plan = env.machine.plan(target, speed, spin, True)
        c = predict_ball_contact(plan, target_z=contact_point[2])
        if c is None:
            break
        delta = contact_point[:2] - c["pos"][:2]
        if np.linalg.norm(delta) < 0.05:
            break
        target = target + delta
    return plan


class ScriptedForehand:
    def __init__(self, env, serve_t=0.8, contact_z=0.95, feed_speed=15.0, feed_spin=2400.0,
                 feed_hint=(8.5, 1.5)):
        self.env = env
        self.serve_t = serve_t
        self.contact_z = contact_z
        m = env.model
        self.site_id = env.ball_ctrl.racket_site
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_ARM_JOINTS]
        self.qadr = [int(m.jnt_qposadr[j]) for j in jid]
        self.dofadr = [int(m.jnt_dofadr[j]) for j in jid]
        wjid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in WAIST_JOINTS]
        self.waist_idx = [env.joint_action_index[mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in wjid]
        self.ready = env.data.qpos[self.qadr].copy()
        self.base_frozen = env.data.qpos[env.act_qpos_adr].copy()
        self.px, self.py = env.data.qpos[7], env.data.qpos[8]
        self.nominal_contact = np.array([self.px - 0.55, self.py + 0.25, contact_z])
        self.face_normal = np.array([-0.85, -0.06, 0.32])
        self.face_normal /= np.linalg.norm(self.face_normal)
        n_des = self.face_normal.copy()
        self._build_swing(self.nominal_contact, serve_t + 1.75, swing_span=0.35)
        t_swing, r_swing, v_swing, n_swing = self._rehearse_swing()
        dist_p = np.linalg.norm(r_swing - self.nominal_contact, axis=1)
        close = np.where(dist_p < 0.20)[0]
        if len(close) == 0:
            close = np.array([int(np.argmin(dist_p))])
        i_best = close[np.argmin(dist_p[close])]
        t_hit = float(t_swing[i_best])
        r_hit = r_swing[i_best]
        speed = float(np.linalg.norm(v_swing[i_best]))
        # Tennis rules: the receiver may not volley — the feed must bounce once
        # on the robot's side before the stroke. Plan the feed through one
        # bounce so the ball rises into the contact point afterwards.
        lo, hi = 8.0, 30.0
        plan = None
        contact = None
        for it in range(8):
            speed_try = 0.5 * (lo + hi) if it else feed_speed
            try:
                plan = plan_feed_for_contact(self.env, r_hit, speed=speed_try,
                                             spin=feed_spin)
                c = predict_ball_contact(plan, target_z=r_hit[2])
            except ValueError:
                lo = speed_try
                continue
            if c is None:
                lo = speed_try
                continue
            ball_t = self.serve_t + c["t"]
            err = ball_t - t_hit
            print("[scripted_forehand] feed iter %d: speed=%.1f ball_t=%.3f err=%+.3f"
                  % (it, speed_try, ball_t, err))
            contact = c
            if abs(err) < 0.05:
                break
            if err < 0:
                hi = speed_try
            else:
                lo = speed_try
        if plan is None:
            plan = plan_feed_for_contact(self.env, r_hit, speed=feed_speed, spin=feed_spin)
            contact = predict_ball_contact(plan, target_z=r_hit[2])
        if contact is not None:
            ball_t = self.serve_t + contact["t"]
        else:
            ball_t = t_hit
        # Timing scan against the full-physics rehearsal (shape kept fixed,
        # only the time base moves, so each try is cheap).
        self._build_swing(self.nominal_contact, ball_t, swing_span=0.35)
        best = None
        for round_ in range(2):
            for dt_try in (0.0, 0.07, -0.07, 0.14, -0.14, 0.21, -0.21):
                self.set_swing_time(ball_t + dt_try)
                out = self._rehearse_full(plan, self.face_normal)
                print("[scripted_forehand] tune r%d dt=%+.3f hit=%s crossed=%s net=%s"
                      % (round_, dt_try, out["hit"], out["crossed"], out["net"]))
                score = 0 if (out["hit"] and out["crossed"]) else (1 if out["hit"] else 2)
                if best is None or score < best[0]:
                    best = (score, ball_t + dt_try)
                if score == 0:
                    break
            if best[0] == 0:
                break
            n_new = np.array([-0.85, -0.05, min(0.85, self.face_normal[2] + 0.12)])
            n_new /= np.linalg.norm(n_new)
            self.face_normal = n_new
            self._build_swing(self.nominal_contact, best[1], swing_span=0.35)
        self.set_swing_time(best[1])
        self.plan = plan
        self.contact_pos = r_hit.copy() if r_hit is not None else self.nominal_contact.copy()
        self.t_contact = best[1]
        self.swing_speed = speed

    def _build_swing(self, contact_point, t_mid, swing_span=0.40):
        px, py = self.px, self.py
        if not hasattr(self, "face_normal") or self.face_normal is None:
            self.face_normal = np.array([-0.88, -0.10, 0.42])
            self.face_normal /= np.linalg.norm(self.face_normal)
        backswing = np.array([px - 0.15, py + 0.45, 1.02])
        # long, fast strike arc so the racket carries real speed through the
        # contact point (the smoothstep path dips to ~0 speed at waypoints)
        pre = contact_point + np.array([0.55, 0.18, 0.06])
        post = contact_point + np.array([-0.90, 0.10, 0.16])
        wrap = contact_point + np.array([-0.45, -0.30, 0.60])
        waypoints = [
            ("ready", np.array([px - 0.55, py + 0.12, 1.00])),
            ("backswing", backswing),
            ("pre", pre),
            ("contact", contact_point),
            ("post", post),
            ("wrap", wrap),
            ("ready", np.array([px - 0.55, py + 0.12, 1.00])),
        ]
        t_c = t_mid
        fast = 0.13
        times = [t_c - 10.0, t_c - 0.55, t_c - fast, t_c,
                 t_c + fast, t_c + 0.45,
                 t_c + 0.95]
        m = self.env.model
        base_qpos = self.env.data.qpos.copy()
        joint_traj = []
        q_prev = self.ready
        for (name, target), t in zip(waypoints, times):
            if name in ("pre", "contact", "post"):
                # keep the face normal through the whole strike zone so the
                # ball can only meet the string bed, never a turned-around face
                q = solve_ik6d(m, base_qpos, self.site_id, target, self.qadr, self.dofadr,
                               q_prev, RIGHT_ARM_JOINTS, self.face_normal,
                               q_ref=q_prev, restarts=4)
            else:
                q = solve_ik(m, base_qpos, self.site_id, target, self.qadr, self.dofadr,
                             q_prev, q_ref=q_prev)
            q_prev = q
            joint_traj.append((t, q))
        waist_angles = [0.0, 0.42, 0.30, 0.05, -0.25, -0.42, 0.0]
        # shape (relative times) separated from the time base, so the timing
        # scan can move the swing without redoing any IK
        self._swing_rel = [(t - t_c, q) for (t, q) in joint_traj]
        self._waist_rel = [(t - t_c, a) for (t, _), a in zip(joint_traj, waist_angles)]
        self._swing_t = t_c

    def set_swing_time(self, t_mid):
        self._swing_t = t_mid

    def _rehearse_full(self, plan, face_normal):
        import mujoco as _mj

        m = self.env.model
        data = _mj.MjData(m)
        data.qpos[:] = self.env.data.qpos
        data.qvel[:] = self.env.data.qvel
        data.ctrl[:] = self.base_frozen
        ctrl = B.BallController(m)
        qadr, vadr = ctrl.qadr, ctrl.vadr
        bid = ctrl.ball_body
        _mj.mj_forward(m, data)
        hit_t = None
        crossed = False
        net_hit = False
        fired = False
        steps = int(4.0 / self.env.dt)
        for k in range(steps):
            t = k * self.env.dt
            if not fired and t >= self.serve_t:
                data.qpos[qadr:qadr + 3] = plan["exit_pos"]
                data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[vadr:vadr + 3] = plan["velocity"]
                data.qvel[vadr + 3:vadr + 6] = plan["omega"]
                data.xfrc_applied[bid] = 0.0
                ctrl.reset()
                fired = True
                _mj.mj_forward(m, data)
            action = self._action_from_traj({"time": t})
            data.ctrl[:] = np.clip(action, self.env.ctrl_lo, self.env.ctrl_hi)
            for _ in range(self.env.n_substeps):
                ctrl.step(data)
            if ctrl.racket_hits > 0 and hit_t is None:
                hit_t = t
            if ctrl.net_hits > 0:
                net_hit = True
            if hit_t is not None:
                pos = data.qpos[qadr:qadr + 3]
                if pos[0] < -0.5:
                    crossed = True
                    break
        return {"hit": hit_t is not None, "crossed": crossed, "net": net_hit}

    def _rehearse_swing(self):
        import mujoco as _mj

        m = self.env.model
        data = _mj.MjData(m)
        data.qpos[:] = self.env.data.qpos
        data.qvel[:] = self.env.data.qvel
        data.ctrl[:] = self.base_frozen
        ctrl = B.BallController(m)
        t_start = self.serve_t + 1.3
        t_end = self.serve_t + 2.4
        ts, rs, vs, ns = [], [], [], []
        n = int((t_end) / self.env.dt)
        for k in range(n):
            t = k * self.env.dt
            action = self._action_from_traj({"time": t})
            data.ctrl[:] = np.clip(action, self.env.ctrl_lo, self.env.ctrl_hi)
            for _ in range(self.env.n_substeps):
                ctrl.step(data)
            if t_start <= t <= t_end:
                ts.append(t)
                rs.append(data.site_xpos[self.site_id].copy())
                vel = np.zeros(6)
                _mj.mj_objectVelocity(m, data, _mj.mjtObj.mjOBJ_SITE, self.site_id, vel, 0)
                vs.append(vel[3:6].copy())
                ns.append(data.site_xmat[self.site_id].reshape(3, 3)[:, 2].copy())
        return np.array(ts), np.array(rs), np.array(vs), np.array(ns)

    def _solve_feed(self, r_hit, t_hit, speed0, spin, hint, iters=6):
        target_t = t_hit - self.serve_t - 0.03
        lo, hi = 6.0, 34.0
        best = None
        speed = float(np.clip(speed0, lo, hi))
        for i in range(iters):
            plan = plan_feed_for_contact(self.env, r_hit, speed=speed, spin=spin)
            c = predict_ball_contact(plan, target_z=r_hit[2])
            if c is None:
                lo = speed
                speed = 0.5 * (speed + hi)
                continue
            err = c["t"] - target_t
            if best is None or abs(err) < best[0]:
                best = (abs(err), plan)
            if abs(err) < 0.03:
                break
            if err > 0:
                lo = speed
                speed = 0.5 * (speed + hi)
            else:
                hi = speed
                speed = 0.5 * (speed + lo)
        if best is None:
            raise RuntimeError("could not solve a feed reaching the racket sweep point")
        return best[1]

    def _action_from_traj(self, obs):
        t = obs["time"]
        action = self.base_frozen.copy()
        arm = self._interp(self._swing_rel, t - self._swing_t, self.ready)
        for k, adr in enumerate(self.qadr):
            action[self.env.joint_action_index[RIGHT_ARM_JOINTS[k]]] = arm[k]
        waist = self._interp(self._waist_rel, t - self._swing_t, 0.0, scalar=True)
        action[self.waist_idx[0]] = waist
        return action

    def reset(self, obs):
        pass

    def act(self, obs):
        return self._action_from_traj(obs)

    def _interp(self, traj, t, default, scalar=False):
        ts = [x[0] for x in traj]
        if t <= ts[0]:
            return default if default is not None else traj[0][1]
        for i in range(len(ts) - 1):
            if ts[i] <= t < ts[i + 1]:
                u = smoothstep((t - ts[i]) / max(ts[i + 1] - ts[i], 1e-6))
                a, b = traj[i][1], traj[i + 1][1]
                return a + (b - a) * u
        return traj[-1][1]
