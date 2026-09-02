import numpy as np

import mujoco

from tennis_sim import aero
from tennis_sim import constants as C

GROUND_Z = (0.0, 0.0, 1.0)


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def restitution_for(speed):
    e = C.RESTITUTION_INTERCEPT + C.RESTITUTION_SLOPE * speed
    return float(np.clip(e, C.RESTITUTION_MIN, C.RESTITUTION_MAX))



def ground_bounce_velocity(v_pre, w_pre):
    v = np.array(v_pre, dtype=float).copy()
    w = np.array(w_pre, dtype=float).copy()
    if v[2] >= -0.3:
        return v, w
    e = restitution_for(-v[2])
    v[2] = -e * v[2]
    u = np.array([v_pre[0], v_pre[1], 0.0])
    u_mag = float(np.linalg.norm(u))
    if u_mag > 0.3:
        t_hat = u / u_mag
        axis = np.cross(GROUND_Z, t_hat)
        w_spin = float(np.dot(w_pre, axis))
        u_out = C.BOUNCE_TANGENTIAL_KEEP * u_mag + C.BOUNCE_SPIN_KICK * C.BALL_RADIUS * w_spin
        v = v - float(np.dot(v, t_hat)) * t_hat + u_out * t_hat
        w = C.BOUNCE_SPIN_KEEP * w_pre
    return v, w


class BallController:
    def __init__(self, model):
        import re as _re

        self.model = model
        self.ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.qadr = model.jnt_qposadr[self.ball_jnt]
        self.vadr = model.jnt_dofadr[self.ball_jnt]
        self.ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.net_geoms = set()
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None and _re.match(r"net_collision_\d+", name):
                self.net_geoms.add(i)
        self.racket_geoms = set()
        for gname in ("racket_strings", "racket_handle"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self.racket_geoms.add(gid)
        self.racket_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_head_site")
        self.reset()

    def reset(self):
        self.was_ground = False
        self.was_racket = False
        self.was_net = False
        self.v_pre = None
        self.w_pre = None
        self.vr_pre = None
        self.last_bounce = None
        self.last_racket_hit = None
        self.last_net_hit = None
        self.ground_bounces = 0
        self.racket_hits = 0
        self.net_hits = 0
        self.v_prev_obs = None
        self.w_prev_obs = None
        self.racket_normal_pre = None

    def ball_states(self, data):
        q = data.qpos[self.qadr:self.qadr + 7]
        pos = q[:3].copy()
        v = data.qvel[self.vadr:self.vadr + 3].copy()
        w_local = data.qvel[self.vadr + 3:self.vadr + 6]
        w = np.zeros(3)
        mujoco.mju_rotVecQuat(w, w_local, q[3:7])
        return pos, v, w, q[3:7].copy()

    def _scan_contacts(self, data):
        ground = False
        racket = False
        net = False
        racket_normal = None
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            pair = {g1, g2}
            if self.ball_geom not in pair:
                continue
            if self.ground_geom in pair:
                ground = True
            elif pair & self.net_geoms:
                net = True
            elif pair & self.racket_geoms:
                racket = True
                n = np.array(con.frame[0:3], dtype=float).copy()
                racket_normal = n
        return ground, racket, net, racket_normal

    def _racket_site_velocity(self, data):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.racket_site,
                                 res, 0)
        return res[3:6].copy()

    def observe_and_correct(self, data):
        ground, racket, net, racket_normal = self._scan_contacts(data)
        pos, v, w, _ = self.ball_states(data)

        v_ref = self.v_prev_obs if self.v_prev_obs is not None else v
        w_ref = self.w_prev_obs if self.w_prev_obs is not None else w

        if net and not self.was_net:
            self.net_hits += 1
            self.last_net_hit = {"pos": pos.copy(), "v": v.copy()}
        if ground and not self.was_ground:
            self.v_pre = v_ref.copy()
            self.w_pre = w_ref.copy()
        if racket and not self.was_racket:
            self.v_pre = v_ref.copy()
            self.w_pre = w_ref.copy()
            self.vr_pre = self._racket_site_velocity(data)
            self.racket_normal_pre = racket_normal

        if not ground and self.was_ground and self.v_pre is not None:
            if self.v_pre[2] < -0.3:
                v_target, w_target = ground_bounce_velocity(self.v_pre, self.w_pre)
                _, _, w_cur, quat = self.ball_states(data)
                data.qvel[self.vadr:self.vadr + 3] = v_target
                w_local_new = np.zeros(3)
                mujoco.mju_rotVecQuat(w_local_new, w_target - w_cur, _quat_conj(quat))
                data.qvel[self.vadr + 3:self.vadr + 6] += w_local_new
                v = data.qvel[self.vadr:self.vadr + 3].copy()
                self.ground_bounces += 1
                self.last_bounce = {
                    "pos": pos.copy(),
                    "v_pre": self.v_pre.copy(),
                    "v_post": v.copy(),
                    "w_pre": self.w_pre.copy(),
                    "restitution": restitution_for(-self.v_pre[2]),
                }
            self.v_pre = None
            self.w_pre = None

        if not racket and self.was_racket and self.v_pre is not None:
            n = self.racket_normal_pre
            if n is not None:
                vr_post = self._racket_site_velocity(data)
                v_rel_pre = self.v_pre - self.vr_pre
                v_rel_post = v - vr_post
                if float(np.dot(v_rel_pre, n)) > 0:
                    n = -n
                ap = float(np.dot(v_rel_pre, n))
                if ap < -0.3:
                    v_rel_n_post = float(np.dot(v_rel_post, n))
                    target = -C.RACKET_RESTITUTION * ap
                    delta = (target - v_rel_n_post)
                    data.qvel[self.vadr:self.vadr + 3] += delta * n
                    v = data.qvel[self.vadr:self.vadr + 3].copy()
                    self.racket_hits += 1
                    self.last_racket_hit = {
                        "pos": pos.copy(),
                        "ball_speed_in": float(np.linalg.norm(self.v_pre)),
                        "racket_speed": float(np.linalg.norm(self.vr_pre)),
                        "ball_speed_out": float(np.linalg.norm(v)),
                        "v_out": v.copy(),
                    }
            self.v_pre = None
            self.w_pre = None
            self.vr_pre = None

        self.was_ground = ground
        self.was_racket = racket
        self.was_net = net
        self.v_prev_obs = v.copy()
        self.w_prev_obs = w.copy()

    def apply_aero(self, data):
        _, v, w, _ = self.ball_states(data)
        f = aero.ball_aero_force(v, w)
        tau = aero.spin_decay_torque(w)
        data.xfrc_applied[self.ball_body, :3] = f
        data.xfrc_applied[self.ball_body, 3:] = tau

    def step(self, data):
        self.observe_and_correct(data)
        self.apply_aero(data)
        mujoco.mj_step(self.model, data)
