import numpy as np

import mujoco

from tennis_sim import ball_machine
from tennis_sim import bounce as B
from tennis_sim import constants as C
from tennis_sim import world as W

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

OBS_SPEC = {
    "qpos": ("nu",),
    "qvel": ("nu",),
    "pelvis_pos": (3,),
    "pelvis_quat": (4,),
    "pelvis_linvel": (3,),
    "pelvis_angvel": (3,),
    "ball_pos": (3,),
    "ball_vel": (3,),
    "ball_omega": (3,),
    "racket_pos": (3,),
    "racket_vel": (3,),
}

OBS_DIM = 60


def quat_up_dot(quat):
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), quat)
    return float(up[2])


def land_zone(x, y):
    if abs(x) <= 11.885 + 0.05 and abs(y) <= 5.485 + 0.05:
        zone = "doubles"
        if abs(y) <= 4.115 + 0.05:
            zone = "singles"
        return {"in": True, "zone": zone, "side": "far" if x < 0 else "near", "x": x, "y": y}
    return {"in": False, "zone": "out", "side": "far" if x < 0 else "near", "x": x, "y": y}


class PolicyAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.is_obj = hasattr(policy, "act")

    def reset(self, obs):
        if self.is_obj and hasattr(self.policy, "reset"):
            self.policy.reset(obs)

    def act(self, obs):
        if self.is_obj:
            return self.policy.act(obs)
        return self.policy(obs)


class TennisEnv:
    def __init__(self, scene_path=None, robot_pos=None, machine_pos=None):
        if scene_path is None:
            scene_path = W.write_scene(robot_pos=robot_pos, machine_pos=machine_pos)
        self.model, self.data = W.load_scene(scene_path)
        m = self.model
        self.ball_ctrl = B.BallController(m)
        self.machine = ball_machine.BallMachine(m, self.data)
        self.pelvis_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.racket_site = self.ball_ctrl.racket_site
        self.nu = m.nu
        self.act_qpos_adr = []
        self.act_dof_adr = []
        self.act_joint_names = []
        for i in range(m.nu):
            jid = int(m.actuator_trnid[i, 0])
            self.act_qpos_adr.append(int(m.jnt_qposadr[jid]))
            self.act_dof_adr.append(int(m.jnt_dofadr[jid]))
            self.act_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        self.act_qpos_adr = np.array(self.act_qpos_adr, dtype=int)
        self.act_dof_adr = np.array(self.act_dof_adr, dtype=int)
        self.joint_action_index = {name: i for i, name in enumerate(self.act_joint_names)}
        self.key_stand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.dt = 1.0 / C.CONTROL_HZ
        self.n_substeps = int(round(self.dt / m.opt.timestep))
        self.t = 0.0
        self.policy = None
        self.events = []
        ctrlrange = m.actuator_ctrlrange.copy()
        self.ctrl_lo = ctrlrange[:, 0]
        self.ctrl_hi = ctrlrange[:, 1]

    def set_policy(self, policy):
        self.policy = PolicyAdapter(policy)

    def joint_target_action(self, targets):
        action = np.zeros(self.nu)
        for name, val in targets.items():
            action[self.joint_action_index[name]] = val
        return action

    def reset(self, machine_pos=None, machine_yaw=None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        if machine_pos is not None or machine_yaw is not None:
            self.machine.set_position(pos=machine_pos, yaw=machine_yaw)
        self.data.ctrl[:] = self.data.qpos[self.act_qpos_adr]
        self.ball_ctrl.reset()
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.events = []
        return self.get_obs()

    def get_obs(self):
        d = self.data
        m = self.model
        qpos = d.qpos[self.act_qpos_adr].copy()
        qvel = d.qvel[self.act_dof_adr].copy()
        ppos = d.xpos[self.pelvis_body].copy()
        pquat = d.xquat[self.pelvis_body].copy()
        pvel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_body, pvel, 0)
        ball_pos, ball_vel, ball_w, _ = self.ball_ctrl.ball_states(d)
        rpos = d.site_xpos[self.racket_site].copy()
        rvel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.racket_site, rvel, 0)
        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "pelvis_pos": ppos,
            "pelvis_quat": pquat,
            "pelvis_linvel": pvel[3:6].copy(),
            "pelvis_angvel": pvel[0:3].copy(),
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "ball_omega": ball_w,
            "racket_pos": rpos,
            "racket_vel": rvel[3:6].copy(),
            "time": self.t,
        }
        flat = np.concatenate([qpos, qvel, ppos, pquat, pvel[3:6], pvel[0:3], ball_pos,
                               ball_vel, ball_w, rpos, rvel[3:6]])
        obs["flat"] = flat
        return obs

    def serve_ball(self, target_xy=None, mode="flat", speed=None, spin_rpm=None, plan=None):
        if plan is not None:
            p = self.machine.execute_serve(plan)
        else:
            p = self.machine.serve(target_xy, mode=mode, speed=speed, spin_rpm=spin_rpm)
        self.ball_ctrl.reset()
        return p

    def step(self, action=None, reward=0.0):
        d = self.data
        if action is not None:
            d.ctrl[:] = np.clip(action, self.ctrl_lo, self.ctrl_hi)
        b0 = self.ball_ctrl.ground_bounces
        r0 = self.ball_ctrl.racket_hits
        n0 = self.ball_ctrl.net_hits
        for _ in range(self.n_substeps):
            self.ball_ctrl.step(d)
        self.t += self.dt
        new_events = []
        for _ in range(self.ball_ctrl.ground_bounces - b0):
            ev = dict(self.ball_ctrl.last_bounce)
            ev["type"] = "bounce"
            ev["t"] = self.t
            ev["zone"] = land_zone(ev["pos"][0], ev["pos"][1])
            new_events.append(ev)
        for _ in range(self.ball_ctrl.racket_hits - r0):
            ev = dict(self.ball_ctrl.last_racket_hit)
            ev["type"] = "racket_hit"
            ev["t"] = self.t
            new_events.append(ev)
        for _ in range(self.ball_ctrl.net_hits - n0):
            ev = dict(self.ball_ctrl.last_net_hit)
            ev["type"] = "net_hit"
            ev["t"] = self.t
            new_events.append(ev)
        self.events.extend(new_events)
        obs = self.get_obs()
        pquat = obs["pelvis_quat"]
        terminated = bool(obs["pelvis_pos"][2] < 0.45 or quat_up_dot(pquat) < 0.5)
        truncated = self.t >= self.max_time if hasattr(self, "max_time") else False
        info = {"events": new_events, "all_events": self.events, "time": self.t}
        return obs, reward, terminated or truncated, info

    def run_episode(self, duration, policy=None, serve_schedule=None, on_step=None):
        obs = self.reset()
        adapter = self.policy if policy is None else PolicyAdapter(policy)
        adapter.reset(obs)
        steps = int(duration / self.dt)
        out = []
        for k in range(steps):
            if serve_schedule is not None:
                serve_schedule(self.t, self)
            action = adapter.act(obs)
            obs, reward, term, info = self.step(action)
            if on_step is not None:
                on_step(self, obs, info)
            out.append((obs, reward, term, info))
            if term:
                break
        return out


if __name__ == "__main__":
    env = TennisEnv()
    obs = env.reset()
    print("obs flat dim:", obs["flat"].shape)
    print("nu:", env.nu)
    hold = obs["qpos"].copy()
    for _ in range(100):
        obs, r, term, info = env.step(hold)
    print("after 2s hold: pelvis z=%.3f ball at %s" % (obs["pelvis_pos"][2],
                                                       np.round(obs["ball_pos"], 3)))
    print("racket head at:", np.round(obs["racket_pos"], 3))
