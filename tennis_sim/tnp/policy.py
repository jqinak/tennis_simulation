"""TaskNPoint policy deployment for the tennis simulation.

Re-implements the tasknpoint deploy pipeline (deploy/simulation/control_node/
control_29dof_dynamic.py + simulation_node goal/ball logic, ROS removed) as a
tennis_sim policy object (reset/act interface, 50 Hz):

  obs(164) = [ command(68)                ref joint pos/vel @frame + goal vec(10)
             | motion_anchor_ori_b(6)     pelvis<-reference orientation error (6D)
             | base_ang_vel(3)            pelvis gyro (local frame)
             | joint_pos(29)              q - default_joint_pos
             | joint_vel(29)
             | actions(29) ]              previous raw action

  action(29): q_des = action * action_scale + default_joint_pos (policy joint
  order, remapped to the tennis-sim actuator order by name).

State machine:
  WARMUP  blend stand pose -> reference frame-0 pose (direct position targets)
  READY   ONNX holds frame 0; ball estimator watches for a feed, selects the
          stroke (nearest nominal contact goal) and triggers when the
          predicted time-to-contact drops below the stroke's contact lead time
  SWING   replay the reference at 50 Hz; while frame <= contact_end the
          position goal is overridden with the predicted ball intercept
          (tasknpoint deploy behaviour)
"""
import os

import numpy as np
import onnxruntime as ort

from tennis_sim.tnp import ball_estimator as BE
from tennis_sim.tnp import goals as G
from tennis_sim.tnp.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rot6d,
    rpy_to_quat,
    yaw_quat,
)

STATE_WARMUP = "warmup"
STATE_READY = "ready"
STATE_SWING = "swing"


class TnpPolicy:
    def __init__(self, env, physics="sim", trigger_margin=0.0, warmup_seconds=1.0,
                 verbose=True):
        self.env = env
        self.physics = physics
        self.trigger_margin = float(trigger_margin)
        self.verbose = verbose

        meta = G.load_metadata()
        self.joint_names = meta["joint_names"]
        self.action_scale = np.asarray(meta["action_scale"], dtype=np.float64)
        self.default_joint_pos = np.asarray(meta["default_joint_pos"], dtype=np.float64)
        self.body_names = meta.get("body_names", [])
        self.anchor_body_name = meta.get("anchor_body_name", "pelvis")

        # map policy joint order -> tennis-sim actuator order (by name)
        name_to_act = {n: i for i, n in enumerate(env.act_joint_names)}
        self.perm = np.array([name_to_act[n] for n in self.joint_names], dtype=int)
        self.inv_perm = np.argsort(self.perm)

        # motions
        self.motions = G.load_motions()
        self.motion_names = list(G.MOTION_ORDER)
        self.time_to_contact = []
        self.contact_end_frame = []
        for name in self.motion_names:
            mo = self.motions[name]
            phase = G.MOTION_GOALS[name]["contact_phase"]
            self.time_to_contact.append(phase * mo["num_frames"] / mo["fps"])
            self.contact_end_frame.append(
                int(mo["num_frames"] * (phase + G.CONTACT_DURATION_PHASE)))
            if self.verbose:
                print(f"[tnp] {name}: frames={mo['num_frames']} "
                      f"ttc={self.time_to_contact[-1]:.3f}s "
                      f"contact_end_frame={self.contact_end_frame[-1]}")

        # nominal contact goals in the reset anchor frame (pelvis frame)
        self.nominal_pos_local = np.stack(
            [G.MOTION_GOALS[n]["pos_mean"] for n in self.motion_names])
        self.nominal_vel_local = np.stack(
            [G.MOTION_GOALS[n]["vel_mean"] for n in self.motion_names])
        self.nominal_ori_rpy = np.stack(
            [G.MOTION_GOALS[n]["ori_mean_rpy"] for n in self.motion_names])

        # ONNX session
        self.sess = ort.InferenceSession(G.ONNX_PATH, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        assert self.input_names[0] == "obs" and len(self.input_names) >= 3, self.input_names

        # runtime state
        self.stand_q = None
        self.motion_idx = 0
        self.state = STATE_WARMUP
        self.warmup_steps = int(round(warmup_seconds * 50))
        self.warmup_k = 0
        self.frame = 0
        self.action = np.zeros(29)
        self.ball_launched = False
        self.swing_count = 0
        self.est_target_time = None
        self.est_target_pos = None
        self.goal_pos_w = None
        self.goal_vel_w = None
        self.goal_quat_w = None
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.anchor_quat0 = None
        self.swing_goal_w = None
        self.log = []

    # ------------------------------------------------------------------ utils
    def _v(self, msg):
        if self.verbose:
            print("[tnp] " + msg)

    def _pelvis(self, obs):
        return obs["pelvis_pos"].astype(float), obs["pelvis_quat"].astype(float)

    def _rot(self, quat):
        w, x, y, z = quat
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    # ------------------------------------------------------------- lifecycle
    def reset(self, obs):
        self.stand_q = obs["qpos"].astype(np.float64).copy()
        self.motion_idx = 0
        self.state = STATE_WARMUP
        self.warmup_k = 0
        self.frame = 0
        self.action = np.zeros(29)
        self.ball_launched = False
        self.swing_count = 0
        self.est_target_time = None
        self.est_target_pos = None
        self.swing_goal_w = None
        self.log = []

        pelvis_pos, pelvis_quat = self._pelvis(obs)
        R0 = self._rot(pelvis_quat)
        # yaw alignment between robot heading and motion frame-0 anchor quat
        q_ref0 = self.motions[self.motion_names[self.motion_idx]]["body_quat_w"][0, 0]
        self.init_quat = quat_multiply(yaw_quat(pelvis_quat), quat_conjugate(yaw_quat(q_ref0)))

        # anchor frame -> world goals. Position/orientation goals are anchored
        # at the reset pelvis pose (training: sampled in anchor frame, stored
        # world). The velocity goal is stored UNROTATED: the obs slot carries
        # the training-world value verbatim (deploy does the same), and the
        # training world was aligned with the motion's initial heading.
        self.goal_pos_w = (R0 @ self.nominal_pos_local.T).T + pelvis_pos
        self.goal_vel_w = self.nominal_vel_local.copy()
        self.goal_quat_w = np.stack(
            [quat_multiply(pelvis_quat, rpy_to_quat(rpy)) for rpy in self.nominal_ori_rpy])
        self._v("reset: pelvis=%s state=warmup (%d steps)"
                % (np.round(pelvis_pos, 3), self.warmup_steps))

    def notify_launch(self):
        """Driver calls this right after serving a ball."""
        self.ball_launched = True
        self._v("ball launched")

    def force_trigger(self, motion_idx=None):
        """Debug/test: start the swing without a ball."""
        if motion_idx is not None:
            self.motion_idx = motion_idx
        self.state = STATE_SWING
        self.frame = 0
        self.swing_count += 1
        self._v("FORCE TRIGGER swing #%d motion=%s"
                % (self.swing_count, self.motion_names[self.motion_idx]))

    # ----------------------------------------------------------------- steps
    def _update_estimator(self, obs):
        """Predict the ball trajectory once and evaluate closest approach
        against all three nominal contact goals (tasknpoint deploy semantics:
        select the stroke by the predicted intercept, trigger by its TTC)."""
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
        # stroke selection: nearest predicted intercept among the motions
        best_m, best_hit = None, None
        for m in range(len(self.motion_names)):
            hit = BE.closest_approach(t, p, self.goal_pos_w[m])
            if hit is None:
                continue
            if best_hit is None or hit[0] < best_hit[0]:
                best_m, best_hit = m, hit
        if best_m is not None and self.state != STATE_SWING:
            if best_m != self.motion_idx:
                self._v("motion select (predicted intercept): %s"
                        % self.motion_names[best_m])
            self.motion_idx = best_m
        self.est_target_time = best_hit[0] if best_hit is not None else None
        self.est_target_pos = best_hit[1] if best_hit is not None else None

    def _select_motion(self, obs):
        pass  # selection handled in _update_estimator

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
            # Freeze the position goal at the predicted intercept (training had
            # a constant goal per episode; the deploy-style continuous override
            # makes the tracker slow down as it catches the moving target).
            if self.est_target_pos is not None:
                self.swing_goal_w = self.est_target_pos.copy()
            else:
                self.swing_goal_w = self.goal_pos_w[self.motion_idx].copy()
            self._v("TRIGGER swing #%d motion=%s ttc=%.3f (threshold %.3f) goal=%s"
                    % (self.swing_count, self.motion_names[self.motion_idx],
                       ttc, threshold, np.round(self.swing_goal_w, 3)))

    def _build_goal_vec(self, obs):
        pelvis_pos, pelvis_quat = self._pelvis(obs)
        R = self._rot(pelvis_quat)
        q_inv = quat_conjugate(pelvis_quat)
        mo = self.motion_names[self.motion_idx]
        use_ball = (self.state == STATE_SWING
                    and self.frame <= self.contact_end_frame[self.motion_idx]
                    and self.swing_goal_w is not None)
        if use_ball:
            p_goal_w = self.swing_goal_w
        else:
            p_goal_w = self.goal_pos_w[self.motion_idx]
        pos_b = R.T @ (p_goal_w - pelvis_pos)
        vel_w = self.goal_vel_w[self.motion_idx]
        ori_b = quat_multiply(q_inv, self.goal_quat_w[self.motion_idx])
        return np.concatenate([pos_b, vel_w, ori_b])

    def _build_obs(self, obs):
        mo = self.motions[self.motion_names[self.motion_idx]]
        f = min(self.frame, mo["num_frames"] - 1)
        command = np.concatenate([mo["joint_pos"][f], mo["joint_vel"][f],
                                  self._build_goal_vec(obs)])

        # motion anchor (pelvis) orientation error, 6D
        pelvis_pos, pelvis_quat = self._pelvis(obs)
        q_ref = mo["body_quat_w"][f, 0]
        ref_corrected = quat_multiply(self.init_quat, q_ref)
        rel = quat_multiply(quat_conjugate(pelvis_quat), ref_corrected)
        anchor_ori_b = quat_to_rot6d(rel)

        # pelvis gyro (local frame)
        R = self._rot(pelvis_quat)
        base_ang_vel_b = R.T @ obs["pelvis_angvel"].astype(float)

        qj = obs["qpos"].astype(np.float64) - self.default_joint_pos
        dqj = obs["qvel"].astype(np.float64)

        return np.concatenate([command, anchor_ori_b, base_ang_vel_b, qj, dqj,
                               self.action]).astype(np.float32)

    def _run_onnx(self, obs):
        x = self._build_obs(obs)
        feed = {self.input_names[0]: x.reshape(1, -1),
                "which_motion": np.array([[self.motion_idx]], dtype=np.float32),
                "time_step": np.array([[self.frame]], dtype=np.float32)}
        out = self.sess.run(None, feed)
        self.action = np.asarray(out[0][0], dtype=np.float64)
        return self.action

    def _action_to_env(self, raw_action):
        """policy joint order raw action -> tennis-sim ctrl targets (our order)."""
        q_des = raw_action * self.action_scale + self.default_joint_pos
        q_des = q_des[self.perm]
        q_des = np.clip(q_des, self.env.ctrl_lo, self.env.ctrl_hi)
        return q_des

    # ------------------------------------------------------------------- act
    def act(self, obs):
        self._update_estimator(obs)
        self._maybe_trigger(obs)

        if self.state == STATE_WARMUP:
            u = np.clip(self.warmup_k / max(self.warmup_steps - 1, 1), 0.0, 1.0)
            s = u * u * (3.0 - 2.0 * u)
            target = (1.0 - s) * self.stand_q + s * self.motions[
                self.motion_names[self.motion_idx]]["joint_pos"][0][self.inv_perm]
            self.warmup_k += 1
            if self.warmup_k >= self.warmup_steps:
                self.state = STATE_READY
                self._v("warmup done -> ready (frame-0 hold)")
            return np.clip(target, self.env.ctrl_lo, self.env.ctrl_hi)

        if self.state == STATE_READY:
            self.frame = 0
            raw = self._run_onnx(obs)
            return self._action_to_env(raw)

        # STATE_SWING
        raw = self._run_onnx(obs)
        mo = self.motions[self.motion_names[self.motion_idx]]
        self.log.append({
            "t": obs["time"], "state": self.state, "motion": self.motion_idx,
            "frame": self.frame, "racket_pos": obs["racket_pos"].copy(),
            "ball_pos": obs["ball_pos"].copy(),
            "est_ttc": self.est_target_time,
            "est_target": None if self.est_target_pos is None else self.est_target_pos.copy(),
        })
        if self.frame >= mo["num_frames"] - 1:
            self.state = STATE_READY
            self.ball_launched = False
            self._v("swing done -> ready")
        else:
            self.frame += 1
        return self._action_to_env(raw)
