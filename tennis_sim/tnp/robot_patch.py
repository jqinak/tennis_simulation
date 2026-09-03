"""Patch the tennis-sim G1 actuators to match the TaskNPoint training dynamics.

The policy was trained (mjlab) against <position> actuators with per-joint
kp/kd (mjlab motor model, omega=10Hz, zeta=2.0) and per-joint effort limits,
plus motor armature. The Menagerie G1 in this repo ships kp=500 actuators
(much stiffer than training), so at load time we retune:

  gainprm[0]  = kp
  biasprm[1]  = -kp
  biasprm[2]  = -kd
  forcerange  = +-effort_limit (and joint actuatorfrcrange likewise)
  dof_armature = armature from the tasknpoint training XML (matched per joint)

Also clamps the waist roll/pitch ctrl targets to +-0.001 rad because the
training XML clamps those joints to the same range (the policy never learned
to use them).
"""
import numpy as np
import mujoco

from tennis_sim.tnp import goals as G

TASKNPOINT_ARMATURE = {
    # mjlab G1 motor model (reflected rotor inertia per joint, computed from
    # mjlab.asset_zoo.robots.unitree_g1.g1_constants; kp/kd recomputed from
    # these match the ONNX-embedded joint_stiffness/joint_damping exactly)
    "hip_pitch_joint": 0.01017752,
    "hip_roll_joint": 0.025101925,
    "hip_yaw_joint": 0.01017752,
    "knee_joint": 0.025101925,
    "ankle_pitch_joint": 0.00721945,
    "ankle_roll_joint": 0.00721945,
    "shoulder_pitch_joint": 0.003609725,
    "shoulder_roll_joint": 0.003609725,
    "shoulder_yaw_joint": 0.003609725,
    "elbow_joint": 0.003609725,
    "wrist_roll_joint": 0.003609725,
    "wrist_pitch_joint": 0.00425,
    "wrist_yaw_joint": 0.00425,
    "waist_yaw_joint": 0.01017752,
    "waist_roll_joint": 0.00721945,
    "waist_pitch_joint": 0.00721945,
}


def patch_robot_to_tasknpoint(env, apply_armature=True, verbose=True):
    m = env.model
    meta = G.load_metadata()
    kp = np.asarray(meta["joint_stiffness"], dtype=float)
    kd = np.asarray(meta["joint_damping"], dtype=float)
    scale = np.asarray(meta["action_scale"], dtype=float)
    default = np.asarray(meta["default_joint_pos"], dtype=float)
    effort = G.EFFORT_LIMITS
    joint_names = meta["joint_names"]

    # sanity: action scale/default/effort must be 29 long
    assert len(kp) == len(kd) == m.nu == len(effort), (len(kp), len(kd), m.nu)

    # map policy joint order -> our actuator order by name
    name_to_act = {name: i for i, name in enumerate(env.act_joint_names)}
    perm = []
    for name in joint_names:
        assert name in name_to_act, f"joint {name} missing in tennis sim model"
        perm.append(name_to_act[name])
    perm = np.asarray(perm, dtype=int)

    for policy_i, act_i in enumerate(perm):
        m.actuator_gainprm[act_i, 0] = kp[policy_i]
        m.actuator_biasprm[act_i, 1] = -kp[policy_i]
        m.actuator_biasprm[act_i, 2] = -kd[policy_i]
        m.actuator_forcerange[act_i] = (-effort[policy_i], effort[policy_i])
        jid = int(m.actuator_trnid[act_i, 0])
        if m.jnt_actfrcrange.shape[1] == 2:
            m.jnt_actfrcrange[jid] = (-effort[policy_i], effort[policy_i])
        if apply_armature:
            base = joint_names[policy_i]
            for suffix, arm in TASKNPOINT_ARMATURE.items():
                if base.endswith(suffix):
                    m.dof_armature[m.jnt_dofadr[jid]] = arm
                    break

    # waist roll/pitch joint ranges clamp (training XML: +-0.001)
    waist_clamp = np.array([-G.WAIST_ROLL_PITCH_LIMIT, G.WAIST_ROLL_PITCH_LIMIT])
    waist_mask = np.zeros(m.nu, dtype=bool)
    for i, name in enumerate(env.act_joint_names):
        if name in ("waist_roll_joint", "waist_pitch_joint"):
            waist_mask[i] = True
            jid = int(m.actuator_trnid[i, 0])
            m.jnt_range[jid] = (waist_clamp[0], waist_clamp[1])
    env.ctrl_lo = np.where(waist_mask, waist_clamp[0], m.actuator_ctrlrange[:, 0])
    env.ctrl_hi = np.where(waist_mask, waist_clamp[1], m.actuator_ctrlrange[:, 1])

    # foot friction: training XML used MuJoCo default friction 1.0 on the sole
    # capsules; the Menagerie foot spheres ship with friction 0.6 (priority=1
    # dominates the pair). Raise foot sliding friction to 1.0 to match training.
    n_foot = 0
    for i in range(m.ngeom):
        bid = m.geom_bodyid[i]
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "ankle_roll_link" in bname and m.geom_contype[i] != 0:
            m.geom_friction[i, 0] = 1.0
            n_foot += 1
    if verbose:
        print("[tnp] foot friction -> 1.0 on %d geoms" % n_foot)

    if verbose:
        print("[tnp] patched actuators to tasknpoint training gains:")
        for i, name in enumerate(env.act_joint_names):
            j = perm[i]
            print("  %-28s kp=%7.3f kd=%6.3f effort=%5.1f scale=%.3f default=%+.3f"
                  % (name, kp[j], kd[j], effort[j], scale[j], default[j]))
    return {"perm": perm, "kp": kp, "kd": kd, "scale": scale,
            "default": default, "effort": effort, "joint_names": joint_names}


# TaskNPoint training racket mount: body "tennis_racket" parented to
# right_wrist_yaw_link at (0.08, 0, 0), shaft along the wrist +z axis
# (quat = ry(-90 deg)), mass 0.1 kg, inertia 0.001/0.01/0.01 @ 0.35 m.
RACKET_TRAIN_MOUNT_POS = (0.08, 0.0, 0.0)
RACKET_TRAIN_MOUNT_QUAT = "0.7071068 0 -0.7071068 0"
RACKET_TRAIN_INERTIAL_POS = (0.35, 0.0, 0.0)
RACKET_TRAIN_MASS = 0.1
RACKET_TRAIN_DIAGINERTIA = "0.001 0.01 0.01"


def patch_racket_to_tasknpoint(env, verbose=True):
    """Runtime alignment of THIS project's racket with the TaskNPoint training
    mount (the same way patch_robot_to_tasknpoint realigns the actuators).

    The built scene keeps its 23-inch racket geometry/materials; only the
    mount pose and inertial of the loaded model are set to the values the
    policy was trained with, so the reference strokes present the racket face
    to the ball exactly like in training. No asset file is modified."""
    import mujoco

    m = env.model
    mount_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "racket_mount")
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "racket")
    assert mount_id >= 0 and body_id >= 0, "racket bodies not found"
    m.body_pos[mount_id] = RACKET_TRAIN_MOUNT_POS
    m.body_quat[mount_id] = (1.0, 0.0, 0.0, 0.0)
    m.body_pos[body_id] = (0.0, 0.0, 0.0)
    m.body_quat[body_id] = np.fromstring(RACKET_TRAIN_MOUNT_QUAT, sep=" ")
    m.body_inertia[body_id] = np.fromstring(RACKET_TRAIN_DIAGINERTIA, sep=" ")
    m.body_mass[body_id] = RACKET_TRAIN_MASS
    m.body_ipos[body_id] = RACKET_TRAIN_INERTIAL_POS
    if verbose:
        print("[tnp] racket mount realigned to tasknpoint training pose "
              "(pos=0.08 0 0, shaft along wrist +z, mass 0.1 kg)")
    return True


def _roll_quat_x(deg):
    r = np.radians(deg) / 2.0
    return np.array([np.cos(r), np.sin(r), 0.0, 0.0])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def patch_racket_roll(env, roll_deg):
    """Roll THIS project's racket about its shaft (local x) so the string face
    points toward the incoming ball. Pure mount orientation: the racket head
    path is unchanged, and the policy (which only sees joint states) is blind
    to it."""
    import mujoco

    m = env.model
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "racket")
    assert body_id >= 0
    base = np.array([0.7071068, 0.0, -0.7071068, 0.0])
    m.body_quat[body_id] = _quat_mul(_roll_quat_x(roll_deg), base)


def patch_feet_to_tasknpoint(env, verbose=True):
    """Runtime alignment of the sole contact points with the TaskNPoint
    training foot (a short row of small r=0.01 contacts at x 0.039-0.075,
    z -0.025, friction 1.0). Same spirit as the actuator patch: no asset file
    is modified, only the loaded model."""
    import mujoco

    m = env.model
    layout = [(0.039, -0.018), (0.051, -0.006), (0.063, 0.006), (0.075, 0.018)]
    n = 0
    last_body = None
    per_foot = []
    for i in range(m.ngeom):
        b = m.geom_bodyid[i]
        nm = mujoco.mj_name2id  # noqa
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "ankle_roll_link" in name and m.geom_contype[i] != 0:
            per_foot.append(i)
    for start in range(0, len(per_foot) - len(per_foot) % 4, 4):
        idxs = per_foot[start:start + 4]
        for k, gi in enumerate(idxs):
            m.geom_size[gi][0] = 0.01
            m.geom_pos[gi] = (layout[k][0], layout[k][1], -0.025)
            m.geom_friction[gi][0] = 1.0
            n += 1
    if verbose:
        print("[tnp] soles realigned to tasknpoint training contact row (%d geoms)" % n)
    return True
