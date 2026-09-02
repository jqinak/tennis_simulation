import os
import xml.etree.ElementTree as ET

import numpy as np

from tennis_sim import constants as C
from tennis_sim import build_court

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
G1_DIR = os.path.join(ASSETS_DIR, "g1")
SCENE_PATH = os.path.join(ASSETS_DIR, "scene.xml")

ROBOT_HOME_POS = (10.9, 0.0, 0.79)
ROBOT_HOME_YAW = np.pi
MACHINE_HOME_POS = (-12.7, -1.2, 0.0)

BALL_FREE_JOINT_SLICE = slice(0, 7)

MACHINE_MATERIALS = {
    "machine_body": "0.22 0.24 0.26 1",
    "machine_accent": "0.85 0.45 0.10 1",
    "machine_wheel": "0.10 0.10 0.11 1",
    "machine_barrel": "0.30 0.32 0.34 1",
}


def _fmt(vals):
    return "%.6f" % tuple(vals) if len(vals) == 1 else " ".join("%.6f" % v for v in vals)


def ensure_ball_texture():
    path = os.path.join(ASSETS_DIR, "ball_seam.png")
    if os.path.exists(path):
        return path
    import imageio.v2 as imageio

    w, h = 1024, 512
    u = np.linspace(0.0, 1.0, w, endpoint=False)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :] = np.array([203, 214, 42], dtype=np.uint8)
    noise = np.random.default_rng(7).normal(0, 4, size=(h, w, 1))
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    v = np.linspace(0.0, 1.0, h)
    seam = 0.5 + 0.30 * np.sin(2 * np.pi * 2.0 * u + 0.4) * np.cos(np.pi * 0.9 * u)
    seam2 = 0.5 - 0.30 * np.sin(2 * np.pi * 2.0 * u + 0.4) * np.cos(np.pi * 0.9 * u)
    yy, xx = np.meshgrid(v, u, indexing="ij")
    d1 = np.abs(yy - seam[None, :])
    d2 = np.abs(yy - seam2[None, :])
    d1_wrapped = np.minimum(d1, 1.0 - d1)
    d2_wrapped = np.minimum(d2, 1.0 - d2)
    width = 0.012
    mask = (d1_wrapped < width) | (d2_wrapped < width)
    base[mask] = np.array([245, 245, 240], dtype=np.uint8)
    imageio.imwrite(path, base)
    return path


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _sub(parent, tag, **attrs):
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        el.set(k, v)
    return el


def build_machine_xml(pos=MACHINE_HOME_POS, yaw=0.0):
    if pos is None:
        pos = MACHINE_HOME_POS
    q = yaw_quat(yaw)
    body = ET.Element(
        "body",
        name="ball_machine",
        mocap="true",
        pos=_fmt(pos),
        quat=_fmt(q),
    )
    _sub(body, "geom", name="mc_cart", type="box", pos="0 0 0.17", size="0.38 0.30 0.17",
         material="machine_body", contype="0", conaffinity="0", group="1")
    for wx in (-0.24, 0.24):
        for wy in (-0.26, 0.26):
            _sub(body, "geom", name="mc_wheel_%d_%d" % (int(wx * 100), int(wy * 100)),
                 type="cylinder", pos="%.3f %.3f 0.08" % (wx, wy), size="0.08 0.03",
                 quat="0.7071 0.7071 0 0", material="machine_wheel", contype="0", conaffinity="0",
                 group="1")
    _sub(body, "geom", name="mc_column", type="cylinder", pos="-0.15 0 0.60", size="0.06 0.27",
         material="machine_body", contype="0", conaffinity="0", group="1")
    _sub(body, "geom", name="mc_head", type="box", pos="0.02 0 1.12", size="0.10 0.11 0.09",
         material="machine_body", contype="0", conaffinity="0", group="1")
    _sub(body, "geom", name="mc_accent", type="box", pos="0.02 0 1.22", size="0.07 0.08 0.012",
         material="machine_accent", contype="0", conaffinity="0", group="1")
    _sub(body, "geom", name="mc_barrel", type="capsule", fromto="0.14 0 1.12 0.66 0 1.12",
         size="0.052", material="machine_barrel", contype="0", conaffinity="0", group="1")
    for dz in (-0.075, 0.075):
        _sub(body, "geom", name="mc_flywheel_%d" % int(dz * 1000), type="cylinder",
             pos="0.64 0 %.4f" % (1.12 + dz), size="0.085 0.012", quat="0.7071 0 0.7071 0",
             material="machine_wheel", contype="0", conaffinity="0", group="1")
    _sub(body, "geom", name="mc_hopper", type="box", pos="-0.32 0 1.28", size="0.16 0.18 0.10",
         material="machine_accent", contype="0", conaffinity="0", group="1")
    _sub(body, "site", name="launch_site", pos="0.70 0 1.12", size="0.012", rgba="1 0 0 0.5")
    return body


def build_ball_xml(pos=(0.0, -8.0, C.BALL_RADIUS)):
    if pos is None:
        pos = (0.0, -8.0, C.BALL_RADIUS)
    body = ET.Element("body", name="ball", pos=_fmt(pos))
    _sub(body, "freejoint", name="ball_joint")
    _sub(body, "inertial", pos="0 0 0", mass=str(C.BALL_MASS),
         diaginertia=" ".join(["%.8e" % C.BALL_INERTIA] * 3))
    _sub(body, "geom", name="ball_geom", type="sphere", size=str(C.BALL_RADIUS),
         material="ball_mat", condim="3", friction="%s 0.005 0.0001" % C.GROUND_FRICTION,
         solref=C.CONTACT_SOLREF, solimp=C.CONTACT_SOLIMP)
    return body


def build_racket_xml():
    body = ET.Element("body", name="racket", pos="0.052 0 0",
                      quat="0.300503 -0.683352 0.267845 -0.609087")
    _sub(body, "inertial", pos="0.331 0 0", mass="0.30",
         diaginertia="0.00082 0.0055 0.0063")
    _sub(body, "geom", name="racket_handle", type="capsule", fromto="0.05 0 0 0.23 0 0",
         size="0.016", material="machine_wheel", contype="1", conaffinity="1", condim="3",
         friction="%s 0.005 0.0001" % C.GROUND_FRICTION, solref=C.CONTACT_SOLREF,
         solimp=C.CONTACT_SOLIMP, group="1")
    _sub(body, "geom", name="racket_head_rim", type="ellipsoid", pos="0.415 0 0",
         size="0.175 0.140 0.014", material="machine_accent", contype="0", conaffinity="0",
         group="1")
    _sub(body, "geom", name="racket_strings", type="box", pos="0.405 0 0",
         size="0.155 0.118 0.012", material="line_white", rgba="0.95 0.95 0.9 0.25",
         contype="1", conaffinity="1", condim="3",
         friction="%s 0.005 0.0001" % C.GROUND_FRICTION, solref=C.CONTACT_SOLREF,
         solimp=C.CONTACT_SOLIMP, group="1")
    _sub(body, "site", name="racket_head_site", pos="0.415 0 0", size="0.012", rgba="0 1 0 0.5")
    return body


def _boost_arm_actuators(root):
    actuator = root.find("actuator")
    if actuator is None:
        return
    worldbody = root.find("worldbody")
    joint_els = {}
    for jel in worldbody.iter("joint"):
        nm = jel.get("name")
        if nm:
            joint_els[nm] = jel
    for act in actuator.findall("position"):
        name = act.get("name", "")
        jname = act.get("joint")
        if "wrist" in name:
            act.set("forcerange", "-30 30")
            act.set("forcelimited", "true")
            act.set("kv", "5")
            if jname in joint_els:
                joint_els[jname].set("actuatorfrcrange", "-30 30")
        elif "shoulder" in name or "elbow" in name:
            act.set("forcerange", "-100 100")
            act.set("forcelimited", "true")
            act.set("kv", "15")
            if jname in joint_els:
                joint_els[jname].set("actuatorfrcrange", "-100 100")


def _load_g1_elements():
    tree = ET.parse(os.path.join(G1_DIR, "g1.xml"))
    root = tree.getroot()
    for tag in ("compiler", "option", "visual", "statistic"):
        for el in root.findall(tag):
            root.remove(el)
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        mesh.set("file", "g1/assets/" + mesh.get("file"))
    worldbody = root.find("worldbody")
    for light in worldbody.findall("light"):
        worldbody.remove(light)
    pelvis = worldbody.find("body[@name='pelvis']")
    wrist = pelvis.find(".//body[@name='right_wrist_yaw_link']")
    wrist.append(build_racket_xml())
    key_qpos = root.find("keyframe/key[@name='stand']")
    return root, pelvis, key_qpos


def build_scene_string(robot=True, machine=True, ball=True,
                       robot_pos=None, robot_yaw=ROBOT_HOME_YAW,
                       machine_pos=None, machine_yaw=0.0):
    if robot_pos is None:
        robot_pos = ROBOT_HOME_POS
    if machine_pos is None:
        machine_pos = MACHINE_HOME_POS
    ensure_ball_texture()
    mujoco = ET.Element("mujoco", model="tennis_scene")
    _sub(mujoco, "compiler", angle="radian")
    _sub(mujoco, "option", timestep=str(C.PHYSICS_DT), gravity="0 0 -9.81", wind="0 0 0",
         density="0", integrator="implicitfast")
    visual = _sub(mujoco, "visual")
    _sub(visual, "global", offwidth="1920", offheight="1080")
    _sub(visual, "quality", shadowsize="4096")
    _sub(visual, "map", fogstart="18", fogend="45")
    _sub(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.35 0.35 0.35",
         specular="0.1 0.1 0.1")

    g1_root, pelvis, key_qpos = _load_g1_elements()
    default_src = g1_root.find("default")
    mujoco.append(default_src)
    asset = _sub(mujoco, "asset")
    build_court.add_assets(asset)
    for name, rgba in MACHINE_MATERIALS.items():
        _sub(asset, "material", name=name, rgba=rgba, specular="0.4", shininess="0.5")
    g1_asset = g1_root.find("asset")
    for el in list(g1_asset):
        asset.append(el)
    _sub(asset, "texture", name="tex_ball", type="2d", file="ball_seam.png")
    _sub(asset, "material", name="ball_mat", texture="tex_ball", specular="0.35",
         shininess="0.5")

    worldbody = _sub(mujoco, "worldbody")
    _sub(worldbody, "light", name="court_light_key", pos="18 -14 22", dir="-0.55 0.45 -1",
         directional="true", castshadow="true", diffuse="0.85 0.82 0.78", specular="0.3 0.3 0.3")
    _sub(worldbody, "light", name="court_light_fill", pos="-16 12 18", dir="0.5 -0.4 -1",
         directional="true", castshadow="false", diffuse="0.35 0.37 0.42", specular="0.1 0.1 0.1")
    build_court.add_ground(worldbody)
    build_court.add_lines(worldbody)
    build_court.add_net(worldbody)
    if machine:
        worldbody.append(build_machine_xml(machine_pos, machine_yaw))
    if ball:
        worldbody.append(build_ball_xml())
    if robot:
        pelvis.set("pos", _fmt(robot_pos))
        pelvis.set("quat", _fmt(yaw_quat(robot_yaw)))
        worldbody.append(pelvis)
        _boost_arm_actuators(g1_root)
        for tag in ("actuator", "sensor"):
            mujoco.append(g1_root.find(tag))
        kf_root = g1_root.find("keyframe")
        qpos_vals = key_qpos.get("qpos").split()
        ball_pos = (0.0, -8.0, C.BALL_RADIUS)
        ball_part = ["%.4f" % ball_pos[0], "%.4f" % ball_pos[1], "%.4f" % ball_pos[2],
                     "1", "0", "0", "0"]
        robot_part = ["%.4f" % robot_pos[0], "%.4f" % robot_pos[1], "%.4f" % robot_pos[2]]
        robot_part += ["%.6f" % v for v in yaw_quat(robot_yaw)]
        key_qpos.set("qpos", " ".join(ball_part + robot_part + qpos_vals[7:]))
        mujoco.append(kf_root)
    ET.indent(mujoco, space="  ")
    return ET.ElementTree(mujoco)


def write_scene(path=SCENE_PATH, **kwargs):
    tree = build_scene_string(**kwargs)
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return path


def load_scene(path=SCENE_PATH):
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def get_id(model, objtype, name):
    import mujoco

    return mujoco.mj_name2id(model, objtype, name)


if __name__ == "__main__":
    p = write_scene()
    model, data = load_scene(p)
    print("scene written:", p)
    print("nq=%d nv=%d nu=%d nbody=%d ngeom=%d" % (model.nq, model.nv, model.nu, model.nbody,
                                                   model.ngeom))
