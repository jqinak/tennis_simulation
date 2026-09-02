import xml.etree.ElementTree as ET

from tennis_sim import constants as C

COURT_LENGTH = 23.77
COURT_WIDTH_DOUBLES = 10.97
COURT_WIDTH_SINGLES = 8.23
SERVICE_LINE_X = 6.40
NET_X = 0.0
BASELINE_X = COURT_LENGTH / 2.0
SINGLES_Y = COURT_WIDTH_SINGLES / 2.0
DOUBLES_Y = COURT_WIDTH_DOUBLES / 2.0
LINE_WIDTH = 0.05
BASELINE_WIDTH = 0.10

GROUND_HALF_X = 22.0
GROUND_HALF_Y = 12.0
GROUND_THICKNESS = 0.3

COLOR_BLUE = "0.118 0.310 0.620 1"
COLOR_GREEN = "0.200 0.430 0.235 1"
COLOR_GREEN_DARK = "0.160 0.360 0.190 1"
COLOR_WHITE = "0.92 0.92 0.92 1"
COLOR_NET = "0.16 0.16 0.18 0.38"
COLOR_POST = "0.10 0.10 0.12 1"


def _sub(parent, tag, **attrs):
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        el.set(k.replace("__", ":"), v)
    return el


def add_assets(asset_el):
    _sub(asset_el, "texture", name="skybox", type="skybox", builtin="gradient",
         rgb1="0.35 0.55 0.85", rgb2="0.82 0.88 0.95", width="800", height="600")
    _sub(asset_el, "material", name="court_blue", rgba=COLOR_BLUE, specular="0.25", shininess="0.35",
         reflectance="0.05")
    _sub(asset_el, "material", name="court_green", rgba=COLOR_GREEN, specular="0.2", shininess="0.3",
         reflectance="0.04")
    _sub(asset_el, "material", name="court_green_dark", rgba=COLOR_GREEN_DARK, specular="0.15",
         shininess="0.25")
    _sub(asset_el, "material", name="line_white", rgba=COLOR_WHITE, specular="0.3", shininess="0.4")
    _sub(asset_el, "material", name="net_dark", rgba=COLOR_NET, specular="0.05", shininess="0.0")
    _sub(asset_el, "material", name="post_dark", rgba=COLOR_POST, specular="0.4", shininess="0.5")


def _box(parent, name, pos, size, material, *, collision=False, friction=None, rgba=None):
    attrs = dict(name=name, pos="%.5f %.5f %.5f" % tuple(pos),
                 size="%.5f %.5f %.5f" % tuple(size), type="box")
    if material:
        attrs["material"] = material
    if rgba:
        attrs["rgba"] = rgba
    if collision:
        attrs["condim"] = "3"
        attrs["friction"] = "%s 0.005 0.0001" % friction
        attrs["solref"] = C.CONTACT_SOLREF
        attrs["solimp"] = C.CONTACT_SOLIMP
    else:
        attrs["contype"] = "0"
        attrs["conaffinity"] = "0"
        attrs["group"] = "1"
    return _sub(parent, "geom", **attrs)


def add_ground(worldbody):
    body = _sub(worldbody, "body", name="court_ground")
    _box(body, "ground", [0, 0, -GROUND_THICKNESS / 2.0],
         [GROUND_HALF_X, GROUND_HALF_Y, GROUND_THICKNESS / 2.0],
         "court_green", collision=True, friction=C.GROUND_FRICTION)
    _box(body, "apron_inner", [0, 0, 0.002],
         [BASELINE_X + 3.0, DOUBLES_Y + 2.6, 0.002], "court_green_dark")
    _box(body, "court_surface", [0, 0, 0.004],
         [BASELINE_X, DOUBLES_Y, 0.003], "court_blue")
    return body


def _line(worldbody, name, cx, cy, lx, ly, z=0.0085):
    _box(worldbody, name, [cx, cy, z], [lx / 2.0, ly / 2.0, 0.0005], "line_white")


def add_lines(worldbody):
    lw = LINE_WIDTH
    for sx in (-1, 1):
        bx = sx * BASELINE_X
        _line(worldbody, "baseline_%d" % (sx > 0), bx, 0, BASELINE_WIDTH, COURT_WIDTH_DOUBLES)
        _line(worldbody, "service_line_%d" % (sx > 0), sx * SERVICE_LINE_X, 0, lw,
              COURT_WIDTH_SINGLES)
        _line(worldbody, "center_mark_%d" % (sx > 0), bx - sx * (BASELINE_WIDTH / 2.0 + 0.15), 0,
              0.3, lw)
    _line(worldbody, "center_service_line", 0, 0, 2 * SERVICE_LINE_X, lw)
    for sy in (-1, 1):
        y = sy * SINGLES_Y
        _line(worldbody, "singles_line_%d" % (sy > 0), 0, y, COURT_LENGTH, lw)
        y = sy * DOUBLES_Y
        _line(worldbody, "doubles_line_%d" % (sy > 0), 0, y, COURT_LENGTH, lw)


def _net_segment(worldbody, name, y0, y1, h0, h1, *, collision):
    y_mid = (y0 + y1) / 2.0
    y_len = abs(y1 - y0)
    h_mid = (h0 + h1) / 2.0
    z_mid = h_mid / 2.0
    x_thickness = C.NET_X_HALF_THICKNESS if collision else 0.005
    _box(worldbody, name, [C.NET_X, y_mid, z_mid],
         [x_thickness, y_len / 2.0, h_mid / 2.0], "net_dark", collision=collision,
         friction=C.GROUND_FRICTION)


def add_net(worldbody):
    n_seg = 14
    prev_y = -C.NET_POST_Y
    prev_h = C.NET_HEIGHT_POST
    for i in range(1, n_seg + 1):
        y = -C.NET_POST_Y + (2 * C.NET_POST_Y) * i / n_seg
        t = i / n_seg
        h = C.NET_HEIGHT_POST - (C.NET_HEIGHT_POST - C.NET_HEIGHT_CENTER) * _sag(t)
        _net_segment(worldbody, "net_collision_%02d" % i, prev_y, y, prev_h, h, collision=True)
        prev_y, prev_h = y, h
    prev_y = -C.NET_POST_Y
    prev_h = C.NET_HEIGHT_POST
    for i in range(1, n_seg + 1):
        y = -C.NET_POST_Y + (2 * C.NET_POST_Y) * i / n_seg
        t = i / n_seg
        h = C.NET_HEIGHT_POST - (C.NET_HEIGHT_POST - C.NET_HEIGHT_CENTER) * _sag(t)
        _net_segment(worldbody, "net_visual_%02d" % i, prev_y, y, prev_h, h, collision=False)
        prev_y, prev_h = y, h
    for side in (-1, 1):
        _sub(worldbody, "geom", name="net_post_%s" % ("neg" if side < 0 else "pos"),
             pos="%.4f %.4f %.4f" % (C.NET_X, side * C.NET_POST_Y, C.NET_HEIGHT_POST / 2.0),
             size="0.035 %.4f" % (C.NET_HEIGHT_POST / 2.0), type="cylinder", material="post_dark",
             contype="0", conaffinity="0", group="1")
        _sub(worldbody, "geom", name="net_band_%s" % ("neg" if side < 0 else "pos"),
             pos="%.4f %.4f %.4f" % (C.NET_X, side * (C.NET_POST_Y / 2.0), C.NET_HEIGHT_POST - 0.025),
             size="0.006 %.4f 0.025" % (C.NET_POST_Y / 2.0), type="box", material="line_white",
             contype="0", conaffinity="0", group="1")
    _sub(worldbody, "geom", name="net_center_strap",
         pos="%.4f 0 %.4f" % (C.NET_X, C.NET_HEIGHT_CENTER / 2.0),
         size="0.006 0.02 %.4f" % (C.NET_HEIGHT_CENTER / 2.0), type="box", material="line_white",
         contype="0", conaffinity="0", group="1")


def _sag(t):
    return 1.0 - (2.0 * t - 1.0) ** 2


def build_court_elements():
    worldbody = ET.Element("body_placeholder")
    worldbody.tag = "placeholder"
    return worldbody


def build_full_court_xml():
    mujoco = ET.Element("mujoco", model="tennis_court")
    ET.SubElement(mujoco, "compiler", angle="radian")
    ET.SubElement(mujoco, "option", timestep=str(C.PHYSICS_DT), gravity="0 0 -9.81",
                  wind="0 0 0", density="0")
    ET.SubElement(mujoco, "visual").append(
        ET.Element("global", offwidth="1920", offheight="1080"))
    ET.SubElement(mujoco, "visual").append(
        ET.Element("quality", shadowsize="4096"))
    asset = ET.SubElement(mujoco, "asset")
    add_assets(asset)
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", name="court_light_key", pos="18 -14 22", dir="-0.55 0.45 -1",
                  directional="true", castshadow="true", diffuse="0.85 0.82 0.78",
                  specular="0.3 0.3 0.3")
    ET.SubElement(worldbody, "light", name="court_light_fill", pos="-16 12 18", dir="0.5 -0.4 -1",
                  directional="true", castshadow="false", diffuse="0.35 0.37 0.42",
                  specular="0.1 0.1 0.1")
    add_ground(worldbody)
    add_lines(worldbody)
    add_net(worldbody)
    ET.indent(mujoco, space="  ")
    return ET.ElementTree(mujoco)


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    tree = build_full_court_xml()
    out_path = os.path.join(out_dir, "court_preview.xml")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    print("wrote", out_path)
