import math
import os
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

COLOR_WINDSCREEN = "0.058 0.105 0.068 1"
COLOR_FENCE_MESH = "0.42 0.46 0.45 0.40"
COLOR_FENCE_FRAME = "0.20 0.22 0.21 1"
COLOR_CHAIR_GREEN = "0.086 0.30 0.17 1"
COLOR_BENCH_METAL = "0.23 0.24 0.26 1"
COLOR_BENCH_SLAT = "0.88 0.88 0.85 1"
COLOR_POLE = "0.34 0.36 0.39 1"
COLOR_POLE_HEAD = "0.11 0.12 0.13 1"
COLOR_LAMP = "0.97 0.97 0.92 1"
COLOR_CART = "0.80 0.42 0.10 1"
COLOR_BANNER = "0.10 0.20 0.45 1"
COLOR_OUTER_GROUND = "0.115 0.185 0.115 1"

FENCE_HALF_X = 16.2
FENCE_HALF_Y = 9.8
FENCE_HEIGHT = 1.0
BACKSTOP_X = -FENCE_HALF_X


def _sub(parent, tag, **attrs):
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        el.set(k.replace("__", ":"), v)
    return el


def _assets_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def _rel_to_assets(path):
    return os.path.relpath(path, _assets_dir())


def _tex_or_none(builder, *args, **kw):
    try:
        return _rel_to_assets(builder(*args, **kw))
    except Exception:
        return None


def add_assets(asset_el):
    try:
        from tennis_sim import textures as TX

        sky = _tex_or_none(TX.ensure_skybox)
    except Exception:
        sky = None
    if sky:
        _sub(asset_el, "texture", name="skybox", type="skybox", file=sky,
             gridsize="3 4", gridlayout="..U.LFRB.D..")
    else:
        _sub(asset_el, "texture", name="skybox", type="skybox", builtin="gradient",
             rgb1="0.13 0.31 0.65", rgb2="0.85 0.90 0.94", width="800", height="600")
    tex_court = _tex_or_none(_surface, "court_surface.png", 3)
    tex_apron = _tex_or_none(_surface, "court_apron.png", 5)
    tex_ground = _tex_or_none(_surface, "court_ground.png", 9)
    if tex_court:
        _sub(asset_el, "texture", name="tex_court", type="2d", file=tex_court)
        _sub(asset_el, "material", name="court_blue", texture="tex_court", texrepeat="12 6",
             rgba=COLOR_BLUE, specular="0.25", shininess="0.35", reflectance="0.05")
    else:
        _sub(asset_el, "material", name="court_blue", rgba=COLOR_BLUE, specular="0.25",
             shininess="0.35", reflectance="0.05")
    if tex_apron:
        _sub(asset_el, "texture", name="tex_apron", type="2d", file=tex_apron)
        _sub(asset_el, "material", name="court_green_dark", texture="tex_apron", texrepeat="15 8",
             rgba=COLOR_GREEN_DARK, specular="0.15", shininess="0.25")
    else:
        _sub(asset_el, "material", name="court_green_dark", rgba=COLOR_GREEN_DARK,
             specular="0.15", shininess="0.25")
    if tex_ground:
        _sub(asset_el, "texture", name="tex_ground", type="2d", file=tex_ground)
        _sub(asset_el, "material", name="court_green", texture="tex_ground", texrepeat="22 12",
             rgba=COLOR_GREEN, specular="0.2", shininess="0.3", reflectance="0.04")
    else:
        _sub(asset_el, "material", name="court_green", rgba=COLOR_GREEN, specular="0.2",
             shininess="0.3", reflectance="0.04")
    _sub(asset_el, "material", name="line_white", rgba=COLOR_WHITE, specular="0.3", shininess="0.4")
    _sub(asset_el, "material", name="net_dark", rgba=COLOR_NET, specular="0.05", shininess="0.0")
    _sub(asset_el, "material", name="post_dark", rgba=COLOR_POST, specular="0.4", shininess="0.5")
    _sub(asset_el, "material", name="windscreen", rgba=COLOR_WINDSCREEN, specular="0.08",
         shininess="0.1")
    _sub(asset_el, "material", name="fence_mesh", rgba=COLOR_FENCE_MESH, specular="0.2",
         shininess="0.3")
    _sub(asset_el, "material", name="fence_frame", rgba=COLOR_FENCE_FRAME, specular="0.3",
         shininess="0.4")
    _sub(asset_el, "material", name="chair_green", rgba=COLOR_CHAIR_GREEN, specular="0.3",
         shininess="0.4")
    _sub(asset_el, "material", name="bench_metal", rgba=COLOR_BENCH_METAL, specular="0.4",
         shininess="0.5")
    _sub(asset_el, "material", name="bench_slat", rgba=COLOR_BENCH_SLAT, specular="0.2",
         shininess="0.3")
    _sub(asset_el, "material", name="pole_gray", rgba=COLOR_POLE, specular="0.35",
         shininess="0.4")
    _sub(asset_el, "material", name="pole_head", rgba=COLOR_POLE_HEAD, specular="0.4",
         shininess="0.5")
    _sub(asset_el, "material", name="lamp_white", rgba=COLOR_LAMP, specular="0.5",
         shininess="0.6")
    _sub(asset_el, "material", name="cart_accent", rgba=COLOR_CART, specular="0.4",
         shininess="0.5")
    _sub(asset_el, "material", name="outer_ground", rgba=COLOR_OUTER_GROUND, specular="0.1",
         shininess="0.15")
    _sub(asset_el, "material", name="banner_blue", rgba=COLOR_BANNER, specular="0.2",
         shininess="0.3")


def _surface(name, seed):
    from tennis_sim import textures as TX

    return TX.ensure_surface_texture(name, seed)


def _deco_box(parent, name, pos, size, material, quat=None):
    attrs = dict(name=name, type="box", pos="%.4f %.4f %.4f" % tuple(pos),
                 size="%.4f %.4f %.4f" % tuple(size), material=material,
                 contype="0", conaffinity="0", group="1")
    if quat is not None:
        attrs["quat"] = quat
    return _sub(parent, "geom", **attrs)


def _deco_capsule(parent, name, fromto, radius, material):
    return _sub(parent, "geom", name=name, type="capsule",
                fromto="%.4f %.4f %.4f %.4f %.4f %.4f" % tuple(fromto),
                size="%.4f" % radius, material=material, contype="0", conaffinity="0",
                group="1")


def _deco_cylinder(parent, name, pos, radius, half_len, material, quat=None):
    attrs = dict(name=name, type="cylinder", pos="%.4f %.4f %.4f" % tuple(pos),
                 size="%.4f %.4f" % (radius, half_len), material=material,
                 contype="0", conaffinity="0", group="1")
    if quat is not None:
        attrs["quat"] = quat
    return _sub(parent, "geom", **attrs)


def _fence_run(parent, prefix, x0, y0, x1, y1, height=FENCE_HEIGHT):
    """Low visual fence between (x0,y0) and (x1,y1) along one axis."""
    length = max(abs(x1 - x0), abs(y1 - y0))
    along_x = abs(x1 - x0) > abs(y1 - y0)
    mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if along_x:
        _deco_box(parent, prefix + "_mesh", [mid_x, mid_y, height * 0.5],
                  [length / 2.0, 0.012, height * 0.42], "fence_mesh")
        for z in (height, height * 0.1):
            _deco_box(parent, prefix + "_rail_%d" % int(z * 100),
                      [mid_x, mid_y, z], [length / 2.0, 0.035, 0.035], "fence_frame")
        n = max(2, int(round(length / 3.25)))
        for k in range(n + 1):
            t = k / n
            _deco_cylinder(parent, "%s_post_%d" % (prefix, k),
                           [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, (height + 0.06) / 2.0],
                           0.032, (height + 0.06) / 2.0, "fence_frame")
    else:
        _deco_box(parent, prefix + "_mesh", [mid_x, mid_y, height * 0.5],
                  [0.012, length / 2.0, height * 0.42], "fence_mesh")
        for z in (height, height * 0.1):
            _deco_box(parent, prefix + "_rail_%d" % int(z * 100),
                      [mid_x, mid_y, z], [0.035, length / 2.0, 0.035], "fence_frame")
        n = max(2, int(round(length / 3.25)))
        for k in range(n + 1):
            t = k / n
            _deco_cylinder(parent, "%s_post_%d" % (prefix, k),
                           [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, (height + 0.06) / 2.0],
                           0.032, (height + 0.06) / 2.0, "fence_frame")


def _backstop(parent, x=BACKSTOP_X, half_y=FENCE_HALF_Y, height=4.0):
    _deco_box(parent, "backstop_panel", [x, 0.0, height / 2.0],
              [0.03, half_y, height / 2.0 - 0.1], "windscreen")
    _deco_box(parent, "backstop_band", [x, 0.0, height - 0.18],
              [0.036, half_y, 0.16], "line_white")
    _deco_box(parent, "backstop_banner", [x, 0.0, height - 0.52],
              [0.036, half_y, 0.10], "banner_blue")
    _deco_box(parent, "backstop_skirt", [x, 0.0, 0.15],
              [0.036, half_y, 0.30], "fence_frame")
    _deco_box(parent, "backstop_cap", [x - 0.02, 0.0, height + 0.03],
              [0.05, half_y, 0.03], "fence_frame")
    n = max(2, int(round(2.0 * half_y / 3.27)))
    for k in range(n + 1):
        y = -half_y + (2.0 * half_y) * k / n
        _deco_cylinder(parent, "backstop_post_%d" % k, [x - 0.06, y, (height + 0.1) / 2.0],
                       0.05, (height + 0.1) / 2.0, "fence_frame")


def _umpire_chair(parent, x=0.0, y=-7.4):
    b = _sub(parent, "body", name="umpire_chair", pos="%.3f %.3f 0" % (x, y))
    for sy in (-1, 1):
        for sx in (-1, 1):
            _deco_capsule(b, "chair_leg_%d_%d" % (sx, sy),
                          [sx * 0.30, sy * 0.26, 0.02, sx * 0.20, sy * 0.18, 1.52],
                          0.020, "chair_green")
    _deco_box(b, "chair_seat", [0, 0, 1.56], [0.17, 0.15, 0.02], "chair_green")
    _deco_box(b, "chair_back", [0, -0.20, 1.98], [0.16, 0.02, 0.225], "chair_green")
    for sy in (-1, 1):
        _deco_box(b, "chair_arm_%d" % sy, [0, sy * 0.15, 1.88], [0.15, 0.015, 0.015],
                  "chair_green")
    for z in (0.42, 0.88):
        _deco_box(b, "chair_step_%d" % int(z * 100), [0, 0, z], [0.15, 0.12, 0.015],
                  "chair_green")


def _bench(parent, x, y):
    name = "bench_%d_%d" % (int(x * 10), int(y * 10))
    b = _sub(parent, "body", name=name, pos="%.3f %.3f 0" % (x, y))
    _deco_box(b, name + "_seat", [0, 0, 0.45], [0.16, 0.85, 0.02], "bench_slat")
    _deco_box(b, name + "_back", [0.17, 0, 0.80], [0.02, 0.85, 0.14], "bench_slat")
    for sy in (-1, 1):
        _deco_box(b, name + "_leg_%d" % sy, [0.05, sy * 0.72, 0.22], [0.14, 0.025, 0.22],
                  "bench_metal")


def _ball_cart(parent, x=-14.0, y=2.4):
    b = _sub(parent, "body", name="ball_cart", pos="%.3f %.3f 0" % (x, y))
    _deco_box(b, "cart_basket", [0, 0, 0.68], [0.17, 0.23, 0.11], "cart_accent")
    for sx in (-1, 1):
        for sy in (-1, 1):
            _deco_cylinder(b, "cart_leg_%d_%d" % (sx, sy),
                           [sx * 0.13, sy * 0.18, 0.30], 0.022, 0.28, "fence_frame")
            _deco_cylinder(b, "cart_wheel_%d_%d" % (sx, sy),
                           [sx * 0.13, sy * 0.20, 0.06], 0.06, 0.02, "pole_head",
                           quat="0.7071 0 0.7071 0")
    _deco_capsule(b, "cart_handle", [-0.30, 0, 0.86, 0.30, 0, 0.86], 0.015, "fence_frame")


def _light_poles(parent, half_x=17.3, half_y=10.7, height=6.6):
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * half_x, sy * half_y
            name = "light_pole_%d_%d" % (sx, sy)
            b = _sub(parent, "body", name=name, pos="%.3f %.3f 0" % (x, y))
            _deco_cylinder(b, name + "_pole", [0, 0, height / 2.0], 0.09, height / 2.0,
                           "pole_gray")
            yaw = math.atan2(-y, -x)
            qstr = "%.5f 0 0 %.5f" % (math.cos(yaw / 2.0), math.sin(yaw / 2.0))
            dx, dy = -sx / 1.4142, -sy / 1.4142
            _deco_box(b, name + "_arm", [dx * 0.45, dy * 0.45, height - 0.12],
                      [0.35, 0.035, 0.035], "pole_gray", quat=qstr)
            _deco_box(b, name + "_head", [dx * 0.85, dy * 0.85, height - 0.20],
                      [0.28, 0.13, 0.06], "pole_head", quat=qstr)
            _deco_box(b, name + "_lamp", [dx * 0.85, dy * 0.85, height - 0.26],
                      [0.25, 0.11, 0.02], "lamp_white", quat=qstr)


def add_decor(worldbody):
    """Visual-only scene decoration. All geoms are non-colliding (contype=0/conaffinity=0)."""
    _deco_box(worldbody, "outer_ground_visual", [0.0, 0.0, -0.008], [42.0, 30.0, 0.004],
              "outer_ground")
    fx, fy = FENCE_HALF_X, FENCE_HALF_Y
    _fence_run(parent=worldbody, prefix="fence_north", x0=-fx, y0=fy, x1=fx, y1=fy)
    _fence_run(parent=worldbody, prefix="fence_south", x0=-fx, y0=-fy, x1=fx, y1=-fy)
    _fence_run(parent=worldbody, prefix="fence_east", x0=fx, y0=-fy, x1=fx, y1=fy)
    _backstop(worldbody)
    _umpire_chair(worldbody)
    _bench(worldbody, 13.3, 6.9)
    _bench(worldbody, 13.3, -6.9)
    _ball_cart(worldbody)
    _light_poles(worldbody)


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
    add_decor(worldbody)
    ET.indent(mujoco, space="  ")
    return ET.ElementTree(mujoco)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    tree = build_full_court_xml()
    out_path = os.path.join(out_dir, "court_preview.xml")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    print("wrote", out_path)
