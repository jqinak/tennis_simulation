import numpy as np

from tennis_sim import constants as C


def magnus_cl(spin_ratio: float) -> float:
    if spin_ratio < 1e-6:
        return 0.0
    cl = 1.0 / (2.0 + 1.0 / spin_ratio)
    return min(cl, C.MAGNUS_CL_MAX)


def ball_aero_force(vel, omega, rho=C.AIR_DENSITY, cd=C.DRAG_CD, radius=C.BALL_RADIUS):
    vel = np.asarray(vel, dtype=float)
    omega = np.asarray(omega, dtype=float)
    speed = np.linalg.norm(vel)
    area = np.pi * radius * radius
    if speed < 1e-9:
        return np.zeros(3)
    f_drag = -0.5 * rho * cd * area * speed * vel
    spin_ratio = radius * np.linalg.norm(omega) / speed
    cl = magnus_cl(spin_ratio)
    if cl <= 0.0:
        return f_drag
    magnus_dir = np.cross(omega, vel)
    m = np.linalg.norm(magnus_dir)
    if m < 1e-9:
        return f_drag
    f_magnus = 0.5 * rho * area * cl * speed * speed * (magnus_dir / m)
    return f_drag + f_magnus


def spin_decay_torque(omega, inertia=C.BALL_INERTIA, decay=C.SPIN_DECAY_RATE):
    return -decay * inertia * np.asarray(omega, dtype=float)


def accel(vel, omega, mass=C.BALL_MASS, gravity=9.81):
    f = ball_aero_force(vel, omega)
    a = f / mass
    a[2] -= gravity
    return a


def rk4_trajectory(pos0, vel0, omega, duration, dt=0.002, spin_decay=True):
    pos = np.asarray(pos0, dtype=float).copy()
    vel = np.asarray(vel0, dtype=float).copy()
    omega = np.asarray(omega, dtype=float).copy()
    traj = [(pos.copy(), vel.copy())]
    n = int(np.ceil(duration / dt))
    for _ in range(n):
        k1v = accel(vel, omega)
        k1p = vel
        k2v = accel(vel + 0.5 * dt * k1v, omega)
        k2p = vel + 0.5 * dt * k1v
        k3v = accel(vel + 0.5 * dt * k2v, omega)
        k3p = vel + 0.5 * dt * k2v
        k4v = accel(vel + dt * k3v, omega)
        k4p = vel + dt * k3v
        pos = pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        vel = vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if spin_decay:
            omega = omega + dt * (spin_decay_torque(omega) / C.BALL_INERTIA)
        traj.append((pos.copy(), vel.copy()))
        if pos[2] < C.BALL_RADIUS:
            break
    return traj, pos, vel


def simulate_until_ground(pos0, vel0, omega, max_t=3.0, dt=0.002, ground_z=C.BALL_RADIUS):
    traj, pos, vel = rk4_trajectory(pos0, vel0, omega, max_t, dt)
    return traj, pos, vel
