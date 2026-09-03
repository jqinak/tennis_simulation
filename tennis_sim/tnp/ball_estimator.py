"""Ball trajectory prediction, ported from the TaskNPoint deploy pipeline
(deploy/simulation/simulation_node/simulation_dynamic.py::_estimate_ball_trajectory
+ deploy/estimation/filtering/ball_estimator_tennis.py, ROS removed).

Two physics backends:
- "tasknpoint": the original deploy model (vacuum ballistic arcs, restitution 0.8,
  ground z=0, no spin).
- "sim": physics matched to this simulator (tennis_sim.aero drag+Magnus RK4 arcs,
  ITF restitution/tangential bounce model from tennis_sim.bounce, spin decay),
  used so the predicted intercept agrees with the simulated ball.

The interface mirrors deploy: predict closest approach of the trajectory to a
nominal target; if within cutoff, emit (target_pos_w, target_time).
"""
import numpy as np

from tennis_sim import aero
from tennis_sim import bounce as B
from tennis_sim import constants as C
from tennis_sim.tnp import goals as G

TASKNPOINT_GRAVITY = 9.81
TASKNPOINT_RESTITUTION = 0.8
TRAJ_DURATION = 3.0
TRAJ_DT = 0.02
BALL_TARGET_CUTOFF_SQ = G.BALL_TARGET_CUTOFF_SQ


def predict_trajectory_tasknpoint(pos, vel, duration=TRAJ_DURATION, dt=TRAJ_DT,
                                  gravity=TASKNPOINT_GRAVITY,
                                  restitution=TASKNPOINT_RESTITUTION):
    """Original deploy model: parabolic arcs + naive bounces on z=0 plane."""
    x, y, z = (float(v) for v in pos)
    vx, vy, vz = (float(v) for v in vel)
    t_elapsed = 0.0
    max_pts = int(np.ceil(duration / dt)) + 1
    buf_t = np.empty(max_pts)
    buf_p = np.empty((max_pts, 3))
    n = 0
    while t_elapsed < duration:
        remaining = duration - t_elapsed
        discriminant = vz ** 2 + 2.0 * gravity * max(z, 0.0)
        t_bounce = (vz + np.sqrt(discriminant)) / gravity
        if t_bounce < 1e-6:
            break
        arc = min(t_bounce, remaining)
        ts = np.arange(0.0, arc, dt)
        if len(ts) == 0 or ts[-1] < arc - 1e-9:
            ts = np.append(ts, arc)
        k = min(len(ts), max_pts - n)
        buf_t[n:n + k] = t_elapsed + ts[:k]
        buf_p[n:n + k, 0] = x + vx * ts[:k]
        buf_p[n:n + k, 1] = y + vy * ts[:k]
        buf_p[n:n + k, 2] = z + vz * ts[:k] - 0.5 * gravity * ts[:k] ** 2
        n += k
        if n >= max_pts:
            break
        t_elapsed += t_bounce
        vz = -restitution * (vz - gravity * t_bounce)
        x += vx * t_bounce
        y += vy * t_bounce
        z = 0.0
    return buf_t[:n], buf_p[:n]


def predict_trajectory_sim(pos, vel, omega=None, duration=TRAJ_DURATION, dt=0.004,
                           max_bounces=4):
    """Sim-matched prediction: aero RK4 arcs + ITF bounce + spin decay."""
    pos = np.asarray(pos, dtype=float).copy()
    vel = np.asarray(vel, dtype=float).copy()
    omega = np.zeros(3) if omega is None else np.asarray(omega, dtype=float).copy()
    all_t, all_p = [], []
    t_total = 0.0
    for _ in range(max_bounces + 1):
        remaining = duration - t_total
        if remaining <= 0:
            break
        traj, _, _ = aero.rk4_trajectory(pos, vel, omega, remaining, dt=dt)
        if len(traj) < 2:
            break
        arr_p = np.array([p for p, _ in traj])
        arr_v = np.array([v for _, v in traj])
        ground = None
        for i in range(1, len(arr_p)):
            if arr_p[i, 2] <= C.BALL_RADIUS:
                ground = i
                break
        end = len(arr_p) if ground is None else ground + 1
        all_t.append(t_total + np.arange(end) * dt)
        all_p.append(arr_p[:end])
        if ground is None or t_total + ground * dt >= duration:
            break
        p_hit, v_hit = arr_p[ground], arr_v[ground]
        t_total += ground * dt
        w_hit = omega * np.exp(-C.SPIN_DECAY_RATE * t_total)
        vel, omega = B.ground_bounce_velocity(v_hit, w_hit)
        pos = np.array([p_hit[0], p_hit[1], max(p_hit[2], C.BALL_RADIUS)])
        if np.linalg.norm(vel) < 0.3:
            break
    if not all_p:
        return np.zeros(0), np.zeros((0, 3))
    return np.concatenate(all_t), np.concatenate(all_p, axis=0)


def closest_approach(times, positions, target_w, cutoff_sq=BALL_TARGET_CUTOFF_SQ):
    """Deploy semantics: nearest predicted sample to the nominal target."""
    if len(times) == 0:
        return None
    d2 = np.sum((positions - np.asarray(target_w, dtype=float)) ** 2, axis=1)
    i = int(np.argmin(d2))
    if d2[i] >= cutoff_sq:
        return None
    return float(times[i]), positions[i].copy()


def estimate_ball_target(pos, vel, target_w, omega=None, physics="sim"):
    """Returns (target_time, target_pos_w) or (None, None) if no valid intercept."""
    if physics == "sim":
        t, p = predict_trajectory_sim(pos, vel, omega=omega)
    else:
        t, p = predict_trajectory_tasknpoint(pos, vel)
    hit = closest_approach(t, p, target_w)
    if hit is None:
        return None, None
    return hit
