from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SkillConfig:
    name: str
    description: str
    machine_pos: Tuple[float, float, float]
    machine_yaw: float
    serve_targets: List[Tuple[float, float]]
    target_jitter: float
    modes: Tuple[str, ...]
    speed: Optional[Tuple[float, float]]
    spin_scale: float
    interval: float
    first_serve_t: float = 0.8
    serve_duration: float = 3.2


SKILLS = {
    "forehand": SkillConfig(
        name="forehand",
        description="G1 forehand drives: machine feeds to the robot's forehand half (+y side)",
        machine_pos=(-12.7, -3.0, 0.0),
        machine_yaw=0.0,
        serve_targets=[(6.0, 1.5), (7.5, 2.2), (8.5, 0.8), (5.5, 2.8)],
        target_jitter=0.4,
        modes=("topspin", "flat"),
        speed=None,
        spin_scale=1.0,
        interval=2.8,
    ),
    "backhand": SkillConfig(
        name="backhand",
        description="G1 backhand drives: machine feeds to the robot's backhand half (-y side)",
        machine_pos=(-12.7, 3.0, 0.0),
        machine_yaw=0.0,
        serve_targets=[(6.0, -1.5), (7.5, -2.2), (8.5, -0.8), (5.5, -2.8)],
        target_jitter=0.4,
        modes=("topspin", "flat"),
        speed=None,
        spin_scale=1.0,
        interval=2.8,
    ),
    "volley": SkillConfig(
        name="volley",
        description="G1 volleys: machine near the net feeds soft, short balls",
        machine_pos=(-5.5, -2.0, 0.0),
        machine_yaw=0.0,
        serve_targets=[(8.0, 0.5), (9.0, -0.8), (8.5, 1.2), (9.5, 0.0)],
        target_jitter=0.3,
        modes=("slice",),
        speed=(14.0, 20.0),
        spin_scale=0.8,
        interval=2.2,
    ),
    "smash": SkillConfig(
        name="smash",
        description="G1 overhead smashes: machine lobs high, slow balls to the mid court",
        machine_pos=(-9.0, 0.0, 0.0),
        machine_yaw=0.0,
        serve_targets=[(5.5, 0.5), (6.5, -0.8), (6.0, 0.0)],
        target_jitter=0.3,
        modes=("lob",),
        speed=None,
        spin_scale=1.0,
        interval=3.2,
    ),
    "serve": SkillConfig(
        name="serve",
        description="G1 serves: ball rests near the robot's left hand; the policy tosses and hits "
                    "into the diagonal service box on the far side",
        machine_pos=(-12.7, -1.2, 0.0),
        machine_yaw=0.0,
        serve_targets=[],
        target_jitter=0.0,
        modes=(),
        speed=None,
        spin_scale=1.0,
        interval=float("inf"),
        first_serve_t=float("inf"),
    ),
}


def get_skill(name):
    return SKILLS[name]


def make_serve_schedule(config, rng=None):
    rng = rng or np.random.default_rng()
    state = {"next_t": config.first_serve_t, "serve_i": 0, "mode_i": 0}

    def schedule(t, env):
        if t < state["next_t"]:
            return None
        if not config.serve_targets:
            return None
        tx, ty = config.serve_targets[state["serve_i"] % len(config.serve_targets)]
        state["serve_i"] += 1
        tx += rng.uniform(-config.target_jitter, config.target_jitter)
        ty += rng.uniform(-config.target_jitter, config.target_jitter)
        mode = config.modes[state["mode_i"] % len(config.modes)]
        state["mode_i"] += 1
        speed = config.speed
        if speed is not None:
            speed = rng.uniform(*speed)
        plan = env.serve_ball(target_xy=(tx, ty), mode=mode, speed=speed)
        state["next_t"] = t + config.interval
        return plan

    return schedule


def evaluate_skill(events, duration):
    racket_hits = [e for e in events if e["type"] == "racket_hit"]
    net_hits = [e for e in events if e["type"] == "net_hit"]
    bounces = [e for e in events if e["type"] == "bounce"]
    returns = []
    for hit in racket_hits:
        after = [b for b in bounces if b["t"] > hit["t"] + 0.01]
        first = after[0] if after else None
        if first is not None and first["zone"]["in"] and first["zone"]["side"] == "far":
            returns.append({"t": hit["t"], "landing": first["pos"].tolist(),
                            "zone": first["zone"]["zone"], "speed_out": hit["ball_speed_out"]})
        elif first is None:
            net = [n for n in net_hits if n["t"] > hit["t"]]
            returns.append({"t": hit["t"], "landing": None, "zone": "net" if net else "unknown",
                            "speed_out": hit["ball_speed_out"]})
        else:
            returns.append({"t": hit["t"], "landing": first["pos"].tolist(),
                            "zone": first["zone"]["zone"], "speed_out": hit["ball_speed_out"]})
    success = any(r.get("zone") in ("singles", "doubles") for r in returns)
    metrics = {
        "success": bool(success),
        "racket_hits": len(racket_hits),
        "net_hits": len(net_hits),
        "returns": returns,
        "duration": duration,
    }
    return metrics
