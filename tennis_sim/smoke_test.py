import sys

import numpy as np

from tennis_sim import skills
from tennis_sim.env import TennisEnv
from tennis_sim.scripted_forehand import ScriptedForehand

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def phase_standing_feeds():
    print("== smoke phase 1: standing feeds ==")
    env = TennisEnv()
    env.max_time = 16.0
    env.reset(machine_pos=(-12.7, -1.2, 0.0), machine_yaw=0.0)
    rng = np.random.default_rng(3)
    feeds = [
        ((7.5, 1.8), "topspin", None), ((6.0, -1.5), "flat", None),
        ((8.5, 0.5), "slice", None), ((5.0, 2.0), "topspin", None),
        ((9.0, -2.0), "flat", None), ((7.0, 0.0), "topspin", None),
    ]
    state = {"i": 0, "next_t": 1.0, "plans": []}

    def schedule(t, env):
        if t >= state["next_t"] and state["i"] < len(feeds):
            tx, ty = feeds[state["i"]][0]
            mode = feeds[state["i"]][1]
            plan = env.serve_ball(target_xy=(tx + rng.uniform(-0.3, 0.3),
                                             ty + rng.uniform(-0.3, 0.3)),
                                  mode=mode)
            state["plans"].append(plan)
            state["i"] += 1
            state["next_t"] = t + 2.2

    obs0 = env.reset()
    hold = obs0["qpos"].copy()
    out = env.run_episode(15.0, policy=lambda obs: hold, serve_schedule=schedule)
    fell = any(term for (_, _, term, _) in out)
    report("G1 stays standing through feeds", not fell)
    report("all 6 feeds fired", state["i"] == 6, "fired=%d" % state["i"])
    bounces = [e for e in env.events if e["type"] == "bounce"]
    report("bounces detected on court", len(bounces) >= 6, "bounces=%d" % len(bounces))
    e_vals = [b["restitution"] for b in bounces]
    report("restitution in ITF band during play",
           len(e_vals) > 0 and all(0.60 <= e <= 0.82 for e in e_vals),
           "e in [%.2f, %.2f]" % (min(e_vals), max(e_vals)) if e_vals else "none")
    net_hits = [e for e in env.events if e["type"] == "net_hit"]
    report("net collisions physical (no tunneling)", len(net_hits) <= 2,
           "net_hits=%d" % len(net_hits))
    in_court = [b for b in bounces if b["zone"]["in"]]
    report("feeds land inside court lines", len(in_court) >= 5, "in=%d/%d" % (len(in_court),
                                                                             len(bounces)))
    return env


def phase_scripted_forehand():
    print("== smoke phase 2: scripted forehand ==")
    env = TennisEnv(robot_pos=(6.5, 0.0, 0.79))
    env.max_time = 6.0
    env.reset(machine_pos=(-12.7, 0.4, 0.0), machine_yaw=0.0)
    serve_t = 0.8
    policy = ScriptedForehand(env, serve_t=serve_t, feed_speed=26.0)
    print("  auto-calibrated: contact t=%.2f swing speed=%.1f m/s"
          % (policy.t_contact, policy.swing_speed))
    fired = {"done": False}

    def schedule(t, env):
        if t >= serve_t and not fired["done"]:
            env.serve_ball(plan=policy.plan)
            fired["done"] = True

    out = env.run_episode(5.5, policy=policy, serve_schedule=schedule)
    racket_hits = [e for e in env.events if e["type"] == "racket_hit"]
    report("racket contacts ball", len(racket_hits) >= 1,
           "hits=%d" % len(racket_hits) + (" out_v=%.1fm/s" % racket_hits[0]["ball_speed_out"]
                                           if racket_hits else ""))
    fell = any(term for (_, _, term, _) in out)
    report("G1 stays standing through the swing", not fell)
    bounces = [e for e in env.events if e["type"] == "bounce"]
    far_returns = [b for b in bounces if b["zone"]["side"] == "far" and b["zone"]["in"]]
    report("ball returns over net into far court", len(far_returns) >= 1,
           "far_in=%d" % len(far_returns) +
           ("" if not bounces else " bounces=%s" % [(round(b["pos"][0], 1),
                                                     round(b["pos"][1], 1),
                                                     b["zone"]["zone"]) for b in bounces[:4]]))
    metrics = skills.evaluate_skill(env.events, 5.5)
    report("skill evaluator success", metrics["success"], str(metrics["returns"][:1]))
    return env


def main():
    env1 = phase_standing_feeds()
    print()
    env2 = phase_scripted_forehand()
    print()
    print("== smoke test summary: %d passed, %d failed ==" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
