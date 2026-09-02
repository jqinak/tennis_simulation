#!/usr/bin/env bash
# One-time environment setup for tennis_simulation.
# Creates .venv inside the project, installs deps, downloads Unitree G1 assets,
# verifies EGL offscreen rendering. All temp/cache writes go to ~/tmp.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="$HOME/tmp"
export PIP_CACHE_DIR="$HOME/tmp/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
UV="$HOME/tmp/uv-bin/uv"
VENV="$PROJECT_DIR/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv at $VENV"
  "$UV" venv "$VENV" --python python3
fi

echo "[setup] installing requirements"
VIRTUAL_ENV="$VENV" "$UV" pip install -r "$PROJECT_DIR/requirements.txt"

echo "[setup] downloading Unitree G1 assets"
bash "$PROJECT_DIR/scripts/download_g1.sh"

echo "[setup] generating court"
VIRTUAL_ENV="$VENV" "$VENV/bin/python" -m tennis_sim.build_court

echo "[setup] verifying EGL offscreen rendering"
MUJOCO_GL=egl "$VENV/bin/python" - <<'PY'
import os
import numpy as np
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
assert img.shape == (64, 64, 3)
r.close()
print("[setup] EGL render OK")
PY

echo "[setup] done. activate with: source $VENV/bin/activate"
