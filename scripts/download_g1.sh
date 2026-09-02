#!/usr/bin/env bash
# Downloads the official Unitree G1 model from MuJoCo Menagerie into assets/g1.
# Uses a sparse partial clone cached in ~/tmp to keep bandwidth small.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="$HOME/tmp/tennis_downloads/menagerie"
DEST="$PROJECT_DIR/assets/g1"

if [ -f "$DEST/g1.xml" ]; then
  echo "[g1] already present at $DEST"
  exit 0
fi

if [ ! -d "$CACHE/.git" ]; then
  echo "[g1] sparse-cloning mujoco_menagerie"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie "$CACHE"
fi

git -C "$CACHE" sparse-checkout set unitree_g1
mkdir -p "$DEST"
cp -r "$CACHE/unitree_g1/." "$DEST/"
echo "[g1] installed $(ls "$DEST")"
