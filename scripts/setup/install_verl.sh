#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
VERL_REPO="${VERL_REPO:-https://github.com/verl-project/verl.git}"
VERL_COMMIT="${VERL_COMMIT:-7aed6b230776f963fa09509c10d9c3a767d1102c}"
PATCH="$PWD/third_party/verl-vgopd.patch"
if [ ! -d third_party/verl/.git ]; then
  git clone "$VERL_REPO" third_party/verl
fi
git -C third_party/verl fetch origin "$VERL_COMMIT" || git -C third_party/verl fetch origin
git -C third_party/verl checkout --detach "$VERL_COMMIT"
git -C third_party/verl apply --check "$PATCH"
git -C third_party/verl apply "$PATCH"
echo "verl $VERL_COMMIT checked out and patched"
uv sync
echo "environment ready: .venv"
