#!/usr/bin/env bash
set -euo pipefail
source "$HOME/tools/oss-cad-suite/environment"
cd /mnt/c/Users/nehal/Documents/Projects/cnn-accel/tb
export SIM=verilator
for eng in conv pool fc add chip; do
  echo "========== ENGINE=$eng =========="
  make ENGINE="$eng" clean
  make ENGINE="$eng"
done
echo "========== ALL PASSED =========="
