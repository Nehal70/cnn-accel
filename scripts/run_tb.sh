#!/usr/bin/env bash
set -euo pipefail
source "$HOME/tools/oss-cad-suite/environment"
cd /mnt/c/Users/nehal/Documents/Projects/cnn-accel/tb
export SIM=verilator
make "$@"
