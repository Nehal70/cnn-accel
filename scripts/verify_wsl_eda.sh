#!/usr/bin/env bash
set -euo pipefail
source "$HOME/tools/oss-cad-suite/environment"
echo "suite: $HOME/tools/oss-cad-suite"
command -v verilator
verilator --version
yosys -V | head -n 1
python3 -c "import cocotb, numpy; print('cocotb', cocotb.__version__); print('numpy', numpy.__version__)"
grep -n "oss-cad-suite" "$HOME/.bashrc" | head
