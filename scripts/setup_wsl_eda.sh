#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${HOME}/tools"
SUITE_DIR="${TOOLS_DIR}/oss-cad-suite"
TAG="2026-06-23"
TGZ_NAME="oss-cad-suite-linux-x64-20260623.tgz"
URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${TAG}/${TGZ_NAME}"
TGZ_PATH="${TOOLS_DIR}/${TGZ_NAME}"

mkdir -p "${TOOLS_DIR}"

if [[ -x "${SUITE_DIR}/bin/verilator" ]]; then
  echo "OSS CAD Suite already present at ${SUITE_DIR}"
else
  if [[ ! -f "${TGZ_PATH}" ]]; then
    echo "Downloading ${TGZ_NAME} (~700MB)..."
    curl -L --retry 3 --fail -o "${TGZ_PATH}.partial" "${URL}"
    mv "${TGZ_PATH}.partial" "${TGZ_PATH}"
  fi
  echo "Extracting to ${TOOLS_DIR}..."
  tar -C "${TOOLS_DIR}" -xzf "${TGZ_PATH}"
  # tarball extracts as oss-cad-suite/
  if [[ ! -x "${SUITE_DIR}/bin/verilator" ]]; then
    echo "Extract did not produce ${SUITE_DIR}/bin/verilator" >&2
    ls -la "${TOOLS_DIR}"
    exit 1
  fi
fi

ENV_LINE="source ${SUITE_DIR}/environment"
if ! grep -qF "${ENV_LINE}" "${HOME}/.bashrc" 2>/dev/null; then
  {
    echo ""
    echo "# OSS CAD Suite (cnn-accel)"
    echo "${ENV_LINE}"
  } >> "${HOME}/.bashrc"
  echo "Appended suite environment to ~/.bashrc"
fi

# shellcheck disable=SC1091
source "${SUITE_DIR}/environment"

echo "---- versions ----"
verilator --version
yosys -V | head -n 1
iverilog -V | head -n 1
python3 --version
python3 -c "import cocotb; print('cocotb', cocotb.__version__)"

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet numpy

python3 -c "import numpy; print('numpy', numpy.__version__)"

echo "---- done ----"
echo "Open WSL Ubuntu and run: source ~/tools/oss-cad-suite/environment"
echo "Or open a new Ubuntu shell (bashrc loads it)."
