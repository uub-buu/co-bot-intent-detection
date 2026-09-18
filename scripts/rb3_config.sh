#!/usr/bin/env bash
#
# rb3_env_check.sh
#
# Run this ON THE RB3, via `source rb3_env_check.sh` (NOT
# `./rb3_env_check.sh` -- it must be sourced, not executed, or the
# exported ADSP_LIBRARY_PATH will vanish the moment the script exits).
#
# Copies the required ARM-side and DSP-side libraries DIRECTLY from the
# QAIRT SDK already unzipped on this board (no /tmp staging, no scp from
# a dev machine needed), installs them into their final locations, sets
# ADSP_LIBRARY_PATH, and verifies everything before you run the binary.
#
# Usage:
#   source rb3_env_check.sh [path_to_unzipped_qairt_sdk]
#
# Defaults to ./qairt/2.50.0.260828 relative to the current directory if
# no path is given.

QAIRT_ROOT="/home/ubuntu/co-bot-intent-detection/libs/qairt/2.50.0.260828"

# Confirmed correct target for this SDK on RB3 Gen 2 (Ubuntu 24.04)
AARCH64_DIR="${QAIRT_ROOT}/lib/aarch64-ubuntu-gcc9.4"

# Confirmed correct Hexagon target for QCS6490 / RB3 Gen 2 (v68)
HEXAGON_DIR="${QAIRT_ROOT}/lib/hexagon-v68/unsigned"

# Confirmed correct paths for this specific board (RB3 Gen 2 / QCS6490,
# Yocto-style /usr/share/qcom layout)
CDSP_DIR="/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp"
ADSP_DIR="/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/adsp"

echo "== Checking QAIRT_ROOT =="
if [ ! -d "${QAIRT_ROOT}" ]; then
  echo "  MISSING: ${QAIRT_ROOT} does not exist."
  echo "  Usage: source rb3_env_check.sh /path/to/unzipped/qairt/2.50.0.260828"
  return 1 2>/dev/null || exit 1
fi
echo "  OK   QAIRT_ROOT=${QAIRT_ROOT}"

echo ""
echo "== Installing ARM-side libraries (DSP + GPU backends) to /usr/lib =="
ARM_LIBS=("libQnnHtp.so" "libQnnHtpV68Stub.so" "libQnnGpu.so")
for lib in "${ARM_LIBS[@]}"; do
  if [ -f "${AARCH64_DIR}/${lib}" ]; then
    sudo cp "${AARCH64_DIR}/${lib}" /usr/lib/
    echo "  copied ${lib}"
  else
    echo "  MISSING in SDK: ${AARCH64_DIR}/${lib}"
  fi
done
sudo ldconfig

echo ""
echo "== Installing DSP-side skel library =="
if [ -f "${HEXAGON_DIR}/libQnnHtpV68Skel.so" ]; then
  if [ -d "${CDSP_DIR}" ]; then
    sudo cp "${HEXAGON_DIR}/libQnnHtpV68Skel.so" "${CDSP_DIR}/"
    echo "  OK"
  else
    echo "  MISSING expected directory: ${CDSP_DIR}"
    echo "  This board's DSP path layout may differ -- run:"
    echo "    find / -type d -iname cdsp 2>/dev/null"
    echo "  and update CDSP_DIR in this script if it's somewhere else."
  fi
else
  echo "  MISSING in SDK: ${HEXAGON_DIR}/libQnnHtpV68Skel.so"
fi

echo ""
echo "== Setting ADSP_LIBRARY_PATH (this session only) =="
export ADSP_LIBRARY_PATH="${CDSP_DIR};${ADSP_DIR}"
echo "  ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH}"
echo "  NOTE: this does not persist across new shells/sessions -- re-source"
echo "  this script every time you reconnect, or add the export line to"
echo "  your ~/.bashrc if you want it permanent."

echo ""
echo "== Verifying everything is actually in place =="

all_ok=true

check_file() {
  if [ -f "$1" ]; then
    echo "  OK   $1"
  else
    echo "  MISSING   $1"
    all_ok=false
  fi
}

check_lib() {
  if ldconfig -p | grep -q "$1"; then
    echo "  OK   $1 registered with ldconfig"
  else
    echo "  MISSING   $1 not found by ldconfig -- did sudo ldconfig run?"
    all_ok=false
  fi
}

check_lib "libQnnHtp.so"
check_lib "libQnnHtpV68Stub.so"
check_lib "libQnnGpu.so"
check_file "${CDSP_DIR}/libQnnHtpV68Skel.so"

echo ""
echo "== Checking DSP firmware booted (informational, not fixable here) =="
if dmesg | grep -qi cdsp; then
  echo "  OK   cdsp firmware message found in dmesg"
else
  echo "  WARNING: no cdsp mention in dmesg -- board may need a reboot,"
  echo "  or the DSP subsystem may not be enabled on this image."
fi

echo ""
echo "== Checking model files =="
check_file "$HOME/model/dlc/pose_landmark_lite.dlc"
check_file "$HOME/model/dlc/lite_stgcn_hmdb51.dlc"

echo ""
if [ "${all_ok}" = true ]; then
  echo "All checks passed. You should be able to run the pipeline binary now."
else
  echo "One or more checks failed -- see MISSING lines above before running."
fi

echo "== Setting ADSP_LIBRARY_PATH (this session only) =="
export ADSP_LIBRARY_PATH="${CDSP_DIR};${ADSP_DIR}"
echo "  ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH}"
echo "  NOTE: this does not persist across new shells/sessions -- re-source"
echo "  this script every time you reconnect, or add the export line to"
echo "  your ~/.bashrc if you want it permanent."

echo ""
echo "== Verifying everything is actually in place =="

all_ok=true

check_file() {
  if [ -f "$1" ]; then
    echo "  OK   $1"
  else
    echo "  MISSING   $1"
    all_ok=false
  fi
}

check_lib() {
  if ldconfig -p | grep -q "$1"; then
    echo "  OK   $1 registered with ldconfig"
  else
    echo "  MISSING   $1 not found by ldconfig -- did sudo ldconfig run?"
    all_ok=false
  fi
}

check_lib "libQnnHtp.so"
check_lib "libQnnHtpV68Stub.so"
check_file "${CDSP_DIR}/libQnnHtpV68Skel.so"

echo ""
echo "== Checking DSP firmware booted (informational, not fixable here) =="
if dmesg | grep -qi cdsp; then
  echo "  OK   cdsp firmware message found in dmesg"
else
  echo "  WARNING: no cdsp mention in dmesg -- board may need a reboot,"
  echo "  or the DSP subsystem may not be enabled on this image."
fi

echo ""
echo "== Checking model files =="
check_file "$HOME/model/dlc/pose_landmark_lite.dlc"
check_file "$HOME/model/dlc/lite_stgcn_hmdb51.dlc"

echo ""
if [ "${all_ok}" = true ]; then
  echo "All checks passed. You should be able to run the pipeline binary now."
else
  echo "One or more checks failed -- see MISSING lines above before running."
fi