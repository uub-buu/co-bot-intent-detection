#!/usr/bin/env bash
#
# rb3_config.sh
#
# Script checks if libs exists in RB3 and ports them to the necessary locations.
# Note: In Progress
# Usage:`source rb3_config.sh /path/to/unzipped/qairt/2.50.0.260828`

if [ -z "${1:-}" ]; then
  echo "Usage: source rb3_config.sh /path/to/unzipped/qairt/<sdk  version>"
  echo "  (the path to your unzipped QAIRT SDK is required)"
  return 1 2>/dev/null || exit 1
fi

QAIRT_ROOT="$1"

AARCH64_DIR="${QAIRT_ROOT}/lib/aarch64-ubuntu-gcc9.4"
HEXAGON_DIR="${QAIRT_ROOT}/lib/hexagon-v68/unsigned"

all_ok=true

echo "Setting Qairt root directory"
if [ ! -d "${QAIRT_ROOT}" ]; then
  echo "  MISSING: ${QAIRT_ROOT} does not exist."
  echo "  Usage: source cb3_config.sh /path/to/unzipped/qairt/2.50.0.260828"
  return 1 2>/dev/null || exit 1
fi
echo "  OK   QAIRT_ROOT=${QAIRT_ROOT}"

echo "Finding DSP directories"
CDSP_DIR="$(find /usr/share/qcom -type d -iname cdsp 2>/dev/null | head -1)"
ADSP_DIR="$(find /usr/share/qcom -type d -iname adsp 2>/dev/null | head -1)"

if [ -z "${CDSP_DIR}" ]; then
  echo "  MISSING: could not find a 'cdsp' directory under /usr/share/qcom"
  all_ok=false
else
  echo "OK CDSP_DIR=${CDSP_DIR}"
fi

if [ -z "${ADSP_DIR}" ]; then
  echo "  MISSING: could not find an 'adsp' directory under /usr/share/qcom"
  all_ok=false
else
  echo "OK ADSP_DIR=${ADSP_DIR}"
fi

echo "Installing libs for RB3 for QNN aka SNPE"
ARM_LIBS=("libQnnHtp.so" "libQnnHtpV68Stub.so" "libQnnGpu.so")
for lib in "${ARM_LIBS[@]}"; do
  if [ -f "${AARCH64_DIR}/${lib}" ]; then
    sudo cp "${AARCH64_DIR}/${lib}" /usr/lib/
    echo "  copied ${lib}"
  else
    echo "  MISSING in SDK: ${AARCH64_DIR}/${lib}"
    all_ok=false
  fi
done
sudo ldconfig

if [ -f "${HEXAGON_DIR}/libQnnHtpV68Skel.so" ] && [ -n "${CDSP_DIR}" ]; then
  sudo cp "${HEXAGON_DIR}/libQnnHtpV68Skel.so" "${CDSP_DIR}/"
  echo "OK"
else
  echo " SKIPPED (missing skel file or CDSP_DIR not found)"
  all_ok=false
fi

echo "Setting ADSP_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="${CDSP_DIR};${ADSP_DIR}"
echo "  ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH}"

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
    echo "  MISSING $1 not found by ldconfig"
    all_ok=false
  fi
}

check_lib "libQnnHtp.so"
check_lib "libQnnHtpV68Stub.so"
check_lib "libQnnGpu.so"
[ -n "${CDSP_DIR}" ] && check_file "${CDSP_DIR}/libQnnHtpV68Skel.so"

echo "Checking if .dlc exists"
check_file "$HOME/co-bot-intent-detection/model/dlc/pose_landmark_lite.dlc"
check_file "$HOME/co-bot-intent-detection/model/dlc/lite_stgcn_hmdb51.dlc"

if [ "${all_ok}" = true ]; then
  echo "All checks passed."
else
  echo "One or more checks failed"
fi