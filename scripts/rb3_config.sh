#!/usr/bin/env bash
#
#!/usr/bin/env bash

set -euo pipefail
VERSION="2.50.0.260828"
wget -O "v${VERSION}.zip" "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${VERSION}/v${VERSION}.zip"
unzip "v${VERSION}.zip"
echo "export QAIRT_ROOT=\"$(pwd)/qairt/${VERSION}\"" >> ~/.bashrc
echo "Done. Run: source ~/.bashrc"

CDSP_DIR="/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/cdsp"
ADSP_DIR="/usr/share/qcom/qcm6490/Thundercomm/RB3gen2/dsp/adsp"

echo "== Installing ARM-side libraries to /usr/lib =="
if [ -f /tmp/libQnnHtp.so ] && [ -f /tmp/libQnnHtpV68Stub.so ]; then
  sudo cp /tmp/libQnnHtp.so /tmp/libQnnHtpV68Stub.so /usr/lib/
  sudo ldconfig
  echo "  OK"
else
  echo "  MISSING /tmp/libQnnHtp.so or /tmp/libQnnHtpV68Stub.so"
  echo "  Did you run push_to_rb3.sh from the dev machine first?"
fi

echo "== Installing DSP-side skel library =="
if [ -f /tmp/libQnnHtpV68Skel.so ]; then
  if [ -d "${CDSP_DIR}" ]; then
    sudo cp /tmp/libQnnHtpV68Skel.so "${CDSP_DIR}/"
    echo "  OK"
  else
    echo "  MISSING expected directory: ${CDSP_DIR}"
    echo "  This board's DSP path layout may differ -- run:"
    echo "    find / -type d -iname cdsp 2>/dev/null"
    echo "  and update CDSP_DIR in this script if it's somewhere else."
  fi
else
  echo "  MISSING /tmp/libQnnHtpV68Skel.so"
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
