#!/usr/bin/env bash
# FlexSfM Studio launcher
#   ./flexsfm.sh                       -> desktop app
#   ./flexsfm.sh gui PROJECT           -> desktop app with a project open
#   ./flexsfm.sh new|run|all|status ...-> command line (see README)
#
# Python environment: uses $FLEXSFM_VENV if set, else ./.venv if it exists, else the python on PATH.
set -e
cd "$(dirname "$(readlink -f "$0")")"
if [ -n "${FLEXSFM_VENV:-}" ]; then source "$FLEXSFM_VENV/bin/activate"
elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export GLOG_minloglevel=1 PYTHONUNBUFFERED=1
if [ $# -eq 0 ]; then exec python -m flexsfm gui; fi
exec python -m flexsfm "$@"
