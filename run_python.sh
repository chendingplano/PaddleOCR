#!/bin/bash
# Invoked by the pdf-parser Go service as: run_python.sh <script> <args...>
# Sets up the paddleocr venv cleanly regardless of the inherited environment.
VENV=/Users/cding/Workspace/ThirdParty/paddleocr/.venv
export VIRTUAL_ENV="$VENV"
export PYTHONPATH=""
unset PYTHONHOME
exec "$VENV/bin/python" "$@"
