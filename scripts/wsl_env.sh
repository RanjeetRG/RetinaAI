#!/usr/bin/env bash
# Training environment for WSL2 Ubuntu. Source it, don't run it:
#     source /mnt/d/path/to/RetinaAI/scripts/wsl_env.sh
#
# TensorFlow 2.21 finds most of the pip-installed CUDA libraries on its own but not libcusolver, so every
# nvidia/*/lib folder in the venv is put on LD_LIBRARY_PATH.

VENV="$HOME/venvs/retinagen"
NVIDIA_LIBS="$(ls -d "$VENV"/lib/python3.12/site-packages/nvidia/*/lib | paste -sd:)"
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_CPP_MIN_LOG_LEVEL=1  # hide TensorFlow's INFO chatter

source "$VENV/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || return
