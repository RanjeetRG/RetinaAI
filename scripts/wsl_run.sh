#!/usr/bin/env bash
# Run a command in the WSL training environment and log it to outputs/<name>.log, ending with EXIT=<code>.
#
#     bash scripts/wsl_run.sh train_efficientnetb0 python -u -m cnn.train --backbone efficientnetb0
#
# From Windows (PowerShell), detached so it survives closing the terminal:
#     Start-Process wsl.exe -ArgumentList "-d","Ubuntu","--","bash","/mnt/d/path/to/RetinaAI/scripts/wsl_run.sh",
#         "train_efficientnetb0","python","-u","-m","cnn.train" -WindowStyle Hidden

source "$(dirname "${BASH_SOURCE[0]}")/wsl_env.sh"
log="outputs/$1.log"
shift
"$@" > "$log" 2>&1
echo "EXIT=$?" >> "$log"
