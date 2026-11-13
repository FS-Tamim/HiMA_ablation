#!/bin/zsh
# run_xai.sh - XAI analysis for HRFDCA on a Mac (CPU). First run creates .venv-mac (needs uv: brew install uv).
#   ./run_xai.sh --light                      # 4 epochs, 100 test samples (~2-2.5 h on an M3, 8 GB)
#   ./run_xai.sh --smoke                      # quick check (~3 min)
#   ./run_xai.sh --light --loss focal         # any xai_analysis.py option can be added
# DATA_ROOT=/path/to/folder ./run_xai.sh ...  # folder that contains hierarchical_aggression.csv
set -e
cd "$(dirname "$0")"
DATA_ROOT="${DATA_ROOT:-$HOME/Documents/Research/datasets}"
if [ ! -x .venv-mac/bin/python ]; then
  uv venv --python 3.11 .venv-mac
  uv pip install --python .venv-mac/bin/python torch torchvision -r requirements.txt protobuf
fi
export ABLATION_NUM_WORKERS=0 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
LOG="xai_log_$(date +%Y%m%d_%H%M).txt"
echo "Log: $LOG"
caffeinate -i .venv-mac/bin/python xai_analysis.py --data-root "$DATA_ROOT" "$@" 2>&1 | tee "$LOG"
