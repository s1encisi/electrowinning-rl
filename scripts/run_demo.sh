#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m electrowinning_rl demo --config "$project_root/configs/demo.json" --output "${1:-runs/demo}"
