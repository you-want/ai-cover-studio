#!/usr/bin/env bash
# 启动 AI翻唱工坊 Web 服务（默认 http://127.0.0.1:8765）
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "尚未安装环境，先运行: bash setup.sh"
  bash setup.sh
fi
source .venv/bin/activate
export TORCH_HOME="${TORCH_HOME:-$PWD/models/torch}"
exec python -m uvicorn app.main:app --host "${VC_HOST:-127.0.0.1}" --port "${VC_PORT:-8765}"
