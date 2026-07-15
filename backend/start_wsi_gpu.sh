#!/usr/bin/env bash
# WSI 서버(8001) 단독 실행 — GPU 서버용. 세팅은 wsi/GPU_SETUP.md 참고.
# 사용법: ./start_wsi_gpu.sh [conda env 이름(기본 chym_wsi)]
set -e
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${1:-chym_wsi}"

python -m uvicorn wsi_main:app --host 0.0.0.0 --port 8001
