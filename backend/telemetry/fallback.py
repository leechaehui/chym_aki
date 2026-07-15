import os
import json
import logging
from logging.handlers import RotatingFileHandler
import gzip
import shutil
from pathlib import Path

class GzipRotatingFileHandler(RotatingFileHandler):
    def rotation_filename(self, default_name):
        return default_name + ".gz"

    def rotate(self, source, dest):
        if not os.path.exists(source):
            return
        with open(source, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)

fallback_logger = logging.getLogger("telemetry_dlq")
fallback_logger.setLevel(logging.INFO)
fallback_logger.propagate = False

# 로그 디렉터리 — 파일 위치 기준(backend/telemetry/fallback.py → backend/data/logs).
# 절대경로 하드코딩 금지: 어느 트리(team 등)에서 실행하든 그 트리 안으로 기록. env 로 재정의 가능.
log_dir = Path(os.getenv("TELEMETRY_LOG_DIR",
                         str(Path(__file__).resolve().parents[1] / "data" / "logs")))
log_dir.mkdir(parents=True, exist_ok=True)

handler = GzipRotatingFileHandler(
    log_dir / "telemetry_fallback.jsonl",
    maxBytes=500 * 1024 * 1024, # 500MB
    backupCount=5
)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

if not fallback_logger.handlers:
    fallback_logger.addHandler(handler)

def append_to_file_fallback(payload: dict, reason: str):
    """
    Tertiary Fallback: Append JSON to local file with gzip rotation
    """
    try:
        record = {
            "payload": payload,
            "reason": reason,
        }
        fallback_logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception as e:
        # Last resort fallback failure
        print(f"CRITICAL: Failed to write to telemetry DLQ file fallback. {e}")
