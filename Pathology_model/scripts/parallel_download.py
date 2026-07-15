"""
KPMP 균형 데이터셋 병렬 다운로더 (재개/무결성 지원)

selected_manifest.csv(371파일)를 읽어 동시 N개로 다운로드한다.
- 이미 받은 파일은 'manifest의 정확한 바이트 수와 일치'할 때만 skip (순차 다운로더를
  강제종료하며 잘렸을 수 있는 partial 파일은 size 불일치로 자동 재다운로드).
- 파일명 규칙은 build_balanced_dataset.py와 동일하게 유지(상호 호환).

사용:
    python parallel_download.py            # 동시 5개(기본)
    python parallel_download.py --workers 6
"""
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

MAX_RETRIES = 4          # 500/타임아웃 시 재시도 횟수
BACKOFF_BASE = 3.0       # 백오프 기준(초): 3,6,12,24s

SELECTED = (Path(__file__).resolve().parents[2] / "selected_manifest.csv")
BASE_DIR = (Path(__file__).resolve().parents[2] / "data/raw")
DOWNLOAD_URL = "https://atlas.kpmp.org/api/v1/file/download/{package_id}/{file_name}"

_print_lock = threading.Lock()
_counter = {"done": 0, "skip": 0, "fail": 0, "bytes": 0}


def dest_path(row):
    grp = row["_grp"]
    pid = str(row["redcap_id"]).replace(";", "_")
    suffix = grp.replace("wsi_", "").upper()
    short = str(row["package_id"])[:8]
    ext = ".tif" if grp == "wsi_if" else ".svs"
    return BASE_DIR / grp / f"{pid}_{suffix}_{short}{ext}"


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def worker(row, total):
    dest = dest_path(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(row["file_size"])

    # 무결성 기반 skip: 존재 + 정확한 크기 일치
    if dest.exists() and dest.stat().st_size == expected:
        with _print_lock:
            _counter["skip"] += 1
            _counter["bytes"] += expected
        return ("skip", dest.name)

    url = DOWNLOAD_URL.format(package_id=row["package_id"], file_name=row["file_name"])
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 18):
                        f.write(chunk)
            size = tmp.stat().st_size
            # 크기 검증 후 원자적 교체
            if expected and abs(size - expected) > max(1024, expected * 0.001):
                tmp.unlink(missing_ok=True)
                raise ValueError(f"size mismatch got={size} expected={expected}")
            tmp.replace(dest)
            with _print_lock:
                _counter["done"] += 1
                _counter["bytes"] += size
                n = _counter["done"] + _counter["skip"]
                log(f"[{n}/{total}] OK {dest.name} ({size/1024**2:.0f}MB) "
                    f"누적 {_counter['bytes']/1024**3:.2f}GB")
            return ("done", dest.name)
        except Exception as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                log(f"  retry {attempt}/{MAX_RETRIES-1} {dest.name} ({e}); {wait:.0f}s 후 재시도")
                time.sleep(wait)
    with _print_lock:
        _counter["fail"] += 1
    log(f"  FAIL {dest.name} - {last_err}")
    return ("fail", dest.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(SELECTED)
    total = len(df)
    rows = df.to_dict("records")
    log(f"병렬 다운로드 시작: {total}파일, 동시 {args.workers}개")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, r, total) for r in rows]
        for _ in as_completed(futs):
            pass

    log("=" * 56)
    log(f"완료: 신규 {_counter['done']} / 기존skip {_counter['skip']} / 실패 {_counter['fail']}")
    log(f"총 용량: {_counter['bytes']/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
