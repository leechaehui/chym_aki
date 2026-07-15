"""
다운로드 watchdog: curl_download.py 가 371개 완료 전에 종료되면 자동 재시작.

curl_download.py 는 재개식(기존 파일 skip)이라 몇 번을 재실행해도 안전하다.
원인 불명의 프로세스 자체종료(관측됨)에 대비한 안전망.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEL = ROOT / "selected_manifest.csv"
BASE = ROOT / "data/raw"
SCRIPT = ROOT / "backend/scripts/curl_download.py"
MAX_ROUNDS = 40


def dest_for(r):
    grp = r["_grp"]; pid = str(r["redcap_id"]).replace(";", "_")
    suf = grp.replace("wsi_", "").upper()
    fid = str(r["file_name"]).split("_")[0][:8]
    ext = ".tif" if grp == "wsi_if" else ".svs"
    return BASE / grp / f"{pid}_{suf}_{fid}{ext}"


def count_done(df):
    done = 0
    for _, r in df.iterrows():
        d = dest_for(r)
        if d.exists() and d.stat().st_size == int(r["file_size"]):
            done += 1
    return done


def main():
    df = pd.read_csv(SEL)
    total = len(df)
    for rnd in range(1, MAX_ROUNDS + 1):
        done = count_done(df)
        print(f"[watchdog] round {rnd}: 시작 전 {done}/{total} 완료", flush=True)
        if done >= total:
            print("[watchdog] 전체 완료. 종료.", flush=True)
            return
        # curl_download.py 실행 (자체 로그는 stdout으로 흘러 watchdog 로그에 합쳐짐)
        rc = subprocess.run([sys.executable, str(SCRIPT)]).returncode
        after = count_done(df)
        print(f"[watchdog] round {rnd} 종료 rc={rc}: {after}/{total} 완료", flush=True)
        if after >= total:
            print("[watchdog] 전체 완료. 종료.", flush=True)
            return
        if after == done:
            # 한 라운드에 0개 진전 -> 서버/네트워크 문제. 백오프 후 재시도
            print("[watchdog] 진전 없음, 30초 후 재시도", flush=True)
            time.sleep(30)
        else:
            time.sleep(5)
    print(f"[watchdog] MAX_ROUNDS 도달, 미완료 잔여 {total-count_done(df)}개", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
