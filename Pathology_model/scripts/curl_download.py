"""
KPMP 균형 데이터셋 다운로더 - curl.exe 엔진 (단일 스트림, stall 자동탈출)

python requests가 이 서버의 대용량 풀파일 전송에서 간헐적으로 connection을
붙잡힌 채 멈추는 문제가 있어, 검증된 curl.exe로 전송을 위임한다.
 - --speed-time 120 --speed-limit 2048 : 120초간 2KB/s 미만이면 끊고 재시도(stall 탈출)
 - --retry 6 --retry-all-errors        : 500/타임아웃 포함 재시도
 - 다운로드 후 manifest의 정확한 바이트와 일치할 때만 최종 파일로 원자적 교체
 - 이미 정확히 받은 파일은 skip (재개)

KPMP 다운로드 엔드포인트는 동시연결을 거부하므로 반드시 단일 스트림으로만 동작.
"""
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd

SELECTED = (Path(__file__).resolve().parents[2] / "selected_manifest.csv")
BASE_DIR = (Path(__file__).resolve().parents[2] / "data/raw")
# stain 그룹을 이 순서로 '완전히 끝낸 뒤' 다음 그룹으로 넘어간다.
GROUP_ORDER = ["wsi_he", "wsi_pas", "wsi_mt", "wsi_silver", "wsi_if"]
GROUP_PASSES = 4  # 그룹 내 재시도 패스 수(일시적 네트워크 실패 흡수)
DOWNLOAD_URL = "https://atlas.kpmp.org/api/v1/file/download/{package_id}/{file_name}"


def dest_path(row):
    grp = row["_grp"]
    pid = str(row["redcap_id"]).replace(";", "_")
    suffix = grp.replace("wsi_", "").upper()
    # 하나의 package에 여러 슬라이드가 있어 package_id는 충돌함.
    # file_name 앞 UUID(file_id)는 파일별로 고유하므로 이것으로 네이밍.
    fid = str(row["file_name"]).split("_")[0][:8]
    ext = ".tif" if grp == "wsi_if" else ".svs"
    return BASE_DIR / grp / f"{pid}_{suffix}_{fid}{ext}"


def have(row):
    d = dest_path(row)
    return d.exists() and d.stat().st_size > 0


def fetch_one(row):
    """단일 파일 다운로드. ('done'|'fail', size, name, msg) 반환."""
    dest = dest_path(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(row["file_size"])
    # 일부 IF 파일명에 공백·()가 있어 URL 인코딩 필수(미인코딩 시 curl (3) malformed)
    url = DOWNLOAD_URL.format(
        package_id=quote(str(row["package_id"])),
        file_name=quote(str(row["file_name"])),
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl.exe", "-sS", "-L", "--fail",
        "--retry", "6", "--retry-delay", "5", "--retry-all-errors",
        "--connect-timeout", "30",
        # 풀파일 연결이 헤더 대기로 멈추는 경우(서버가 range는 주는데 full을 붙잡음)
        # speed-time(전송 시작 후에만 동작)이 안 걸리므로 max-time으로 강제 상한.
        # 최대 파일도 이 안에 받으며, 멈춘 연결은 끊고 --retry로 새 연결 재시도.
        "--max-time", "420",
        "--speed-time", "45", "--speed-limit", "2048",
        "-o", str(tmp), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    size = tmp.stat().st_size if tmp.exists() else 0
    if r.returncode == 0 and size > 0:
        tmp.replace(dest)
        warn = ""
        if expected and abs(size - expected) > max(4096, expected * 0.02):
            warn = f"  [!카탈로그 {expected/1024**2:.0f}MB와 불일치]"
        return ("done", size, dest.name, warn)
    if tmp.exists():
        tmp.unlink()
    msg = (r.stderr or "").strip()[:120] or f"rc={r.returncode} size={size}/{expected}"
    return ("fail", 0, dest.name, msg)


def main():
    df = pd.read_csv(SELECTED)
    rows_all = df.to_dict("records")
    total = len(rows_all)
    # 시작 시점에 이미 받아둔 파일을 진척에 반영
    skip = sum(1 for r in rows_all if have(r))
    acc_bytes = sum(dest_path(r).stat().st_size for r in rows_all if have(r))
    done = 0
    unfinished = []

    print(f"curl 다운로더 시작: {total}파일 (단일 스트림, 그룹순서 처리). "
          f"기존 {skip}개 보유", flush=True)

    # 그룹을 GROUP_ORDER 순으로 '완전히' 끝낸 뒤 다음 그룹으로 진행
    for grp in GROUP_ORDER:
        rows_g = [r for r in rows_all if r["_grp"] == grp]
        if not rows_g:
            continue
        for p in range(1, GROUP_PASSES + 1):
            pending = [r for r in rows_g if not have(r)]
            if not pending:
                break
            if p > 1:
                print(f"  [{grp}] 재시도 패스 {p}/{GROUP_PASSES}: 남은 {len(pending)}개, 10초 대기",
                      flush=True)
                time.sleep(10)
            for row in pending:
                st, size, name, msg = fetch_one(row)
                if st == "done":
                    done += 1
                    acc_bytes += size
                    print(f"[{skip+done}/{total}] OK {name} ({size/1024**2:.0f}MB) "
                          f"누적 {acc_bytes/1024**3:.2f}GB{msg}", flush=True)
                else:
                    print(f"  FAIL {name} - {msg}", flush=True)
        got = sum(1 for r in rows_g if have(r))
        still = [r for r in rows_g if not have(r)]
        print(f"=== [{grp}] 그룹 완료: {got}/{len(rows_g)} "
              f"{'(전부 완료)' if not still else f'(미완 {len(still)}개)'} ===", flush=True)
        unfinished.extend(dest_path(r).name for r in still)

    print("=" * 56, flush=True)
    print(f"전체: 보유 {skip} / 신규 {done} / 미완 {len(unfinished)} / "
          f"총 {acc_bytes/1024**3:.2f}GB", flush=True)
    if unfinished:
        print("미완 목록:", flush=True)
        for n in unfinished:
            print(f"  - {n}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
