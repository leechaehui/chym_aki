"""requirements.txt ↔ 실제 설치 패키지 동기화 유틸.

사용법:
    python -m db.sync_requirements          # 불일치 보고만
    python -m db.sync_requirements --apply  # requirements.txt 자동 갱신

동작:
  1. 현재 pip freeze 결과와 requirements.txt 를 비교
  2. 버전 변경·신규 추가·삭제된 패키지를 감지
  3. --apply 시 requirements.txt 에 반영 (주석·섹션 구조 유지)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REQ_FILE = BACKEND_DIR / "requirements.txt"

# requirements.txt 에서 무시할 접두사 (빌드/에디팅 전용 패키지)
_IGNORE_PREFIXES = {"pip", "setuptools", "wheel", "pkg-resources", "distribute"}


def _parse_req_file(path: Path) -> dict[str, dict]:
    """requirements.txt 를 파싱.
    반환: {패키지명(소문자): {version, line_num, raw_line, comment_section}}
    """
    packages: dict[str, dict] = {}
    current_section = ""
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if line.startswith("#"):
            current_section = line.lstrip("# ").strip()
            continue
        if not line or line.startswith("-"):
            continue
        # 패키지==버전 또는 패키지 (버전 없음)
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\[?[^\]]*\]?\s*(?:[><=!~]+\s*(.+))?", line)
        if m:
            name = m.group(1).lower().replace("-", "_")
            ver = (m.group(2) or "").strip()
            packages[name] = {
                "version": ver,
                "line_num": i,
                "raw_line": raw,
                "section": current_section,
            }
    return packages


def _pip_freeze() -> dict[str, str]:
    """pip freeze 실행 → {패키지명(소문자): 버전}."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    )
    packages: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "==" in line:
            name, ver = line.split("==", 1)
            name_norm = name.lower().replace("-", "_")
            if name_norm not in _IGNORE_PREFIXES:
                packages[name_norm] = ver.strip()
    return packages


def sync(apply: bool = False) -> list[str]:
    """requirements.txt 와 실제 설치 패키지 비교.

    반환: 변경 로그 문자열 목록.
    apply=True 시 requirements.txt 를 갱신한다.
    """
    req_pkgs = _parse_req_file(REQ_FILE)
    installed = _pip_freeze()
    logs: list[str] = []

    # 1) 버전 변경 감지
    for name, info in req_pkgs.items():
        if name in installed:
            if info["version"] and info["version"] != installed[name]:
                logs.append(
                    f"[UPDATE] {name}: {info['version']} → {installed[name]}"
                )
        else:
            # 주석 처리된 패키지(# scikit-learn 등)는 무시
            if not info["raw_line"].strip().startswith("#"):
                logs.append(f"[MISSING] {name} (requirements.txt 에 있지만 미설치)")

    # 2) 신규 패키지 감지 (requirements.txt 에 없는데 설치됨)
    req_names = set(req_pkgs.keys())
    for name, ver in sorted(installed.items()):
        if name not in req_names:
            # 핵심 패키지의 의존성(sub-dependency)은 보고만 하고 자동 추가하지 않음
            logs.append(f"[NEW] {name}=={ver} (설치됨, requirements.txt 에 없음)")

    if not logs:
        logs.append("[OK] requirements.txt 와 설치 패키지가 일치합니다.")
        return logs

    if not apply:
        logs.insert(0, "[DRY-RUN] --apply 플래그로 실제 반영 가능")
        return logs

    # 3) 적용: requirements.txt 갱신
    lines = REQ_FILE.read_text(encoding="utf-8").splitlines()
    updated = False

    for name, info in req_pkgs.items():
        if name in installed and info["version"] and info["version"] != installed[name]:
            old_line = lines[info["line_num"]]
            new_line = old_line.replace(f"=={info['version']}", f"=={installed[name]}")
            lines[info["line_num"]] = new_line
            updated = True

    if updated:
        REQ_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logs.append("[APPLIED] requirements.txt 버전 갱신 완료")

    return logs


def main():
    parser = argparse.ArgumentParser(description="requirements.txt 동기화")
    parser.add_argument("--apply", action="store_true", help="실제 반영")
    args = parser.parse_args()

    print("=" * 60)
    print("requirements.txt ↔ pip freeze 비교")
    print("=" * 60)
    for line in sync(apply=args.apply):
        print(f"  {line}")
    print("=" * 60)


if __name__ == "__main__":
    main()
