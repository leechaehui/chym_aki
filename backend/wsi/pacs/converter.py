"""SVS/TIFF/NDPI 등 → DICOM WSI 변환 (wsidicomizer, openslide 백엔드).

PACS(Orthanc)는 DICOM 만 받으므로, 비-DICOM WSI 는 업로드 전 DICOM WSI 로 변환한다.
변환은 무겁다(대형 슬라이드는 수 분) → 반드시 백그라운드 job 에서 호출할 것.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# openslide 가 읽을 수 있는 대표 WSI 포맷(이외는 변환 시도 후 실패 시 에러)
WSI_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".svslide", ".bif"}


@lru_cache(maxsize=1)
def _ensure_native_libs() -> None:
    """opentile/wsidicomizer 의 네이티브 의존(libjpeg-turbo, turbojpeg.dll)을 런타임이 찾게 보장.

    서버를 'conda activate' 없이 python 을 직접 띄우면 <env>/Library/bin 이 Windows DLL
    검색경로에 없어 turbojpeg.dll 로드가 실패한다(→ 변환 전부 FileNotFoundError). 여기서 현재
    인터프리터(sys.prefix) 기준으로 그 경로를 등록한다 — 절대경로 하드코딩 없이 어느 PC에서나 동작.
    """
    lib_bin = Path(sys.prefix) / "Library" / "bin"          # conda(win) 네이티브 DLL 위치
    dll = lib_bin / "turbojpeg.dll"
    if dll.exists():
        os.environ.setdefault("TURBOJPEG", str(dll))        # opentile 가 우선 참조하는 힌트
        if hasattr(os, "add_dll_directory"):                # py3.8+ Windows: ctypes 검색경로 추가
            try:
                os.add_dll_directory(str(lib_bin))
            except OSError:
                pass


def is_convertible(filename: str) -> bool:
    return Path(filename).suffix.lower() in WSI_SUFFIXES


def convert_to_dicom(src: Path, out_dir: Path, *, tile_size: int = 512) -> list[Path]:
    """src(WSI) → out_dir 에 DICOM WSI(.dcm) 생성. 생성된 파일 경로 목록 반환."""
    _ensure_native_libs()
    from wsidicomizer import WsiDicomizer
    out_dir.mkdir(parents=True, exist_ok=True)
    created = WsiDicomizer.convert(str(src), output_path=str(out_dir), tile_size=tile_size)
    paths = [Path(p) for p in created]
    if not paths:
        raise RuntimeError("변환 결과 DICOM 파일이 없습니다")
    return paths
