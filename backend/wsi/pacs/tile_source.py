"""PACS DICOM WSI 타일소스 — 인스턴스를 받아 wsidicom 으로 열고 DZI 타일/썸네일 생성.

PACS 는 타일 스트리밍 엔드포인트가 없으므로, 케이스의 DICOM 인스턴스를 로컬 캐시로 받아
wsidicom 으로 피라미드를 열고 OpenSeadragon 용 DZI 를 on-the-fly 로 만든다.

⚠️ DZI 변환부(_read_dzi_tile)는 실제 DICOM WSI 데이터로 검증 필요(현재 creds/데이터 미보유).
썸네일·다운로드·열기 구조는 표준 wsidicom API 기준으로 구현.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path

from wsi.core.config import WsiSettings
from wsi.core.errors import PacsError, TileUnavailableError
from wsi.pacs.gateway import PacsGateway

_TILE = 254
_OVERLAP = 1


class PacsDicomTileSource:
    def __init__(self, settings: WsiSettings, gateway: PacsGateway):
        self._s = settings
        self._gw = gateway
        self._root = settings.cache_dir / "pacs"      # 케이스별 DICOM 캐시

    # ── 케이스 DICOM 캐시 확보 + wsidicom 열기 ───────────────────────────
    def _ensure_downloaded(self, case_id: str) -> Path:
        d = self._root / case_id
        if d.exists() and any(d.iterdir()):
            return d
        d.mkdir(parents=True, exist_ok=True)

        try:
            files = self._gw.get_manifest(case_id).get("files", [])
        except PacsError:
            files = []  # manifest 미지원/실패 → 아래에서 전체 ZIP 다운로드로 폴백

        if files:
            from concurrent.futures import ThreadPoolExecutor

            def _download_one(f):
                iid = f["instance_id"]
                target = d / f"{iid}.dcm"
                if target.exists():
                    return
                try:
                    data = self._gw.download_instance(case_id, iid)
                    target.write_bytes(data)
                except Exception as e:
                    import logging
                    logging.getLogger("wsi.pacs").warning(f"인스턴스 다운로드 실패 {iid}: {e}")

            with ThreadPoolExecutor(max_workers=15) as executor:
                list(executor.map(_download_one, files))
        else:
            # manifest 엔드포인트가 없는 PACS 서버 대응 — 케이스 전체 ZIP(/download)을 받아 풀어쓴다.
            import io
            import zipfile
            zip_bytes = self._gw.download_case(case_id)
            try:
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    zf.extractall(d)
            except zipfile.BadZipFile:
                raise TileUnavailableError(
                    f"PACS 케이스 다운로드 결과가 비어있거나 손상됨: {case_id}"
                ) from None

        if not any(d.iterdir()):
            raise TileUnavailableError(f"PACS 케이스 인스턴스 없음: {case_id}")

        return d

    def open_wsi(self, case_id: str):
        """공개 접근자 — ABMIL 분석 파이프라인이 슬라이드 크기/썸네일 조회에 사용."""
        return self._open(case_id)

    def ensure_downloaded_dir(self, case_id: str) -> Path:
        """공개 접근자 — 다운로드를 보장하고 로컬 DICOM 캐시 디렉토리를 반환.
        openslide(4.0+, DICOM 자동인식)로 직접 여는 등 wsidicom 외 용도에 사용."""
        return self._ensure_downloaded(case_id)

    @lru_cache(maxsize=4)
    def _open(self, case_id: str):
        from wsidicom import WsiDicom
        folder = self._ensure_downloaded(case_id)
        try:
            return WsiDicom.open(str(folder))
        except Exception as e:
            raise TileUnavailableError(f"wsidicom 열기 실패({case_id}): {e}") from e

    def _base_size(self, slide) -> tuple[int, int]:
        size = slide.size                              # wsidicom Size(width,height)
        return int(size.width), int(size.height)

    # ── DZI ──────────────────────────────────────────────────────────────
    def dzi_descriptor(self, case_id: str) -> str:
        w, h = self._base_size(self._open(case_id))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
            f'Format="jpeg" Overlap="{_OVERLAP}" TileSize="{_TILE}">'
            f'<Size Width="{w}" Height="{h}"/></Image>'
        )

    def dzi_tile(self, case_id: str, level: int, col: int, row: int) -> bytes:
        slide = self._open(case_id)
        w, h = self._base_size(slide)
        max_level = math.ceil(math.log2(max(w, h)))
        if level < 0 or level > max_level:
            raise TileUnavailableError(f"잘못된 DZI 레벨 {level}")

        scale = 2 ** (max_level - level)               # base 대비 이 DZI 레벨의 다운샘플
        wsi_level = max_level - level                  # wsidicom 가상 레벨(2^level 다운샘플)

        # 이 DZI 레벨 좌표계에서의 전체 크기 + 타일 영역(표준 DZI overlap 규약)
        w_l = math.ceil(w / scale)
        h_l = math.ceil(h / scale)
        x = col * _TILE - (_OVERLAP if col > 0 else 0)
        y = row * _TILE - (_OVERLAP if row > 0 else 0)
        tw = _TILE + (2 * _OVERLAP if col > 0 else _OVERLAP)
        th = _TILE + (2 * _OVERLAP if row > 0 else _OVERLAP)
        tw = min(tw, w_l - x)
        th = min(th, h_l - y)
        if tw <= 0 or th <= 0:
            raise TileUnavailableError(f"타일 영역 없음 L{level}({col},{row})")

        # wsidicom 이 실제로 고를 피라미드 레벨과, 그 레벨에서 이 타일이 쓸 수 있는 픽셀수.
        # (DZI 가상레벨의 ceil 크기와 실제 레벨 크기가 달라 오버뷰·가장자리 타일이 실제 레벨
        #  경계를 넘어 out-of-bounds 404 나던 문제 방지.)
        try:
            wl = slide.pyramids.get(slide.selected_pyramid).get_closest_by_level(wsi_level)
            sf = max(1, wl.calculate_scale(wsi_level))
            avail_w = wl.size.width // sf - x
            avail_h = wl.size.height // sf - y
        except Exception:
            wl, avail_w, avail_h = None, tw, th        # 내부 API 변동 시 기본 동작 유지

        try:
            if wl is not None and (avail_w < 1 or avail_h < 1):
                # DZI 레벨이 가장 깊은 실제 피라미드 레벨보다도 더 축소(최저 오버뷰):
                # 실제 레벨 전체를 읽어 이 타일 크기로 축소한다.
                full = slide.read_region((0, 0), int(wl.level),
                                         (int(wl.size.width), int(wl.size.height)))
                region = full.resize((int(tw), int(th)))
            else:
                if wl is not None:                     # 실제 레벨 경계 안으로 클램프
                    tw = min(tw, avail_w)
                    th = min(th, avail_h)
                # wsidicom 은 요청 level 좌표계로 영역을 읽고 가장 가까운 실제 피라미드 레벨에서
                # 다운샘플해 돌려준다 → level-0 전체를 읽던 OOM/지연을 회피.
                region = slide.read_region((int(x), int(y)), int(wsi_level),
                                           (int(tw), int(th)))
        except Exception as e:
            raise TileUnavailableError(f"PACS 타일 생성 실패 L{level}({col},{row}): {e}") from e
        return _jpeg(region)

    def patch(self, case_id: str, x: int, y: int, width: int, height: int) -> bytes:
        """레벨 0(원본 해상도) 기준 임의 좌표 크롭 — 패치 확대 검사(WSIViewer 클릭 줌)용."""
        slide = self._open(case_id)
        try:
            region = slide.read_region((int(x), int(y)), 0, (int(width), int(height)))
        except Exception as e:
            raise TileUnavailableError(f"PACS 패치 크롭 실패({case_id}): {e}") from e
        return _jpeg(region)

    def thumbnail(self, case_id: str, size: int) -> bytes:
        slide = self._open(case_id)
        try:
            img = slide.read_thumbnail((size, size))
        except Exception as e:
            raise TileUnavailableError(f"PACS 썸네일 실패({case_id}): {e}") from e
        return _jpeg(img)


def _jpeg(img, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
