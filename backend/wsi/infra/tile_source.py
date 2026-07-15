"""TileSource 구현 — openslide DeepZoom 및 Tissue ROI 기반 가상 DeepZoom 생성.

프론트 OpenSeadragon 이 dziUrl(.dzi 디스크립터)을 받아 타일을 요청한다.
조직 검출(Tissue Detection) 결과에 근거하여 특정 조직의 Bounding Box 영역만 잘라 서빙할 수 있도록 구현.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path
from PIL import Image

from wsi.core.errors import TileUnavailableError
from wsi.infra.tissue_detector import detect_tissues

_TILE_SIZE = 254
_OVERLAP = 1


@lru_cache(maxsize=16)
def _open(svs: str):
    """OpenSlide 핸들 캐시."""
    try:
        import openslide
        osr = openslide.OpenSlide(svs)
        return osr
    except Exception as e:
        raise TileUnavailableError(f"SVS 열기 실패: {svs} ({e})") from e


@lru_cache(maxsize=32)
def _get_tissues_cached(svs: str) -> list[dict[str, int]]:
    """슬라이드별 조직 Bounding Box 목록 캐시."""
    return detect_tissues(Path(svs))


class OpenSlideTileSource:
    def dzi_descriptor(self, svs: Path, tissue_index: int = 0) -> str:
        osr = _open(str(svs))
        tissues = _get_tissues_cached(str(svs))
        
        if not tissues or tissue_index < 0 or tissue_index >= len(tissues):
            # fallback: 전체 슬라이드
            w, h = osr.dimensions
            rx, ry, rw, rh = 0, 0, w, h
        else:
            t = tissues[tissue_index]
            rx, ry, rw, rh = t["x"], t["y"], t["w"], t["h"]
            
        # XML 형식의 DZI 디스크립터 생성 (오픈시드래곤 네임스페이스 호환)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       TileSize="{_TILE_SIZE}"
       Overlap="{_OVERLAP}"
       Format="jpeg">
  <Size Width="{rw}" Height="{rh}"/>
</Image>"""

    @lru_cache(maxsize=4096)
    def _get_tile_bytes_cached(self, svs_str: str, level: int, col: int, row: int, tissue_index: int) -> bytes:
        osr = _open(svs_str)
        tissues = _get_tissues_cached(svs_str)
        
        if not tissues or tissue_index < 0 or tissue_index >= len(tissues):
            w, h = osr.dimensions
            rx, ry, rw, rh = 0, 0, w, h
        else:
            t = tissues[tissue_index]
            rx, ry, rw, rh = t["x"], t["y"], t["w"], t["h"]
            
        img = self._get_roi_tile(osr, rx, ry, rw, rh, level, col, row)
        return _jpeg(img)

    def dzi_tile(self, svs: Path, level: int, col: int, row: int, tissue_index: int = 0) -> bytes:
        try:
            return self._get_tile_bytes_cached(str(svs), level, col, row, tissue_index)
        except Exception as e:
            raise TileUnavailableError(
                f"ROI 타일 생성 실패 L{level} ({col},{row}), tissue={tissue_index}: {e}"
            ) from e

    def thumbnail(self, svs: Path, size: int, tissue_index: int = 0) -> bytes:
        osr = _open(str(svs))
        tissues = _get_tissues_cached(str(svs))
        
        if not tissues or tissue_index < 0 or tissue_index >= len(tissues):
            # fallback: 전체 슬라이드
            return _jpeg(osr.get_thumbnail((size, size)))
        
        t = tissues[tissue_index]
        rx, ry, rw, rh = t["x"], t["y"], t["w"], t["h"]
        
        # 지정된 ROI에 대한 썸네일 생성
        # 스케일 비율 계산
        scale = max(rw, rh) / size
        # 최적의 openslide level 획득
        best_level = 0
        for l_idx, downsample in enumerate(osr.level_downsamples):
            if downsample <= scale:
                best_level = l_idx
            else:
                break
        
        ds_factor = osr.level_downsamples[best_level]
        read_w = max(1, int(round(rw / ds_factor)))
        read_h = max(1, int(round(rh / ds_factor)))
        
        img = osr.read_region((rx, ry), best_level, (read_w, read_h))
        img_rgb = img.convert("RGB")
        
        # 최종 size에 맞춰 리사이즈
        thumb_w = int(round(rw / scale))
        thumb_h = int(round(rh / scale))
        thumb_w = max(1, thumb_w)
        thumb_h = max(1, thumb_h)
        
        img_resized = img_rgb.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        return _jpeg(img_resized)

    def _get_roi_tile(
        self, osr, rx: int, ry: int, rw: int, rh: int, level: int, col: int, row: int
    ) -> Image.Image:
        max_level = int(math.ceil(math.log2(max(rw, rh))))
        if level < 0 or level > max_level:
            raise ValueError(f"Invalid level {level}")

        scale = 2 ** (max_level - level)
        
        # level L 에서의 전체 크기
        w_L = int(math.ceil(rw / scale))
        h_L = int(math.ceil(rh / scale))

        # level L 에서의 타일 픽셀 영역 구하기 (overlap 고려)
        x_in_level = col * _TILE_SIZE - (_OVERLAP if col > 0 else 0)
        y_in_level = row * _TILE_SIZE - (_OVERLAP if row > 0 else 0)
        
        tile_w = _TILE_SIZE + (2 * _OVERLAP if col > 0 else _OVERLAP)
        tile_h = _TILE_SIZE + (2 * _OVERLAP if row > 0 else _OVERLAP)

        # 경계 클리핑
        tile_w = min(tile_w, w_L - x_in_level)
        tile_h = min(tile_h, h_L - y_in_level)

        if tile_w <= 0 or tile_h <= 0:
            raise ValueError("Tile dimension is zero or negative")

        # level 0 (원본) 기준의 픽셀 영역으로 변환
        x_in_level0 = rx + int(round(x_in_level * scale))
        y_in_level0 = ry + int(round(y_in_level * scale))
        w_in_level0 = int(round(tile_w * scale))
        h_in_level0 = int(round(tile_h * scale))

        # 이미지 경계 보호
        orig_w, orig_h = osr.dimensions
        x_in_level0 = max(0, min(x_in_level0, orig_w - 1))
        y_in_level0 = max(0, min(y_in_level0, orig_h - 1))
        w_in_level0 = max(1, min(w_in_level0, orig_w - x_in_level0))
        h_in_level0 = max(1, min(h_in_level0, orig_h - y_in_level0))

        # 최적의 openslide level 결정
        best_level = 0
        for l_idx, downsample in enumerate(osr.level_downsamples):
            if downsample <= scale:
                best_level = l_idx
            else:
                break

        ds_factor = osr.level_downsamples[best_level]
        read_w = max(1, int(round(w_in_level0 / ds_factor)))
        read_h = max(1, int(round(h_in_level0 / ds_factor)))
        
        # read_region 호출
        img = osr.read_region((x_in_level0, y_in_level0), best_level, (read_w, read_h))
        img_rgb = img.convert("RGB")
        
        if img_rgb.size != (tile_w, tile_h):
            img_rgb = img_rgb.resize((tile_w, tile_h), Image.Resampling.BILINEAR)

        return img_rgb


def _jpeg(img, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

