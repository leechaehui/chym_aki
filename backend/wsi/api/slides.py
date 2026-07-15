"""슬라이드/타일 Controller — HTTP ↔ 유스케이스 변환 및 Tissue ROI 타일 라우팅.

DZI 라우팅: 프론트 OpenSeadragon 이 dziUrl 로부터 타일 URL(`..._files/{level}/{col}_{row}.jpeg`)을
유도하므로 디스크립터/타일 두 경로를 모두 제공. _t{index}_files 서픽스를 파싱하여 tissue_index를 판별.
"""
from __future__ import annotations

import re
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from wsi.core.auth import require_view
from wsi.core.errors import TileUnavailableError
from wsi.infra.engine_factory import (
    get_analysis_service, get_pacs_tile_source, get_repository, get_tile_source,
)
from wsi.schemas.wsi import WsiSlideList, WsiStain

# 병리과 화면 전용 — pathology/admin 만 접근(라우터 전체 게이트).
router = APIRouter(tags=["wsi"], dependencies=[Depends(require_view)])


@router.get("/slides", response_model=WsiSlideList)
def list_slides(stain: WsiStain = Query(...)):
    return get_analysis_service().list_slides(stain)


def _parse_id(raw_id: str) -> tuple[str, int]:
    # 30-10018_t1 -> slide_id="30-10018", tissue_index=1
    m = re.match(r"^(.*?)(?:_t(\d+))?$", raw_id)
    if m:
        return m.group(1), int(m.group(2)) if m.group(2) else 0
    return raw_id, 0


@router.get("/dzi/{stain}/{slide_id}")
def dzi_descriptor(stain: WsiStain, slide_id: str, tissue_index: int = Query(0)):
    parsed_id, t_idx = _parse_id(slide_id)
    if t_idx > 0:
        tissue_index = t_idx
    svs = get_repository().svs_path(parsed_id, stain)
    if svs is None:
        raise HTTPException(404, "SVS 없음")
    return Response(
        get_tile_source().dzi_descriptor(svs, tissue_index=tissue_index),
        media_type="application/xml"
    )


@router.get("/dzi/{stain}/{slide_seg}/{level}/{tile}")
def dzi_tile(stain: WsiStain, slide_seg: str, level: int, tile: str):
    seg_name = slide_seg[:-6] if slide_seg.endswith("_files") else slide_seg
    parsed_id, tissue_index = _parse_id(seg_name)

    try:
        col, row = (int(x) for x in tile.split(".")[0].split("_"))
    except ValueError:
        raise HTTPException(400, "타일 좌표 형식 오류")

    svs = get_repository().svs_path(parsed_id, stain)
    if svs is None:
        raise HTTPException(404, "SVS 없음")

    try:
        data = get_tile_source().dzi_tile(svs, level, col, row, tissue_index=tissue_index)
    except TileUnavailableError:
        raise HTTPException(404, "타일 없음")
    return Response(data, media_type="image/jpeg")


@router.get("/thumbnail/{stain}/{slide_id}")
def thumbnail(stain: WsiStain, slide_id: str, size: int = 400, tissue_index: int = Query(0)):
    parsed_id, t_idx = _parse_id(slide_id)
    if t_idx > 0:
        tissue_index = t_idx
    svs = get_repository().svs_path(parsed_id, stain)
    if svs is None:
        raise HTTPException(404, "SVS 없음")
    return Response(
        get_tile_source().thumbnail(svs, size, tissue_index=tissue_index),
        media_type="image/jpeg"
    )



@router.get("/patch/{stain}/{slide_id}")
def get_patch_image(
    stain: WsiStain,
    slide_id: str,
    x: int = Query(...),
    y: int = Query(...),
    width: int = Query(...),
    height: int = Query(...)
):
    """지정된 원본 슬라이드 좌표(x, y)와 크기(width, height)로 패치 이미지를 크롭하여 반환.
    로컬 SVS 가 없으면(PACS 슬라이드) PACS DICOM 캐시에서 크롭한다."""
    svs = get_repository().svs_path(slide_id, stain)
    if svs is None:
        return Response(get_pacs_tile_source().patch(slide_id, x, y, width, height), media_type="image/jpeg")
    try:
        import openslide
        osr = openslide.OpenSlide(str(svs))
        img = osr.read_region((x, y), 0, (width, height)).convert("RGB")
        from wsi.infra.tile_source import _jpeg
        return Response(_jpeg(img), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, f"패치 크롭 실패: {e}")

