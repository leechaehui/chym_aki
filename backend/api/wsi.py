"""WSI 병리 API — PACS Primary, 로컬 캐시 기반 DZI 타일 서빙.

슬라이드 목록·이미지 타일은 PACS 서버에서 가져오고, ABMIL 추론만 로컬 .pt 파일을 사용한다.
첫 접근 시 PACS에서 DICOM ZIP을 다운로드해 로컬에 캐시하고, 이후 요청은 캐시를 사용한다.
"""
import asyncio
import hashlib
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from core.config import settings
from core.deps import get_current_user
from models.user import User
from services import pacs_client

router = APIRouter(prefix="/wsi", tags=["wsi"])

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
WSI_BASE  = Path(settings.wsi_base_dir) if settings.wsi_base_dir else Path(".")
WSI_CACHE = Path(settings.wsi_cache_dir)
WSI_CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WSI_BASE / "src"))

# ABMIL 로컬 설정 (feature 파일·모델 체크포인트)
CONFIGS: dict[str, dict] = {
    "HE": {
        "feat_dir":    WSI_BASE / "features_phikon_512_he",
        "ckpt":        WSI_BASE / "checkpoints_abmil_phikon_512_he" / "abmil_he_final.pt",
        "model_label": "phikon-v2 ABMIL",
    },
    "MT": {
        "feat_dir":    WSI_BASE / "features_hibou_512_mt",
        "ckpt":        WSI_BASE / "checkpoints_abmil_hibou_512_mt" / "abmil_mt_final.pt",
        "model_label": "Hibou-L ABMIL",
    },
}

TARGET_META: dict[str, dict] = {
    "fibrosisRatio": {"label": "간질 섬유화",   "unit": "%",  "scale": 100.0},
    "atrophyRatio":  {"label": "세뇨관 위축",   "unit": "%",  "scale": 100.0},
    "tubularInjury": {"label": "세뇨관 손상",   "unit": "%",  "scale": 100.0},
    "inflammation":  {"label": "간질 염증",     "unit": "%",  "scale": 100.0},
    "artHyalinosis": {"label": "소동맥 유리화", "unit": "/3", "scale": 3.0},
}

ANALYSIS_CACHE_DIR = WSI_CACHE / "analysis"
ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_executor        = ThreadPoolExecutor(max_workers=2)
_models:          dict[str, object]       = {}
_wsi_cache:       dict[str, object]       = {}   # case_id → wsidicom.WsiDicom
_svs_cache:       dict[str, object]       = {}   # case_id → openslide.OpenSlide
_dz_cache:        dict[str, object]       = {}   # case_id → DeepZoomGenerator
_download_locks:  dict[str, asyncio.Lock] = {}
_download_status: dict[str, dict]         = {}
DEVICE = None

_TILE_SIZE = 256

# SVS 로컬 파일 디렉토리 (stain별)
SVS_DIRS = {
    "HE": WSI_BASE / "wsi_he_local",
    "MT": WSI_BASE / "wsi_mt_local",  # TRI 파일 포함
}

# PACS case_id → {case_code, stain, feat: Path|None}
_slide_registry: dict[str, dict] = {}


# ── 스테인 감지 (PACS 메타데이터 기반) ────────────────────────────────────────

def _detect_stain(case_code: str, description: str | None) -> str | None:
    """case_code / description 에서 스테인을 추론한다."""
    code = case_code.upper()
    desc = (description or "").upper()
    if "_HE_" in code or "_H&E_" in code or "H&E" in desc:
        return "HE"
    if "_MT_" in code or "_TRI_" in code or desc == "MT":
        return "MT"
    return None


# ── ABMIL feature 파일 탐색 ────────────────────────────────────────────────────

def _feat_variants(case_code: str) -> list[str]:
    """PACS case_code(_MT_)와 로컬 파일명(_TRI_) 불일치 보정."""
    codes = [case_code]
    if "_MT_" in case_code:
        codes.append(case_code.replace("_MT_", "_TRI_"))
    elif "_TRI_" in case_code:
        codes.append(case_code.replace("_TRI_", "_MT_"))
    return codes


def _find_feat(stain: str, case_code: str) -> Path | None:
    feat_dir = CONFIGS[stain]["feat_dir"]
    if not feat_dir.exists():
        return None
    for code in _feat_variants(case_code):
        hits = list(feat_dir.glob(f"*_{code}.pt"))
        if hits:
            return hits[0]
    return None


# ── PACS 캐시 관리 ────────────────────────────────────────────────────────────

def _case_cache_dir(case_id: str) -> Path:
    return WSI_CACHE / case_id


_INSTANCE_SIZE_LIMIT_MB = 100  # 이 크기 초과 인스턴스는 건너뜀 (최고해상도 886MB 제외)


def _is_cached(case_id: str) -> bool:
    d = _case_cache_dir(case_id)
    return d.exists() and any(d.glob("*.dcm"))


async def _get_service_token() -> str:
    return await pacs_client.get_token(settings.pacs_employee_id)


# ── SVS 로컬 파일 지원 ────────────────────────────────────────────────────────

def _find_svs(slide_id: str) -> "Path | None":
    """slide_id(UUID 또는 case_code)로 HE/MT 디렉토리에서 로컬 SVS 파일을 찾는다.
    파일명 패턴: {uuid}_{case_code}.svs 또는 {case_code}.svs 또는 {uuid}_*.svs"""
    case_code = _slide_registry.get(slide_id, {}).get("case_code")
    search_keys = [slide_id]
    if case_code and case_code != slide_id:
        search_keys.append(case_code)

    for svs_dir in SVS_DIRS.values():
        if not svs_dir.exists():
            continue
        for key in search_keys:
            for pat in (f"{key}_*.svs", f"{key}.svs", f"*_{key}.svs", f"*_{key}_*.svs"):
                hits = list(svs_dir.glob(pat))
                if hits:
                    return hits[0]
    return None


def _svs_stain_from_path(svs: Path) -> str | None:
    """SVS 파일명에서 HE/MT 스테인을 추출."""
    parts = svs.stem.split("_")
    if len(parts) < 3:
        return None
    s = parts[2].upper()
    if "HE" in s or "H&E" in s:
        return "HE"
    if "TRI" in s or "MT" in s:
        return "MT"
    return None


def _get_openslide_sync(slide_id: str):
    """OpenSlide + DeepZoomGenerator 를 캐시해서 반환."""
    if slide_id not in _svs_cache:
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
        entry = _slide_registry.get(slide_id, {})
        svs   = entry.get("svs") or _find_svs(slide_id)
        if not svs:
            raise HTTPException(404, f"SVS 파일 없음: {slide_id}")
        slide = openslide.OpenSlide(str(svs))
        _svs_cache[slide_id] = slide
        _dz_cache[slide_id]  = DeepZoomGenerator(
            slide, tile_size=_TILE_SIZE, overlap=0, limit_bounds=True
        )
    return _svs_cache[slide_id], _dz_cache[slide_id]


_core_roi_cache: dict[str, dict | None] = {}


def _compute_core_roi_from_thumb(thumb_img, w0: int, h0: int) -> "dict | None":
    """PIL 썸네일 + 원본 크기로 가장 큰 조직 코어의 OSD ROI 계산.
    OSD 좌표계: cx/cy 모두 slide_width 기준 정규화 (width=1, height=aspect_ratio).
    """
    import cv2
    import numpy as np

    aspect   = h0 / max(w0, 1)
    thumb_np = np.array(thumb_img.convert("RGB"))
    hsv      = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    _, mask  = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel   = np.ones((5, 5), np.uint8)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total_px  = int((mask > 0).sum())
    valid_ids = [i for i in range(1, n_labels)
                 if stats[i, cv2.CC_STAT_AREA] >= max(50, total_px * 0.10)]
    if not valid_ids:
        return None

    largest_i = max(valid_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mh, mw    = mask.shape                    # 썸네일 픽셀 크기
    x_t = stats[largest_i, cv2.CC_STAT_LEFT]
    y_t = stats[largest_i, cv2.CC_STAT_TOP]
    w_t = stats[largest_i, cv2.CC_STAT_WIDTH]
    h_t = stats[largest_i, cv2.CC_STAT_HEIGHT]
    pad = 0.05
    # OSD에서 x, y 모두 slide_width 기준이므로 y 변환 시 aspect(=h0/w0=mh/mw) 적용
    return {
        "x": round(max(0.0, x_t / mw - pad * w_t / mw), 5),
        "y": round(max(0.0, y_t / mw - pad * h_t / mw), 5),
        "w": round(min(1.0,      w_t / mw * (1 + 2 * pad)), 5),
        "h": round(min(aspect,   h_t / mw * (1 + 2 * pad)), 5),
    }


def _compute_core_roi_svs(svs_path: Path) -> "dict | None":
    """SVS 썸네일로 가장 큰 코어 OSD ROI 계산."""
    import openslide
    slide  = openslide.OpenSlide(str(svs_path))
    w0, h0 = slide.dimensions
    thumb  = slide.get_thumbnail((max(1, w0 // 32), max(1, h0 // 32)))
    slide.close()
    return _compute_core_roi_from_thumb(thumb, w0, h0)


def _compute_core_roi_dicom(wsi) -> "dict | None":
    """DICOM WSI 썸네일로 가장 큰 코어 OSD ROI 계산."""
    w0, h0 = wsi.size.width, wsi.size.height
    thumb  = wsi.get_thumbnail((min(512, w0), min(512, h0)))
    return _compute_core_roi_from_thumb(thumb, w0, h0)


def _get_exact_patch_coords(
    svs_path: Path,
) -> tuple[list[tuple[float, float] | None], float, dict | None]:
    """피처 추출(05b_extract_features_phikon_wsi.py)과 완전히 동일한 알고리즘으로
    패치 OSD 좌표를 재현한다.

    같은 SVS + 같은 알고리즘 → 정확히 같은 coords 순서 보장.
    OSD 좌표: cx = px_x / slide_w, cy = px_y / slide_w (너비=1 기준)

    멀티코어 슬라이드(두 조직 코어)는 가장 큰 코어만 clean 마스크에 포함.
    파편/작은 코어 패치는 None으로 인덱스를 유지하여 .pt 파일 매핑을 보존한다.
    반환: (coords, patch_r, core_roi)
      core_roi — 가장 큰 코어의 OSD 정규화 좌표 {'x','y','w','h'} 또는 None
    """
    import cv2
    import numpy as np
    import openslide

    import openslide

    PATCH_SIZE       = 512
    TARGET_MAG       = 20.0
    TISSUE_THRESHOLD = 0.5

    slide   = openslide.OpenSlide(str(svs_path))
    w0, h0  = slide.dimensions

    # native magnification → ds_factor (피처 추출과 동일)
    native_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "40"))
    ds_factor  = native_mag / TARGET_MAG
    patch_l0   = int(PATCH_SIZE * ds_factor)

    # 1/32 썸네일 tissue mask (피처 추출과 완전히 동일)
    thumb    = slide.get_thumbnail((max(1, w0 // 32), max(1, h0 // 32)))
    slide.close()

    import cv2
    import numpy as np
    hsv      = cv2.cvtColor(np.array(thumb.convert("RGB")), cv2.COLOR_RGB2HSV)
    _, mask  = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel   = np.ones((5, 5), np.uint8)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 연결 성분 분석 — 가장 큰 코어 하나만 사용
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total_tissue_px = int((mask > 0).sum())
    valid_ids = [i for i in range(1, n_labels)
                 if stats[i, cv2.CC_STAT_AREA] >= max(50, total_tissue_px * 0.10)]
    clean    = np.zeros_like(mask)
    core_roi = None

    if valid_ids:
        largest_i = max(valid_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        clean[labels == largest_i] = 255
        aspect    = h0 / max(w0, 1)
        mh_px, mw_px = mask.shape
        core_roi  = {
            "x": round(max(0.0, stats[largest_i, cv2.CC_STAT_LEFT] / mw_px - 0.05 * stats[largest_i, cv2.CC_STAT_WIDTH] / mw_px), 5),
            "y": round(max(0.0, stats[largest_i, cv2.CC_STAT_TOP] / mh_px * aspect - 0.05 * stats[largest_i, cv2.CC_STAT_HEIGHT] / mh_px * aspect), 5),
            "w": round(min(1.0, stats[largest_i, cv2.CC_STAT_WIDTH] / mw_px * 1.10), 5),
            "h": round(min(aspect, stats[largest_i, cv2.CC_STAT_HEIGHT] / mh_px * aspect * 1.10), 5),
        }
        _core_roi_cache[str(svs_path)] = core_roi
    mh, mw = mask.shape

    # y outer, x inner 스캔 (피처 추출과 동일한 순서)
    # 피처 파일 인덱스와 1:1 대응 유지:
    #   - 주요 조직 패치 → (cx, cy)
    #   - 파편/2번째 코어 패치 → None (렌더링에서 건너뜀)
    coords: list[tuple[float, float] | None] = []
    for py in range(0, h0 - patch_l0, patch_l0):
        for px in range(0, w0 - patch_l0, patch_l0):
            mx1 = int(px / w0 * mw)
            my1 = int(py / h0 * mh)
            mx2 = max(mx1 + 1, int((px + patch_l0) / w0 * mw))
            my2 = max(my1 + 1, int((py + patch_l0) / h0 * mh))
            mx2 = min(mx2, mw); my2 = min(my2, mh)
            if mask[my1:my2, mx1:mx2].mean() / 255 >= TISSUE_THRESHOLD:
                if clean[my1:my2, mx1:mx2].mean() / 255 >= TISSUE_THRESHOLD:
                    cx = (px + patch_l0 / 2) / w0
                    cy = (py + patch_l0 / 2) / w0
                    coords.append((round(cx, 5), round(cy, 5)))
                else:
                    coords.append(None)  # 파편/2번째 코어 — 인덱스 유지

    patch_r = round(patch_l0 / 2 / w0, 5)
    return coords, patch_r, core_roi


async def _download_case_background(case_id: str) -> None:
    """PACS에서 인스턴스 개별 다운로드 (100MB 초과 제외).
    OpenSlide 4.0은 DICOM 파일 하나를 열면 같은 디렉토리의 피라미드를 자동 인식한다."""
    if case_id not in _download_locks:
        _download_locks[case_id] = asyncio.Lock()

    async with _download_locks[case_id]:
        if _is_cached(case_id):
            _download_status[case_id] = {"status": "ready"}
            return

        cache_dir = _case_cache_dir(case_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            token = await _get_service_token()

            # manifest → 실패 시 /download endpoint fallback
            files: list = []
            use_bulk_download = False
            try:
                manifest = await pacs_client.get_manifest(token, case_id)
                files = manifest.get("files", [])
                print(f"[pacs] manifest ok, files={len(files)}")
            except Exception as manifest_err:
                print(f"[pacs] manifest 실패({manifest_err}), /download fallback 시도")
                use_bulk_download = True

            _download_status[case_id] = {"status": "downloading", "downloaded_mb": 0}
            downloaded_mb = 0.0

            # /download endpoint: 전체 케이스를 하나의 파일로 스트리밍
            if use_bulk_download:
                dcm_path = cache_dir / "SM000000.dcm"
                async with httpx.AsyncClient(timeout=3600) as c:
                    async with c.stream(
                        "GET",
                        f"{pacs_client._base()}/{case_id}/download",
                        headers={"Authorization": f"Bearer {token}", **pacs_client._headers()},
                        follow_redirects=True,
                    ) as r:
                        ct = r.headers.get("content-type", "")
                        print(f"[pacs] /download status={r.status_code} content-type={ct}")
                        r.raise_for_status()
                        with open(dcm_path, "wb") as f:
                            async for chunk in r.aiter_bytes(512 * 1024):
                                f.write(chunk)
                                downloaded_mb += len(chunk) / 1024 / 1024
                                _download_status[case_id] = {
                                    "status": "downloading",
                                    "downloaded_mb": round(downloaded_mb, 1),
                                }
                _download_status[case_id] = {"status": "ready"}
                return

            total = len(files)
            if total == 0:
                raise RuntimeError("다운로드할 파일이 없습니다")

            async with httpx.AsyncClient(timeout=600) as c:
                for idx, file_info in enumerate(files):
                    download_url = file_info["download_url"]
                    dcm_path     = cache_dir / f"SM{idx:06d}.dcm"

                    if dcm_path.exists():
                        downloaded_mb += dcm_path.stat().st_size / 1024 / 1024
                        continue

                    # 스트리밍 다운로드, 크기 초과 인스턴스 건너뜀
                    buf  = bytearray()
                    skip = False
                    async with c.stream("GET", download_url, headers={"Authorization": f"Bearer {token}"}) as r:
                        r.raise_for_status()
                        async for chunk in r.aiter_bytes(512 * 1024):
                            buf.extend(chunk)
                            if len(buf) > _INSTANCE_SIZE_LIMIT_MB * 1024 * 1024:
                                skip = True
                                break

                    if not skip:
                        dcm_path.write_bytes(buf)
                        downloaded_mb += len(buf) / 1024 / 1024

                    _download_status[case_id] = {
                        "status":        "downloading",
                        "downloaded_mb": round(downloaded_mb, 1),
                        "step":          f"{idx + 1}/{total}",
                    }

            _download_status[case_id] = {"status": "ready"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            _download_status[case_id] = {"status": "error", "message": str(e)}


# ── wsidicom DZI 서버 ─────────────────────────────────────────────────────────

def _open_wsi_sync(dcm_dir: Path):
    import wsidicom
    return wsidicom.WsiDicom.open(str(dcm_dir))


async def _get_wsi(case_id: str):
    if case_id not in _wsi_cache:
        if not _is_cached(case_id):
            raise HTTPException(503, "슬라이드 준비 중 — POST /wsi/prepare/{slide_id} 를 먼저 호출하세요.")
        wsi = await asyncio.get_event_loop().run_in_executor(
            _executor, _open_wsi_sync, _case_cache_dir(case_id)
        )
        _wsi_cache[case_id] = wsi
    return _wsi_cache[case_id]


def _dzi_xml(width: int, height: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
        f' Format="jpeg" Overlap="0" TileSize="{_TILE_SIZE}">'
        f'<Size Width="{width}" Height="{height}"/></Image>'
    )


def _native_wsidicom_level(wsi, target_scale: float) -> tuple[int, int, float]:
    """가장 적합한 원본 DICOM 레벨과 wsidicom level 파라미터를 반환.

    wsidicom.read_region(pos, level, size):
      - level=N → scale = 2^N 배 축소 공간의 좌표/크기
      - 하지만 실제로는 원본 DICOM 레벨(wsi.levels) 기준으로 내부 변환함
      - native_scale = full_w / lvl.size.width, native_level = round(log2(native_scale))
      - 해당 native_level 을 초과하는 level 파라미터는 경계 초과 오류 유발
    반환: (native_wsi_idx, native_wsidicom_level, native_scale)
    """
    full_w = wsi.size.width
    best_idx = 0
    for i in range(len(wsi.levels) - 1, -1, -1):
        ns = full_w / wsi.levels[i].size.width
        if ns <= target_scale + 0.5:
            best_idx = i
            break
    native_scale = full_w / wsi.levels[best_idx].size.width
    native_level = round(math.log2(max(1.0, native_scale)))
    return best_idx, native_level, native_scale


def _wsi_tile_sync(wsi, dzi_level: int, col: int, row: int) -> bytes:
    from PIL import Image as PILImage

    full_w = wsi.size.width
    full_h = wsi.size.height
    max_dzi = math.ceil(math.log2(max(full_w, full_h)))

    dzi_scale = 2 ** (max_dzi - dzi_level)
    level_w   = math.ceil(full_w / dzi_scale)
    level_h   = math.ceil(full_h / dzi_scale)
    x0 = col * _TILE_SIZE;  y0 = row * _TILE_SIZE
    tw = min(_TILE_SIZE, level_w - x0)
    th = min(_TILE_SIZE, level_h - y0)

    if tw <= 0 or th <= 0:
        raise ValueError(f"tile out of bounds level={dzi_level} ({col},{row})")

    best_idx, native_level, native_scale = _native_wsidicom_level(wsi, dzi_scale)
    best_lvl    = wsi.levels[best_idx]
    extra_scale = dzi_scale / native_scale   # DZI 픽셀 1개 = native 픽셀 extra_scale 개

    # DZI 타일 좌표 → native 레벨 좌표
    x0n = math.floor(x0 * extra_scale)
    y0n = math.floor(y0 * extra_scale)
    x1n = min(best_lvl.size.width,  math.ceil((x0 + tw) * extra_scale))
    y1n = min(best_lvl.size.height, math.ceil((y0 + th) * extra_scale))
    read_w = max(1, x1n - x0n)
    read_h = max(1, y1n - y0n)

    region = wsi.read_region((x0n, y0n), native_level, (read_w, read_h))
    if region.size != (tw, th):
        region = region.resize((tw, th), PILImage.LANCZOS)

    buf = io.BytesIO()
    region.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ── registry helper ────────────────────────────────────────────────────────────

def _registry_entry(slide_id: str) -> dict:
    entry = _slide_registry.get(slide_id)
    if not entry:
        raise HTTPException(404, f"슬라이드 없음: {slide_id} — /wsi/slides 먼저 호출")
    return entry


# ── 모델 로드 (서버 시작 시 1회) ──────────────────────────────────────────────

def load_wsi_models() -> None:
    global DEVICE
    try:
        import torch
        from mil_model import ABMIL, HE_TARGETS, MT_TARGETS
    except ImportError:
        return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_counts = {"HE": len(HE_TARGETS), "MT": len(MT_TARGETS)}
    for stain, cfg in CONFIGS.items():
        ckpt_path = cfg["ckpt"]
        if not ckpt_path.exists():
            continue
        model = ABMIL(in_dim=1024, n_targets=target_counts[stain]).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _models[stain] = model


# ── 레이어 / 리포트 빌더 ──────────────────────────────────────────────────────

def _build_layers(stain: str) -> list[dict]:
    if stain == "HE":
        return [
            {"key": "fibrosisRatio", "label": "간질 섬유화",   "color": "rgba(37,99,235,0.45)",  "count": None, "visible": True},
            {"key": "atrophyRatio",  "label": "세뇨관 위축",   "color": "rgba(245,158,11,0.45)", "count": None, "visible": True},
            {"key": "tubularInjury", "label": "세뇨관 손상",   "color": "rgba(239,68,68,0.45)",  "count": None, "visible": True},
            {"key": "inflammation",  "label": "간질 염증",     "color": "rgba(34,197,94,0.4)",   "count": None, "visible": False},
            {"key": "artHyalinosis", "label": "소동맥 유리화", "color": "rgba(168,85,247,0.4)",  "count": None, "visible": False},
        ]
    return [
        {"key": "fibrosisRatio", "label": "간질 섬유화 (Collagen)", "color": "rgba(37,99,235,0.45)",  "count": None, "visible": True},
        {"key": "atrophyRatio",  "label": "세뇨관 위축",             "color": "rgba(245,158,11,0.45)", "count": None, "visible": True},
        {"key": "artHyalinosis", "label": "소동맥 유리화",           "color": "rgba(168,85,247,0.4)",  "count": None, "visible": False},
    ]


def _build_report(stain: str, metrics: list[dict]) -> dict:
    mv = {m["key"]: m["value"] for m in metrics}
    if stain == "HE":
        findings = (
            f"간질 섬유화 {mv.get('fibrosisRatio', 0):.1f}%, "
            f"세뇨관 위축 {mv.get('atrophyRatio', 0):.1f}%. "
            f"세뇨관 손상 {mv.get('tubularInjury', 0):.1f}%, "
            f"간질 염증 {mv.get('inflammation', 0):.1f}%. "
            f"소동맥 유리화 {mv.get('artHyalinosis', 0):.2f}/3."
        )
    else:
        findings = (
            f"간질 섬유화 {mv.get('fibrosisRatio', 0):.1f}%, "
            f"세뇨관 위축 {mv.get('atrophyRatio', 0):.1f}%. "
            f"소동맥 유리화 {mv.get('artHyalinosis', 0):.2f}/3."
        )
    # diagnosis 는 '진단'이 아니라 정량 요약임을 명확히 한다(과거 고정 문구가 실제 진단처럼
    # 보여 metrics 와 안 맞는다는 오해를 부름). 표준 병리 보고서 초안은 프론트 판독 화면의
    # buildStandardizedReport(reportDraft.ts)가 metrics 로부터 생성한다.
    return {"findings": findings, "diagnosis": "", "status": "draft", "updatedAt": None}


# ── ABMIL 추론 ────────────────────────────────────────────────────────────────

def _detect_tissue_positions(wsi, n_patches: int) -> tuple[list[tuple[float, float]], float]:
    """WSI 썸네일에서 조직 픽셀만 골라 OSD 정규화 좌표를 반환한다.

    근본 한계: PACS DICOM(~10x)과 피처 추출 SVS(40x)는 해상도가 달라
    정확한 위치 복원이 불가능하다. 이 함수는 '조직 위 어딘가' 만 보장한다.

    OSD 좌표: cx = px_x / slide_w, cy = px_y / slide_w (너비=1 기준)
    """
    import numpy as np
    from PIL import Image as PILImage

    slide_w = wsi.size.width
    slide_h = wsi.size.height

    # ── 썸네일 생성 (최대 512px) ────────────────────────────────────────────────
    lvl    = wsi.levels[-1]
    region = wsi.read_region((0, 0), wsi.levels[-1].level,
                              (lvl.size.width, lvl.size.height))
    thumb  = region.convert("RGB")
    MAX_T  = 512
    if lvl.size.width > MAX_T or lvl.size.height > MAX_T:
        sc    = MAX_T / max(lvl.size.width, lvl.size.height)
        thumb = thumb.resize((max(1, int(lvl.size.width * sc)),
                               max(1, int(lvl.size.height * sc))), PILImage.LANCZOS)
    arr = np.array(thumb)
    mask_h, mask_w = arr.shape[:2]

    # ── 조직 마스크: H&E/MT 배경(흰 유리)은 R,G,B 모두 > 215 ─────────────────
    r, g, b  = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    is_bg    = (r > 215) & (g > 215) & (b > 215)
    tissue_m = (~is_bg).astype(np.uint8)   # 1=조직, 0=배경

    # ── 조직 픽셀 좌표 목록 (썸네일 스케일) ──────────────────────────────────
    ys, xs = np.where(tissue_m > 0)
    if len(ys) == 0:
        # 조직 감지 실패 → 빈 리스트 반환
        return [], 0.02

    # 조직 픽셀을 scan order(y 우선)로 정렬 후 n_patches 개 균등 샘플
    order    = np.lexsort((xs, ys))        # (y, x) 기준 정렬
    ys_s     = ys[order]; xs_s = xs[order]
    n_tissue = len(ys_s)
    if n_tissue >= n_patches:
        idxs  = np.linspace(0, n_tissue - 1, n_patches, dtype=int)
    else:
        idxs  = np.arange(n_tissue)        # 있는 것만

    # ── OSD 좌표 변환 ──────────────────────────────────────────────────────────
    # thumb x → OSD cx: cx = (xs / mask_w)          (이미지 너비=1)
    # thumb y → OSD cy: cy = (ys / mask_h) * (slide_h / slide_w)
    slide_aspect = slide_h / max(slide_w, 1)
    coords: list[tuple[float, float]] = []
    for i in idxs:
        cx = float(xs_s[i]) / mask_w
        cy = float(ys_s[i]) / mask_h * slide_aspect
        coords.append((round(cx, 5), round(cy, 5)))

    # patch_r: 썸네일 1픽셀이 OSD 너비 기준으로 얼마인지
    patch_r = round(3.0 / mask_w, 5)    # 시각적으로 적당한 크기
    return coords, patch_r


def _run_abmil(stain: str, slide_id: str,
               patch_positions: list[tuple[float, float]] | None = None,
               patch_r: float | None = None,
               slide_aspect: float = 1.0,
               core_roi: dict | None = None) -> dict:
    import torch
    import numpy as np
    from mil_model import HE_TARGETS, MT_TARGETS
    targets = HE_TARGETS if stain == "HE" else MT_TARGETS

    entry = _registry_entry(slide_id)
    feat_p = entry.get("feat")
    if feat_p is None:
        raise HTTPException(404, f"피처 파일 없음: {entry['case_code']} — AI 분석 불가 (로컬 .pt 필요)")

    feat = torch.load(feat_p, map_location="cpu", weights_only=False).float()
    H = feat.to(DEVICE or "cpu")
    model = _models[stain]
    with torch.no_grad():
        preds, attn = model(H)
    preds_np = preds.cpu().numpy()
    attn_np  = attn.cpu().numpy()

    metrics = []
    for i, t in enumerate(targets):
        meta = TARGET_META[t]
        raw  = float(preds_np[i])
        metrics.append({"key": t, "label": meta["label"], "value": round(raw * meta["scale"], 2), "unit": meta["unit"], "raw": round(raw, 5)})

    top_k   = min(200, len(attn_np))
    top_idx = __import__("numpy").argsort(attn_np)[::-1][:top_k]
    heatmap = [{"patch_idx": int(i), "weight": round(float(attn_np[i]), 5)} for i in top_idx]

    n_total = int(len(attn_np))
    w_max   = float(attn_np.max()) if n_total > 0 else 1.0

    # 조직 위치 기반 좌표 매핑
    # patch_positions = 전체 조직 픽셀에서 균등 샘플 (모두 조직 위)
    # → n_total 개로 재샘플해 전체 조직 영역에 균등 배치
    if patch_positions and len(patch_positions) > 0:
        r_val = patch_r or 0.015
        if len(patch_positions) == n_total:
            # SVS 정확 복원: None(파편) 포함 1:1 매핑 유지
            pos_list = list(patch_positions)
        elif len(patch_positions) > n_total:
            idxs     = np.linspace(0, len(patch_positions) - 1, n_total, dtype=int)
            pos_list = [patch_positions[i] for i in idxs]
        else:
            pos_list = list(patch_positions) + [None] * (n_total - len(patch_positions))
    else:
        # fallback: sqrt 그리드 (조직 감지 실패 시)
        n_cols   = max(1, int(math.sqrt(n_total)))
        n_rows   = max(1, math.ceil(n_total / n_cols))
        r_val    = round(0.5 / n_cols, 6)
        pos_list = [((c + 0.5) / n_cols, (r + 0.5) / n_rows * slide_aspect)
                    for r in range(n_rows) for c in range(n_cols)]

    # 지표별 contrib 가중치 (전체 raw score 비율로 분배)
    metric_raws = {m["key"]: max(float(m["raw"]), 0.0) for m in metrics}
    raw_sum     = sum(metric_raws.values()) or 1.0
    contrib_w   = {k: v / raw_sum for k, v in metric_raws.items()}

    attn_overlays = []
    for idx in top_idx:
        i      = int(idx)
        if i >= len(pos_list) or pos_list[i] is None:
            continue   # 범위 초과 또는 파편 패치 건너뜀
        w_norm = round(float(attn_np[i]) / w_max, 5) if w_max > 0 else 0.0
        cx, cy = pos_list[i]
        attn_overlays.append({
            "cx":     round(cx, 5),
            "cy":     round(cy, 5),
            "r":      r_val,
            "weight": w_norm,
            "contrib": {k: round(v * w_norm, 4) for k, v in contrib_w.items()},
        })

    return {
        "stain": stain, "slide_id": slide_id,
        "model_label": CONFIGS[stain]["model_label"],
        "metrics": metrics, "layers": _build_layers(stain),
        "report": _build_report(stain, metrics),
        "heatmap": heatmap, "attn_overlays": attn_overlays,
        "n_patches": n_total,
        "core_roi": core_roi,
    }


def _analysis_cache_path(slide_id: str, stain: str) -> Path:
    key = hashlib.md5(f"{slide_id}_{stain}".encode()).hexdigest()[:12]
    return ANALYSIS_CACHE_DIR / f"{key}.json"


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@router.get("/slides")
async def list_slides(stain: str = "HE", user: User = Depends(get_current_user)):
    stain = stain.upper()
    if stain not in CONFIGS:
        raise HTTPException(400, "stain은 HE 또는 MT")

    emp_id = user.employee_id or settings.pacs_employee_id or user.username
    try:
        token      = await pacs_client.get_token(emp_id)
        pacs_cases = await pacs_client.list_cases(token)
    except Exception as e:
        raise HTTPException(502, f"PACS 연결 오류 (employee_id={emp_id!r}): {e}")

    # PACS cases 응답 타입 검증
    if not isinstance(pacs_cases, list):
        raise HTTPException(502, f"PACS /cases 응답이 list 가 아님: {type(pacs_cases).__name__} — raw={str(pacs_cases)[:200]}")

    slides = []
    try:
        for case in pacs_cases:
            case_id   = case.get("id", "")
            case_code = case.get("case_code", "")
            if not case_id or not case_code:
                continue
            if _detect_stain(case_code, case.get("description")) != stain:
                continue

            feat   = _find_feat(stain, case_code)
            svs    = _find_svs(case_id)          # SVS 로컬 파일 탐색
            cached = svs is not None or _is_cached(case_id)

            _slide_registry[case_id] = {
                "case_code": case_code, "stain": stain,
                "feat": feat, "svs": svs,
            }

            slides.append({
                "slide_id":     case_id,
                "case_code":    case_code,
                "stain":        stain,
                "has_features": feat is not None,
                "cached":       cached,
                "description":  case.get("description", ""),
            })
    except Exception as e:
        import traceback
        raise HTTPException(500, f"슬라이드 목록 처리 오류: {e}\n{traceback.format_exc()}")

    return {"stain": stain, "slides": slides, "total": len(slides)}


@router.post("/prepare/{slide_id}")
async def prepare_slide(
    slide_id: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(get_current_user),
):
    """PACS에서 슬라이드를 백그라운드로 다운로드해 캐시한다."""
    entry = _slide_registry.get(slide_id, {})
    if entry.get("svs") or _find_svs(slide_id) or _is_cached(slide_id):
        return {"status": "ready"}

    current = _download_status.get(slide_id, {}).get("status")
    if current == "downloading":
        return {
            "status":        "downloading",
            "downloaded_mb": _download_status[slide_id].get("downloaded_mb", 0),
        }

    background_tasks.add_task(_download_case_background, slide_id)
    _download_status[slide_id] = {"status": "downloading", "downloaded_mb": 0}
    return {"status": "downloading", "downloaded_mb": 0}


@router.get("/cache-status/{slide_id}")
async def cache_status(slide_id: str, _: User = Depends(get_current_user)):
    """슬라이드 캐시 상태를 반환한다."""
    entry = _slide_registry.get(slide_id, {})
    if entry.get("svs") or _find_svs(slide_id) or _is_cached(slide_id):
        return {"status": "ready"}
    return _download_status.get(slide_id, {"status": "not_started"})


@router.get("/core-roi/{stain}/{slide_id:path}")
async def get_core_roi(stain: str, slide_id: str, _: User = Depends(get_current_user)):
    """가장 큰 조직 코어의 OSD 뷰포트 ROI 반환. SVS·DICOM 모두 지원."""
    cache_key = f"{stain}:{slide_id}"
    if cache_key in _core_roi_cache:
        return {"core_roi": _core_roi_cache[cache_key]}

    entry = _registry_entry(slide_id)
    svs   = entry.get("svs")

    try:
        if svs:
            roi = await asyncio.get_event_loop().run_in_executor(
                _executor, _compute_core_roi_svs, svs
            )
        else:
            # DICOM 슬라이드: WSI 썸네일 사용
            wsi = await _get_wsi(slide_id)
            roi = await asyncio.get_event_loop().run_in_executor(
                _executor, _compute_core_roi_dicom, wsi
            )
    except Exception:
        roi = None

    _core_roi_cache[cache_key] = roi
    return {"core_roi": roi}


@router.get("/thumbnail/{stain}/{slide_id:path}")
async def thumbnail(stain: str, slide_id: str, size: int = 400):
    entry = _registry_entry(slide_id)

    if entry.get("svs"):
        def _make_svs():
            from PIL import Image as PILImage
            import openslide
            slide = openslide.OpenSlide(str(entry["svs"]))
            w, h  = slide.dimensions
            thumb = slide.get_thumbnail((size, max(1, int(size * h / max(w, 1)))))
            slide.close()
            buf = io.BytesIO()
            thumb.convert("RGB").save(buf, format="JPEG", quality=82)
            return buf.getvalue()
        data = await asyncio.get_event_loop().run_in_executor(_executor, _make_svs)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    wsi = await _get_wsi(slide_id)

    def _make():
        from PIL import Image as PILImage
        _, native_level, _ = _native_wsidicom_level(wsi, wsi.size.width)
        lvl    = wsi.levels[-1]
        region = wsi.read_region((0, 0), native_level, (lvl.size.width, lvl.size.height))
        region.thumbnail((size, size), PILImage.LANCZOS)
        buf = io.BytesIO()
        region.convert("RGB").save(buf, format="JPEG", quality=82)
        return buf.getvalue()

    data = await asyncio.get_event_loop().run_in_executor(_executor, _make)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/dzi/{stain}/{slide_id:path}.dzi")
async def dzi_info(stain: str, slide_id: str, _: User = Depends(get_current_user)):
    entry = _registry_entry(slide_id)

    if entry.get("svs"):
        def _make_svs():
            _, dz = _get_openslide_sync(slide_id)
            return dz.get_dzi("jpeg").encode()
        data = await asyncio.get_event_loop().run_in_executor(_executor, _make_svs)
        return Response(content=data, media_type="application/xml")

    wsi = await _get_wsi(slide_id)
    return Response(content=_dzi_xml(wsi.size.width, wsi.size.height), media_type="application/xml")


@router.get("/dzi/{stain}/{slide_id:path}_files/{level}/{col}_{row}.jpeg")
async def dzi_tile(stain: str, slide_id: str, level: int, col: int, row: int, _: User = Depends(get_current_user)):
    entry = _registry_entry(slide_id)

    if entry.get("svs"):
        def _make_svs():
            _, dz = _get_openslide_sync(slide_id)
            try:
                tile = dz.get_tile(level, (col, row))
            except Exception as e:
                raise HTTPException(404, str(e))
            buf = io.BytesIO()
            tile.convert("RGB").save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        data = await asyncio.get_event_loop().run_in_executor(_executor, _make_svs)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})

    wsi = await _get_wsi(slide_id)

    def _make():
        try:
            return _wsi_tile_sync(wsi, level, col, row)
        except ValueError as e:
            raise HTTPException(404, str(e))

    data = await asyncio.get_event_loop().run_in_executor(_executor, _make)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


class AnalyzeRequest(BaseModel):
    slide_id:  str
    stain:     Literal["HE", "MT"] = "HE"
    use_cache: bool = True


@router.post("/analyze")
async def analyze(req: AnalyzeRequest, _: User = Depends(get_current_user)):
    stain = req.stain.upper()
    if stain not in _models:
        raise HTTPException(503, f"ABMIL 모델 미로드 (stain={stain}) — 서버 재시작 필요. 로드된 모델: {list(_models.keys())}")

    cache_file = _analysis_cache_path(req.slide_id, stain)
    if req.use_cache and cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("attn_overlays"):   # 빈 캐시(이전 버전)는 무시하고 재분석
            return cached

    # 패치 좌표 계산: SVS 있으면 피처 추출과 동일한 정확한 coords, 없으면 DICOM 근사
    patch_positions: list[tuple[float, float]] | None = None
    patch_r_val:     float | None = None
    core_roi_val:    dict | None  = None
    slide_aspect_val: float       = 1.0
    try:
        entry = _registry_entry(req.slide_id)
        svs   = entry.get("svs")
        if svs:
            # SVS 모드: 피처 추출과 완전히 동일한 알고리즘 → 정확한 위치
            import openslide as _osl
            _s = _osl.OpenSlide(str(svs)); w0, h0 = _s.dimensions; _s.close()
            slide_aspect_val = h0 / max(w0, 1)
            patch_positions, patch_r_val, core_roi_val = await asyncio.get_event_loop().run_in_executor(
                _executor, _get_exact_patch_coords, svs
            )
        else:
            # DICOM 모드: 썸네일 기반 근사
            wsi = await _get_wsi(req.slide_id)
            slide_aspect_val = wsi.size.height / max(wsi.size.width, 1)
            def _pos():
                return _detect_tissue_positions(wsi, n_patches=300)
            patch_positions, patch_r_val = await asyncio.get_event_loop().run_in_executor(_executor, _pos)
    except Exception:
        pass  # 실패 시 sqrt-grid fallback

    def _make():
        try:
            return _run_abmil(stain, req.slide_id, patch_positions, patch_r_val, slide_aspect_val, core_roi_val)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            raise RuntimeError(traceback.format_exc()) from e

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, _make)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(500, f"추론 오류: {e}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result


@router.get("/result/{stain}/{slide_id:path}")
async def get_result(stain: str, slide_id: str, _: User = Depends(get_current_user)):
    cache_file = _analysis_cache_path(slide_id, stain.upper())
    if not cache_file.exists():
        raise HTTPException(404, "결과 없음 — POST /wsi/analyze 먼저 호출")
    with open(cache_file, encoding="utf-8") as f:
        return json.load(f)
