"""
WSI 타일 샘플링 (tifffile + zarr, openslide 불필요)

조직(tissue) 영역을 검출해 그 안에서만 타일을 샘플링한다.
배경/공백 타일과 저품질(거의 흰색) 타일은 제외.
"""
import numpy as np
import tifffile
import zarr


def _read_level(path, level):
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        lv = level if level < len(s.levels) else len(s.levels) - 1
        za = zarr.open(s.levels[lv].aszarr(), mode="r")
        shapes = [l.shape for l in s.levels]
    return za, shapes, lv


def _tissue_mask(thumb_rgb, sat_thresh=0.10, val_thresh=0.95):
    """HSV 채도 기반 조직 마스크. 채도 높고 너무 밝지 않은 픽셀=조직."""
    import cv2
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    return (s > sat_thresh) & (v < val_thresh)


def sample_tissue_tiles(path, n_tiles=16, tile=256, work_level=1,
                        min_tissue_frac=0.5, seed=0, max_tries=400):
    """
    work_level에서 tile x tile 타일을 조직 영역 위주로 최대 n_tiles개 반환.
    반환: list[np.ndarray(tile,tile,3) uint8]
    """
    rng = np.random.default_rng(seed)
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        nlev = len(s.levels)
        wl = min(work_level, nlev - 1)
        za = zarr.open(s.levels[wl].aszarr(), mode="r")
        H, W = za.shape[0], za.shape[1]
        # 썸네일(최저 레벨)로 조직 마스크
        thumb = np.asarray(zarr.open(s.levels[-1].aszarr(), mode="r")[:])
    if thumb.ndim == 2:
        thumb = np.stack([thumb] * 3, -1)
    mask = _tissue_mask(thumb[..., :3])
    th, tw = mask.shape
    tiles = []
    tries = 0
    while len(tiles) < n_tiles and tries < max_tries:
        tries += 1
        y = int(rng.integers(0, max(1, H - tile)))
        x = int(rng.integers(0, max(1, W - tile)))
        # 마스크 좌표로 매핑해 조직 여부 빠르게 사전판정
        my = min(th - 1, int(y / H * th))
        mx = min(tw - 1, int(x / W * tw))
        if not mask[my, mx]:
            continue
        patch = np.asarray(za[y:y + tile, x:x + tile])[..., :3]
        if patch.shape[:2] != (tile, tile):
            continue
        # 타일 내부 조직 비율 재확인(흰 배경 과다 제외)
        import cv2
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        frac = ((hsv[..., 1] / 255.0 > 0.10) & (hsv[..., 2] / 255.0 < 0.95)).mean()
        if frac < min_tissue_frac:
            continue
        tiles.append(patch)
    return tiles
