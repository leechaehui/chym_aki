"""온디맨드 패치추출 + 인코딩 — PACS DICOM → 조직 패치 → phikon/hibou 임베딩(.pt).

미리 만들어둔 .pt 피처가 없는 슬라이드(신규 PACS 업로드 등)를 위한 "다음 단계".
aki_wsi_ai/05b_extract_features_phikon_wsi.py(HE)·05g_extract_features_hibou_512_wsi.py(MT)의
조직 마스킹·패치 추출 로직을 그대로 이식했다 — 차이는 .svs 대신 PACS 캐시의 .dcm 파일을
openslide(4.0+, DICOM 자동인식)로 직접 여는 것뿐(검증 완료, 코드 변경 불필요).
백그라운드 스레드로 실행 + 진행률(dict)만 폴링하도록 노출(PACS 다운로드 진행률과 동일 패턴).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from wsi.core.config import WsiSettings
from wsi.infra.manifest_slide_repo import ManifestSlideRepository

log = logging.getLogger("wsi.extract")


def _safe_norm(patch: np.ndarray, norm) -> np.ndarray:
    """stain normalizer 적용(실패 시 원본 유지) — embed_patches._safe_norm 과 동일 규약."""
    if norm is None:
        return patch
    try:
        return norm.transform(patch)
    except Exception:
        return patch

PATCH_SIZE = 512
TARGET_MAG = 20.0
TISSUE_THRESHOLD = 0.5
THUMB_DOWNSAMPLE = 32

# 사용자가 요청하는 stain(HE/MT/PAS) -> 실제로 뽑아야 할 인코더 태그 목록.
# HE 는 두 번 뽑는다: ABMIL(HE, phikon)용과 CdssEngine(HE_CT, ctranspath)용 — 두 모델이
# 서로 다른 인코더로 학습돼 하나로 겸용할 수 없다(차원도 다름: 1024 vs 768).
_ENCODER_TAGS: dict[str, list[str]] = {
    "HE": ["HE", "HE_CT"],
    "MT": ["MT"],
    "PAS": ["PAS"],
}


def _get_tissue_mask(slide) -> np.ndarray:
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((max(1, w // THUMB_DOWNSAMPLE), max(1, h // THUMB_DOWNSAMPLE)))
    thumb_np = np.array(thumb.convert("RGB"))
    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _tissue_patch_coords(slide) -> tuple[list[tuple[int, int]], int, int]:
    """20배율 기준 512px 타일 좌표(level-0) 목록 + (level, patch_l0) 반환."""
    raw_mag = slide.properties.get("openslide.objective-power", "40")
    native_mag = float(str(raw_mag).rstrip("xX ") or "40")
    ds_factor = native_mag / TARGET_MAG
    level = slide.get_best_level_for_downsample(ds_factor)
    patch_l0 = int(PATCH_SIZE * ds_factor)

    mask = _get_tissue_mask(slide)
    mask_h, mask_w = mask.shape
    w0, h0 = slide.dimensions

    coords: list[tuple[int, int]] = []
    for y in range(0, h0 - patch_l0, patch_l0):
        for x in range(0, w0 - patch_l0, patch_l0):
            mx1 = int(x / w0 * mask_w); my1 = int(y / h0 * mask_h)
            mx2 = int((x + patch_l0) / w0 * mask_w); my2 = int((y + patch_l0) / h0 * mask_h)
            region = mask[my1:my2, mx1:mx2]
            if region.size > 0 and region.mean() / 255 >= TISSUE_THRESHOLD:
                coords.append((x, y))
    return coords, level, patch_l0


def _find_openable_file(case_dir: Path) -> Path:
    dcm = sorted(case_dir.glob("*.dcm"))
    if dcm:
        return dcm[0]
    svs = sorted(case_dir.glob("*.svs"))
    if svs:
        return svs[0]
    raise FileNotFoundError(f"열 수 있는 WSI 파일 없음: {case_dir}")


class FeatureExtractionService:
    """PACS 슬라이드 → 조직 패치 → phikon(HE)/hibou(MT) 임베딩(.pt) 온디맨드 추출.

    무거운 인코더(transformers)는 stain 별로 지연 로드(첫 요청 때만).
    진행 상황은 (stain, slide_id) 키의 dict 로 노출 — PACS 다운로드 진행률과 동일 폴링 패턴.
    """

    def __init__(self, *, settings: WsiSettings, repo: ManifestSlideRepository, pacs_tiles_provider):
        self._s = settings
        self._repo = repo
        self._pacs_tiles_provider = pacs_tiles_provider
        self._models: dict[str, tuple] = {}          # stain -> (model, processor|None, device)
        self._stain_norms: dict | None = None         # PAS/HE_CT 동결 normalizer(지연 로드)
        self._status: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    # ── stain 정규화 reference 지연 로드 (CdssEngine train/serve 일치) ────────
    def _get_stain_norms(self) -> dict:
        """PAS(Reinhard)/HE_CT(Macenko) 동결 reference 로 normalizer 복원.
        stain_ref.json 부재/오류 시 {} 반환 → 정규화 생략(기존 동작·경고). 학습이 정규화한
        피처로 CdssEngine 이 학습됐으므로, 정규화 없이 서빙하면 색공간 스큐가 발생한다."""
        if self._stain_norms is not None:
            return self._stain_norms
        self._stain_norms = {}
        path = self._s.stain_ref_path
        if not path.exists():
            log.warning("stain_ref.json 없음 (%s) — PAS/HE_CT 정규화 생략(train/serve 스큐 주의). "
                        "mil/build_stain_ref.py 로 생성 필요.", path)
            return self._stain_norms
        try:
            import sys
            if str(self._s.pkg_root) not in sys.path:
                sys.path.insert(0, str(self._s.pkg_root))
            from mil.stain_norm import ReinhardNormalizer, MacenkoNormalizer
            data = json.loads(path.read_text(encoding="utf-8"))
            for tag, e in data.get("refs", {}).items():
                method = e.get("method")
                if method == "reinhard":
                    n = ReinhardNormalizer()
                    n.t_mean = np.asarray(e["t_mean"], dtype=np.float64)
                    n.t_std = np.asarray(e["t_std"], dtype=np.float64)
                elif method == "macenko":
                    n = MacenkoNormalizer(Io=e.get("Io", 240), beta=e.get("beta", 0.15),
                                          alpha=e.get("alpha", 1))
                    n.t_HE = np.asarray(e["t_HE"], dtype=np.float32)
                    n.t_maxC = np.asarray(e["t_maxC"], dtype=np.float64)
                else:
                    continue
                self._stain_norms[tag] = n
            log.info("stain_ref 로드 완료: %s (%s)", list(self._stain_norms.keys()), path.name)
        except Exception:
            log.exception("stain_ref 로드 실패 — PAS/HE_CT 정규화 생략")
            self._stain_norms = {}
        return self._stain_norms

    # ── 인코더 지연 로드 ─────────────────────────────────────────────────────
    def _ensure_model(self, stain: str):
        if stain in self._models:
            return self._models[stain]
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = self._s.hf_token or None
        from transformers import AutoModel
        if stain == "HE":
            log.info("phikon-v2 로드 중...")
            model = AutoModel.from_pretrained("owkin/phikon-v2", token=token).eval().to(device)
            entry = (model, None, device)
        elif stain == "MT":
            hibou_src = self._s.hibou_local_path or "histai/hibou-L"
            log.info("Hibou-L 로드 중... (%s)", hibou_src)
            from transformers import AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(hibou_src, trust_remote_code=True, token=token)
            model = AutoModel.from_pretrained(hibou_src, trust_remote_code=True, token=token).eval().to(device)
            entry = (model, processor, device)
        elif stain in ("PAS", "HE_CT"):
            log.info("ctranspath 로드 중... (%s)", stain)
            import sys
            if str(self._s.pkg_root) not in sys.path:
                sys.path.insert(0, str(self._s.pkg_root))       # mil.encoders 임포트 경로
            from mil.encoders import build_encoder
            model, _dim, pp = build_encoder("ctranspath", device, img_size=224)
            entry = (model, pp, device)
        else:
            raise ValueError(f"지원 안 하는 stain: {stain}")
        self._models[stain] = entry
        return entry

    # ── 상태 조회 ────────────────────────────────────────────────────────────
    def status(self, slide_id: str, stain: str) -> dict:
        return self._status.get((stain, slide_id), {"status": "not_started"})

    # ── 시작(백그라운드) ─────────────────────────────────────────────────────
    def start(self, slide_id: str, stain: str, case_code: str | None = None) -> dict:
        key = (stain, slide_id)
        current = self._status.get(key, {}).get("status")
        if current in ("downloading", "extracting", "encoding"):
            return self._status[key]
        if self._repo.has_feature(slide_id, stain, case_code):
            self._status[key] = {"status": "ready"}
            return self._status[key]

        self._status[key] = {"status": "downloading", "progress": 0}
        threading.Thread(target=self._run, args=(slide_id, stain, case_code), daemon=True).start()
        return self._status[key]

    def _run(self, slide_id: str, stain: str, case_code: str | None) -> None:
        key = (stain, slide_id)
        try:
            tiles = self._pacs_tiles_provider()
            case_dir = tiles.ensure_downloaded_dir(slide_id)
            wsi_file = _find_openable_file(case_dir)

            self._status[key] = {"status": "extracting", "progress": 0}
            import openslide
            slide = openslide.OpenSlide(str(wsi_file))
            try:
                slide_w, slide_h = slide.dimensions
                coords, level, patch_l0 = _tissue_patch_coords(slide)
                if not coords:
                    self._status[key] = {"status": "error", "message": "조직 패치를 찾지 못함(빈 슬라이드/마스킹 실패)"}
                    return

                # 패치 좌표는 인코더와 무관 — 한 번만 뽑고, 태그별로 인코딩만 반복(HE 는 phikon+ctranspath 2회).
                tags = _ENCODER_TAGS[stain]
                for i, tag in enumerate(tags):
                    model, processor, device = self._ensure_model(tag)
                    self._status[key] = {"status": "encoding", "progress": 0, "total": len(coords)}
                    feats = self._encode(slide, coords, level, patch_l0, tag, model, processor, device, key)

                    out_path = self._save(feats, slide_id, tag, case_code)
                    self._save_coords(out_path, coords, patch_l0, slide_w, slide_h)
                    self._repo.register_feature(slide_id, tag, out_path, case_code)
                    log.info("피처 추출 완료 slide=%s tag=%s n_patches=%d -> %s",
                             slide_id, tag, feats.shape[0], out_path.name)
            finally:
                slide.close()

            self._status[key] = {"status": "ready"}
        except Exception as e:
            log.exception("피처 추출 실패 slide=%s stain=%s", slide_id, stain)
            self._status[key] = {"status": "error", "message": str(e)}

    def _encode(self, slide, coords, level, patch_l0, stain, model, processor, device, key,
               batch_size: int = 32):
        import torch
        from PIL import Image as PILImage
        from torchvision import transforms

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        to_tensor = transforms.ToTensor()

        feats: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(coords), batch_size):
                batch_coords = coords[i:i + batch_size]
                imgs = []
                for x, y in batch_coords:
                    patch = slide.read_region((x, y), level, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                    imgs.append(patch)

                if stain == "HE":
                    imgs_224 = [im.resize((224, 224), PILImage.LANCZOS) for im in imgs]
                    tensors = torch.stack([normalize(to_tensor(im)) for im in imgs_224])
                    out = model(pixel_values=tensors.to(device))
                    batch_feat = out.last_hidden_state[:, 0, :].cpu().numpy()
                elif stain in ("PAS", "HE_CT"):
                    # processor 자리에 mil.encoders.build_encoder 의 preprocess 함수(pp)가 들어옴 —
                    # numpy(B,H,W,3) uint8 -> 정규화된 텐서(디바이스 이미 적용)를 직접 반환.
                    # 학습(embed_patches)과 동일: read 512 -> stain norm @512 -> pp 가 224 다운샘플.
                    # PAS=Reinhard, HE_CT=Macenko (동결 reference). norm 없으면 원본(경고 후 생략).
                    norm = self._get_stain_norms().get(stain)
                    imgs_np = np.stack([_safe_norm(np.asarray(im), norm) for im in imgs])
                    x = processor(imgs_np)
                    out = model(x)
                    batch_feat = out.cpu().numpy()
                else:
                    inputs = processor(images=imgs, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = model(**inputs)
                    batch_feat = out.last_hidden_state[:, 0, :].cpu().numpy()

                feats.append(batch_feat)
                self._status[key] = {"status": "encoding", "progress": min(i + batch_size, len(coords)),
                                     "total": len(coords)}

        empty_dim = 768 if stain in ("PAS", "HE_CT") else 1024
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, empty_dim), dtype=np.float32)

    def _save(self, feats: np.ndarray, slide_id: str, stain: str, case_code: str | None) -> Path:
        import torch
        feat_dir = {"HE": self._s.he_feature_dir, "MT": self._s.mt_feature_dir,
                   "PAS": self._s.pas_feature_dir, "HE_CT": self._s.he_ctranspath_feature_dir}[stain]
        feat_dir.mkdir(parents=True, exist_ok=True)
        name_part = case_code or slide_id
        out_path = feat_dir / f"{slide_id}_{name_part}.pt"
        torch.save(torch.from_numpy(feats), out_path)
        return out_path

    @staticmethod
    def coords_path(pt_path: Path) -> Path:
        """피처(.pt) 옆에 있는 패치 좌표 사이드카 경로(있으면 정확 위치, 없으면 근사)."""
        return pt_path.parent / (pt_path.stem + "_coords.npz")

    def _save_coords(self, pt_path: Path, coords: list[tuple[int, int]], patch_l0: int,
                     slide_w: int, slide_h: int) -> None:
        """패치별 level-0 좌표 + 슬라이드 크기를 저장 — 나중에 attention 오버레이를
        근사(썸네일 조직 샘플링) 대신 정확한 위치로 그릴 수 있게 한다."""
        np.savez(self.coords_path(pt_path),
                 coords=np.array(coords, dtype=np.int64),
                 patch_l0=patch_l0, slide_w=slide_w, slide_h=slide_h)
