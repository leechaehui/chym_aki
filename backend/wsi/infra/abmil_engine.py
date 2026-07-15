"""ABMIL 분석 서비스 — HE/MT 전용 슬라이드 수준 다중 구조 지표 회귀(8010 aki_wsi 모델 이식).

CdssEngine(Pathology_model, PAS·멀티스테인)과는 입출력 형태가 달라(연속값 회귀 vs
QC/ABSTAIN 리포트) InferenceEngine/EngineOutput 포트에 억지로 끼워맞추지 않고,
WsiAnalysisResult 를 직접 구성하는 병렬 파이프라인으로 둔다(mapper.py 는 CdssEngine 전용).
피처 로딩(SlideRepository)·타일소스(PACS)·캐시(ResultCache) 포트는 기존 것을 그대로 재사용.

공간 오버레이(attn_overlays)는 패치추출 시 저장된 정확한 좌표(_coords.npz, feature_extraction.py)가
있을 때만 그린다 — 조직 위치를 썸네일 등으로 근사해서 "그럴듯한 위치"를 지어내지 않는다(의료 뷰어
원칙: 가짜 좌표 금지). 좌표 사이드카가 없으면 attn_overlays 는 빈 배열로 반환(heatmap 은 위치와
무관한 patch_idx 랭킹이라 그대로 제공).
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from wsi.core.config import WsiSettings
from wsi.core.errors import EngineUnavailableError, SlideNotFoundError
from wsi.domain.ports import ResultCache, SlideRepository
from wsi.infra.feature_extraction import FeatureExtractionService
from wsi.schemas.wsi import (
    WsiAnalysisResult, WsiAttnOverlay, WsiHeatmapCell, WsiLayer, WsiMetric, WsiReport,
)

log = logging.getLogger("wsi.abmil")

TARGET_META: dict[str, dict] = {
    "fibrosisRatio": {"label": "간질 섬유화",   "unit": "%",  "scale": 100.0, "color": "96,165,250"},
    "atrophyRatio":  {"label": "세뇨관 위축",   "unit": "%",  "scale": 100.0, "color": "251,191,36"},
    "tubularInjury": {"label": "세뇨관 손상",   "unit": "%",  "scale": 100.0, "color": "239,68,68"},
    "inflammation":  {"label": "간질 염증",     "unit": "%",  "scale": 100.0, "color": "74,222,128"},
    "artHyalinosis": {"label": "소동맥 유리화", "unit": "/3", "scale": 3.0,   "color": "168,85,247"},
}

_HEATMAP_TOP = 200
_OVERLAY_TOP = 40


class AbmilAnalysisService:
    """HE/MT 전용 ABMIL 추론 유스케이스 — repo 에서 피처(.pt) 로드, 좌표 사이드카가 있을 때만 공간 오버레이."""

    def __init__(self, *, settings: WsiSettings, repo: SlideRepository, cache: ResultCache):
        self._s = settings
        self._repo = repo
        self._cache = cache
        self._models: dict[str, object] = {}
        self._device = None

    # ── 모델 로드(지연, 1회) ────────────────────────────────────────────────
    def _ensure_models(self) -> None:
        if self._models:
            return
        try:
            import torch
        except Exception as e:
            raise EngineUnavailableError(f"ABMIL 로드 실패(torch): {e}") from e

        # aki_wsi/src 경로 탐색 — data_root(=.../aki_wsi) 하위 src 우선.
        candidates = [self._s.data_root / "src", self._s.data_root]
        for c in candidates:
            if (c / "mil_model.py").exists():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                break
        else:
            raise EngineUnavailableError(f"mil_model.py 를 찾을 수 없음(탐색: {candidates})")

        try:
            from mil_model import ABMIL, HE_TARGETS, MT_TARGETS
        except Exception as e:
            raise EngineUnavailableError(f"ABMIL 모델 모듈 임포트 실패: {e}") from e

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_counts = {"HE": len(HE_TARGETS), "MT": len(MT_TARGETS)}
        ckpts = {"HE": self._s.he_ckpt_path, "MT": self._s.mt_ckpt_path}
        for stain, ckpt_path in ckpts.items():
            if not ckpt_path.exists():
                log.warning("ABMIL 체크포인트 없음(%s): %s", stain, ckpt_path)
                continue
            model = ABMIL(in_dim=1024, n_targets=target_counts[stain]).to(self._device)
            ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._models[stain] = model
        if not self._models:
            raise EngineUnavailableError("로드된 ABMIL 체크포인트가 없음")

    # ── 정확 좌표 로드(추출 시 저장된 사이드카) — 없으면 None(근사 금지) ───────
    def _load_exact_positions(
        self, slide_id: str, stain: str,
    ) -> tuple[list[tuple[float, float]], float, np.ndarray, int] | None:
        pt_path = self._repo.feature_path(slide_id, stain)
        if pt_path is None:
            return None
        coords_path = FeatureExtractionService.coords_path(pt_path)
        if not coords_path.exists():
            return None
        with np.load(coords_path) as npz:
            coords = npz["coords"]           # (N, 2) level-0 (x, y)
            patch_l0 = int(npz["patch_l0"])
            slide_w = int(npz["slide_w"])
        if slide_w <= 0 or len(coords) == 0:
            return None
        half = patch_l0 / 2.0
        positions = [(round((x + half) / slide_w, 5), round((y + half) / slide_w, 5))
                     for x, y in coords]
        patch_r = round(half / slide_w, 5)
        return positions, patch_r, coords, patch_l0

    # ── 추론 ────────────────────────────────────────────────────────────────
    def _run(self, stain: str, slide_id: str) -> WsiAnalysisResult:
        import torch
        from mil_model import HE_TARGETS, MT_TARGETS
        targets = HE_TARGETS if stain == "HE" else MT_TARGETS

        if stain not in self._models:
            raise EngineUnavailableError(f"ABMIL 모델 미로드(stain={stain})")

        bag = self._repo.load_bag(slide_id)
        feat = bag.get(stain)
        if feat is None:
            raise SlideNotFoundError(f"피처 없음: slide={slide_id} stain={stain}")

        H = torch.from_numpy(feat).float().to(self._device or "cpu")
        model = self._models[stain]
        with torch.no_grad():
            preds, attn = model(H)
        preds_np = preds.cpu().numpy()
        attn_np = attn.cpu().numpy()

        metrics: list[WsiMetric] = []
        for i, t in enumerate(targets):
            meta = TARGET_META[t]
            raw = float(preds_np[i])
            metrics.append(WsiMetric(key=t, label=meta["label"],
                                     value=round(raw * meta["scale"], 2),
                                     unit=meta["unit"], raw=round(raw, 5)))

        layers = [WsiLayer(key=t, label=TARGET_META[t]["label"], color=TARGET_META[t]["color"],
                           count=None, visible=(t not in ("inflammation", "artHyalinosis")))
                 for t in targets]

        findings = ", ".join(f"{TARGET_META[t]['label']} {m.value:.1f}{m.unit}"
                             for t, m in zip(targets, metrics))
        # diagnosis 는 '진단'/신뢰도 해석이 아니라 비움 — ABMIL 은 회귀값만 내고 ABSTAIN/신뢰도
        # 판정이 없다(항상 ALLOW). 표준 보고서·신뢰도 표기는 프론트가 실제 필드로 구성한다.
        report = WsiReport(findings=findings, diagnosis="", status="ALLOW", updatedAt=None)

        # 공간 오버레이 — 추출 시 저장된 정확 좌표(_coords.npz)가 있을 때만. 없으면 근사하지 않고
        # attn_overlays 를 비워서 반환한다(의료 뷰어: 가짜/추정 위치 표시 금지).
        n_total = int(len(attn_np))
        w_max = float(attn_np.max()) if n_total > 0 else 1.0
        exact = self._load_exact_positions(slide_id, stain)
        pos_list: list[tuple[float, float]] | None = None
        patch_r_val = 0.0
        px_coords: np.ndarray | None = None
        patch_l0 = 0
        if exact is not None:
            positions, patch_r_val, px_coords, patch_l0 = exact
            if len(positions) == n_total:
                pos_list = positions
            else:
                log.warning("좌표 개수 불일치(slide=%s stain=%s): coords=%d attn=%d — 오버레이 생략",
                           slide_id, stain, len(positions), n_total)
                px_coords = None

        top_idx = np.argsort(attn_np)[::-1][:_HEATMAP_TOP]
        heatmap = [WsiHeatmapCell(patch_idx=int(i), weight=round(float(attn_np[i]) / w_max, 5))
                  for i in top_idx if w_max > 0]

        metric_raws = {m.key: max(m.raw, 0.0) for m in metrics}
        raw_sum = sum(metric_raws.values()) or 1.0
        contrib_w = {k: v / raw_sum for k, v in metric_raws.items()}

        overlays: list[WsiAttnOverlay] = []
        if pos_list is not None:
            for idx in top_idx[:_OVERLAY_TOP]:
                i = int(idx)
                w_norm = round(float(attn_np[i]) / w_max, 5) if w_max > 0 else 0.0
                cx, cy = pos_list[i]
                px, py = (int(px_coords[i][0]), int(px_coords[i][1])) if px_coords is not None else (0, 0)
                overlays.append(WsiAttnOverlay(
                    cx=round(cx, 5), cy=round(cy, 5), r=patch_r_val, weight=w_norm,
                    contrib={k: round(v * w_norm, 4) for k, v in contrib_w.items()},
                    px=px, py=py, psize=patch_l0,
                ))

        return WsiAnalysisResult(
            stain=stain, slide_id=slide_id, model_label="ABMIL (aki_wsi, phikon/hibou 인코더)",
            metrics=metrics, layers=layers, report=report,
            heatmap=heatmap, attn_overlays=overlays, n_patches=n_total,
            tissues=[], slide_w=0, slide_h=0,
        )

    # ── 캐시/공개 API(AnalysisService 와 동일 시그니처) ─────────────────────
    # ResultCache 는 (stain, slide_id) 로만 키가 잡혀 CdssEngine(PAS)과 공유된다. HE/MT 는 과거
    # CdssEngine ABSTAIN 캐시가 남아있을 수 있어 "abmil:" 네임스페이스로 분리해 충돌을 막는다.
    def _cache_key(self, stain: str) -> str:
        return f"abmil_{stain}"

    def analyze(self, *, slide_id: str, stain: str, use_cache: bool = True) -> WsiAnalysisResult:
        if use_cache:
            cached = self._cache.get(self._cache_key(stain), slide_id)
            if cached is not None:
                log.info("abmil analyze cache-hit slide=%s stain=%s", slide_id, stain)
                return WsiAnalysisResult(**cached)

        self._ensure_models()
        result = self._run(stain, slide_id)
        self._cache.set(self._cache_key(stain), slide_id, result.model_dump())
        log.info("abmil analyze done slide=%s stain=%s patches=%d", slide_id, stain, result.n_patches)
        return result

    def get_result(self, *, stain: str, slide_id: str) -> WsiAnalysisResult:
        cached = self._cache.get(self._cache_key(stain), slide_id)
        if cached is None:
            raise SlideNotFoundError(f"캐시된 결과 없음: {stain}/{slide_id}")
        return WsiAnalysisResult(**cached)
