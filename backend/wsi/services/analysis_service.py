"""분석 유스케이스 — 단방향 파이프라인 오케스트레이션.

Template Method: analyze 의 단계 순서(캐시→전처리→추론→후처리→매핑→캐시→발행)를 고정.
DIP: 구현이 아니라 포트(SlideRepository/InferenceEngine/ResultCache)에만 의존.
엔진은 provider 로 지연 주입 → 가중치/torch 로드 실패가 목록/타일 기능을 죽이지 않음(장애 격리).
관측성: 단계별 로깅 + 완료 이벤트(Observer seam).
"""
from __future__ import annotations

import logging
from typing import Callable

from wsi.core.errors import SlideNotFoundError
from wsi.domain.mapper import to_result
from wsi.domain.ports import InferenceEngine, ResultCache, SlideRepository
from wsi.schemas.wsi import WsiAnalysisResult, WsiSlideList

log = logging.getLogger("wsi.analysis")


class AnalysisService:
    def __init__(
        self,
        *,
        repo: SlideRepository,
        cache: ResultCache,
        engine_provider: Callable[[], InferenceEngine],
    ):
        self._repo = repo
        self._cache = cache
        self._engine_provider = engine_provider     # 지연 주입(엔진 로드 격리)

    def list_slides(self, stain: str) -> WsiSlideList:
        slides = self._repo.list_slides(stain)
        return WsiSlideList(stain=stain, slides=slides, total=len(slides))

    def analyze(self, *, slide_id: str, stain: str, use_cache: bool = True) -> WsiAnalysisResult:
        from wsi.core.config import cache_version, active_model_variant
        requested = active_model_variant()                       # intent(엔진 없이·GET용)
        ver_req = cache_version(requested)                       # GET = requested 키
        if use_cache:
            cached = self._cache.get(stain, slide_id, ver_req)
            if cached is not None:
                res = WsiAnalysisResult(**cached)
                # cache hit ⇒ requested-키 파일 존재 ⇒ 과거 SET(resolved==requested) ⇒ resolved=requested (정확)
                self._ab_event(slide_id, stain, requested, requested, "cached", True, res.report.status)
                log.info("analyze cache-hit slide=%s stain=%s ver=%s", slide_id, stain, ver_req)
                return res

        bag = self._repo.load_bag(slide_id)                      # 1) 전처리(임베딩 로드)
        # CdssEngine 은 전 stain 768d(ctranspath)로 학습됨. ABMIL용 HE(phikon, 1024d)가 섞여 들어오면
        # np.concatenate 에서 차원이 안 맞아 죽으므로, ctranspath로 뽑아둔 HE_CT 로 반드시 교체한다.
        if "HE_CT" in bag:
            bag["HE"] = bag.pop("HE_CT")
        elif "HE" in bag:
            bag.pop("HE")   # ctranspath-HE 없으면 phikon-HE(차원 불일치)는 빼고 나머지 stain만 사용
        # [infer-single] 멀티모달로 학습했지만 추론은 '표시된 그 슬라이드' 단일 stain 으로 — 같은 검체의
        # 다른 stain(HE 짝 등)이 딸려와도 제외한다. HE 없는 PAS 단독도 판독되게(멀티 게이트 완화와 짝).
        bag = {s: v for s, v in bag.items() if s == stain}
        engine = self._engine_provider()                         # 2) 엔진(지연 로드)
        info = getattr(engine, "variant_info", {"requested": requested, "resolved": requested, "status": "ok"})
        resolved = info.get("resolved", requested)               # Execution(실제 로드된 모델)
        out = engine.analyze(bag, slide_id=slide_id)             # 3) 추론(QC·라우팅·ensemble·attn)
        coords = self._repo.patch_coords(slide_id)               # 4) 후처리 데이터(좌표)

        # Tissue 검출 정보 가져오기
        tissues = None
        svs = self._repo.svs_path(slide_id, stain)
        if svs is not None:
            try:
                from wsi.infra.tile_source import _get_tissues_cached
                tissues = _get_tissues_cached(str(svs))
            except Exception as e:
                log.error("조직 검출 실패 (slide=%s): %s", slide_id, e)

        result = to_result(out, coords, slide_id=slide_id,       # 5) 계약 매핑(Adapter)
                           stain=stain, model_label=engine.model_label,
                           tissues=tissues)
        ver_res = cache_version(resolved)                        # SET = resolved 키(fallback 오염 방지)
        self._cache.set(stain, slide_id, result.model_dump(), ver_res)  # 6) 상태 외부화(versioned)
        self._ab_event(slide_id, stain, requested, resolved, info.get("status", "ok"),
                       False, result.report.status)                # A/B: recompute(miss), 2층 로그
        log.info("analyze computed slide=%s stain=%s req=%s resolved=%s status=%s decision=%s",
                 slide_id, stain, requested, resolved, info.get("status", "ok"), result.report.status)
        self._publish(result)                                    # 7) 완료 발행(Observer)
        return result

    def _ab_event(self, slide_id, stain, requested, resolved, load_status, cache_hit, decision) -> None:
        """A/B pairing 이벤트 — Intent(requested) vs Execution(resolved) 2층.
        grouping=experiment_id(requested 무관), 성능비교=resolved 기준. 실패해도 분석 무영향."""
        import json
        from datetime import datetime
        from wsi.core.config import cache_version, get_settings
        try:
            f = get_settings().cache_dir.parent / "ab_events.jsonl"
            rec = {"ts": datetime.now().isoformat(timespec="seconds"),
                   "experiment_id": f"{slide_id}__{stain}", "slide_id": slide_id, "stain": stain,
                   "requested_variant": requested, "resolved_variant": resolved,
                   "load_status": load_status, "cache_version": cache_version(resolved),
                   "cache_hit": bool(cache_hit), "decision": decision}
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


    def get_result(self, *, stain: str, slide_id: str) -> WsiAnalysisResult:
        from wsi.core.config import cache_version
        cached = self._cache.get(stain, slide_id, cache_version())
        if cached is None:
            raise SlideNotFoundError(f"캐시된 결과 없음: {stain}/{slide_id}")
        return WsiAnalysisResult(**cached)

    def _publish(self, result: WsiAnalysisResult) -> None:
        # Observer seam: 향후 report 생성/알림 구독자를 여기에 연결(현재는 관측 로그).
        log.info("wsi.inference.completed slide=%s stain=%s decision=%s patches=%d",
                 result.slide_id, result.stain, result.report.status, result.n_patches)
