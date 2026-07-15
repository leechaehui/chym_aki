"""InferenceEngine 구현 — 실제 CdssEngine(5-seed ordinal ensemble) Adapter.

Adapter 패턴: 외부 자산(Pathology_model)의 인터페이스를 우리 도메인 포트(EngineOutput)로
변환. 모델 구현이 바뀌어도 서비스는 영향 없음.
무거운 의존(torch)은 이 파일에만 격리 → 임상 백엔드(8010)는 가볍게 유지.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from wsi.core.config import WsiSettings
from wsi.core.errors import EngineUnavailableError
from wsi.domain.ports import EngineOutput

_MODEL_LABEL = "Task-Attention MIL · ordinal · 5-seed ensemble (SHADOW·비진단)"


class CdssEngineAdapter:
    def __init__(self, settings: WsiSettings):
        self._s = settings
        if str(settings.pkg_root) not in sys.path:
            sys.path.insert(0, str(settings.pkg_root))      # mil.* 임포트 경로
        try:
            from mil.cdss_engine import CdssEngine
            self._engine = CdssEngine(device=settings.device)
        except Exception as e:                              # 가중치/torch 실패 → 503로 격리
            raise EngineUnavailableError(f"엔진 로드 실패: {e}") from e

    @property
    def model_label(self) -> str:
        return _MODEL_LABEL

    @property
    def variant_info(self) -> dict:
        """Intent/Execution 상태(requested/resolved/status). 캐시 키·A/B 로그의 실행기준."""
        try:
            return dict(self._engine.variant_info)
        except Exception:
            return {"requested": "ln", "resolved": "baseline", "status": "unknown", "reason": ""}

    def analyze(self, bag: dict[str, np.ndarray], *, slide_id: str) -> EngineOutput:
        # 모델이 학습한 stain 만 사용 — 단일 stain 변종(pas)에 같은 검체의 companion(HE 등)이 섞이면
        # 미학습 encoder 로 오염돼 과-abstain/차원불일치(768 vs 1024)가 난다. 멀티는 STAIN_KEEP 라 무변.
        stains = getattr(self._engine, "model_stains", None)
        if stains:
            bag = {s: v for s, v in bag.items() if s in stains}
        # STEP 1: 임상 리포트(QC·라우팅·조건부 ensemble) — 엔진 내부 로직 그대로.
        report = self._engine.analyze(bag, slide_id=slide_id, batch=True)
        n_total = int(sum(v.shape[0] for v in bag.values() if v is not None))

        # ABSTAIN 이면 설명가능성 추출 생략(불필요한 forward 회피).
        if report.get("decision") != "ALLOW":
            return EngineOutput(report=report, n_patches=n_total, stain_ids=[], task_attn={})

        # STEP 2: 설명가능성 — ensemble forward 로 패치 attention/ stain_ids 추출.
        stain_ids, task_attn = self._extract_attention(bag)
        return EngineOutput(report=report, n_patches=len(stain_ids) or n_total,
                            stain_ids=stain_ids, task_attn=task_attn)

    def _extract_attention(self, bag: dict[str, np.ndarray]):
        import torch
        dev = self._engine.device
        bag_t = {s: torch.from_numpy(v).to(dev) for s, v in bag.items()
                 if v is not None and v.shape[0] > 0}
        with torch.no_grad():
            outs = [m(bag_t) for m in self._engine.models]
        stain_ids = list(outs[0]["stain_ids"])
        tasks = list(outs[0]["task_attn"].keys())
        task_attn = {
            t: np.mean([o["task_attn"][t].cpu().numpy() for o in outs], axis=0).astype(float).tolist()
            for t in tasks
        }
        return stain_ids, task_attn
