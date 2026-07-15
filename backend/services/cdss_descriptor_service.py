"""CDSS v5 디스크립터(OOF) 공급 서비스 — v6.1 §6 tubular injury spectrum 용.

라이브 KPMP 모델(8001 Task-Attention MIL)은 fibrosis/atrophy/inflammation 3개만 예측하고
ATI(ati_severity)·tubulitis(immune) 는 미예측한다. 그러나 cdss_v5(CLAM-lite/CTransPath)
파이프라인이 이 헤드들을 **실제 학습**해 pooled-OOF 예측을 cdss_core(settings.cdss_core_path)에 남겼다.

본 서비스는 그 검증된 OOF 예측을 **환자별(case_code = patient_id)** 로 공급한다:
- ati_severity : 구조적 급성 세뇨관 손상(ATI) — 실제 학습 헤드.
- immune       : 면역성 손상(tubulitis) — 실제 학습 헤드(저커버리지/약AUROC → SHADOW 등급).
- chronic      : 만성 배경(CKD baseline 보조).
- stage3       : KDIGO stage≥3.
OOF 는 라이브 forward 가 아닌 out-of-fold 검증 예측(실데이터·재현 가능). stub/난수 아님.

파일 부재(에어갭 외 PC 등) 시 빈 결과를 반환해 호출측이 graceful 폴백하도록 한다.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from core.config import settings
from core.logging import get_logger

log = get_logger("chym.cdss_descriptor")

# OOF task → 응답 키(camelCase). stage3 는 보조.
_TASK_KEY = {
    "ati_severity": "atiSeverity",
    "immune": "immune",
    "chronic": "chronic",
    "stage3": "stage3",
}


class CdssDescriptorService:
    """cdss_v5 OOF 예측·메트릭 로더(프로세스 1회 로드, 환자별 조회).

    book-keeping 없는 순수 조회 → 결정적·audit 재현. 파일 없으면 available=False.
    """

    def __init__(self, oof_path: Path, metrics_path: Path | None):
        self._preds: dict[str, dict[str, float]] = {}  # {patient_id: {key: p}}
        self._metrics: dict[str, dict] = {}            # {key: {auroc, ci95, n}}
        self.available = False
        if oof_path.exists():
            self._load_oof(oof_path)
            if metrics_path and metrics_path.exists():
                self._load_metrics(metrics_path)
            self.available = True
            log.info("CDSS descriptor OOF loaded: %d patients (%s)", len(self._preds), oof_path)
        else:
            log.warning("CDSS descriptor OOF not found: %s → descriptors unavailable", oof_path)

    def _load_oof(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = _TASK_KEY.get(row.get("task", ""))
                if key is None:
                    continue
                try:
                    self._preds.setdefault(row["patient_id"], {})[key] = round(float(row["p"]), 4)
                except (KeyError, ValueError):
                    continue

    def _load_metrics(self, path: Path) -> None:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("CDSS descriptor metrics load failed: %s", path)
            return
        tasks = meta.get("tasks", {})
        for task, key in _TASK_KEY.items():
            t = tasks.get(task) or {}
            self._metrics[key] = {
                "auroc": t.get("auroc"),
                "ci95": t.get("auroc_ci95"),
                "n": t.get("n"),
            }

    def get(self, case_code: str) -> dict | None:
        """환자(case_code)의 디스크립터 + 메트릭. 코호트 미포함 시 None."""
        pred = self._preds.get(case_code)
        if not pred:
            return None
        return {
            "caseCode": case_code,
            "descriptors": pred,                 # {atiSeverity, immune, chronic, stage3}
            "metrics": self._metrics,            # {key: {auroc, ci95, n}}
            "source": "cdss_v5_full_72p/CLAM-lite/ctranspath (pooled-OOF)",
        }


@lru_cache(maxsize=1)
def get_descriptor_service() -> CdssDescriptorService:
    """프로세스 싱글턴. OOF/메트릭 1회 로드."""
    base = settings.cdss_core_path
    return CdssDescriptorService(base / settings.cdss_oof_file, base / settings.cdss_metrics_file)
