"""Retrieval CDSS DTO — Evidence Display 전용(진단/reasoning 필드 없음)."""
from schemas.base import CamelModel


class RetrievalQueryIn(CamelModel):
    """Clinical Concept 입력 = 객관 임상값. etiology 는 선택(hint, 저가중)."""
    kdigo_stage: int | None = None
    cr_trend_slope: float | None = None
    oliguria: int | None = None
    egfr: float | None = None
    proteinuria_mg_g: float | None = None
    a1c_pct: float | None = None
    age: float | None = None
    sex: str | None = None
    diabetes: bool | None = None
    hypertension: bool | None = None
    etiology_hint: str = "unknown"
    subject_id: int | None = None
    stay_id: int | None = None          # 지정 시 MIMIC ICU stay 에서 concept 자동 추출
    k: int = 5


class RepresentativeWsiOut(CamelModel):
    slide_id: str
    stain: str | None = None
    source: str = "KPMP"


class RetrievalHitOut(CamelModel):
    prototype_id: str
    label: str
    similarity: float
    confidence: float                   # ⑦ calibration 전까지 = similarity
    n_members: int
    is_rare: bool
    kdigo_band: str | None = None
    etiology_hint: str | None = None
    egfr_mean: float | None = None
    representative_wsi: RepresentativeWsiOut | None = None


class ConceptOut(CamelModel):
    kdigo_stage: int | None = None
    kdigo_band: str
    egfr: float | None = None
    proteinuria: int | None = None
    a1c: int | None = None
    age: float | None = None
    etiology_hint: str
    completeness: float


class OodOut(CamelModel):
    score: float | None = None
    is_ood: bool = False
    message: str | None = None


class RetrievalResultOut(CamelModel):
    concept: ConceptOut
    ood: OodOut
    hits: list[RetrievalHitOut]
    reference_notice: str
