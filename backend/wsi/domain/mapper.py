"""Mapper — EngineOutput(내부 표현) → WsiAnalysisResult(프론트 계약).

이 파일이 '모델 세계'와 '프론트 세계'의 유일한 접점이다(Adapter). 모델 출력이 바뀌어도
프론트 계약은 여기만 고치면 된다 → 결합을 한 곳에 격리(변경 대응력).

원칙: 의료 뷰어이므로 '가짜 좌표 금지'. 공간 오버레이는 manifest 좌표가 실제로
해결될 때만 생성하고, 불가하면 빈 배열로 graceful degrade(프론트가 처리).
"""
from __future__ import annotations

from datetime import datetime

from wsi.domain.ports import EngineOutput, StainCoords
from wsi.domain.strategy import DESCRIPTORS, grade_to_percent
from wsi.schemas.wsi import (
    WsiAnalysisResult, WsiAttnOverlay, WsiHeatmapCell, WsiLayer, WsiMetric, WsiReport,
)

_HEATMAP_TOP = 200      # 패치별 heatmap 상한(payload 제어)
_OVERLAY_TOP = 40       # 공간 오버레이 상한(종합 주목도 기준, 시각 혼잡/payload 제어)
_PER_TASK_TOPK = 12     # 지표별 상위 K 패치는 종합 순위에 밀려도 포함(지표별 히트맵 보존)
_OVL_HARD_MAX = 90      # 종합 상위 + 지표별 top-K 합집합 총 상한(payload 제어)


def _load_expl_calib():
    """EXPL 시각 정규화 파라미터(backend/wsi/expl_visual_calibration.json). VISUALIZATION ONLY —
    decision/router/margin/abstain 에는 어떤 경우에도 영향 없음. 파일 없으면 안전 기본값."""
    import json
    from pathlib import Path
    try:
        c = json.loads((Path(__file__).resolve().parents[1] / "expl_visual_calibration.json")
                       .read_text(encoding="utf-8"))
        return (int(c.get("schema_version", 1)),
                int(c.get("norm_percentile", 99)), float(c.get("overlay_coverage_top_frac", 0.10)),
                int(c.get("overlay_max", _OVERLAY_TOP)), int(c.get("heatmap_top", _HEATMAP_TOP)),
                int(c.get("overlay_per_task_topk", _PER_TASK_TOPK)), int(c.get("overlay_hard_max", _OVL_HARD_MAX)))
    except Exception:
        return 1, 99, 0.10, _OVERLAY_TOP, _HEATMAP_TOP, _PER_TASK_TOPK, _OVL_HARD_MAX


_SCHEMA_V, _NORM_PCT, _COV_FRAC, _OVL_MAX, _HM_TOP, _PER_TASK_K, _OVL_HARDMAX = _load_expl_calib()
# 버전 문자열 = 실제 mapper 가 쓰는 frozen 값에서 도출(캐시 키·A/B의 1:1 보장; JSON만 바뀌고
# 렌더는 그대로인 불일치 방지 — 이 값이 곧 실제 렌더 동작을 대표).
EXPL_VERSION = f"expl{_SCHEMA_V}"
NORM_VERSION = f"p{_NORM_PCT}c{int(round(_COV_FRAC * 100))}"


def _pct(sorted_vals, pct):
    """정렬된 리스트의 pct(0-100) 백분위. 빈 리스트면 1e-9."""
    n = len(sorted_vals)
    if n == 0:
        return 1e-9
    return sorted_vals[min(n - 1, int(round((pct / 100.0) * (n - 1))))]


def to_result(
    out: EngineOutput,
    coords: dict[str, StainCoords],
    *,
    slide_id: str,
    stain: str,
    model_label: str,
    tissues: list[dict[str, int]] | None = None,
) -> WsiAnalysisResult:
    report = out.report
    decision = report.get("decision", "ABSTAIN")
    allowed = decision == "ALLOW"
    grades: dict = report.get("finalPrediction") or {}

    calconf: dict = report.get("calibratedConfidence") or {}
    metrics = _metrics(grades, calconf) if allowed else []
    layers = _layers() if allowed else []
    heatmap, overlays = (_explain(out, coords) if allowed else ([], []))

    sc = coords.get(stain)
    slide_w = sc.slide_w if sc else 0
    slide_h = sc.slide_h if sc else 0

    from wsi.schemas.wsi import WsiTissue
    tissues_dto = []
    if tissues:
        for idx, t in enumerate(tissues):
            tissues_dto.append(WsiTissue(id=idx, x=t["x"], y=t["y"], w=t["w"], h=t["h"], area=t["area"]))

    return WsiAnalysisResult(
        stain=stain, slide_id=slide_id, model_label=model_label,
        metrics=metrics, layers=layers,
        report=_report(report, decision),
        heatmap=heatmap, attn_overlays=overlays, n_patches=out.n_patches,
        tissues=tissues_dto, slide_w=slide_w, slide_h=slide_h,
    )


def _metrics(grades: dict, calconf: dict | None = None) -> list[WsiMetric]:
    calconf = calconf or {}
    out = []
    for d in DESCRIPTORS:
        g = grades.get(d.engine_task)
        if g is None:
            continue
        conf = calconf.get(d.engine_task)
        out.append(WsiMetric(key=d.metric_key, label=d.label, unit="%",
                             value=grade_to_percent(g, d.banff_edges), raw=float(g),
                             confidence=float(conf) if conf is not None else None))
    return out


def _layers() -> list[WsiLayer]:
    return [WsiLayer(key=d.metric_key, label=d.label, color=d.color, count=None, visible=True)
            for d in DESCRIPTORS]


def _report(report: dict, decision: str) -> WsiReport:
    if decision == "ALLOW":
        findings = "; ".join(report.get("pathologyFindingsSummary", []))
        diagnosis = report.get("confidenceInterpretation", "")
    else:
        findings = report.get("uncertaintyLimitation", "신뢰도 부족 — 보고 보류")
        diagnosis = report.get("clinicalRecommendation", "병리의 직접 검토 필요")
        if isinstance(diagnosis, list):
            diagnosis = " ".join(diagnosis)
    unc = report.get("uncertainty")
    return WsiReport(findings=findings, diagnosis=diagnosis, status=decision,
                     updatedAt=datetime.now().isoformat(timespec="seconds"),
                     uncertainty=float(unc) if unc is not None else None)


def _explain(out: EngineOutput, coords: dict[str, StainCoords]):
    """task_attn → heatmap(비공간) + attn_overlays(공간, 좌표 가능 시)."""
    n = out.n_patches
    if n == 0 or not out.task_attn:
        return [], []
    tasks = list(out.task_attn.keys())
    # 패치별 평균 attention(모든 task)
    mean_attn = [sum(out.task_attn[t][i] for t in tasks) / len(tasks) for i in range(n)]
    # [EXPL 시각 정합] per-slide robust scale(p99)+coverage → baseline/LN 색·밀도 비교가능(시각화 전용).
    sv = sorted(mean_attn)
    scale = _pct(sv, _NORM_PCT) or 1e-9                       # max 대신 p99(collapse 편향 제거)
    cov_thr = _pct(sv, 100.0 * (1.0 - _COV_FRAC))             # 상위 coverage 만 오버레이(변종 무관 동일밀도)

    top_idx = sorted(range(n), key=lambda i: mean_attn[i], reverse=True)[:_HM_TOP]
    heatmap = [WsiHeatmapCell(patch_idx=i, weight=round(min(1.0, mean_attn[i] / scale), 4)) for i in top_idx]

    overlays = _overlays(out, coords, mean_attn, scale, cov_thr)
    return heatmap, overlays


def _overlays(out: EngineOutput, coords: dict[str, StainCoords], mean_attn, scale, cov_thr) -> list[WsiAttnOverlay]:
    tasks = list(out.task_attn.keys())
    task_scale = {t: (_pct(sorted(out.task_attn[t]), _NORM_PCT) or 1e-9) for t in tasks}  # per-task p99(색 비교가능)
    metric_key = {d.engine_task: d.metric_key for d in DESCRIPTORS}

    # 지표별 상위 K 패치(전역 인덱스)를 미리 확보 → 종합(mean) 순위에 밀려도 포함해야
    # 지표별 히트맵이 비지 않는다. 순위는 task 내 상대값이라 raw attention 정렬로 충분.
    n = out.n_patches
    keep_idx: set[int] = set()
    for t in tasks:
        if t not in metric_key:
            continue
        vals = out.task_attn[t]
        keep_idx.update(sorted(range(n), key=lambda i: vals[i], reverse=True)[:_PER_TASK_K])

    cand: list[tuple[int, WsiAttnOverlay]] = []
    for stain, sc in coords.items():
        if not sc.coords or not sc.slide_w or not sc.displayed_slide:
            continue
        rows = [i for i, s in enumerate(out.stain_ids) if s == stain]
        # patch_coords 가 load_bag 임베딩 순서와 1:1 정렬(EXPL2) → rows[k] ↔ sc.coords[k].
        for k in range(min(len(rows), len(sc.coords))):
            i = rows[k]
            if mean_attn[i] < cov_thr and i not in keep_idx:  # coverage 컷 — 단, 지표별 top-K 는 통과
                continue
            x, y, size, slide_file = sc.coords[k]
            if slide_file != sc.displayed_slide:
                # 환자당 물리 슬라이드가 여러 개 — 화면에 띄운 파일이 아니면 좌표 공간이 달라 제외
                continue
            w = min(1.0, mean_attn[i] / scale)
            contrib = {metric_key[t]: round(min(1.0, out.task_attn[t][i] / task_scale[t]), 3)
                       for t in tasks if t in metric_key}
            cand.append((i, WsiAttnOverlay(
                cx=round((x + size / 2) / sc.slide_w, 5),
                cy=round((y + size / 2) / sc.slide_w, 5),
                r=round((size / 2) / sc.slide_w, 5),
                weight=round(w, 4), contrib=contrib,
                px=int(round(x)),
                py=int(round(y)),
                psize=int(round(size)),
            )))

    # 종합 상위 _OVL_MAX 를 기본으로 두고, weight cap 밖으로 밀린 지표별 top-K 를
    # _OVL_HARDMAX 상한까지 추가로 포함(payload 제어).
    cand.sort(key=lambda io: io[1].weight, reverse=True)
    selected = cand[:_OVL_MAX]
    chosen = {ci for ci, _ in selected}
    for ci, o in cand[_OVL_MAX:]:
        if len(selected) >= _OVL_HARDMAX:
            break
        if ci in keep_idx and ci not in chosen:
            selected.append((ci, o))
            chosen.add(ci)
    return [o for _, o in selected]

