"""A/B pairing 분석기 — ab_events.jsonl 을 experiment_id(=slide__stain)로 묶어
baseline vs LN 을 자동 매칭하고, cache/inference 지표와 판정 일치도를 계산한다.

이 레이어가 "A/B 가능"을 "A/B 자동 분석"으로 완성한다:
  - Experiment Trace ID(experiment_id)로 pairing
  - variant별 cache_hit_rate / recompute(miss) rate
  - pair(양 variant 모두 존재)에서 판정 일치/변화

사용: python -m wsi.tools.ab_report            # 기본 경로(backend/data/ab_events.jsonl)
      python -m wsi.tools.ab_report <path.jsonl>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _default_log() -> Path:
    # backend/wsi/tools/ab_report.py → backend/data/ab_events.jsonl
    return Path(__file__).resolve().parents[2] / "data" / "ab_events.jsonl"


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _resolved(e: dict) -> str:
    # Execution 기준(성능 비교). 구버전 로그(model_variant)와 호환.
    return e.get("resolved_variant") or e.get("model_variant") or "?"


def build_report(events: list[dict]) -> dict:
    # variant별 카운트(resolved=실제 로드 기준) + intent/execution mismatch 계측
    per_variant = defaultdict(lambda: {"n": 0, "hit": 0})
    mismatch = 0
    # experiment_id(requested 무관 grouping) → resolved variant → 최신 이벤트(ts 순)
    exp = defaultdict(dict)
    for e in sorted(events, key=lambda x: x.get("ts", "")):
        v = _resolved(e)                                 # 성능 비교는 resolved 기준
        per_variant[v]["n"] += 1
        per_variant[v]["hit"] += 1 if e.get("cache_hit") else 0
        if e.get("requested_variant") and e.get("requested_variant") != v:
            mismatch += 1                                # Intent–Execution mismatch(fallback)
        exp[e.get("experiment_id", "?")][v] = e  # 최신값으로 덮어씀

    # variant 지표
    variants = {}
    for v, c in per_variant.items():
        n = c["n"] or 1
        variants[v] = {"requests": c["n"], "cache_hit_rate": round(c["hit"] / n, 3),
                       "recompute_rate": round(1 - c["hit"] / n, 3)}

    # pair(양 variant) — baseline vs ln 자동 매칭
    pairs, agree, shifts = [], 0, defaultdict(int)
    for eid, byv in exp.items():
        if "baseline" in byv and "ln" in byv:
            db, dl = byv["baseline"].get("decision"), byv["ln"].get("decision")
            pairs.append({"experiment_id": eid, "baseline": db, "ln": dl, "same": db == dl})
            agree += 1 if db == dl else 0
            shifts[f"{db} -> {dl}"] += 1

    rc_diff = None
    if "baseline" in variants and "ln" in variants:
        rc_diff = round(variants["ln"]["recompute_rate"] - variants["baseline"]["recompute_rate"], 3)

    return {
        "n_events": len(events),
        "n_experiments": len(exp),
        "n_pairs": len(pairs),
        "pair_agreement": round(agree / len(pairs), 3) if pairs else None,
        "decision_shifts": dict(shifts),
        "variants": variants,
        "recompute_rate_diff_ln_minus_baseline": rc_diff,
        "intent_execution_mismatch": mismatch,   # requested != resolved (fallback). >0이면 A/B 신뢰도 주의
        "pairs": pairs,
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_log()
    ev = load_events(path)
    rep = build_report(ev)
    print(f"== A/B report ({path}) ==")
    print(f"events={rep['n_events']} experiments={rep['n_experiments']} pairs={rep['n_pairs']} "
          f"agreement={rep['pair_agreement']}")
    print("variant 지표(cache/inference):")
    for v, m in rep["variants"].items():
        print(f"  {v:9} requests={m['requests']} cache_hit_rate={m['cache_hit_rate']} "
              f"recompute_rate={m['recompute_rate']}")
    print(f"recompute_rate_diff(ln - baseline) = {rep['recompute_rate_diff_ln_minus_baseline']} "
          f"(양수면 LN이 miss-heavy → 비교 skew 주의)")
    print(f"intent-execution mismatch(fallback) = {rep['intent_execution_mismatch']} "
          f"(>0 이면 요청≠실행 — A/B 신뢰도 저하, resolved 기준으로만 해석)")
    print("판정 변화(baseline -> ln, resolved 기준):", rep["decision_shifts"])
    return rep


if __name__ == "__main__":
    main()
