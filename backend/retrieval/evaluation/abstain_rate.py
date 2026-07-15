"""Abstain Rate — OOD 게이트로 retrieval 을 기각한 비율.

cross-cohort 안전성의 핵심 지표. 너무 낮으면(=거의 안 기각) OOD가 무력, 너무 높으면 과보수.
"""
from __future__ import annotations

from collections.abc import Iterable


def abstain_rate(logs: Iterable[dict]) -> dict:
    """logs: [{is_ood: bool, ...}]. 반환: {n, n_abstain, abstain_rate}."""
    rows = list(logs)
    n = len(rows)
    n_ab = sum(1 for r in rows if r.get("is_ood"))
    return {
        "n": n,
        "n_abstain": n_ab,
        "abstain_rate": round(n_ab / n, 4) if n else None,
    }
