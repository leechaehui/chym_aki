"""Prototype Distribution — 어떤 프로토타입이 얼마나 자주 검색됐는가.

소수 프로토타입에 편중되면(=대부분 같은 것만 반환) retrieval 이 무의미. 엔트로피로 다양성 측정.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np


def prototype_distribution(logs: Iterable[dict], *, top_only: bool = True) -> dict:
    """top_only=True 면 top1 만 집계. 반환: 사용횟수·커버리지·정규화 엔트로피."""
    counter: Counter = Counter()
    for r in logs:
        hits = r.get("hits") or []
        if not hits:
            continue
        for h in (hits[:1] if top_only else hits):
            counter[h["prototype_id"]] += 1

    total = sum(counter.values())
    n_used = len(counter)
    if total == 0:
        return {"total_hits": 0, "n_prototypes_used": 0, "entropy_norm": None, "top": []}
    p = np.array(list(counter.values()), float) / total
    ent = float(-(p * np.log(p)).sum())
    ent_norm = round(ent / np.log(n_used), 4) if n_used > 1 else 0.0
    return {
        "total_hits": total,
        "n_prototypes_used": n_used,
        "entropy_norm": ent_norm,              # 1=균등, 0=완전편중
        "top": counter.most_common(10),
    }
