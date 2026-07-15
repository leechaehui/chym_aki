"""Retrieval Consistency — 유사 concept → 유사 결과인가(근접/섭동 안정성).

Ground Truth 가 없으므로 "비슷한 임상 입력은 비슷한 프로토타입을 검색해야 한다"는
일관성을 평가한다. concept 벡터 간 거리와 top1 일치/순위 상관을 본다.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def neighbor_consistency(
    records: Iterable[dict], *, k: int = 5, concept_key: str = "concept_vec",
) -> dict:
    """records: [{concept_vec: list[float], top1: prototype_id}].

    각 레코드의 concept-공간 kNN 이웃이 같은 top1 을 검색한 비율(=local consistency).
    """
    rows = [r for r in records if r.get(concept_key) is not None and r.get("top1")]
    n = len(rows)
    if n < 2:
        return {"n": n, "consistency_at_k": None}
    X = np.stack([np.asarray(r[concept_key], float) for r in rows])
    tops = [r["top1"] for r in rows]
    # cosine 거리 행렬
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    agree = []
    for i in range(n):
        nn = np.argsort(sim[i])[::-1][:k]
        agree.append(np.mean([tops[j] == tops[i] for j in nn]))
    return {"n": n, "k": k, "consistency_at_k": round(float(np.mean(agree)), 4)}


def perturbation_stability(
    concept_vec: list[float], retrieve_fn: Callable[[np.ndarray], str],
    *, n_trials: int = 20, sigma: float = 0.02, seed: int = 42,
) -> dict:
    """concept 벡터에 작은 noise 를 줘도 top1 이 유지되는가(robustness)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(concept_vec, float)
    base_top = retrieve_fn(base)
    same = sum(retrieve_fn(base + rng.normal(0, sigma, base.shape)) == base_top
               for _ in range(n_trials))
    return {"base_top1": base_top, "stability": round(same / n_trials, 4), "n_trials": n_trials}
