"""
#7 Missing Modality 처리 — 프레임워크 무관 참조 구현 (numpy)

핵심 원칙:
 - 없는 stain을 0/평균 벡터로 '대체하지 않는다'.
 - fusion은 '존재하는 stain만'으로 수행(평균/attention의 분모·softmax에서 missing 제외).
 - stain 존재 여부(mask) 자체를 feature로 부착(stain availability encoding).

모델(torch) 단계에서 이 로직을 nn.Module로 1:1 이식한다.
attention score는 학습 파라미터로 대체되며, missing 위치는 -inf 마스킹 후 softmax.

STAINS 순서 = [HE, PAS, MT, SILVER, IF]
embeddings: dict{stain: vec(D,)} — 존재하는 stain만 key로 포함
"""
import numpy as np

STAINS = ["HE", "PAS", "MT", "SILVER", "IF"]


def mask_vector(embeddings):
    """존재하는 stain을 1로 표시한 (5,) 마스크."""
    return np.array([1.0 if s in embeddings else 0.0 for s in STAINS])


def stain_availability_encoding(embeddings):
    """mask(5,) + n_stains(1,) 를 이은 availability feature."""
    m = mask_vector(embeddings)
    return np.concatenate([m, [m.sum()]])


def masked_mean_fusion(embeddings):
    """존재하는 stain 임베딩만 평균. (missing은 분모에서 제외 → 0-fill 편향 없음)"""
    present = [embeddings[s] for s in STAINS if s in embeddings]
    if not present:
        raise ValueError("stain이 하나도 없음")
    return np.mean(present, axis=0)


def masked_attention_fusion(embeddings, score_fn=None):
    """
    존재하는 stain에 대해서만 attention. missing은 softmax 분모에서 제외.
    score_fn: vec(D,)->scalar (모델에선 학습 attention). 기본=L2 norm(데모용).
    반환: (fused vec(D,), 전체 5칸 attention 가중치(missing=0))
    """
    if score_fn is None:
        score_fn = lambda v: float(np.linalg.norm(v))
    present = [s for s in STAINS if s in embeddings]
    if not present:
        raise ValueError("stain이 하나도 없음")
    scores = np.array([score_fn(embeddings[s]) for s in present])
    w = np.exp(scores - scores.max())
    w = w / w.sum()                       # 존재 stain에 대해서만 정규화
    fused = sum(wi * embeddings[s] for wi, s in zip(w, present))
    full_w = {s: 0.0 for s in STAINS}     # 설명가능성: 5칸 전체 가중치(missing=0)
    for wi, s in zip(w, present):
        full_w[s] = float(wi)
    return fused, full_w


def patient_representation(embeddings, method="attention"):
    """환자 수준 표현 = fused(존재 stain만) ⊕ availability encoding."""
    if method == "mean":
        fused = masked_mean_fusion(embeddings)
        attn = None
    else:
        fused, attn = masked_attention_fusion(embeddings)
    rep = np.concatenate([fused, stain_availability_encoding(embeddings)])
    return rep, attn


def _selftest():
    rng = np.random.default_rng(0)
    D = 8
    # 환자 A: HE/PAS/MT (Silver/IF 없음)
    embA = {s: rng.normal(2.0, 0.5, D) for s in ["HE", "PAS", "MT"]}
    print("mask(A):", mask_vector(embA), " availability:", stain_availability_encoding(embA))

    masked = masked_mean_fusion(embA)
    # 잘못된 방식: 없는 stain을 0벡터로 채워 5개 평균 -> 0 쪽으로 편향
    zerofill = np.mean([embA.get(s, np.zeros(D)) for s in STAINS], axis=0)
    print(f"\nmasked-mean    노름 = {np.linalg.norm(masked):.3f}")
    print(f"zero-fill평균 노름 = {np.linalg.norm(zerofill):.3f}  (3/5로 축소 → 편향)")
    assert np.linalg.norm(masked) > np.linalg.norm(zerofill), "masked는 0-fill 편향이 없어야"
    # masked-mean은 실제 존재 3개 평균과 정확히 일치(없는 stain 영향 0)
    assert np.allclose(masked, np.mean([embA[s] for s in ["HE", "PAS", "MT"]], 0))

    rep, attn = patient_representation(embA, "attention")
    print("\nattention 가중치(missing=0):", {k: round(v, 3) for k, v in attn.items()})
    assert attn["SILVER"] == 0.0 and attn["IF"] == 0.0
    assert abs(sum(attn.values()) - 1.0) < 1e-6
    print("patient representation dim:", rep.shape, "(fused", D, "+ mask5 + n1)")
    print("\n[OK] missing-modality: 0-fill 안 함, 존재 stain만 융합, mask는 feature.")


if __name__ == "__main__":
    _selftest()
