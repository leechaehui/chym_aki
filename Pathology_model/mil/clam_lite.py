"""
CLAM-lite — MIL 필요성 검증용 baseline 모델 (설계 §8)

역할(중요): CLAM-lite 는 '메인 prediction 모델'이 아니라 **기준선(baseline) + sanity check** 다.
StainAwareMIL/MultiScaleMIL 의 stain-aware fusion·multi-scale·SILVER 제약이 정말 성능에
기여하는지 가르기 위한 최소복잡도 대조군이다.

구조(CLAM 계열을 경량화):
  patch embedding only
    → linear instance scoring (인스턴스 1개당 점수)
    → Top-K selection (K=16, 점수 상위 패치만)
    → simple attention pooling (Top-K 위에서만 softmax attention)
    → slide-level multi-task 예측

명시적 비사용(설계 §8/§10 실패조건 방지):
  - SILVER 미사용(메인/구조 모두). 인스턴스 풀은 MAIN stain(HE/PAS/MT 등 비-SILVER)만.
  - multi-scale 미사용. stain-aware fusion mask 미사용.
  - 인스턴스 풀은 '존재하는 비-SILVER stain 패치를 한 덩어리로' 합쳐 단순화.

태스크: StainAwareMIL 과 동일(immune/chronic/stage3 BCE, ati_severity MSE) — 공정 비교용.
"""
import torch
import torch.nn as nn

from mil.model import SILVER

TOPK = 16   # 설계 §8 핵심 상수


class ClamLite(nn.Module):
    """최소복잡도 Top-K attention MIL baseline."""

    def __init__(self, in_dim=2048, dim=256, att=128, dropout=0.25, topk=TOPK):
        super().__init__()
        self.topk = topk
        # 인스턴스 점수기(linear) — 패치 1개를 스칼라로
        self.instance_scorer = nn.Linear(in_dim, 1)
        # 패치 표현 프로젝션 + 단순 attention pooling
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn = nn.Linear(dim, 1)
        self.head_immune = nn.Linear(dim, 1)
        self.head_ati = nn.Linear(dim, 1)
        self.head_stage3 = nn.Linear(dim, 1)
        self.head_chronic = nn.Linear(dim, 1)

    @staticmethod
    def pool_instances(bag):
        """존재하는 비-SILVER stain 패치를 하나의 인스턴스 풀로 합친다. (n_total, in_dim)."""
        mats = [v for s, v in bag.items()
                if s != SILVER and v is not None and v.shape[0] > 0]
        if not mats:
            # 비-SILVER 가 전무하면(엣지) 가용 패치 전체로 폴백
            mats = [v for v in bag.values() if v is not None and v.shape[0] > 0]
        if not mats:
            raise ValueError("bag에 패치 없음")
        return torch.cat(mats, dim=0)

    def forward(self, bag):
        patches = self.pool_instances(bag)              # (N, in_dim)
        scores = self.instance_scorer(patches).squeeze(-1)   # (N,)
        k = min(self.topk, patches.shape[0])
        top = torch.topk(scores, k).indices             # 상위 K 인스턴스
        sel = patches[top]                              # (k, in_dim)
        h = self.proj(sel)                              # (k, dim)
        a = torch.softmax(self.attn(h), dim=0)          # (k,1) 단순 attention
        z = (a * h).sum(0)                              # (dim,)
        return {
            "immune": self.head_immune(z).squeeze(-1),
            "ati_severity": self.head_ati(z).squeeze(-1),
            "stage3": self.head_stage3(z).squeeze(-1),
            "chronic": self.head_chronic(z).squeeze(-1),
            "n_instances": int(patches.shape[0]),
            "topk": int(k),
        }
