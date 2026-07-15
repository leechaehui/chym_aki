"""
Phase C — Stain-Aware Multi-task MIL 모델 (torch)

구조:
 - 입력: 환자별 bag = {stain: (n_patches, 2048)} (존재하는 stain만) + modality mask
 - stain별 독립 인코더: proj(2048->D) + gated-attention MIL pooling -> stain 벡터(D) + 패치 attention
 - missing-aware fusion: 존재 stain 벡터에 대해서만 attention(softmax, missing 제외) -> 융합벡터
   + stain availability encoding(mask 5 + n) 부착  ([[missing_modality]] 원칙 이식)
 - 3 멀티태스크 헤드:
     immune (BCE)        : AIN proxy
     ati_severity (MSE)  : kdigo proxy 1..3 (정규화)
     chronic (BCE)       : 섬유화 burden proxy

설명가능성(#10): 패치 attention + stain contribution(=fusion attention) 반환.

SILVER 역할(설계 확정): SILVER는 '독립 prediction stain'이 아니라 구조 일관성 제약이다.
StainAwareMIL.silver_mode 로 분기:
  "prediction"      : (레거시) SILVER도 예측 융합에 포함 — exp1~6 재현용 기본값.
  "off"             : SILVER 완전 제외(예측·일관성 모두 미사용).
  "consistency"     : SILVER를 예측 융합에서 제외하고, 메인 융합표현을 SILVER 구조벡터에
                      정렬시키는 일관성 제약(L_silver_consistency) 신호로만 사용.
  "consistency_attn": consistency + 메인 stain 패치 attention 엔트로피 안정화 정규화.
SILVER 존재 여부(modality availability)는 어떤 모드에서도 feature 로 유지된다(0-fill 금지).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STAINS = ["HE", "PAS", "MT", "SILVER", "IF"]
SILVER = "SILVER"
SILVER_MODES = ("prediction", "off", "consistency", "consistency_attn")
TASKS_ALL = ["immune", "chronic", "stage3", "ati_severity"]


class GatedAttentionMIL(nn.Module):
    """Simple Linear Attention Pooling (Replaced ABMIL due to variance collapse)."""

    def __init__(self, in_dim=2048, dim=256, att=128, dropout=0.25):
        super().__init__()
        # Project patch features to 'dim' space
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(dim)
        # Simple attention linear layer
        self.attn = nn.Linear(dim, 1)
        self.last_stats = {}

    def forward(self, patches):                     # patches: (n, in_dim)
        h = self.proj(patches)                      # (n, dim)
        h_ln = self.ln(h)
        logit = self.attn(h_ln)                     # (n, 1)
        
        self.last_stats = {
            "h_mean": float(h_ln.mean()), "h_std": float(h_ln.std(unbiased=False)),
            "v_mean": 0.0, "v_std": 0.0,  # removed V
            "u_mean": 0.0, "u_std": 0.0,  # removed U
            "logit_mean": float(logit.mean()), "logit_std": float(logit.std(unbiased=False))
        }
        
        a = torch.softmax(logit, dim=0)                 # 패치 attention
        z = (a * h_ln).sum(0)                          # (dim,)
        return z, a.squeeze(-1)


class _GatedFusion(nn.Module):
    """존재하는 항목 벡터들에 대해서만 simple linear attention softmax 융합."""

    def __init__(self, dim, att):
        super().__init__()
        # Simple attention linear layer
        self.attn = nn.Linear(dim, 1)

    def forward(self, vecs):                 # vecs: (k, dim), k=존재 항목 수
        logit = self.attn(vecs)
        a = torch.softmax(logit, 0)
        return (a * vecs).sum(0), a.squeeze(-1)


class MultiScaleMIL(nn.Module):
    """
    Exp6 — 다중스케일 late fusion (스펙 6/7).
    입력 bag = {(stain, scale): (n,in_dim)}  scale in {"10","40"}, stain in MAIN(HE/PAS/MT).
    스케일별: stain 인코더 -> stain gated fusion -> scale 벡터.
    스케일 fusion(gated, 존재 스케일만) -> 융합 + scale availability -> 4 head.
    """

    def __init__(self, in_dim=768, dim=256, att=128, dropout=0.25,
                 scales=("10", "40"), stains=("HE", "PAS", "MT")):
        super().__init__()
        self.scales = list(scales); self.stains = list(stains)
        self.enc = nn.ModuleDict(
            {f"{sc}_{st}": GatedAttentionMIL(in_dim, dim, att, dropout)
             for sc in self.scales for st in self.stains})
        self.stain_fuse = nn.ModuleDict({sc: _GatedFusion(dim, att) for sc in self.scales})
        self.scale_fuse = _GatedFusion(dim, att)
        fused_dim = dim + len(self.scales)        # 융합 + scale availability mask
        self.head_immune = nn.Linear(fused_dim, 1)
        self.head_ati = nn.Linear(fused_dim, 1)
        self.head_stage3 = nn.Linear(fused_dim, 1)
        self.head_chronic = nn.Linear(fused_dim, 1)

    def forward(self, bag):
        scale_vecs, present, scale_contrib = [], [], {}
        for sc in self.scales:
            svecs = []
            for st in self.stains:
                key = (st, sc)
                if key in bag and bag[key] is not None and bag[key].shape[0] > 0:
                    z, _ = self.enc[f"{sc}_{st}"](bag[key]); svecs.append(z)
            if svecs:
                v, _ = self.stain_fuse[sc](torch.stack(svecs, 0))
                scale_vecs.append(v); present.append(sc)
        if not scale_vecs:
            raise ValueError("bag에 (stain,scale) 없음")
        SV = torch.stack(scale_vecs, 0)
        fused, fa = self.scale_fuse(SV) if SV.shape[0] > 1 else (SV[0], torch.ones(1, device=SV.device))
        mask = torch.tensor([1.0 if s in present else 0.0 for s in self.scales], device=fused.device)
        rep = torch.cat([fused, mask])
        if SV.shape[0] > 1:
            scale_contrib = {s: float(fa[i]) for i, s in enumerate(present)}
        else:
            scale_contrib = {present[0]: 1.0}
        return {"immune": self.head_immune(rep).squeeze(-1),
                "ati_severity": self.head_ati(rep).squeeze(-1),
                "stage3": self.head_stage3(rep).squeeze(-1),
                "chronic": self.head_chronic(rep).squeeze(-1),
                "scale_contrib": scale_contrib}


class _TaskAttention(nn.Module):
    """task별 gated attention pooling (Ilse 2018 ABMIL).

    배포 체크포인트(models/cdss_shadow/ordinal_ms.pt, 5-seed ensemble)가 gated(V/U/w)로
    학습되어 있어 그 구조와 일치시킨다(8001 재시작 후 state_dict 로딩 호환).
    A = w·(tanh(V·H) ⊙ sigmoid(U·H)) → softmax → 가중합.
    """

    def __init__(self, dim, att):
        super().__init__()
        self.V = nn.Linear(dim, att)
        self.U = nn.Linear(dim, att)
        self.w = nn.Linear(att, 1)

    def forward(self, H, temperature=1.0):      # H: (N, dim)
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))   # (N,1)
        a = torch.softmax(A / temperature, 0)       # temperature>1 → attention 분산↑
        return (a * H).sum(0), a.squeeze(-1)


class TaskAttentionMIL(nn.Module):
    """
    개선2 — Task-Specific Attention (설명가능성 핵심).

    구조: stain별 Shared Encoder(proj) -> 패치 feature 풀 H -> task별 독립 attention -> task head.
    Shared attention(StainAwareMIL) 과 달리 immune/chronic/stage3/ati_severity 가 '각자' 패치를
    본다 → task별 attention map·stain contribution 을 분리 해석 가능.

    SILVER: prediction 풀에서 제외(silver_mode=off/consistency/consistency_attn). consistency 계열은
    공유 feature(H 평균)를 SILVER 구조벡터에 정렬하는 L_silver_consistency 로만 사용. 존재여부는 feature.
    """

    def __init__(self, in_dim=768, dim=256, att=128, dropout=0.25, silver_mode="off",
                 tasks=None, out_dims=None):
        super().__init__()
        assert silver_mode in SILVER_MODES, f"silver_mode∈{SILVER_MODES}"
        self.silver_mode = silver_mode
        self.tasks = list(tasks) if tasks is not None else list(TASKS_ALL)  # 임의 task 집합 지원
        self.out_dims = dict(out_dims or {})       # task별 출력 차원(ordinal CORN=K-1, 기본 1)
        # Shared per-stain 인코더(패치 feature 추출, pooling 없음)
        self.encoders = nn.ModuleDict(
            {s: nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
             for s in STAINS})
        self.task_attn = nn.ModuleDict({t: _TaskAttention(dim, att) for t in self.tasks})
        self.silver_proj = nn.Linear(dim, dim)
        fused_dim = dim + len(STAINS) + 1           # task 슬라이드벡터 + availability(mask5+n)
        self.heads = nn.ModuleDict(
            {t: nn.Linear(fused_dim, self.out_dims.get(t, 1)) for t in self.tasks})

    def forward(self, bag, temperature=1.0):
        use_silver = self.silver_mode == "prediction"
        feats, stain_ids, present_all, silver_feat = [], [], [], None
        for s in STAINS:
            if s in bag and bag[s] is not None and bag[s].shape[0] > 0:
                present_all.append(s)
                h = self.encoders[s](bag[s])        # (n_s, dim)
                if s == SILVER and not use_silver:
                    if self.silver_mode != "off":
                        silver_feat = h.mean(0)     # 구조 앵커(mean pool)
                    continue
                feats.append(h); stain_ids += [s] * h.shape[0]
        if not feats:                               # SILVER 단독 엣지 → fallback
            if SILVER in bag and bag[SILVER] is not None and bag[SILVER].shape[0] > 0:
                h = self.encoders[SILVER](bag[SILVER])
                feats.append(h); stain_ids += [SILVER] * h.shape[0]; silver_feat = None
            else:
                raise ValueError("bag에 stain 없음")
        H = torch.cat(feats, 0)                      # (N, dim) 공유 패치 feature
        mask = torch.tensor([1.0 if s in present_all else 0.0 for s in STAINS], device=H.device)
        avail = torch.cat([mask, mask.sum().unsqueeze(0)])

        out = {}
        task_ent = {}                               # task별 정규화 attention entropy(항상 산출 — reg/진단용)
        per_attn, per_contrib, per_contrib_norm = {}, {}, {}
        for t in self.tasks:
            z, a = self.task_attn[t](H, temperature)  # task별 attention(+temperature)
            o = self.heads[t](torch.cat([z, avail]))
            out[t] = o.squeeze(-1) if o.shape[-1] == 1 else o   # dim1=스칼라, dim>1=ordinal 벡터(K-1)
            n = a.shape[0]
            if n > 1:
                task_ent[t] = -(a * (a + 1e-8).log()).sum() \
                    / torch.log(torch.tensor(float(n), device=a.device))
            if not self.training:                   # 설명가능성 산출(학습 중엔 생략 → 경량)
                ad = a.detach()
                per_attn[t] = ad
                contrib, contrib_n = {}, {}
                for s in present_all:
                    if s == SILVER and not use_silver:
                        continue
                    sidx = [i for i, x in enumerate(stain_ids) if x == s]
                    msum = float(ad[sidx].sum())
                    contrib[s] = msum                      # raw: attention mass 합(패치수 confound)
                    contrib_n[s] = msum / len(sidx)        # normalized: 패치당 평균 attention
                per_contrib[t] = contrib
                per_contrib_norm[t] = contrib_n
        out["task_attn"] = per_attn                 # task별 패치 attention(eval)
        out["task_stain_contrib"] = per_contrib     # task별 stain 기여(raw, eval)
        out["task_stain_contrib_norm"] = per_contrib_norm  # task별 stain 기여(패치수 정규화, eval)
        out["stain_ids"] = stain_ids                # 좌표/마스킹 역추적용
        out["task_entropy"] = task_ent              # task별 정규화 attention entropy(0~1)
        if self.silver_mode in ("consistency", "consistency_attn") and silver_feat is not None:
            out["silver_align"] = self.silver_proj(H.mean(0))
            out["silver_target"] = silver_feat.detach()
        else:
            out["silver_align"] = None; out["silver_target"] = None
        out["main_attn_entropy"] = (torch.stack(list(task_ent.values())).mean()
                                    if (self.silver_mode == "consistency_attn" and task_ent) else None)
        return out


class StainAwareMIL(nn.Module):
    def __init__(self, in_dim=2048, dim=256, att=128, fusion_att=128, dropout=0.25,
                 silver_mode="prediction"):
        super().__init__()
        assert silver_mode in SILVER_MODES, f"silver_mode∈{SILVER_MODES}"
        self.silver_mode = silver_mode
        # stain별 독립 인코더
        self.encoders = nn.ModuleDict(
            {s: GatedAttentionMIL(in_dim, dim, att, dropout) for s in STAINS})
        # 각 Stain별 고유 Embedding (Fusion 시 식별력 강화)
        self.stain_emb = nn.ParameterDict(
            {s: nn.Parameter(torch.randn(dim) * 0.1) for s in STAINS})
        # stain간 fusion attention (gated)
        self.fV = nn.Linear(dim, fusion_att)
        self.fU = nn.Linear(dim, fusion_att)
        self.fw = nn.Linear(fusion_att, 1)
        fused_dim = dim + len(STAINS) + 1           # 융합 + availability(mask5 + n)
        self.head_immune = nn.Linear(fused_dim, 1)
        self.head_ati = nn.Linear(fused_dim, 1)       # ati_severity (회귀)
        self.head_stage3 = nn.Linear(fused_dim, 1)    # Stage3 vs non (보조 이진)
        self.head_chronic = nn.Linear(fused_dim, 1)
        # 메인 융합표현 -> SILVER 구조 공간 정렬 프로젝션(consistency 모드에서만 사용)
        self.silver_proj = nn.Linear(dim, dim)

    def forward(self, bag):
        """
        bag: dict{stain: FloatTensor(n_patches, in_dim)} — 존재하는 stain만 포함.
        반환: dict(logits + 설명용 attention + SILVER 일관성 신호)
        """
        use_silver_in_pred = self.silver_mode == "prediction"
        stain_vecs, present, patch_attn = [], [], {}
        silver_vec = None
        for s in STAINS:
            if s in bag and bag[s] is not None and bag[s].shape[0] > 0:
                z, a = self.encoders[s](bag[s])
                patch_attn[s] = a                       # 설명가능성: SILVER 포함 전 stain attention
                if s == SILVER and not use_silver_in_pred:
                    if self.silver_mode != "off":
                        silver_vec = z                  # 일관성 제약의 구조 앵커로만 보관
                    continue                            # 예측 융합에서는 제외
                stain_vecs.append(z)
                present.append(s)

        # ====== Modality Dropout (Phase 2: Gradient Starvation 해결) ======
        # 각 Stain(H&E, PAS, MT, Silver)에 대해 독립적으로 15% 확률로 Drop (최소 1개 유지)
        if self.training and len(stain_vecs) > 1:
            device = stain_vecs[0].device
            keep_mask = torch.rand(len(stain_vecs), device=device) > 0.15
            
            # 최소 1개의 모달리티는 반드시 남도록 보장
            if not keep_mask.any():
                keep_mask[torch.randint(0, len(stain_vecs), (1,), device=device)] = True
                
            filtered_vecs, filtered_present = [], []
            for i in range(len(stain_vecs)):
                if keep_mask[i]:
                    filtered_vecs.append(stain_vecs[i])
                    filtered_present.append(present[i])
            stain_vecs = filtered_vecs
            present = filtered_present
        # ====================================================================

        if not stain_vecs:
            # 메인 stain 이 전무한 엣지(예: SILVER 단독 보유) — 크래시 대신 SILVER로 예측
            if SILVER in patch_attn:
                z, _ = self.encoders[SILVER](bag[SILVER])
                stain_vecs.append(z); present.append(SILVER); silver_vec = None
            else:
                raise ValueError("bag에 stain 없음")
        
        # Stain Embedding 더하기
        stain_vecs_emb = [z + self.stain_emb[s] for z, s in zip(stain_vecs, present)]
        Z = torch.stack(stain_vecs_emb, 0)          # (n_present, dim)
        
        # missing-aware fusion attention (존재 stain만 softmax)
        fa = self.fw(torch.tanh(self.fV(Z)) * torch.sigmoid(self.fU(Z)))  # (n_present,1)
        # Temperature Softmax (Collapse 방지)
        fusion_temperature = 2.0
        fa = torch.softmax(fa / fusion_temperature, dim=0)
        fused = (fa * Z).sum(0)                     # (dim,)
        # availability encoding (0-fill 아님: 실제 stain 존재 여부를 명시 feature로 — SILVER 포함)
        present_all = [s for s in STAINS
                       if s in bag and bag[s] is not None and bag[s].shape[0] > 0]
        mask = torch.tensor([1.0 if s in present_all else 0.0 for s in STAINS],
                            device=fused.device)
        avail = torch.cat([mask, mask.sum().unsqueeze(0)])
        rep = torch.cat([fused, avail])
        stain_contrib = {s: float(fa[i].detach()) for i, s in enumerate(present)}
        out = {
            "immune": self.head_immune(rep).squeeze(-1),
            "ati_severity": self.head_ati(rep).squeeze(-1),
            "stage3": self.head_stage3(rep).squeeze(-1),
            "chronic": self.head_chronic(rep).squeeze(-1),
            "patch_attn": patch_attn,          # 설명가능성 #10
            "stain_contrib": stain_contrib,    # 설명가능성 #10
            "silver_align": None,              # consistency: 메인 융합의 구조 정렬 벡터
            "silver_target": None,             # consistency: SILVER 구조 앵커(정지구배)
            "main_attn_entropy": None,         # consistency_attn: 메인 attention 정규화 엔트로피
        }
        # SILVER 일관성 제약: 메인 융합표현을 SILVER 구조 공간에 정렬(앵커는 detach → 단방향 정규화)
        if self.silver_mode in ("consistency", "consistency_attn") and silver_vec is not None:
            out["silver_align"] = self.silver_proj(fused)
            out["silver_target"] = silver_vec.detach()
        # 메인 stain 패치 attention 안정화: 정규화 엔트로피(0~1, 1=완전 분산) 평균
        if self.silver_mode == "consistency_attn":
            ents = []
            for s in present:
                a = patch_attn[s]
                n = a.shape[0]
                if n > 1:
                    ent = -(a * (a + 1e-8).log()).sum() / torch.log(
                        torch.tensor(float(n), device=a.device))
                    ents.append(ent)
            if ents:
                out["main_attn_entropy"] = torch.stack(ents).mean()
        return out
