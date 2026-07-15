"""단일 stain(PAS) 배포 모델 빌드 — 전체 코호트 5-seed 앙상블, ordinal descriptor.

멀티스테인 cdss_engine.fit() 과 동일 저장 포맷을 쓰되:
  - keep={PAS} (단일 stain — CTransPath 멀티 융합 제거. 검증서 descriptor MAE 멀티대비 개선 확인)
  - layernorm=False (단일 stain 은 모달리티 붕괴가 없어 LN 불필요)
  - ckpt["stains"]=["PAS"] 표식 → 엔진이 REQUIRED_STAINS 를 {PAS} 로 완화(HE 없어도 ALLOW)

라이브(cdss_engine.py)는 건드리지 않는다. 산출물은 별도 파일(ordinal_pas.pt) → 배선 후 env 로 opt-in.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]          # Pathology_model
sys.path.insert(0, str(_PKG))
import torch
import torch.nn.functional as F
from mil.train import SEED, load_bags, seed_all
from mil.cdss_paths import p as _p
from mil.cdss_engine import ORD, KBINS, BANFF, SEEDS, _new_model, to_ord

_ART = _PKG / "artifacts"
OUT_PATH = _PKG / "models" / "cdss_shadow" / "ordinal_pas.pt"
KEEP = {"PAS"}


def build():
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = pd.read_csv(_p("embeddings") / "ctranspath" / "index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"].isin(["10", "40"]) & idx["stain"].isin(KEEP)]
    embed_dim = int(idx["embed_dim"].iloc[0])

    # ORD descriptor 라벨(Banff 4-bin) — 경로는 코드 상대(artifacts), 마이그레이션 무관.
    desc = pd.read_csv(_ART / "descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    src = {"fibrosis": "interstitial_fibrosis_pct", "atrophy": "tubular_atrophy",
           "inflammation": "interstitial_mononuclear_wbc_pct"}
    lab = pd.DataFrame({"patient_id": desc["patient_id"]})
    for t, c in src.items():
        v = pd.to_numeric(desc[c], errors="coerce"); v = v.where(v < 999)
        lab[t] = v.map(lambda x: to_ord(x, BANFF[t]))

    bags = load_bags(lab["patient_id"].tolist(), idx, keep=KEEP)
    lab = lab[lab["patient_id"].isin(bags) & lab[ORD].notna().any(axis=1)].reset_index(drop=True)
    print(f"cohort={len(lab)} embed_dim={embed_dim} keep={sorted(KEEP)}", flush=True)

    def train_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        model = _new_model(embed_dim).to(device)      # layernorm=False
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(40):
            model.train()
            for r in lab.sample(frac=1, random_state=seed + ep).itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                out = model(bag); loss = 0.0; nt = 0
                for t in ORD:
                    yv = rd[t]
                    if yv == yv:
                        lv = torch.tensor([1.0 if yv > k else 0.0 for k in range(KBINS - 1)], device=device)
                        loss = loss + F.binary_cross_entropy_with_logits(out[t], lv); nt += 1
                if nt:
                    opt.zero_grad(); loss.backward(); opt.step()
        return {k: v.cpu() for k, v in model.state_dict().items()}

    ensemble = [train_seed(s) for s in SEEDS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ensemble": ensemble, "seeds": SEEDS, "embed_dim": embed_dim,
                "tasks": ORD, "kbins": KBINS, "banff": BANFF, "layernorm": False,
                "stains": ["PAS"],
                "trained": datetime.now().isoformat(timespec="seconds"),
                "note": "SHADOW; PAS single-stain(멀티 융합 제거); 5-seed CORAL. "
                        "검증(train_descriptor_ms_ordinal --stains PAS, 5-seed): descriptor MAE 멀티대비 개선"},
               OUT_PATH)
    print(f"저장 -> {OUT_PATH} (ensemble={len(ensemble)}, cohort={len(lab)})", flush=True)


if __name__ == "__main__":
    build()
