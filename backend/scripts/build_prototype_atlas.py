"""② Prototype Atlas Builder — KPMP 임베딩 → 프로토타입 → DB 적재.

파이프라인(Frozen):
  CTransPath 임베딩(D:/cdss_core) → slide-level 표현(환자별 mean-pool)
   → batch 보정(z-score) → HDBSCAN 군집(noise=rare 보존)
   → 대표 WSI(medoid) → Prototype Metadata(멤버 ConceptVector 평균) → DB 적재.

검색 대상(metadata_vec)은 임상 concept 평균이고, 군집은 WSI 임베딩(병리 유사도)으로 한다
→ "병리 패턴이 비슷한 프로토타입"을 만들되 검색은 임상 concept으로 설명 가능.

사용: (team venv) PYTHONPATH=. python scripts/build_prototype_atlas.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from core.config import settings
from core.database import SessionLocal
from retrieval.concept.builder import load_kpmp_concepts
from retrieval.concept.schema import ConceptVector
from retrieval.prototype.metadata import build_prototype_metadata
import models.retrieval as R

ART = Path(__file__).resolve().parent.parent.parent / "Pathology_model" / "artifacts"


def cluster_embeddings(X: np.ndarray, *, pca: int, min_cluster_size: int,
                       min_samples: int) -> np.ndarray:
    """L2 정규화(→cosine 등가) + PCA 차원축소(고차원 HDBSCAN 쏠림 완화) → HDBSCAN labels."""
    Xn = normalize(X)                                   # L2 → euclidean≈cosine
    if pca > 0 and pca < min(X.shape):
        Xn = PCA(n_components=pca, whiten=True, random_state=42).fit_transform(Xn)
        Xn = normalize(Xn)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     metric="euclidean").fit_predict(Xn)
    return labels


def _entropy_norm(sizes: list[int]) -> float:
    p = np.array(sizes, float) / sum(sizes)
    ent = float(-(p * np.log(p)).sum())
    return round(ent / np.log(len(sizes)), 4) if len(sizes) > 1 else 0.0


def _emb_root() -> Path:
    return settings.cdss_core_path


def load_slide_embeddings(pids: set[str]) -> dict[str, np.ndarray]:
    """환자별 slide-level 임베딩(모든 stain/배율 패치 mean-pool). {pid: (768,)}."""
    root = _emb_root()
    idx = pd.read_csv(root / "data_processed/embeddings/ctranspath/index.csv",
                      dtype={"magnification": str})
    idx["patient_id"] = idx["patient_id"].astype(str)
    out: dict[str, np.ndarray] = {}
    for pid, sub in idx[idx["patient_id"].isin(pids)].groupby("patient_id"):
        mats = []
        for _, r in sub.iterrows():
            p = root / r["npy_path"]
            if p.exists():
                a = np.load(p)
                if a.shape[0] > 0:
                    mats.append(a.astype(np.float32))
        if mats:
            out[str(pid)] = np.vstack(mats).mean(axis=0)   # patch mean-pool
    return out


def _label(members: list[ConceptVector]) -> str:
    etio = max(set(m.etiology_hint for m in members),
               key=lambda e: sum(x.etiology_hint == e for x in members))
    band = max(set(m.kdigo_band() for m in members),
               key=lambda b: sum(x.kdigo_band() == b for x in members))
    return f"{etio}/KDIGO {band}"


def main() -> None:
    ap = argparse.ArgumentParser()
    # 튜닝 선정값(쏠림 제거: max군집 88%→13%, entropy 0.956). L2정규화+PCA가 핵심.
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--pca", type=int, default=15, help="0=PCA 미사용")
    ap.add_argument("--dry-run", action="store_true", help="DB 미적재, 클러스터 분포만 출력")
    args = ap.parse_args()

    concepts = load_kpmp_concepts(ART)
    emb = load_slide_embeddings(set(concepts))
    pids = [p for p in concepts if p in emb]
    print(f"[atlas] cohort with embedding+concept: {len(pids)}")
    if len(pids) < args.min_cluster_size:
        print("[atlas] 코호트 부족 — 중단")
        return

    X = np.stack([emb[p] for p in pids])               # (n, 768)
    Xn = normalize(X)                                  # medoid 계산용(정규화 공간)
    labels = cluster_embeddings(X, pca=args.pca, min_cluster_size=args.min_cluster_size,
                                min_samples=args.min_samples)
    uniq = sorted(set(labels))
    n_clusters = len([u for u in uniq if u >= 0])
    n_noise = int((labels == -1).sum())
    sizes = sorted(Counter(int(c) for c in labels if c >= 0).values(), reverse=True)
    print(f"[atlas] HDBSCAN(min_cluster={args.min_cluster_size},pca={args.pca}): "
          f"clusters={n_clusters} noise(rare)={n_noise}")
    print(f"        cluster sizes(top10)={sizes[:10]} | size-entropy={_entropy_norm(sizes) if sizes else 0}")
    if args.dry_run:
        print("[atlas] dry-run — DB 미적재. (분포 확인용)")
        return

    # noise(-1)는 rare 프로토타입으로 각자 보존
    db = SessionLocal()
    try:
        db.query(R.PrototypeMember).delete()
        db.query(R.WsiMetadata).delete()
        db.query(R.Prototype).delete()
        db.commit()

        cluster_groups: dict[int, list[int]] = {}
        next_singleton = max([u for u in uniq if u >= 0], default=-1) + 1
        for i, lab in enumerate(labels):
            key = lab if lab >= 0 else next_singleton + i  # noise → 고유 singleton
            cluster_groups.setdefault(key, []).append(i)

        n_proto = 0
        for cid, idxs in cluster_groups.items():
            members = [concepts[pids[i]] for i in idxs]
            meta = build_prototype_metadata(members)
            # medoid: 군집 centroid 에 가장 가까운 환자 = 대표 WSI (정규화 공간)
            sub = Xn[idxs]
            centroid = sub.mean(0)
            medoid_local = int(np.argmin(((sub - centroid) ** 2).sum(1)))
            medoid_pid = pids[idxs[medoid_local]]
            is_rare = len(idxs) < args.min_cluster_size

            proto = R.Prototype(
                label=_label(members), cluster_id=int(cid), n_members=len(idxs),
                metadata_vec=[float(x) for x in meta.metadata_vec],
                kdigo_band=meta.kdigo_band, etiology_hint=meta.etiology_hint,
                egfr_mean=meta.egfr_mean, chronicity=None,
                is_rare=is_rare, batch_adjusted=True, validated_by=None,
            )
            db.add(proto)
            db.flush()  # proto.id 확보
            for i in idxs:
                db.add(R.PrototypeMember(
                    prototype_id=proto.id, patient_id=pids[i],
                    concept_vec=[float(x) for x in concepts[pids[i]].to_vector()],
                    is_medoid=(pids[i] == medoid_pid)))
            db.add(R.WsiMetadata(
                slide_id=medoid_pid, prototype_id=proto.id, is_representative=True,
                stain="HE", source="KPMP"))
            n_proto += 1
        db.commit()
        print(f"[atlas] 적재 완료: prototypes={n_proto}, members={len(pids)}")

        # 요약
        rows = db.query(R.Prototype).all()
        for p in sorted(rows, key=lambda x: -x.n_members)[:12]:
            print(f"  {p.id} | {p.label:18s} n={p.n_members} rare={p.is_rare} egfr~{p.egfr_mean}")
    finally:
        db.close()
    print("[atlas] done.")


if __name__ == "__main__":
    main()
