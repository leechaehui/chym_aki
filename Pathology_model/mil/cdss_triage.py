"""
Shadow CDSS — Triage / ABSTAIN-우선 비진단 보조 추론 계층 (최종 사양)

⚠️ 운영 제한: 본 계층이 감싸는 모델은 임상 미검증(AUROC 95%CI가 0.5 포함, R²≈0, N≈60,
외부검증 없음)이다. 따라서 본 시스템은 SHADOW(비가시 보조분석) + ABSTAIN-우선으로만 운용한다.
- 진단/치료/확정 판정 금지. 임상 의사결정 단독 사용 금지. 검사 우선순위 자동결정 금지.
- 'high confidence' 개념 사용 안 함. 출력은 '불확실성 포함 참고 신호'일 뿐.
- 원칙: "출력 정확도보다 안전한 침묵(ABSTAIN)을 우선한다."

흐름: QC 게이트 → (모델) multi-scale + stain routing → MIL → MC-dropout 불확실성 →
ABSTAIN 규칙(강화) → 고정 JSON. 모든 결과 audit 로깅. 최종판단은 병리의(HITL).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# 비진단 표현 강제: '진단적 단정' 표현만 차단(안전문구 '비진단/진단이 아닌'은 허용 → 정밀 매칭)
BANNED = ["확진", "거부반응", "rejection", "diagnosis", "진단 결과", "진단됨",
          "진단합니다", "확정 진단", "처방", "치료 결정"]
ABSTAIN_MSG = "신뢰도가 낮아 분석을 보류합니다. 병리의 검토가 필요합니다."
AUDIT_LOG = (Path(__file__).resolve().parents[2] / "Pathology_model/results/09_cdss_shadow/audit_log.jsonl")

# task head → 비진단 risk 신호 매핑
SIGNAL_HEAD = {"inflammation": "wbc_pct", "fibrosis": "fibrosis_pct",
               "atrophy": "atrophy", "tubulitis": "tubulitis"}
SIGNAL_PHRASE = {
    "inflammation": "염증 패턴 가능성 신호",
    "fibrosis": "만성 구조 변화 가능성 신호",
    "atrophy": "세뇨관 구조 변화 가능성 신호",
    "tubulitis": "활동성 염증 패턴 가능성 신호",
}
REQUIRED_STAINS = {"HE", "PAS"}        # 최소 필수 stain (누락 시 ABSTAIN)


class ShadowTriage:
    def __init__(self, model, device="cpu", *, mc_passes=20,
                 min_patches=30, unc_abstain=0.12, conflict_band=0.10,
                 emb_norm_range=(0.1, 100.0)):
        """
        model: descriptor 5-head TaskAttentionMIL (학습된 가중치 — 미검증이면 shadow 전용).
        unc_abstain: MC-dropout 신호 표준편차 평균이 이 값 초과면 ABSTAIN(불확실성↑).
        conflict_band: 신호 평균이 0.5±band 이고 분산 크면 '모델 충돌'로 ABSTAIN.
        """
        import torch  # noqa
        self.model = model.to(device).eval()
        self.device = device
        self.mc = mc_passes
        self.min_patches = min_patches
        self.unc_abstain = unc_abstain
        self.conflict_band = conflict_band
        self.emb_lo, self.emb_hi = emb_norm_range

    # ---------- QC 게이트 ----------
    def _qc(self, bag_np):
        present = {s for s, v in bag_np.items() if v is not None and getattr(v, "shape", [0])[0] > 0}
        if not REQUIRED_STAINS.issubset(present):
            return f"필수 stain 누락(HE/PAS): 보유={sorted(present)}"
        total = sum(v.shape[0] for v in bag_np.values() if v is not None)
        if total < self.min_patches:
            return f"조직/패치 부족(patch={total} < {self.min_patches})"
        all_v = np.concatenate([v for v in bag_np.values() if v is not None and v.shape[0] > 0], 0)
        if not np.isfinite(all_v).all():
            return "임베딩 비정상(NaN/Inf) — 초점불량/아티팩트 의심"
        norm = float(np.linalg.norm(all_v, axis=1).mean())
        if not (self.emb_lo <= norm <= self.emb_hi):
            return f"임베딩 신뢰도 낮음/OOD(평균 norm {norm:.2f} 범위 밖)"
        return None

    # ---------- MC-dropout 추론 ----------
    def _mc_predict(self, bag_np):
        import torch
        # dropout만 train 모드로(나머지 eval) → epistemic 근사
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
        bag = {s: torch.from_numpy(v).to(self.device) for s, v in bag_np.items()
               if v is not None and v.shape[0] > 0}
        samples = {sig: [] for sig in SIGNAL_HEAD}
        top_stain = {}
        with torch.no_grad():
            for t in range(self.mc):
                out = self.model(bag)
                for sig, head in SIGNAL_HEAD.items():
                    if head in out:
                        samples[sig].append(float(torch.sigmoid(out[head])))
                if t == 0:  # 근거영역(첫 패스 attention 기여)
                    for sig, head in SIGNAL_HEAD.items():
                        c = out.get("task_stain_contrib_norm", {}).get(head, {})
                        if c:
                            top_stain[sig] = max(c, key=c.get)
        mean = {sig: float(np.mean(v)) if v else None for sig, v in samples.items()}
        std = {sig: float(np.std(v)) if v else None for sig, v in samples.items()}
        return mean, std, top_stain

    # ---------- 메인 ----------
    def analyze(self, bag_np, *, slide_id="unknown"):
        qc = self._qc(bag_np)
        if qc:
            return self._abstain(reason=f"QC: {qc}", slide_id=slide_id)

        mean, std, top_stain = self._mc_predict(bag_np)
        valid = [s for s in std.values() if s is not None]
        uncertainty = float(np.mean(valid)) if valid else 1.0
        # 모델 충돌: 신호 평균이 결정경계 근처인데 분산 큼
        conflict = any(m is not None and abs(m - 0.5) < self.conflict_band
                       and std[sig] is not None and std[sig] > self.unc_abstain * 0.7
                       for sig, m in mean.items())
        present = {s for s, v in bag_np.items() if v is not None and v.shape[0] > 0}
        missing_optional = sorted({"MT", "SILVER"} - present)

        # ABSTAIN 규칙(강화): 불확실성↑ 또는 충돌
        if uncertainty > self.unc_abstain:
            return self._abstain(reason=f"불확실성 높음(평균 std {uncertainty:.3f}>{self.unc_abstain})",
                                 slide_id=slide_id, extra={"uncertainty": round(uncertainty, 3)})
        if conflict:
            return self._abstain(reason="MC 표본 간 결과 충돌(결정경계 불안정)", slide_id=slide_id)

        signals = {sig: round(m, 2) for sig, m in mean.items() if m is not None}
        confidence = round(1.0 - uncertainty, 3)   # 'high confidence' 미사용 — 단순 보조 수치
        # 위험도: 보수적(최댓값 기준, abstain-우선이므로 임계 보수적)
        mx = max(signals.values()) if signals else 0.0
        risk = "high" if mx >= 0.66 else ("medium" if mx >= 0.4 else "low")
        active = [SIGNAL_PHRASE[s] for s, v in signals.items() if v >= 0.4]
        att = "; ".join(f"{s}↦{top_stain.get(s, '?')}" for s in signals) or "근거 영역 추정 제한"
        interp = ("참고용(비진단) 신호입니다. 관찰된 패턴: "
                  + (", ".join(active) if active else "유의 신호 미관찰")
                  + ". 병리의 검토가 필요합니다.")
        if missing_optional:
            interp += f" (참고: {','.join(missing_optional)} 미보유로 신호 제한)"
        out = {"mode": "SHADOW", "risk_level": risk, "signals": signals,
               "confidence": confidence, "uncertainty": round(uncertainty, 3),
               "attention_summary": att, "interpretation": interp}
        self._guard(out)
        self._audit(slide_id, out)
        return out

    def _abstain(self, *, reason, slide_id, extra=None):
        out = {"mode": "SHADOW", "risk_level": "abstain", "signals": {},
               "confidence": None, "uncertainty": (extra or {}).get("uncertainty"),
               "attention_summary": None, "interpretation": ABSTAIN_MSG, "abstain_reason": reason}
        self._guard(out)
        self._audit(slide_id, out)
        return out

    @staticmethod
    def _guard(out):
        blob = json.dumps(out, ensure_ascii=False)
        hit = [w for w in BANNED if w in blob]
        if hit:
            raise ValueError(f"비진단 표현 위반(금지어 {hit}) — 출력 차단")

    @staticmethod
    def _audit(slide_id, out):
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "slide_id": slide_id, **out}
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 데모/스모크: 합성 bag으로 계약·ABSTAIN 동작 검증 (미학습 모델 = shadow 로직 검증용)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Pathology_model"))
    import torch
    from mil.model import TaskAttentionMIL
    TASKS = ["immune", "tubulitis", "wbc_pct", "fibrosis_pct", "atrophy"]
    model = TaskAttentionMIL(in_dim=768, silver_mode="prediction", tasks=TASKS)
    tri = ShadowTriage(model, mc_passes=20)
    D = 768
    print("1) PAS 누락 →", tri.analyze({"HE": np.random.randn(40, D).astype("f4")},
                                       slide_id="case_noPAS")["risk_level"])
    print("2) 패치 부족 →", tri.analyze({"HE": np.random.randn(5, D).astype("f4"),
                                        "PAS": np.random.randn(5, D).astype("f4")},
                                       slide_id="case_fewpatch")["risk_level"])
    full = {s: np.random.randn(60, D).astype("f4") for s in ["HE", "PAS", "MT", "SILVER"]}
    r = tri.analyze(full, slide_id="case_full")
    print("3) 정상 bag →", json.dumps(r, ensure_ascii=False, indent=2))
