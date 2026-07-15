"""AI Draft 파이프라인 실행 산출물 생성기 (작업지시서 11 OUTPUT REQUIREMENTS).

샘플 환자에 대해 전체 파이프라인을 실행하고 다음을 docs/vv/ai_draft_pipeline_report.md 로 출력:
  - full pipeline execution result
  - SOAP with evidence mapping
  - problem list mapping
  - risk score breakdown
  - timeline event record
  - validation report

실행:
  cd backend && set PYTHONPATH=. && .venv\\Scripts\\python.exe tests\\vv_runners\\run_ai_draft_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent        # <repo>/tests/vv_runners
BACKEND = HERE.parent.parent / "backend"       # <repo>/backend
sys.path.insert(0, str(BACKEND))

_tmp = Path(tempfile.gettempdir()) / "chym_ai_draft_demo.db"
if _tmp.exists():
    _tmp.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp.as_posix()}"
os.environ["SEED_ON_STARTUP"] = "false"

from core.database import SessionLocal, init_db  # noqa: E402
from models.base import utcnow  # noqa: E402
from models.patient import Patient, PatientLab, PatientTrendPoint, PatientUrinePoint  # noqa: E402
from models.timeline import TimelineEvent  # noqa: E402
from services.ai_draft_service import AiDraftService  # noqa: E402

OUT = BACKEND / "docs" / "vv" / "ai_draft_pipeline_report.md"
TRANSCRIPT = "어제부터 다리가 붓고 소변이 줄었어요. 메스껍고 기운이 없어요."


def _seed_patient(db) -> Patient:
    pid = "demo-" + uuid.uuid4().hex[:6]
    db.add(Patient(id=pid, mrn="DEMO-" + pid.upper(), name="홍데모", sex="M", age=67,
                   diagnosis="당뇨병성 신증", admitted_at=utcnow().isoformat(),
                   attending="신장내과", room="W-12", ai_risk_score=0))
    db.add_all([
        PatientTrendPoint(patient_id=pid, date="2026-06-01", creatinine=1.0, egfr=82, bun=20),
        PatientTrendPoint(patient_id=pid, date="2026-06-03", creatinine=4.2, egfr=14, bun=60),
        PatientUrinePoint(patient_id=pid, date="2026-06-03", value=0.2),
        PatientLab(patient_id=pid, seq=0, key="cr", label="Creatinine", value=4.2, unit="mg/dL", flag="high"),
        PatientLab(patient_id=pid, seq=1, key="k", label="Potassium", value=6.3, unit="mmol/L", flag="high"),
        PatientLab(patient_id=pid, seq=2, key="egfr", label="eGFR", value=14, unit="mL/min", flag="low"),
    ])
    db.commit()
    return db.get(Patient, pid)


def _md(result: dict, events: list[TimelineEvent]) -> str:
    soap, problems, risk, val = result["soap"], result["problem_list"], result["risk"], result["validation"]
    L = ["# AI Draft 파이프라인 실행 결과 (작업지시서 11)", ""]
    L += ["_Audio → STT → SOAP → Problem List → CDSS Risk → Timeline_", ""]
    L += ["## 1. Transcript (STT)", "", f"> {result['transcript']}", ""]

    L += ["## 2. SOAP (evidence mapping)", ""]
    L += [f"- **S**: {soap['S']}", f"- **O**: {soap['O']}", ""]
    for key, tk in (("A", "assessment"), ("P", "plan")):
        sec = soap[key]
        L += [f"**{key} — {sec[tk]}**", ""]
        if sec["statements"]:
            L += ["| statement | evidence |", "|---|---|"]
            for st in sec["statements"]:
                L.append(f"| {st['statement']} | {'; '.join(st['evidence'])} |")
        L.append("")

    L += ["## 3. Problem List (A 기반)", "", "| problem | source | confidence | evidence |", "|---|---|---:|---|"]
    for p in problems:
        L.append(f"| {p['problem']} | {p['source']} | {p['confidence']} | {'; '.join(p['evidence'])} |")
    L.append("")

    L += ["## 4. CDSS Risk Score breakdown", "",
          f"**risk_score = {risk['risk_score']} → {risk['tier']}** "
          f"(alert: {risk['alert']}, nephrology_trigger: {risk['nephrology_trigger']})", "",
          "| component | value | weight | contribution | explanation |",
          "|---|---:|---:|---:|---|"]
    for c in risk["breakdown"]:
        L.append(f"| {c['component']} | {c['value']} | {c['weight']} | {c['contribution']} | {c['explanation']} |")
    L += ["", f"score = " + " + ".join(f"{c['weight']}×{c['value']}" for c in risk["breakdown"]), ""]

    L += ["## 5. Timeline event record", "", "| event_type | severity | title | payload |", "|---|---|---|---|"]
    for e in events:
        L.append(f"| {e.event_type} | {e.severity} | {e.title} | {e.payload_json or ''} |")
    L.append("")

    L += ["## 6. Validation report", "", f"**passed: {val['passed']}**", ""]
    for v in val["validators"]:
        status = "✅" if v["passed"] else "❌"
        L.append(f"- {status} `{v['validator']}` — errors={len(v['errors'])}, warnings={len(v['warnings'])}")
        for e in v["errors"]:
            L.append(f"  - ⚠️ {e}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        patient = _seed_patient(db)
        svc = AiDraftService(db)
        note = svc.generate_draft(patient_id=patient.id, transcript=TRANSCRIPT, actor_id="demo-user")
        result = svc.to_out(note)
        events = (
            db.query(TimelineEvent)
            .filter(TimelineEvent.patient_id == patient.id)
            .order_by(TimelineEvent.event_time)
            .all()
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(_md(result, events), encoding="utf-8")
        print(f"[OK] {OUT}")
        print(f"     risk={result['risk']['risk_score']} tier={result['risk']['tier']} "
              f"problems={len(result['problem_list'])} validation_passed={result['validation']['passed']}")
        print("     pipeline result JSON:")
        print(json.dumps({k: result[k] for k in ('problem_list', 'risk')}, ensure_ascii=False)[:300])
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
