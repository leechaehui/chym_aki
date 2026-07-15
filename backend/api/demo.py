import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from core.database import SessionLocal
from core.fake_name import fake_korean_name
from models.base import new_id
from queries.constants import STAGE_LABELS
from services.icu_monitor_service import _predictions_cached, _predictions, _model_feature_cols
from services.alert_service import reset_dedup as _reset_alert_dedup
from api.events import _publish_lab

router = APIRouter(prefix="/demo", tags=["demo"])

# RobustScaler 임상 변환 파라미터
_STAGE_CR_MULTIPLIER = {0: 1.0, 1: 1.7, 2: 2.3, 3: 3.3}
_STAGE_BUN_ADD = {0: 0.0, 1: 8.0, 2: 18.0, 3: 30.0}
_STAGE_K_ADD = {0: 0.0, 1: 0.3, 2: 0.7, 3: 1.2}
_STAGE_UO = {0: 1.2, 1: 0.6, 2: 0.35, 3: 0.15}

def _patient_variation(subject_id: int | None) -> float:
    """subject_id 기반 결정적 미세 편차 계수(-1.0~+1.0). 재적재해도 값이 고정돼
    같은 스테이지 환자끼리 수치가 우연히 겹치는 것을 막는 용도."""
    if subject_id is None:
        return 0.0
    bucket = (subject_id * 2654435761) % 1000  # 0..999 결정적 해시
    return bucket / 999.0 * 2.0 - 1.0


def _clinical_values_for_stage(
    stage: int,
    baseline_cr: float = 0.9,
    subject_id: int | None = None,
    severity: float | None = None,
) -> dict[str, float]:
    """스테이지별 목표 임상수치. subject_id/severity가 주어지면 같은 스테이지
    안에서도 환자별로 값이 벌어지도록 결정적 편차를 더한다.

    severity: 같은 스테이지 내 상대 중증도(0.0~1.0, 높을수록 나쁨). 모델 입력
    피처(creatinine_max) 순위에서 뽑아 넘긴다. None이면 편차 방향은 id 노이즈만.
    """
    s = stage if stage in _STAGE_CR_MULTIPLIER else 0

    # 편차 방향(-1~+1): 실제 피처 순위(중증도)가 주 성분, id 노이즈가 충돌 방지 보조.
    sev_component = 0.0 if severity is None else (severity * 2.0 - 1.0)
    id_component = _patient_variation(subject_id)
    # 스테이지 0(정상)은 편차 없이 전원 정상 기준값 유지.
    offset = 0.0 if s == 0 else (0.6 * sev_component + 0.4 * id_component)

    return {
        "cr": round(baseline_cr * _STAGE_CR_MULTIPLIER[s] * (1.0 + 0.12 * offset), 2),
        "bun": round(15.0 + _STAGE_BUN_ADD[s] * (1.0 + 0.15 * offset), 1),
        "k": round(4.2 + _STAGE_K_ADD[s] * (1.0 + 0.20 * offset), 2),
        # 나쁠수록(offset↑) 소변량은 반대로 줄어든다.
        "uo": round(_STAGE_UO[s] * (1.0 - 0.15 * offset), 2),
    }

def _sync_ai_risk(db: Session, subject_ids: list[int], *, update_diagnosis: bool = True) -> dict[int, int]:
    df_pred = _predictions()
    scores: dict[int, int] = {}
    for sid in subject_ids:
        match = df_pred[df_pred["subject_id"] == sid]
        if match.empty:
            continue
        real_score = int(match.iloc[0]["risk_score"])
        scores[sid] = real_score
        if update_diagnosis:
            pred = int(match.iloc[0]["pred"])
            diag = f"AI 예측: {STAGE_LABELS[pred]} (위험도 {real_score}%)"
            db.execute(
                text("UPDATE chym.patients SET ai_risk_score = :score, diagnosis = :diag WHERE mimic_subject_id = :sid"),
                {"sid": sid, "score": real_score, "diag": diag},
            )
        else:
            db.execute(
                text("UPDATE chym.patients SET ai_risk_score = :score WHERE mimic_subject_id = :sid"),
                {"sid": sid, "score": real_score},
            )
    return scores

# dataset 경로
DATASET_CSV = Path(__file__).resolve().parent.parent.parent / "AKI" / "preprocessing" / "final_dataset" / "test_final.csv"
if not DATASET_CSV.exists():
    DATASET_CSV = Path("C:/dev/chym_aki/dataset/test.csv")

PROTAGONIST_SUBJECT_ID = 10218191
DEMO_COHORT_SIZE = {"stage_0": 220, "stage_1": 60, "stage_2_3": 20}

# advance-hour 롤링 배치: 위험군(stage1 60 + stage2_3 20 = 80명)을 한 번에 10명씩
# 악화시킨다. 8번 누르면 전체 소진.
DEMO_BATCH_SIZE = 10

# 추이/소변 그래프 x축 라벨. character varying(10) 제약 + 문자열 정렬로 시간순이
# 유지되도록 모두 'MM-DD HHh'(9자) 형식으로 통일한다.
SETUP_LABEL = "07-03 00h"     # 0h 기준선
TRIGGER_LABEL = "07-05 12h"   # 오준현 트리거(마지막 시점)
_ADVANCE_BATCH_LABELS = {
    1: "07-03 06h", 2: "07-03 12h", 3: "07-03 18h", 4: "07-04 00h",
    5: "07-04 06h", 6: "07-04 12h", 7: "07-04 18h", 8: "07-05 00h",
}

# advance-hour 롤링 진행 상태(= 지금까지 처리한 배치 수). setup/reset 시 0으로 초기화.
_advance_progress = {"time_step": 0}

def _select_demo_subjects() -> dict[str, list[int]]:
    fallback = {
        "stage_0": [
            10015272, 10018328, 10038332, 10032409, 10001725,
            10044916, 10065354, 10067480, 10070311, 10072364,
            10054464, 10057070, 10060142, 10081559, 10083218,
            10100289, 10101901, 10102102, 10103103, 10104104,
            10105105
        ],
        "stage_1": [10032207, 10027730, 10061633, 10096175, 10054716, 10100811],
        "stage_2_3": [PROTAGONIST_SUBJECT_ID, 10243766, 10318966],
    }
    try:
        df = pd.read_csv(DATASET_CSV)
        df = df.drop_duplicates(subset=["subject_id"]).sort_values("subject_id")

        # 데모 코호트 선별 두 단계(둘 다 test_final.csv 의 결측 대치 아티팩트 대응):
        #  (1) 피처 전부-0 환자 제외: 측정값이 없어 35개 모델 피처가 통째로 0/NaN→0 인 환자가
        #      17%(917/5362)나 된다. 이들은 모델 입력이 0이라 SHAP·변수·리포트가 전부 0으로
        #      떠(예: '홍인은' subject 10011365) 데모에 부적합 → 제외.
        #  (2) 피처벡터 중복 제거: 남은 환자 중에서도 피처가 완전히 동일한 환자는 결정론적
        #      모델 특성상 점수·SHAP 가 100% 똑같아 클릭 시 구별이 안 된다 → 고유 벡터만.
        feat_in_df = [c for c in _model_feature_cols() if c in df.columns]
        if feat_in_df:
            has_signal = ~(df[feat_in_df].fillna(0) == 0).all(axis=1)
            df_distinct = df[has_signal].drop_duplicates(subset=feat_in_df)
        else:
            df_distinct = df

        def pick(stages: list[int], n: int) -> list[int]:
            return df_distinct[df_distinct["aki_stage"].isin(stages)]["subject_id"].astype(int).tolist()[:n]

        stage_0 = pick([0], DEMO_COHORT_SIZE["stage_0"])
        stage_1 = pick([1], DEMO_COHORT_SIZE["stage_1"])
        stage_2_3 = pick([2, 3], DEMO_COHORT_SIZE["stage_2_3"])
        
        if PROTAGONIST_SUBJECT_ID not in stage_2_3:
            stage_2_3 = ([PROTAGONIST_SUBJECT_ID] + stage_2_3)[:DEMO_COHORT_SIZE["stage_2_3"]]
        if not stage_0 or not stage_1 or not stage_2_3:
            return fallback
        return {"stage_0": stage_0, "stage_1": stage_1, "stage_2_3": stage_2_3}
    except Exception:
        return fallback

DEMO_PATIENTS_CONFIG = _select_demo_subjects()
ALL_DEMO_SUBJECTS = (
    DEMO_PATIENTS_CONFIG["stage_0"] +
    DEMO_PATIENTS_CONFIG["stage_1"] +
    DEMO_PATIENTS_CONFIG["stage_2_3"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/setup")
def setup_demo(db: Session = Depends(get_db)):
    """데모용 환자 데이터를 DB에 초기 주입 (0시간 시점).
    모든 환자의 크레아티닌을 0.9(정상) 등 초기 청정 상태로 세팅합니다.
    """
    try:
        _advance_progress["time_step"] = 0
        _reset_alert_dedup()

        # 1. 기존 데이터 비우기
        db.execute(text("TRUNCATE TABLE chym.patients CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.patient_labs CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.patient_trend_points CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.patient_urine_points CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.alerts CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.cohort CASCADE;"))
        db.execute(text("TRUNCATE TABLE chym.icu_features_scaled CASCADE;"))
        
        # 2. cohort 데이터셋 로드
        if not DATASET_CSV.exists():
            raise HTTPException(500, "시뮬레이션용 CSV 데이터셋이 없습니다.")
            
        df_all = pd.read_csv(DATASET_CSV)
        df_demo = df_all[df_all["subject_id"].isin(ALL_DEMO_SUBJECTS)].drop_duplicates(subset=["subject_id"])
        
        # 3. DB 적재
        # CSV(test_final.csv)의 age 는 모델 입력용 z-score 표준화 값(-2~1)이라
        # 그대로 넣으면 화면 나이가 0/1/-1 로 깨진다. 원본 나이는
        # mimiciv_hosp.patients.anchor_age 에서 subject_id 로 조회해 사용한다.
        demo_sids = [int(s) for s in df_demo["subject_id"].tolist()]
        age_map: dict[int, int] = {}
        if demo_sids:
            for r in db.execute(
                text("SELECT subject_id, anchor_age FROM mimiciv_hosp.patients "
                     "WHERE subject_id = ANY(:ids)"),
                {"ids": demo_sids},
            ):
                if r[1] is not None:
                    age_map[int(r[0])] = int(r[1])

        base_admit = datetime(2026, 7, 3, 12, 0, 0)
        for idx, (_, row) in enumerate(df_demo.iterrows()):
            sid = int(row["subject_id"])
            stay_id = int(row["stay_id"])
            name = fake_korean_name(sid)
            # 원본 나이(anchor_age). 데모 대상은 모두 mimiciv_hosp.patients 에 존재해
            # age_map 에 항상 값이 있다(patients.age 는 NOT NULL). 방어적으로만 fallback.
            age = age_map.get(sid)
            if age is None:
                continue  # anchor_age 없는 예외 subject 는 데모에서 제외(정상 경로에선 발생 안 함)
            gender = str(row["gender"])
            admitted_at = (base_admit - timedelta(minutes=idx)).isoformat()

            # 초기 정상값 (모두가 정상범위 Cr 0.9, BUN 15.0, K 4.2)
            cr0 = 0.9
            bun0 = 15.0
            k0 = 4.2
            egfr0 = 85.0

            # cohort 적재
            db.execute(text("""
                INSERT INTO chym.cohort (stay_id, subject_id, age, gender, first_careunit, icu_los_hours, icu_intime, hadm_id)
                VALUES (:stay_id, :subject_id, :age, :gender, :careunit, 48.0, '2026-07-03 12:00:00', :hadm_id)
            """), {
                "stay_id": stay_id, "subject_id": sid, "age": age, "gender": gender,
                "careunit": "Medical Intensive Care Unit (MICU)", "hadm_id": int(row["hadm_id"])
            })

            # patients 적재 (0시간 정상 상태)
            db.execute(text("""
                INSERT INTO chym.patients (id, mrn, name, sex, age, diagnosis, admitted_at, attending, room, ai_risk_score, mimic_subject_id)
                VALUES (:id, :mrn, :name, :sex, :age, :diagnosis, :admitted_at, '홍민준', 'ICU-B1', 12, :sid)
            """), {
                "id": f"p-{sid}", "mrn": f"AKI-{sid}", "name": name, "sex": gender, "age": age,
                "diagnosis": "중환자 모니터링 및 급성 신손상 의증", "admitted_at": admitted_at, "sid": sid
            })

            # 초기 검사 결과 (Lab)
            labs = [
                (0, "cr", "Creatinine", cr0, "mg/dL", 0.7, 1.3),
                (1, "egfr", "eGFR", egfr0, "mL/min", 60.0, None),
                (2, "bun", "BUN", bun0, "mg/dL", 8.0, 20.0),
                (3, "na", "Na", 138.0, "mmol/L", 135.0, 145.0),
                (4, "k", "K", k0, "mmol/L", 3.5, 5.1)
            ]
            for seq, key, label, val, unit, r_low, r_high in labs:
                db.execute(text("""
                    INSERT INTO chym.patient_labs (patient_id, seq, key, label, value, unit, ref_low, ref_high, flag)
                    VALUES (:pid, :seq, :key, :label, :value, :unit, :low, :high, 'normal')
                """), {
                    "pid": f"p-{sid}", "seq": seq, "key": key, "label": label, "value": val, "unit": unit,
                    "low": r_low, "high": r_high
                })

            # 초기 소변량 (정상 속도 1.2)
            db.execute(text("""
                INSERT INTO chym.patient_urine_points (patient_id, date, value)
                VALUES (:pid, :date, 1.2)
            """), {"pid": f"p-{sid}", "date": SETUP_LABEL})

            # 초기 추이 그래프용 포인트 (0h)
            db.execute(text("""
                INSERT INTO chym.patient_trend_points (patient_id, date, creatinine, egfr, bun)
                VALUES (:pid, :date, :cr, :egfr, :bun)
            """), {"pid": f"p-{sid}", "date": SETUP_LABEL, "cr": cr0, "egfr": egfr0, "bun": bun0})

            # 초기 피처 적재 (scaled 테이블도 0h 기준 정상값 주입)
            feat_cols = _model_feature_cols()
            keep_cols = ["stay_id", "subject_id", "age", "gender", "aki_label", "aki_stage"] + feat_cols

            feat_vals = {c: 0.0 for c in keep_cols}
            feat_vals.update({
                "stay_id": stay_id, "subject_id": sid, "age": age, "gender": gender,
                "aki_label": 0, "aki_stage": 0,
                "creatinine_max": cr0, "creatinine_min": 0.9, "creatinine_delta": 0.0,
                "urine_output_sum": 2000.0, "urine_ml_kg_hr": 1.2, "oliguria_flag": 0.0,
                "bun_max": bun0, "potassium_max": k0, "sodium_min": 138.0
            })
            feat_vals = {c: feat_vals.get(c, 0.0) for c in keep_cols}

            cols = ", ".join(f'"{c}"' for c in keep_cols)
            placeholders = ", ".join(f":{c}" for c in keep_cols)
            db.execute(text(f"INSERT INTO chym.icu_features_scaled ({cols}) VALUES ({placeholders})"), feat_vals)

        db.execute(text("TRUNCATE TABLE chym.notifications CASCADE;"))
        db.commit()

        _predictions_cached.cache_clear()
        _sync_ai_risk(db, ALL_DEMO_SUBJECTS)
        db.commit()
        return {"status": "success", "message": f"데모 초기 세팅 완료 ({len(ALL_DEMO_SUBJECTS)}명 환자 정상상태로 로드됨)"}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"세팅 중 오류 발생: {e}")

@router.post("/advance-hour")
def advance_hour(db: Session = Depends(get_db)):
    """위험군 환자를 10명씩 롤링으로 악화시킵니다. 누를 때마다 다음 10명 배치를
    각자의 aki_stage 목표 임상수치로 전환하고 KDIGO 알람 파이프라인까지 태웁니다.
    위험군 80명 기준 8번 누르면 전체 소진됩니다.
    """
    try:
        if not DATASET_CSV.exists():
            raise HTTPException(500, "시뮬레이션용 CSV 데이터셋이 없습니다.")

        # 실제 chym.patients 에 존재하는 위험군만 대상으로 삼는다.
        # setup 이 부분 실패했거나, --reload 로 코호트 설정이 바뀌어 in-memory 목록과
        # DB 가 어긋나도 없는 환자를 건드려 FK 위반이 나는 일을 원천 차단한다.
        wanted = DEMO_PATIENTS_CONFIG["stage_1"] + DEMO_PATIENTS_CONFIG["stage_2_3"]
        present = {
            int(r[0]) for r in db.execute(
                text("SELECT mimic_subject_id FROM chym.patients WHERE mimic_subject_id = ANY(:ids)"),
                {"ids": wanted},
            )
        }
        worsening_pool = [s for s in wanted if s in present]
        if not worsening_pool:
            raise HTTPException(400, "먼저 '시나리오 초기화(Setup)'를 실행해 주세요. (대상 위험 환자가 DB에 없습니다)")

        batch_no = _advance_progress["time_step"] + 1
        start = (batch_no - 1) * DEMO_BATCH_SIZE
        if start >= len(worsening_pool):
            return {"status": "success", "message": "모든 위험군 환자 악화가 반영되었습니다. (초기화 후 다시 진행해 주세요)"}

        batch = worsening_pool[start:start + DEMO_BATCH_SIZE]
        _advance_progress["time_step"] = batch_no
        step_label = _ADVANCE_BATCH_LABELS.get(batch_no, "07-05 00h")

        df_all = pd.read_csv(DATASET_CSV)
        df_batch = df_all[df_all["subject_id"].isin(batch)].drop_duplicates(subset=["subject_id"])

        feat_cols = _model_feature_cols()
        keep_cols = ["stay_id", "subject_id", "age", "gender", "aki_label", "aki_stage"] + feat_cols

        # 같은 스테이지 안에서 환자별 상대 중증도(creatinine_max 순위, 0~1) 산출.
        # 이 값으로 랩 수치 편차의 방향을 잡아 화면 수치와 위험도 순서가 어긋나지 않게 한다.
        sev_map: dict[int, float] = {}
        if "creatinine_max" in df_batch.columns:
            for _stage_val, grp in df_batch.groupby("aki_stage"):
                lo, hi = float(grp["creatinine_max"].min()), float(grp["creatinine_max"].max())
                for _, r in grp.iterrows():
                    sid_ = int(r["subject_id"])
                    sev_map[sid_] = (float(r["creatinine_max"]) - lo) / (hi - lo) if hi > lo else 0.5

        for _, row in df_batch.iterrows():
            sid = int(row["subject_id"])
            stage_gt = int(row["aki_stage"])
            target = _clinical_values_for_stage(stage_gt, subject_id=sid, severity=sev_map.get(sid))

            # 이 배치 환자는 곧바로 목표(stage별) 임상수치로 전환한다.
            cr_val = target["cr"]
            bun_val = target["bun"]
            k_val = target["k"]
            uo_val = target["uo"]
            egfr_val = round(max(5.0, 85.0 * 0.9 / max(cr_val, 0.1)), 1)

            # DB의 실시간 EMR 테이블 업데이트
            flag = "high" if cr_val > 1.3 else "normal"
            db.execute(text("UPDATE chym.patient_labs SET value = :val, flag = :flag WHERE patient_id = :pid AND key = 'cr'"), {"pid": f"p-{sid}", "val": cr_val, "flag": flag})
            db.execute(text("UPDATE chym.patient_labs SET value = :val, flag = :flag WHERE patient_id = :pid AND key = 'bun'"), {"pid": f"p-{sid}", "val": bun_val, "flag": flag})
            db.execute(text("UPDATE chym.patient_labs SET value = :val, flag = :flag WHERE patient_id = :pid AND key = 'k'"), {"pid": f"p-{sid}", "val": k_val, "flag": flag})

            # 추이 그래프 포인트 추가 (INSERT) — 과거 기록은 그대로 두고 새 시점만 쌓는다.
            db.execute(text("""
                INSERT INTO chym.patient_urine_points (patient_id, date, value)
                VALUES (:pid, :date, :val)
            """), {"pid": f"p-{sid}", "date": step_label, "val": uo_val})

            db.execute(text("""
                INSERT INTO chym.patient_trend_points (patient_id, date, creatinine, egfr, bun)
                VALUES (:pid, :date, :cr, :egfr, :bun)
            """), {"pid": f"p-{sid}", "date": step_label, "cr": cr_val, "egfr": egfr_val, "bun": bun_val})

            # AI 모델 입력용 피처를 CSV 최종 스케일값(정답 상태)으로 전환
            update_fields = {}
            for col in keep_cols:
                if col in ("stay_id", "subject_id", "gender"):
                    continue
                final_val = row.get(col)
                if final_val is None or pd.isna(final_val):
                    final_val = 0.0
                else:
                    final_val = float(final_val)
                update_fields[col] = final_val

            update_fields["aki_stage"] = stage_gt
            update_fields["aki_label"] = int(1 if stage_gt > 0 else 0)

            set_clause = ", ".join(f'"{c}" = :{c}' for c in update_fields.keys())
            update_fields["sid"] = sid
            db.execute(text(f"UPDATE chym.icu_features_scaled SET {set_clause} WHERE subject_id = :sid"), update_fields)

            # KDIGO 알람 파이프라인 전송 (수치가 임상 경보 한계를 넘었을 때 alerts에 뜸)
            _publish_lab(
                f"AKI-{sid}",
                {"creatinine": cr_val, "egfr": egfr_val, "urineOutput": uo_val},
                age=int(row["age"]) if not pd.isna(row.get("age")) else None,
                priors=[0.9, 0.9],
                subject_id=sid,
            )

        db.commit()
        _predictions_cached.cache_clear()
        _sync_ai_risk(db, batch)
        db.commit()

        # 방금 악화된 배치 중 위험도가 가장 높은 환자로 알림 딥링크 연결
        top_risk_row = db.execute(text("""
            SELECT mrn, name FROM chym.patients
            WHERE mimic_subject_id = ANY(:sids)
            ORDER BY ai_risk_score DESC LIMIT 1
        """), {"sids": batch}).first()

        top_patient_mrn = top_risk_row[0] if top_risk_row else f"AKI-{batch[0]}"
        top_patient_name = top_risk_row[1] if top_risk_row else "고위험"

        done = min(start + len(batch), len(worsening_pool))
        remaining = len(worsening_pool) - done

        db.execute(text("""
            INSERT INTO chym.notifications (id, severity, department, title, message, read, link)
            VALUES (:id, 'ACTION_REQUIRED', 'nephrology', :title, :message, false, :link)
        """), {
            "id": new_id("demo-alert"),
            "title": f"AKI 경보 - {top_patient_name} 감지",
            "message": f"신규 위험 환자 {len(batch)}명 감지 (누적 {done}/{len(worsening_pool)}명). {top_patient_name} 환자의 위험도가 대폭 상승했습니다.",
            "link": f"/nephrology?patient={top_patient_mrn}",
        })
        db.commit()

        msg = f"위험군 {len(batch)}명 악화 반영 완료 (누적 {done}/{len(worsening_pool)}명)"
        msg += f", 남은 {remaining}명 계속 진행하세요." if remaining > 0 else ". 전체 위험군 소진 완료."
        return {"status": "success", "message": msg}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"시간 경과 처리 중 오류 발생: {e}")

@router.post("/trigger-event")
def trigger_event(db: Session = Depends(get_db)):
    """시나리오 연출용 주인공 환자(오준현 subject_id: 10218191)의 신기능을 강제로 폭등(악화)시켜 경보를 유도시킵니다.
    """
    try:
        sid = PROTAGONIST_SUBJECT_ID

        # setup 이 실행되지 않아 주인공 환자가 DB에 없으면 FK 위반 대신 안내를 준다.
        if not db.execute(
            text("SELECT 1 FROM chym.patients WHERE mimic_subject_id = :sid"), {"sid": sid}
        ).first():
            raise HTTPException(400, "먼저 '시나리오 초기화(Setup)'를 실행해 주세요. (오준현 환자가 DB에 없습니다)")

        db.execute(text("UPDATE chym.patient_labs SET value = 3.6, flag = 'high' WHERE patient_id = :pid AND key = 'cr'"), {"pid": f"p-{sid}"})
        severe = _clinical_values_for_stage(3)
        db.execute(text("UPDATE chym.patient_labs SET value = :val, flag = 'high' WHERE patient_id = :pid AND key = 'k'"), {"pid": f"p-{sid}", "val": severe["k"]})

        # 트리거 포인트 인서트 — 과거 기록은 유지, 새 시점만 추가.
        db.execute(text("""
            INSERT INTO chym.patient_urine_points (patient_id, date, value)
            VALUES (:pid, :date, 0.1)
        """), {"pid": f"p-{sid}", "date": TRIGGER_LABEL})
        db.execute(text("UPDATE chym.patients SET diagnosis = '급성 세뇨관괴사 의증, 고칼륨혈증 동반' WHERE mimic_subject_id = :sid"), {"sid": sid})
        
        df_all = pd.read_csv(DATASET_CSV)
        row = df_all[df_all["subject_id"] == sid].iloc[0]
        
        feat_cols = _model_feature_cols()
        keep_cols = ["stay_id", "subject_id", "age", "gender", "aki_label", "aki_stage"] + feat_cols
        
        update_fields = {}
        for col in keep_cols:
            if col in ("stay_id", "subject_id", "gender"):
                continue
            val = row.get(col)
            if val is None or pd.isna(val):
                val = 0.0
            elif isinstance(val, (int, float)):
                val = float(val)
            update_fields[col] = val
            
        update_fields.update({
            "creatinine_max": 3.6,
            "creatinine_delta": 2.7,
            "aki_stage": 3,
            "aki_label": 1
        })
        update_fields = {c: update_fields[c] for c in keep_cols if c not in ("stay_id", "subject_id", "gender")}
        
        set_clause = ", ".join(f'"{c}" = :{c}' for c in update_fields.keys())
        update_fields["sid"] = sid
        db.execute(text(f"UPDATE chym.icu_features_scaled SET {set_clause} WHERE subject_id = :sid"), update_fields)

        # 48h 시점 추이 포인트 추가 (9자 날짜 데이터 입력)
        bun_val = severe["bun"]
        egfr_est = max(5.0, round(85.0 * 0.9 / max(3.6, 0.1), 1))
        db.execute(text("""
            INSERT INTO chym.patient_trend_points (patient_id, date, creatinine, egfr, bun)
            VALUES (:pid, :date, :cr, :egfr, :bun)
        """), {"pid": f"p-{sid}", "date": TRIGGER_LABEL, "cr": 3.6, "egfr": egfr_est, "bun": bun_val})

        db.commit()

        _publish_lab(
            f"AKI-{sid}",
            {"creatinine": 3.6, "egfr": egfr_est, "urineOutput": 0.1},
            age=int(row["age"]) if not pd.isna(row.get("age")) else None,
            priors=[0.9, 0.9],
            subject_id=sid,
        )

        _predictions_cached.cache_clear()
        real_score = _sync_ai_risk(db, [sid], update_diagnosis=False).get(sid, 0)

        db.commit()
        return {"status": "success", "message": f"오준현 환자 급성 악화 시나리오 트리거 완료 (모델 위험도 {real_score}%)"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"시나리오 트리거 중 오류 발생: {e}")

@router.post("/reset")
def reset_demo(db: Session = Depends(get_db)):
    return setup_demo(db)
