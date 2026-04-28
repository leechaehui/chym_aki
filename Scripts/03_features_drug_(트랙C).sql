-- ═════════════════════════════════════════════════════
-- 03_drug_exposure_features_optimized.sql
-- 핵심 최적화:
--   1. prescriptions 사전 필터링 (20M → ~500K행)
--   2. 약물명 정규식을 WHERE로 끌어올려 JOIN 전에 필터
--   3. 중간 테이블에 인덱스 생성
--   4. ANALYZE로 통계 갱신
-- ═════════════════════════════════════════════════════

-- aki_project 스키마 생성
CREATE SCHEMA IF NOT EXISTS aki_project;

-- ═════════════════════════════════════════════════════
-- STEP 0-A. 관측 윈도우 기준 테이블 + 인덱스
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.aki_cohort_window CASCADE;
CREATE TABLE aki_project.aki_cohort_window AS
SELECT
    a.stay_id,
    a.subject_id,
    a.hadm_id,
    a.icu_intime,
    a.icu_outtime,
    a.aki_label,
    a.aki_stage,
    a.first_source,
    a.both_criteria,
    a.aki_onset_time,
    a.prediction_cutoff,
    a.hours_to_aki,
    COALESCE(a.prediction_cutoff,
             a.icu_intime + INTERVAL '40 hours') AS effective_cutoff,
    CASE WHEN a.prediction_cutoff IS NULL
         THEN 1 ELSE 0 END AS is_pseudo_cutoff
FROM aki_stage_final a;
 
CREATE INDEX idx_cohort_subject ON aki_project.aki_cohort_window (subject_id);
CREATE INDEX idx_cohort_stay    ON aki_project.aki_cohort_window (stay_id);
ANALYZE aki_project.aki_cohort_window;



-- ═════════════════════════════════════════════════════
-- STEP 0-B. prescriptions 사전 필터링 (핵심 최적화)
--
-- 원본: 20,292,611행 (전체)
-- 필터 후: 코호트 환자 + 신독성 약물 행만 → 10~20배 축소 기대
--
-- 적용 원칙:
--   - subject_id 먼저 제한 (인덱스 활용)
--   - 약물명 정규식으로 신독성 약물만
--   - 시간 범위 제한 (icu_intime ~ effective_cutoff)
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.nephrotoxic_rx_raw CASCADE;
CREATE TABLE aki_project.nephrotoxic_rx_raw AS
SELECT
    p.subject_id,
    p.drug,
    p.starttime,
    p.stoptime,
    p.dose_val_rx,
    p.route
FROM mimiciv_hosp.prescriptions p
WHERE p.subject_id IN (SELECT subject_id FROM aki_project.aki_cohort_window)
  AND (
    LOWER(p.drug) LIKE '%vancomycin%'       OR
    LOWER(p.drug) LIKE '%piperacillin%'     OR
    LOWER(p.drug) LIKE '%zosyn%'            OR
    LOWER(p.drug) LIKE '%pip/tazo%'         OR
    LOWER(p.drug) LIKE '%pip-tazo%'         OR
    LOWER(p.drug) LIKE '%gentamicin%'       OR
    LOWER(p.drug) LIKE '%tobramycin%'       OR
    LOWER(p.drug) LIKE '%amikacin%'         OR
    LOWER(p.drug) LIKE '%amphotericin%'     OR
    LOWER(p.drug) LIKE '%meropenem%'        OR
    LOWER(p.drug) LIKE '%imipenem%'         OR
    LOWER(p.drug) LIKE '%ertapenem%'        OR
    LOWER(p.drug) LIKE '%ketorolac%'        OR
    LOWER(p.drug) LIKE '%ibuprofen%'        OR
    LOWER(p.drug) LIKE '%indomethacin%'     OR
    LOWER(p.drug) LIKE '%diclofenac%'       OR
    LOWER(p.drug) LIKE '%lisinopril%'       OR
    LOWER(p.drug) LIKE '%enalapril%'        OR
    LOWER(p.drug) LIKE '%captopril%'        OR
    LOWER(p.drug) LIKE '%ramipril%'         OR
    LOWER(p.drug) LIKE '%losartan%'         OR
    LOWER(p.drug) LIKE '%valsartan%'        OR
    LOWER(p.drug) LIKE '%irbesartan%'       OR
    LOWER(p.drug) LIKE '%furosemide%'       OR
    LOWER(p.drug) LIKE '%lasix%'            OR
    LOWER(p.drug) LIKE '%hydrochlorothiazide%' OR
    LOWER(p.drug) LIKE '%tacrolimus%'       OR
    LOWER(p.drug) LIKE '%prograf%'          OR
    LOWER(p.drug) LIKE '%cyclosporine%'     OR
    LOWER(p.drug) LIKE '%cyclosporin%'      OR
    LOWER(p.drug) LIKE '%metformin%'        OR
    LOWER(p.drug) LIKE '%pantoprazole%'     OR
    LOWER(p.drug) LIKE '%omeprazole%'       OR
    LOWER(p.drug) LIKE '%esomeprazole%'
  );
 
CREATE INDEX idx_nrx_subject_time
    ON aki_project.nephrotoxic_rx_raw (subject_id, starttime);
ANALYZE aki_project.nephrotoxic_rx_raw;

-- ═════════════════════════════════════════════════════
-- STEP 0-C. inputevents 사전 필터
--   STEP 4(수액·조영제)에서 사용
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.inputevents_filtered CASCADE;
CREATE TABLE aki_project.inputevents_filtered AS
SELECT
    i.stay_id, i.itemid, i.starttime, i.amount
FROM mimiciv_icu.inputevents i
WHERE i.stay_id IN (SELECT stay_id FROM aki_project.aki_cohort_window)
  AND i.itemid IN (
      220954, 220955, 220956, 226364, 226375,  -- IV fluids
      225943, 225944, 227522                    -- Contrast
  );
 
CREATE INDEX idx_inputs_stay_item
    ON aki_project.inputevents_filtered (stay_id, itemid, starttime);
ANALYZE aki_project.inputevents_filtered;



-- ═════════════════════════════════════════════════════
-- STEP 1. ICU 내 신독성 약물 처방
-- 좁아진 nephrotoxic_rx_raw 사용 → 훨씬 빠른 JOIN
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.icu_nephrotoxic_rx CASCADE;
CREATE TABLE aki_project.icu_nephrotoxic_rx AS
 
WITH rx_flags AS (
    SELECT
        c.stay_id,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
                 THEN 1 ELSE 0 END)            AS vancomycin_rx,
 
        COALESCE(SUM(CASE
            WHEN LOWER(p.drug) LIKE '%vancomycin%'
            THEN EXTRACT(EPOCH FROM
                   LEAST(COALESCE(p.stoptime, c.effective_cutoff),
                         c.effective_cutoff) - p.starttime
                 ) / 3600.0
        END), 0)                               AS vancomycin_exposure_hours,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%piperacillin%'
                   OR LOWER(p.drug) LIKE '%zosyn%'
                   OR LOWER(p.drug) LIKE '%pip/tazo%'
                   OR LOWER(p.drug) LIKE '%pip-tazo%'
                 THEN 1 ELSE 0 END)            AS piptazo_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%gentamicin%'
                   OR LOWER(p.drug) LIKE '%tobramycin%'
                   OR LOWER(p.drug) LIKE '%amikacin%'
                 THEN 1 ELSE 0 END)            AS aminoglycoside_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%amphotericin%'
                 THEN 1 ELSE 0 END)            AS amphotericin_b_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%meropenem%'
                   OR LOWER(p.drug) LIKE '%imipenem%'
                   OR LOWER(p.drug) LIKE '%ertapenem%'
                 THEN 1 ELSE 0 END)            AS carbapenem_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'
                 THEN 1 ELSE 0 END)            AS ketorolac_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ibuprofen%'
                 THEN 1 ELSE 0 END)            AS ibuprofen_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'
                   OR LOWER(p.drug) LIKE '%ibuprofen%'
                   OR LOWER(p.drug) LIKE '%indomethacin%'
                   OR LOWER(p.drug) LIKE '%diclofenac%'
                 THEN 1 ELSE 0 END)            AS nsaid_any_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'
                   OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'
                   OR LOWER(p.drug) LIKE '%ramipril%'
                 THEN 1 ELSE 0 END)            AS ace_inhibitor_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%losartan%'
                   OR LOWER(p.drug) LIKE '%valsartan%'
                   OR LOWER(p.drug) LIKE '%irbesartan%'
                 THEN 1 ELSE 0 END)            AS arb_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'
                   OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'
                   OR LOWER(p.drug) LIKE '%losartan%'
                   OR LOWER(p.drug) LIKE '%valsartan%'
                 THEN 1 ELSE 0 END)            AS acei_arb_any_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%furosemide%'
                   OR LOWER(p.drug) LIKE '%lasix%'
                 THEN 1 ELSE 0 END)            AS furosemide_rx,
 
        COALESCE(SUM(CASE
            WHEN (LOWER(p.drug) LIKE '%furosemide%'
               OR LOWER(p.drug) LIKE '%lasix%')
             AND p.dose_val_rx ~ '^[0-9.]+$'
            THEN p.dose_val_rx::FLOAT
        END), 0)                               AS furosemide_cumulative_mg,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%hydrochlorothiazide%'
                 THEN 1 ELSE 0 END)            AS hctz_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%tacrolimus%'
                   OR LOWER(p.drug) LIKE '%prograf%'
                 THEN 1 ELSE 0 END)            AS tacrolimus_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%cyclosporine%'
                   OR LOWER(p.drug) LIKE '%cyclosporin%'
                 THEN 1 ELSE 0 END)            AS cyclosporine_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%metformin%'
                 THEN 1 ELSE 0 END)            AS metformin_rx,
 
        MAX(CASE WHEN LOWER(p.drug) LIKE '%pantoprazole%'
                   OR LOWER(p.drug) LIKE '%omeprazole%'
                   OR LOWER(p.drug) LIKE '%esomeprazole%'
                 THEN 1 ELSE 0 END)            AS ppi_rx
 
    FROM aki_project.aki_cohort_window c
    LEFT JOIN aki_project.nephrotoxic_rx_raw p
      ON p.subject_id = c.subject_id
     AND p.starttime >= c.icu_intime
     AND p.starttime <  c.effective_cutoff
    GROUP BY c.stay_id
)
 
SELECT
    cw.stay_id,
    cw.aki_label,
    cw.aki_stage,
    cw.icu_intime,
    cw.effective_cutoff,
    cw.is_pseudo_cutoff,
    COALESCE(r.vancomycin_rx, 0)              AS vancomycin_rx,
    COALESCE(r.vancomycin_exposure_hours, 0)  AS vancomycin_exposure_hours,
    COALESCE(r.piptazo_rx, 0)                 AS piptazo_rx,
    COALESCE(r.aminoglycoside_rx, 0)          AS aminoglycoside_rx,
    COALESCE(r.amphotericin_b_rx, 0)          AS amphotericin_b_rx,
    COALESCE(r.carbapenem_rx, 0)              AS carbapenem_rx,
    COALESCE(r.ketorolac_rx, 0)               AS ketorolac_rx,
    COALESCE(r.ibuprofen_rx, 0)               AS ibuprofen_rx,
    COALESCE(r.nsaid_any_rx, 0)               AS nsaid_any_rx,
    COALESCE(r.ace_inhibitor_rx, 0)           AS ace_inhibitor_rx,
    COALESCE(r.arb_rx, 0)                     AS arb_rx,
    COALESCE(r.acei_arb_any_rx, 0)            AS acei_arb_any_rx,
    COALESCE(r.furosemide_rx, 0)              AS furosemide_rx,
    COALESCE(r.furosemide_cumulative_mg, 0)   AS furosemide_cumulative_mg,
    COALESCE(r.hctz_rx, 0)                    AS hctz_rx,
    COALESCE(r.tacrolimus_rx, 0)              AS tacrolimus_rx,
    COALESCE(r.cyclosporine_rx, 0)            AS cyclosporine_rx,
    COALESCE(r.metformin_rx, 0)               AS metformin_rx,
    COALESCE(r.ppi_rx, 0)                     AS ppi_rx
FROM aki_project.aki_cohort_window cw
LEFT JOIN rx_flags r ON cw.stay_id = r.stay_id;
 
CREATE INDEX idx_icu_rx_stay ON aki_project.icu_nephrotoxic_rx (stay_id);
ANALYZE aki_project.icu_nephrotoxic_rx;

-- ═════════════════════════════════════════════════════
-- STEP 2. 약물 조합 위험도 → nephrotoxic_combo_risk
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.nephrotoxic_combo_risk CASCADE;
CREATE TABLE aki_project.nephrotoxic_combo_risk AS
SELECT
    r.stay_id,
    r.aki_label,
 
    -- 고위험 병용 조합
    r.vancomycin_rx * r.piptazo_rx              AS vanco_piptazo_combo,
    r.vancomycin_rx * r.aminoglycoside_rx       AS vanco_aminogly_combo,
    r.vancomycin_rx * r.carbapenem_rx           AS vanco_carbapenem_combo,
    r.nsaid_any_rx * r.acei_arb_any_rx          AS nsaid_acei_combo,
 
    -- Triple Whammy (NSAIDs + ACEi/ARB + 이뇨제)
    CASE WHEN r.nsaid_any_rx    = 1
          AND r.acei_arb_any_rx = 1
          AND r.furosemide_rx   = 1
         THEN 1 ELSE 0
    END                                         AS triple_whammy,
 
    -- 이뇨제 과다 + 신독성 항생제
    CASE WHEN r.furosemide_cumulative_mg > 200
          AND (r.vancomycin_rx = 1 OR r.aminoglycoside_rx = 1)
         THEN 1 ELSE 0
    END                                         AS diuretic_overload_flag,
 
    -- Metformin + 신독성 동시
    CASE WHEN r.metformin_rx = 1
          AND (r.vancomycin_rx = 1 OR r.nsaid_any_rx = 1)
         THEN 1 ELSE 0
    END                                         AS metformin_risk_flag,
 
    -- 누적 신독성 부담 (0~8)
    r.vancomycin_rx
    + r.aminoglycoside_rx
    + r.piptazo_rx
    + r.amphotericin_b_rx
    + r.nsaid_any_rx
    + r.acei_arb_any_rx
    + r.tacrolimus_rx
    + CASE WHEN r.furosemide_cumulative_mg > 200
           THEN 1 ELSE 0 END                    AS nephrotoxic_burden_score,
 
    -- 약물 처방 위험 스코어 (0~5)
    (CASE WHEN r.vancomycin_rx = 1 AND r.piptazo_rx = 1
          THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_rx + r.aminoglycoside_rx
              + r.amphotericin_b_rx >= 2
           THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_exposure_hours > 48
           THEN 1 ELSE 0 END)
    + (CASE WHEN r.nsaid_any_rx = 1
             AND r.acei_arb_any_rx = 1
            THEN 1 ELSE 0 END)
    + (CASE WHEN r.furosemide_cumulative_mg > 200
           THEN 1 ELSE 0 END)                    AS drug_risk_score
FROM aki_project.icu_nephrotoxic_rx r;
 
CREATE INDEX idx_combo_stay ON aki_project.nephrotoxic_combo_risk (stay_id);
ANALYZE aki_project.nephrotoxic_combo_risk;

-- ═════════════════════════════════════════════════════
-- STEP 3. 응급실 방문 피처 → ed_encounter_features  
--   컬럼명: '_flag' 접미사 일관 적용
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.ed_encounter_features CASCADE;
CREATE TABLE aki_project.ed_encounter_features AS
 
WITH
ed_stay AS (
    SELECT e.stay_id AS ed_stay_id,
           e.subject_id,
           e.intime   AS ed_intime,
           e.outtime  AS ed_outtime
    FROM mimiciv_ed.edstays e
    WHERE e.subject_id IN (SELECT subject_id FROM aki_project.aki_cohort_window)
),
triage_v AS (
    SELECT t.stay_id AS ed_stay_id,
           t.sbp           AS triage_sbp,
           t.heartrate     AS triage_hr,
           t.temperature   AS triage_temp,
           t.o2sat         AS triage_o2sat,
           t.acuity        AS triage_ktas,
           CASE WHEN t.acuity <= 2 THEN 1 ELSE 0 END AS triage_critical_flag
    FROM mimiciv_ed.triage t
    WHERE EXISTS (SELECT 1 FROM ed_stay es WHERE es.ed_stay_id = t.stay_id)
),
preadmission_meds AS (
    SELECT
        m.stay_id AS ed_stay_id,
        MAX(CASE WHEN LOWER(m.name) SIMILAR TO
            '%(ibuprofen|naproxen|ketorolac|diclofenac|meloxicam|celecoxib)%'
            THEN 1 ELSE 0 END) AS nsaid_preadmission_flag,
        MAX(CASE WHEN LOWER(m.name) SIMILAR TO
            '%(lisinopril|enalapril|captopril|ramipril|losartan|valsartan|irbesartan|olmesartan)%'
            THEN 1 ELSE 0 END) AS acei_arb_preadmission_flag,
        MAX(CASE WHEN LOWER(m.name) LIKE '%metformin%'
            THEN 1 ELSE 0 END) AS metformin_preadmission_flag,
        MAX(CASE WHEN LOWER(m.name) SIMILAR TO
            '%(furosemide|hydrochlorothiazide|spironolactone|bumetanide)%'
            THEN 1 ELSE 0 END) AS diuretic_preadmission_flag,
        MAX(CASE WHEN LOWER(m.name) SIMILAR TO
            '%(tacrolimus|cyclosporine|mycophenolate|sirolimus)%'
            THEN 1 ELSE 0 END) AS immunosuppressant_preadmission_flag
    FROM mimiciv_ed.medrecon m
    WHERE EXISTS (SELECT 1 FROM ed_stay es WHERE es.ed_stay_id = m.stay_id)
    GROUP BY m.stay_id
)
 
SELECT
    cw.stay_id,
    CASE WHEN es.ed_intime IS NULL THEN 0
         ELSE EXTRACT(EPOCH FROM
            LEAST(es.ed_outtime, cw.effective_cutoff) - es.ed_intime
         ) / 3600.0
    END                                          AS ed_los_hours,
    CASE WHEN es.ed_intime IS NULL
         THEN 1 ELSE 0 END                       AS direct_icu_admit_flag,
    tv.triage_sbp,
    tv.triage_hr,
    tv.triage_temp,
    tv.triage_o2sat,
    COALESCE(tv.triage_ktas, 3)                  AS triage_ktas,
    COALESCE(tv.triage_critical_flag, 0)         AS triage_critical_flag,
    COALESCE(pm.nsaid_preadmission_flag, 0)      AS nsaid_preadmission_flag,
    COALESCE(pm.acei_arb_preadmission_flag, 0)   AS acei_arb_preadmission_flag,
    COALESCE(pm.metformin_preadmission_flag, 0)  AS metformin_preadmission_flag,
    COALESCE(pm.diuretic_preadmission_flag, 0)   AS diuretic_preadmission_flag,
    COALESCE(pm.immunosuppressant_preadmission_flag, 0)
                                                 AS immunosuppressant_preadmission_flag
FROM aki_project.aki_cohort_window cw
LEFT JOIN ed_stay            es ON cw.subject_id = es.subject_id
LEFT JOIN triage_v           tv ON es.ed_stay_id = tv.ed_stay_id
LEFT JOIN preadmission_meds  pm ON es.ed_stay_id = pm.ed_stay_id;
 
CREATE INDEX idx_ed_stay ON aki_project.ed_encounter_features (stay_id);
ANALYZE aki_project.ed_encounter_features;

-- ═════════════════════════════════════════════════════
-- STEP 4. IV 수액 + 조영제 노출 → iv_fluid_contrast_exposure  
-- ═════════════════════════════════════════════════════
DROP TABLE IF EXISTS aki_project.iv_fluid_contrast_exposure CASCADE;
CREATE TABLE aki_project.iv_fluid_contrast_exposure AS
SELECT
    cw.stay_id,
    COALESCE(SUM(CASE
        WHEN i.itemid IN (220954, 220955, 220956, 226364, 226375)
         AND i.starttime < cw.effective_cutoff
        THEN i.amount END), 0)                   AS iv_fluid_total_ml,
    COALESCE(SUM(CASE
        WHEN i.itemid IN (220954, 220955, 220956, 226364, 226375)
         AND i.starttime < cw.effective_cutoff
        THEN i.amount END), 0)
        / NULLIF(w.weight_kg, 0)                 AS iv_fluid_per_kg,
    MAX(CASE WHEN i.itemid IN (225943, 225944, 227522)
         AND i.starttime < cw.effective_cutoff
        THEN 1 ELSE 0 END)                       AS contrast_exposure_flag,
    w.weight_kg,
    w.weight_source
FROM aki_project.aki_cohort_window         cw
LEFT JOIN aki_project.inputevents_filtered i  ON cw.stay_id = i.stay_id
LEFT JOIN patient_weight_filled            w  ON cw.stay_id = w.stay_id
GROUP BY cw.stay_id, cw.effective_cutoff, w.weight_kg, w.weight_source;
 
CREATE INDEX idx_iv_stay ON aki_project.iv_fluid_contrast_exposure (stay_id);
ANALYZE aki_project.iv_fluid_contrast_exposure;

-- ═════════════════════════════════════════════════════
-- STEP 5. 최종 통합 → aki_drug_exposure_features
-- ═════════════════════════════════════════════════════
-- 테이블명:
--   aki_project.aki_drug_exposure_features
--
-- 목적:
--   AKI 예측 모델을 위한 최종 통합 feature dataset 생성
--   ICU 약물, ED 상태, 수액/조영제, 약물 조합 위험도를 통합하여
--   머신러닝 및 통계 분석에 사용할 수 있는 구조로 정리
--
-- 데이터 기준:
--   aki_project.aki_cohort_window (ICU AKI 분석 코호트)
--
-- 생성 구조 (파이프라인):
--   STEP 1: ICU 약물 노출 (prescriptions 기반 exposure)
--   STEP 2: 약물 조합 위험도 (nephrotoxic interaction / burden score)
--   STEP 3: ED 초기 상태 (vital signs + triage + preadmission meds)
--   STEP 4: IV fluid & contrast exposure (ICU inputevents 기반)
--
-- 최종 산출물:
--   각 ICU stay 단위로 AKI label과 모든 exposure feature를 결합한
--   모델 입력용 wide-format feature table
-- ═════════════════════════════════════════════════════

DROP TABLE IF EXISTS aki_project.aki_drug_exposure_features CASCADE;
CREATE TABLE aki_project.aki_drug_exposure_features AS
SELECT
    -- [기준 코호트]
    cw.stay_id,
    cw.subject_id,
    cw.aki_label,
    cw.aki_stage,
    cw.icu_intime,
    cw.effective_cutoff,
    cw.is_pseudo_cutoff,
    cw.hours_to_aki,
 
    -- [STEP 1: ICU 약물 노출]
    rx.vancomycin_rx,
    rx.vancomycin_exposure_hours,
    rx.piptazo_rx,
    rx.aminoglycoside_rx,
    rx.amphotericin_b_rx,
    rx.carbapenem_rx,
    rx.ketorolac_rx,
    rx.nsaid_any_rx,
    rx.ace_inhibitor_rx,
    rx.arb_rx,
    rx.acei_arb_any_rx,
    rx.furosemide_rx,
    rx.furosemide_cumulative_mg,
    rx.tacrolimus_rx,
    rx.cyclosporine_rx,
    rx.metformin_rx,
    rx.ppi_rx,
 
    -- [STEP 2: 약물 조합 위험도]
    combo.vanco_piptazo_combo,
    combo.vanco_aminogly_combo,
    combo.vanco_carbapenem_combo,
    combo.nsaid_acei_combo,
    combo.triple_whammy,
    combo.diuretic_overload_flag,
    combo.metformin_risk_flag,
    combo.nephrotoxic_burden_score,
    combo.drug_risk_score,
 
    -- [STEP 3: ED 정보]
    ed.ed_los_hours,
    ed.direct_icu_admit_flag,
    ed.triage_sbp,
    ed.triage_hr,
    ed.triage_temp,
    ed.triage_o2sat,
    ed.triage_ktas,
    ed.triage_critical_flag,
    ed.nsaid_preadmission_flag,
    ed.acei_arb_preadmission_flag,
    ed.metformin_preadmission_flag,
    ed.diuretic_preadmission_flag,
    ed.immunosuppressant_preadmission_flag,
 
    -- [STEP 4: 수액·조영제]
    fc.iv_fluid_total_ml,
    fc.iv_fluid_per_kg,
    fc.contrast_exposure_flag,
    fc.weight_kg,
    fc.weight_source,
 
    -- [파생 변수] ICU 약물 + 입원 전 복용 통합 신독성 부담
    combo.nephrotoxic_burden_score
    + COALESCE(ed.nsaid_preadmission_flag, 0)
    + COALESCE(ed.acei_arb_preadmission_flag, 0)
    + COALESCE(ed.immunosuppressant_preadmission_flag, 0)
                                                AS total_nephrotoxic_burden,
 
    -- [파생 변수] 조영제 × NSAIDs 병용 (CI-AKI 위험)
    fc.contrast_exposure_flag
    * GREATEST(rx.nsaid_any_rx,
               COALESCE(ed.nsaid_preadmission_flag, 0))
                                                AS contrast_nsaid_combo
 
FROM aki_project.aki_cohort_window             cw
LEFT JOIN aki_project.icu_nephrotoxic_rx       rx    ON cw.stay_id = rx.stay_id
LEFT JOIN aki_project.nephrotoxic_combo_risk   combo ON cw.stay_id = combo.stay_id
LEFT JOIN aki_project.ed_encounter_features    ed    ON cw.stay_id = ed.stay_id
LEFT JOIN aki_project.iv_fluid_contrast_exposure fc  ON cw.stay_id = fc.stay_id;
 
CREATE INDEX idx_final_stay  ON aki_project.aki_drug_exposure_features (stay_id);
CREATE INDEX idx_final_label ON aki_project.aki_drug_exposure_features (aki_label);
ANALYZE aki_project.aki_drug_exposure_features;

-- ═════════════════════════════════════════════════════
-- 검증 쿼리
-- ═════════════════════════════════════════════════════
 
-- 1. 기본 sanity check
SELECT
    COUNT(*)                                  AS n_rows,
    COUNT(DISTINCT stay_id)                   AS n_unique_stay,
    SUM(CASE WHEN aki_label IS NULL
             THEN 1 ELSE 0 END)               AS missing_label,
    SUM(CASE WHEN vancomycin_rx IS NULL
             THEN 1 ELSE 0 END)               AS null_vanco,
    SUM(CASE WHEN piptazo_rx IS NULL
             THEN 1 ELSE 0 END)               AS null_piptazo
FROM aki_project.aki_drug_exposure_features;
 
 
-- 2. 약물 분포 (AKI vs Non-AKI)
SELECT
    aki_label,
    COUNT(*)                                              AS n,
    ROUND((AVG(vancomycin_rx)       * 100)::NUMERIC, 1)   AS pct_vancomycin,
    ROUND((AVG(piptazo_rx)          * 100)::NUMERIC, 1)   AS pct_piptazo,
    ROUND((AVG(vanco_piptazo_combo) * 100)::NUMERIC, 1)   AS pct_vanco_pip_combo,
    ROUND((AVG(aminoglycoside_rx)   * 100)::NUMERIC, 1)   AS pct_aminogly,
    ROUND((AVG(ketorolac_rx)        * 100)::NUMERIC, 1)   AS pct_ketorolac,
    ROUND((AVG(furosemide_rx)       * 100)::NUMERIC, 1)   AS pct_furosemide,
    ROUND((AVG(triple_whammy)       * 100)::NUMERIC, 1)   AS pct_triple_whammy,
    ROUND(AVG(nephrotoxic_burden_score)::NUMERIC, 2)      AS avg_burden,
    ROUND(AVG(drug_risk_score)::NUMERIC,          2)      AS avg_drug_risk,
    ROUND(AVG(total_nephrotoxic_burden)::NUMERIC, 2)      AS avg_total_burden
FROM aki_project.aki_drug_exposure_features
GROUP BY aki_label
ORDER BY aki_label DESC;
 
 
-- 3. ED feature 부착 여부
SELECT
    COUNT(*)                                  AS n,
    SUM(CASE WHEN ed_los_hours IS NULL
             THEN 1 ELSE 0 END)               AS missing_ed_los,
    SUM(direct_icu_admit_flag)                AS n_direct_icu_admit,
    SUM(CASE WHEN triage_sbp IS NULL
             THEN 1 ELSE 0 END)               AS missing_triage
FROM aki_project.aki_drug_exposure_features;
 
 
-- 4. IV fluid / contrast 확인
SELECT
    COUNT(*)                                  AS n,
    ROUND(AVG(iv_fluid_total_ml)::NUMERIC, 1) AS avg_fluid,
    ROUND(AVG(iv_fluid_per_kg)::NUMERIC,   1) AS avg_fluid_per_kg,
    SUM(contrast_exposure_flag)               AS contrast_cases
FROM aki_project.aki_drug_exposure_features;
 
 
-- 5. coherence check (flag vs exposure 불일치)
SELECT
    COUNT(*)                                  AS n,
    SUM(CASE WHEN vancomycin_rx = 1
              AND vancomycin_exposure_hours = 0
             THEN 1 ELSE 0 END)               AS vanco_flag_but_zero_hours,
    SUM(CASE WHEN furosemide_rx = 1
              AND furosemide_cumulative_mg = 0
             THEN 1 ELSE 0 END)               AS lasix_flag_but_zero_dose
FROM aki_project.aki_drug_exposure_features;

-- ═════════════════════════════════════════════════════
-- 성능 비교: 원본 prescriptions vs 최적화 nephrotoxic_rx_raw
-- ═════════════════════════════════════════════════════
-- 캐시 영향 최소화
DISCARD ALL;


-- ═════════════════════════════════════════════════════
-- [비교 1-A] 원본: prescriptions 직접 JOIN (데이터 흐름 검증)

-- ═════════════════════════════════════════════════════
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    COUNT(*)                                           AS total_join_rows,
    COUNT(p.*)                                         AS matched_rows,
    COUNT(*) - COUNT(p.*)                              AS unmatched_rows,
    ROUND(100.0 * COUNT(p.*) / COUNT(*), 2)            AS match_rate_pct
FROM aki_project.aki_cohort_window c
LEFT JOIN mimiciv_hosp.prescriptions p
    ON c.subject_id = p.subject_id
   AND p.starttime >= c.icu_intime
   AND p.starttime <  c.effective_cutoff;


-- ═════════════════════════════════════════════════════
-- [비교 1-B] 최적화: nephrotoxic_rx_raw 사용
-- 사전 필터된 테이블로 같은 검증 수행
-- ═════════════════════════════════════════════════════
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    COUNT(*)                                           AS total_join_rows,
    COUNT(p.*)                                         AS matched_rows,
    COUNT(*) - COUNT(p.*)                              AS unmatched_rows,
    ROUND(100.0 * COUNT(p.*) / COUNT(*), 2)            AS match_rate_pct
FROM aki_project.aki_cohort_window c
LEFT JOIN aki_project.nephrotoxic_rx_raw p
    ON c.subject_id = p.subject_id
   AND p.starttime >= c.icu_intime
   AND p.starttime <  c.effective_cutoff;



-- ═════════════════════════════════════════════════════
-- [비교 2-A] 원본: 환자별 처방 건수 Top 10 (폭발 원인)
-- Image 2와 동일 — 29,872 ms 예상
-- ═════════════════════════════════════════════════════
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.stay_id,
    COUNT(p.*)                                         AS rx_count
FROM aki_project.aki_cohort_window c
LEFT JOIN mimiciv_hosp.prescriptions p
    ON c.subject_id = p.subject_id
GROUP BY c.stay_id
ORDER BY rx_count DESC
LIMIT 10;


-- ═════════════════════════════════════════════════════
-- [비교 2-B] 최적화: 신독성 약물 처방 건수 Top 10
-- 사전 필터된 테이블로 같은 랭킹 수행
-- ═════════════════════════════════════════════════════
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.stay_id,
    COUNT(p.*)                                         AS rx_count
FROM aki_project.aki_cohort_window c
LEFT JOIN aki_project.nephrotoxic_rx_raw p
    ON c.subject_id = p.subject_id
GROUP BY c.stay_id
ORDER BY rx_count DESC
LIMIT 10;



-- ═════════════════════════════════════════════════════
-- [비교 3] 전체 파이프라인 — STEP 1 완전 재현
-- 실무에서 가장 중요한 비교: 실제 사용할 쿼리의 전체 시간
-- ═════════════════════════════════════════════════════

-- 3-A. 원본 기반 집계 (프로덕션에서 발생한 실제 폭발)
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.stay_id,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
             THEN 1 ELSE 0 END)                        AS vancomycin_rx,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%piperacillin%'
               OR LOWER(p.drug) LIKE '%zosyn%'
             THEN 1 ELSE 0 END)                        AS piptazo_rx,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%furosemide%'
               OR LOWER(p.drug) LIKE '%lasix%'
             THEN 1 ELSE 0 END)                        AS furosemide_rx
FROM aki_project.aki_cohort_window c
LEFT JOIN mimiciv_hosp.prescriptions p
    ON c.subject_id = p.subject_id
   AND p.starttime >= c.icu_intime
   AND p.starttime <  c.effective_cutoff
GROUP BY c.stay_id;


-- 3-B. 최적화 기반 집계
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.stay_id,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
             THEN 1 ELSE 0 END)                        AS vancomycin_rx,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%piperacillin%'
               OR LOWER(p.drug) LIKE '%zosyn%'
             THEN 1 ELSE 0 END)                        AS piptazo_rx,
    MAX(CASE WHEN LOWER(p.drug) LIKE '%furosemide%'
               OR LOWER(p.drug) LIKE '%lasix%'
             THEN 1 ELSE 0 END)                        AS furosemide_rx
FROM aki_project.aki_cohort_window c
LEFT JOIN aki_project.nephrotoxic_rx_raw p
    ON c.subject_id = p.subject_id
   AND p.starttime >= c.icu_intime
   AND p.starttime <  c.effective_cutoff
GROUP BY c.stay_id;
