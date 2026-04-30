-- ═══════════════════════════════════════════════════════════════════════════
-- AKI CDSS : 피처 추출 SQL 파이프라인 (SCR-03 → 04 → 05 → 06 → 07)
--
-- ┌ 프로젝트 목적 ────────────────────────────────────────────────────────┐
-- │ MIMIC-IV ICU 데이터 기반으로 급성 신손상(AKI) 발생을 prediction_     │
-- │ cutoff 시점에서 12시간 이내 예측하는 CDSS 피처 파이프라인.           │
-- │ 화면설계서 순서와 1:1로 대응하여 SQL 유지보수성을 확보한다.          │
-- └───────────────────────────────────────────────────────────────────────┘
--
-- ┌ 데이터 소스 (MIMIC-IV PostgreSQL) ────────────────────────────────────┐
-- │ mimiciv_hosp.admissions    입퇴원 기록, 원내 사망 시각               │
-- │ mimiciv_hosp.patients      환자 기본 정보 (생년·성별·사망일)          │
-- │ mimiciv_hosp.prescriptions 처방 기록 (~20 M행)                       │
-- │ mimiciv_hosp.labevents     혈액검사 결과 (~300 M행)                  │
-- │ mimiciv_icu.chartevents    ICU 활력징후 차트 (~300 M행)              │
-- │ mimiciv_icu.inputevents    ICU 투약·수액 기록                        │
-- └───────────────────────────────────────────────────────────────────────┘
--
-- ┌ 실행 순서 ─────────────────────────────────────────────────────────────┐
-- │ [공통] cdss_cohort_window          코호트 기준 메타 테이블             │
-- │ STEP 1 (SCR-03) cdss_nephrotoxic_rx_raw                              │
-- │                 cdss_icu_nephrotoxic_rx                              │
-- │                 cdss_nephrotoxic_combo_risk                          │
-- │ STEP 2 (SCR-04) cdss_raw_lab_values                                 │
-- │                 cdss_lab_features                                    │
-- │ STEP 3 (SCR-05) cdss_raw_map_values                                 │
-- │                 cdss_feat_map_ischemia                               │
-- │                 cdss_feat_map_summary                                │
-- │                 cdss_feat_shock_index                                │
-- │                 cdss_feat_vasopressor                                │
-- │                 cdss_ischemic_features                               │
-- │ STEP 4 (SCR-06) cdss_rule_score_features                            │
-- │ STEP 5 (SCR-07) cdss_master_features  ← XGBoost 단일 입력 소스      │
-- └───────────────────────────────────────────────────────────────────────┘
--
-- ┌ 테이블명 규칙 ─────────────────────────────────────────────────────────┐
-- │ cdss_ 접두사 : 이 파이프라인에서 신규 생성하는 테이블.                │
-- │ 기존 DB의 aki_cohort_window 등과 이름 충돌을 방지한다.               │
-- │ 인덱스도 cdss_idx_ 접두사로 통일한다.                                │
-- └───────────────────────────────────────────────────────────────────────┘
-- ═══════════════════════════════════════════════════════════════════════════


-- ═══════════════════════════════════════════════════════════════════════════
-- [공통] cdss_cohort_window  —  모든 STEP의 LEFT JOIN 기준 테이블
--
-- ▶ 역할 요약
--   환자 1명 = 1행. stay_id를 기본 키로 삼아 이후 모든 STEP 테이블이
--   LEFT JOIN으로 이 테이블에 붙는다. effective_cutoff 컬럼이 핵심이며,
--   이 값 이전 데이터만 피처 추출에 사용해 미래 누수(Data Leakage)를 막는다.
--
-- ▶ effective_cutoff 설계
--   AKI 환자  → prediction_cutoff (AKI 발생 12시간 전)
--   비AKI 환자 → icu_intime + 40 h (pseudo cutoff)
--   사망 환자  → LEAST(위 두 값, death_time)  ← 사망 이후 데이터 차단
--
-- ▶ 사망 환자 3가지 처리 전략
--   ① 제외  icu_los_hours < 24 : 측정값이 거의 없어 피처 신뢰 불가
--           입원 후 6 h 이내 사망 : 최소 관측 구간조차 없음
--   ② 플래그 competed_with_death = 1 :
--           사망 시각 < prediction_cutoff → aki_label=0이 "비AKI"가 아닌
--           "AKI 발생 전 사망"이므로, XGBoost 학습 시 load_data()에서 제외
--   ③ 캡     effective_cutoff = LEAST(pseudo, death_time)
--           사망 이후 charttime 데이터가 피처에 혼입되는 것을 원천 차단
--
-- ▶ 백엔드 연결 (db.py → PatientBase)
--   hospital_expire_flag, competed_with_death, hours_to_death 필드가
--   PatientBase 모델에 포함되어 API 응답으로 전달되며,
--   competed_with_death=1이면 SCR-06 AI 점수 해석 주의 경고를 표시한다.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_cohort_window CASCADE;
-- CASCADE : 이 테이블에 종속된 뷰·인덱스가 있어도 함께 삭제

CREATE TABLE aki_project.cdss_cohort_window AS

WITH death_info AS (
    -- ── 사망 시각 정규화 ────────────────────────────────────────────────
    -- MIMIC-IV에는 사망 정보가 두 곳에 분산됨.
    --   admissions.deathtime : ICU 내 사망 시각 (분 단위로 정확)
    --   patients.dod         : 사망 날짜만 있음 (병원 외 사망 포함)
    -- → deathtime 우선, 없으면 dod 당일 23:59:59 (보수적 추정).
    --   당일 중 가장 늦은 시각을 써서 피처 포함 범위를 최대화한다.
    SELECT
        ad.hadm_id,
        ad.subject_id,
        COALESCE(
            ad.deathtime,
            p.dod + INTERVAL '23 hours 59 minutes'
        )                        AS death_time,
        ad.hospital_expire_flag  -- 1 = 이번 입원 중 사망, 0 = 생존 퇴원
    FROM mimiciv_hosp.admissions ad
    JOIN mimiciv_hosp.patients   p  ON ad.subject_id = p.subject_id
    WHERE ad.hospital_expire_flag = 1   -- 사망 환자만
       OR p.dod IS NOT NULL             -- 또는 사망일자 기록이 있는 경우
)

SELECT
    -- ── 식별자 ─────────────────────────────────────────────────────────
    a.stay_id,        -- ICU 체류 ID, PK 역할. 이후 모든 STEP JOIN 기준
    a.subject_id,     -- 환자 고유 ID. prescriptions·labevents JOIN에 사용
    a.hadm_id,        -- 입원 ID. admissions JOIN에 사용

    -- ── ICU 체류 정보 ───────────────────────────────────────────────────
    c.icu_intime,     -- ICU 입실 시각 ← 모든 피처의 시작 기준선
    c.icu_outtime,    -- ICU 퇴실 시각
    c.age,            -- 나이 ← CKD-EPI eGFR 공식에 필수 (0.9938^age)
    c.gender,         -- 성별 M/F ← CKD-EPI 성별 계수(κ) 결정에 필수
    c.first_careunit, -- 최초 배치 ICU 유형 (MICU·SICU·CCU 등)
    c.icu_los_hours,  -- ICU 재원 시간 ← 24 h 미만 제외 기준

    -- ── AKI 레이블 ─────────────────────────────────────────────────────
    a.aki_label,       -- 타겟: 0=비AKI, 1=AKI 발생
    a.aki_stage,       -- AKI 중증도: 0(없음)/1/2/3 (KDIGO 기준)
    a.aki_onset_time,  -- AKI 실제 발생 시각
    a.prediction_cutoff, -- AKI 발생 12h 전 시각, 비AKI는 NULL
    a.hours_to_aki,    -- 피처 추출 시점 → AKI까지 남은 시간

    -- ── 사망 관련 컬럼 ─────────────────────────────────────────────────
    d.death_time,
    d.hospital_expire_flag,
    -- ICU 입원 이후 사망까지 경과 시간 (시간 단위)
    -- 용도: hours_to_death < 6 인 환자 제외 필터 기준
    EXTRACT(EPOCH FROM (d.death_time - c.icu_intime)) / 3600.0
        AS hours_to_death,

    -- ── 경쟁 위험 플래그 ────────────────────────────────────────────────
    -- 사망 시각이 prediction_cutoff보다 앞선 경우.
    -- 이 환자의 aki_label=0은 "비AKI"가 아니라 "AKI 발생 전 사망"이므로
    -- XGBoost 학습에서 제외해야 한다. (→ xgboost_pipeline.py load_data 참고)
    CASE
        WHEN d.death_time IS NOT NULL
         AND d.death_time < COALESCE(a.prediction_cutoff,
                                     c.icu_intime + INTERVAL '40 hours')
        THEN 1 ELSE 0
    END AS competed_with_death,

    -- ── effective_cutoff : 피처 추출의 절대 상한 ─────────────────────
    -- 계산 우선순위:
    --   1순위 prediction_cutoff (AKI 환자, 가장 정확한 기준)
    --   2순위 icu_intime + 40 h  (비AKI pseudo cutoff)
    --   캡    LEAST(..., death_time) ← 사망 이후 데이터 차단
    --
    -- 예시 (사망 환자):
    --   icu_intime=00:00, death_time=30:00, pseudo=40:00
    --   → LEAST(40h, 30h) = 30:00 ✅ 사망 후 10시간 데이터 차단
    LEAST(
        COALESCE(a.prediction_cutoff, c.icu_intime + INTERVAL '40 hours'),
        COALESCE(d.death_time, '9999-12-31'::TIMESTAMP)
        -- 생존 환자: death_time=NULL → 9999년 → LEAST에서 자동 제외
    ) AS effective_cutoff,

    -- pseudo cutoff 사용 여부 지시변수
    -- XGBoost 학습 후 하위그룹 분석(pseudo vs real cutoff) 에 활용
    CASE WHEN a.prediction_cutoff IS NULL THEN 1 ELSE 0 END
        AS is_pseudo_cutoff

FROM aki_stage_final a
JOIN cohort          c  ON a.stay_id = c.stay_id
LEFT JOIN death_info d  ON a.hadm_id = d.hadm_id
-- LEFT JOIN 필수: 생존 환자는 death_info에 없으므로 INNER JOIN 쓰면 전부 제외됨

WHERE
    -- ─ 제외 조건 1: 재원 24 h 미만 ──────────────────────────────────
    -- 24 h 이내에는 혈액검사 1~2회, MAP 측정 수십 건 수준으로 집계 피처의
    -- 신뢰도가 너무 낮다. 강화하려면 48로 변경.
    c.icu_los_hours >= 24

    -- ─ 제외 조건 2: 입원 후 6 h 이내 사망 ──────────────────────────
    -- 6 h 이내는 기본 혈액검사(CBC·BMP) 결과조차 나오지 않은 시점.
    -- 피처가 사실상 전부 NULL → 모델에 노이즈만 추가.
    AND NOT (
        d.death_time IS NOT NULL
        AND EXTRACT(EPOCH FROM (d.death_time - c.icu_intime)) / 3600.0 < 6
    );

-- 인덱스
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_stay    ON aki_project.cdss_cohort_window (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_subject ON aki_project.cdss_cohort_window (subject_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_hadm    ON aki_project.cdss_cohort_window (hadm_id);
CREATE INDEX IF NOT EXISTS cdss_idx_cohort_win_death   ON aki_project.cdss_cohort_window (competed_with_death);
-- ANALYZE aki_project.cdss_cohort_window;

-- ── 코호트 구성 검증 (실행 후 반드시 확인) ──────────────────────────────
-- aki_pct     정상 범위 20~40 %. 너무 낮으면 코호트 기준 재검토
-- n_competing 전체의 5~15 %가 정상. 너무 높으면 cutoff 로직 재검토
-- avg_icu_los 24 h 필터 후에도 72~120 h 수준이면 정상
SELECT
    COUNT(*)                                                  AS n_total,
    SUM(aki_label)                                            AS n_aki,
    ROUND(100.0 * SUM(aki_label) / COUNT(*), 1)               AS aki_pct,
    SUM(CASE WHEN hospital_expire_flag = 1 THEN 1 END)        AS n_died,
    SUM(competed_with_death)                                  AS n_competing,
    SUM(is_pseudo_cutoff)                                     AS n_pseudo_cutoff,
    ROUND(100.0 * SUM(competed_with_death)
          / NULLIF(SUM(CASE WHEN hospital_expire_flag=1 THEN 1 END),0), 1)
                                                              AS pct_death_competing,
    ROUND(AVG(icu_los_hours)::NUMERIC, 1)                     AS avg_icu_los_h
FROM aki_project.cdss_cohort_window;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 1-A  (SCR-03 처방 약물 관리 > 처방 목록 테이블)
-- cdss_nephrotoxic_rx_raw  —  신독성 처방 사전 필터링
--
-- ▶ 역할 요약
--   prescriptions 전체 20 M행 중 코호트 환자 + 신독성 약물만 선별해
--   약 500 K행으로 압축한다. JOIN 전에 먼저 줄이므로 이후 쿼리가 10~20배 빨라진다.
--
-- ▶ 시간 필터는 이 단계에서 걸지 않는다
--   effective_cutoff 기반 시간 필터는 STEP 1-B 집계 단계에서 적용.
--   여기서는 넓게 포함해 재사용성을 높인다.
--
-- ▶ 신독성 약물 분류 체계
--   직접 신독성 : 신세포에 직접 손상 (아미노글리코사이드·amphotericin B)
--   혈역학적    : 신혈류 감소를 통한 간접 손상
--     NSAIDs  → 프로스타글란딘 억제 → 구심성 세동맥 수축 → GFR 감소
--     ACEi/ARB → 원심성 세동맥 이완 → 사구체 여과압 감소
--     이뇨제   → 혈관 내 용적 감소 → 신관류압 저하
--   조합 증폭  : Vanco+Pip/Tazo → AKI 3.7배↑ (2019 ASHP 메타분석)
--               Triple Whammy   → NSAIDs+ACEi/ARB+이뇨제 → GFR 0 수렴
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_nephrotoxic_rx_raw CASCADE;

CREATE TABLE aki_project.cdss_nephrotoxic_rx_raw AS
SELECT
    p.subject_id,
    p.drug,          -- 약물명 (LOWER 변환 후 LIKE 비교)
    p.starttime,     -- 처방 시작 시각 ← STEP 1-B의 effective_cutoff 필터 기준
    p.stoptime,      -- 처방 종료 시각 (NULL = 계속 투여 중)
    p.dose_val_rx,   -- 용량 수치 ← furosemide 누적 mg 계산에 사용
    p.dose_unit_rx,  -- 용량 단위 (mg·mcg 등)
    p.route          -- 투여 경로 (PO=경구, IV=정맥, IM=근육)
FROM mimiciv_hosp.prescriptions p
WHERE
    -- ① 코호트 환자만 — IN 서브쿼리로 먼저 줄여 대형 JOIN 방지
    p.subject_id IN (SELECT subject_id FROM aki_project.cdss_cohort_window)

    -- ② 신독성 약물 키워드 LIKE 매칭 (브랜드명+성분명 모두 포함)
    AND (
        -- 항생제·항진균제 (직접 신독성)
        LOWER(p.drug) LIKE '%vancomycin%'      -- 글리코펩타이드. ICU 신독성 1위
     OR LOWER(p.drug) LIKE '%piperacillin%'    -- β-lactam. vanco 병용 시 3.7배↑
     OR LOWER(p.drug) LIKE '%zosyn%'           -- pip/taz 브랜드명
     OR LOWER(p.drug) LIKE '%pip/tazo%'
     OR LOWER(p.drug) LIKE '%pip-tazo%'
     OR LOWER(p.drug) LIKE '%gentamicin%'      -- 아미노글리코사이드. 근위세관 직접 손상
     OR LOWER(p.drug) LIKE '%tobramycin%'
     OR LOWER(p.drug) LIKE '%amikacin%'
     OR LOWER(p.drug) LIKE '%amphotericin%'    -- 항진균. 신독성 발생률 30~80%
     OR LOWER(p.drug) LIKE '%meropenem%'       -- 카르바페넴. vanco 병용 시 위험↑
     OR LOWER(p.drug) LIKE '%imipenem%'
     OR LOWER(p.drug) LIKE '%ertapenem%'
        -- NSAIDs (혈역학적 신독성)
     OR LOWER(p.drug) LIKE '%ketorolac%'       -- ICU 가장 흔한 NSAIDs
     OR LOWER(p.drug) LIKE '%ibuprofen%'
     OR LOWER(p.drug) LIKE '%indomethacin%'    -- 동맥관 폐쇄 목적으로도 사용
     OR LOWER(p.drug) LIKE '%diclofenac%'
        -- ACEi (원심성 세동맥 이완 → GFR 압력 감소)
     OR LOWER(p.drug) LIKE '%lisinopril%'
     OR LOWER(p.drug) LIKE '%enalapril%'
     OR LOWER(p.drug) LIKE '%captopril%'
     OR LOWER(p.drug) LIKE '%ramipril%'
        -- ARB (ACEi와 동일 기전)
     OR LOWER(p.drug) LIKE '%losartan%'
     OR LOWER(p.drug) LIKE '%valsartan%'
     OR LOWER(p.drug) LIKE '%irbesartan%'
        -- 이뇨제 (혈관 내 용적 감소 → 신관류압 저하)
     OR LOWER(p.drug) LIKE '%furosemide%'      -- Loop 이뇨제. ICU 가장 흔함
     OR LOWER(p.drug) LIKE '%lasix%'           -- furosemide 브랜드명
     OR LOWER(p.drug) LIKE '%hydrochlorothiazide%'
        -- 면역억제제
     OR LOWER(p.drug) LIKE '%tacrolimus%'      -- 칼시뉴린 억제제. 신이식 필수약
     OR LOWER(p.drug) LIKE '%prograf%'         -- tacrolimus 브랜드명
     OR LOWER(p.drug) LIKE '%cyclosporine%'
     OR LOWER(p.drug) LIKE '%cyclosporin%'
        -- 기타
     OR LOWER(p.drug) LIKE '%metformin%'       -- 신기능 저하 시 젖산산증 위험
     OR LOWER(p.drug) LIKE '%pantoprazole%'
     OR LOWER(p.drug) LIKE '%omeprazole%'
     OR LOWER(p.drug) LIKE '%esomeprazole%'
    );

-- (subject_id, starttime) 복합 인덱스
-- STEP 1-B의 "subject_id=c.subject_id AND starttime < cutoff" 조건에 최적화
CREATE INDEX IF NOT EXISTS cdss_idx_nrx_subject_time
    ON aki_project.cdss_nephrotoxic_rx_raw (subject_id, starttime);
-- ANALYZE aki_project.cdss_nephrotoxic_rx_raw;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 1-B  (SCR-03 처방 약물 관리 > 처방 목록 테이블 + AI 신독성 배너)
-- cdss_icu_nephrotoxic_rx  —  환자별 ICU 신독성 약물 노출 집계
--
-- ▶ 역할 요약
--   STEP 1-A의 약 500 K행을 환자 1명당 1행으로 집계한다.
--   약물 노출 여부(0/1), 노출 시간(시간), 누적 용량(mg)을 계산해
--   SCR-03 처방 목록 테이블과 AI 신독성 배너의 원천 데이터를 제공한다.
--
-- ▶ effective_cutoff 필터 위치
--   starttime < effective_cutoff → prediction 이후 처방 차단 (누수 방지)
--   stoptime이 cutoff를 넘어가도 cutoff까지만 시간을 잘라 계산한다.
--
-- ▶ vancomycin_exposure_hours 계산 원리
--   LEAST(stoptime, effective_cutoff) — cutoff를 넘어가는 처방은 cutoff에서 자름
--   COALESCE(stoptime, effective_cutoff) — 아직 종료 안 된 처방은 cutoff까지 인정
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_icu_nephrotoxic_rx CASCADE;

CREATE TABLE aki_project.cdss_icu_nephrotoxic_rx AS

WITH rx_flags AS (
    SELECT
        c.stay_id,

        -- ─ Vancomycin ───────────────────────────────────────────────────
        -- SCR-03 처방 목록 "주의" 상태 표시 기준.
        -- ICU 환자 30~40%가 투여받으며 신독성 발생률 5~43%로 보고됨.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
                 THEN 1 ELSE 0 END)                     AS vancomycin_rx,

        -- vancomycin 노출 시간 (시간 단위).
        -- 48 h 초과 시 만성 노출로 신독성 위험 급증 → SCR-03 AI 배너 장기 노출 경고.
        COALESCE(SUM(
            CASE WHEN LOWER(p.drug) LIKE '%vancomycin%'
            THEN EXTRACT(EPOCH FROM (
                     LEAST(COALESCE(p.stoptime, c.effective_cutoff),
                           c.effective_cutoff)
                     - p.starttime
                 )) / 3600.0
            END
        ), 0)                                           AS vancomycin_exposure_hours,

        -- ─ Pip/Tazo ────────────────────────────────────────────────────
        -- 단독 신독성은 낮으나 vancomycin과 병용 시 AKI 3.7배↑.
        -- zosyn·pip/tazo·pip-tazo 표기 변형을 모두 포착.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%piperacillin%'
                   OR LOWER(p.drug) LIKE '%zosyn%'
                   OR LOWER(p.drug) LIKE '%pip/tazo%'
                   OR LOWER(p.drug) LIKE '%pip-tazo%'
                 THEN 1 ELSE 0 END)                     AS piptazo_rx,

        -- ─ 아미노글리코사이드 ───────────────────────────────────────────
        -- 신장 근위세관 상피세포에 직접 축적 → 세포사.
        -- 1회 고용량(once-daily) 방식이 분할 투여보다 신독성이 낮음.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%gentamicin%'
                   OR LOWER(p.drug) LIKE '%tobramycin%'
                   OR LOWER(p.drug) LIKE '%amikacin%'
                 THEN 1 ELSE 0 END)                     AS aminoglycoside_rx,

        -- ─ Amphotericin B ───────────────────────────────────────────────
        -- 항진균제 중 가장 강한 직접 신독성 (발생률 30~80%).
        -- liposomal 제형은 신독성이 낮지만 비용이 매우 높음.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%amphotericin%'
                 THEN 1 ELSE 0 END)                     AS amphotericin_b_rx,

        -- ─ 카르바페넴 ───────────────────────────────────────────────────
        -- 단독 신독성은 낮으나 vancomycin 병용 시 위험 증가.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%meropenem%'
                   OR LOWER(p.drug) LIKE '%imipenem%'
                   OR LOWER(p.drug) LIKE '%ertapenem%'
                 THEN 1 ELSE 0 END)                     AS carbapenem_rx,

        -- ─ NSAIDs ───────────────────────────────────────────────────────
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'
                 THEN 1 ELSE 0 END)                     AS ketorolac_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ibuprofen%'
                 THEN 1 ELSE 0 END)                     AS ibuprofen_rx,
        -- Triple Whammy 계산에 사용하는 통합 플래그
        MAX(CASE WHEN LOWER(p.drug) LIKE '%ketorolac%'
                   OR LOWER(p.drug) LIKE '%ibuprofen%'
                   OR LOWER(p.drug) LIKE '%indomethacin%'
                   OR LOWER(p.drug) LIKE '%diclofenac%'
                 THEN 1 ELSE 0 END)                     AS nsaid_any_rx,

        -- ─ ACEi ─────────────────────────────────────────────────────────
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'
                   OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'
                   OR LOWER(p.drug) LIKE '%ramipril%'
                 THEN 1 ELSE 0 END)                     AS ace_inhibitor_rx,

        -- ─ ARB ──────────────────────────────────────────────────────────
        MAX(CASE WHEN LOWER(p.drug) LIKE '%losartan%'
                   OR LOWER(p.drug) LIKE '%valsartan%'
                   OR LOWER(p.drug) LIKE '%irbesartan%'
                 THEN 1 ELSE 0 END)                     AS arb_rx,

        -- Triple Whammy 계산: ACEi 또는 ARB 중 하나라도 있으면 1
        MAX(CASE WHEN LOWER(p.drug) LIKE '%lisinopril%'
                   OR LOWER(p.drug) LIKE '%enalapril%'
                   OR LOWER(p.drug) LIKE '%captopril%'
                   OR LOWER(p.drug) LIKE '%losartan%'
                   OR LOWER(p.drug) LIKE '%valsartan%'
                 THEN 1 ELSE 0 END)                     AS acei_arb_any_rx,

        -- ─ 이뇨제 ───────────────────────────────────────────────────────
        -- SCR-03 처방 목록 "유지" 상태 표시 (신독성 약물은 아님).
        MAX(CASE WHEN LOWER(p.drug) LIKE '%furosemide%'
                   OR LOWER(p.drug) LIKE '%lasix%'
                 THEN 1 ELSE 0 END)                     AS furosemide_rx,

        -- furosemide 누적 용량 (mg).
        -- 200 mg 초과 → 이뇨제 과부하 → 혈관 내 용적 감소 → 신독성 약물 농도 상승.
        -- REGEXP '^[0-9.]+$' : 숫자가 아닌 "80-120" 같은 범위 표기 필터링.
        COALESCE(SUM(
            CASE WHEN (LOWER(p.drug) LIKE '%furosemide%'
                    OR LOWER(p.drug) LIKE '%lasix%')
                  AND p.dose_val_rx ~ '^[0-9.]+$'
            THEN p.dose_val_rx::FLOAT
            END
        ), 0)                                           AS furosemide_cumulative_mg,

        MAX(CASE WHEN LOWER(p.drug) LIKE '%hydrochlorothiazide%'
                 THEN 1 ELSE 0 END)                     AS hctz_rx,

        -- ─ 면역억제제 ───────────────────────────────────────────────────
        -- SCR-03 비고란 "면역억제 효과 모니터링 필요" 표시에 사용.
        MAX(CASE WHEN LOWER(p.drug) LIKE '%tacrolimus%'
                   OR LOWER(p.drug) LIKE '%prograf%'
                 THEN 1 ELSE 0 END)                     AS tacrolimus_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%cyclosporine%'
                   OR LOWER(p.drug) LIKE '%cyclosporin%'
                 THEN 1 ELSE 0 END)                     AS cyclosporine_rx,

        MAX(CASE WHEN LOWER(p.drug) LIKE '%metformin%'
                 THEN 1 ELSE 0 END)                     AS metformin_rx,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%pantoprazole%'
                   OR LOWER(p.drug) LIKE '%omeprazole%'
                   OR LOWER(p.drug) LIKE '%esomeprazole%'
                 THEN 1 ELSE 0 END)                     AS ppi_rx

    FROM aki_project.cdss_cohort_window c
    LEFT JOIN aki_project.cdss_nephrotoxic_rx_raw p
           ON p.subject_id = c.subject_id
              AND p.starttime >= c.icu_intime       -- ICU 입실 이후 처방만
              AND p.starttime <  c.effective_cutoff  -- 미래 누수 차단
    GROUP BY c.stay_id
    -- LEFT JOIN + COALESCE: 처방 기록 없는 환자도 0행으로 포함
)

SELECT
    cw.stay_id, cw.aki_label, cw.aki_stage,
    cw.icu_intime, cw.effective_cutoff, cw.is_pseudo_cutoff,
    -- NULL → 0 : 처방 기록 없음 = 노출 없음
    COALESCE(r.vancomycin_rx,             0) AS vancomycin_rx,
    COALESCE(r.vancomycin_exposure_hours, 0) AS vancomycin_exposure_hours,
    COALESCE(r.piptazo_rx,                0) AS piptazo_rx,
    COALESCE(r.aminoglycoside_rx,         0) AS aminoglycoside_rx,
    COALESCE(r.amphotericin_b_rx,         0) AS amphotericin_b_rx,
    COALESCE(r.carbapenem_rx,             0) AS carbapenem_rx,
    COALESCE(r.ketorolac_rx,              0) AS ketorolac_rx,
    COALESCE(r.ibuprofen_rx,              0) AS ibuprofen_rx,
    COALESCE(r.nsaid_any_rx,              0) AS nsaid_any_rx,
    COALESCE(r.ace_inhibitor_rx,          0) AS ace_inhibitor_rx,
    COALESCE(r.arb_rx,                    0) AS arb_rx,
    COALESCE(r.acei_arb_any_rx,           0) AS acei_arb_any_rx,
    COALESCE(r.furosemide_rx,             0) AS furosemide_rx,
    COALESCE(r.furosemide_cumulative_mg,  0) AS furosemide_cumulative_mg,
    COALESCE(r.hctz_rx,                   0) AS hctz_rx,
    COALESCE(r.tacrolimus_rx,             0) AS tacrolimus_rx,
    COALESCE(r.cyclosporine_rx,           0) AS cyclosporine_rx,
    COALESCE(r.metformin_rx,              0) AS metformin_rx,
    COALESCE(r.ppi_rx,                    0) AS ppi_rx
FROM aki_project.cdss_cohort_window cw
LEFT JOIN rx_flags r ON cw.stay_id = r.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_icu_rx_stay ON aki_project.cdss_icu_nephrotoxic_rx (stay_id);
-- ANALYZE aki_project.cdss_icu_nephrotoxic_rx;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 1-C  (SCR-03 처방 약물 관리 > + 처방 추가 팝업 + AI 신독성 배너)
-- cdss_nephrotoxic_combo_risk  —  위험 약물 조합 패턴 감지
--
-- ▶ 역할 요약
--   단일 약물 노출보다 더 위험한 "조합 패턴"을 플래그로 산출한다.
--   SCR-03의 "+ 처방 추가" 팝업에서 기존 처방과 신규 약물의 교차 위험을
--   자동 경고하고, 하단 AI 신독성 배너의 위험도 근거 수치로 사용된다.
--
-- ▶ 핵심 조합 임상 근거
--   vanco_piptazo_combo  Vanco+Pip/Tazo → AKI 3.7배↑
--                        기전: pip이 vanco 신장 배설 경쟁 억제 → 혈중 농도↑
--   triple_whammy        NSAIDs+ACEi/ARB+이뇨제 → 신혈류 3중 차단
--                        구심성↑ + 원심성↓ + 혈관 내 용적↓ → GFR≈0
--   diuretic_overload    furosemide>200mg+신독성 항생제
--                        → 탈수로 분포용적 감소 → 항생제 혈중 농도↑
--
-- ▶ 플래그 계산 방식
--   두 플래그 곱셈(×) : 하나라도 0이면 0, 둘 다 1이면 1 (AND 조건 구현)
--   CASE WHEN 3개 조건 : Triple Whammy 등 3요소 조합
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_nephrotoxic_combo_risk CASCADE;

CREATE TABLE aki_project.cdss_nephrotoxic_combo_risk AS
SELECT
    r.stay_id,
    r.aki_label,

    -- ─ 고위험 병용 조합 ────────────────────────────────────────────────
    r.vancomycin_rx * r.piptazo_rx        AS vanco_piptazo_combo,   -- AKI 3.7배
    r.vancomycin_rx * r.aminoglycoside_rx AS vanco_aminogly_combo,  -- 신독성 상가
    r.vancomycin_rx * r.carbapenem_rx     AS vanco_carbapenem_combo, -- 경험적 치료 병용
    r.nsaid_any_rx  * r.acei_arb_any_rx   AS nsaid_acei_combo,      -- 신혈류 이중 감소

    -- Triple Whammy : 세 조건 모두 1이어야 1
    CASE WHEN r.nsaid_any_rx = 1 AND r.acei_arb_any_rx = 1
              AND r.furosemide_rx = 1
         THEN 1 ELSE 0 END                AS triple_whammy,

    -- furosemide > 200 mg + 신독성 항생제 : 탈수로 약물 농도↑
    CASE WHEN r.furosemide_cumulative_mg > 200
              AND (r.vancomycin_rx = 1 OR r.aminoglycoside_rx = 1)
         THEN 1 ELSE 0 END                AS diuretic_overload_flag,

    -- metformin + 신독성 약물 : eGFR 저하 시 젖산산증 위험
    CASE WHEN r.metformin_rx = 1
              AND (r.vancomycin_rx = 1 OR r.nsaid_any_rx = 1)
         THEN 1 ELSE 0 END                AS metformin_risk_flag,

    -- ─ 누적 신독성 부담 점수 (0~8) ────────────────────────────────────
    -- SCR-03 AI 배너: 3점 이상 → "위험도: 높음"
    r.vancomycin_rx + r.aminoglycoside_rx + r.piptazo_rx
    + r.amphotericin_b_rx + r.nsaid_any_rx + r.acei_arb_any_rx
    + r.tacrolimus_rx
    + CASE WHEN r.furosemide_cumulative_mg > 200 THEN 1 ELSE 0 END
                                          AS nephrotoxic_burden_score,

    -- ─ 약물 위험 스코어 (0~5, SCR-06 XGBoost 입력 피처) ──────────────
    -- 단순 종류 수가 아닌 "위험 패턴"의 가중 합산
    (CASE WHEN r.vancomycin_rx = 1 AND r.piptazo_rx = 1       THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_rx + r.aminoglycoside_rx
              + r.amphotericin_b_rx >= 2                       THEN 1 ELSE 0 END)
    + (CASE WHEN r.vancomycin_exposure_hours > 48              THEN 1 ELSE 0 END)
    + (CASE WHEN r.nsaid_any_rx=1 AND r.acei_arb_any_rx=1     THEN 1 ELSE 0 END)
    + (CASE WHEN r.furosemide_cumulative_mg > 200              THEN 1 ELSE 0 END)
                                          AS drug_risk_score,

    -- ICU 내 신독성 약물 종류 수 (master_features total_burden 계산에 사용)
    r.vancomycin_rx + r.aminoglycoside_rx + r.piptazo_rx
    + r.amphotericin_b_rx + r.nsaid_any_rx + r.acei_arb_any_rx
    + r.tacrolimus_rx
    + CASE WHEN r.furosemide_cumulative_mg > 200 THEN 1 ELSE 0 END
                                          AS icu_nephrotoxic_count

FROM aki_project.cdss_icu_nephrotoxic_rx r;

CREATE INDEX IF NOT EXISTS cdss_idx_combo_stay ON aki_project.cdss_nephrotoxic_combo_risk (stay_id);
-- ANALYZE aki_project.cdss_nephrotoxic_combo_risk;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 2-A  (SCR-04 검사 결과 > 날짜 탭 + 결과 테이블 4행)
-- cdss_raw_lab_values  —  혈액검사 원본 추출
--
-- ▶ 역할 요약
--   labevents에서 코호트 환자의 ICU 입실 이후 ~ effective_cutoff 이전 혈액검사를
--   추출한다. itemid 6종으로 제한해 300 M행을 대폭 축소한다.
--
-- ▶ SCR-04 화면 연결
--   charttime의 날짜(DATE)를 기준으로 SCR-04 날짜 탭(04.24·04.25·04.26)을 구성.
--   itemid별 valuenum이 각 행(크레아티닌·BUN·eGFR·헤모글로빈)에 표시된다.
--
-- ▶ itemid 임상 의미
--   50912 크레아티닌 : AKI 판정의 핵심. 0.3 이상 상승 → KDIGO Stage 1
--   51006 BUN        : BUN/Cr 비율로 신전성·신성 AKI 감별
--   50882 중탄산염   : 22 미만 → 대사산증 → 신장 산 배설 능력 저하
--   50971 칼륨       : 5.5 이상 → 고칼륨혈증 → 심실세동 위험
--   51222 헤모글로빈 : 7 미만 → 심각한 빈혈 → 신장 산소 공급 부족
--   50813 젖산       : 2.0 이상 → 조직 저산소 → 허혈성 AKI 의심
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_raw_lab_values CASCADE;

CREATE TABLE aki_project.cdss_raw_lab_values AS
SELECT
    c.stay_id,
    c.effective_cutoff,
    c.aki_label,
    l.charttime,   -- 검사 시행 시각 ← SCR-04 날짜 탭 기준
    l.itemid,      -- 검사 항목 코드
    l.valuenum     -- 검사 수치 (단위는 itemid별로 고정)
FROM aki_project.cdss_cohort_window c
JOIN mimiciv_hosp.labevents l
      ON  l.subject_id = c.subject_id   -- labevents에는 stay_id가 없어 subject_id 사용
      AND l.charttime >= c.icu_intime
      AND l.charttime <= c.effective_cutoff
WHERE l.itemid IN (50912, 51006, 50882, 50971, 51222, 50813)
  AND l.valuenum IS NOT NULL;  -- NULL 검사값 제외 (채혈 실패·용혈 등)

CREATE INDEX IF NOT EXISTS cdss_idx_raw_lab_stay ON aki_project.cdss_raw_lab_values (stay_id, itemid);
-- ANALYZE aki_project.cdss_raw_lab_values;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 2-B  (SCR-04 검사 결과 > 결과 테이블 + 추이 화살표 + AKI 배너)
-- cdss_lab_features  —  혈액검사 항목별 집계
--
-- ▶ 역할 요약
--   원본 행들을 환자 1명·항목 그룹 단위로 집계한다.
--   MIN·MAX·AVG·DELTA를 각각 계산하며 결측 지시변수도 함께 생성한다.
--
-- ▶ SCR-04 화면 연결
--   cr_max, bun_max   → 결과 테이블 "결과값" 열
--   cr_delta          → 0.3 이상이면 빨간 행(높음) 상태 결정의 근거
--   bun_cr_ratio      → AKI 배너 "신전성 AKI 패턴" 문구 트리거
--   cr_missing        → "검사 없음" 상태로 회색 처리
--
-- ▶ LAG 윈도우함수 사용 이유
--   prev_value로 이전 측정값을 보유해야 추이 화살표(↑↓→) 계산이 가능.
--   단, SELECT 집계 단계에서는 LAG 결과가 GROUP BY와 맞지 않아
--   실제 SCR-04 추이는 백엔드(scr04_lab_monitoring.py)에서 계산함.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_lab_features CASCADE;

CREATE TABLE aki_project.cdss_lab_features AS

WITH ranked AS (
    SELECT *,
        -- 직전 측정값. 백엔드 calculate_trend_direction_from_sequential_values()가
        -- ↑↓→ 화살표 결정에 사용하나, 이 CTE 결과는 직접 SELECT에서는 집계 불가.
        LAG(valuenum) OVER (
            PARTITION BY stay_id, itemid
            ORDER BY charttime
        ) AS prev_value
    FROM aki_project.cdss_raw_lab_values
)
SELECT
    stay_id,
    aki_label,

    -- ─ 크레아티닌 (SCR-04 첫 번째 행) ─────────────────────────────────
    MIN(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_min,
    MAX(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_max,   -- 빨강 행 기준: >1.5
    AVG(CASE WHEN itemid = 50912 THEN valuenum END) AS cr_mean,
    -- cr_delta = 최댓값 - 최솟값. KDIGO 0.3 이상 → AKI Stage 1 판정 가능.
    MAX(CASE WHEN itemid = 50912 THEN valuenum END)
    - MIN(CASE WHEN itemid = 50912 THEN valuenum END)            AS cr_delta,

    -- ─ BUN (SCR-04 두 번째 행, 정상 7~20) ──────────────────────────────
    MAX(CASE WHEN itemid = 51006 THEN valuenum END) AS bun_max,  -- 빨강 기준: >30
    AVG(CASE WHEN itemid = 51006 THEN valuenum END) AS bun_mean,
    -- BUN/Cr 비율: >20 신전성 AKI(혈류 부족), <10 신성 AKI(실질 손상)
    CASE WHEN AVG(CASE WHEN itemid = 50912 THEN valuenum END) > 0
         THEN AVG(CASE WHEN itemid = 51006 THEN valuenum END)
            / AVG(CASE WHEN itemid = 50912 THEN valuenum END)
         ELSE NULL
    END                                                          AS bun_cr_ratio,

    -- ─ 칼륨 ─────────────────────────────────────────────────────────────
    MAX(CASE WHEN itemid = 50971 THEN valuenum END) AS potassium_max,  -- >5.5 고칼륨
    AVG(CASE WHEN itemid = 50971 THEN valuenum END) AS potassium_mean,

    -- ─ 중탄산염 ──────────────────────────────────────────────────────────
    MIN(CASE WHEN itemid = 50882 THEN valuenum END) AS bicarbonate_min, -- <22 산증
    AVG(CASE WHEN itemid = 50882 THEN valuenum END) AS bicarbonate_mean,

    -- ─ 헤모글로빈 (SCR-04 네 번째 행, 정상 12~16 g/dL) ──────────────────
    MIN(CASE WHEN itemid = 51222 THEN valuenum END) AS hemoglobin_min,  -- <7 심각
    AVG(CASE WHEN itemid = 51222 THEN valuenum END) AS hemoglobin_mean,

    -- ─ 젖산 (SCR-05 허혈 지표, SCR-04 배너에서도 참조) ──────────────────
    MAX(CASE WHEN itemid = 50813 THEN valuenum END) AS lactate_max,   -- >2.0 저산소
    AVG(CASE WHEN itemid = 50813 THEN valuenum END) AS lactate_mean,

    -- ─ 결측 지시변수 ──────────────────────────────────────────────────────
    -- "측정 안 함" 자체가 임상 정보. 젖산은 쇼크 의심 시에만 측정하므로
    -- lactate_missing=1이 오히려 "안정 상태"를 의미하기도 함.
    CASE WHEN COUNT(CASE WHEN itemid = 50912 THEN 1 END) = 0
         THEN 1 ELSE 0 END                                       AS cr_missing,
    CASE WHEN COUNT(CASE WHEN itemid = 50813 THEN 1 END) = 0
         THEN 1 ELSE 0 END                                       AS lactate_missing

FROM aki_project.cdss_raw_lab_values
GROUP BY stay_id, aki_label;

CREATE INDEX IF NOT EXISTS cdss_idx_lab_stay ON aki_project.cdss_lab_features (stay_id);
-- ANALYZE aki_project.cdss_lab_features;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-A  (SCR-05 카디오 필터 > 좌·우 카드 공통 원본)
-- cdss_raw_map_values  —  MAP 원본 추출
--
-- ▶ 역할 요약
--   chartevents에서 MAP 측정값을 추출한다. 이 테이블을 기반으로
--   STEP 3-B(허혈 시간)·3-C(현재 MAP)·3-D(Shock Index)가 파생된다.
--
-- ▶ SCR-05 화면 연결
--   cdss_feat_map_ischemia.map_below65_hours → 좌측 카드 "총 허혈 시간"
--   cdss_feat_map_summary.current_map        → 우측 카드 "현재 MAP"
--
-- ▶ itemid 선택 이유
--   220052 침습적 동맥라인 MAP : 가장 정확하나 모든 환자에게 없음
--   220181 비침습적(커프) MAP  : 간헐적이지만 더 보편적
--   225312 수술실 연속 MAP     : 수술 후 입실 환자 포함
--   → 세 itemid를 합산해 MAP 커버리지를 최대화
--
-- ▶ 이상치 필터 기준
--   MAP 20 미만 : 심정지 수준으로 측정 오류 가능성 높음
--   MAP 300 이상: 기기 오류 (생리적으로 불가능)
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_raw_map_values CASCADE;

CREATE TABLE aki_project.cdss_raw_map_values AS
SELECT
    c.stay_id,
    c.effective_cutoff,
    ce.charttime,
    ce.valuenum AS map
FROM aki_project.cdss_cohort_window c
JOIN mimiciv_icu.chartevents ce
      ON  ce.stay_id   = c.stay_id
      AND ce.charttime >= c.icu_intime
      AND ce.charttime <= c.effective_cutoff
WHERE ce.itemid IN (220052, 220181, 225312)
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum BETWEEN 20 AND 300;

CREATE INDEX IF NOT EXISTS cdss_idx_raw_map_stay_time
    ON aki_project.cdss_raw_map_values (stay_id, charttime);
-- ANALYZE aki_project.cdss_raw_map_values;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-B  (SCR-05 카디오 필터 > 좌측 카드 "총 허혈 시간")
-- cdss_feat_map_ischemia  —  MAP < 65 mmHg 지속 시간 누적 계산
--
-- ▶ 역할 요약
--   MAP < 65 mmHg였던 구간의 시간을 모두 합산한다.
--   단순 횟수가 아닌 지속 시간을 쓰는 이유:
--   1분짜리 MAP=64와 2시간짜리 MAP=50은 신장 손상 정도가 전혀 다르기 때문.
--
-- ▶ SCR-05 화면 연결 (좌측 카드)
--   map_below65_hours × 60 → "총 허혈 시간 : 145 분" 대형 숫자
--   flag_ischemia_over120min = 1 → 카드 테두리 빨간색 + "권장(120분) 초과" 경고
--
-- ▶ SCR-06 연결
--   flag_ischemia_over120min = 1 → 위험 요인 "허혈 시간" +15점 트리거
--
-- ▶ 구간 면적법 계산 원리
--   현재 MAP < 65이면, 다음 측정 시각까지 허혈 상태가 지속된 것으로 간주.
--   1시간 캡을 두는 이유: 측정 누락이 길어도 무한 허혈로 집계되는 오류 방지.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_feat_map_ischemia CASCADE;

CREATE TABLE aki_project.cdss_feat_map_ischemia AS

WITH time_ordered AS (
    SELECT
        stay_id,
        effective_cutoff,
        charttime,
        map,
        -- 다음 측정 시각. 마지막 행은 NULL → COALESCE로 cutoff 사용.
        LEAD(charttime) OVER (PARTITION BY stay_id ORDER BY charttime) AS next_charttime
    FROM aki_project.cdss_raw_map_values
),
ischemia_duration AS (
    SELECT
        stay_id,
        CASE
            WHEN map < 65 THEN
                EXTRACT(EPOCH FROM (
                    LEAST(
                        COALESCE(next_charttime, effective_cutoff), -- 다음 측정 없으면 cutoff
                        effective_cutoff,                           -- cutoff를 넘어가면 잘라냄
                        charttime + INTERVAL '1 hour'               -- 측정 간격 1h 이상이면 캡
                    ) - charttime
                )) / 3600.0   -- 초 → 시간
            ELSE 0            -- MAP >= 65 : 허혈 없음
        END AS ischemia_hours
    FROM time_ordered
)
SELECT
    stay_id,
    SUM(ischemia_hours)                              AS map_below65_hours,
    CASE WHEN SUM(ischemia_hours) > 2 THEN 1 ELSE 0 END
                                                     AS flag_ischemia_over120min
FROM ischemia_duration
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_map_isch_stay ON aki_project.cdss_feat_map_ischemia (stay_id);
-- ANALYZE aki_project.cdss_feat_map_ischemia;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-C  (SCR-05 카디오 필터 > 우측 카드 "현재 MAP")
-- cdss_feat_map_summary  —  MAP 집계 + 최근 측정값
--
-- ▶ 역할 요약
--   전체 구간의 MAP 최솟값·평균과 가장 최근 측정값을 산출한다.
--   current_map이 SCR-05 우측 카드에 대형 숫자로 표시된다.
--
-- ▶ Bug Fix : LAST_VALUE + GROUP BY 동시 사용 불가 (PostgreSQL 에러)
--   해결: CTE 두 개로 분리 — agg(집계) + latest(최근값) — 후 JOIN
--   DISTINCT ON (stay_id) ORDER BY charttime DESC → 가장 최근 1행만 추출
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_feat_map_summary CASCADE;

CREATE TABLE aki_project.cdss_feat_map_summary AS

WITH agg AS (
    SELECT
        stay_id,
        MIN(map) AS map_min,   -- 구간 내 최솟값 → SCR-05 목표치 미달 판단
        AVG(map) AS map_mean   -- 구간 내 평균 → SCR-06 허혈 정도 반영
    FROM aki_project.cdss_raw_map_values
    GROUP BY stay_id
),
latest AS (
    -- DISTINCT ON: stay_id별로 ORDER BY 기준 첫 번째 행만 반환 (PostgreSQL 전용)
    -- charttime DESC → 가장 최근 시각이 첫 번째
    SELECT DISTINCT ON (stay_id)
        stay_id,
        map AS current_map   -- SCR-05 우측 카드 "현재 MAP : 62 mmHg"
    FROM aki_project.cdss_raw_map_values
    ORDER BY stay_id, charttime DESC
)
SELECT a.stay_id, a.map_min, a.map_mean, l.current_map
FROM agg a JOIN latest l ON a.stay_id = l.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_map_sum_stay ON aki_project.cdss_feat_map_summary (stay_id);
-- ANALYZE aki_project.cdss_feat_map_summary;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-D  (SCR-05 카디오 필터 > AI IRI 배너 보조 지표)
-- cdss_feat_shock_index  —  Shock Index 계산
--
-- ▶ 역할 요약
--   Shock Index = 심박수(HR) / 수축기혈압(SBP).
--   1.0 이상이면 쇼크 의심 → 신혈류 감소 → 허혈성 AKI 위험.
--   단독으로 사용하기보다 MAP·승압제와 조합해 cardio_risk_score를 구성.
--
-- ▶ 이상치 필터
--   HR  20~250 bpm 범위 밖 → 기기 오류
--   SBP 40~300 mmHg 범위 밖 → 측정 오류
--   NULLIF(sbp, 0) : SBP=0이면 NULL 반환 → 0 나누기 에러 방지
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_feat_shock_index CASCADE;

CREATE TABLE aki_project.cdss_feat_shock_index AS
WITH hr_sbp AS (
    SELECT
        c.stay_id,
        ce.charttime,
        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)            AS hr,
        AVG(CASE WHEN ce.itemid IN (220179, 220050) THEN ce.valuenum END) AS sbp
    FROM aki_project.cdss_cohort_window c
    JOIN mimiciv_icu.chartevents ce
          ON  ce.stay_id   = c.stay_id
          AND ce.charttime >= c.icu_intime
          AND ce.charttime <= c.effective_cutoff
    WHERE ce.itemid IN (220045, 220179, 220050)
      AND (   (ce.itemid = 220045 AND ce.valuenum BETWEEN 20  AND 250)
           OR (ce.itemid IN (220179, 220050) AND ce.valuenum BETWEEN 40 AND 300))
    GROUP BY c.stay_id, ce.charttime
)
SELECT
    stay_id,
    AVG(hr / NULLIF(sbp, 0)) AS shock_index   -- 시점별 SI 평균으로 경향 반영
FROM hr_sbp
WHERE hr IS NOT NULL AND sbp IS NOT NULL
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_shock_stay ON aki_project.cdss_feat_shock_index (stay_id);
-- ANALYZE aki_project.cdss_feat_shock_index;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-E  (SCR-05 카디오 필터 > AI IRI 배너 경고 트리거)
-- cdss_feat_vasopressor  —  승압제 사용 여부
--
-- ▶ 역할 요약
--   ICU 재원 중 승압제를 한 번이라도 사용했으면 1.
--   혈압을 스스로 유지 못해 약물로 보조하는 상태이므로
--   신관류압 불안정 → AKI 고위험 신호다.
--
-- ▶ SCR-05 화면 연결
--   vasopressor_flag = 1 → AI IRI 배너 cardio_risk_score 1점 추가
--   → "카디오 필터 프로토콜 즉시 적용 권고" 배너 활성화 기준
--
-- ▶ 단위 오류 필터
--   phenylephrine (221749) 에서 'mg/min' 단위로 잘못 기록된 건 제외
--   vasopressin (222315) 에서 'units/min' 잘못 기록된 건 제외
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_feat_vasopressor CASCADE;

CREATE TABLE aki_project.cdss_feat_vasopressor AS
SELECT
    a.stay_id,
    CASE WHEN COUNT(rv.stay_id) > 0 THEN 1 ELSE 0 END AS vasopressor_flag
FROM aki_project.cdss_cohort_window a
LEFT JOIN (
    SELECT c.stay_id
    FROM aki_project.cdss_cohort_window c
    JOIN mimiciv_icu.inputevents ie
          ON  ie.stay_id   = c.stay_id
          AND ie.starttime >= c.icu_intime
          AND ie.starttime <= c.effective_cutoff
    WHERE ie.itemid IN (
        221906,  -- Norepinephrine : α1 작용, 1차 선택 승압제
        221289,  -- Epinephrine    : α+β, 심정지·아나필락시스
        221749,  -- Phenylephrine  : 순수 α1, 심박수 올리지 않음
        222315,  -- Vasopressin    : V1 수용체, 패혈증 추가 요법
        221662,  -- Dopamine       : 저용량 신혈류↑, 고용량 혈관수축
        221653   -- Dobutamine     : β1, 심박출량 증가
    )
    AND ie.rate IS NOT NULL
    AND NOT (ie.itemid = 221749 AND ie.rateuom = 'mg/min')
    AND NOT (ie.itemid = 222315 AND ie.rateuom = 'units/min')
) rv ON a.stay_id = rv.stay_id
GROUP BY a.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_vaso_stay ON aki_project.cdss_feat_vasopressor (stay_id);
-- ANALYZE aki_project.cdss_feat_vasopressor;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 3-F  (SCR-05 카디오 필터 > 전체 화면)
-- cdss_ischemic_features  —  허혈성 피처 통합
--
-- ▶ 역할 요약
--   MAP·허혈시간·Shock Index·승압제·젖산·헤모글로빈을 1행으로 통합.
--   SCR-05 전체 화면 데이터의 단일 소스이자 SCR-06 score_ischemia·score_map의 원천.
--
-- ▶ 결측 지시변수 의미
--   map_missing=1        : MAP 측정 기록 자체가 없음 → 모니터링 미실시
--   shock_index_missing=1: HR 또는 SBP 없음 → Shock Index 계산 불가
--   isch_lactate_missing=1: 젖산 미측정 → 안정 상태일 수도, 처방 누락일 수도
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_ischemic_features CASCADE;

CREATE TABLE aki_project.cdss_ischemic_features AS
SELECT
    cw.stay_id, cw.aki_label, cw.aki_stage,
    cw.prediction_cutoff, cw.hours_to_aki,
    ms.current_map,                                        -- SCR-05 우측 카드
    ms.map_mean             AS isch_map_mean,
    ms.map_min              AS isch_map_min,
    COALESCE(mi.map_below65_hours, 0)         AS map_below65_hours,        -- SCR-05 좌측 카드
    COALESCE(mi.flag_ischemia_over120min, 0)  AS flag_ischemia_over120min, -- SCR-06 +15점 기준
    si.shock_index          AS isch_shock_index,           -- SCR-05 보조 지표
    COALESCE(vp.vasopressor_flag, 0)          AS vasopressor_flag,         -- SCR-05 IRI 배너
    lf.lactate_max          AS isch_lactate_max,
    lf.hemoglobin_min       AS isch_hemo_min,
    CASE WHEN ms.map_min     IS NULL THEN 1 ELSE 0 END AS map_missing,
    CASE WHEN si.shock_index IS NULL THEN 1 ELSE 0 END AS shock_index_missing,
    CASE WHEN lf.lactate_max IS NULL THEN 1 ELSE 0 END AS isch_lactate_missing,
    CASE WHEN lf.hemoglobin_min IS NULL THEN 1 ELSE 0 END AS hemoglobin_missing
FROM aki_project.cdss_cohort_window           cw
LEFT JOIN aki_project.cdss_feat_map_summary   ms ON cw.stay_id = ms.stay_id
LEFT JOIN aki_project.cdss_feat_map_ischemia  mi ON cw.stay_id = mi.stay_id
LEFT JOIN aki_project.cdss_feat_shock_index   si ON cw.stay_id = si.stay_id
LEFT JOIN aki_project.cdss_feat_vasopressor   vp ON cw.stay_id = vp.stay_id
LEFT JOIN aki_project.cdss_lab_features       lf ON cw.stay_id = lf.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_isch_stay ON aki_project.cdss_ischemic_features (stay_id);
-- ANALYZE aki_project.cdss_ischemic_features;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 4  (SCR-06 AI 급성 신손상 예측도 > 대형 숫자 + 위험 요인 테이블 5행)
-- cdss_rule_score_features  —  규칙 기반 위험도 점수 계산
--
-- ▶ 역할 요약
--   5개 임상 지표의 임계값 초과 여부를 플래그화하고 기여 점수를 합산한다.
--   XGBoost 모델이 없을 때 이 점수만으로도 SCR-06 화면을 운영할 수 있고,
--   XGBoost가 있으면 rule_based_score가 모델 입력 피처의 하나로 사용된다.
--
-- ▶ SCR-06 화면 연결
--   rule_based_score   → 대형 숫자 박스 기본값 (XGBoost 있으면 보정됨)
--   flag_* / score_*   → 위험 요인 테이블 5행의 "상태"·"기여도" 열
--   high_risk_flag = 1 → 박스 배경색 빨간색 + 상단 "고위험" 배너
--
-- ▶ 점수 배분 근거 (합계 100점)
--   크레아티닌 > 1.5  +30점 : AKI의 직접 생화학 지표, 가장 중요
--   BUN > 30          +20점 : 신기능 저하 누적 지표
--   eGFR < 45         +20점 : 신기능 저하 정도
--   허혈 시간 > 120분 +15점 : 허혈성 손상 위험
--   MAP < 65          +15점 : 현재 신관류압 부족
--   합계 ≥ 70점       → 고위험 (high_risk_flag = 1)
--
-- ▶ eGFR CKD-EPI 2021 공식 (SQL 내 계산)
--   여성 (κ=0.7) : 142 × min(Cr/0.7,1)^-0.241 × max(Cr/0.7,1)^-1.200
--                  × 0.9938^age × 1.012
--   남성 (κ=0.9) : 142 × min(Cr/0.9,1)^-0.302 × max(Cr/0.9,1)^-1.200
--                  × 0.9938^age
--   ※ 백엔드(scr04_lab_monitoring.py, scr06_ai_risk_score.py)와
--      동일한 공식을 사용해야 값이 일치한다.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_rule_score_features CASCADE;

CREATE TABLE aki_project.cdss_rule_score_features AS

WITH
egfr AS (
    -- CKD-EPI 2021 공식 (성별·연령·크레아티닌 필요)
    -- cr_mean 우선, 없으면 cr_min 사용 (평균이 더 대표값에 가까움)
    SELECT
        cw.stay_id,
        CASE
            WHEN cw.gender = 'F' THEN
                CASE
                    WHEN COALESCE(l.cr_mean, l.cr_min) <= 0.7
                    THEN 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.7, -0.241)
                         * POWER(0.9938, cw.age) * 1.012
                    ELSE 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.7, -1.200)
                         * POWER(0.9938, cw.age) * 1.012
                END
            ELSE  -- 남성
                CASE
                    WHEN COALESCE(l.cr_mean, l.cr_min) <= 0.9
                    THEN 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.9, -0.302)
                         * POWER(0.9938, cw.age)
                    ELSE 142.0 * POWER(COALESCE(l.cr_mean,l.cr_min)/0.9, -1.200)
                         * POWER(0.9938, cw.age)
                END
        END AS egfr_ckdepi
    FROM aki_project.cdss_cohort_window cw
    JOIN aki_project.cdss_lab_features  l   ON cw.stay_id = l.stay_id
    WHERE COALESCE(l.cr_mean, l.cr_min) IS NOT NULL
      AND COALESCE(l.cr_mean, l.cr_min) > 0   -- 0이면 측정 오류
),

threshold_flags AS (
    -- 임계값 초과 여부 + 현재 수치 추출
    -- SCR-06 테이블 "현재값" 열 = val_*, "상태" 열 = flag_*
    SELECT
        cw.stay_id,
        CASE WHEN COALESCE(l.cr_max, l.cr_mean) > 1.5  THEN 1 ELSE 0 END AS flag_cr,
        COALESCE(l.cr_max, l.cr_mean)                   AS val_cr,
        CASE WHEN l.bun_max > 30                        THEN 1 ELSE 0 END AS flag_bun,
        l.bun_max                                       AS val_bun,
        CASE WHEN e.egfr_ckdepi < 45                    THEN 1 ELSE 0 END AS flag_egfr,
        e.egfr_ckdepi                                   AS val_egfr,
        -- map_below65_hours(시간) × 60 → 분으로 변환해 SCR-05/06 화면에 표시
        CASE WHEN COALESCE(isch.map_below65_hours,0) > 2 THEN 1 ELSE 0 END AS flag_ischemia,
        COALESCE(isch.map_below65_hours, 0) * 60        AS val_ischemia_min,
        CASE WHEN COALESCE(isch.isch_map_min,999) < 65  THEN 1 ELSE 0 END AS flag_map,
        isch.isch_map_min                               AS val_map
    FROM aki_project.cdss_cohort_window    cw
    LEFT JOIN aki_project.cdss_lab_features         l    ON cw.stay_id = l.stay_id
    LEFT JOIN egfr                          e    ON cw.stay_id = e.stay_id
    LEFT JOIN aki_project.cdss_ischemic_features    isch ON cw.stay_id = isch.stay_id
)

SELECT
    cw.stay_id,
    cw.aki_label,
    e.egfr_ckdepi,              -- SCR-04 세 번째 행, SCR-06 위험 요인 3
    tf.val_cr, tf.val_bun, tf.val_egfr, tf.val_ischemia_min, tf.val_map,
    tf.flag_cr, tf.flag_bun, tf.flag_egfr, tf.flag_ischemia, tf.flag_map,
    tf.flag_cr       * 30 AS score_cr,       -- +30점
    tf.flag_bun      * 20 AS score_bun,      -- +20점
    tf.flag_egfr     * 20 AS score_egfr,     -- +20점
    tf.flag_ischemia * 15 AS score_ischemia, -- +15점
    tf.flag_map      * 15 AS score_map,      -- +15점
    -- 규칙 기반 총점 (SCR-06 대형 숫자, XGBoost 없을 때 표시)
    tf.flag_cr*30 + tf.flag_bun*20 + tf.flag_egfr*20
    + tf.flag_ischemia*15 + tf.flag_map*15               AS rule_based_score,
    -- 70점 이상 → 빨간 배경 + "고위험" 상단 배너
    CASE WHEN (tf.flag_cr*30+tf.flag_bun*20+tf.flag_egfr*20
               +tf.flag_ischemia*15+tf.flag_map*15) >= 70
         THEN 1 ELSE 0 END                               AS high_risk_flag
FROM aki_project.cdss_cohort_window  cw
LEFT JOIN egfr            e   ON cw.stay_id = e.stay_id
LEFT JOIN threshold_flags tf  ON cw.stay_id = tf.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_rule_score_stay ON aki_project.cdss_rule_score_features (stay_id);
-- ANALYZE aki_project.cdss_rule_score_features;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 5  (SCR-07 AKI 위험도 시계열 > 전체 + XGBoost 학습 단일 입력)
-- cdss_master_features  —  모든 STEP 피처 통합 마스터 테이블
--
-- ▶ 역할 요약
--   모든 STEP 결과를 LEFT JOIN으로 통합해 환자 1명당 1행으로 구성한다.
--   XGBoost 학습(xgboost_pipeline.py)과 추론(scr06_ai_risk_score.py)의
--   단일 입력 소스이자, 각 화면 API의 기본 조회 테이블이다.
--
-- ▶ LEFT JOIN 사용 이유
--   특정 STEP 피처가 NULL이어도 환자가 제외되지 않아야 한다.
--   예: MAP 측정이 없는 환자 → ischemic_features가 NULL이어도 약물 피처는 보존.
--
-- ▶ hospital_expire_flag·competed_with_death 포함 이유
--   백엔드 main.py의 환자 검색 API와 PatientBase 모델에서 이 필드를 참조.
--   competed_with_death=1이면 SCR-06 화면에 "AI 점수 해석 주의" 경고를 표시.
--
-- ▶ total_nephrotoxic_burden 설계
--   약물 부담 + 임상 이상 지표(Cr·eGFR·승압제)를 가중 합산.
--   Cr·eGFR 이상에 ×2 가중치: 신기능 직접 지표이므로 더 중요.
--   SCR-07 시계열 배너의 "종합 위험 지수" 문구에 사용.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_master_features CASCADE;

CREATE TABLE aki_project.cdss_master_features AS
SELECT
    -- ── 식별자·메타 ──────────────────────────────────────────────────────
    cw.stay_id, cw.subject_id, cw.hadm_id,
    cw.age, cw.gender, cw.first_careunit,
    cw.icu_intime, cw.icu_outtime, cw.icu_los_hours,
    cw.aki_label, cw.aki_stage,
    cw.prediction_cutoff, cw.effective_cutoff,
    cw.hours_to_aki, cw.is_pseudo_cutoff,
    -- 사망 관련 (PatientBase·환자 검색 API에서 참조)
    cw.hospital_expire_flag, cw.competed_with_death, cw.hours_to_death,

    -- ══ SCR-03 약물 피처 ════════════════════════════════════════════════
    rx.vancomycin_rx, rx.vancomycin_exposure_hours, rx.piptazo_rx,
    rx.aminoglycoside_rx, rx.amphotericin_b_rx, rx.carbapenem_rx,
    rx.ketorolac_rx, rx.nsaid_any_rx,
    rx.ace_inhibitor_rx, rx.arb_rx, rx.acei_arb_any_rx,
    rx.furosemide_rx, rx.furosemide_cumulative_mg,
    rx.tacrolimus_rx, rx.cyclosporine_rx, rx.metformin_rx, rx.ppi_rx,
    cb.vanco_piptazo_combo, cb.vanco_aminogly_combo, cb.vanco_carbapenem_combo,
    cb.nsaid_acei_combo, cb.triple_whammy, cb.diuretic_overload_flag,
    cb.nephrotoxic_burden_score, cb.drug_risk_score,

    -- ══ SCR-04 혈액검사 피처 ════════════════════════════════════════════
    lf.cr_min, lf.cr_max, lf.cr_mean, lf.cr_delta,
    lf.bun_max, lf.bun_mean, lf.bun_cr_ratio,
    lf.potassium_max, lf.potassium_mean,
    lf.bicarbonate_min, lf.bicarbonate_mean,
    lf.hemoglobin_min, lf.hemoglobin_mean,
    lf.lactate_max, lf.lactate_mean,
    lf.cr_missing, lf.lactate_missing,
    rs.egfr_ckdepi,          -- SCR-04 세 번째 행

    -- ══ SCR-05 허혈성 피처 ══════════════════════════════════════════════
    isch.current_map, isch.isch_map_mean, isch.isch_map_min,
    isch.map_below65_hours,        -- SCR-05 좌측 카드 (시간)
    isch.flag_ischemia_over120min, -- SCR-06 허혈 +15점 기준
    isch.isch_shock_index, isch.vasopressor_flag,
    isch.isch_lactate_max, isch.isch_hemo_min,
    isch.map_missing, isch.shock_index_missing,
    isch.isch_lactate_missing, isch.hemoglobin_missing,

    -- ══ SCR-06 규칙 기반 점수 ═══════════════════════════════════════════
    rs.val_cr, rs.val_bun, rs.val_egfr, rs.val_ischemia_min, rs.val_map,
    rs.flag_cr, rs.flag_bun, rs.flag_egfr, rs.flag_ischemia, rs.flag_map,
    rs.score_cr, rs.score_bun, rs.score_egfr, rs.score_ischemia, rs.score_map,
    rs.rule_based_score,   -- SCR-06 대형 숫자 기본값
    rs.high_risk_flag,     -- SCR-06 빨간 배경 기준

    -- ══ SCR-07 종합 신독성 부담 지수 ════════════════════════════════════
    COALESCE(cb.nephrotoxic_burden_score, 0)
    + COALESCE(rs.flag_cr,   0) * 2   -- Cr 이상 : 직접 지표라 ×2 가중
    + COALESCE(rs.flag_egfr, 0) * 2   -- eGFR 이상: 직접 지표라 ×2 가중
    + COALESCE(isch.vasopressor_flag, 0)
                                                  AS total_nephrotoxic_burden

FROM aki_project.cdss_cohort_window               cw
LEFT JOIN aki_project.cdss_icu_nephrotoxic_rx     rx   ON cw.stay_id = rx.stay_id
LEFT JOIN aki_project.cdss_nephrotoxic_combo_risk cb   ON cw.stay_id = cb.stay_id
LEFT JOIN aki_project.cdss_lab_features           lf   ON cw.stay_id = lf.stay_id
LEFT JOIN aki_project.cdss_ischemic_features      isch ON cw.stay_id = isch.stay_id
LEFT JOIN aki_project.cdss_rule_score_features    rs   ON cw.stay_id = rs.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_master_stay     ON aki_project.cdss_master_features (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_master_label    ON aki_project.cdss_master_features (aki_label);
CREATE INDEX IF NOT EXISTS cdss_idx_master_highrisk ON aki_project.cdss_master_features (high_risk_flag);
-- ANALYZE aki_project.cdss_master_features;



-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6 (Track D) : 방사선 NLP 피처 추출 및 master_features 통합
--
-- ▶ 선행 조건
--   이 STEP 실행 전에 반드시 NLP CSV를 DB에 임포트해야 한다.
--   실행: python kidney_nlp/import_nlp_csv.py
--
-- ▶ 처리 흐름
--   스테이징(stg_nlp_*)  →  STEP 6-A cdss_nlp_keyword_raw
--                        →  STEP 6-B cdss_nlp_radiology_extra
--                        →  STEP 6-C cdss_nlp_features (병합)
--                        →  STEP 6-D cdss_master_features 컬럼 추가
-- ═══════════════════════════════════════════════════════════════════════════

-- STEP 6-A : nlp_keyword_features.csv 스테이징 임포트
--
-- ▶ 역할
--   Python(pandas)으로 사전 처리된 NLP 키워드 CSV를 PostgreSQL에 적재한다.
--   이 테이블은 원본 보존용으로, 필터링은 STEP 1에서 수행한다.
--
-- ▶ 실행 방법 (psql 또는 Python psycopg2)
--   psql 방식:
--     \COPY aki_project.stg_nlp_keyword_features
--       FROM 'kidney_nlp/nlp_keyword_features.csv'
--       CSV HEADER ENCODING 'UTF8';
--
--   Python 방식 (권장 — 경로 유연성):
--     python kidney_nlp/import_nlp_csv.py   ← 별도 임포트 스크립트
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.stg_nlp_keyword_features CASCADE;

CREATE TABLE aki_project.stg_nlp_keyword_features (
    stay_id             BIGINT      PRIMARY KEY,
    nlp_text_combined   TEXT,                   -- 판독문 원문 (findings+impression 결합)

    -- ── 사전 추출 키워드 플래그 (0/1) ──────────────────────────────────
    -- 출처: nlp_keyword_features.csv (Python NLP 전처리 결과)
    kw_oliguria         SMALLINT    DEFAULT 0,  -- 핍뇨 (소변량 감소) → AKI 직접 지표
    kw_anuria           SMALLINT    DEFAULT 0,  -- 무뇨 (소변 없음)   → AKI 심각 지표
    kw_edema            SMALLINT    DEFAULT 0,  -- 부종 (체액 저류)   → 신기능 저하 신호
    kw_hydronephrosis   SMALLINT    DEFAULT 0,  -- 수신증 (폐색성 AKI) → 신후성 AKI 원인
    kw_aki_mention      SMALLINT    DEFAULT 0,  -- AKI 직접 언급      → 임상의 판단 포함
    kw_renal_abnormal   SMALLINT    DEFAULT 0,  -- 신장 이상 소견     → 구조적 문제
    kw_fluid_overload   SMALLINT    DEFAULT 0,  -- 수액 과부하        → 신기능 저하 상태

    -- 메타
    imported_at         TIMESTAMPTZ DEFAULT NOW()
);

-- CSV 임포트 명령 (psql에서 실행):
-- \COPY aki_project.stg_nlp_keyword_features
--   (stay_id, nlp_text_combined,
--    kw_oliguria, kw_anuria, kw_edema,
--    kw_hydronephrosis, kw_aki_mention, kw_renal_abnormal, kw_fluid_overload)
-- FROM 'kidney_nlp/nlp_keyword_features.csv'
-- CSV HEADER ENCODING 'UTF8';

COMMENT ON TABLE aki_project.stg_nlp_keyword_features IS
    'Track D 스테이징: nlp_keyword_features.csv 원본 적재. 27,049행.';


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6-B : radiology_nlp_text.csv 스테이징 임포트
--
-- ▶ 역할
--   방사선 판독 원문(findings·impression)을 PostgreSQL에 적재한다.
--   원문 텍스트에서 STEP 2에서 추가 키워드를 정규표현식으로 추출한다.
--
-- ▶ 파일 규모
--   전체 약 35 MB. psql \COPY가 Python 루프보다 10배 이상 빠름.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.stg_radiology_nlp_text CASCADE;

CREATE TABLE aki_project.stg_radiology_nlp_text (
    note_id     TEXT        NOT NULL,           -- 리포트 고유 ID (예: 12466550-RR-26)
    subject_id  BIGINT      NOT NULL,
    hadm_id     BIGINT,
    stay_id     BIGINT,                         -- NULL인 경우 있음 (ICU 연결 전 note)
    charttime   TIMESTAMPTZ,                    -- 리포트 작성 시각
    findings    TEXT,                           -- 방사선 소견 (주요 서술 부분)
    impression  TEXT,                           -- 최종 인상 (핵심 요약, 23.8% NULL)
    imported_at TIMESTAMPTZ DEFAULT NOW()
);

-- CSV 임포트 명령 (psql에서 실행):
-- \COPY aki_project.stg_radiology_nlp_text
--   (note_id, subject_id, hadm_id, stay_id, charttime, findings, impression)
-- FROM 'kidney_nlp/radiology_nlp_text.csv'
-- CSV HEADER ENCODING 'UTF8'
-- NULL '';

-- 조회 성능을 위한 인덱스
CREATE INDEX IF NOT EXISTS cdss_idx_rad_stay    ON aki_project.stg_radiology_nlp_text (stay_id);
CREATE INDEX IF NOT EXISTS cdss_idx_rad_subject ON aki_project.stg_radiology_nlp_text (subject_id);
CREATE INDEX IF NOT EXISTS cdss_idx_rad_time    ON aki_project.stg_radiology_nlp_text (charttime);
-- ANALYZE aki_project.stg_radiology_nlp_text;

COMMENT ON TABLE aki_project.stg_radiology_nlp_text IS
    'Track D 스테이징: radiology_nlp_text.csv 원본 적재. findings+impression 원문.';


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6-C : 스테이징 → cdss_nlp_keyword_raw
--           코호트 필터링 + effective_cutoff 기준 데이터 누수 차단
--
-- ▶ effective_cutoff 필터 이유
--   NLP 키워드는 방사선 리포트 시각(charttime) 기반이다.
--   prediction_cutoff 이후에 작성된 리포트의 키워드는 미래 정보 → 누수.
--   stg_nlp_keyword_features에는 charttime이 없으므로 stay_id 기준으로
--   코호트 필터만 적용하고, 원문 필터는 STEP 2에서 수행한다.
--
-- ▶ 결측 처리
--   NLP 데이터가 없는 환자(38.6%): 모든 kw_ 컬럼을 0으로 채워
--   피처 결측이 XGBoost 학습에 영향을 주지 않게 한다.
--   nlp_available 지시변수로 "NLP 데이터 없음" 패턴을 모델이 학습할 수 있게 한다.
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_nlp_keyword_raw CASCADE;

CREATE TABLE aki_project.cdss_nlp_keyword_raw AS
SELECT
    -- 코호트 기준 컬럼 (LEFT JOIN 기준)
    cw.stay_id,
    cw.effective_cutoff,
    cw.aki_label,

    -- ── 사전 추출 키워드 (stg_nlp_keyword_features 원본) ───────────────
    -- COALESCE: NLP 데이터 없는 환자는 0 (미관찰 = 키워드 없음으로 처리)
    COALESCE(n.kw_oliguria,       0) AS kw_oliguria,
    COALESCE(n.kw_anuria,         0) AS kw_anuria,
    COALESCE(n.kw_edema,          0) AS kw_edema,
    COALESCE(n.kw_hydronephrosis, 0) AS kw_hydronephrosis,
    COALESCE(n.kw_aki_mention,    0) AS kw_aki_mention,
    COALESCE(n.kw_renal_abnormal, 0) AS kw_renal_abnormal,
    COALESCE(n.kw_fluid_overload, 0) AS kw_fluid_overload,

    -- ── 텍스트 부재 여부 지시변수 ─────────────────────────────────────
    -- "방사선 리포트 없음" 자체가 임상 패턴 (예: 중증 상태로 촬영 불가)
    CASE WHEN n.stay_id IS NULL THEN 1 ELSE 0 END AS nlp_missing,

    -- 결합 키워드 점수: 각 kw_ 합산 (0~7점)
    -- SCR-07 total_nephrotoxic_burden 계산에 부가 지표로 사용
    COALESCE(n.kw_oliguria, 0)
    + COALESCE(n.kw_anuria, 0)
    + COALESCE(n.kw_edema, 0)
    + COALESCE(n.kw_hydronephrosis, 0)
    + COALESCE(n.kw_aki_mention, 0)
    + COALESCE(n.kw_renal_abnormal, 0)
    + COALESCE(n.kw_fluid_overload, 0)
                                      AS nlp_keyword_score

FROM aki_project.cdss_cohort_window              cw
LEFT JOIN aki_project.stg_nlp_keyword_features   n
       ON cw.stay_id = n.stay_id;
-- LEFT JOIN: NLP 없는 환자도 포함 (COALESCE로 0 처리)

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_kw_stay ON aki_project.cdss_nlp_keyword_raw (stay_id);
-- ANALYZE aki_project.cdss_nlp_keyword_raw;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6-D : 방사선 원문 → cdss_nlp_radiology_extra
--           effective_cutoff 이전 리포트에서 추가 AKI 관련 키워드 추출
--
-- ▶ 사전 추출 CSV에 없는 추가 임상 키워드 (방사선 리포트 특화)
--   실제 분포 (2000행 샘플 기준):
--     pulmonary_edema:    11.5%  (폐부종 — 신기능 저하 시 흔한 소견)
--     pleural_effusion:   25.7%  (흉수 — 신부전 또는 심부전 동반)
--     ascites:             2.8%  (복수 — 신증후군/간신증후군 신호)
--     contrast_agent:      5.3%  (조영제 사용 — 조영제 신증 위험)
--     hydronephrosis:      1.7%  (원문 직접 언급, kw_ 중복이나 다른 문맥 포착)
--     renal_calculus:      0.3%  (신결석 — 폐색성 AKI 원인)
--     cardiomegaly:        5.8%  (심비대 — 심신증후군 신호)
--     foley_catheter:      0.7%  (유치 도뇨관 — 폐색 모니터링 중)
--
-- ▶ effective_cutoff 필터
--   charttime <= effective_cutoff: prediction_cutoff 이전 리포트만 포함
--   이 단계에서 데이터 누수를 완전히 차단한다.
--
-- ▶ 환자당 집계 방식
--   여러 리포트 중 하나라도 키워드가 있으면 1 (MAX 집계)
--   가장 최근 리포트의 combined 텍스트도 별도 보존 (last_rad_text)
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_nlp_radiology_extra CASCADE;

CREATE TABLE aki_project.cdss_nlp_radiology_extra AS

WITH
-- ── effective_cutoff 이전 리포트만 필터링 ─────────────────────────────
filtered_notes AS (
    SELECT
        r.stay_id,
        r.charttime,
        -- findings + impression 결합 텍스트 (소문자 변환)
        -- impression이 NULL인 경우(23.8%) findings만 사용
        LOWER(
            COALESCE(r.findings,   '') || ' '
            || COALESCE(r.impression, '')
        ) AS note_text
    FROM aki_project.stg_radiology_nlp_text r
    JOIN aki_project.cdss_cohort_window      cw
      ON r.stay_id = cw.stay_id
    WHERE r.stay_id   IS NOT NULL                    -- ICU 연결된 리포트만
      AND r.charttime <= cw.effective_cutoff          -- 미래 데이터 누수 차단
      AND r.charttime >= cw.icu_intime               -- ICU 입실 이후 리포트만
      AND (r.findings IS NOT NULL                    -- findings/impression 중
           OR r.impression IS NOT NULL)              -- 적어도 하나는 있어야 함
),

-- ── 리포트별 키워드 플래그 ────────────────────────────────────────────
note_flags AS (
    SELECT
        stay_id,
        charttime,

        -- ─ 폐·흉부 소견 (신기능 저하의 간접 지표) ─────────────────────
        -- 폐부종: 신장이 체액을 제거 못하면 폐에 수분 축적
        CASE WHEN note_text ~* '\mpulmonary\s+edema\M
                              |\mpulmonary\s+congestion\M
                              |\mcongestive\s+heart\s+failure\M'
             THEN 1 ELSE 0 END                        AS kw_pulmonary_edema,

        -- 흉수: 신부전 또는 심부전 동반 시 흔하게 발생
        CASE WHEN note_text ~* '\mpleural\s+effusion\M
                              |\mpleural\s+fluid\M'
             THEN 1 ELSE 0 END                        AS kw_pleural_effusion,

        -- ─ 복부 소견 (복수 — 신증후군/간신증후군) ───────────────────────
        CASE WHEN note_text ~* '\mascit[eis]\M
                              |\mperitoneal\s+fluid\M
                              |\mfree\s+fluid.*abdomen\M'
             THEN 1 ELSE 0 END                        AS kw_ascites,

        -- ─ 조영제 관련 (조영제 신증, CI-AKI 위험) ───────────────────────
        -- 조영제 사용 자체가 신독성 트랙C와 겹치지만 방사선 소견으로 확인
        CASE WHEN note_text ~* '\mcontrast\M
                              |\miodinated\s+contrast\M
                              |\mgadolinium\M
                              |\mnephropathy\M'
             THEN 1 ELSE 0 END                        AS kw_contrast_agent,

        -- ─ 신장 구조 이상 (원문 직접 추출, 더 넓은 패턴) ────────────────
        -- kw_hydronephrosis CSV와 중복이나 문맥이 다른 경우 포착
        CASE WHEN note_text ~* '\mhydronephrosis\M
                              |\mhydroureter\M
                              |\mrenal\s+obstruction\M
                              |\mureteral\s+obstruction\M'
             THEN 1 ELSE 0 END                        AS kw_rad_hydronephrosis,

        -- 신결석: 폐색성 AKI(신후성)의 직접 원인
        CASE WHEN note_text ~* '\mrenal\s+calcul[ui]\M
                              |\mkidney\s+stone\M
                              |\mnephrolithiasis\M
                              |\murolithiasis\M
                              |\mureteral\s+calcul[ui]\M'
             THEN 1 ELSE 0 END                        AS kw_renal_calculus,

        -- ─ 심장 소견 (심신증후군 신호) ──────────────────────────────────
        -- 심비대 + 신기능 저하 = 심신증후군(Cardiorenal Syndrome) 위험
        CASE WHEN note_text ~* '\mcardiomegaly\M
                              |\mcardiac\s+enlargement\M
                              |\menlarged\s+(cardiac\s+)?silhouette\M'
             THEN 1 ELSE 0 END                        AS kw_cardiomegaly,

        -- ─ 도뇨관 (요로 폐색 모니터링 중) ──────────────────────────────
        CASE WHEN note_text ~* '\mfoley\M
                              |\murinary\s+(catheter|drain)\M
                              |\mbladder\s+catheter\M'
             THEN 1 ELSE 0 END                        AS kw_foley_catheter,

        -- ─ 신기능 직접 언급 (임상의 판독 소견) ─────────────────────────
        CASE WHEN note_text ~* '\mrenal\s+failure\M
                              |\macute\s+kidney\s+injury\M
                              |\m\baki\b\M
                              |\mrenal\s+insufficiency\M'
             THEN 1 ELSE 0 END                        AS kw_rad_aki_mention

    FROM filtered_notes
)

-- ── 환자별 집계 (복수 리포트 → 1행) ────────────────────────────────────
SELECT
    stay_id,

    -- 하나라도 있으면 1 (MAX = OR 집계)
    MAX(kw_pulmonary_edema)   AS kw_pulmonary_edema,
    MAX(kw_pleural_effusion)  AS kw_pleural_effusion,
    MAX(kw_ascites)           AS kw_ascites,
    MAX(kw_contrast_agent)    AS kw_contrast_agent,
    MAX(kw_rad_hydronephrosis)AS kw_rad_hydronephrosis,
    MAX(kw_renal_calculus)    AS kw_renal_calculus,
    MAX(kw_cardiomegaly)      AS kw_cardiomegaly,
    MAX(kw_foley_catheter)    AS kw_foley_catheter,
    MAX(kw_rad_aki_mention)   AS kw_rad_aki_mention,

    -- 방사선 리포트 수 (많을수록 중증 경과 가능성)
    COUNT(*)                  AS rad_report_count,

    -- 결측 지시변수: 이 stay_id에 방사선 리포트가 있으면 0
    0                         AS rad_text_missing

FROM note_flags
GROUP BY stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_rad_stay ON aki_project.cdss_nlp_radiology_extra (stay_id);
-- ANALYZE aki_project.cdss_nlp_radiology_extra;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6-E : cdss_nlp_features — 두 소스 병합 + 전체 결측 처리
--
-- ▶ 역할
--   STEP 1 (사전추출 키워드) + STEP 2 (원문 추출 키워드)를 stay_id 기준으로
--   결합하여 XGBoost 입력에 사용할 최종 Track D 피처를 생성한다.
--
-- ▶ 결측 처리 전략
--   NLP 데이터 없는 환자 (35.9%): 모든 kw_ = 0, nlp_missing = 1
--   방사선 리포트 없는 환자:       rad_* = 0, rad_text_missing = 1
--   두 지시변수를 XGBoost 피처로 포함해 "관찰 없음 패턴"을 학습에 반영
--
-- ▶ 최종 피처 목록 (feature_config.py FEAT_NLP에 추가할 것)
--   사전추출 (7개): kw_oliguria, kw_anuria, kw_edema, kw_hydronephrosis,
--                   kw_aki_mention, kw_renal_abnormal, kw_fluid_overload
--   원문추출 (9개): kw_pulmonary_edema, kw_pleural_effusion, kw_ascites,
--                   kw_contrast_agent, kw_rad_hydronephrosis, kw_renal_calculus,
--                   kw_cardiomegaly, kw_foley_catheter, kw_rad_aki_mention
--   집계 (3개):     nlp_keyword_score, rad_report_count, nlp_missing
-- ═══════════════════════════════════════════════════════════════════════════
-- DROP TABLE IF EXISTS aki_project.cdss_nlp_features CASCADE;

CREATE TABLE aki_project.cdss_nlp_features AS
SELECT
    cw.stay_id,
    cw.aki_label,
    cw.effective_cutoff,

    -- ══ 사전 추출 키워드 (nlp_keyword_features.csv 기반) ════════════════
    COALESCE(k.kw_oliguria,       0) AS kw_oliguria,      -- 핍뇨 (소변 감소)
    COALESCE(k.kw_anuria,         0) AS kw_anuria,        -- 무뇨
    COALESCE(k.kw_edema,          0) AS kw_edema,         -- 부종 (31.5%)
    COALESCE(k.kw_hydronephrosis, 0) AS kw_hydronephrosis,-- 수신증
    COALESCE(k.kw_aki_mention,    0) AS kw_aki_mention,   -- AKI 직접 언급
    COALESCE(k.kw_renal_abnormal, 0) AS kw_renal_abnormal,-- 신장 이상
    COALESCE(k.kw_fluid_overload, 0) AS kw_fluid_overload,-- 수액 과부하 (42.9%)
    COALESCE(k.nlp_keyword_score, 0) AS nlp_keyword_score,-- 키워드 합산 점수

    -- ══ 원문 추출 키워드 (radiology_nlp_text.csv → STEP 2) ══════════════
    COALESCE(r.kw_pulmonary_edema,    0) AS kw_pulmonary_edema,
    COALESCE(r.kw_pleural_effusion,   0) AS kw_pleural_effusion,
    COALESCE(r.kw_ascites,            0) AS kw_ascites,
    COALESCE(r.kw_contrast_agent,     0) AS kw_contrast_agent,    -- CI-AKI 위험
    COALESCE(r.kw_rad_hydronephrosis, 0) AS kw_rad_hydronephrosis,
    COALESCE(r.kw_renal_calculus,     0) AS kw_renal_calculus,
    COALESCE(r.kw_cardiomegaly,       0) AS kw_cardiomegaly,      -- 심신증후군
    COALESCE(r.kw_foley_catheter,     0) AS kw_foley_catheter,
    COALESCE(r.kw_rad_aki_mention,    0) AS kw_rad_aki_mention,
    COALESCE(r.rad_report_count,      0) AS rad_report_count,      -- 리포트 수

    -- ══ 결측 지시변수 ════════════════════════════════════════════════════
    COALESCE(k.nlp_missing,      1)      AS nlp_missing,       -- 1=NLP 데이터 없음
    COALESCE(r.rad_text_missing, 1)      AS rad_text_missing,  -- 1=방사선 원문 없음

    -- ══ 복합 위험 플래그 ════════════════════════════════════════════════
    -- 신장 직접 관련 소견: 수신증 또는 신결석 또는 AKI 직접 언급
    CASE WHEN COALESCE(k.kw_hydronephrosis, 0) = 1
           OR COALESCE(r.kw_rad_hydronephrosis, 0) = 1
           OR COALESCE(r.kw_renal_calculus, 0) = 1
           OR COALESCE(k.kw_aki_mention, 0) = 1
           OR COALESCE(r.kw_rad_aki_mention, 0) = 1
         THEN 1 ELSE 0 END                AS nlp_direct_renal_flag,

    -- 체액 과부하 복합: 부종 + 수액과부하 + 폐부종 + 흉수 중 2개 이상
    CASE WHEN (COALESCE(k.kw_edema, 0)
              + COALESCE(k.kw_fluid_overload, 0)
              + COALESCE(r.kw_pulmonary_edema, 0)
              + COALESCE(r.kw_pleural_effusion, 0)
              + COALESCE(r.kw_ascites, 0)) >= 2
         THEN 1 ELSE 0 END                AS nlp_fluid_burden_flag

FROM aki_project.cdss_cohort_window          cw
LEFT JOIN aki_project.cdss_nlp_keyword_raw   k  ON cw.stay_id = k.stay_id
LEFT JOIN aki_project.cdss_nlp_radiology_extra r ON cw.stay_id = r.stay_id;

CREATE INDEX IF NOT EXISTS cdss_idx_nlp_feat_stay ON aki_project.cdss_nlp_features (stay_id);
-- ANALYZE aki_project.cdss_nlp_features;


-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 6-F : cdss_master_features에 Track D NLP 컬럼 추가
--
-- ▶ 역할
--   기존 cdss_master_features에 NLP 피처 컬럼을 ALTER TABLE로 추가한다.
--   master_features를 처음부터 재생성하지 않아도 되므로 효율적이다.
--
-- ▶ 주의
--   ALTER TABLE은 전체 테이블 재작성을 유발할 수 있다. (PostgreSQL 12+는 빠름)
--   master_features가 매우 크면 새로 CREATE TABLE ... AS 방식이 더 안전하다.
-- ═══════════════════════════════════════════════════════════════════════════

-- 컬럼 추가 (이미 있으면 에러 → 주석 처리 후 UPDATE만 실행)
ALTER TABLE aki_project.cdss_master_features
    ADD COLUMN IF NOT EXISTS kw_oliguria            SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_anuria              SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_edema               SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_hydronephrosis      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_aki_mention         SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_renal_abnormal      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_fluid_overload      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_pulmonary_edema     SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_pleural_effusion    SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_ascites             SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_contrast_agent      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_rad_hydronephrosis  SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_renal_calculus      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_cardiomegaly        SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_foley_catheter      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS kw_rad_aki_mention     SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_keyword_score      SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS rad_report_count       INT      DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_missing            SMALLINT DEFAULT 1,
    ADD COLUMN IF NOT EXISTS rad_text_missing       SMALLINT DEFAULT 1,
    ADD COLUMN IF NOT EXISTS nlp_direct_renal_flag  SMALLINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS nlp_fluid_burden_flag  SMALLINT DEFAULT 0;

-- cdss_nlp_features 값으로 master_features 업데이트
UPDATE aki_project.cdss_master_features m
SET
    kw_oliguria            = n.kw_oliguria,
    kw_anuria              = n.kw_anuria,
    kw_edema               = n.kw_edema,
    kw_hydronephrosis      = n.kw_hydronephrosis,
    kw_aki_mention         = n.kw_aki_mention,
    kw_renal_abnormal      = n.kw_renal_abnormal,
    kw_fluid_overload      = n.kw_fluid_overload,
    kw_pulmonary_edema     = n.kw_pulmonary_edema,
    kw_pleural_effusion    = n.kw_pleural_effusion,
    kw_ascites             = n.kw_ascites,
    kw_contrast_agent      = n.kw_contrast_agent,
    kw_rad_hydronephrosis  = n.kw_rad_hydronephrosis,
    kw_renal_calculus      = n.kw_renal_calculus,
    kw_cardiomegaly        = n.kw_cardiomegaly,
    kw_foley_catheter      = n.kw_foley_catheter,
    kw_rad_aki_mention     = n.kw_rad_aki_mention,
    nlp_keyword_score      = n.nlp_keyword_score,
    rad_report_count       = n.rad_report_count,
    nlp_missing            = n.nlp_missing,
    rad_text_missing       = n.rad_text_missing,
    nlp_direct_renal_flag  = n.nlp_direct_renal_flag,
    nlp_fluid_burden_flag  = n.nlp_fluid_burden_flag
FROM aki_project.cdss_nlp_features n
WHERE m.stay_id = n.stay_id;

-- ANALYZE aki_project.cdss_master_features;


-- ── 최종 검증 ─────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                         AS n_total,
    SUM(1 - nlp_missing)                             AS n_nlp_available,
    ROUND(100.0 * SUM(1-nlp_missing) / COUNT(*), 1) AS nlp_coverage_pct,
    SUM(kw_edema)                                    AS n_kw_edema,
    SUM(kw_fluid_overload)                           AS n_kw_fluid_overload,
    SUM(kw_pulmonary_edema)                          AS n_kw_pulmonary_edema,
    SUM(kw_pleural_effusion)                         AS n_kw_pleural_effusion,
    SUM(kw_contrast_agent)                           AS n_kw_contrast,
    SUM(nlp_direct_renal_flag)                       AS n_direct_renal,
    SUM(nlp_fluid_burden_flag)                       AS n_fluid_burden
FROM aki_project.cdss_master_features;


-- ═══════════════════════════════════════════════════════════════════════════
-- 최종 검증 쿼리  —  트랙 A·B·C·D 통합 후 전체 확인
-- ═══════════════════════════════════════════════════════════════════════════
SELECT
    COUNT(*)                                         AS n_total,
    SUM(aki_label)                                   AS n_aki,
    ROUND(100.0*SUM(aki_label)/COUNT(*),1)            AS aki_pct,
    ROUND(AVG(rule_based_score)::NUMERIC,1)           AS avg_rule_score,
    SUM(high_risk_flag)                              AS n_high_risk,
    SUM(vanco_piptazo_combo)                         AS n_vanco_pip,
    SUM(triple_whammy)                               AS n_triple_whammy,
    SUM(competed_with_death)                         AS n_competing_risk,
    -- Track D NLP 통계
    SUM(1 - nlp_missing)                             AS n_nlp_available,
    ROUND(100.0*SUM(1-nlp_missing)/COUNT(*),1)        AS nlp_coverage_pct,
    SUM(kw_edema)                                    AS n_kw_edema,
    SUM(kw_fluid_overload)                           AS n_kw_fluid_overload,
    SUM(kw_pulmonary_edema)                          AS n_kw_pulmonary_edema,
    SUM(kw_pleural_effusion)                         AS n_kw_pleural_effusion,
    SUM(nlp_direct_renal_flag)                       AS n_direct_renal,
    SUM(nlp_fluid_burden_flag)                       AS n_fluid_burden
FROM aki_project.cdss_master_features;
