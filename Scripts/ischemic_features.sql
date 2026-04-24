-- 1) chartevents 혈압/심박수,map itemid 확인
-- 라벨,itemid 확인
SELECT itemid, label, abbreviation, linksto, category
FROM mimiciv_icu.d_items
WHERE label ILIKE '%mean%'
   OR label ILIKE '%arterial pressure%'
   OR label ILIKE '%map%'
   OR label ILIKE '%heart rate%'
   or label ILIKE '%systolic%'
   OR label ILIKE '%sbp%'
ORDER BY label;
-- 데이터 수 확인
SELECT
    c.itemid,
    d.label,
    COUNT(*) AS n_records,
    COUNT(DISTINCT c.stay_id) AS n_stays
FROM mimiciv_icu.chartevents c
JOIN mimiciv_icu.d_items d ON c.itemid = d.itemid
WHERE c.itemid IN (
    225312, 220052, 220181,   -- MAP
    220045,                  -- HR
    220050, 220179, 225309    -- SBP
)
  AND c.valuenum IS NOT NULL
GROUP BY c.itemid, d.label
ORDER BY d.label, n_records DESC;
-- hr은 220045 하나로 모든 환자 커버 가능
-- map에서 225312는 데이터가 너무 적어 제외
-- sbp에서 225309는 220050랑 같은 계열이며 220050하나로 충분하기 때문에 제외
-------------------------------------------------------------------------------------------

-- 2) inputevents 승압제 itemid 확인
SELECT itemid, label, abbreviation, linksto, category
FROM mimiciv_icu.d_items
WHERE label ILIKE '%norepinephrine%'
   OR label ILIKE '%epinephrine%'
   OR label ILIKE '%phenylephrine%'
   OR label ILIKE '%vasopressin%'
   OR label ILIKE '%dopamine%'
   OR label ILIKE '%dobutamine%'
ORDER BY label;

-- 데이터 건수 확인
SELECT
    i.itemid,
    d.label,
    i.rateuom,
    COUNT(*) AS n_records,
    COUNT(DISTINCT i.stay_id) AS n_stays
FROM mimiciv_icu.inputevents i
JOIN mimiciv_icu.d_items d
    ON i.itemid = d.itemid
WHERE i.itemid IN (
    221906, -- Norepinephrine
    221289, -- Epinephrine
    221749, -- Phenylephrine
    222315, -- Vasopressin
    221662, -- Dopamine
    221653, -- Dobutamine
    229617, -- Epinephrine
    229632, -- Phenylephrine
    229631, -- Phenylephrine
    229630  -- Phenylephrine
)
  AND i.rate IS NOT NULL
GROUP BY i.itemid, d.label, i.rateuom
ORDER BY d.label, n_records DESC;
--221906는 단위가 여러개라 (mg/kg/min)단위 제거
--221749는 단위가 여러개라 (mg/min)단위 제거
--222315는 (units/hour)를 주로 사용하기 때문에 (units/min)제거
--229632 데이터 적어서 제거
--229631 데이터 적어서 제거
--229617는 221289랑 같은 약 중복이라 제거 
----------------------------------------------------------------------------
-- 3) labevents 젖산/헤모글로빈 itemid 후보 확인
SELECT itemid, label, fluid, category
FROM mimiciv_hosp.d_labitems
WHERE label ILIKE '%lactate%'
   OR label ILIKE '%hemoglobin%'
ORDER BY label;
-- 데이터 건수 확인
SELECT
    l.itemid,
    d.label,
    d.fluid,
    d.category,
    COUNT(*) AS total_records,
    COUNT(l.valuenum) AS nonnull_valuenum
FROM mimiciv_hosp.labevents l
JOIN mimiciv_hosp.d_labitems d
    ON l.itemid = d.itemid
WHERE l.itemid IN (50813, 52442, 51222, 50811, 50855)
GROUP BY l.itemid, d.label, d.fluid, d.category
ORDER BY d.label, total_records DESC;
-- 52442는 데이터 없으므로 제거
-- 50855는 실제값 없으므로 제거
------------------------------------------------------------------------------
-- 1) chartevents에서 map,hr,sbp 가져와서 각각 테이블 생성
-- map원본테이블 만들기
DROP TABLE IF EXISTS raw_map;

CREATE TABLE raw_map AS
SELECT
    a.stay_id,				-- ICU 체류 단위 ID (환자-ICU stay 기준)
    a.prediction_cutoff,	-- 예측 기준 시점 (이 이후 데이터는 사용하면 안됨)
    ce.charttime,			-- 해당 MAP 측정 시간
    ce.valuenum AS map		-- MAP 값 (Mean Arterial Pressure, 평균 동맥압)
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
   AND ce.charttime >= c.icu_intime
   AND ce.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ce.charttime <= a.prediction_cutoff
  AND ce.itemid IN (220052, 220181, 225312)
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum > 0
  AND ce.valuenum < 300
  AND ce.valuenum >= 20;

select * from raw_map LIMIT 20;
SELECT COUNT(*) FROM raw_map;	-- 결과:174,818
----------------------------------------------------------------------------

-- 디스플레이에서 현재 map수치를 보여주기 위해 만드는 테이블 
--DROP TABLE IF EXISTS feat_current_map;
--
--CREATE TABLE feat_current_map AS
--SELECT DISTINCT ON (stay_id)
--    stay_id,
--    map AS current_map
--FROM raw_map
--ORDER BY stay_id, charttime DESC;
--
--select * from feat_current_map;
-----------------------------------------------------------------------------
-- shock_index를 만들기 위해 심박수,수축기혈압 가져오기
DROP TABLE IF EXISTS raw_hr_sbp;

CREATE TABLE raw_hr_sbp AS
SELECT
    a.stay_id,
    a.prediction_cutoff,
    ce.charttime,
    ce.itemid,
    ce.valuenum
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = c.stay_id
   AND ce.charttime >= c.icu_intime
   AND ce.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ce.itemid IN (220045, 220179, 220050)
  AND ce.valuenum IS NOT NULL
  AND (
        (ce.itemid = 220045 AND ce.valuenum BETWEEN 20 AND 250) OR
        (ce.itemid IN (220179,220050) AND ce.valuenum BETWEEN 40 AND 300)
      );

select * from raw_hr_sbp LIMIT 10;
SELECT COUNT(*) FROM raw_hr_sbp;	--결과:339,917
-----------------------------------------------------------------------------
-- 2) inputevents에서 승압제 관련 가져와서 테이블 생성
-- 승압제 가져오기
DROP TABLE IF EXISTS raw_vasopressor;

CREATE TABLE raw_vasopressor AS
SELECT
    a.stay_id,
    a.prediction_cutoff,
    ie.itemid,
    ie.starttime,		-- 투여 시작 시간
    ie.endtime,			-- 투여 종료 시간
    ie.rate,			-- 투여 속도 (용량)
    ie.rateuom,			-- 용량 단위 (mcg/kg/min 등)
    ie.patientweight	-- 환자 체중 (단위 보정용)
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_icu.inputevents ie
    ON ie.stay_id = c.stay_id
   AND ie.starttime >= c.icu_intime
   AND ie.starttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE ie.itemid IN (
      221653,
      221662,
      221289,
      221906,
      221749,
      229630,
      222315
)
  AND ie.rate IS NOT NULL
  AND NOT (
        ie.itemid = 221749 
        AND ie.rateuom = 'mg/min'
  )
  AND NOT (
        ie.itemid = 222315 
        AND ie.rateuom = 'units/min'
  );

select * from raw_vasopressor LIMIT 10;
SELECT COUNT(*) FROM raw_vasopressor;	--결과:18,218
-----------------------------------------------------------------------------

-- 노르에피네프린 가져오기
DROP TABLE IF EXISTS raw_norepi;

CREATE TABLE raw_norepi AS
SELECT *
FROM raw_vasopressor
WHERE itemid = 221906 AND rateuom = 'mcg/kg/min';

select * from raw_norepi LIMIT 10;
SELECT COUNT(*) FROM raw_norepi;	--결과:8,784
-----------------------------------------------------------------------------
-- 3) labevents에서 젖산,헤모글로빈 가져와서 테이블 생성
-- 젖산 가져오기
DROP TABLE IF EXISTS raw_lactate;

CREATE TABLE raw_lactate AS
SELECT
    a.stay_id,
    a.hadm_id,				-- 병원 입원 ID (lab은 이걸로 연결)
    a.prediction_cutoff,
    le.charttime,			-- 검사 시간
    le.valuenum AS lactate	-- 젖산 수치
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_hosp.labevents le
    ON le.hadm_id = c.hadm_id
   AND le.charttime >= c.icu_intime
   AND le.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE le.itemid = 50813
  AND le.valuenum IS NOT NULL;

SELECT * FROM raw_lactate LIMIT 10;
SELECT COUNT(*) FROM raw_lactate;	--결과:14,362
-----------------------------------------------------------------------------
-- 헤모글로빈 가져오기
DROP TABLE IF EXISTS raw_hemoglobin;

CREATE TABLE raw_hemoglobin AS
SELECT
    a.stay_id,
    a.hadm_id,
    a.prediction_cutoff,
    le.charttime,
    le.valuenum AS hemoglobin	-- 헤모글로빈 수치
FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id
JOIN mimiciv_hosp.labevents le
    ON le.hadm_id = c.hadm_id
   AND le.charttime >= c.icu_intime
   AND le.charttime <= COALESCE(a.prediction_cutoff, c.icu_outtime)
WHERE le.itemid IN (50811, 51222)
  AND le.valuenum IS NOT NULL;

SELECT * FROM raw_hemoglobin LIMIT 10;
SELECT COUNT(*) FROM raw_hemoglobin;	--결과:25,689
-----------------------------------------------------------------------------
-- [파생변수 생성하기]
-- 1) chartevents
-- map 파생변수(map_mean,map_min)
DROP TABLE IF EXISTS feat_map_basic;

CREATE TABLE feat_map_basic AS
SELECT
    stay_id,
    AVG(map) AS map_mean,   -- 평균 MAP
    MIN(map) AS map_min     -- 최저 MAP
FROM raw_map
GROUP BY stay_id;

SELECT * FROM feat_map_basic LIMIT 10;
SELECT COUNT(*) FROM feat_map_basic;	--결과:2,080
-----------------------------------------------------------------------------
-- map 파생변수(map_below65_hours)
-- 단순히 "65 미만인 횟수"를 세는 게 아니라,
-- "얼마나 오랫동안 65 미만이었는지"를 시간으로 계산
DROP TABLE IF EXISTS feat_map_below65;

CREATE TABLE feat_map_below65 AS
WITH t AS (
    SELECT
        stay_id,
        prediction_cutoff,			-- 마지막 측정 이후에는 cutoff까지만 시간 계산해야 leakage 방지
        charttime,
        map,
        LEAD(charttime) OVER (
            PARTITION BY stay_id
            ORDER BY charttime
        ) AS next_time
    FROM raw_map
),
dur AS (
    SELECT
        stay_id,
        CASE
            WHEN map < 65 THEN
                EXTRACT(
				    EPOCH FROM (
				        LEAST(
				            COALESCE(next_time, prediction_cutoff),
				            prediction_cutoff,
				            charttime + INTERVAL '1 hour'
				        ) - charttime
				    )
				) / 3600.0	-- 현재 MAP가 65 미만이면 현재 시점 ~ 다음 시점(또는 cutoff)까지의 시간을 시간 단위로 계산
            ELSE 0	-- MAP가 65 이상이면 위험시간 0시간
        END AS hours
    FROM t
)
SELECT
    stay_id,
    SUM(hours) AS map_below65_hours	-- 한 환자에서 MAP<65였던 모든 시간 구간을 합산 -> 총 허혈시간(proxy) 역할
FROM dur
GROUP BY stay_id;

SELECT * FROM feat_map_below65 LIMIT 10;
SELECT COUNT(*) FROM feat_map_below65;	--결과:2,080
-----------------------------------------------------------------------------
-- hr,sbp 파생변수(shock_index)
DROP TABLE IF EXISTS feat_shock_index;

CREATE TABLE feat_shock_index AS
WITH vs AS (
    SELECT
        stay_id,
        charttime,
        AVG(CASE WHEN itemid = 220045 THEN valuenum END) AS hr,
        -- 같은 시간에 여러 측정 방식이 있으면 평균으로 정리
        AVG(CASE WHEN itemid IN (220179,220050) THEN valuenum END) AS sbp
    FROM raw_hr_sbp
    GROUP BY stay_id, charttime
),
si AS (
    SELECT
        stay_id,
        -- SBP가 0이면 나누기 오류가 나므로 NULLIF로 방지
        hr / NULLIF(sbp, 0) AS shock_index
    FROM vs
    -- HR과 SBP가 둘 다 있는 시점만 사용
    WHERE hr IS NOT NULL AND sbp IS NOT NULL
)
SELECT
    stay_id,
    -- 시점별 shock index를 환자 단위 평균으로 요약
    AVG(shock_index) AS shock_index
FROM si
GROUP BY stay_id;

SELECT * FROM feat_shock_index LIMIT 10;
SELECT COUNT(*) FROM feat_shock_index;	--결과:31,899
-----------------------------------------------------------------------------
-- 2) inputevents
-- 승압제 파생변수(vasopressor_flag)
DROP TABLE IF EXISTS feat_vaso_flag;

CREATE TABLE feat_vaso_flag AS
SELECT
    stay_id,
    -- 승압제 투여 기록이 한 번이라도 있으면 1, 없으면 0
    -- 환자의 전반적인 혈역학적 불안정성 여부를 반영
    CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END AS vasopressor_flag
FROM raw_vasopressor
GROUP BY stay_id;

SELECT * FROM feat_vaso_flag LIMIT 10;
SELECT COUNT(*) FROM feat_vaso_flag;	--결과:8,184
-----------------------------------------------------------------------------
-- 승압제 파생변수(vasopressor_flag_6h)
DROP TABLE IF EXISTS feat_vaso_flag_6h;

CREATE TABLE feat_vaso_flag_6h AS
SELECT
    a.stay_id,
    CASE WHEN COUNT(rv.*) > 0 THEN 1 ELSE 0 END AS vasopressor_flag_6h
FROM aki_stage_final a
LEFT JOIN raw_vasopressor rv
    ON a.stay_id = rv.stay_id
   AND rv.starttime <= a.prediction_cutoff
   AND COALESCE(rv.endtime, a.prediction_cutoff) >= a.prediction_cutoff - INTERVAL '6 hour'
GROUP BY a.stay_id;

SELECT * FROM feat_vaso_flag_6h LIMIT 10;
SELECT COUNT(*) FROM feat_vaso_flag_6h;	--결과:42,210
-----------------------------------------------------------------------------
-- 승압제 파생변수(vasopressor_hours)
DROP TABLE IF EXISTS feat_vaso_hours;

CREATE TABLE feat_vaso_hours AS
SELECT
    stay_id,
    SUM(
        EXTRACT(
            EPOCH FROM (
                LEAST(
                    COALESCE(endtime, prediction_cutoff),
                    prediction_cutoff,
                    starttime + INTERVAL '24 hour'
                ) - starttime
            )
        ) / 3600.0
    ) AS vasopressor_hours
FROM raw_vasopressor
WHERE starttime < prediction_cutoff
  AND (
        endtime IS NOT NULL
        OR prediction_cutoff - starttime < INTERVAL '48 hour'
      )
  AND LEAST(
        COALESCE(endtime, prediction_cutoff),
        prediction_cutoff,
        starttime + INTERVAL '24 hour'
      ) > starttime
GROUP BY stay_id
HAVING SUM(
    EXTRACT(
        EPOCH FROM (
            LEAST(
                COALESCE(endtime, prediction_cutoff),
                prediction_cutoff,
                starttime + INTERVAL '24 hour'
            ) - starttime
        )
    ) / 3600.0
) <= 200;

SELECT * FROM feat_vaso_hours LIMIT 10;
SELECT COUNT(*) FROM feat_vaso_hours;	--결과:771
-- 데이터 결과값에서 max가 556시간으로 비정상 확인됨
-- 1. endtime이 NULL인 경우 → 과도하게 길어짐
-- 2. 한 row가 비정상적으로 길게 기록됨
-- 3. 여러 row가 누적되면 500시간 이상 발생
-- endtime 없으면 cutoff까지,ㅣcutoff 이후 데이터 차단 (leakage 방지),최대 24시간까지만 인정으로 수정
-----------------------------------------------------------------------------

-- 노르에피네프린 파생변수(norepi_dose_max)
DROP TABLE IF EXISTS feat_norepi_max;

CREATE TABLE feat_norepi_max AS
SELECT
    stay_id,
    MAX(rate) AS norepi_dose_max
FROM raw_norepi
WHERE rate > 0
  AND rate <= 5
GROUP BY stay_id;

SELECT * FROM feat_norepi_max LIMIT 10;
SELECT COUNT(*) FROM feat_norepi_max;	--결과:469
-----------------------------------------------------------------------------
-- 3) labevents
-- 젖산 파생변수(lactate_max)
DROP TABLE IF EXISTS feat_lactate;

CREATE TABLE feat_lactate AS
SELECT
    stay_id,
    -- 관찰구간 중 가장 높았던 lactate
    -- 조직 저산소/저관류 상태의 최대 강도를 반영
    MAX(lactate) AS lactate_max
FROM raw_lactate
WHERE lactate > 0
AND lactate <= 30
GROUP BY stay_id;

SELECT * FROM feat_lactate LIMIT 10;
SELECT COUNT(*) FROM feat_lactate;	--결과:17,090
-----------------------------------------------------------------------------
-- 헤모글로빈 파생변수(hemoglobin_min)
DROP TABLE IF EXISTS feat_hemo;

CREATE TABLE feat_hemo AS
SELECT
    stay_id,
    -- 관찰구간 중 가장 낮았던 hemoglobin
    -- 산소 운반 능력이 가장 떨어졌던 시점을 반영
    MIN(hemoglobin) AS hemoglobin_min
FROM raw_hemoglobin
WHERE hemoglobin >= 3 AND hemoglobin <= 25 --이상치 생기지 않게 조건 설정
GROUP BY stay_id;

SELECT * FROM feat_hemo LIMIT 10;
SELECT COUNT(*) FROM feat_hemo;	--결과:33,145
--------------------------------------------------
-- 데이터 체크해보기
-- 전체 행 수 확인
SELECT 'aki_stage_final' AS table_name, COUNT(*) AS n_rows, COUNT(DISTINCT stay_id) AS n_stays FROM aki_stage_final
UNION ALL
SELECT 'feat_map_basic', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_map_basic
UNION ALL
SELECT 'feat_map_below65', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_map_below65
UNION ALL
SELECT 'feat_shock_index', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_shock_index
UNION ALL
SELECT 'feat_vaso_flag', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_vaso_flag
UNION ALL
SELECT 'feat_vaso_flag_6h', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_vaso_flag_6h
UNION ALL
SELECT 'feat_vaso_hours', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_vaso_hours
UNION ALL
SELECT 'feat_norepi_max', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_norepi_max
UNION ALL
SELECT 'feat_lactate', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_lactate
UNION ALL
SELECT 'feat_hemo', COUNT(*), COUNT(DISTINCT stay_id) FROM feat_hemo;
-- 미래 데이터 섞였는지 확인
-- 다 0이어야 함
SELECT COUNT(*) AS leakage_map
FROM raw_map
WHERE charttime > prediction_cutoff;

SELECT COUNT(*) AS leakage_hr_sbp
FROM raw_hr_sbp
WHERE charttime > prediction_cutoff;

SELECT COUNT(*) AS leakage_lactate
FROM raw_lactate
WHERE charttime > prediction_cutoff;

SELECT COUNT(*) AS leakage_hemo
FROM raw_hemoglobin
WHERE charttime > prediction_cutoff;

SELECT COUNT(*) AS leakage_vaso
FROM raw_vasopressor
WHERE starttime > prediction_cutoff;
-- 값 범위 확인
SELECT
    MIN(map_mean) AS map_mean_min,
    MAX(map_mean) AS map_mean_max,
    MIN(map_min) AS map_min_min,
    MAX(map_min) AS map_min_max
FROM feat_map_basic;

SELECT
    MIN(map_below65_hours) AS min_hours,
    MAX(map_below65_hours) AS max_hours,
    AVG(map_below65_hours) AS avg_hours
FROM feat_map_below65;

SELECT
    MIN(shock_index) AS min_si,
    MAX(shock_index) AS max_si,
    AVG(shock_index) AS avg_si
FROM feat_shock_index;

SELECT
    MIN(vasopressor_hours) AS min_vaso_h,
    MAX(vasopressor_hours) AS max_vaso_h,
    AVG(vasopressor_hours) AS avg_vaso_h
FROM feat_vaso_hours;

SELECT
    MIN(norepi_dose_max) AS min_norepi,
    MAX(norepi_dose_max) AS max_norepi,
    AVG(norepi_dose_max) AS avg_norepi
FROM feat_norepi_max;

SELECT
    MIN(lactate_max) AS min_lactate,
    MAX(lactate_max) AS max_lactate,
    AVG(lactate_max) AS avg_lactate
FROM feat_lactate;

SELECT
    MIN(hemoglobin_min) AS min_hgb,
    MAX(hemoglobin_min) AS max_hgb,
    AVG(hemoglobin_min) AS avg_hgb
FROM feat_hemo;
--중복 확인(아무것도 안나와야 함)
SELECT stay_id, COUNT(*)
FROM feat_map_basic
GROUP BY stay_id
HAVING COUNT(*) > 1;

SELECT stay_id, COUNT(*)
FROM feat_shock_index
GROUP BY stay_id
HAVING COUNT(*) > 1;

SELECT stay_id, COUNT(*)
FROM feat_lactate
GROUP BY stay_id
HAVING COUNT(*) > 1;

SELECT stay_id, COUNT(*)
FROM feat_hemo
GROUP BY stay_id
HAVING COUNT(*) > 1;
-----------------------------------------------------------------------------
-- 최종 테이블 완성
DROP TABLE IF EXISTS ischemic_features;

CREATE TABLE ischemic_features AS
SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.age,
    c.gender,
    c.icu_intime,
    c.icu_outtime,
    c.icu_los_hours,
    c.first_careunit,

    -- AKI 레이블
    a.aki_label,
    a.aki_stage,
    a.aki_onset_time,
    a.prediction_cutoff,
    a.hours_to_aki,

    -- MAP 피처
    cm.current_map,
    mb.map_mean,
    mb.map_min,
    COALESCE(m65.map_below65_hours, 0) AS map_below65_hours,

    -- Shock index
    si.shock_index,

    -- 승압제 피처
    COALESCE(vf.vasopressor_flag, 0) AS vasopressor_flag,
    COALESCE(vf6.vasopressor_flag_6h, 0) AS vasopressor_flag_6h,
    COALESCE(vh.vasopressor_hours, 0) AS vasopressor_hours,
    COALESCE(nm.norepi_dose_max, 0) AS norepi_dose_max,

    -- Lab 피처
    lac.lactate_max,
    hemo.hemoglobin_min,

    -- 결측 여부 지시변수
    CASE WHEN cm.current_map IS NULL THEN 1 ELSE 0 END AS current_map_missing,
    CASE WHEN mb.map_mean IS NULL THEN 1 ELSE 0 END AS map_missing,
    CASE WHEN si.shock_index IS NULL THEN 1 ELSE 0 END AS shock_index_missing,
    CASE WHEN lac.lactate_max IS NULL THEN 1 ELSE 0 END AS lactate_missing,
    CASE WHEN hemo.hemoglobin_min IS NULL THEN 1 ELSE 0 END AS hemoglobin_missing

FROM cohort c
JOIN aki_stage_final a
    ON c.stay_id = a.stay_id

LEFT JOIN feat_current_map cm
    ON c.stay_id = cm.stay_id

LEFT JOIN feat_map_basic mb
    ON c.stay_id = mb.stay_id

LEFT JOIN feat_map_below65 m65
    ON c.stay_id = m65.stay_id

LEFT JOIN feat_shock_index si
    ON c.stay_id = si.stay_id

LEFT JOIN feat_vaso_flag vf
    ON c.stay_id = vf.stay_id

LEFT JOIN feat_vaso_flag_6h vf6
    ON c.stay_id = vf6.stay_id

LEFT JOIN feat_vaso_hours vh
    ON c.stay_id = vh.stay_id

LEFT JOIN feat_norepi_max nm
    ON c.stay_id = nm.stay_id

LEFT JOIN feat_lactate lac
    ON c.stay_id = lac.stay_id

LEFT JOIN feat_hemo hemo
    ON c.stay_id = hemo.stay_id;
-----------------------------------------------------------------------------
-- 데이터 확인
select * from final_ischemic_features limit 30;
SELECT COUNT(*) FROM final_ischemic_features;
SELECT stay_id, COUNT(*)
FROM final_ischemic_features
GROUP BY stay_id
HAVING COUNT(*) > 1;

SELECT
    COUNT(*) AS total,

    SUM(CASE WHEN map_mean IS NULL THEN 1 ELSE 0 END) AS map_mean_null,
    ROUND(100.0 * SUM(CASE WHEN map_mean IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS map_mean_null_pct,

    SUM(CASE WHEN shock_index IS NULL THEN 1 ELSE 0 END) AS shock_index_null,
    ROUND(100.0 * SUM(CASE WHEN shock_index IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS shock_index_null_pct,

    SUM(CASE WHEN lactate_max IS NULL THEN 1 ELSE 0 END) AS lactate_null,
    ROUND(100.0 * SUM(CASE WHEN lactate_max IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS lactate_null_pct,

    SUM(CASE WHEN hemoglobin_min IS NULL THEN 1 ELSE 0 END) AS hemoglobin_null,
    ROUND(100.0 * SUM(CASE WHEN hemoglobin_min IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS hemoglobin_null_pct,

    SUM(CASE WHEN vasopressor_flag = 0 THEN 1 ELSE 0 END) AS vaso_no_use,
    ROUND(100.0 * SUM(CASE WHEN vasopressor_flag = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS vaso_no_use_pct
FROM final_ischemic_features;

