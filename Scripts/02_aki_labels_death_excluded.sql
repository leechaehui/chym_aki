-- 코호트환자 42,210명 각각에 "이 환자는 ICU에 있는 동안 AKI가 생겼는가, 생겼다면 언제" 라는 정답표를 만드는 작업!

-- 크레아티닌 기반
-- 48시간 이내 크레아티닌이 0.3mg/dL 이상 상승 또는 7일 이내 기저치 대비 1.5 ml/kg/h 미만

-- 소변량 기반
-- 6시간 이산 연속으로 소변량 0.5 ml/kg/h미만

-- 이 단계 목표 정리
--cr_timeseries 크레아티닌 시계열
-- baseline_creatinine 기저 크레아티닌 (AKI판정 기준점)
--aki_stage_creatinine 크레아티닌 기준으로 AKI판정

-- urine_hourly 시간별 소변량
--patient_weight_filled 체중 (소변량 보정용)
-- urine_rate mL/kg/h계산

--aki_stage_urine 소변량 기준으로 AKI 판정
--aki_stage_final 두 기준 통합 -> 최종!!!!



-- !!!!!!!여기에 cr_timeseries (ICU크레아티닌 시계열), baseline_creatinine (환자별 기저 크레아티닌)테이블이 있음!!!




-- step1크레아티닌 시계열 추출 --
-- 42,210명의 ICU환자들이 ICU에 있는 동안 혈액 검사로 크레아티닌을 측정한 기록 전부를 시간 순서대로 모은 테이블
-- 이 테이블의 결과 250,192행의 뜻은 42,210명이 ICU에 있는 동안 총 250,192번 크레아티닌 검사를 받았다는 뜻
-- 즉 ICU체류 동안 하루에 1~2번 검사 받은 셈
DROP TABLE IF EXISTS cr_timeseries;

CREATE TABLE cr_timeseries AS
SELECT
    c.subject_id,
    c.stay_id,
    c.hadm_id,
    c.icu_intime,
    c.icu_outtime,
    l.charttime,
    l.valuenum AS creatinine,
    EXTRACT(EPOCH FROM (l.charttime - c.icu_intime))
        / 3600.0 AS hours_from_icu_admit
FROM cohort_death_excluded c     -- 코호트 정의 반영한 구간!!
JOIN mimiciv_hosp.labevents l     -- 코호트 안에 있는 subject_id와 일치하는 기록만 가져옴!
    ON  c.subject_id = l.subject_id
    AND l.charttime >= c.icu_intime     -- ICU 입실 이후만
    AND l.charttime <= c.icu_outtime  -- ICU퇴실 이전까지만
WHERE
    l.itemid = 50912  -- 크레아티닌 항목 코드 
    AND l.valuenum IS NOT NULL
    AND l.valuenum BETWEEN 0.1 AND 20.0
ORDER BY c.stay_id, l.charttime;

SELECT COUNT(*) FROM cr_timeseries;	-- 247,968







-- step2 기조 크에아티닌 계산
-- 이 환자의 평소 크레아티닌
-- KDIGO AKI판정 기준 중 하나가 기저치 대비 1.5배 이상 상승 이기 때문에 기저치 고려를 위한 테이블
-- 환자 A 기저치: 0.9 mg/dL
-- ICU 3일차:    1.4 mg/dL
-- 비율: 1.4 / 0.9 = 1.56배 → 1.5배 이상 → AKI
-- 아래 코드 실행결과 42,210명 중 37,447명은 입원 전 1년 이내에 외래 크레아티닌 기록이 있음.
-- 이 환자들은 평소 신장 기능을 가장 정확하게 반영하는 기저치를 가짐. 평균 0.94mg/dL로 정상 범위
-- ICU입실 후 첫값 11.2%
--4,705명은 입원 전 외래 기록이 없어서 ICU입실 후 측정값을 기저치로 썼음
-- 평균이 1.13으로 외래값보다 약간 높은데, 이미 아픈 상태에서 ICU에 들어온 직후 값이라 자연스럽게 높게 나옴


DROP TABLE IF EXISTS baseline_creatinine;

CREATE TABLE baseline_creatinine AS

WITH pre_icu AS (
    SELECT DISTINCT ON (c.stay_id)
        c.stay_id,
        l.valuenum      AS baseline_cr,
        l.charttime     AS baseline_time,
        1               AS priority
    FROM cohort_death_excluded c
    JOIN mimiciv_hosp.labevents l
        ON  c.subject_id = l.subject_id
        AND l.charttime  < c.icu_intime
        AND l.charttime >= c.icu_intime - INTERVAL '365 days'
    WHERE
        l.itemid = 50912
        AND l.valuenum BETWEEN 0.1 AND 20.0
    ORDER BY c.stay_id, l.charttime DESC
),

post_icu_first AS (
    SELECT DISTINCT ON (stay_id)
        stay_id,
        creatinine      AS baseline_cr,
        charttime       AS baseline_time,
        2               AS priority
    FROM cr_timeseries
    ORDER BY stay_id, charttime ASC
)

SELECT DISTINCT ON (stay_id)
    stay_id,
    baseline_cr,
    baseline_time,
    priority,
    CASE
        WHEN priority = 1 THEN '입원전 외래값'
        ELSE 'ICU입실후 첫값'
    END AS baseline_source
FROM (
    SELECT * FROM pre_icu
    UNION ALL
    SELECT * FROM post_icu_first
) combined
ORDER BY stay_id, priority ASC;


SELECT
    baseline_source,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct,
    ROUND(AVG(baseline_cr)::NUMERIC, 2) AS avg_baseline_cr
FROM baseline_creatinine
GROUP BY baseline_source;





-- step3. 크레아티닌 기준 AKI판정 -- 
-- 위에서 만든 두 테이블을 비교해서 KDIGO기준을 충족하는 시점을 찾는 작업
-- 조건 1 : 48시간 이내 0.3 mg/dL 이상 상승
-- 조건 2 : 7일 이내 기저치 대비 1.5배 이상 상승
-- 비교해서 두 조건 중 하나 충족하면 AKI

-- 얼마나 올랐는지에 따라 Stage 분류
-- stage1 : 기저치 1.5~1.9배  OR  +0.3 mg/dL
-- stage2 : 기저치 2.0~2.9배
-- stage3 : 기저치 3.0배 이상 or 절대값 4.0 mg/dL 이상 or 투석 시작

-- 코드실행결과
-- 5,693 + 340 + 711 = 6,744명
-- 6,744 / 42,210 = 약 16%
-- 크레아티닌 기준만으로 전체 코호트의 16%에서 AKI가 발생한걸 볼 수 있다. 

--어떤 환자는 48시간 안에 0.3이 안 올라도, 일주일에 걸쳐 서서히 1.5배가 될 수 있습니다. 
--이런 경우도 AKI. 
-- 두 조건을 함께 쓰는 이유가 여기 있음. 서로 다른 패턴의 AKI를 각각 잡기 위해서.

DROP TABLE IF EXISTS aki_stage_creatinine;

CREATE TABLE aki_stage_creatinine AS

WITH cr_with_baseline AS (
    SELECT
        cr.stay_id,
        cr.charttime,
        cr.creatinine,
        b.baseline_cr,

        cr.creatinine - MIN(cr_prev.creatinine)
            AS cr_delta_48h,

        CASE
            WHEN b.baseline_cr > 0
            THEN cr.creatinine / b.baseline_cr
            ELSE NULL
        END AS cr_ratio

    FROM cr_timeseries cr
    JOIN baseline_creatinine b
        ON cr.stay_id = b.stay_id
    LEFT JOIN cr_timeseries cr_prev
        ON  cr.stay_id        = cr_prev.stay_id
        AND cr_prev.charttime < cr.charttime
        AND cr_prev.charttime >= cr.charttime - INTERVAL '48 hours'
    GROUP BY
        cr.stay_id,
        cr.charttime,
        cr.creatinine,
        b.baseline_cr
),

rrt_events AS (
    SELECT
        stay_id,
        MIN(starttime) AS rrt_start_time
    FROM mimiciv_icu.procedureevents
    WHERE itemid IN (
        225441, 225802, 225803,
        225805, 224270, 225809, 225955
    )
    GROUP BY stay_id
),

cr_staged AS (
    SELECT
        cb.stay_id,
        cb.charttime,
        cb.creatinine,
        cb.baseline_cr,
        cb.cr_delta_48h,
        cb.cr_ratio,
        r.rrt_start_time,

        CASE
            WHEN r.rrt_start_time IS NOT NULL
             AND r.rrt_start_time <= cb.charttime
            THEN 3
            WHEN cb.cr_ratio >= 3.0
              OR cb.creatinine >= 4.0
            THEN 3
            WHEN cb.cr_ratio >= 2.0
            THEN 2
            WHEN cb.cr_ratio >= 1.5
              OR cb.cr_delta_48h >= 0.3
            THEN 1
            ELSE 0
        END AS cr_stage

    FROM cr_with_baseline cb
    LEFT JOIN rrt_events r
        ON cb.stay_id = r.stay_id
)

SELECT DISTINCT ON (stay_id)
    stay_id,
    charttime       AS cr_aki_onset,
    creatinine,
    baseline_cr,
    cr_delta_48h,
    cr_ratio,
    cr_stage,
    rrt_start_time
FROM cr_staged
WHERE cr_stage > 0
ORDER BY stay_id, charttime ASC;

SELECT
    cr_stage,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct,
    ROUND(AVG(cr_delta_48h)::NUMERIC, 2) AS avg_delta,
    ROUND(AVG(cr_ratio)::NUMERIC, 2) AS avg_ratio
FROM aki_stage_creatinine
GROUP BY cr_stage
ORDER BY cr_stage;





-- step 4 시간별 소변량 집계 (urine_hourly)
-- chartevent테이블에서 코호트 환자들의 소변량 기록을 1시간 단위로 합산하는 작업
-- 왜 1시간 단위로 했냐면, KDIGO 소변량 기준이 "시간당 몇 mL/kg"이기 때문
-- 시간 단위로 집계해야 "연속 6시간 이상 기준 미달을 계산할 수 있음"
-- chartevents 원본데이터에서의 불규칙한 시간 간격을 1시간 단위로 환산해서 정리하는 테이블

SELECT itemid, label, COUNT(*) AS n
FROM mimiciv_icu.d_items
WHERE label ILIKE '%urine%'
   OR label ILIKE '%foley%'
   OR label ILIKE '%void%'
GROUP BY itemid, label
ORDER BY n DESC;

--위에 이거 실행해보면 itemid라는 숫자 코드 나옴
-- d_items테이블보면 itemid의 label을 볼 수 있음 각각이 뭘 의미하는지
-- d_item 테이블 = itemid번호와 이름을 연결해주는 사전이라고 이해
-- 226559 = Foley 유치도뇨과으로 측정한 소변량
-- 226560 = Void 환자가 스스로 화장실 가서 잰 소변량
-- 226627 = OR Urine 수술실에서 측정한 소변량
-- 226631 = PACU Urine 회복실에서 측정, 수술후 마취 회복 중 소변량
-- 이라는걸 이 코드로 알 수 있음 이 코드를 참고해서 아래 코드 구성함

DROP TABLE IF EXISTS urine_hourly;

CREATE TABLE urine_hourly AS
SELECT
    ce.stay_id,
    DATE_TRUNC('hour', ce.charttime)        AS hour_bucket,
    SUM(ce.valuenum)                        AS urine_ml

FROM mimiciv_icu.chartevents ce
JOIN cohort c
    ON  ce.stay_id    = c.stay_id
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_outtime

WHERE
    ce.itemid IN (
        226559,   -- Foley (가장 많이 쓰임)
        226560,   -- Void (자가 배뇨)
        226627,   -- OR Urine (수술실)
        226631    -- PACU Urine (회복실)
    )
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum >= 0
    AND ce.valuenum < 2000    -- 시간당 2L 이상은 이상치

GROUP BY ce.stay_id, DATE_TRUNC('hour', ce.charttime)
ORDER BY ce.stay_id, hour_bucket;

-- 위 코드에서 왜 227489(GU Irrigant)를 제외 했냐면, 이건 방광 세척에 쓴 세척액으로 소변이 섞인 값이기 때문에 실제 소변량이 아니라서 제외함
-- 이 코드를 넣으면 소변량이 실제보다 훨씬 많게 계산되서 AKI를 놓치게 됨

-- 왜 2000mL상한을 두었을까?
-- 간호사가 교대할 때 누적 소변량을 한번에 입력하는 경우가 있음
-- 12시간치 소변을 한 번에 1500ml로 입력
-- 이걸 1시간 데이터로 쓰면 완전히 틀린값이 되기 때문에 
-- 2000mL 이상은 이상치로 제거함

DROP TABLE IF EXISTS urine_hourly;

CREATE TABLE urine_hourly AS
SELECT
    oe.stay_id,
    DATE_TRUNC('hour', oe.charttime)        AS hour_bucket,
    SUM(oe.value)                           AS urine_ml

FROM mimiciv_icu.outputevents oe
JOIN cohort_death_excluded c
    ON  oe.stay_id    = c.stay_id
    AND oe.charttime >= c.icu_intime
    AND oe.charttime <= c.icu_outtime

WHERE
    oe.itemid IN (
        226559,   -- Foley (핵심, 360만건)
        226560,   -- Void
        226561,   -- Condom Cath
        226563,   -- Suprapubic
        226627,   -- OR Urine
        226631,   -- PACU Urine
        226584    -- Ileoconduit
    )
    AND oe.value IS NOT NULL
    AND oe.value >= 0
    AND oe.value < 2000

GROUP BY oe.stay_id, DATE_TRUNC('hour', oe.charttime)
ORDER BY oe.stay_id, hour_bucket;

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT stay_id) AS n_patients,
    ROUND(AVG(urine_ml)::NUMERIC, 1) AS avg_urine_per_hour
FROM urine_hourly; -- 2,124,076/40,914/153.5

-- step4 결과 해석
-- total_rows   2,137,737행
-- 41,693명의 환자가 ICU에 있는 동안 시간 단위로 집계된 소변량 기록이 총 2,137,737개

-- n_patients     41,693명
-- 코호트 42,210명 중 41,693명에게 소변량 기록이 있음
-- 517명은 소변량 기록이 없는데, 이는 측정을 안 했거나 기록이 누락된 경우
-- 이 환자들은 소변량 기준 AKI 판정에서 자동으로 제외

-- avg_urine_per_hour     153.3 mL
-- KDIGO 정상 기준: 0.5 mL/kg/h 이상




-- step5 환자 체중추출 (patient_weight_filled)
-- 이 단계는 소변량 기준 AKI는 단순히 몇 mL가 아니라 체중 1kg당 시간당 몇 mL로 판정하기 때문에
-- 체중 50kg 환자: 시간당 20mL → 20 ÷ 50 = 0.4 mL/kg/h → 기준 미달
-- 체중 50kg 환자: 시간당 30mL → 30 ÷ 50 = 0.6 mL/kg/h → 정상
-- 체중 100kg 환자: 시간당 20mL → 20 ÷ 100 = 0.2 mL/kg/h → 기준 미달

SELECT
    d.itemid,
    d.label,
    COUNT(*) AS n_records
FROM mimiciv_icu.chartevents ce
JOIN mimiciv_icu.d_items d ON ce.itemid = d.itemid
WHERE
    d.label ILIKE '%weight%'
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum BETWEEN 20 AND 300
GROUP BY d.itemid, d.label
ORDER BY n_records DESC;

-- 위 코드는 체중관련 itemid 확인
-- 224639  Daily Weight           310,695건
-- 226512  Admission Weight (Kg)   87,246건  이 두가지만 체택해서 이어서 코드

DROP TABLE IF EXISTS patient_weight;

CREATE TABLE patient_weight AS
SELECT DISTINCT ON (c.stay_id)
    c.stay_id,
    ce.valuenum                             AS weight_kg,
    ce.charttime                            AS weight_time,
    d.label                                 AS weight_source
FROM cohort_death_excluded c
JOIN mimiciv_icu.chartevents ce
    ON  ce.stay_id    = c.stay_id
    AND ce.charttime >= c.icu_intime
    AND ce.charttime <= c.icu_intime + INTERVAL '24 hours'
JOIN mimiciv_icu.d_items d
    ON  ce.itemid = d.itemid
WHERE
    ce.itemid IN (
        224639,   -- Daily Weight (kg)
        226512    -- Admission Weight (Kg)
    )
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum BETWEEN 20 AND 300
ORDER BY c.stay_id, ce.charttime ASC;

-- 왜 ICU입실 후 24시간 이내만 보는가
-- 체중은 입실 초기에 측정한 값이 기준이 됨 나중에 측정한 값은 부종이나 수액 투여로 실제 체중과 다를 수 있음
-- 입실 후 24시간 이내 첫 번째 값 = 가장 실제 체중에 가까운 값
-- 몸무게를 20~300kg으로 범위를 정한 이유는 그 이상과 이하는 성인에게 불가능한 몸무게라고 판단했기 때문(작성 오류 예상)
-- distinct on (stay_id) 로 여러번 측정했어도 첫 번째 값 1개만 가져옴 order by charttime ASC로 가장 이른 시각 기준을 적용

DROP TABLE IF EXISTS patient_weight_filled;

CREATE TABLE patient_weight_filled AS
SELECT
    c.stay_id,
    COALESCE(
        w.weight_kg,
        CASE WHEN p.gender = 'M' THEN 70.0 ELSE 60.0 END
    ) AS weight_kg,
    CASE
        WHEN w.weight_kg IS NOT NULL THEN 'measured'
        WHEN p.gender = 'M' THEN 'imputed_male_70kg'
        ELSE 'imputed_female_60kg'
    END AS weight_source
FROM cohort_death_excluded c
JOIN mimiciv_hosp.patients p ON c.subject_id = p.subject_id
LEFT JOIN patient_weight w ON c.stay_id = w.stay_id;

-- 위 쿼리에서 왜 체중없는 환자를 표준체중으로 대체 하냐면
-- 문헌에 보면 체중을 모를 때 쓰는 성별 기반 표준 체중이 남성 70, 여성 60임
-- 완벽하진 않지만 체중 기록 없는 환자를 버리는 것보다 이 값으로 대체하는게 더 나을거라고 판단해서


SELECT
    weight_source,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct,
    ROUND(AVG(weight_kg)::NUMERIC, 1) AS avg_weight
FROM patient_weight_filled
GROUP BY weight_source
ORDER BY n DESC;


-- step5 결과분석
-- 실측값 38.5% 42,210명 중 16,249명만 ICU 입실 후 24시간 이내 체중 기록이 있음.
-- ICU에서는 중증 환자라 입실 초기에 체중 측정을 못하는 경우가 많기 때문

-- 그래서 표준값으로 대체한 비율이 61.5%
-- 남성 대체: 14,202명 (33.6%) → 70kg
-- 여성 대체: 11,759명 (27.9%) → 60kg





-- step6 mL/kg/h 변환 (urine_rate)
-- urine_hourly(시간별 소변량)와 patient_weight_filled(체중)를 합쳐서 체중 보정 소변량을 계산하는 단계

DROP TABLE IF EXISTS urine_rate;

CREATE TABLE urine_rate AS
SELECT
    u.stay_id,
    u.hour_bucket,
    u.urine_ml,
    w.weight_kg,

    -- mL/kg/h 계산
    u.urine_ml / w.weight_kg                AS urine_rate_ml_kg_h,

    -- 0.5 mL/kg/h 미만이면 1 (기준 미달)
    CASE
        WHEN u.urine_ml / w.weight_kg < 0.5
        THEN 1
        ELSE 0
    END                                     AS below_threshold,

    -- 행 번호 (나중에 연속 구간 계산에 필요)
    ROW_NUMBER() OVER (
        PARTITION BY u.stay_id
        ORDER BY u.hour_bucket
    )                                       AS rn

FROM urine_hourly          u
JOIN patient_weight_filled w
    ON u.stay_id = w.stay_id;

-- 소변량 25mL, 체중 70kg
-- 25 / 70 = 0.357 mL/kg/h → 0.5 미만 → below_threshold = 1

--row_number() 롤 각 환자별로 시간 순서대로번호를 매김 


SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT stay_id) AS n_patients,
    SUM(below_threshold) AS n_below,
    ROUND(SUM(below_threshold) * 100.0
        / COUNT(*), 1) AS pct_below,
    ROUND(AVG(urine_rate_ml_kg_h)::NUMERIC, 2)
        AS avg_rate
FROM urine_rate;


-- step6 분석결과 
-- total_rows     2,137,737행
-- n_patients        41,693명
-- n_below          312,642행
-- pct_below           14.6%
-- avg_rate             2.16 mL/kg/h

-- avg_rate 평균적으로는 정상 기준의 4배 이상, 대부분의 시간에는 소변이 정상적으로 나오고 있음을 의미
--pct_below 14.6% 전체 시간 구간 중 14.6%에서 소변량이 기준 미달
--2,137,737행 중 312,642행이 below_threshold = 1
-- 이 중에서 연속 6시간 이상 지속되는 구간만 AKI로 판정할 것 (Step 7)
-- 그럼 결과적으로 단순 14.6%보다 훨씬 적은 수가 실제 AKI 가 될것임



-- step7 소변량 기준 AKI판정  (aki_stage_urine)
-- urine_rate에서 below_threshold = 1이 연속으로 6시간 이상 지속되는 구간을 찾는 작업입니다.
-- 연속 6시간 → AKI Stage 1 와 같은 방식으로 판정하도록
--  SQL 기법이 gaps-and-islands 패턴 사용

DROP TABLE IF EXISTS aki_stage_urine;

CREATE TABLE aki_stage_urine AS

WITH
-- 기준 미달 행만 추출하고 순서번호 부여
below_rows AS (
    SELECT
        stay_id,
        hour_bucket,
        urine_rate_ml_kg_h,
        rn,
        ROW_NUMBER() OVER (
            PARTITION BY stay_id
            ORDER BY hour_bucket
        )                                   AS below_rn
    FROM urine_rate
    WHERE below_threshold = 1
),

-- 그룹 키 계산 (rn - below_rn = 연속이면 같은 값)
grouped AS (
    SELECT
        stay_id,
        hour_bucket,
        urine_rate_ml_kg_h,
        rn - below_rn                       AS grp
    FROM below_rows
),

-- 각 연속 구간의 시작/끝/길이 계산
streaks AS (
    SELECT
        stay_id,
        grp,
        MIN(hour_bucket)                    AS streak_start,
        MAX(hour_bucket)                    AS streak_end,
        COUNT(*)                            AS streak_hours,
        MIN(urine_rate_ml_kg_h)             AS min_rate,
        AVG(urine_rate_ml_kg_h)             AS avg_rate
    FROM grouped
    GROUP BY stay_id, grp
    HAVING COUNT(*) >= 6
)

-- 6시간 이상 구간만 → Stage 분류
-- 각 환자의 가장 이른 발생 시점만 남김
SELECT DISTINCT ON (stay_id)
    stay_id,
    streak_start + INTERVAL '5 hours'       AS uo_aki_onset,
    streak_start,
    streak_end,
    streak_hours,
    ROUND(min_rate::NUMERIC, 3)             AS min_rate,
    ROUND(avg_rate::NUMERIC, 3)             AS avg_rate,

    CASE
        -- Stage 3: 12시간 이상 + 무뇨 (0.1 미만)
        WHEN streak_hours >= 12
         AND min_rate < 0.1
        THEN 3

        -- Stage 3: 24시간 이상 핍뇨
        WHEN streak_hours >= 24
        THEN 3

        -- Stage 2: 12시간 이상 핍뇨
        WHEN streak_hours >= 12
        THEN 2

        -- Stage 1: 6~12시간 핍뇨
        ELSE 1
    END                                     AS uo_stage

FROM streaks
ORDER BY stay_id, streak_start ASC;

-- 왜 streak_start + INTERVAL '5 hours'인가
-- AKI 발생 시점은 기준 미달이 시작된 순간이 아님, 6시간째가 지난 후에야 KDIGO 기준 충족
-- streak_start = 06:00 (기준 미달 시작), 6시간 후 = 12:00 (AKI 공식 발생)

-- Stage 3 조건을 먼저 확인 (가장 심각한 것 우선)
--  → 12시간 이상 + 무뇨(0.1 미만)
--  → 24시간 이상 지속

-- Stage 2
-- → 12시간 이상 지속

-- Stage 1
--  → 6~12시간 지속

SELECT
    uo_stage,
    COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER(), 1)          AS pct,
    ROUND(AVG(streak_hours)::NUMERIC, 1)    AS avg_hours,
    ROUND(AVG(min_rate)::NUMERIC, 3)        AS avg_min_rate
FROM aki_stage_urine
GROUP BY uo_stage
ORDER BY uo_stage;

-- step7 결과 분석
-- 4,983 + 753 + 816 = 6,552명 전체 소변량 기준 AKI 환자
-- 6,552 / 42,210 = 약 15.5%


-- 중간점검!
-- 크레아티닌 기준  6,744명 (16.0%)
-- 소변량 기준      6,552명 (15.5%)


-- step8 최종 레이블 통합 (aki_stage_final)
-- 두가지 원칙을 두고 만듬

-- 원칙 1 : 발생시점 -> 더 이른것 
-- 크레아티닌 기준: 1월 8일 10시 AKI 발생
-- 소변량 기준:    1월 7일 22시 AKI 발생
--  → 최종 발생 시점: 1월 7일 22시 (더 이른 것)

--원칙2 : Stage → 더 높은 것
-- 크레아티닌 기준: Stage 1
-- 소변량 기준:    Stage 2
--  → 최종 Stage: 2 (더 심각한 것)

-- prediction_cutoff = aki_onset_time - 48시간
-- 피처 추출의 기준 시점 
-- 이 시각 이전 데이터만 피처로 사용가능하게 함

DROP TABLE IF EXISTS aki_stage_final_death_excluded ;

CREATE TABLE aki_stage_final_death_excluded AS

WITH all_aki AS (
    -- 크레아티닌 기준
    SELECT
        stay_id,
        cr_aki_onset                        AS onset_time,
        cr_stage                            AS stage,
        'creatinine'                        AS aki_source
    FROM aki_stage_creatinine

    UNION ALL

    -- 소변량 기준
    SELECT
        stay_id,
        uo_aki_onset                        AS onset_time,
        uo_stage                            AS stage,
        'urine_output'                      AS aki_source
    FROM aki_stage_urine
),

-- 환자별 집계
-- 발생 시점: 가장 이른 것
-- Stage: 가장 높은 것
per_patient AS (
    SELECT
        stay_id,
        MIN(onset_time)                     AS first_aki_onset,
        MAX(stage)                          AS final_stage,
        -- 어느 기준이 먼저 발생했는지
        (ARRAY_AGG(
            aki_source ORDER BY onset_time ASC
        ))[1]                               AS first_source,
        -- 두 기준 모두 충족했는지
        CASE
            WHEN COUNT(DISTINCT aki_source) = 2
            THEN TRUE
            ELSE FALSE
        END                                 AS both_criteria
    FROM all_aki
    GROUP BY stay_id
)

-- cohort 전체와 LEFT JOIN
-- AKI 없는 환자도 포함 (label = 0)
SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.icu_intime,
    c.icu_outtime,

    -- AKI 여부 (0 또는 1)
    CASE WHEN p.stay_id IS NOT NULL
         THEN 1 ELSE 0
    END                                     AS aki_label,

    -- AKI 발생 시각
    p.first_aki_onset                       AS aki_onset_time,

    -- 최종 Stage (0 = AKI 없음)
    COALESCE(p.final_stage, 0)             AS aki_stage,

    -- 어느 기준으로 먼저 발생했는지
    p.first_source,

    -- 두 기준 모두 충족했는지
    p.both_criteria,

    -- ICU 입실 후 AKI까지 걸린 시간
    CASE WHEN p.first_aki_onset IS NOT NULL
        THEN EXTRACT(EPOCH FROM
            (p.first_aki_onset - c.icu_intime)
        ) / 3600.0
        ELSE NULL
    END                                     AS hours_to_aki,

    -- prediction_cutoff (피처 추출 기준 시점)
    -- AKI 발생 48시간 전
    CASE WHEN p.first_aki_onset IS NOT NULL
        THEN p.first_aki_onset - INTERVAL '48 hours'
        ELSE NULL
    END                                     AS prediction_cutoff

FROM cohort_death_excluded        c
LEFT JOIN per_patient p ON c.stay_id = p.stay_id
ORDER BY c.stay_id;

-- prediction_cutoff가 왜 중요한가
-- 이 컬럼이 이후 모든 피처 추출의 기준점이 되기 때문

SELECT
    aki_label,
    aki_stage,
    COUNT(*)                                AS n,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER(), 1)          AS pct,
    ROUND(AVG(hours_to_aki)::NUMERIC, 1)    AS avg_hours_to_aki
FROM aki_stage_final_death_excluded
GROUP BY aki_label, aki_stage
ORDER BY aki_label DESC, aki_stage;

-- 통합 결과 해석
-- 전체AKI 발생 환자
-- 8,267 + 1,032 + 1,367 = 10,666명
-- 10,666 / 42,210 = 25.3%

-- ICU 환자의 약 25%에서 AKI 발생

-- Stage별 해석
-- Stage 1  8,267명 (19.6%)
-- 평균 43.7시간 후 발생
 --→ 입실 후 약 2일째 발생
--  → 경증이지만 가장 많음

--Stage 2  1,032명 (2.4%)
--  평균 30.9시간 후 발생
--  → Stage 1보다 더 빨리 발생
--  → 더 빨리 악화됐다는 의미

--Stage 3  1,367명 (3.2%)
--  평균 15.0시간 후 발생
--  → 가장 빨리 발생
 -- → 입실 초기부터 심각한 상태
--  → 즉각적인 처치 필요 대상

-- stage가 높을 수록 더 빨리 발생 
--Stage 1: 평균 43.7시간
--Stage 2: 평균 30.9시간
--Stage 3: 평균 15.0시간

-- 크레아티닌, 소변량 기여도 확인 쿼리
SELECT
    first_source,
    both_criteria,
    COUNT(*) AS n,
    ROUND(AVG(aki_stage)::NUMERIC, 2) AS avg_stage
FROM aki_stage_final_death_excluded
WHERE aki_label = 1
GROUP BY first_source, both_criteria
ORDER BY first_source, both_criteria;


-- 전체 AKI 환자 중 25%는 두 기준을 모두 충족

--크레아티닌만 썼다면:  5,349명 포착
--소변량만 썼다면:      5,317명 포착
--둘 다 쓰면:          10,666명 포착

-- 크레아티닌만 쓰면 소변량으로만 잡히는
--3,922명을 놓침

--소변량만 쓰면 크레아티닌으로만 잡히는
-- 4,114명을 놓침

-- KDIGO가 두 기준을 모두 요구하는 이유가 여기 있음




















