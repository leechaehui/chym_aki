/* =========================================================
   CDSS Cardio Filter용 테이블
   1) 총 허혈시간: MAP < 65인 누적 시간
   2) 현재 평균동맥압: index_time 직전 가장 최근 MAP
========================================================= */


/* =========================
   1. MAP 원본
========================= */
DROP TABLE IF EXISTS cdss_raw_map;

CREATE TABLE cdss_raw_map AS
WITH base AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.icu_intime,
        c.icu_outtime,
        a.aki_label,
        a.prediction_cutoff,
        COALESCE(a.prediction_cutoff, c.icu_outtime) AS index_time
    FROM cohort_death_excluded c
    JOIN aki_stage_final_death_excluded a
        ON c.stay_id = a.stay_id
)
SELECT
    b.stay_id,
    b.subject_id,
    b.hadm_id,
    b.aki_label,
    b.prediction_cutoff,
    b.index_time,
    ce.charttime,
    ce.itemid,
    ce.valuenum AS map_value
FROM base b
JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = b.stay_id
   AND ce.charttime >= b.icu_intime
   AND ce.charttime <= b.index_time
WHERE ce.itemid IN (220052, 220181, 225312)
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum BETWEEN 20 AND 300;

/* =========================
   2. 총 허혈시간 테이블
   - MAP < 65인 구간의 누적 시간
   - 단위: hour
========================= */
DROP TABLE IF EXISTS cdss_total_ischemic_time;

CREATE TABLE cdss_total_ischemic_time AS
WITH ordered_map AS (
    SELECT
        stay_id,
        subject_id,
        hadm_id,
        aki_label,
        prediction_cutoff,
        index_time,
        charttime,
        map_value,

        LEAD(charttime) OVER (
            PARTITION BY stay_id
            ORDER BY charttime
        ) AS next_charttime
    FROM cdss_raw_map
),

interval_map AS (
    SELECT
        stay_id,
        subject_id,
        hadm_id,
        aki_label,
        prediction_cutoff,
        index_time,
        charttime,
        map_value,

        CASE
            WHEN next_charttime IS NULL THEN index_time
            WHEN next_charttime > index_time THEN index_time
            ELSE next_charttime
        END AS interval_end
    FROM ordered_map
)

SELECT
    stay_id,
    subject_id,
    hadm_id,
    aki_label,
    prediction_cutoff,
    index_time,

    SUM(
        CASE
            WHEN map_value < 65
             AND interval_end > charttime
            THEN EXTRACT(EPOCH FROM (interval_end - charttime)) / 3600.0
            ELSE 0
        END
    ) AS total_ischemic_hours

FROM interval_map
GROUP BY
    stay_id,
    subject_id,
    hadm_id,
    aki_label,
    prediction_cutoff,
    index_time;

/* =========================
   3. 현재 평균동맥압 테이블
   - index_time 직전 가장 최근 MAP
========================= */
DROP TABLE IF EXISTS cdss_current_map;

CREATE TABLE cdss_current_map AS
SELECT DISTINCT ON (stay_id)
    stay_id,
    subject_id,
    hadm_id,
    aki_label,
    prediction_cutoff,
    index_time,
    charttime AS current_map_time,
    map_value AS current_map
FROM cdss_raw_map
ORDER BY stay_id, charttime DESC;

-------------------------------------
-- 확인용
SELECT *
FROM cdss_total_ischemic_time
LIMIT 10;

SELECT *
FROM cdss_current_map
LIMIT 10;

-------------------------------------
--두개를 만약에 합쳐야 한다면
--DROP TABLE IF EXISTS cdss_cardio_filter;
--
--CREATE TABLE cdss_cardio_filter AS
--SELECT
--    t.stay_id,
--    t.subject_id,
--    t.hadm_id,
--    t.aki_label,
--    t.prediction_cutoff,
--    t.index_time,
--    t.total_ischemic_hours,
--    c.current_map,
--    c.current_map_time
--FROM cdss_total_ischemic_time t
--LEFT JOIN cdss_current_map c
--    ON t.stay_id = c.stay_id;