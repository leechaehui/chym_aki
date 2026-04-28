-- 기존 cohort에서 24~48시간 내 사망 환자만 제거한 코호트
-- 기존 cohort: 42,210명
-- 제거 대상: ICU 체류 24~48시간 내 사망 801명
-- 예상 결과: 41,409명

DROP TABLE IF EXISTS cohort_death_excluded;

CREATE TABLE cohort_death_excluded AS
SELECT *
FROM cohort
WHERE NOT (
    -- 24~48시간 체류 중 사망한 환자
    icu_los_hours BETWEEN 24 AND 48
    AND hospital_expire_flag = 1
);

SELECT
    COUNT(*) AS n_total,
    SUM(hospital_expire_flag) AS n_death,
    ROUND(AVG(age)::NUMERIC, 1) AS avg_age,
    ROUND(AVG(icu_los_hours)::NUMERIC, 1) AS avg_icu_hours
FROM cohort_death_excluded;

