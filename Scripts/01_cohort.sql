DROP TABLE IF EXISTS cohort;

-- 기본 코호트 생성 --
CREATE TABLE cohort AS
SELECT
    p.subject_id,
    p.gender,
    p.anchor_age AS age,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.admission_type,
    a.hospital_expire_flag,
    i.stay_id,
    i.first_careunit,
    i.intime AS icu_intime,
    i.outtime AS icu_outtime,
    EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 AS icu_los_hours
FROM mimiciv_hosp.patients p
JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
JOIN mimiciv_icu.icustays i ON a.hadm_id = i.hadm_id
WHERE p.anchor_age >= 18
AND EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600.0 >= 24;

SELECT COUNT(*) FROM cohort;




-- ICD 코드 제외 확인 -- -- 이미 신장 문제가 있는 환자 제외 확인 --
-- 처음부터 신장에 문제가 있는 환자 제외 --
-- N17 : AKI 제외, 입실할 때 이미 크레아티닌이 높은 사람 
-- N18 : CKD 제외, CKD환자는 평소 크레아티닌이 항상 높음
-- Z992, Z49 : ESRD+투석 제외, 투석 환자는 신장이 이미 완전히 망가진 상태기 때문에 제외 시킴 
-- 
SELECT
    CASE
        WHEN icd_code LIKE 'N17%' THEN 'AKI (N17)'
        WHEN icd_code LIKE 'N18%' THEN 'CKD (N18)'
        WHEN icd_code LIKE 'Z992%' THEN 'ESRD 투석 (Z992)'
        WHEN icd_code LIKE 'Z49%' THEN '투석 처치 (Z49)'
        WHEN icd_code LIKE 'N185%' THEN 'ESRD (N185)'
    END AS exclude_reason,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM mimiciv_hosp.diagnoses_icd
WHERE
    icd_code LIKE 'N17%'
    OR icd_code LIKE 'N18%'
    OR icd_code LIKE 'Z992%'
    OR icd_code LIKE 'Z49%'
    OR icd_code LIKE 'N185%'
GROUP BY 1
ORDER BY 2 DESC;





-- ICD 코드 기반 실제 제외 --
-- 위에서 확인한 결과 반영하기 위한 쿼리 입니다. --
-- 이미 신장 문제가 있는 환자를 제외 했을때를 확인해 볼 수 있는 쿼리 입니다. -- 
DELETE FROM cohort
WHERE hadm_id IN (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE
        icd_code LIKE 'N17%'
        OR icd_code LIKE 'N18%'
        OR icd_code LIKE 'Z992%'
        OR icd_code LIKE 'Z49%'
        OR icd_code LIKE 'N185%'
);

SELECT COUNT(*) FROM cohort;


-- 중간점검!!--
--기본 코호트 74,829명 --
--ICD제외 후 56,595명 -- 
-- 제외된 인원 18,234명 --





-- eGFR기반 CKD추가 제외 확인 단계!ㅎ--
-- ICD코드로는 CKD환자의 절반만 잡힘. 크레아티닌 기록으로 추가로 확인 하는 단계 -- 
-- eGFR 제외 의의는 실제 신장 기능이 나쁜 환자를 제외하는것
-- CKD환자여도 의사가 항상 N18코드를 기록하지 않음
-- eGFR45 는 CKD Stage3 즉 신장 기능이 원래 나쁨 하지만 N18코드가 없을 수 있음 의사가 기록 안해서
-- 주 진단이 폐력미고 부진단이 CKD일 경우 이런 경우가 생김
-- 이렇게 데이터를 제외하지 않는다면
-- 원래 높은 크레아티닌 농도를 가진 CKD 사람이 입원후 더 높아져 AKI를 충족 해도 이게 새로운 AKI인지 CKD자연경과인지 알수 없게 되어 모델이 잘못된 패턴을 학습하게 됨
-- ICD코드가 없더라도 입원 전 1년 이내 크레아티닌 기록을 직접 봐서 신장 기능이 나쁜 환자를 추가로 잡는 과정인것임
-- 1차 측정과 2차 측정 즉 2회이상 eGFR 60미만인 사람을 CKD로 판단하고 제외하는 쿼리
WITH pre_admission_egfr AS (
    SELECT
        c.stay_id,
        c.subject_id,
        l.valuenum AS creatinine,
        CASE
            WHEN p.gender = 'F' AND l.valuenum <= 0.7
                THEN 144 * POWER(l.valuenum / 0.7, -0.329)
                         * POWER(0.993, p.anchor_age)
            WHEN p.gender = 'F' AND l.valuenum > 0.7
                THEN 144 * POWER(l.valuenum / 0.7, -1.209)
                         * POWER(0.993, p.anchor_age)
            WHEN p.gender = 'M' AND l.valuenum <= 0.9
                THEN 141 * POWER(l.valuenum / 0.9, -0.411)
                         * POWER(0.993, p.anchor_age)
            ELSE
                141 * POWER(l.valuenum / 0.9, -1.209)
                     * POWER(0.993, p.anchor_age)
        END AS egfr
    FROM cohort c
    JOIN mimiciv_hosp.patients p ON c.subject_id = p.subject_id
    JOIN mimiciv_hosp.labevents l ON c.subject_id = l.subject_id
    WHERE
        l.itemid = 50912
        AND l.valuenum BETWEEN 0.1 AND 20.0
        AND l.charttime < c.icu_intime
        AND l.charttime >= c.icu_intime - INTERVAL '365 days'
),
ckd_by_egfr AS (
    SELECT stay_id
    FROM pre_admission_egfr
    WHERE egfr < 60
    GROUP BY stay_id
    HAVING COUNT(*) >= 2
)
SELECT COUNT(*) AS n_to_exclude FROM ckd_by_egfr;





-- eGFR기반 실제 제외 -- 
WITH pre_admission_egfr AS (
    SELECT
        c.stay_id,
        c.subject_id,
        l.valuenum AS creatinine,
        CASE
            WHEN p.gender = 'F' AND l.valuenum <= 0.7
                THEN 144 * POWER(l.valuenum / 0.7, -0.329)
                         * POWER(0.993, p.anchor_age)
            WHEN p.gender = 'F' AND l.valuenum > 0.7
                THEN 144 * POWER(l.valuenum / 0.7, -1.209)
                         * POWER(0.993, p.anchor_age)
            WHEN p.gender = 'M' AND l.valuenum <= 0.9
                THEN 141 * POWER(l.valuenum / 0.9, -0.411)
                         * POWER(0.993, p.anchor_age)
            ELSE
                141 * POWER(l.valuenum / 0.9, -1.209)
                     * POWER(0.993, p.anchor_age)
        END AS egfr
    FROM cohort c
    JOIN mimiciv_hosp.patients p ON c.subject_id = p.subject_id
    JOIN mimiciv_hosp.labevents l ON c.subject_id = l.subject_id
    WHERE
        l.itemid = 50912
        AND l.valuenum BETWEEN 0.1 AND 20.0
        AND l.charttime < c.icu_intime
        AND l.charttime >= c.icu_intime - INTERVAL '365 days'
),
ckd_by_egfr AS (
    SELECT stay_id
    FROM pre_admission_egfr
    WHERE egfr < 60
    GROUP BY stay_id
    HAVING COUNT(*) >= 2
)
DELETE FROM cohort
WHERE stay_id IN (SELECT stay_id FROM ckd_by_egfr);

SELECT COUNT(*) FROM cohort;



-- 다시한번 중간점검! --
-- 기본 코호트 74,829명
-- ICD 코드 제외 후 56,595명 (-18,234명)
-- eGFR 기반 제외 후 42,210명 (-14,385명)



-- 조기 사망 제외 (6시간 이내 사망 )
SELECT COUNT(*) AS n_to_exclude
FROM cohort
WHERE
    hospital_expire_flag = 1
    AND icu_los_hours < 6;

-- 이 단계에서 0명이 나옴
-- 생각해보니 ICU 24시간 이상 조건을 걸었기 때문에 이미 24시간 이상 체류한 환자만 남아있으니 6시간 이내 사망은 이미 걸러진 상태로 보임



-- 인덱스 생성(이후 다른 테이블과 조인할때 속도 빠르게 하기 위해 넣었습니다!)
CREATE INDEX idx_cohort_stay_id ON cohort (stay_id);
CREATE INDEX idx_cohort_subject_id ON cohort (subject_id);
CREATE INDEX idx_cohort_hadm_id ON cohort (hadm_id);



-- 최종검증--
-- 검증 1 : 전체 규모 및 기본 통계
SELECT
    COUNT(*)                                        AS n_total,
    ROUND(AVG(age), 1)                              AS avg_age,
    MIN(age)                                        AS min_age,
    MAX(age)                                        AS max_age,
    SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END)  AS n_male,
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END)  AS n_female,
    ROUND(AVG(icu_los_hours), 1)                    AS avg_icu_hours,
    SUM(hospital_expire_flag)                       AS n_deaths
FROM cohort;

-- 검증 2 : ICU 종류별 분포
SELECT
    first_careunit,
    COUNT(*)                                        AS n,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
FROM cohort
GROUP BY first_careunit
ORDER BY n DESC;

-- 검증 3 : 단계별 제외 인원흐름 
SELECT '① 전체 ICU 입실' AS step,
    COUNT(DISTINCT i.stay_id) AS n
FROM mimiciv_icu.icustays i

UNION ALL

SELECT '② 18세 이상',
    COUNT(DISTINCT i.stay_id)
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
WHERE p.anchor_age >= 18

UNION ALL

SELECT '③ ICU 24h 이상',
    COUNT(DISTINCT i.stay_id)
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
WHERE p.anchor_age >= 18
AND EXTRACT(EPOCH FROM (i.outtime - i.intime)) / 3600 >= 24

UNION ALL

SELECT '④ 최종 코호트', COUNT(*) FROM cohort;









