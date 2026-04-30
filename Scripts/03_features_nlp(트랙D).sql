SELECT
    r.note_id,
    r.subject_id,
    r.hadm_id,
    c.stay_id,
    r.charttime,
    TRIM(SUBSTRING(r.text FROM 'FINDINGS:([\s\S]*?)(?=\n[A-Z ]+:|$)'))   AS findings,
    TRIM(SUBSTRING(r.text FROM 'IMPRESSION:([\s\S]*?)(?=\n[A-Z ]+:|$)')) AS impression
FROM mimiciv_note.radiology r
JOIN cohort c
    ON  r.hadm_id   = c.hadm_id
    AND r.charttime >= c.icu_intime                        -- ICU 입실 이후
    AND r.charttime <= c.icu_intime + INTERVAL '24' HOUR  -- 24시간 이내
ORDER BY c.stay_id, r.charttime;


SELECT stay_id FROM cohort;