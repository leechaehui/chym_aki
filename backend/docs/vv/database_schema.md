# Database Schema (작업지시서 9)

총 **17** 테이블. SQLAlchemy 메타데이터에서 자동 추출.

## `audit_logs`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| user_id | VARCHAR(40) | Y |  |
| action | VARCHAR(60) | N |  |
| target_type | VARCHAR(40) | N |  |
| target_id | VARCHAR(40) | Y |  |
| payload | TEXT | Y |  |
| created_at | DATETIME | N |  |

**Indexes:** `ix_audit_target`(target_type, target_id); `ix_audit_user`(user_id)

## `bed_details`

| Column | Type | Null | Key |
|---|---|---|---|
| bed_id | VARCHAR(40) | N | PK |
| diagnosis | VARCHAR(200) | N |  |
| attending | VARCHAR(60) | N |  |
| admitted_at | VARCHAR(40) | N |  |
| aki_risk | BOOLEAN | N |  |
| aki_stage | VARCHAR(60) | Y |  |
| medications_json | TEXT | N |  |
| labs_json | TEXT | N |  |
| recent_inputs_json | TEXT | Y |  |
| treatment_report | TEXT | Y |  |
| treatment_items_json | TEXT | Y |  |

## `bed_reservations`

| Column | Type | Null | Key |
|---|---|---|---|
| bed_id | VARCHAR(40) | N | PK |
| patient_name | VARCHAR(60) | N |  |
| reserved_at | VARCHAR(40) | N |  |
| scheduled_at | VARCHAR(40) | N |  |
| accepted_by | VARCHAR(80) | N |  |
| requested_by | VARCHAR(80) | N |  |
| note | TEXT | Y |  |

## `consultations`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| kind | VARCHAR(20) | N |  |
| patient_mrn | VARCHAR(40) | N |  |
| patient_name | VARCHAR(60) | N |  |
| diagnosis | VARCHAR(200) | N |  |
| key_labs | VARCHAR(300) | N |  |
| reason | TEXT | N |  |
| urgency | VARCHAR(20) | N |  |
| status | VARCHAR(20) | N |  |
| requested_by | VARCHAR(60) | N |  |
| requested_at | DATETIME | N |  |
| bed_label | VARCHAR(20) | Y |  |
| reply_json | TEXT | Y |  |

**Indexes:** `ix_consultations_kind_status`(kind, status)

## `emergency_patients`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| name | VARCHAR(60) | N |  |
| sex | VARCHAR(1) | N |  |
| age | INTEGER | N |  |
| arrived_at | VARCHAR(40) | N |  |
| acuity | VARCHAR(20) | N |  |
| chief_complaint | VARCHAR(200) | N |  |
| status | VARCHAR(60) | N |  |
| attending | VARCHAR(60) | N |  |

**Indexes:** `ix_emergency_acuity`(acuity)

## `notifications`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| department | VARCHAR(20) | N |  |
| severity | VARCHAR(20) | N |  |
| title | VARCHAR(120) | N |  |
| message | TEXT | N |  |
| link | VARCHAR(200) | Y |  |
| tone | VARCHAR(20) | Y |  |
| read | BOOLEAN | N |  |
| created_at | DATETIME | N |  |

## `pathology_results`

| Column | Type | Null | Key |
|---|---|---|---|
| consult_id | VARCHAR(40) | N | PK |
| stain | VARCHAR(20) | N |  |
| image_url | VARCHAR(300) | Y |  |
| layers_json | TEXT | N |  |
| metrics_json | TEXT | N |  |
| report_findings | TEXT | N |  |
| report_diagnosis | TEXT | N |  |
| report_status | VARCHAR(10) | N |  |
| report_updated_at | VARCHAR(40) | Y |  |

## `patients`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| mrn | VARCHAR(40) | N |  |
| name | VARCHAR(60) | N |  |
| sex | VARCHAR(1) | N |  |
| age | INTEGER | N |  |
| diagnosis | VARCHAR(200) | N |  |
| admitted_at | VARCHAR(40) | N |  |
| attending | VARCHAR(60) | N |  |
| room | VARCHAR(80) | N |  |
| ai_risk_score | INTEGER | N |  |
| mimic_subject_id | INTEGER | Y |  |

**Indexes:** `ix_patients_admitted_at`(admitted_at)

## `users`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| username | VARCHAR(60) | N |  |
| password_hash | VARCHAR(120) | N |  |
| name | VARCHAR(60) | N |  |
| role | VARCHAR(20) | N |  |
| department | VARCHAR(60) | N |  |
| approval | VARCHAR(20) | N |  |
| last_login_at | DATETIME | Y |  |
| created_at | DATETIME | N |  |

**Indexes:** `ix_users_approval_role`(approval, role)

## `ai_draft_notes`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| transcript | TEXT | N |  |
| symptoms_json | TEXT | N |  |
| draft_text | TEXT | N |  |
| status | VARCHAR(20) | N |  |
| created_at | DATETIME | N |  |

**Indexes:** `ix_ai_draft_patient`(patient_id)

## `beds`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| zone | VARCHAR(20) | N |  |
| label | VARCHAR(20) | N |  |
| state | VARCHAR(20) | N |  |
| patient_id | VARCHAR(40) | Y | FK→patients.id |
| patient_name | VARCHAR(60) | Y |  |

**Indexes:** `ix_beds_zone_state`(zone, state)

## `consult_events`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| consultation_id | VARCHAR(40) | N | FK→consultations.id |
| stage | VARCHAR(20) | N |  |
| label | VARCHAR(80) | N |  |
| at | VARCHAR(40) | N |  |
| actor | VARCHAR(80) | N |  |

**Indexes:** `ix_consult_events_consultation`(consultation_id)

## `patient_labs`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| seq | INTEGER | N |  |
| key | VARCHAR(20) | N |  |
| label | VARCHAR(40) | N |  |
| value | FLOAT | N |  |
| unit | VARCHAR(20) | N |  |
| ref_low | FLOAT | Y |  |
| ref_high | FLOAT | Y |  |
| flag | VARCHAR(10) | N |  |

**Indexes:** `ix_patient_labs_patient`(patient_id)

## `patient_trend_points`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| date | VARCHAR(10) | N |  |
| creatinine | FLOAT | N |  |
| egfr | FLOAT | N |  |
| bun | FLOAT | N |  |

**Indexes:** `ix_patient_trend_patient`(patient_id)

## `patient_urine_points`

| Column | Type | Null | Key |
|---|---|---|---|
| id | INTEGER | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| date | VARCHAR(10) | N |  |
| value | FLOAT | N |  |

**Indexes:** `ix_patient_urine_patient`(patient_id)

## `timeline_events`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| event_type | VARCHAR(30) | N |  |
| severity | VARCHAR(20) | N |  |
| title | VARCHAR(120) | N |  |
| description | TEXT | Y |  |
| source | VARCHAR(30) | N |  |
| actor | VARCHAR(80) | Y |  |
| event_time | DATETIME | N |  |
| payload_json | TEXT | Y |  |
| created_at | DATETIME | N |  |

**Indexes:** `ix_timeline_patient_severity`(patient_id, severity); `ix_timeline_patient_time`(patient_id, event_time)

## `admissions`

| Column | Type | Null | Key |
|---|---|---|---|
| id | VARCHAR(40) | N | PK |
| patient_id | VARCHAR(40) | N | FK→patients.id |
| bed_id | VARCHAR(40) | N | FK→beds.id |
| status | VARCHAR(20) | N |  |
| admitted_at | DATETIME | N |  |
| discharged_at | DATETIME | Y |  |

**Indexes:** `ix_admissions_bed_status`(bed_id, status); `ix_admissions_patient_status`(patient_id, status)
