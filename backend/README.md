# CHYM-AKI Backend

병원 EMR 운영 + AI 음성 진료 + AKI 분석 + 병상 트랜잭션 시스템 (FastAPI).

단순 API 서버가 아니라 **실제 병원 운영 로직을 트랜잭션 기반으로 디지털화**한 시스템이다.

---

## 빠른 실행 (로컬, SQLite — 1번에 실행)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows (mac/linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # (mac/linux: cp .env.example .env)
python -m uvicorn main:app --reload
```

- 서버: http://localhost:8000
- 문서(Swagger): http://localhost:8000/docs
- 헬스: http://localhost:8000/health

시작 시 SQLite(`chym_aki.db`)가 생성되고 시드 데이터가 자동 적재된다.

### 데모 계정

| 역할 | 아이디 | 비밀번호 |
|---|---|---|
| 관리자 | `admin` | `Admin2026!` |
| 응급의학과 | `er_kim` | `Emer2026!` |
| 신장내과 | `neph_hong` | `Neph2026!` |
| 병리과 | `path_lee` | `Path2026!` |

---

## Docker 실행 (PostgreSQL)

```bash
cd backend
docker compose -f docker/docker-compose.yml up --build
```

`db`(Postgres) 헬스체크 통과 후 `backend`가 기동되며 시드가 적재된다.

---

## 아키텍처 (Clean Architecture / SOLID)

```
api          → 요청 처리 (얇게: 검증·위임만, 비즈니스 로직 금지)
services     → 비즈니스 로직 + 트랜잭션 경계
repositories → DB 접근 (Repository Pattern)
models       → SQLAlchemy ORM
schemas      → Pydantic DTO (camelCase ↔ snake_case 자동변환)
core         → 설정·DB·보안·예외·RBAC·쿼리최적화
```

- **DIP**: API → Service → Repository 단방향 의존.
- **SRP**: 파일 1개 = 1 역할 (`bed_service` 와 `admission_service` 분리 등).
- **OCP/LSP**: STT/NLP/AKI 는 전략(Strategy)+팩토리(Factory)로 교체 가능.

### GoF 패턴 적용

| 패턴 | 위치 |
|---|---|
| Factory | `stt/factory.py`, `nlp/factory.py`, `ai_draft/aki_model.py:get_aki_predictor` |
| Strategy | `stt/*`, `nlp/*`, `ai_draft/aki_model.py` (Model/Rule 예측기) |
| Repository | `repositories/*` |
| Service Layer | `services/*` |

---

## 트랜잭션 규칙 (핵심)

모든 상태 변경은 단일 DB 트랜잭션 + 감사로그(audit) + 타임라인(EMR 이벤트) 기록.

- **병상 배정** (`POST /api/beds/{id}/assign`): `SELECT bed FOR UPDATE` → 상태확인 → 환자확정 → 입원 중복체크 → 입원 생성 → 병상 occupied → BED_CHANGE 타임라인 + 감사 → COMMIT.
- **병상 해제** (`POST /api/beds/{id}/release`): bed lock → 활성 입원 종료 → 병상 cleaning + 환자 해제 → 타임라인 + 감사 → COMMIT.
- **AI 초안** (`POST /api/voice/draft`): transcript → NLP → AKI → SOAP → 저장 → 감사 → COMMIT.

비관적 잠금은 PostgreSQL 에서 실제 행 잠금, SQLite 에서 쓰기 직렬화로 동작한다.

---

## 타임라인 (EMR 공통 이벤트 레이어)

TIMELINE 은 독립 CRUD 가 아니라 EMR 공통 시스템이다. 모든 도메인 이벤트가 흘러든다:
`BED_CHANGE / LAB_RESULT / CONSULTATION / AI_ALERT / DIAGNOSIS_UPDATE`.

- **EMERGENCY**: 이벤트 생성(쓰기) + 조회.
- **NEPHROLOGY**: 읽기 전용(쓰기 금지) — RBAC 로 강제. AI_ALERT 자동기록만 예외.

---

## AKI 모델 연동

`AKI/modeling/final_aki_model.py` 의 2-stage 융합(Stage1 LR + Stage2 LGBM)을 재현한다.

- 학습 번들(`.pkl`)을 `AKI_MODEL_DIR`(기본 `backend/ml_models/`)에 두면 모델 추론 사용:
  - `stage1_LR_full.pkl`, `stage2_LGBM_v13_full_classweight.pkl`
- 번들이 없으면 **rule-based KDIGO 폴백**으로 자동 동작(항상 실행 가능).
- 결과: 3-class(Non-AKI / Stage1 / Stage2+3) + 위험점수(0–100) + 판정 근거.

---

## 쿼리 최적화 (`core/query_optimizer.py`)

- `SELECT *` 금지 → `load_only`.
- N+1 방지 → `selectinload`.
- 모든 목록 조회 pagination 기본 적용(상한 클램프).
- 핵심 조회 경로에 복합 인덱스 정의(병상 zone+state, 타임라인 patient+time 등).

---

## 주요 엔드포인트

| 도메인 | 메서드 · 경로 |
|---|---|
| Auth | `POST /api/auth/login` · `POST /api/auth/signup` · `GET /api/auth/me` · `GET /api/auth/accounts` · `PATCH /api/auth/accounts/{id}/approval` |
| Patients | `GET /api/patients` · `GET /api/patients/{id}` |
| Beds | `GET /api/beds` · `GET /api/beds/summary` · `GET /api/beds/emergency-patients` · `POST /api/beds/{id}/assign` · `POST /api/beds/{id}/release` |
| Nephrology | `POST /api/nephrology/aki/analyze` · `POST /api/nephrology/aki/analyze/{patient_id}` · `GET /api/nephrology/timeline/{patient_id}` |
| Voice | `POST /api/voice/transcribe` · `POST /api/voice/draft` · `GET /api/voice/drafts/{patient_id}` |
| Consultation | `GET/POST /api/consultations` · `POST /api/consultations/{id}/accept` · `POST /api/consultations/{id}/reply` |
| Timeline | `GET /api/timeline/patient/{id}` · `POST /api/timeline/event` |
| Audit | `GET /api/audit` (admin) |
