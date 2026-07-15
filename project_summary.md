# 📋 CHYM-AKI 프로젝트 종합 운영 및 경로 정리 가이드

이 문서는 CHYM-AKI EMR 프로젝트의 전체 실행/종료 방법(배치 런처), 프로젝트 파일 구조 및 핵심 코드 역할, 백엔드 경로/설정 체계에 대해 종합적으로 정리한 문서입니다. SVN 협업 및 로컬 개발 시 참고해 주시기 바랍니다.

---

## 1. 🚀 전체 실행 및 종료 (배치 런처)

프로젝트 루트 폴더(`chym_aki`)에는 개발 서버 구축 및 구동을 원클릭으로 처리할 수 있는 윈도우 배치 스크립트가 존재합니다.

### 1) 실행 런처: `start_dev.bat`
개발 환경을 자동으로 점검, 세팅하고 백엔드와 프론트엔드 서버를 동시에 구동합니다.
* **동작 순서**:
  1. **소스 코드 백업**: 실행 시각 기준으로 `backup/backup_YYYYMMDD_HHMMSS` 폴더를 생성하여 가상환경/라이브러리를 제외한 순수 소스코드를 자동 백업합니다. (설정 실패 시 이전 소스코드로 롤백하는 안전장치 탑재)
  2. **백엔드 검사 및 가상환경 세팅**: `backend/.venv` 가상환경 존재 여부를 확인하고 없으면 자동 생성 후 `pip install -r requirements.txt`를 실행합니다.
  3. **프론트엔드 라이브러리 검사**: `frontend/node_modules` 존재 여부를 확인하고 `npm install`을 자동 수행합니다.
  4. **데이터베이스 초기화 및 시딩 (PostgreSQL)**: `.env`에 정의된 데이터베이스 주소로 접근하여 테이블 스키마 생성 및 초기 데모 데이터(시드)를 주입하고 텔레메트리 스키마를 적용합니다.
  5. **서버 동시 구동**: 새로운 CMD 창을 2개 띄워 백엔드(uvicorn, 8010포트)와 프론트엔드(Vite, 5174포트) 서버를 동시에 실행합니다.

### 2) 종료 런처: `stop_dev.bat`
구동 중인 서버들을 안전하게 강제 종료하고 리소스를 반환합니다.
* **동작 내용**:
  * 실행 중인 `CHYM Backend` 및 `CHYM Frontend` 타이틀을 가진 CMD 창 프로세스를 `taskkill` 명령어로 일괄 종료합니다.

---

## 2. 📂 디렉토리 구조 및 핵심 파일 역할

전체 프로젝트는 크게 백엔드, 프론트엔드, 그리고 머신러닝/연구 산출물이 담긴 AKI 폴더로 나뉩니다.

```text
chym_aki/
├── start_dev.bat           # 개발 서버 구동 런처
├── stop_dev.bat            # 개발 서버 종료 런처
├── conda_setup_guide.md    # Conda 가상환경 세팅 가이드
├── svn_ignore_guide.md     # SVN Ignore 속성 등록 가이드
├── AKI/                    # 머신러닝 연구 산출물 (전처리 데이터셋, 피처, 모델 학습 코드)
│   ├── preprocessing/      # 최종 데이터셋_전처리 완료된 (train/valid/test_final.csv)
│   └── modeling/           # 모델링 실험 코드 및 모델 출력 폴더
├── backend/                # FastAPI 백엔드
│   ├── core/               # 데이터베이스 연결, JWT 보안, 전역 설정, 예외 처리
│   ├── api/                # REST API 라우터 (auth, patients, beds, nephrology, voice 등)
│   ├── services/           # 비즈니스 도메인 서비스 로직 (핵심 기능 구현체)
│   ├── models/             # SQLAlchemy DB ORM 엔티티 모델 정의
│   ├── schemas/            # Pydantic 데이터 검증 및 직렬화 스펙
│   ├── db/                 # DB 시드 데이터 입력(seed.py), 피처 CSV 로더 스크립트
│   ├── ml_models/          # 학습 완료된 AKI 2-stage 모델 파일 (.pkl) 및 로컬 재학습 스크립트
│   ├── telemetry/          # 사용자 행동 로깅 및 성능 모니터링 미들웨어/수집 파이프라인
│   ├── .env                # 데이터베이스, CORS, JWT 비밀값 등 로컬 환경 변수 파일
│   └── main.py             # 백엔드 진입점 (FastAPI 인스턴스 구성 및 미들웨어 마운트)
└── frontend/               # React + Vite 프론트엔드
    ├── src/                # React 컴포넌트, 상태 관리, API 연동 코드
    ├── package.json        # 프론트엔드 의존성 및 스크립트 정의
    └── vite.config.ts      # Vite 빌드 및 개발 서버 프록시 설정
```

---

## 3. ⚙️ 백엔드 핵심 설정 및 경로 처리 체계

### 1) `.env` 환경 변수 관리
백엔드 설정은 `backend/.env` 파일에서 집중 관리하며, 주요 변수들은 다음과 같습니다:
* **`DATABASE_URL`**: PostgreSQL 연결 DSN (`postgresql+psycopg2://username:password@IP:5432/dbname`)
* **`CORS_ORIGINS`**: 웹 브라우저 CORS 허용 주소 목록 (로컬 IP, localhost 등 쉼표 구분 입력)
* **`AKI_MODEL_DIR`**: 예측 모델이 보관된 디렉토리 주소 (기본값: `./ml_models`)

### 2) 동적 절대경로 변환 (중요)
소스 코드 내부에서는 협업 개발자 간의 로컬 경로 차이(예: `C:\dev` vs `D:\dev\workspace_proj`)로 인한 에러를 방지하기 위해 파이썬 내장 객체인 `Path(__file__)`을 기반으로 **상대 경로를 동적 절대경로로 자동 변환**하여 사용합니다.
* **설정 모듈 (`backend/core/config.py`)**:
  ```python
  BACKEND_DIR = Path(__file__).resolve().parent.parent # backend/ 절대경로 탐색
  
  @property
  def aki_model_path(self) -> Path:
      p = Path(self.aki_model_dir)
      # 상대 경로 입력 시 자동으로 백엔드 절대 경로를 조합하여 절대 경로로 반환
      return p if p.is_absolute() else (BACKEND_DIR / p)
  ```
* **모니터 서비스 (`backend/services/icu_monitor_service.py`)**:
  ```python
  BACKEND = Path(__file__).resolve().parent.parent # backend 절대경로
  MODELS = BACKEND / "ml_models"
  FEATURE_CSV_DIR = BACKEND.parent / "AKI" / "preprocessing" / "final_dataset"
  ```
  따라서 개발자의 컴퓨터가 Windows든 Linux든 상관없이 폴더 구조만 유지된다면 완벽하게 데이터셋과 모델 경로를 추적해 냅니다.

### 3) scikit-learn 라이브러리 하위 호환성 Monkey-patch
파이썬 가상환경에 세팅된 scikit-learn 최신 라이브러리(1.6.0 이상)와 예전 버전으로 생성된 학습 모델(`.pkl`) 간의 구조 차이로 인한 `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` 에러를 자동으로 예방해 주는 호환성 코드가 탑재되어 있습니다.
* **해당 파일**: `icu_monitor_service.py`, `model_metrics_service.py`, `ai_draft/aki_model.py`
* **동작**: 모델 역직렬화(Unpickling) 즉시 누락된 속성을 검사하여 주입합니다.
  ```python
  lr = stage1["model"]
  if not hasattr(lr, "multi_class"):
      lr.multi_class = "auto" # 최신 sklearn 구조에 맞춤 대응
  ```
  *(참고: 만약 모델을 최신 라이브러리 환경에서 새로 학습하여 저장하고 싶다면 `backend/ml_models/train_aki_models.py`를 직접 실행하시면 가상환경 버전에 맞는 깨끗한 모델 파일이 재생성됩니다.)*
 