# CHYM-AKI 프로젝트 전체 폴더 및 파일 구조 상세 안내

이 문서는 `chym_aki` 프로젝트의 최상위 폴더부터 하위 폴더, 그리고 개별 파일들의 목적과 역할을 상세하게 정리한 문서입니다.

---

## 📁 1. 최상위 디렉토리 (Root Directories)

### `backend/`
FastAPI 기반의 백엔드 서버 소스코드가 담긴 폴더입니다.
- **`api/`**: REST API 라우터 및 엔드포인트 정의 (컨트롤러 역할)
- **`core/`**: 설정, 예외 처리, 데이터베이스 연결 등 핵심 공통 로직
- **`db/`**: 데이터베이스 마이그레이션 및 시드 데이터
- **`docker/`**: 백엔드 서버 배포를 위한 `Dockerfile` 및 `docker-compose.yml` (PostgreSQL 등)
- **`models/` & `schemas/`**: ORM 데이터베이스 모델(Entity) 및 Pydantic 데이터 검증 스키마
- **`repositories/`**: 데이터베이스 접근 로직 (Repository 패턴 구현)
- **`services/`**: 비즈니스 로직(핵심 기능)이 구현된 서비스 계층
- **`ml_models/` & `wsi/`**: 머신러닝 추론 파이프라인 및 WSI(Whole Slide Image) 처리 코드
- **`main.py`**: FastAPI 백엔드 어플리케이션 진입점 (앱 실행)
- **`requirements.txt`**: 백엔드 구동에 필요한 파이썬 라이브러리(패키지) 목록

### `frontend/`
React + TypeScript 기반의 웹 클라이언트(프론트엔드) 소스코드가 담긴 폴더입니다.
- **`src/`**: 실제 화면을 구성하는 모든 코드
  - `components/`: 재사용 가능한 UI 컴포넌트 모음
  - `features/`: 도메인/기능별로 분리된 로직 및 컴포넌트 (예: pathology, retrieval)
  - `layouts/`: 화면의 공통 레이아웃 (TopBar, Sidebar 등)
  - `services/`: 백엔드 API와의 통신을 담당하는 함수 모음
  - `store/`: 전역 상태 관리 (Zustand 기반)
  - `types/`: TypeScript 타입 정의 모음
- **`package.json`**: 프론트엔드 구동을 위한 npm 패키지 의존성 목록
- **`vite.config.ts`**: Vite 빌드 도구 설정 파일

### `Pathology_model/`
병리 이미지 AI 분석을 위한 딥러닝 모델 소스코드 및 연구 데이터 폴더입니다.
- **`mil/`**: 다중 인스턴스 학습(Multiple Instance Learning) 관련 모델 아키텍처 코드
- **`models/`**: CTransPath 등 백본 모델 가중치 및 설정 코드
- **`generate_v1_heatmap.py`**: AI 추론 결과를 바탕으로 시각화된 히트맵(Heatmap)을 생성하는 스크립트
- **`cdss_v1_e2e_test.py`**: 시스템의 처음부터 끝까지(End-to-End) 제대로 동작하는지 테스트하는 스크립트

### `scripts/`
프로젝트 운영 및 관리에 필요한 유틸리티 스크립트가 모여 있습니다.
- **`ensure_postgres.ps1`**: PostgreSQL 데이터베이스가 정상 구동 중인지 확인하는 파워쉘 스크립트
- **`rollback_notifier.py`**: 시스템 롤백 발생 시 알림을 보내는 관리용 파이썬 스크립트

### `tests/`
코드의 안정성을 확보하기 위한 유닛 테스트(Unit Test) 및 통합 테스트 코드 모음입니다.
- **`test_*.py`**: 기능별(환자 조회, 병리 분석, 알림 등) 동작을 자동으로 검증하는 모듈별 테스트 코드 모음
- **`vv_runners/`**: 시스템 검증(V&V, Verification and Validation) 및 자동 테스트 리포트 PDF를 생성하는 도구들

### 🚫 버전 관리 및 대용량 데이터 폴더 (Git 제외됨)
- **`aki_wsi/`, `aki_wsi_ai/`**: 원본 병리 이미지(WSI) 및 수 기가바이트에 달하는 AI 모델 가중치(`.safetensors` 등)가 저장된 폴더 (용량 제한으로 제외)
- **`data/`, `uploads/`, `pacs_cache/`**: 대규모 학습 데이터셋 및 PACS 서버에서 내려받은 대용량 의료 영상(DICOM) 캐시 폴더
- **`wandb/`**: Weights & Biases (머신러닝 실험 로깅 플랫폼)의 로컬 실험 기록 및 캐시 파일
- **`.agents/`, `.claude/`**: AI 코딩 에이전트 구동 기록 및 설정 파일
- **`.svn/`, `.git_backup/`**: 이전 버전 관리 시스템인 SVN의 기록 및 임시 백업 폴더

---

## 📄 2. 기능별 분리 폴더 (`utils/`, `analysis_results/`)

최상위 폴더가 지저분해지는 것을 방지하고 나중에 쉽게 찾을 수 있도록, 성격이 비슷한 개별 파일들을 나누어 정리했습니다.

### 🛠️ `utils/` 폴더 (유틸리티 스크립트)
각종 파이썬 스크립트 파일들이 모여있는 폴더입니다.
- **데이터 무결성 검증 용도**: `check_cols.py`, `check_coords.py`, `check_csvs.py`, `check_img.py` 등
- **결과 시각화(차트 생성) 용도**: `make_chart.py`, `make_scatter.py`, `make_n78_chart.py`, `make_notion_chart.py` 등
- **데이터 변환 및 추출 용도**: `dump_experiments.py`, `dump_jsons.py`, `md_to_pdf.py`, `md_to_pdf_chrome.py` 등
- **기타 수치 계산 용도**: `get_qwk.py` 등

### 📊 `analysis_results/` 폴더 (데이터 및 결과 리포트)
분석 과정에 쓰인 메타데이터와 최종 결과물이 모여있는 폴더입니다.
- **대용량 엑셀/데이터(CSV)**: `kpmp.csv`, `patches_manifest.csv`, `split_manifest.csv`
- **분석 로그 및 결과 리포트**: `ab_events.jsonl`, `Integrated_Master_Report.html`
- **디버그용 임시 텍스트 파일**: `debug.txt`, `diff.txt`

### ⚙️ (루트 경로 유지) 필수 실행 및 설정 파일
실행 편의성과 시스템 정상 동작(라이브러리 인식 등)을 위해 바깥(루트)에 그대로 둔 파일들입니다.
- **실행 스크립트**: `start_dev.bat`, `stop_dev.bat`, `run_train_wandb.ps1`, `run_overnight_171.bat`
- **환경 설정 파일**: `package.json`(프론트엔드), `pytest.ini`(테스트), `pdf_config.json`(문서 변환)
