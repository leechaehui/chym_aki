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

## 📄 2. 최상위(Root) 주요 개별 파일 목록

### 🐍 파이썬 유틸리티 스크립트 (*.py)
- **데이터 무결성 검증 용도**
  - `check_cols.py`, `check_coords.py`, `check_csvs.py`, `check_img.py` 등: CSV 파일의 열 데이터 누락 여부, 좌표 값 정합성, 이미지가 정상적인지 검사합니다.
- **결과 시각화(차트 생성) 용도**
  - `make_chart.py`, `make_scatter.py`, `make_n78_chart.py`, `make_notion_chart.py`: 모델 학습 결과나 분석 데이터를 바탕으로 통계 그래프를 그려 파일로 저장합니다.
- **데이터 변환 및 추출 용도**
  - `dump_experiments.py`, `dump_jsons.py`: DB 내역이나 실험 결과들을 한데 모아 덤프(백업)합니다.
  - `md_to_pdf.py`, `md_to_pdf_chrome.py`: 마크다운 기반의 텍스트 문서를 깔끔한 PDF 파일 포맷으로 일괄 변환해줍니다.
- **기타 수치 계산 용도**
  - `get_qwk.py`: QWK (Quadratic Weighted Kappa) 등의 성능 통계 지표를 계산합니다.

### 📜 마크다운 문서 (*.md)
- **`project_summary.md`**: 프로젝트 전체의 목적, 아키텍처 구조, 활용된 기술 스택 등을 한눈에 요약한 핵심 문서입니다.
- **`DEMO_SCENARIO_GUIDE.md`**: 발표나 데모 시연을 진행할 때 참고하기 위해 작성된 스크립트 및 시나리오 안내서입니다.
- **`PATIENT_SLIDE_LINKING_TODO.md`**: 환자 데이터베이스와 실제 병리 슬라이드 이미지를 매핑(연결)하는 작업의 남은 할 일(TODO) 목록입니다.
- **`conda_setup_guide.md`**: 개발 환경 세팅을 위해 로컬 컴퓨터에 파이썬 가상환경(Conda)을 설정하는 방법을 정리한 문서입니다.

### ⚙️ 실행 및 설정 파일
- **`start_dev.bat` / `stop_dev.bat`**: 로컬 컴퓨터에서 백엔드, 프론트엔드 환경을 클릭 한 번에 모두 실행하거나 일괄 종료하는 단축 배치 파일입니다.
- **`run_train_wandb.ps1` / `run_overnight_171.bat`**: 딥러닝 모델 학습을 시작하거나, 밤새 오래 걸리는 훈련 작업을 자동화하기 위한 스크립트입니다.
- **`pytest.ini`**: 파이썬 자동화 테스트 프레임워크인 Pytest의 세부 실행 옵션을 정의한 설정 파일입니다.
- **`pdf_config.json`**: 문서를 PDF로 변환할 때 사용할 용지 크기나 여백 등을 지정하는 설정 파일입니다.

### 📊 분석 데이터 및 텍스트 파일
- **`kpmp.csv`, `patches_manifest.csv`, `split_manifest.csv`**: 대용량 환자 메타데이터 및 병리 이미지에서 쪼개낸 패치 좌표들의 매니페스트 파일들입니다.
- **`ab_events.jsonl`**: 특정 분석 과정에서 발생한 이벤트 로깅 데이터를 한 줄씩 읽기 편한 JSON Lines 형태로 정리한 파일입니다.
- **`Integrated_Master_Report.html`**: 여러 가지 모델 예측 및 실험 분석 결과를 하나로 취합하여 웹 브라우저에서 열어볼 수 있도록 렌더링한 마스터 리포트 파일입니다.
- **`debug.txt` / `diff.txt`**: 프로그램 디버깅 중 찍어본 로그나, 파일 간의 차이점 비교 결과 등을 임시로 저장해둔 텍스트 파일입니다.
