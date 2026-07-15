"""애플리케이션 설정 (Single Source of Truth).

이 모듈의 책임은 환경변수/.env 로드와 타입 안전한 설정 노출 단 하나다.
- 다른 레이어는 절대 os.environ 을 직접 읽지 않는다 → 설정 접근은 여기로 단일화.
- pydantic-settings 로 검증/형변환을 수행한다.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 디렉터리 절대경로 (모델 디렉터리·SQLite 상대경로 기준점)
BACKEND_DIR = Path(__file__).resolve().parent.parent
# 리포 루트(C:/team/chym_aki) — 데이터/CDSS 상대경로 기준점(머신 비의존)
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """환경설정 값. .env 파일과 OS 환경변수에서 로드된다."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 앱 ---
    app_name: str = "RENAI"
    app_env: str = "local"
    debug: bool = True

    # --- DB ---
    database_url: str = "sqlite:///./chym_aki.db"
    # 앱 테이블 스키마(PostgreSQL). mimic4 DB 에서 MIMIC 테이블과 분리하기 위한 전용 스키마.
    # SQLite 에서는 스키마 개념이 없어 무시된다.
    app_schema: str = "chym"

    # --- 보안(JWT) ---
    # RS256(비대칭): 개인키로 서명(8010만), 공개키로 검증(8001/PACS 등). 검증자가 털려도 위조 불가.
    # HS256 으로 되돌리려면 jwt_algorithm=HS256 + jwt_secret 사용(하위호환).
    jwt_algorithm: str = "RS256"
    jwt_secret: str = "dev-only-change-me-in-production-0123456789abcdef"  # HS256 폴백용
    jwt_private_key_path: str = "keys/jwt_private.pem"   # 서명용 개인키 — 절대 배포/커밋 금지
    jwt_public_key_path: str = "keys/jwt_public.pem"     # 검증용 공개키 — 자유 배포 가능
    access_token_expire_minutes: int = 720

    # --- CORS ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- AKI 모델 ---
    aki_model_dir: str = "./ml_models"
    aki_stage1_file: str = "stage1_LR_full.pkl"
    aki_stage2_file: str = "stage2_LGBM_v13_full_classweight.pkl"

    # --- STT ---
    stt_strategy: str = "passthrough"
    whisper_model_size: str = "base"

    # --- 시드 ---
    seed_on_startup: bool = True

    # --- PACS ---
    pacs_base_url: str = ""
    pacs_service_id: str = ""
    pacs_service_api_key: str = ""
    pacs_employee_id: str = ""  # 케이스 목록 조회용 서비스 계정 사번

    # --- WSI ---
    # ABMIL feature 파일·모델 체크포인트가 있는 로컬 베이스 디렉토리
    wsi_base_dir: str = ""
    # PACS에서 다운로드한 슬라이드를 캐시할 디렉토리
    wsi_cache_dir: str = "./wsi_cache"

    # --- CDSS v5 디스크립터(OOF) ---
    # cdss_v5(CLAM-lite/CTransPath) 의 pooled-OOF 예측·메트릭이 있는 디렉토리.
    # 라이브 KPMP 모델이 미예측하는 ati_severity·immune(=tubulitis) 등 실제 학습 헤드를
    # 환자별로 공급한다(v6.1 §6 tubular injury spectrum). 파일 부재 시 엔드포인트는 빈 응답.
    # 상대경로면 리포 루트 기준으로 해석(cdss_core_path). 다른 PC/에어갭에서도 동작.
    # 절대경로가 필요하면 .env 의 CDSS_CORE_DIR 로 덮는다.
    cdss_core_dir: str = "data/cdss_core"
    cdss_oof_file: str = "oof_cdss_v5_full_72p_ctranspath.csv"
    cdss_metrics_file: str = "mil_cv_cdss_v5_full_72p_ctranspath.json"

    @property
    def cors_origin_list(self) -> list[str]:
        """콤마 구분 문자열을 리스트로 변환 (CORS 미들웨어 입력용)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (BACKEND_DIR / p)

    @property
    def jwt_signing_key(self) -> str:
        """토큰 서명 키 — RS256이면 개인키 PEM, HS256이면 공유 시크릿."""
        if self.jwt_algorithm.startswith("RS"):
            return self._resolve(self.jwt_private_key_path).read_text(encoding="utf-8")
        return self.jwt_secret

    @property
    def jwt_verify_key(self) -> str:
        """토큰 검증 키 — RS256이면 공개키 PEM, HS256이면 공유 시크릿."""
        if self.jwt_algorithm.startswith("RS"):
            return self._resolve(self.jwt_public_key_path).read_text(encoding="utf-8")
        return self.jwt_secret

    @property
    def aki_model_path(self) -> Path:
        """AKI 모델 디렉터리 절대경로."""
        p = Path(self.aki_model_dir)
        return p if p.is_absolute() else (BACKEND_DIR / p)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def cdss_core_path(self) -> Path:
        """CDSS 코어 데이터 루트 절대경로. 상대경로면 리포 루트(REPO_DIR) 기준으로 해석."""
        p = Path(self.cdss_core_dir)
        return p if p.is_absolute() else (REPO_DIR / p)


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴. lru_cache 로 프로세스당 1회만 생성한다."""
    return Settings()


settings = get_settings()
