"""WSI 추론 서버 설정 — 경로/임계/런타임의 단일 소스(Single Source of Truth).

경계 원칙: 도메인/서비스는 절대경로를 모른다. 모든 외부 경로는 여기로 주입된다.
이식성(다른 PC 연동) 원칙 — 절대경로 하드코딩 금지:
- CODE/MODEL 루트(pkg_root): 이 파일 기준으로 자동 도출(repo/Pathology_model). 환경 무관.
- DATA 루트(data_root)·SVS 루트(svs_root): Pathology_model/config.json 에서 읽음(머신별 1곳만 수정).
- 모든 값은 환경변수(CHYM_WSI_*)로 재정의 가능 → 12-factor.
PC마다 다른 것은 config.json(또는 env)만 바꾸면 되고, 코드/번들은 그대로 이식된다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv_to_environ() -> None:
    """backend/.env → os.environ 주입(1회, setdefault=OS env 우선).
    8001 은 pydantic .env 로드를 안 쓰므로, CdssEngine(Pathology_model) 등 os.getenv/os.environ 로
    읽는 하위까지 .env 값(CDSS_MODEL_VARIANT·HF_TOKEN 등)이 닿게 하려면 여기서 명시 로드해야 한다."""
    _envf = Path(__file__).resolve().parents[2] / ".env"       # backend/.env
    if not _envf.exists():
        return
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _s = _line.strip()
        if _s.startswith("#") or "=" not in _s:
            continue
        _k, _v = _s.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


_load_dotenv_to_environ()


@dataclass(frozen=True)
class WsiSettings:
    # ── 경로 경계 ──────────────────────────────────────────────────────────
    pkg_root: Path        # .../Pathology_model (코드 + ordinal_ms.pt)
    data_root: Path       # .../chym_aki (data/embeddings, data/raw/wsi_*)
    cache_dir: Path       # 분석 결과 캐시(stateless 서비스 → 상태는 여기로 외부화)
    svs_root: Path        # SVS 원본 검색 루트 — manifest 경로가 깨졌을 때 파일명으로 재해결(데이터 이동 대응)

    # ── 임베딩/모델 파라미터 ──────────────────────────────────────────────
    encoder: str = "ctranspath"
    mags: tuple[str, ...] = ("10", "40")
    # 프론트가 노출하는 표시 stain(WsiStain). 분석 bag 은 전체 stain 사용.
    display_stains: tuple[str, ...] = ("HE", "MT")
    device: str = "cpu"

    # ── 런타임 ────────────────────────────────────────────────────────────
    port: int = 8001
    cors_origins: tuple[str, ...] = ("http://localhost:5174", "http://127.0.0.1:5174")

    # ── PACS (선택) — 자격증명은 env 만(비밀: config.json·프론트 저장 금지). 서버↔서버 BFF. ──
    pacs_base_url: str = ""
    pacs_service_id: str = ""
    pacs_service_api_key: str = ""
    pacs_employee_id: str = ""
    # 오프라인 모드: 슬라이드 목록을 라이브 PACS 대신 로컬 피처(.pt)로 구성한다.
    # (PACS 미가용 네트워크에서 목록 조회가 타임아웃으로 지연되는 것을 방지. 캐시된 슬라이드의
    #  이미지 뷰어는 여전히 로컬 DICOM 캐시로 동작하므로 PACS 자격증명은 그대로 두어도 된다.)
    offline: bool = False

    # ── HuggingFace (선택) — hibou-L 등 gated 모델 다운로드용 토큰. ──
    hf_token: str = ""
    # 로컬에 미리 받아둔 hibou-L 스냅샷(config/safetensors/모델 코드) 경로. 있으면 허브 다운로드 생략.
    hibou_local_path: str = ""

    @property
    def pacs_enabled(self) -> bool:
        """자격증명 4개가 모두 있을 때만 PACS 연동 활성."""
        return bool(self.pacs_base_url and self.pacs_service_id
                    and self.pacs_service_api_key and self.pacs_employee_id)

    @property
    def index_csv(self) -> Path:
        return self.data_root / "data" / "embeddings" / self.encoder / "index.csv"

    @property
    def patches_manifest(self) -> Path:
        return self.pkg_root / "artifacts" / "patches_manifest.csv"

    @property
    def he_feature_dir(self) -> Path:
        return self.data_root / "features_phikon_512_he"

    @property
    def mt_feature_dir(self) -> Path:
        return self.data_root / "features_hibou_512_mt"

    @property
    def pas_feature_dir(self) -> Path:
        return self.data_root / "features_ctranspath_512_pas"

    @property
    def he_ctranspath_feature_dir(self) -> Path:
        """CdssEngine(멀티스테인 융합)용 HE — ABMIL용 phikon(he_feature_dir)과 인코더가 달라 별도 보관.
        CdssEngine 체크포인트가 768차원(ctranspath)으로 학습돼 HE도 PAS/MT와 같은 인코더가 필요하다."""
        return self.data_root / "features_ctranspath_512_he"

    @property
    def stain_ref_path(self) -> Path:
        """CdssEngine(ctranspath, PAS/HE_CT) train/serve 정규화 일치용 동결 reference.
        mil/build_stain_ref.py 가 학습과 동일 규칙으로 생성. 없으면 서빙은 정규화 생략(경고)."""
        return self.pkg_root / "models" / "cdss_shadow" / "stain_ref.json"

    @property
    def he_ckpt_path(self) -> Path:
        return self.data_root / "checkpoints_abmil_phikon_512_he" / "abmil_he_final.pt"

    @property
    def mt_ckpt_path(self) -> Path:
        return self.data_root / "checkpoints_abmil_hibou_512_mt" / "abmil_mt_final.pt"


def _pkg_config(pkg_root: Path) -> dict:
    """Pathology_model/config.json 읽기(경로 단일소스). 없으면 빈 dict."""
    try:
        return json.loads((pkg_root / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env(key: str, default: str = "") -> str:
    """OS 환경변수 우선, 없으면 backend/.env 파싱. (WSI 서버는 pydantic .env 로드를 안 쓰므로 직접 읽음.)"""
    if os.getenv(key):
        return os.environ[key]
    env = Path(__file__).resolve().parents[2] / ".env"      # backend/.env
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return default


@lru_cache(maxsize=1)
def get_settings() -> WsiSettings:
    """프로세스 단일 설정. 우선순위: 환경변수 > config.json > 자동도출 기본값.

    - pkg_root: 이 파일 위치에서 자동 도출(repo 루트/Pathology_model) → PC 무관 이식.
    - data_root/svs_root: config.json 에서(머신별 1곳). env 로 덮어쓰기 가능.
    """
    repo_root = Path(__file__).resolve().parents[3]            # .../<repo>
    pkg_root = Path(os.getenv("CHYM_WSI_PKG_ROOT", str(repo_root / "Pathology_model")))
    cfg = _pkg_config(pkg_root)

    data_root = Path(_env("CHYM_WSI_DATA_ROOT") or cfg.get("data_root") or str(repo_root))
    # SVS 루트: env > config.json[wsi_svs_root] > data_root(features 와 같은 루트)
    svs_root = Path(_env("CHYM_WSI_SVS_ROOT") or cfg.get("wsi_svs_root") or str(data_root))

    cache_dir = Path(_env("CHYM_WSI_CACHE_DIR")
                     or str(Path(__file__).resolve().parents[2] / "data" / "wsi_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    return WsiSettings(
        pkg_root=pkg_root, data_root=data_root, cache_dir=cache_dir,
        svs_root=svs_root, device=os.getenv("CHYM_WSI_DEVICE", "cpu"),
        # PACS 자격증명 — env/​.env 단일 소스(비밀). 없으면 pacs_enabled=False 로 비활성.
        pacs_base_url=_env("PACS_BASE_URL"),
        pacs_service_id=_env("PACS_SERVICE_ID"),
        pacs_service_api_key=_env("PACS_SERVICE_API_KEY"),
        pacs_employee_id=_env("PACS_EMPLOYEE_ID"),
        offline=_env("CHYM_WSI_OFFLINE").strip().lower() in ("1", "true", "yes", "on"),
        hf_token=_env("HF_TOKEN"),
        hibou_local_path=_env("HIBOU_L_LOCAL_PATH"),
    )


# ── A/B versioned caching — 캐시 키에 model_variant·expl·norm 버전 포함(실험 통제) ──
def active_model_variant() -> str:
    """torch 로드 없이 '의도된' 모델 variant 해석(캐시 키·로그용). ln 파일 없으면 baseline.
    (엔진의 실제 로드 실패 fallback 은 별도지만, 파일 존재 기준이라 통상 일치.)"""
    s = get_settings()
    shadow = s.pkg_root / "models" / "cdss_shadow"
    v = os.getenv("CDSS_MODEL_VARIANT", "ln").lower()
    if v == "pas" and (shadow / "ordinal_pas.pt").exists():
        return "pas"
    if v == "ln" and (shadow / "ordinal_ms_ln.pt").exists():
        return "ln"
    return "baseline"


def expl_versions() -> tuple[str, str]:
    """EXPL 시각화·정규화 버전(캐시 키·A/B). expl_visual_calibration.json 단일 소스.
    정규화 파라미터(p99·coverage)가 바뀌면 norm 버전이 바뀌어 캐시 자동 무효화."""
    try:
        c = json.loads((Path(__file__).resolve().parents[1] / "expl_visual_calibration.json")
                       .read_text(encoding="utf-8"))
        ev = f"expl{c.get('schema_version', 1)}"
        nv = f"p{int(c.get('norm_percentile', 99))}c{int(round(float(c.get('overlay_coverage_top_frac', 0.10)) * 100))}"
        return ev, nv
    except Exception:
        return "expl1", "p99c10"


def cache_version(variant: str | None = None) -> str:
    """캐시 키 버전 접미사 = model_variant__expl__norm. A/B·재현·오염방지의 단일 통제점.
    variant 지정 시 그 값 사용(GET=requested, SET=resolved 로 Intent/Execution 분리).
    expl/norm 은 mapper 의 '실제 frozen 값'에서 도출(1:1 보장). mapper 미가용 시 JSON 폴백."""
    mv = variant if variant is not None else active_model_variant()
    try:
        from wsi.domain.mapper import EXPL_VERSION, NORM_VERSION
        return f"{mv}__{EXPL_VERSION}__{NORM_VERSION}"
    except Exception:
        ev, nv = expl_versions()
        return f"{mv}__{ev}__{nv}"
