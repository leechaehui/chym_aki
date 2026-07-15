"""SlideRepository 구현 — PACS 전용. 로컬 SVS 스캔 없음.

슬라이드 목록은 PACS 서버에서만 가져온다.
피처(.pt)는 slide_id(PACS case_id, 전역 유일) 기준 1:1로 찾는다 — 환자ID/검체번호로
합치면 같은 환자의 서로 다른 물리 슬라이드(연속 절편 1of2/2of2 등)가 뒤섞여 엉뚱한
슬라이드의 attention이 그려지는 사고가 난다(실제로 겪음). 환자ID는 오직 "이 슬라이드에
없는 다른 stain을 같은 환자에서 보충"하는 멀티스테인 bag 병합(CdssEngine)에만 쓴다.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from wsi.core.config import WsiSettings
from wsi.core.errors import SlideNotFoundError
from wsi.domain.ports import StainCoords
from wsi.schemas.wsi import WsiSlide

_log = logging.getLogger("wsi.repo")


def _specimen_code(text: str) -> str | None:
    """환자 매칭 키 추출 — 우선순위:
    1) 괄호 안 환자ID "(NN-NNNNN)" — stain 간 실제로 겹치는 진짜 매칭 키.
    2) 괄호 없이 그 자체가 환자ID 형식(예: "30-10018_PAS").
    3) 검체 접수번호 S-YYYY-NNNNNN — 위 둘이 없을 때만(환자 매칭 보장 안 됨, 그룹핑용 폴백).
    """
    m = re.search(r'\(([0-9]{2,3}-[0-9]{3,7})\)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?<![\w-])([0-9]{2,3}-[0-9]{4,7})(?!\d)', text)
    if m:
        return m.group(1)
    m = re.search(r'(S-\d{4}-\d+)', text.upper())
    return m.group(1) if m else None


def _detect_stain(case_code: str, description: str | None) -> str | None:
    code = case_code.upper()
    desc = (description or "").upper()
    if "_HE" in code or "H&E" in desc or "HE" in desc:
        return "HE"
    if "_MT" in code or "_TRI" in code or "MT" in desc or "TRICHROME" in desc:
        return "MT"
    if "_PAS" in code or "PAS" in desc:
        return "PAS"
    return None


class ManifestSlideRepository:
    def __init__(self, settings: WsiSettings):
        self._s = settings

        # ── 피처 인덱스: slide_id -> {stain_tag: Path} — slide_id(PACS case_id)는 전역
        # 유일이라 충돌이 절대 없다. 파일명 규칙: {slide_id}_{case_code}.pt
        self._pt_by_slide: dict[str, dict[str, Path]] = {}
        # ── 환자ID -> stain -> 대표 slide_id 1개. 멀티스테인 bag 병합(CdssEngine)에만 쓴다 —
        # 같은 환자가 같은 stain 슬라이드를 여러 장 가지면 먼저 스캔된 것 하나만 대표로 삼는다
        # (그 슬라이드 "자신"을 볼 때는 절대 이 경로를 안 타므로 안전 — load_bag 참고).
        self._specimen_stain_slide: dict[str, dict[str, str]] = {}

        for feat_dir, stain in [
            (settings.he_feature_dir, "HE"),
            (settings.mt_feature_dir, "MT"),
            (settings.pas_feature_dir, "PAS"),
            (settings.he_ctranspath_feature_dir, "HE_CT"),   # CdssEngine 전용(768d) — "HE"와 별도 키
        ]:
            if feat_dir.is_dir():
                for pt in feat_dir.glob("*.pt"):
                    slide_id = pt.stem.split("_", 1)[0]
                    self._pt_by_slide.setdefault(slide_id, {})[stain] = pt
                    spec = _specimen_code(pt.stem)
                    if spec and stain not in self._specimen_stain_slide.get(spec, {}):
                        self._specimen_stain_slide.setdefault(spec, {})[stain] = slide_id
            else:
                _log.warning("피처 디렉토리 없음: %s", feat_dir)
        _log.info("피처 인덱스: %d 슬라이드", len(self._pt_by_slide))

        # PACS case_id → specimen_code 매핑 (list_slides 호출 시 채워짐)
        self._case_specimen: dict[str, str] = {}

    # ── 목록 (PACS 우선, 미가용 시 로컬 피처로 오프라인 폴백) ────────────────
    def list_slides(self, stain: str) -> list[WsiSlide]:
        # 오프라인 모드: PACS 를 아예 시도하지 않고(타임아웃 지연 제거) 로컬 피처로 목록 구성.
        if getattr(self._s, "offline", False):
            return self._list_slides_offline(stain)
        try:
            from wsi.infra.engine_factory import get_pacs_repository
            pacs_cases = get_pacs_repository().list_wsi_cases()
        except Exception as e:
            # PACS 미가용(자격증명 미설정/오프라인/타 네트워크) → 로컬 피처(.pt)로 목록 구성.
            # 데모/오프라인 환경에서도 슬라이드 목록·AI 분석이 동작하도록 한다.
            _log.warning("PACS 목록 조회 실패 — 로컬 피처 기반 오프라인 목록으로 폴백: %s", e)
            return self._list_slides_offline(stain)

        slides: list[WsiSlide] = []
        for c in pacs_cases:
            case_id   = c.get("case_id")
            case_code = c.get("case_code")
            desc      = c.get("description")
            if not case_id or not case_code:
                continue
            if _detect_stain(case_code, desc) != stain:
                continue

            # 표준 명명(S-YYYY-NNNNNN)이 없는 케이스(PAS 데모 케이스 등)는 case_id 자체를 키로 폴백.
            spec = _specimen_code(case_code) or case_id
            self._case_specimen[case_id] = spec

            has_features = stain in self._pt_by_slide.get(case_id, {})
            cached = self._is_pacs_cached(case_id)

            slides.append(WsiSlide(
                slide_id=case_id,
                case_code=case_code,
                stain=stain,
                has_features=has_features,
                cached=cached,
                is_pacs=True,
                description=desc,
            ))

        return sorted(slides, key=lambda s: s.case_code)

    # 스테인 → 대표 피처 디렉토리(ABMIL HE=phikon / MT=hibou / PAS=ctranspath).
    def _feature_dir_for(self, stain: str) -> Path | None:
        return {
            "HE": self._s.he_feature_dir,
            "MT": self._s.mt_feature_dir,
            "PAS": self._s.pas_feature_dir,
        }.get(stain)

    def _list_slides_offline(self, stain: str) -> list[WsiSlide]:
        """PACS 미가용 시 로컬 피처(.pt)에서 슬라이드 목록을 복원한다.

        파일명 규칙 {slide_id}_{case_code}.pt 에서 slide_id/case_code 를 분리한다.
        slide_id 는 analyze/load_bag 이 쓰는 키와 동일하게 유지되어 오프라인에서도
        목록→분석이 그대로 연결된다. cached(뷰어용 DICOM 캐시)는 있으면 True.
        """
        feat_dir = self._feature_dir_for(stain)
        if not feat_dir or not feat_dir.is_dir():
            _log.warning("오프라인 목록: %s 피처 디렉토리 없음", stain)
            return []

        slides: list[WsiSlide] = []
        seen: set[str] = set()
        for pt in sorted(feat_dir.glob("*.pt")):
            parts = pt.stem.split("_", 1)
            if len(parts) < 2:
                continue
            slide_id, case_code = parts[0], parts[1]
            if slide_id in seen:
                continue
            seen.add(slide_id)

            spec = _specimen_code(case_code) or slide_id
            self._case_specimen[slide_id] = spec
            slides.append(WsiSlide(
                slide_id=slide_id,
                case_code=case_code,
                stain=stain,
                has_features=True,
                cached=self._is_pacs_cached(slide_id),
                is_pacs=True,
                description=None,
            ))
        _log.info("오프라인 목록(%s): %d 슬라이드", stain, len(slides))
        return sorted(slides, key=lambda s: s.case_code)

    def _is_pacs_cached(self, case_id: str) -> bool:
        d = self._s.cache_dir / "pacs" / case_id
        return d.exists() and any(d.iterdir())

    def _resolve_specimen(self, slide_id: str) -> str:
        spec = self._case_specimen.get(slide_id)
        if spec:
            return spec
        try:
            from wsi.infra.engine_factory import get_pacs_repository
            cases = get_pacs_repository().list_wsi_cases()
            for c in cases:
                if c.get("case_id") == slide_id:
                    spec = _specimen_code(c.get("case_code", "")) or slide_id
                    self._case_specimen[slide_id] = spec
                    return spec
        except Exception:
            pass
        return slide_id

    def _merged_entries(self, slide_id: str) -> dict[str, Path]:
        """이 슬라이드 자신의 피처(HE/HE_CT 등) + 같은 환자의 다른 stain(대표 슬라이드 1개씩) 병합.
        자신의 피처가 항상 우선(다른 슬라이드로 덮어쓰지 않음) — 오늘 실제로 겪은 사고
        (다른 슬라이드 attention이 엉뚱한 슬라이드에 그려짐) 재발 방지."""
        spec = self._resolve_specimen(slide_id)
        entries: dict[str, Path] = dict(self._pt_by_slide.get(slide_id, {}))
        for stain, other_slide_id in self._specimen_stain_slide.get(spec, {}).items():
            if stain in entries:
                continue
            p = self._pt_by_slide.get(other_slide_id, {}).get(stain)
            if p:
                entries[stain] = p
        return entries

    # ── 임베딩 bag (이 슬라이드 자신 + 같은 환자의 보충 stain, CdssEngine 멀티스테인용) ───
    def load_bag(self, slide_id: str) -> dict[str, np.ndarray]:
        entries = self._merged_entries(slide_id)
        if not entries:
            raise SlideNotFoundError(f"피처 없음: slide={slide_id}")

        import torch
        bag: dict[str, np.ndarray] = {}
        for stain, pt_path in entries.items():
            tensor = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            arr = tensor.numpy().astype(np.float32)
            if arr.shape[0] > 0:
                bag[stain] = arr
        if not bag:
            raise SlideNotFoundError(f"피처 비어있음: slide={slide_id}")
        return bag

    # ── 공간 좌표 (오버레이) — 온디맨드 추출 시 저장한 _coords.npz 사이드카에서 읽는다 ─────
    def patch_coords(self, slide_id: str) -> dict[str, StainCoords]:
        entries = dict(self._merged_entries(slide_id))
        if not entries:
            return {}
        # CdssEngine 은 HE_CT(ctranspath)가 있으면 그걸 "HE" bag 으로 쓴다(analysis_service.py 와
        # 동일 치환) — 좌표도 실제로 추론에 쓰인 파일(HE_CT) 기준으로 맞춰야 stain_ids 와 어긋나지 않는다.
        if "HE_CT" in entries:
            entries["HE"] = entries.pop("HE_CT")
        entries.pop("HE_CT", None)

        result: dict[str, StainCoords] = {}
        for stain, pt_path in entries.items():
            coords_path = pt_path.parent / (pt_path.stem + "_coords.npz")
            if not coords_path.exists():
                continue
            try:
                with np.load(coords_path) as npz:
                    coords_arr = npz["coords"]
                    patch_l0 = float(npz["patch_l0"])
                    slide_w = int(npz["slide_w"])
                    slide_h = int(npz["slide_h"])
            except Exception:
                continue
            if slide_w <= 0 or len(coords_arr) == 0:
                continue
            # 이 stain 피처가 실제로 온 물리 슬라이드 id (pt 파일명 {slide_id}_{case_code}.pt).
            # mapper 가 displayed_slide 와 비교해 '다른 물리 슬라이드' 패치를 걸러 이미지 밖 오버레이를 막는다
            # (멀티스테인 CdssEngine bag: PAS 를 볼 때 딸려온 HE/MT 패치는 좌표공간이 달라 제외해야 함).
            src_slide = pt_path.stem.split("_", 1)[0]
            coords_list = [(float(x), float(y), patch_l0, src_slide) for x, y in coords_arr]
            result[stain] = StainCoords(coords=coords_list, slide_w=slide_w, slide_h=slide_h,
                                        displayed_slide=slide_id)
        return result

    def feature_path(self, slide_id: str, stain: str) -> Path | None:
        """slide_id 의 피처(.pt) 실제 경로 — 좌표 사이드카(_coords.npz) 위치 찾기용."""
        return self._pt_by_slide.get(slide_id, {}).get(stain)

    # ── SVS 경로 (PACS 슬라이드는 로컬 SVS 없음 → None) ─────────────────────
    def svs_path(self, slide_id: str, stain: str) -> Path | None:
        return None

    # ── 런타임 피처 등록(추출 파이프라인 완료 후 재시작 없이 즉시 반영) ──────
    def register_feature(self, slide_id: str, stain: str, path: Path, case_code: str | None = None) -> None:
        self._pt_by_slide.setdefault(slide_id, {})[stain] = path
        spec = (case_code and _specimen_code(case_code)) or self._case_specimen.get(slide_id) or slide_id
        self._case_specimen[slide_id] = spec
        if stain not in self._specimen_stain_slide.get(spec, {}):
            self._specimen_stain_slide.setdefault(spec, {})[stain] = slide_id
        _log.info("피처 런타임 등록: slide=%s stain=%s spec=%s path=%s", slide_id, stain, spec, path.name)

    def has_feature(self, slide_id: str, stain: str, case_code: str | None = None) -> bool:
        return stain in self._pt_by_slide.get(slide_id, {})

    def _slide_dims(self, slide_id: str, stain: str) -> tuple[int, int]:
        return 0, 0
