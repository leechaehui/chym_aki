"""WSI 도메인 예외 — HTTP 세부사항과 분리(도메인은 상태코드를 모른다).

api 계층이 이 예외를 잡아 적절한 HTTP 응답으로 변환한다(경계 격리).
"""
from __future__ import annotations


class WsiError(Exception):
    """WSI 도메인 기본 예외."""


class SlideNotFoundError(WsiError):
    """요청한 slide_id/stain 의 임베딩 bag 이 없음 → 404."""


class TileUnavailableError(WsiError):
    """원본 SVS 부재/손상으로 타일(DZI/썸네일) 생성 불가 → 404(프론트는 graceful degrade)."""


class EngineUnavailableError(WsiError):
    """추론 엔진(가중치/torch) 로드 실패 → 503. 임상 API(8010)와 격리되어야 함."""


class AuthError(WsiError):
    """토큰 누락/무효/만료 → 401."""


class ForbiddenError(WsiError):
    """권한 부족(병리과/admin 아님) → 403."""


class PacsError(WsiError):
    """PACS 게이트웨이 연동 실패(인증/네트워크/응답) → 502."""


class PacsDisabledError(WsiError):
    """PACS 자격증명 미설정 → 503(기능 비활성)."""
