"""도메인 예외.

책임: 비즈니스 규칙 위반을 표현하는 예외 타입 정의.
API 레이어에서 HTTP 상태코드로 매핑된다(exception handler).
"""


class DomainError(Exception):
    """도메인 예외 베이스. status_code 를 통해 HTTP 응답으로 변환된다.

    error_code: 표준화된 에러 코드(작업지시서 3.3 — '에러 코드 표준화').
    클라이언트는 메시지 문자열이 아니라 이 코드로 분기한다.
    """

    status_code: int = 400
    error_code: str = "DOMAIN_ERROR"

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        # 호출부에서 상태코드/에러코드를 직접 지정하면 클래스 기본값을 덮어쓴다.
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code


class NotFoundError(DomainError):
    """대상 리소스 없음."""

    status_code = 404
    error_code = "NOT_FOUND"


class ConflictError(DomainError):
    """상태 충돌(예: 이미 점유된 병상, 중복 입원, 중복 아이디)."""

    status_code = 409
    error_code = "CONFLICT"


class AuthError(DomainError):
    """인증 실패."""

    status_code = 401
    error_code = "UNAUTHENTICATED"


class PermissionError_(DomainError):
    """권한 부족(RBAC)."""

    status_code = 403
    error_code = "FORBIDDEN"


class ValidationFailedError(DomainError):
    """V&V 검증 실패(예: SOAP/CDSS 안전 규칙 위반). 결과 영속화 차단용."""

    status_code = 422
    error_code = "VALIDATION_FAILED"

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}
