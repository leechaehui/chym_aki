"""Incident API.

운영 장애 조회, 조치, 배정, 전자서명, 잠금 처리 엔드포인트.
"""
from fastapi import APIRouter, Depends, Request

from core.database import get_db
from core.deps import get_current_user
from models.user import User
from schemas.incident import (
    IncidentOut,
    IncidentStatusUpdate,
    IncidentResolveUpdate,
    IncidentSignUpdate,
    IncidentAssignUpdate,
    IncidentAnalysisOut,
)
from services.audit_service import AuditService
from services.incident_service import IncidentService

router = APIRouter(prefix="/incidents", tags=["incident"])


def get_incident_service(db=Depends(get_db)) -> IncidentService:
    return IncidentService(db, AuditService(db))


def get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("", response_model=list[IncidentOut])
def list_incidents(
    status: str | None = None,
    sort_by: str = "latest",
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """Incident 목록 조회."""
    return svc.list_incidents(status=status, sort_by=sort_by)


@router.get("/{incident_id}", response_model=IncidentOut)
def get_incident(
    incident_id: str,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """Incident 단건 상세 조회."""
    return svc.get_incident(incident_id)


@router.patch("/{incident_id}/status", response_model=IncidentOut)
def update_incident_status(
    incident_id: str,
    payload: IncidentStatusUpdate,
    request: Request,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """상태 변경."""
    return svc.update_status(
        incident_id, 
        payload.status, 
        current_user.id, 
        current_user.name,
        get_client_ip(request),
        request.headers.get("user-agent", "unknown")
    )


@router.patch("/{incident_id}/assign", response_model=IncidentOut)
def assign_incident(
    incident_id: str,
    payload: IncidentAssignUpdate,
    request: Request,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """담당자 지정 및 변경."""
    return svc.assign_incident(
        incident_id,
        payload.assigned_to_user_id,
        payload.assigned_to_name,
        current_user.id,
        current_user.name,
        get_client_ip(request),
        request.headers.get("user-agent", "unknown")
    )


@router.patch("/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(
    incident_id: str,
    payload: IncidentResolveUpdate,
    request: Request,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """조치 내역(원인, 조치) 입력."""
    return svc.resolve_incident(
        incident_id,
        payload.root_cause,
        payload.action_taken,
        current_user.id,
        current_user.name,
        get_client_ip(request),
        request.headers.get("user-agent", "unknown")
    )


@router.post("/{incident_id}/sign", response_model=IncidentOut)
def sign_incident(
    incident_id: str,
    payload: IncidentSignUpdate,
    request: Request,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """프로필에 저장된 전자서명을 사용하여 SIGNED 상태 전환."""
    return svc.sign_incident(
        incident_id,
        current_user.id,
        current_user.name,
        current_user.signature_path,
        get_client_ip(request),
        request.headers.get("user-agent", "unknown")
    )


@router.post("/{incident_id}/lock", response_model=IncidentOut)
def lock_incident(
    incident_id: str,
    request: Request,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """최종 승인 및 LOCKED 상태 전환."""
    return svc.update_status(
        incident_id,
        "LOCKED",
        current_user.id,
        current_user.name,
        get_client_ip(request),
        request.headers.get("user-agent", "unknown")
    )


@router.post("/{incident_id}/analyze", response_model=IncidentAnalysisOut)
def analyze_incident(
    incident_id: str,
    svc: IncidentService = Depends(get_incident_service),
    current_user: User = Depends(get_current_user),
):
    """장애 원인 및 조치 내역 자동(휴리스틱 AI) 생성."""
    incident = svc.get_incident(incident_id)
    err = (incident.error_message or "").lower()
    
    cause = "시스템 내부 오류가 발생했습니다."
    action = "서버 로그를 확인하고 관련 모듈을 점검합니다."
    
    if "timeout" in err:
        cause = f"[{incident.module_name}] 데이터베이스 또는 외부 API 응답 지연(Timeout)으로 인한 병목 발생."
        action = "1. 커넥션 풀 및 타임아웃 설정값 조정\n2. 트래픽 폭주에 대비한 스케일 아웃 검토\n3. 느린 쿼리(Slow Query) 튜닝 적용"
    elif "connection refused" in err or "network" in err:
        cause = f"[{incident.module_name}] 백엔드 인프라 노드 간 네트워크 단절 또는 포트 닫힘 현상."
        action = "1. 방화벽 및 인바운드/아웃바운드 룰 검토\n2. 네트워크 장비 재시작 및 헬스체크\n3. 페일오버(Fallback) 로직 추가"
    elif "memory" in err or "oom" in err:
        cause = f"[{incident.module_name}] 일시적인 트래픽 폭증으로 인한 가용 메모리 고갈 (OOM)."
        action = "1. Memory Guard 임계치(Threshold) 튜닝\n2. 인스턴스 메모리 증설(Scale-up)\n3. 대용량 데이터 로드 로직 최적화 및 배치(Batch) 처리 적용"
    elif "syntax" in err or "undefined" in err or "not found" in err:
        cause = f"[{incident.module_name}] 존재하지 않는 리소스 참조 또는 잘못된 구문(Syntax) 실행으로 인한 애플리케이션 크래시."
        action = "1. 배포된 코드 버전 확인 및 핫픽스(Hotfix) 롤백\n2. 프론트엔드-백엔드 간 API 스펙 정합성 검증\n3. 누락된 DB 테이블/컬럼 마이그레이션 적용"
    else:
        cause = f"[{incident.module_name}] 컴포넌트 내 예외 발생: {incident.error_message}"
        action = "1. Sentry/Datadog 상세 로그 추적\n2. 재현 테스트 스크립트 작성\n3. 예외 처리(Try-Catch) 로직 보강 및 방어 코드 추가"

    svc.update_draft(incident_id, cause, action)
    return IncidentAnalysisOut(root_cause=cause, action_taken=action)
