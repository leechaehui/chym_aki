"""RENAI 백엔드 진입점.

책임: FastAPI 앱 구성(미들웨어/예외핸들러/라우터) + 시작 시 DB 초기화/시드.
얇게 유지 — 모든 도메인 로직은 service 레이어에 있다.
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.router import api_router
from core.config import settings
from core.database import init_db
from core.exceptions import DomainError
from core.logging import (
    configure_logging,
    get_logger,
    get_request_id,
    set_request_id,
)
from telemetry.middleware import TelemetryMiddleware
from telemetry.streamer import streamer

configure_logging()
log = get_logger("chym.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """시작 시: 테이블 생성 + (옵션) 시드 + 이벤트 파이프라인 구독. 종료 시: 정리 없음."""
    import asyncio

    from services.pipeline import init_pipeline

    init_db()
    if settings.seed_on_startup:
        from db.seed import seed

        seed()
    # WSI ABMIL 모델 로드
    from api.wsi import load_wsi_models
    load_wsi_models()
    # 이벤트 소비자 등록 + WS 브리지용 running 정루프 주입
    await streamer.start()
    init_pipeline(asyncio.get_running_loop())
    yield
    await streamer.stop()


app = FastAPI(
    title=f"{settings.app_name} Backend",
    description="병원 EMR 운영 + AI 음성 진료 + AKI 분석 + 병상 트랜잭션 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 프론트엔드 개발 서버 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex="https?://.*",
    # allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TelemetryMiddleware)


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """요청 단위 추적: request_id 발급/전파 + 지연시간 로깅(3.3).

    - 클라이언트가 X-Request-ID 를 주면 그대로 사용(분산 추적 연결), 없으면 발급.
    - 처리 시간(ms)을 측정해 로그/응답 헤더에 남긴다(8. 성능 p95 추적 근거).
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    set_request_id(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.exception(
            "request failed %s %s (%.1fms)",
            request.method, request.url.path, elapsed_ms,
        )
        
        # Incident 캡처 로직 연동
        # 주의: 여기서는 Session 을 새로 열거나 주입받아야 함
        from core.database import SessionLocal
        from services.incident_service import IncidentService
        from services.audit_service import AuditService
        
        db = SessionLocal()
        try:
            svc = IncidentService(db, AuditService(db))
            svc.report_exception(exc, module_name="API", endpoint=request.url.path)
        except Exception as inner_exc:
            log.error(f"Failed to report incident: {inner_exc}")
        finally:
            db.close()
            
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    log.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError):
    """도메인 예외 → 일관된 HTTP 응답 매핑(표준 error_code 포함)."""
    log.warning(
        "domain error %s %s: [%s] %s",
        request.method, request.url.path, exc.error_code, exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "request_id": get_request_id(),
        },
    )


@app.get("/health", tags=["system"])
def health():
    """헬스체크 — docker/로드밸런서용."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


import os
from fastapi.staticfiles import StaticFiles

# 서명 이미지 제공용 디렉토리 마운트
os.makedirs(os.path.join(os.getcwd(), "uploads", "signatures"), exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(api_router)


# [경고] 어떤 일이 있더라도 서버는 8010으로 유지한다. (실행 전 이 주석을 반드시 확인할 것)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=settings.debug)
