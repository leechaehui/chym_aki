"""CHYM WSI 추론 서버 — 포트 8001 (임상 백엔드 8010 과 분리된 별 프로세스).

분리 이유(장애 격리·의존성 격리): 무거운 torch/openslide 추론이 임상 API 를 죽이지 않게,
또 모델 재배포가 임상 백엔드 재배포를 강제하지 않게 한다.
도메인 예외를 여기서 HTTP 로 변환(경계 격리) — 도메인은 상태코드를 모른다.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from wsi.api import analysis, features, pacs, slides
from wsi.core.config import get_settings
from wsi.core.errors import (
    AuthError, EngineUnavailableError, ForbiddenError, PacsDisabledError,
    PacsError, SlideNotFoundError, TileUnavailableError, WsiError,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("wsi")

settings = get_settings()
app = FastAPI(title="CHYM WSI Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=list(settings.cors_origins),
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def observability(request: Request, call_next):
    """관측성 — request_id 부여 + 지연 측정(운영 추적)."""
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    log.info("rid=%s %s %s -> %s %.0fms",
             rid, request.method, request.url.path, response.status_code, dt)
    return response


# ── 도메인 예외 → HTTP (한 곳에서 매핑) ──────────────────────────────────────
@app.exception_handler(SlideNotFoundError)
async def _not_found(_: Request, e: SlideNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(TileUnavailableError)
async def _tile(_: Request, e: TileUnavailableError):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(AuthError)
async def _auth(_: Request, e: AuthError):
    return JSONResponse(status_code=401, content={"detail": str(e)})


@app.exception_handler(ForbiddenError)
async def _forbidden(_: Request, e: ForbiddenError):
    return JSONResponse(status_code=403, content={"detail": str(e)})


@app.exception_handler(EngineUnavailableError)
async def _engine(_: Request, e: EngineUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(e)})


@app.exception_handler(PacsDisabledError)
async def _pacs_off(_: Request, e: PacsDisabledError):
    return JSONResponse(status_code=503, content={"detail": str(e)})


@app.exception_handler(PacsError)
async def _pacs_err(_: Request, e: PacsError):
    return JSONResponse(status_code=502, content={"detail": str(e)})


@app.exception_handler(WsiError)
async def _generic(_: Request, e: WsiError):
    return JSONResponse(status_code=500, content={"detail": str(e)})


app.include_router(slides.router)
app.include_router(analysis.router)
app.include_router(features.router)
app.include_router(pacs.router)
app.include_router(pacs.view_router)  # 읽기 전용 케이스 썸네일(require_view) — 협진 임상의 조회용


@app.get("/health")
def health():
    return {"status": "ok", "service": "wsi", "port": settings.port}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wsi_main:app", host="0.0.0.0", port=settings.port, reload=False)
