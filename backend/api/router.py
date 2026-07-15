"""API 라우터 집계 — 모든 도메인 라우터를 /api 프리픽스로 묶는다."""
from fastapi import APIRouter

from api import (
    audit,
    auth,
    beds,
    chat,
    consultation,
    events,
    nephrology,
    notifications,
    pathology,
    patients,
    predict,
    timeline,
    voice,
    ws,
    incident,
    wsi,
    retrieval,
    demo,
)
from telemetry.admin_api import admin_router

api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(patients.router)
api_router.include_router(beds.router)
api_router.include_router(nephrology.router)
api_router.include_router(predict.router)
api_router.include_router(voice.router)
api_router.include_router(consultation.router)
api_router.include_router(pathology.router)
api_router.include_router(notifications.router)
api_router.include_router(timeline.router)
api_router.include_router(audit.router)
api_router.include_router(events.router)
api_router.include_router(events.alerts_router)
api_router.include_router(ws.router)
api_router.include_router(admin_router)
api_router.include_router(incident.router)
api_router.include_router(wsi.router)
api_router.include_router(retrieval.router)
api_router.include_router(demo.router)
