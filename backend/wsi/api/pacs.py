"""PACS Controller — 병리과 전용 PACS 브라우저/뷰어 엔드포인트(BFF).

브라우저는 여기(8001)만 호출하고, 서버가 PACS 게이트웨이로 대리 요청한다. 자격증명은 노출 0.
모든 엔드포인트 require_pathology(병리과/admin) 게이트.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from wsi.core.auth import require_pathology, require_view
from wsi.core.config import get_settings
from wsi.core.errors import TileUnavailableError
from wsi.infra.engine_factory import (
    get_job_store, get_pacs_gateway, get_pacs_repository, get_pacs_tile_source,
)
from wsi.pacs.converter import convert_to_dicom, is_convertible

router = APIRouter(prefix="/pacs", tags=["pacs"], dependencies=[Depends(require_pathology)])

# 읽기 전용 케이스 썸네일은 협진 리포트를 받은 임상의(신장내과/응급)도 조회 가능해야 한다
# (AI 분석 근거의 Heatmap 배경). 병리 전용 게이트 대신 require_view 로 분리한다. 나머지
# PACS 엔드포인트(다운로드·변환 등)는 그대로 require_pathology 유지.
view_router = APIRouter(prefix="/pacs", tags=["pacs"], dependencies=[Depends(require_view)])


@router.get("/status")
def status():
    """프론트가 PACS 탭 노출 여부 판단용 — 자격증명 설정 여부만 반환(비밀 노출 X)."""
    return {"enabled": get_settings().pacs_enabled}


@router.get("/cases")
def list_cases():
    return get_pacs_repository().list_wsi_cases()


@router.get("/projects")
def list_projects():
    return get_pacs_gateway().list_projects()


@router.post("/upload")
async def upload_case(
    case_code: str = Form(...),
    project_code: str | None = Form(None),
    modality: str | None = Form(None),
    description: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    """DICOM/WSI 업로드 — 브라우저 → (우리 백엔드) → 실 PACS. 자격증명 노출 없음."""
    payload = [(f.filename or "file.dcm", await f.read(), f.content_type) for f in files]
    return get_pacs_gateway().upload_case(
        case_code=case_code, project_code=project_code,
        modality=modality, description=description, files=payload)


def _run_convert_job(job_id: str, src: Path, case_code: str,
                     project_code: str | None, description: str | None) -> None:
    """백그라운드: SVS/TIFF → DICOM 변환 → 실 PACS 업로드. 상태를 job 스토어에 갱신."""
    store = get_job_store()
    work = src.parent

    def _is_cancelled() -> bool:
        j = store.get(job_id)
        return bool(j and j.cancelled)

    try:
        if _is_cancelled():                                 # 시작 직전 취소
            store.update(job_id, status="cancelled", stage="취소됨", progress=0)
            return
        store.update(job_id, status="converting", stage="DICOM 변환 중", progress=20)
        dcm = convert_to_dicom(src, work / "dicom")
        if _is_cancelled():                                 # PACS 전송 직전 취소(전송 안 함 → PACS 무변경)
            store.update(job_id, status="cancelled", stage="취소됨", progress=0)
            return
        store.update(job_id, status="uploading", stage="PACS 업로드 중", progress=70)
        files = [(p.name, p.read_bytes(), "application/dicom") for p in dcm]
        res = get_pacs_gateway().upload_case(
            case_code=case_code, project_code=project_code,
            modality="SM", description=description, files=files)
        store.update(job_id, status="done", stage="완료", progress=100, result=res)
    except Exception as e:                                   # 변환/업로드 실패 → 작업만 실패(서버 무사)
        if _is_cancelled():                                 # 취소 중 발생한 예외는 실패로 덮지 않음
            store.update(job_id, status="cancelled", stage="취소됨", progress=0)
        else:
            store.update(job_id, status="error", stage="실패", error=str(e))
    finally:
        shutil.rmtree(work, ignore_errors=True)


@router.post("/convert-upload")
async def convert_upload(
    case_code: str = Form(...),
    project_code: str | None = Form(None),
    description: str | None = Form(None),
    file: UploadFile = File(...),
):
    """SVS/TIFF 등 → DICOM 변환 업로드(비동기). job_id 반환 후 /pacs/jobs/{id} 폴링."""
    if not is_convertible(file.filename or ""):
        raise HTTPException(400, "변환 지원 형식이 아닙니다(.svs/.tiff/.ndpi 등). DICOM 은 /pacs/upload 사용")
    job = get_job_store().create(case_code)
    work = get_settings().cache_dir / "pacs_convert" / job.id
    work.mkdir(parents=True, exist_ok=True)
    src = work / (file.filename or "input.svs")
    with src.open("wb") as f:                                # 청크 저장(대용량 메모리 회피)
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    threading.Thread(target=_run_convert_job,
                     args=(job.id, src, case_code, project_code, description),
                     daemon=True).start()
    return job.as_dict()


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return job.as_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """변환 업로드 작업 취소 — PACS 전송 전이면 실제로 중단된다(전송 후/완료는 멱등 no-op).
    업스트림 PACS 는 케이스 삭제 API 가 없어, 전송 완료된 건은 되돌릴 수 없다."""
    job = get_job_store().cancel(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return job.as_dict()


@router.get("/cases/{case_id}")
def case_detail(case_id: str):
    return get_pacs_gateway().get_case(case_id)


@router.get("/cases/{case_id}/dicom-info")
def dicom_info(case_id: str):
    return get_pacs_gateway().get_dicom_info(case_id)


@router.get("/cases/{case_id}/download")
def download_case(case_id: str):
    data = get_pacs_gateway().download_case(case_id)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="case_{case_id}.zip"'})


@router.get("/cases/{case_id}/manifest")
def manifest(case_id: str):
    return get_pacs_repository().manifest(case_id)


@view_router.get("/cases/{case_id}/thumbnail")
def thumbnail(case_id: str, size: int = 400):
    return Response(get_pacs_tile_source().thumbnail(case_id, size), media_type="image/jpeg")


@router.get("/cases/{case_id}/dzi")
def dzi_descriptor(case_id: str):
    return Response(get_pacs_tile_source().dzi_descriptor(case_id), media_type="application/xml")


@router.get("/cases/{case_id}/dzi_files/{level}/{tile}")
def dzi_tile(case_id: str, level: int, tile: str):
    try:
        col, row = (int(x) for x in tile.split(".")[0].split("_"))
    except ValueError:
        raise HTTPException(400, "타일 좌표 형식 오류")
    try:
        data = get_pacs_tile_source().dzi_tile(case_id, level, col, row)
    except TileUnavailableError:
        raise HTTPException(404, "타일 없음")
    return Response(data, media_type="image/jpeg")
