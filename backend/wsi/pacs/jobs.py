"""변환 업로드 작업(job) 스토어 — 인메모리, 진행상태 폴링용.

SVS/TIFF → DICOM 변환은 무겁고 길어서 HTTP 요청 동기로 처리하면 타임아웃된다.
→ 업로드 요청은 job 을 만들고 즉시 job_id 반환, 백그라운드 스레드가 변환·업로드하며 상태를 갱신.
프론트는 /pacs/jobs/{id} 를 폴링한다. (단일 프로세스 가정; 재시작 시 job 소실 — 운영 확장 시 DB/큐로 교체.)
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Job:
    id: str
    status: str = "queued"        # queued | converting | uploading | done | error | cancelled
    stage: str = "대기 중"
    progress: int = 0             # 0~100
    case_code: str = ""
    result: dict | None = None
    error: str | None = None
    cancelled: bool = False       # 취소 요청 플래그(워커가 PACS 전송 직전 확인)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def as_dict(self) -> dict:
        return {
            "job_id": self.id, "status": self.status, "stage": self.stage,
            "progress": self.progress, "case_code": self.case_code,
            "result": self.result, "error": self.error, "created_at": self.created_at,
        }


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, case_code: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], case_code=case_code)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def cancel(self, job_id: str) -> Job | None:
        """취소 요청 — 아직 종료(done/error/cancelled)되지 않은 작업만 플래그를 세운다.
        실제 중단은 워커 스레드가 PACS 전송 직전 cancelled 를 확인해 수행(협력적 취소)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status not in ("done", "error", "cancelled"):
                job.cancelled = True
                job.stage = "취소 요청됨"
            return job
