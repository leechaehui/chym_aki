"""PACS Gateway 클라이언트 — "공용 PACS Gateway API"(Orthanc 백엔드)와 서버↔서버 통신.

BFF 원칙: 브라우저는 PACS 를 직접 호출하지 않는다. 자격증명(env)은 이 서버에만 있고,
여기서 employee-token 을 받아(캐시) PACS 를 대리 호출한다. 토큰/키는 절대 프론트로 안 나간다.

엔드포인트(OpenAPI 기준):
  POST /api/auth/employee-token  (X-Service-Id, X-Service-Api-Key + {employee_id}) -> {access_token,...}
  GET  /api/cases                                          -> [CaseOut]
  GET  /api/cases/{id}/manifest                            -> {files:[{instance_id,series_id,download_url}]}
  GET  /api/cases/{id}/instances/{instance_id}/file        -> DICOM bytes
"""
from __future__ import annotations

import base64
import json
import time

import requests

from wsi.core.config import WsiSettings
from wsi.core.errors import PacsDisabledError, PacsError

_TIMEOUT = 15            # 일반 요청 타임아웃(초)
_DL_TIMEOUT = 120        # 인스턴스(대용량 DICOM) 다운로드 타임아웃
_TOKEN_SKEW = 60         # 만료 여유(초)


class PacsGateway:
    def __init__(self, settings: WsiSettings):
        if not settings.pacs_enabled:
            raise PacsDisabledError("PACS 자격증명 미설정(env PACS_BASE_URL/SERVICE_ID/API_KEY/EMPLOYEE_ID)")
        self._s = settings
        self._base = settings.pacs_base_url.rstrip("/")
        self._token: str | None = None
        self._token_exp: float = 0.0

    # ── 인증(employee-token, 캐시) ────────────────────────────────────────
    def _employee_token(self) -> str:
        if self._token and time.time() < self._token_exp - _TOKEN_SKEW:
            return self._token
        try:
            res = requests.post(
                f"{self._base}/api/auth/employee-token",
                headers={"X-Service-Id": self._s.pacs_service_id,
                         "X-Service-Api-Key": self._s.pacs_service_api_key},
                json={"employee_id": self._s.pacs_employee_id},
                timeout=_TIMEOUT,
            )
        except requests.RequestException as e:
            raise PacsError(f"PACS 인증 요청 실패: {e}") from e
        if res.status_code != 200:
            raise PacsError(f"PACS 인증 실패 {res.status_code}: {res.text[:200]}")
        tok = res.json().get("access_token")
        if not tok:
            raise PacsError("PACS 응답에 access_token 없음")
        self._token = tok
        self._token_exp = _jwt_exp(tok) or (time.time() + 50 * 60)
        return tok

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._employee_token()}"}

    def _get(self, path: str, *, timeout: int = _TIMEOUT) -> requests.Response:
        try:
            res = requests.get(f"{self._base}{path}", headers=self._headers(), timeout=timeout)
        except requests.RequestException as e:
            raise PacsError(f"PACS 요청 실패 {path}: {e}") from e
        if res.status_code == 401:                       # 토큰 만료 추정 → 1회 재발급 후 재시도
            self._token = None
            try:
                res = requests.get(f"{self._base}{path}", headers=self._headers(), timeout=timeout)
            except requests.RequestException as e:
                raise PacsError(f"PACS 재시도 실패 {path}: {e}") from e
        if res.status_code != 200:
            raise PacsError(f"PACS {path} -> {res.status_code}: {res.text[:200]}")
        return res

    # ── 케이스/매니페스트/인스턴스 ───────────────────────────────────────
    def list_cases(self, modality: str | None = None) -> list[dict]:
        cases = self._get("/api/cases").json()
        if modality:
            cases = [c for c in cases if c.get("modality") == modality]
        return cases

    def get_case(self, case_id: str) -> dict:
        return self._get(f"/api/cases/{case_id}").json()

    def get_manifest(self, case_id: str) -> dict:
        """{case_id, case_code, study_uid, file_count, files:[{instance_id, series_id, download_url}]}"""
        return self._get(f"/api/cases/{case_id}/manifest").json()

    def download_instance(self, case_id: str, instance_id: str) -> bytes:
        return self._get(f"/api/cases/{case_id}/instances/{instance_id}/file",
                         timeout=_DL_TIMEOUT).content

    def list_projects(self) -> list[dict]:
        return self._get("/api/projects").json()

    def get_dicom_info(self, case_id: str) -> dict:
        return self._get(f"/api/cases/{case_id}/dicom-info").json()

    def download_case(self, case_id: str) -> bytes:
        """케이스 전체 ZIP 다운로드(대용량 가능)."""
        return self._get(f"/api/cases/{case_id}/download", timeout=_DL_TIMEOUT).content

    def upload_case(self, *, case_code: str, project_code: str | None, modality: str | None,
                    description: str | None, files: list[tuple[str, bytes, str | None]]) -> dict:
        """케이스 업로드(multipart) — files: [(filename, bytes, content_type)]."""
        data = {"case_code": case_code}
        if project_code:
            data["project_code"] = project_code
        if modality:
            data["modality"] = modality
        if description:
            data["description"] = description
        multipart = [("files", (fn, content, ct or "application/dicom")) for fn, content, ct in files]
        try:
            res = requests.post(f"{self._base}/api/cases/upload", headers=self._headers(),
                                data=data, files=multipart, timeout=_DL_TIMEOUT)
        except requests.RequestException as e:
            raise PacsError(f"PACS 업로드 실패: {e}") from e
        if res.status_code == 401:
            self._token = None
            res = requests.post(f"{self._base}/api/cases/upload", headers=self._headers(),
                                data=data, files=multipart, timeout=_DL_TIMEOUT)
        if res.status_code not in (200, 201):
            raise PacsError(f"PACS 업로드 {res.status_code}: {res.text[:200]}")
        return res.json()


def _jwt_exp(token: str) -> float | None:
    """JWT exp 클레임(초). 디코드 실패 시 None(기본 만료 사용)."""
    try:
        payload_b64 = token.split(".")[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
        return float(payload["exp"]) if "exp" in payload else None
    except Exception:
        return None
