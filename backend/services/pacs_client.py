import httpx
from core.config import settings


def _headers() -> dict:
    return {
        "X-Service-Id": settings.pacs_service_id,
        "X-Service-Api-Key": settings.pacs_service_api_key,
    }


def _base() -> str:
    return f"{settings.pacs_base_url}/api/cases"


async def get_token(employee_id: str) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{settings.pacs_base_url}/api/auth/employee-token",
            headers=_headers(),
            json={"employee_id": employee_id},
        )
        r.raise_for_status()
        return r.json()["access_token"]


async def list_cases(token: str) -> list:
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{settings.pacs_base_url}/api/cases",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


async def get_manifest(token: str, case_id: str) -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{settings.pacs_base_url}/api/cases/{case_id}/manifest",
            headers={"Authorization": f"Bearer {token}", **_headers()},
        )
        if not r.is_success:
            print(f"[pacs] manifest {r.status_code} body: {r.text[:1000]}")
        r.raise_for_status()
        return r.json()


async def get_case(token: str, case_id: str) -> dict:
    """단일 케이스 상세 정보 (파일/인스턴스 목록 포함 여부 확인용)."""
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{settings.pacs_base_url}/api/cases/{case_id}",
            headers={"Authorization": f"Bearer {token}", **_headers()},
        )
        if not r.is_success:
            print(f"[pacs] get_case {r.status_code} body: {r.text[:500]}")
        r.raise_for_status()
        result = r.json()
        print(f"[pacs] get_case keys={list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
        return result


async def list_instances(token: str, case_id: str) -> list:
    """case의 instance 목록 반환. manifest가 실패할 때 fallback으로 사용."""
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{settings.pacs_base_url}/api/cases/{case_id}/instances",
            headers={"Authorization": f"Bearer {token}", **_headers()},
        )
        if not r.is_success:
            print(f"[pacs] instances {r.status_code} body: {r.text[:500]}")
        r.raise_for_status()
        return r.json()


async def download_instance(token: str, case_id: str, instance_id: str) -> bytes:
    async with httpx.AsyncClient() as c:
        r = await c.get(
            f"{settings.pacs_base_url}/api/cases/{case_id}/instances/{instance_id}/file",
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=True,
        )
        r.raise_for_status()
        return r.content
