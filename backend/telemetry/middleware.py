import time
import uuid
import jwt
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone
import json

from .context import request_id_ctx, session_id_ctx, user_id_ctx, get_request_id, get_session_id, get_user_id
from .streamer import streamer
from core.config import settings  # 앱과 동일한 JWT 검증키/알고리즘(RS256) 재사용

class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. 단 1회 Request ID 생성 및 ContextVar 설정
        req_id = str(uuid.uuid4())
        request_id_ctx.set(req_id)
        
        # 2. JWT 분석 및 session/user 설정
        session_id = None
        user_id = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, settings.jwt_verify_key, algorithms=[settings.jwt_algorithm])
                session_id = payload.get("sid")
                user_id = payload.get("sub") or payload.get("user_id")
            except jwt.ExpiredSignatureError:
                pass
            except jwt.InvalidTokenError:
                pass
                
        session_id_ctx.set(session_id)
        user_id_ctx.set(user_id)

        # 3. 측정 시작
        start_time = time.time()
        request_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        
        from .active_requests import add_active_request, remove_active_request
        add_active_request(req_id, {
            "request_id": req_id,
            "session_id": session_id,
            "user_id": user_id,
            "endpoint": request.url.path,
            "method": request.method,
            "ip_address": request.client.host if request.client else None,
            "start_time": request_dt
        })
        
        # Request Body 추출 (Latency에 영향 최소화를 위해 크기 확인 후 선택적 읽기)
        body = {}
        # 주의: Request Body를 미리 읽으면 라우터에서 재사용 문제가 생길 수 있어,
        # 여기서는 생략하거나 필요 시 request.stream()을 wrap해서 읽어야 함.
        # 본 구현에서는 예시로 query param이나 간단한 정보만 payload로 로깅.

        exception_occurred = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            exception_occurred = str(e)
            raise e
        finally:
            remove_active_request(req_id)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            response_dt = datetime.now(timezone.utc).replace(tzinfo=None)
            
            # Non-blocking Streamer Push
            log_entry = {
                "request_id": get_request_id(),
                "session_id": get_session_id(),
                "user_id": get_user_id(),
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "request_time": request_dt,
                "response_time": response_dt,
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "payload": {"query": str(request.query_params)},
                "exception": exception_occurred
            }
            streamer.push(log_entry)
