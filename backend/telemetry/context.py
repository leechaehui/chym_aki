import contextvars
from typing import Optional

request_id_ctx = contextvars.ContextVar[Optional[str]]("request_id_ctx", default=None)
session_id_ctx = contextvars.ContextVar[Optional[str]]("session_id_ctx", default=None)
user_id_ctx = contextvars.ContextVar[Optional[str]]("user_id_ctx", default=None)

def get_request_id() -> Optional[str]:
    return request_id_ctx.get()

def get_session_id() -> Optional[str]:
    return session_id_ctx.get()

def get_user_id() -> Optional[str]:
    return user_id_ctx.get()
