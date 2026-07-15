import uuid
from datetime import datetime, timedelta, timezone
import jwt
from typing import Optional

from core.config import settings  # 앱과 동일한 JWT 키/알고리즘(RS256) 재사용 — 하드코딩 시크릿 제거

async def create_user_session(db_pool, user_id: str, ip_address: str, user_agent: str) -> dict:
    """
    Creates a user session in DB and returns the JWT token.
    """
    session_id = str(uuid.uuid4())
    jti = str(uuid.uuid4())
    
    # DB Insert
    async with db_pool.acquire() as conn:
        query = """
            INSERT INTO user_sessions (id, user_id, ip_address, user_agent)
            VALUES ($1, $2, $3, $4)
        """
        await conn.execute(query, session_id, user_id, ip_address, user_agent)
        
    # Create JWT
    payload = {
        "sub": user_id,
        "sid": session_id,
        "jti": jti,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24)
    }
    
    token = jwt.encode(payload, settings.jwt_signing_key, algorithm=settings.jwt_algorithm)
    return {"access_token": token, "session_id": session_id}
