from core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='chym' AND table_name='users' AND column_name='signature_path'"
    ))
    print('있음' if result.fetchone() else '없음')
