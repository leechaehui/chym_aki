from core.database import engine
from sqlalchemy import text

with engine.begin() as conn:
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN signature_path VARCHAR(255);"))
        print("users table altered.")
    except Exception as e:
        print("users alter failed:", e)

    try:
        conn.execute(text("ALTER TABLE incidents ADD COLUMN assigned_to_user_id VARCHAR(40);"))
        conn.execute(text("ALTER TABLE incidents ADD COLUMN assigned_to_name VARCHAR(60);"))
        print("incidents table altered.")
    except Exception as e:
        print("incidents alter failed:", e)
