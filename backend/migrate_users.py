from core.database import engine
from sqlalchemy import text

cols = [
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS signature_path VARCHAR(255)",
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS rejection_reason TEXT",
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS approved_by VARCHAR(40)",
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP WITH TIME ZONE",
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS rejected_by VARCHAR(40)",
    "ALTER TABLE chym.users ADD COLUMN IF NOT EXISTS rejected_at TIMESTAMP WITH TIME ZONE",
]

with engine.connect() as conn:
    for sql in cols:
        conn.execute(text(sql))
        print(f"완료: {sql.split('ADD COLUMN IF NOT EXISTS ')[1].split(' ')[0]}")
    conn.commit()

print("마이그레이션 완료!")
