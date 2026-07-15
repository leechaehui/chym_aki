import os
import sys

# 프로젝트 루트(backend)를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database import engine, SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    with engine.begin() as conn:
        logger.info("Adding columns to users table...")
        # 기존 users 테이블에 필드 추가 (이미 존재하면 무시되거나 예외 발생)
        try:
            conn.execute(text("""
                ALTER TABLE chym.users 
                ADD COLUMN rejection_reason TEXT,
                ADD COLUMN approved_by VARCHAR(36),
                ADD COLUMN approved_at TIMESTAMP,
                ADD COLUMN rejected_by VARCHAR(36),
                ADD COLUMN rejected_at TIMESTAMP;
            """))
            logger.info("Columns added successfully.")
        except Exception as e:
            logger.warning(f"Columns might already exist or error occurred: {e}")

    # 새 트랜잭션에서 테이블 생성
    with engine.begin() as conn:
        logger.info("Creating approval_history table...")
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chym.approval_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    admin_id VARCHAR(36),
                    actor_type VARCHAR(20) NOT NULL,
                    old_status VARCHAR(20),
                    new_status VARCHAR(20) NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

                    CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES chym.users(id) ON DELETE RESTRICT,
                    CONSTRAINT fk_admin_id FOREIGN KEY (admin_id) REFERENCES chym.users(id) ON DELETE SET NULL,
                    CONSTRAINT chk_actor_type CHECK (actor_type IN ('ADMIN', 'SYSTEM', 'WORKER')),
                    CONSTRAINT chk_new_status CHECK (new_status IN ('pending', 'approved', 'rejected'))
                );
            """))
            logger.info("Table approval_history created successfully.")
        except Exception as e:
            logger.warning(f"Error creating approval_history table: {e}")
            
    with engine.begin() as conn:
        logger.info("Creating indexes...")
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_approval_history_user_id ON chym.approval_history(user_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_approval_history_created_at ON chym.approval_history(created_at);"))
            logger.info("Indexes created successfully.")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

if __name__ == "__main__":
    migrate()
    logger.info("Migration finished.")
