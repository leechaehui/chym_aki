import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from core.database import SessionLocal
from models.user import User
from models.approval_history import ApprovalHistory
from models.base import utcnow

def backfill():
    db = SessionLocal()
    try:
        users = db.scalars(select(User)).all()
        for user in users:
            # Check if history exists
            exists = db.scalar(select(ApprovalHistory).where(ApprovalHistory.user_id == user.id).limit(1))
            if not exists:
                history = ApprovalHistory(
                    user_id=user.id,
                    admin_id=None,
                    actor_type="SYSTEM",
                    old_status=None,
                    new_status=user.approval,
                    reason=user.rejection_reason,
                    created_at=user.created_at or utcnow()
                )
                db.add(history)
        db.commit()
        print("Backfill completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"Error during backfill: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    backfill()
