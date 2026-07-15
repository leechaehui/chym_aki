"""
새로운 DB 테이블 생성 및 다운로드된 WSI 파일 매핑 (Phase 2 실행)
"""
import os
import sys

# 백엔드 루트를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import SessionLocal, init_db
from services.phase2_wsi_acquisition import Phase2WsiAcquisitionService

def main():
    print("="*60)
    print("Phase 2: DB 초기화 및 WSI 매핑 시작")
    print("="*60)
    
    # DB 테이블 생성 (models/__init__.py에 등록된 Phase10, Phase2 등 생성됨)
    print(" - DB 테이블 스키마 초기화 중...")
    init_db()
    
    # 세션 획득 및 매핑 실행
    db = SessionLocal()
    try:
        service = Phase2WsiAcquisitionService(db)
        # 다운로더가 저장한 디렉토리 위치 (frontend, backend와 동일선상의 data/raw)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
        
        mapped_count = service.scan_and_map_directories(base_dir=data_dir)
        print(f" - 스캔 완료! 새로 매핑된 슬라이드 수: {mapped_count}개")
        
    except Exception as e:
        print(f"매핑 중 오류 발생: {e}")
    finally:
        db.close()
        
    print("="*60)
    print("완료되었습니다.")
    print("="*60)

if __name__ == "__main__":
    main()
