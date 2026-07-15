"""
Phase 2 (사전 준비): KPMP WSI 자동 다운로더
kpmp.csv 파일을 읽어들여 필요한 염색(H&E, PAS, SIL, TRI)의 SVS 파일을
각 디렉토리에 맞게 다운로드합니다.
"""

import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path

# 설정
CSV_PATH = (Path(__file__).resolve().parents[2] / "kpmp.csv")
BASE_DIR = (Path(__file__).resolve().parents[2] / "data/raw")
# KPMP Atlas 실제 다운로드 엔드포인트 (구 /packages/{id}/files/{name} 경로는 404)
API_ENDPOINT = "https://atlas.kpmp.org/api/v1/file/download/{package_id}/{file_name}"

STAIN_MAP = {
    "H&E stain": "wsi_he",
    "Frozen H&E stain": "wsi_he",
    "PAS stain": "wsi_pas",
    "SIL stain": "wsi_silver",
    "TRI stain": "wsi_mt"
}

def download_file(url: str, dest_path: Path) -> bool:
    """지정된 URL에서 파일을 다운로드하며 진행률을 표시합니다."""
    if dest_path.exists():
        print(f"이미 존재함: {dest_path.name}")
        return True
        
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"다운로드 실패: {dest_path.name} - {e}")
        # 실패한 빈 파일은 삭제
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    print("="*60)
    print("KPMP WSI 이미지 다운로더 시작")
    print("="*60)
    
    if not CSV_PATH.exists():
        print(f"오류: {CSV_PATH} 파일을 찾을 수 없습니다.")
        sys.exit(1)
        
    # CSV 로드
    df = pd.read_csv(CSV_PATH)
    
    target_columns = ["Internal Package ID", "File Name", "Workflow Type", "Participant ID", "Access"]
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        print(f"CSV에 다음 컬럼이 없습니다: {missing_cols}")
        sys.exit(1)
        
    # Open Access 파일이면서 SVS 파일인 것만 필터링
    open_access_df = df[(df["Access"] == "open") & (df["File Name"].str.endswith(".svs"))].copy()
    
    # 지원하는 염색만 필터링
    supported_df = open_access_df[open_access_df["Workflow Type"].isin(STAIN_MAP.keys())]
    print(f"총 {len(df)}개 파일 중 대상 파일 수: {len(supported_df)}개")
    
    # 디렉토리 생성
    for stain_dir in set(STAIN_MAP.values()):
        os.makedirs(BASE_DIR / stain_dir, exist_ok=True)
        
    # 너무 많은 파일이 다운로드되는 것을 방지하기 위해 염색별로 최대 2개씩만 테스트 다운로드
    # (실제 전체 다운로드가 필요할 경우 아래 MAX_PER_STAIN 관련 로직을 제거하면 됩니다)
    MAX_PER_STAIN = 2
    downloaded_count = {stain: 0 for stain in set(STAIN_MAP.values())}
    
    for index, row in supported_df.iterrows():
        workflow_type = row["Workflow Type"]
        stain_dir = STAIN_MAP.get(workflow_type)
        
        if downloaded_count[stain_dir] >= MAX_PER_STAIN:
            continue
            
        package_id = row["Internal Package ID"]
        file_name = row["File Name"]
        participant_id = row["Participant ID"]
        
        # 다운로드 URL 구성
        url = API_ENDPOINT.format(package_id=package_id, file_name=file_name)
        
        # 매핑하기 쉽도록 파일명을 ParticipantID_Stain_ID.svs 형태로 저장
        stain_suffix = stain_dir.replace("wsi_", "").upper()
        short_pkg = str(package_id)[:8]
        new_file_name = f"{participant_id}_{stain_suffix}_{short_pkg}.svs"
        
        dest_path = BASE_DIR / stain_dir / new_file_name
        
        if download_file(url, dest_path):
            downloaded_count[stain_dir] += 1
        
    print("\n[다운로드 요약]")
    for stain_dir, count in downloaded_count.items():
        print(f" - {stain_dir}: {count}개 완료")
        
    print("="*60)
    print("다운로더가 모두 완료되었습니다.")
    print("="*60)

if __name__ == "__main__":
    main()
