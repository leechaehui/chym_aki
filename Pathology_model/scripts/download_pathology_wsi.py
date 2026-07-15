"""
KPMP Pathology WSI 50GB 다운로더

pathology_img.csv 파일을 읽어들여 50GB 제한선까지
필요한 염색(H&E, PAS, SIL, TRI, TOL)의 SVS 파일을 각 디렉토리에 맞게 다운로드합니다.
"""

import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import random

# 설정
CSV_PATH = (Path(__file__).resolve().parents[2] / "pathology_img.csv")
BASE_DIR = (Path(__file__).resolve().parents[2] / "data/raw")
# KPMP Atlas 실제 다운로드 엔드포인트 (구 /packages/{id}/files/{name} 경로는 404)
API_ENDPOINT = "https://atlas.kpmp.org/api/v1/file/download/{package_id}/{file_name}"
MAX_TOTAL_BYTES = 50 * 1024 * 1024 * 1024  # 50GB 제한

STAIN_MAP = {
    "H&E stain": "wsi_he",
    "Frozen H&E stain": "wsi_he",
    "PAS stain": "wsi_pas",
    "SIL stain": "wsi_silver",
    "TRI stain": "wsi_mt",
    "TOL stain": "wsi_tol"
}

# 이번 실행에서 받을 염색 종류만 지정 (None 또는 빈 리스트면 STAIN_MAP 전체 대상)
TARGET_STAINS = ["PAS stain"]

def parse_size(size_str: str) -> int:
    """문자열 사이즈(예: '63.6 MB')를 바이트 정수로 변환합니다."""
    if pd.isna(size_str):
        return 0
        
    size_str = str(size_str).strip().upper()
    try:
        val_str, unit = size_str.split(" ")
        val = float(val_str)
    except ValueError:
        return 0
        
    multiplier = 1
    if unit == "KB":
        multiplier = 1024
    elif unit == "MB":
        multiplier = 1024 ** 2
    elif unit == "GB":
        multiplier = 1024 ** 3
    elif unit == "B":
        multiplier = 1
        
    return int(val * multiplier)

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
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    print("="*60)
    print(f"KPMP Pathology WSI 다운로더 시작 (최대 한도: {MAX_TOTAL_BYTES / 1024**3:.2f} GB)")
    print("="*60)
    
    if not CSV_PATH.exists():
        print(f"오류: {CSV_PATH} 파일을 찾을 수 없습니다.")
        sys.exit(1)
        
    # CSV 로드
    df = pd.read_csv(CSV_PATH)
    
    # 1. 지원하는 SVS 파일 및 Stain 필터링
    df_filtered = df[(df["Data Format"] == "svs") & (df["Access"] == "open")].copy()
    allowed_stains = set(TARGET_STAINS) if TARGET_STAINS else set(STAIN_MAP.keys())
    df_filtered = df_filtered[df_filtered["Workflow Type"].isin(allowed_stains)]
    print(f"이번 실행 대상 염색: {sorted(allowed_stains)}")
    
    # 2. 파일 사이즈 파싱
    df_filtered["SizeBytes"] = df_filtered["Size"].apply(parse_size)
    
    # 다양한 샘플을 받기 위해 랜덤 셔플 (재현성을 위해 시드 고정)
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"총 {len(df)}개 파일 중 대상 svs 파일 수: {len(df_filtered)}개")
    
    # 디렉토리 생성
    for stain_dir in set(STAIN_MAP.values()):
        os.makedirs(BASE_DIR / stain_dir, exist_ok=True)
        
    accumulated_bytes = 0
    downloaded_count = {stain: 0 for stain in set(STAIN_MAP.values())}
    
    for index, row in df_filtered.iterrows():
        file_size = row["SizeBytes"]
        
        # 50GB 한도를 넘으면 즉시 중단
        if accumulated_bytes + file_size > MAX_TOTAL_BYTES:
            print(f"\\n[INFO] 누적 다운로드 용량 한도({MAX_TOTAL_BYTES / 1024**3:.2f} GB)에 도달하여 다운로드를 중단합니다.")
            break
            
        workflow_type = row["Workflow Type"]
        stain_dir = STAIN_MAP.get(workflow_type)
            
        package_id = row["Internal Package ID"]
        file_name = row["File Name"]
        participant_id = row["Participant ID"]
        
        # 파일명 중복을 피하고 관리하기 쉽도록 네이밍 
        stain_suffix = stain_dir.replace("wsi_", "").upper()
        short_pkg = str(package_id)[:8]
        new_file_name = f"{participant_id}_{stain_suffix}_{short_pkg}.svs"
        dest_path = BASE_DIR / stain_dir / new_file_name
        
        url = API_ENDPOINT.format(package_id=package_id, file_name=file_name)
        
        if download_file(url, dest_path):
            downloaded_count[stain_dir] += 1
            accumulated_bytes += file_size
            
    print("\\n[다운로드 요약]")
    print(f"누적 다운로드 용량: {accumulated_bytes / 1024**3:.2f} GB")
    for stain_dir, count in downloaded_count.items():
        print(f" - {stain_dir}: {count}개 완료")
        
    print("="*60)
    print("다운로더가 모두 완료되었습니다.")
    print("="*60)

if __name__ == "__main__":
    main()
