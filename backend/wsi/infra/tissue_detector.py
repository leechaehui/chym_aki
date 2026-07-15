"""Tissue Detector — WSI 썸네일에서 조직(Tissue) 영역을 자동 검출 및 Bounding Box 계산.

OpenCV Connected Component Analysis 를 활용하여 2개 이상의 조직을 분리 관리 가능하게 함.
"""
from __future__ import annotations

import logging
from pathlib import Path
import cv2
import numpy as np

log = logging.getLogger("wsi.tissue")

def detect_tissues(svs_path: Path) -> list[dict[str, int]]:
    """WSI 에서 조직 영역을 검출하여 원본 레벨 0 기준 Bounding Box 목록을 반환.
    
    기본 반환 형식: [{'x': rx, 'y': ry, 'w': rw, 'h': rh, 'area': area}]
    면적(area) 내림차순 정렬.
    """
    try:
        import openslide
        osr = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        log.error("openslide 로드 실패: %s", e)
        return []

    orig_w, orig_h = osr.dimensions

    # 1) 썸네일 생성 (최대 1024 크기로 축소하여 고속 연산)
    thumb_target_size = 1024
    try:
        thumb = osr.get_thumbnail((thumb_target_size, thumb_target_size))
    except Exception as e:
        log.error("썸네일 생성 실패: %s", e)
        # 실패 시 전체 슬라이드를 단일 ROI로 반환
        return [{"x": 0, "y": 0, "w": orig_w, "h": orig_h, "area": orig_w * orig_h}]

    img = np.array(thumb)
    thumb_w, thumb_h = thumb.width, thumb.height

    # 2) Grayscale 변환 및 Gaussian Blur
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Background 제거 (Threshold)
    # 조직 부위는 상대적으로 어두우므로 반전 임계 처리
    _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)

    # 4) morphology close (자잘한 구멍 메우기)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5) Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    scale_x = orig_w / thumb_w
    scale_y = orig_h / thumb_h

    tissues = []
    min_area_ratio = 0.005 # 전체 이미지의 0.5% 미만인 잡음 제거

    for i in range(1, num_labels): # 0은 배경
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        # 면적 비율 검사
        if area_pixels < (thumb_w * thumb_h) * min_area_ratio:
            continue

        tx = stats[i, cv2.CC_STAT_LEFT]
        ty = stats[i, cv2.CC_STAT_TOP]
        tw = stats[i, cv2.CC_STAT_WIDTH]
        th = stats[i, cv2.CC_STAT_HEIGHT]

        # 원본 좌표계로 스케일링
        rx = int(tx * scale_x)
        ry = int(ty * scale_y)
        rw = int(tw * scale_x)
        rh = int(th * scale_y)

        # 이미지 경계 제한
        rx = max(0, min(rx, orig_w - 1))
        ry = max(0, min(ry, orig_h - 1))
        rw = max(1, min(rw, orig_w - rx))
        rh = max(1, min(rh, orig_h - ry))

        tissues.append({
            "x": rx,
            "y": ry,
            "w": rw,
            "h": rh,
            "area": int(rw * rh)
        })

    # 면적 내림차순 정렬
    tissues.sort(key=lambda t: t["area"], reverse=True)

    # 검출된 조직이 전혀 없으면 전체 이미지를 반환
    if not tissues:
        tissues.append({
            "x": 0,
            "y": 0,
            "w": orig_w,
            "h": orig_h,
            "area": orig_w * orig_h
        })

    log.info("detect_tissues: 검출된 조직 수 = %d (WSI=%s)", len(tissues), svs_path.name)
    return tissues
