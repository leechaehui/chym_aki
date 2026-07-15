"""
디지털 병리 파이프라인 (Phase 3 ~ 19) 통합 실험 스크립트.
구축된 클래스와 인터페이스(SOLID)가 정상적으로 맞물려 돌아가는지 확인합니다.
"""
import os
import sys

# 백엔드 루트를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Phase 모듈 임포트
from ml_models.phase4_normalization import Phase4NormalizationContext, Phase4MacenkoNormalization
from ml_models.phase5_7_patching import Phase5TissueDetector, Phase6PatchExtractor, Phase7PatchFilter
from ml_models.phase8_foundation_models import Phase8ModelContext, Phase8UniModel
from ml_models.phase9_mil import Phase9ClamSbModel
from services.phase15_18_validation import Phase15AttentionValidator
from services.phase19_fusion import Phase19FusionEngine

def run_experiment():
    print("="*60)
    print("[실험 시작] CHYM-AKI 디지털 병리 파이프라인 통합 테스트")
    print("="*60)
    
    # 1. 패치 전처리 (Phase 5~7)
    print("\n▶ [Phase 5~7] 썸네일 로드 및 패치 추출 시뮬레이션")
    dummy_thumbnail = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    phase5_detector = Phase5TissueDetector()
    tissue_mask = phase5_detector.detect_tissue_mask(dummy_thumbnail)
    print(f" - 조직 마스크 생성 완료 (크기: {tissue_mask.shape})")
    
    phase6_extractor = Phase6PatchExtractor(patch_size=512, stride=512)
    patches = phase6_extractor.extract_patches(slide_obj=None, tissue_mask=tissue_mask)
    print(f" - 패치 좌표 추출 완료 (총 {len(patches)}개 패치)")
    
    phase7_filter = Phase7PatchFilter()
    valid_patches = phase7_filter.filter_patches(patches)
    print(f" - 패치 필터링 통과 (유효 패치: {len(valid_patches)}개)")
    
    # 2. 정규화 (Phase 4)
    print("\n▶ [Phase 4] 염색 정규화 (Macenko 전략 적용)")
    norm_context = Phase4NormalizationContext(Phase4MacenkoNormalization())
    # 가상의 패치 이미지 데이터
    dummy_patch_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    normalized_patch = norm_context.execute_normalization(dummy_patch_img)
    print(f" - 정규화 완료")
    
    # 3. 파운데이션 모델 (Phase 8)
    print("\n▶ [Phase 8] 파운데이션 모델 임베딩 (UNI 모델 전략 적용)")
    # 유효 패치 개수만큼 더미 이미지 리스트 생성
    patch_images = [dummy_patch_img for _ in range(len(valid_patches))]
    model_context = Phase8ModelContext(Phase8UniModel())
    embeddings = model_context.get_embeddings(patch_images)
    print(f" - 임베딩 추출 완료 (Shape: {embeddings.shape})")
    
    # 4. MIL 추론 (Phase 9)
    print("\n▶ [Phase 9] MIL 기반 예측 (CLAM-SB 적용)")
    mil_model = Phase9ClamSbModel()
    mil_result = mil_model.predict(embeddings)
    print(f" - 슬라이드 예측 점수: {mil_result['score']:.4f}")
    
    # 5. 어텐션 검증 (Phase 15)
    print("\n▶ [Phase 15] 어텐션 안정성 검증")
    validator = Phase15AttentionValidator()
    stability = validator.validate_attention_stability(mil_result['attention'])
    print(f" - 엔트로피: {stability['entropy']:.4f}, 분산: {stability['variance']:.4f}")
    
    # 6. 임상 데이터 융합 (Phase 19)
    print("\n▶ [Phase 19] 임상 + 병리 Fusion Engine 판정")
    fusion_engine = Phase19FusionEngine()
    dummy_clinical = {"creatinine_max": 2.5}
    fusion_result = fusion_engine.generate_fusion_assessment(dummy_clinical, mil_result['score'])
    print(f" - 최종 융합 점수: {fusion_result['fusion_score']:.4f}")
    print(f" - 판정 결과: {fusion_result['conclusion']}")
    
    print("\n" + "="*60)
    print("[실험 종료] 파이프라인이 성공적으로 연결되어 구동되었습니다.")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
