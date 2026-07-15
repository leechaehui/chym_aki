"""CHYM Retrieval CDSS — Clinical Phenotype 기반 Pathology Reference Retrieval.

Architecture (frozen):
  Clinical Data(MIMIC) → Clinical Concept Layer → OOD Detection
    → Prototype Retrieval(KPMP Atlas) → Top-K Representative WSI → Evidence Display

원칙: clinical→pathology *예측* 아님. 임상 phenotype이 유사한 KPMP 프로토타입을 *참조*로 검색.
MIMIC·KPMP 는 paired 가 아니며, 양쪽이 보유한 **공통 임상 concept 공간**에서 유사도를 비교한다.
"""
