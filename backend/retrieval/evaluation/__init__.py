"""Retrieval Evaluation — Ground Truth 없는 cross-cohort retrieval 평가.

논문/발표의 핵심 질문 "성능을 어떻게 평가했는가?"에 답하기 위해 처음부터 별도 모듈로 구축.
Top-K 정답이 없으므로 분포·일관성·기각률 중심으로 자동 기록한다:
  - abstain_rate        : OOD 게이트로 기각된 비율
  - ood_statistics      : OOD score 분포·임계 통계
  - prototype_distribution: 어떤 프로토타입이 얼마나 검색됐는가(편중 점검)
  - retrieval_consistency: 유사 concept → 유사 결과인가(섭동/근접 안정성)
모두 retrieval_logs(또는 주입 레코드)에서 산출 → 결정적·재현 가능.
"""
