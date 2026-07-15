# Legacy — v6.1 Prediction Panel (보존, 미사용)

여기 파일들은 **이전 v6.1 "예측형" CDSS UX**입니다(EMR+WSI → 환자의 tubular injury/만성변화/염증을
*예측·단언*). 현재 시스템은 **Clinical Phenotype 기반 Pathology Reference Retrieval**로 전환되어,
"예측"이 아니라 "유사 임상 phenotype의 KPMP 프로토타입을 *참조*로 검색"한다.

두 출력을 동시에 노출하면("Tubular Injury 82%" vs "ATI Prototype sim 0.84") 사용자가
"결국 예측한 것 아닌가?"라고 혼동하므로, 예측 패널은 **삭제하지 않고 숨겨** 둔다.

용도: 이전 포트폴리오/논문/비교 실험 참조. 라우팅·메인 UI에는 연결하지 않는다.

- `CdssV61Panel.tsx` — 3-layer 예측 패널 (L1 risk / L2 3축 / L3 descriptor)
- `cdssV61.ts` — 예측 판정 로직(late fusion: ABMIL + cdss_v5 OOF + EMR risk)
