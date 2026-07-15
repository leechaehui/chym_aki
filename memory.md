# 🧠 프로젝트 통합 메모리 (Project Memory)

이 문서는 프로젝트 진행 중 발견된 핵심 이슈, 진단 결과, 그리고 기술적 결정 사항들을 영구적으로 기록하는 메모리 파일입니다.

---

## [2026-06-29] Pathology Model: Multi-Stain MIL Attention Collapse 진단 및 가설 수정

### 📌 문제 정의
WSI 병리 모델(`StainAwareMIL`, `TaskAttentionMIL`)에서 **MT(Masson's Trichrome) Stain 가중치가 0.1%로 수렴**하고, Attention Heatmap이 병변을 찾지 못하고 **전체적으로 파랗게(Blue) 나오는 현상** 발생.

### 🔍 초기 진단(Epoch 2)의 한계와 새로운 가설 (중요!)
1. **Temperature Scaling은 오답이다**
   - 초기 진단에서 Raw Logit 분산이 `0.003`으로 낮아 Temperature Scaling을 제안했으나 이는 **잘못된 접근**임.
   - Logit 차이가 없는 상태에서 Temperature를 낮춰 분산을 강제로 벌려도, **패치 간의 순위(Rank)는 변하지 않으므로 잘못된 패치를 더 강하게 보게 만들 뿐**임.
2. **Epoch 2 결과의 함정**
   - MIL 모델은 본래 학습 초반(Epoch 2)에 Attention이 거의 Uniform(Entropy ≈ 1.0)하게 나옴. 
   - 따라서 초반 2 Epoch만 보고 Collapse라고 단정할 수 없으며, 후반(Epoch 30~50)까지 진행 과정을 봐야 함.
   - 실제로 H&E 75%, PAS 30%, MT 0.1%라는 Fusion Weight 붕괴도 후반 학습에서 생기는 현상임.

### 💡 새로운 핵심 의심 지점: "Attention Network가 아예 학습되지 않고 있다"
모델이 패치를 구별하지 못하는 진짜 이유는 Attention 계층(`self.w`, `self.U`, `self.V`)으로 **Gradient가 흐르지 않아 가중치가 업데이트되지 않고 있을 확률이 매우 높음**.
(예상 원인: Learning rate, Weight initialization, Gradient flow 단절, Gated Attention 구현 오류 등)

### 💡 확정된 원인 (50 Epoch & Embedding & Single-Stain 진단 결과 확인)
진단 스크립트 실행 결과, **입력 피처(CTransPath) 및 모달리티 자체의 한계가 아닌, Early Fusion 구조로 인한 Gradient Starvation**이 원인으로 최종 확정되었습니다.

**1. Embedding Health Check (`diag_embedding_collapse.py`)**
- 모든 Stain(HE, PAS, MT)의 Embedding Variance가 `0.00237 ~ 0.00241`로 동일함.
- Intra-bag Cosine Similarity 역시 모두 `0.86` 수준으로 동일함.
- 즉, 입력 피처 공간에서는 PAS와 MT가 HE와 동등한 수준의 풍부한 정보량과 분산을 가지고 있음. (Encoder 붕괴 가설 폐기)

**2. Gradient Starvation (`diag_gradient_flow.py`)**
- 학습 극초반(Epoch 1~5)에는 HE와 MT의 Z 벡터(Slide vector)로 가는 그래디언트가 동등(0.19)함.
- Epoch 10 기점으로 HE의 그래디언트는 0.43으로 폭등하고 MT는 0.05로 급락함.
- Epoch 50에는 MT/PAS 그래디언트가 0.008로 영구 소멸함. 즉, Starvation Point가 정확히 **Fusion Layer(`fw`)**임이 시각적으로 증명됨.

**3. Single-Stain Upper Bound 실험 (Experiment 0)**
각 Stain을 단독으로 학습시켜 한계 성능을 측정한 결과, PAS와 MT가 본래 무능한 것이 아니라 **특정 Task에서는 H&E를 능가하는 최고의 정보원**임이 증명되었습니다 (스모킹 건).
- **Immune**: H&E (0.689) 최고
- **Chronic**: **PAS (0.781)** 로 H&E(0.732) 압도
- **Stage 3**: **MT (0.630)** 로 H&E(0.543) 압도
- **ATI Severity**: **MT (0.282)** 로 H&E(0.095) 압도
**결론**: H&E가 초반 Loss를 빠르게 줄이면서 Fusion 밸브를 독점해버렸고, 이 구조적 결함(Early Fusion) 때문에 MT와 PAS가 가장 잘 푸는 Task마저 H&E에 갇혀 완전히 망가진 상태임.

### 🚀 해결 가이드 (Next Steps: 5-Step Architecture Evaluation)
구조를 갈아엎기 위해 한 번에 수정하지 않고, 다음 순서로 인과효과를 입증해 나가는 마스터플랜을 수행합니다.
1. **Experiment A (Baseline)**: Early Fusion의 한계점 기록 (Gradient, Fusion Entropy 등)
2. **Experiment B (Modality Dropout)**: H&E 편향을 깨기 위해 각 Stain별 독립 15% Drop 적용 (최소 1개 유지).
3. **Experiment C (Late Fusion)**: 분리된 Shared Head를 통해 강제 Loss 전파.
4. **Experiment D (Delayed Fusion)**: Cross-Stain Transformer 활용.
- **평가지표 필수 포함**: QWK, AUROC, Attention Entropy, Gradient Flow Map, **Fusion Diversity(Shannon Entropy)**, ECE.
1. **Late Fusion (앙상블)**: 예측 헤드를 각각 두고 마지막 확률을 앙상블.
2. **Modality Dropout**: 학습 중 H&E 피처를 확률적으로 마스킹하여 다른 브랜치로 그래디언트를 뚫어줌.
3. **가시화 개선 (Percentile Scaling)**: 히트맵 출력 시 상/하위 5~99 Percentile 기반의 동적 스케일링을 적용하여 미세한 차이를 시각화.

### 💡 [업데이트] Patch-level Attention (Instance Encoder) 완전 붕괴 확인 및 해결
- `diag_fusion_collapse.py`를 통해 상위 20개 패치를 추출한 결과, PAS/MT 모달리티에서 **모든 패치에 동일한 Attention Score(약 0.09)를 부여하는 Uniform Distribution** 상태임을 확인.
- 원인 분석 (Initialization Variance Collapse): `GatedAttentionMIL` 내부의 `tanh`와 `sigmoid` 활성화 함수 포화(Saturation)로 인해 Raw Logit의 표준편차가 수학적으로 `0.14`를 넘지 못하는 **초기화 병목** 발견. 5,000개 패치 환경에서 Logit 분산이 너무 작아 Gradient가 극도로 희석됨.
- 해결 및 검증 (Option B): 복잡한 Gated(tanh*sigmoid) 구조를 제거하고, `cdss_v5_model`에서 검증된 **Simple Linear Attention (`nn.Linear(dim, 1)`)**으로 아예 교체함.
- 검증 결과: 교체 직후 Logit의 표준편차가 `0.3 ~ 0.5` 수준으로 3~5배 이상 치솟으며, Attention의 대칭성(Symmetry)이 완전히 깨짐. PAS/MT 모델이 더 이상 파란 배경(Uniform)만 보지 않고 H&E처럼 의미 있는 점수 격차를 벌리기 시작함!

### ?? [������Ʈ] �ð�ȭ �ڵ� ���� (Color Normalization & Text Overlap) ����
- **Color Normalization ����**: CDSS v5 �ð�ȭ ��ũ��Ʈ(\cdss_v5_viz_heatmap40.py\)���� Heatmap ������ �� ������(Max)�� �Ķ���(Min)�� ����ġ�� \min\, \max\�� HE, PAS, MT ��޸�Ƽ ��ü ��ġ�� ���ļ� �۷ι��ϰ� ����ϰ� �־���. HE�� ��ó�� ��õ������ ������ ���� �۷ι� Vmax�� �����ϸ鼭, ��������� �������� ���� PAS�� MT�� �ֽ����� �ְ���(Red)�� �������� ���ϰ� û�ϻ�/�Ķ���(Cyan/Blue)���� �������Ǵ� �ð��� �ְ� �߰�.
- **�ذ�**: \min\�� \max\�� Stain(��޸�Ƽ) ���� �������� �̵�����, �� ���� �����̵庰�� �������� �����ϸ�(Local Normalization)�� �����ϵ��� ����. �� ��� PAS�� MT������ ������ �ֽ����� ���������� �����.
- **UI Layout ����**: ���� 'Top 16 Patches' �ؽ�Ʈ�� ���� ��ġ ��ȣ ���̺��� ���� ���� �������� ��ġ�� ���� �߰�.
- **�ذ�**: \x3.set_title(..., pad=25)\ ���� ������ �߰��Ͽ� ��ħ ���� �Ϻ� �ذ�.
