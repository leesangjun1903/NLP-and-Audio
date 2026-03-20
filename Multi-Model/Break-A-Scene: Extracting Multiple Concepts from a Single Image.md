# Break-A-Scene: Extracting Multiple Concepts from a Single Image

---

## 1. 핵심 주장 및 주요 기여 요약

**Break-A-Scene**은 단일 이미지에서 여러 시각적 개념(concept)을 동시에 추출하여 각각에 대한 고유한 텍스트 토큰(handle)을 학습하고, 이를 활용해 개별 또는 조합된 개념을 새로운 맥락에서 재합성할 수 있는 **텍스트 장면 분해(textual scene decomposition)** 방법을 제안합니다.

### 주요 기여:
1. **새로운 과제 정의**: 단일 이미지에서 여러 개념을 분리 추출하는 "textual scene decomposition" 과제를 최초로 정의
2. **새로운 학습 파이프라인**: 2단계 최적화(two-phase customization), 마스크 기반 확산 손실(masked diffusion loss), 교차 주의(cross-attention) 손실, 합집합 샘플링(union-sampling)을 결합한 방법 제안
3. **자동 평가 메트릭 및 사용자 연구**: COCO 데이터셋 기반 자동 평가 파이프라인과 AMT 사용자 연구를 통한 정량적·정성적 검증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 텍스트-이미지 모델 개인화(personalization) 방법들(Textual Inversion, DreamBooth 등)은 **하나의 개념**을 학습하기 위해 **여러 장의 이미지**를 필요로 합니다. 그러나 실제 응용에서는 **단일 이미지 안에 여러 개념**이 존재하며, 이를 각각 독립적으로 추출하여 새로운 장면에서 재합성하는 것이 필요합니다.

기존 방법을 이 설정에 적용하면 **재구성-편집성 트레이드오프(reconstruction-editability tradeoff)**가 발생합니다:
- **Textual Inversion (TI)**: 편집성은 유지하지만 개념의 정체성(identity) 보존에 실패
- **DreamBooth (DB)**: 정체성은 보존하지만 과적합(overfitting)으로 인해 텍스트 프롬프트를 따르지 못함

### 2.2 제안하는 방법

본 논문은 4가지 핵심 구성 요소를 결합한 파이프라인을 제안합니다.

#### (1) 2단계 최적화 (Two-Phase Customization)

- **1단계**: 모델 가중치를 동결(freeze)하고, 새로 추가된 텍스트 임베딩(handles) $\{v_i\}_{i=1}^{N}$만 높은 학습률($5 \times 10^{-4}$)로 최적화
- **2단계**: 모델 가중치를 해동(unfreeze)하여 텍스트 토큰과 함께 낮은 학습률($2 \times 10^{-6}$)로 미세 조정

이 전략은 과적합을 방지하면서도 충분한 재구성 능력을 확보합니다.

#### (2) 합집합 샘플링 (Union-Sampling)

각 학습 단계에서 $N$개의 개념 중 무작위 부분집합 $s = \{i_1, \ldots, i_k\} \subseteq [N]$을 선택하여 텍스트 프롬프트 "a photo of $[v_{i_1}]$ and ... $[v_{i_k}]$"를 구성합니다. 손실은 대응하는 마스크의 합집합 $M_s = \bigcup M_{i_k}$에 대해 계산됩니다. 이를 통해 여러 개념의 조합 생성 능력을 향상시킵니다.

#### (3) 마스크 확산 손실 (Masked Diffusion Loss)

마스크 영역 내의 픽셀에 대해서만 표준 확산 손실을 적용합니다:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{z, s, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon \odot M_s - \epsilon_\theta(z_t, t, p_s) \odot M_s\|_2^2\right]$$

여기서 $z_t$는 시간 단계 $t$에서의 노이즈 잠재 변수, $p_s$는 텍스트 프롬프트, $M_s$는 마스크 합집합, $\epsilon$은 추가된 노이즈, $\epsilon_\theta$는 디노이징 네트워크입니다.

이 손실은 각 핸들이 지정된 개념을 재구성하도록 보장하지만, **하나의 핸들이 여러 개념에 연관되는 것을 방지하지는 못합니다**.

#### (4) 교차 주의 손실 (Cross-Attention Loss)

개념 간 얽힘(entanglement)을 방지하기 위해, 교차 주의 맵이 각 핸들의 대응 마스크와 일치하도록 하는 추가 손실을 도입합니다:

$$\mathcal{L}_{\text{attn}} = \mathbb{E}_{z, k, t}\left[\|CA_\theta(v_i, z_t) - M_{i_k}\|_2^2\right]$$

여기서 $CA_\theta(v_i, z_t)$는 토큰 $v_i$와 노이즈 잠재 변수 $z_t$ 사이의 교차 주의 맵(16×16 해상도에서 평균 및 정규화)입니다.

#### 최종 손실 함수:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{attn}} \mathcal{L}_{\text{attn}}$$

여기서 $\lambda_{\text{attn}} = 0.01$입니다.

### 2.3 모델 구조

- **기반 모델**: Stable Diffusion V2.1 (Latent Diffusion Model)
- **UNet 구조**: self-attention → cross-attention 레이어가 해상도 8, 16, 32, 64에서 반복
- **최적화 대상**:
  - 1단계: 새로 추가된 텍스트 임베딩만
  - 2단계: UNet 가중치 + 텍스트 인코더 가중치 + 텍스트 임베딩
- **옵티마이저**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.99$, weight decay $= 1 \times 10^{-8}$)
- **학습 단계**: 각 단계 400 스텝

### 2.4 성능 향상

**정량적 결과** (COCO 데이터셋 기반 5400개 이미지-텍스트 쌍):
- **Prompt Similarity** (CLIP 유사도): TI-m/CD-m 수준의 높은 프롬프트 대응력 유지
- **Identity Similarity** (DINO 임베딩): DB-m에 근접하는 정체성 보존
- **Pareto Front**에 위치하여 두 메트릭 간 최적의 균형 달성

**사용자 연구** (AMT, Likert 1-5 척도):

| Method | Identity Similarity | Prompt Similarity |
|--------|-------------------|-------------------|
| TI-m | $2.69 \pm 1.3$ | $3.88 \pm 1.21$ |
| DB-m | $3.97 \pm 0.95$ | $2.37 \pm 1.11$ |
| CD-m | $2.47 \pm 1.3$ | $4.08 \pm 1.12$ |
| ELITE | $3.05 \pm 1.31$ | $3.53 \pm 1.31$ |
| **Ours** | $\mathbf{3.56 \pm 1.27}$ | $\mathbf{3.85 \pm 1.21}$ |

모든 베이스라인 대비 통계적으로 유의한 차이 확인 ($p < 0.05$, Tukey's HSD).

**Ablation 결과**:
- 1단계 제거 → 프롬프트 유사도 크게 하락 (과적합)
- 마스크 손실 제거 → 배경 학습으로 프롬프트 유사도 하락
- 교차 주의 손실 제거 → 개념 얽힘으로 정체성 유사도 하락
- Union-sampling 제거 → 다중 개념 생성 능력 저하

### 2.5 한계

1. **조명 불일치 (Inconsistent Lighting)**: 단일 이미지의 조명 조건이 학습된 개념에 얽혀, 새로운 환경에서도 원래 조명이 유지됨
2. **자세 고착 (Pose Fixation)**: 입력 이미지의 자세가 정체성과 얽혀 다른 자세 생성이 어려움
3. **다수 개념 과소적합**: 4개 이상의 개념 추출 시 정체성 학습 실패
4. **높은 계산 비용**: 단일 장면당 약 4.5분 소요, 실시간 응용에 제약

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 강점

Break-A-Scene의 일반화 성능은 다음 메커니즘에 의해 확보됩니다:

1. **2단계 학습의 과적합 방지**: 1단계에서 모델 가중치를 동결하고 텍스트 임베딩만 최적화하여 사전 학습된 모델의 일반성을 보존합니다. 2단계의 낮은 학습률($2 \times 10^{-6}$)은 미세 조정의 과적합 위험을 최소화합니다.

2. **마스크 기반 학습**: $\mathcal{L}_{\text{rec}}$에서 마스크 영역만 페널라이즈하여 배경 정보가 개념에 혼입되는 것을 방지하고, 새로운 배경에서의 일반화를 촉진합니다.

3. **교차 주의 손실**: $\mathcal{L}_{\text{attn}}$이 각 핸들의 주의를 해당 개념 영역으로 제한하여, 개념 간 분리(disentanglement)를 통한 독립적 일반화를 가능하게 합니다.

4. **Union-Sampling**: 다양한 개념 조합을 학습 시 노출시켜, 추론 시 임의의 조합에 대한 일반화 능력을 향상시킵니다.

### 3.2 일반화 성능의 한계와 개선 방향

1. **단일 이미지의 근본적 한계**: 단일 뷰에서만 개념을 관찰하므로, 자세·조명·시점의 다양성이 부족합니다. **데이터 증강(augmentation)** 전략의 강화나 **3D-aware 표현**의 도입이 일반화를 개선할 수 있습니다.

2. **개념 수 확장성**: 4개 이상의 개념에서 성능이 저하됩니다. 이는 제한된 텍스트 임베딩 공간의 용량 문제로, **확장된 역전 공간**(XTI [Voynov et al. 2023]의 P+ 공간 등)이나 **계층적 개념 분해**가 해결책이 될 수 있습니다.

3. **빠른 개인화 인코더와의 결합**: 현재 최적화 기반 방법의 높은 계산 비용은 일반화 실험의 규모를 제한합니다. **인코더 기반 접근법**(E4T, ELITE 등)과의 결합을 통해 빠른 초기화 후 미세 조정하는 하이브리드 방식이 일반화와 효율성을 동시에 개선할 수 있습니다.

4. **도메인 일반화**: COCO 데이터셋 중심의 평가로, **예술 작품, 의료 영상, 위성 사진** 등 다양한 도메인에서의 일반화 성능은 검증되지 않았습니다. 사전 학습 데이터의 다양성과 도메인 적응 기법의 적용이 필요합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **새로운 연구 방향 개척**: "단일 이미지 다중 개념 추출"이라는 새로운 과제를 정의하여, 이후 ConceptExpress (2024), InstantFamily (2024) 등 후속 연구에 영감을 제공
2. **교차 주의 맵의 활용 확장**: 단순 이미지 편집을 넘어 개념 분리에 교차 주의 맵을 활용하는 새로운 패러다임 제시
3. **개인화 기법의 실용화**: 단일 이미지만으로 다중 개념을 추출할 수 있어, 데이터 수집 부담을 획기적으로 줄임
4. **평가 체계 기여**: 다중 주체(multi-subject) 정체성 유사도 평가 방법을 제안하여 후속 연구의 평가 표준에 기여

### 4.2 향후 연구 시 고려할 점

1. **효율성 개선**: 4.5분의 최적화 시간을 줄이기 위한 인코더 기반 초기화, LoRA/SVD 기반 경량 미세 조정 등
2. **더 많은 개념 지원**: 현재 4개 한계를 넘어 10개 이상의 개념을 다룰 수 있는 확장성 확보
3. **3D 및 비디오 확장**: 단일 이미지에서 3D 객체로의 확장, 비디오 내 다중 개념 추출
4. **자동 마스크 생성과의 통합**: SAM(Segment Anything Model) 등과의 완전 자동 파이프라인 구축
5. **개념의 속성 분리**: 형태, 질감, 색상 등 세부 속성 수준의 분리와 재조합
6. **윤리적 고려**: 저작권 침해 및 딥페이크 악용 가능성에 대한 안전장치 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 단일 이미지 | 다중 개념 | 핵심 차별점 |
|------|------|-----------|----------|-----------|
| **Textual Inversion** [Gal et al.] | 2022 | ✗ | ✗ | 텍스트 임베딩만 최적화, 다수 이미지 필요 |
| **DreamBooth** [Ruiz et al.] | 2023 | ✗ | ✗ | 모델 전체 미세 조정, 높은 재구성/낮은 편집성 |
| **Custom Diffusion** [Kumari et al.] | 2023 | ✗ | ✓ | 교차 주의 가중치만 미세 조정, 다수 이미지 필요 |
| **ELITE** [Wei et al.] | 2023 | ✓ | ✗ | 인코더 기반 빠른 개인화, 단일 개념만 지원 |
| **E4T** [Gal et al.] | 2023 | ✓ | ✗ | 인코더 기반 도메인 튜닝 |
| **SVDiff** [Han et al.] | 2023 | ✗ | ✓ | SVD 기반 경량 미세 조정, 다수 이미지 필요 |
| **Perfusion** [Tewel et al.] | 2023 | ✗ | ✓ | Rank-1 업데이트 + key locking |
| **P+** (XTI) [Voynov et al.] | 2023 | ✓ | ✗ | 확장된 텍스트 역전 공간 활용 |
| **Break-A-Scene** (본 논문) | 2023 | **✓** | **✓** | **단일 이미지 + 다중 개념 최초 지원** |
| **IP-Adapter** [Ye et al.] | 2023 | ✓ | ✗ | 이미지 프롬프트 어댑터, 튜닝 불필요 |
| **BLIP-Diffusion** [Li et al.] | 2023 | ✓ | ✗ | 사전 학습된 멀티모달 인코더 활용 |
| **Subject-driven T2I via Apprenticeship** [Chen et al.] | 2023 | ✗ | ✗ | Apprenticeship learning 기반 |
| **InstantBooth** [Shi et al.] | 2023 | ✓ | ✗ | 테스트 시 미세 조정 불필요 |

### 주요 비교 분석:

**vs. Textual Inversion / DreamBooth**: Break-A-Scene은 두 방법의 장점을 결합합니다. TI의 편집성(모델 동결)과 DB의 재구성 능력(가중치 미세 조정)을 2단계 학습으로 통합하되, 마스크와 교차 주의 손실로 다중 개념 분리를 추가합니다.

**vs. Custom Diffusion**: CD는 다중 개념을 지원하지만 각 개념에 대해 별도의 이미지 세트가 필요합니다. Break-A-Scene은 단일 이미지에서 동작하며, union-sampling으로 개념 조합 생성 능력을 확보합니다.

**vs. ELITE**: ELITE는 인코더 기반으로 빠르지만 단일 개념만 지원하고, 다중 개념 시 정체성 보존이 현저히 저하됩니다. Break-A-Scene은 최적화 기반이라 느리지만 다중 개념의 정체성을 더 잘 보존합니다.

**vs. SVDiff**: SVDiff는 두 개념의 혼합/분리를 지원하지만 각 개념에 여러 이미지가 필요하고, 두 객체의 나란한 배치만 가능합니다. Break-A-Scene은 단일 이미지에서 최대 4개 개념의 임의 배치를 지원합니다.

---

## 참고자료

1. Avrahami, O., Aberman, K., Fried, O., Cohen-Or, D., & Lischinski, D. (2023). "Break-A-Scene: Extracting Multiple Concepts from a Single Image." *SIGGRAPH Asia 2023 Conference Papers*. ACM. arXiv:2305.16311
2. Gal, R., et al. (2022). "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." *ICLR 2023*.
3. Ruiz, N., et al. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." *CVPR 2023*.
4. Kumari, N., et al. (2023). "Multi-Concept Customization of Text-to-Image Diffusion." *CVPR 2023*.
5. Wei, Y., et al. (2023). "ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation." arXiv:2302.13848
6. Han, L., et al. (2023). "SVDiff: Compact Parameter Space for Diffusion Fine-Tuning." arXiv:2303.11305
7. Hertz, A., et al. (2022). "Prompt-to-Prompt Image Editing with Cross-Attention Control." *ICLR 2023*.
8. Tewel, Y., et al. (2023). "Key-Locked Rank One Editing for Text-to-Image Personalization." *SIGGRAPH 2023*.
9. Voynov, A., et al. (2023). "P+: Extended Textual Conditioning in Text-to-Image Generation." arXiv:2303.09522
10. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
11. Gal, R., et al. (2023). "Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models." *ACM TOG*.
12. Roich, D., et al. (2021). "Pivotal Tuning for Latent-based Editing of Real Images." *ACM TOG*.
13. Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
14. Kirillov, A., et al. (2023). "Segment Anything." arXiv:2304.02643
15. 프로젝트 페이지: https://omriavrahami.com/break-a-scene/
