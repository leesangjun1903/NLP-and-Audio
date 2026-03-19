# Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models

---

## 1. 핵심 주장과 주요 기여 (요약)

최근 텍스트-이미지 생성 모델들은 텍스트 프롬프트 기반의 다양하고 창의적인 이미지 생성에서 비할 데 없는 능력을 보여주었지만, 현존하는 최첨단 확산 모델들은 여전히 주어진 텍스트 프롬프트의 의미를 완전히 전달하는 이미지 생성에 실패할 수 있다.

이 논문(Chefer et al., SIGGRAPH 2023, ACM Transactions on Graphics)의 핵심 기여는 다음과 같다:

1. **Catastrophic Neglect 문제 정의**: 공개적으로 사용 가능한 Stable Diffusion 모델을 분석하여, 모델이 입력 프롬프트에서 하나 이상의 주체(subject)를 생성하지 못하는 **치명적 무시(catastrophic neglect)** 현상의 존재를 확인하였다.
2. **속성 바인딩 오류 식별**: 일부 경우에서 모델이 속성(예: 색상)을 해당하는 주체에 올바르게 바인딩하지 못하는 것을 발견하였다.
3. **Generative Semantic Nursing (GSN) 개념 도입**: 이러한 실패 사례를 완화하기 위해, 추론 시간(inference time)에 실시간으로(on the fly) 생성 과정에 개입하여 생성된 이미지의 충실도를 향상시키는 **Generative Semantic Nursing(GSN)** 개념을 도입하였다.
4. **Attend-and-Excite 방법 제안**: GSN의 어텐션 기반 구현인 Attend-and-Excite를 통해, cross-attention 유닛이 텍스트 프롬프트의 모든 주체 토큰에 주의를 기울이고 그 활성화를 강화(excite)하도록 모델을 안내하여, 텍스트 프롬프트에 기술된 모든 주체를 생성하도록 유도한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 텍스트-이미지 모델의 두 가지 의미적 문제를 관찰한다. 첫째, **치명적 무시(catastrophic neglect)**—모델이 하나 이상의 주체를 생성하지 않는 것(예: 고양이가 생성되지 않음). 둘째, **잘못된 속성 바인딩(incorrect attribute binding)**—색상과 같은 속성이 잘못된 주체에 매칭되는 것(예: 벤치가 갈색 대신 노란색으로 색칠됨).

예를 들어, "a cat and a dog" 프롬프트에서 Stable Diffusion은 고양이만 생성하고 개를 무시하는 경우가 빈번하며, "a red car and a blue bicycle"에서 색상이 뒤바뀌는 문제가 발생한다.

### 2.2 제안하는 방법 및 수식

#### Cross-Attention 메커니즘의 분석

Stable Diffusion에서 텍스트 조건화는 cross-attention 메커니즘을 통해 수행된다. 어텐션 행렬은 텍스트 토큰별 공간 맵(spatial map)으로 재구성할 수 있으며, 직관적으로 토큰이 생성 이미지에 나타나려면 해당 맵에서 적어도 하나의 패치가 높은 활성값을 가져야 한다.

Cross-attention의 기본 구조는 다음과 같다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

여기서:
- $Q = W_Q \cdot \phi(z_t)$: 공간 특성(latent features)에서 나온 **Query**
- $K = W_K \cdot \psi(P)$: 텍스트 임베딩에서 나온 **Key**
- $V = W_V \cdot \psi(P)$: 텍스트 임베딩에서 나온 **Value**
- $\phi(z_t)$: 시간 $t$에서의 latent 특성
- $\psi(P)$: 프롬프트 $P$의 텍스트 인코딩
- $d$: 특성 차원

Cross-attention 맵 $A$는 다음과 같이 정의된다:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \in \mathbb{R}^{N \times L}$$

여기서 $N$은 공간 패치 수, $L$은 텍스트 토큰 수이다. 각 토큰 $s$에 대한 공간 어텐션 맵은 $A_{:,s} \in \mathbb{R}^{N}$이다.

#### Attend-and-Excite의 핵심 손실 함수

주어진 프롬프트(예: "A lion with a crown")에서 주체 토큰(lion, crown)과 그에 대응하는 어텐션 맵을 추출하고, 각 어텐션 맵에 가우시안 커널을 적용하여 이웃 패치를 고려한 평활화된 어텐션 맵을 얻는다. 최적화는 시간 $t$에서 가장 무시된(most neglected) 토큰의 최대 활성값을 강화하고, 이에 따라 latent 코드를 업데이트한다.

**가우시안 평활화(Gaussian Smoothing):**

$$\hat{A}_{:,s} = G_\sigma * A_{:,s}$$

여기서 $G_\sigma$는 표준편차 $\sigma$인 가우시안 커널이며, $*$는 합성곱 연산이다.

**핵심 손실 함수 — 가장 무시된 주체 토큰의 최대 활성값 최대화:**

주체 토큰 집합 $S = \{s_1, s_2, \ldots, s_k\}$에 대해, 각 토큰 $s$의 평활화된 어텐션 맵에서의 최대 활성값:

$$\text{max val}(s) = \max_{n} \hat{A}_{n, s}$$

**가장 무시된 토큰(most neglected subject token):**

$$s^* = \arg\min_{s \in S} \text{max val}(s)$$

**최종 손실 함수:**

```math
\mathcal{L}_{\text{A\&E}} = 1 - \text{max\_val}(s^*) = 1 - \min_{s \in S} \max_{n} \hat{A}_{n, s}
```

이 손실은 가장 무시된 토큰의 활성화가 최대화되도록 유도하며, 이를 통해 모든 주체 토큰이 공간에서 적어도 하나의 높은 활성값을 갖도록 보장한다.

#### Latent 업데이트 규칙

각 디노이징 타임스텝 $t$에서, latent 코드 $z_t$를 gradient descent를 통해 업데이트한다:

```math
z_t \leftarrow z_t - \alpha_t \cdot \nabla_{z_t} \mathcal{L}_{\text{A\&E}}
```

여기서 $\alpha_t$는 시간 $t$에서의 step size(학습률)이다. 이 업데이트를 $n$ 회 반복적 정제(iterative refinement) 후, 표준 디노이징 스텝으로 진행한다.

Attend-and-Excite는 이 직관을 구체화하여, latent를 이동시킴으로써 텍스트의 모든 주체 토큰에 어텐션을 기울이도록 유도한다.

#### 전체 알고리즘 (Generative Semantic Nursing)

각 디노이징 타임스텝 $t = T, T-1, \ldots, 1$에서:

1. 현재 $z_t$로부터 U-Net의 forward pass를 수행하여 cross-attention 맵 $A_t$ 추출
2. 주체 토큰별 가우시안 평활화 적용: $\hat{A}\_{t,:,s} = G_\sigma * A_{t,:,s}$
3. 가장 무시된 토큰 탐색: $s^* = \arg\min_{s \in S} \max_{n} \hat{A}_{t,n,s}$
4. 손실 계산: $\mathcal{L} = 1 - \max_{n} \hat{A}_{t,n,s^*}$
5. Latent 업데이트: $z_t \leftarrow z_t - \alpha_t \nabla_{z_t}\mathcal{L}$ (반복)
6. 표준 디노이징 스텝으로 $z_{t-1}$ 생성

이 과정은 **추론 시간에만** 수행되며, **모델 재학습이 필요 없다(training-free).**

### 2.3 모델 구조

Attend-and-Excite는 **Stable Diffusion (v1.4)** 을 기반 모델로 사용하며, 모델의 아키텍처를 수정하지 않는다:

- **텍스트 인코더**: CLIP 텍스트 인코더 ($\psi$)
- **U-Net**: latent space에서의 디노이징 네트워크 ($\epsilon_\theta$)
- **VAE 디코더**: latent → pixel space 변환

사전 학습된 텍스트-이미지 확산 모델(예: Stable Diffusion)이 주어지면, Attend-and-Excite는 이미지 합성 과정에서 cross-attention 값을 수정하도록 생성 모델을 안내하여, 입력 텍스트 프롬프트를 더 충실하게 묘사하는 이미지를 생성한다.

핵심은 U-Net 내부의 **cross-attention layer**에 대한 개입이며, 특히 16×16 해상도의 cross-attention 맵을 사용한다.

### 2.4 성능 향상

대안적 접근법들과 비교하여, 다양한 텍스트 프롬프트에 걸쳐 원하는 개념을 더 충실하게 전달함을 입증하였다.

**정량적 평가 지표:**
- 이미지 기반 CLIP 유사도(image-based CLIP similarity)와 BLIP 캡셔닝 후 텍스트 기반 CLIP 유사도(text-based CLIP similarity)를 사용하여 정량적 실험을 수행하였다.

**비교 대상**:
- 기본 Stable Diffusion
- Composable Diffusion (Liu et al., 2022)
- StructureDiffusion (Feng et al., 2023)

Attend-and-Excite는 Generative Semantic Nursing의 도입과 cross-attention 메커니즘의 신중한 조작을 통해, 모델 재학습 없이 구성적 이해에서 상당한 개선을 달성하였다. 종합적인 평가에서 여러 지표와 인간 선호도에서 명확한 이점을 보여주며, 어텐션 기반 접근법은 모델의 해석 가능성(interpretability)도 향상시킨다.

### 2.5 한계점

논문 및 후속 연구에서 지적된 한계점:

1. **추론 비용 증가**: 각 타임스텝마다 반복적 latent 업데이트가 필요하여 추론 시간이 증가함
2. Attend-and-Excite와 같은 내부 어텐션 최적화 방법은 특정 의미적 객체의 생성을 보장하는 데 집중하지만(cross-attention 그래프 조정), '객체 존재'에만 초점을 맞추며 전체 레이아웃이나 구성 요소 간 관계와 같은 복잡한 구조적 속성에 대한 명시적 모델링이 부족하다.
3. 이 방법의 손실 함수는 주어진 타임스텝에서 가장 작은 어텐션 맵의 존재를 최대화하여 선택된 토큰 집합이 생성 이미지에 포함되도록 보장하는 것을 목표로 하지만, 언어적으로 관련된 단어들의 쌍별 관계(pairwise relations)에는 의존하지 않는다.
4. **속성 바인딩의 불완전 해결**: 주체 존재는 개선하지만 색상-객체 바인딩 문제를 완전히 해결하지는 못함
5. **복잡한 프롬프트 한계**: 3개 이상의 주체나 복잡한 공간적 관계를 포함하는 프롬프트에서는 여전히 어려움이 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Training-Free 접근법의 일반화 강점

Attend-and-Excite는 추론 시 cross-attention 맵을 조작하여, 모델 재학습 없이(without requiring model retraining) 지정된 모든 주체가 생성 이미지에 존재하고 올바르게 속성이 부여되도록 보장하며, 의미적 충실도와 신뢰성을 향상시킨다.

이러한 **training-free** 특성은 일반화 가능성에서 핵심적 장점을 제공한다:

1. **모델 불가지론적(Model-Agnostic) 적용**: cross-attention 메커니즘을 사용하는 모든 diffusion 모델에 적용 가능 (예: SD v1.x, v2.x, SDXL 등)
2. **도메인 독립성**: 특정 데이터셋에 의존하지 않으므로, 다양한 도메인(사진, 예술, 일러스트 등)에 걸쳐 일반화 가능
3. **플러그인 호환성**: 학습된 BoxNet과 어텐션 마스크 제어를 원본 SD와 Attend-and-Excite, GLIGEN 등 두 가지 변형에 통합하여 즉시 사용 가능한 플러그인으로 활용할 수 있다.

### 3.2 일반화의 한계와 발전 방향

텍스트-이미지 생성 모델의 급속한 발전에도 불구하고, Stable Diffusion이나 DALL-E 3 같은 최첨단 모델들은 여전히 여러 객체를 일관된 장면으로 구성하는 데 어려움을 겪으며, 잘못된 속성 바인딩, 오류 카운팅, 결함 있는 객체 관계 등의 문제가 존재한다.

일반화 성능 향상을 위한 방향:

1. **DiT(Diffusion Transformer) 아키텍처 확장**: DiT 아키텍처에서의 편집 강도 제어를 위한 training-free 방법이 필요하며, 기존 어텐션 조작 방법들이 Key 공간에만 집중하고 Value 공간은 미활용하고 있어, DiT의 multi-modal attention layers에서 Key와 Value 프로젝션 모두 layer-specific bias 벡터 주위에 밀집하는 bias-delta 구조를 보인다.

2. **LLM 기반 레이아웃 가이던스와의 결합**: 최근 연구들은 LLM이 생성한 레이아웃을 사용하지만, 비용이 높은 추론과 엄격한 제약 조건을 부과하여 자연스럽지 않은 출력으로 이어진다.

3. **Fine-tuning 기반 접근과의 결합**: 모델의 능력을 최대한 유지하면서 구성적 생성 능력을 효과적으로 향상시키기 위해, 모델의 관련 부분만 fine-tuning하는 것이 필수적이며, 이미지 배치가 어텐션 맵과 크게 상관되기 때문에 어텐션 모듈의 query와 key 프로젝션 레이어만을 fine-tuning하는 전략이 효과적이다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

이 연구는 보다 신뢰성 있고 제어 가능한 생성 AI 시스템의 기반을 확립하며, 정밀한 의미적 제어가 필요한 실용적 응용에 한 걸음 더 가까이 다가갔다.

**핵심 영향:**

1. **Cross-attention 분석 패러다임 확립**: Prompt-to-Prompt에 이어, cross-attention 맵의 조작을 통한 생성 제어라는 연구 패러다임을 확립
2. **Training-Free Guidance 연구의 촉발**: 추론 시 개입(test-time intervention)이라는 연구 방향의 주요 기여
3. **벤치마크 기여**: Attend-and-Excite에서 구축한 데이터셋은 DrawBench, CC-500과 함께 객체-속성 바인딩 평가에 가장 일반적으로 사용되는 데이터셋 중 하나가 되었다.

### 4.2 향후 연구 시 고려할 점

1. **복잡한 구성적 관계 모델링**: 단순 객체 존재를 넘어, 공간적 관계, 수량, 부정(negation) 등 더 복잡한 의미적 관계를 다루는 방법 필요
2. **효율성 개선**: 반복적 latent 최적화의 계산 비용을 줄이기 위한 방법론 연구
3. **새로운 아키텍처 적응**: DiT 기반 모델(SD3, FLUX 등)에서의 적용 방안 탐구
4. **종합적 벤치마크 사용**: T2I-CompBench++와 같은 확장된 벤치마크를 활용한 보다 포괄적인 평가 필요—속성 바인딩, 객체 관계, 생성 수량, 복잡한 구성 등을 포함하는 8개 하위 카테고리로 분류된 8,000개의 구성적 텍스트 프롬프트가 제공된다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 주요 차이점 | Training-Free |
|------|------|-----------|------------|:---:|
| **DDPM** (Ho et al.) | 2020 | 디노이징 확산 확률 모델 | 확산 모델의 기반 아키텍처 | ✗ |
| **Stable Diffusion** (Rombach et al.) | 2022 | 잠재 확산 모델 | Attend-and-Excite가 개선하고자 하는 기반 모델 | ✗ |
| **Prompt-to-Prompt** (Hertz et al.) | 2022 | Cross-attention 맵 교체를 통한 이미지 편집 | Cross-attention 맵 조작을 개척했으나 사용자 지정 편집에 초점; Attend-and-Excite는 초기 생성 시 catastrophic neglect 자동 보정에 적용 | ✓ |
| **Composable Diffusion** (Liu et al.) | 2022 | 다중 프롬프트 합성 | Composable Diffusion은 다중 주체 이미지 생성을 위해 확산 모델의 출력을 합성하지만, '주체 혼합(subject mixing)' 문제가 관찰됨 | ✓ |
| **StructureDiffusion** (Feng et al.) | 2023 | 언어 구조 기반 cross-attention 안내 | StructureDiffusion은 언어적 구조를 사용하여 이미지-텍스트 cross-attention을 안내하지만, 샘플 수준에서 의미적 문제를 해결하는 데 자주 부족 | ✓ |
| **Attend-and-Excite** (Chefer et al.) | 2023 | GSN + cross-attention 활성화 강화 | 가장 무시된 토큰의 max activation 최적화 | ✓ |
| **SynGen** (Rassin et al.) | 2023 | 언어적 바인딩 + 어텐션 맵 정렬 | A&E의 손실이 최소 어텐션 맵의 최대화를 목표로 하는 반면, SynGen의 손실은 언어적으로 관련된 단어들의 쌍별 관계에 의존하여 프롬프트의 언어 구조에 대한 확산 과정 정렬을 목표 | ✓ |
| **Bounded Attention** (Dahary et al.) | 2024 | Self/cross-attention 정보 흐름 제한 | Bounded Attention은 샘플링 과정에서 정보 흐름을 제한하여 주체 간의 유해한 누출(leakage)을 방지하고, 복잡한 다중 주체 조건화에서도 각 주체의 개별성을 촉진하는 training-free 방법 | ✓ |
| **EBAMA** (Zhang et al.) | 2024 | 객체-속성 일관성 손실 | Attend-and-Excite가 어텐션을 사용하여 latent를 업데이트하는 반면, EBAMA는 객체-속성 일관성 손실(object-attribute consistency losses)을 도입 | ✓ |
| **Attention Refocusing** (Phung et al.) | 2024 | Grounded text-to-image + 어텐션 재초점 | LLM 기반 레이아웃 + 어텐션 맵 리포커싱 | ✓ |
| **EvoGen** (Progressive Compositionality) | 2024 | Curriculum training | Curriculum training이 확산 모델에 구성성에 대한 근본적 이해를 부여하는 데 핵심적이라고 주장 | ✗ |
| **IterComp** | 2024 | 반복적 피드백 학습 | IterComp는 다중 모델로부터 구성 인식 모델 선호도를 집계하고, 반복적 피드백 학습 접근법을 사용하여 구성적 생성을 향상 | ✗ |
| **Focused Cross-Attention (FCA)** | 2024 | 구문적 제약 기반 어텐션 제어 | FCA는 입력 문장에서 발견된 구문적 제약에 의해 시각적 어텐션 맵을 제어하며, 추가 학습 없이 최첨단 확산 모델에 쉽게 통합 가능 | ✓ |

### 핵심 트렌드 분석

Attend-and-Excite 이후의 연구들은 다음과 같은 방향으로 발전하고 있다:

1. **객체 존재 → 속성 바인딩 → 관계 모델링**으로의 점진적 확장
2. 어텐션 맵은 이미지 및 비디오 확산 모델에서 레이아웃과 구성을 조종하고, 세밀한 텍스처와 스타일을 전이하며, 다중 프롬프트 정렬을 강제하고, 비디오의 시간적 일관성을 유지하는 등 컴팩트한 training-free 제어 인터페이스가 되었다.
3. **Training-free 방법의 한계 인식**: 복잡한 구성적 생성에서는 fine-tuning이나 curriculum training 기반 접근이 보완적으로 필요

---

## 참고 자료 및 출처

1. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., & Cohen-Or, D. (2023). "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models." *ACM Transactions on Graphics (TOG)*, 42(4), Article 148. (SIGGRAPH 2023) — [arXiv:2301.13826](https://arxiv.org/abs/2301.13826)
2. 공식 프로젝트 페이지: https://attendandexcite.github.io/Attend-and-Excite/
3. 공식 GitHub: https://github.com/yuval-alaluf/Attend-and-Excite
4. ACM Digital Library: https://dl.acm.org/doi/10.1145/3592116
5. alphaXiv 분석: https://www.alphaxiv.org/overview/2301.13826
6. Semantic Scholar: https://www.semanticscholar.org/paper/c3c7464acb90049c5f520b0732dc7435ba3690bd
7. Dahary, O. et al. (2024). "Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation." ECCV 2024 — Springer
8. "Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds" (2024) — arXiv:2411.18810
9. "Progressive Compositionality in Text-to-Image Generative Models" (2024) — arXiv:2410.16719
10. "Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models" (2023) — arXiv:2305.13921
11. ResearchGate: Linguistic Binding in Diffusion Models, EBAMA, Predicated Diffusion 등 관련 연구 — https://www.researchgate.net/publication/372672708
