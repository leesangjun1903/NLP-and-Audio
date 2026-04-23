
# Personalized Residuals for Concept-Driven Text-to-Image Generation

> **논문 정보**
> - **저자**: Cusuh Ham, Matthew Fisher, James Hays, Nicholas Kolkin, Yuchen Liu, Richard Zhang, Tobias Hinz
> - **학회**: CVPR 2024 (pp. 8186–8195)
> - **arXiv**: [2405.12978](https://arxiv.org/abs/2405.12978)
> - **소속**: Adobe Research

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 **Personalized Residuals(개인화 잔차)**와 **Localized Attention-Guided (LAG) Sampling**을 제안하여, 사전 학습된 텍스트-이미지 확산 모델을 기반으로 효율적인 개념 중심(concept-driven) 이미지 생성을 달성한다. 핵심 아이디어는 사전 학습된 확산 모델의 가중치를 동결(freeze)한 채로, 모델 레이어의 일부 집합에 대한 **저랭크 잔차(low-rank residuals)**를 학습하는 것이다.

이 방법은 **단일 GPU에서 약 3분** 만에 개념의 정체성(identity)을 효과적으로 포착하며, 정규화 이미지(regularization images) 없이, 이전 모델보다 적은 파라미터로 동작하고, Localized Sampling은 이미지의 넓은 영역에서 원본 모델을 강력한 prior로 활용한다.

### 주요 기여 (Contributions) 요약

| 기여 항목 | 내용 |
|---|---|
| Personalized Residuals | LoRA 기반 저랭크 잔차 학습으로 개념 정체성 포착 |
| LAG Sampling | 크로스-어텐션 맵 기반 공간 지역화 샘플링 |
| 정규화 이미지 불필요 | Prior-preservation 없이 개념 학습 가능 |
| 파라미터 효율성 | 전체 모델의 약 0.1% 수준의 파라미터만 학습 |
| 빠른 학습 속도 | 단일 GPU에서 ~3분 내 학습 완료 |

---

## 2. 논문의 상세 분석

### 2-1. 해결하고자 하는 문제

기존의 개인화(personalization) 접근법들은 학습 속도가 느리고, 높은 연산 요구량을 가지며, 정규화 이미지를 필요로 하거나, 대상 개념을 다른 맥락으로 재구성(recontextualization)하는 데 어려움이 있었다.

특히 오픈 도메인 접근법의 주요 과제는 모델의 원래 학습에서 습득한 개념의 망각(forgetting)을 완화하기 위한 정규화(regularization)의 필요성과, 각 개념에 대해 새로운 파라미터 집합을 파인튜닝하는 데 따른 계산 오버헤드였다.

또한 많은 개인화 접근법들은 대상 개념에 과적합(overfitting)하여 특정 배경을 렌더링하거나 새로운 객체를 추가하는 데 어려움을 겪는다. 이러한 시나리오를 위해 저자들은 새로운 **Localized Attention-Guided (LAG) 샘플링** 방식을 제안한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) Personalized Residuals (저랭크 잔차)

대부분의 접근법이 크로스-어텐션 레이어의 Key/Value 가중치를 파인튜닝하는 데 집중하는 것과 달리, 이 논문은 각 크로스-어텐션 레이어 이후의 **출력 프로젝션 Conv 레이어 가중치에 대한 저랭크 잔차(low-rank residual)를 예측**한다. 이를 통해 기존 방법보다 훨씬 적은 파라미터(기반 모델의 약 0.1%)를 파인튜닝할 수 있다.

LoRA (Low-Rank Adaptation)의 핵심 수식을 기반으로, 가중치 행렬 $W_0 \in \mathbb{R}^{d \times k}$에 대해 업데이트는 다음과 같이 표현된다:

$$W = W_0 + \Delta W = W_0 + BA$$

여기서:
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ (단, $r \ll \min(d, k)$ )
- $\Delta W = BA$가 학습되는 저랭크 잔차

각 트랜스포머 블록 $i$에 대해, 출력 프로젝션 컨볼루션 레이어의 가중치 행렬 $W_i \in \mathbb{R}^{m_i \times m_i \times 1}$에 대한 랭크 $r_i$를 계산한다.

각 블록별 잔차는 다음과 같이 적용된다:

$$W_i^{\text{pers}} = W_i + \Delta W_i = W_i + B_i A_i$$

여기서 $B_i \in \mathbb{R}^{m_i \times r_i}$, $A_i \in \mathbb{R}^{r_i \times m_i}$는 학습 가능한 행렬이며, $W_i$는 동결(frozen)된 원본 가중치이다.

학습 목표(Loss)는 표준 확산 모델의 노이즈 예측 손실을 사용한다:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right]$$

여기서:
- $x_0$: 참조 이미지
- $t$: 타임스텝
- $\epsilon \sim \mathcal{N}(0, I)$: 추가된 노이즈
- $c$: 텍스트 조건(고유 식별자 토큰 $V^*$ 포함)
- $\epsilon_\theta$: 파인튜닝된 노이즈 예측 모델

---

#### (B) Localized Attention-Guided (LAG) Sampling

각 트랜스포머 블록의 크로스-어텐션 레이어는 텍스트 토큰과 이미지 영역 간의 대응 관계를 학습하며, 각 크로스-어텐션 레이어는 프롬프트의 각 토큰 $y_i$에 대한 어텐션 맵 $A_{y_i}$를 계산하여, 해당 토큰이 생성 이미지의 어느 위치에 영향을 미치는지 나타낸다. 개념을 지정하는 고유 식별자 및 매크로 클래스 토큰(예: "V*"와 "dog")의 집합 $\mathcal{C}$를 기준으로, 해당 어텐션 값을 합산하여 개념의 위치를 예측한다.

LAG Sampling의 작동 방식을 수식으로 나타내면:

**어텐션 맵 합산:**

$$M = \sum_{y_i \in \mathcal{C}} A_{y_i}$$

**이진 마스크 생성 (임계값 $\tau$ 적용):**

$$\hat{M}(p) = \begin{cases} 1 & \text{if } M(p) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

**개념 위치 기반 특징 합성:**

$$h^{\text{final}} = \hat{M} \odot h^{\text{pers}} + (1 - \hat{M}) \odot h^{\text{orig}}$$

여기서:
- $h^{\text{pers}}$: 개인화된 잔차(Personalized Residuals)를 적용해 생성한 특징
- $h^{\text{orig}}$: 원본 동결된 확산 모델로 생성한 특징
- $\hat{M}$: 개념 위치 예측 마스크
- $\odot$: 원소별(element-wise) 곱

이 샘플링 방식은 추가적인 학습이나 데이터를 필요로 하지 않으며, 모델 평가 횟수를 늘리지 않으므로 샘플링 시간도 증가하지 않는다.

---

### 2-3. 모델 구조

이 논문의 방법은 **Stable Diffusion v1.4**를 기반 모델로 사용한다.

전체 구조는 다음과 같이 요약할 수 있다:

```
[Reference Images (3~5장)]
        ↓
[사전 학습된 Stable Diffusion (동결)]
        ↓
[크로스-어텐션 레이어 이후 Output Projection Conv Layer]
        ↓
[저랭크 잔차 ΔW = BA (학습 가능)]
        ↓
[학습된 Personalized Residuals]
        ↓
    (추론 시)
  ┌──────────────────────────────────┐
  │ LAG Sampling                     │
  │  - Cross-Attention Map 기반       │
  │    개념 위치 마스크 추정           │
  │  - 마스크 내부: pers. residuals    │
  │  - 마스크 외부: 원본 모델         │
  └──────────────────────────────────┘
        ↓
[최종 생성 이미지]
```

LoRA 기반 접근법을 통해 개념의 정체성을 표현하는 소규모 잔차 집합을 학습하며, 학습 가능한 파라미터 수와 학습 시간을 줄이고, 도메인 정규화에 대한 의존성을 제거하면서도 편집 유연성을 유지한다.

---

### 2-4. 성능 향상

사용자 연구 및 정량적 평가 결과, 이 방법은 다른 기준 모델(baseline)과 동등하거나 더 나은 성능을 보이며, 제안된 샘플링 방식은 배경 변경과 같은 특정 재맥락화(recontextualization) 시나리오의 어려움을 해결할 수 있다.

| 비교 항목 | 기존 방법 (DreamBooth 등) | 본 논문 방법 |
|---|---|---|
| 학습 시간 | 수십 분 ~ 수 시간 | ~3분 (단일 GPU) |
| 파라미터 수 | 전체 모델 또는 대부분 | 전체의 ~0.1% |
| 정규화 이미지 | 필요 | 불필요 |
| 배경 생성 품질 | 과적합으로 저하 | LAG로 원본 prior 활용 |
| 편집 유연성 | 제한적 | 높음 |

---

### 2-5. 한계

LAG 샘플링은 학습된 잔차가 참조 이미지에 과적합하고, 참조 이미지의 모호성이나 모델 편향(예: 가구는 실내에서 자주 촬영됨)으로 인해 대상 개념이 배경과 효과적으로 분리되지 않은 경우에 유익하다. 개념 토큰에서의 어텐션 맵을 활용해 잔차를 지역화함으로써 배경에 영향을 미치지 않도록 한다.

구체적인 한계점:
1. **단일 개념 학습에 집중**: 논문은 주로 단일 개념(single concept)의 개인화에 초점을 맞추며, 다중 개념(multi-concept) 동시 생성은 직접적으로 다루지 않는다.
2. **Stable Diffusion v1.4 기반**: 더 최신의 대규모 확산 모델(SDXL, DiT 기반 등)로의 직접 확장성이 명시적으로 검증되지 않았다.
3. **어텐션 맵의 정확도 의존성**: LAG Sampling은 크로스-어텐션 맵의 정확도에 크게 의존하므로, 어텐션이 부정확하게 예측될 경우 성능이 저하될 수 있다.
4. **개념 모호성 문제**: 참조 이미지가 모호하거나 대상 개념과 배경이 명확히 분리되지 않으면 과적합이 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 가능하게 하는 핵심 설계

잔차 기반 접근법은 개념이 크로스-어텐션을 통해 지역화된 영역에만 학습된 잔차를 적용하고, 그 외의 모든 영역에는 원본 확산 가중치를 적용하는 샘플링 기법을 직접적으로 가능하게 한다. Localized Sampling은 이를 통해 학습된 개념의 정체성과 기반 확산 모델의 기존 생성 prior를 결합한다.

이것이 일반화에 중요한 이유:

- 원본 모델의 **방대한 학습 prior가 배경 및 비개념 영역에서 완전히 보존**됨
- 개념이 완전히 새로운 장면, 배경, 스타일과 결합될 때 **분포 외(out-of-distribution) 시나리오에서도 강건**

### 3-2. 편집 유연성 (Editability)

LoRA 기반 접근법을 통해 편집 유연성을 유지하면서도, 학습 가능한 파라미터 수와 학습 시간을 줄이고, 도메인 정규화에 대한 의존성을 제거한다.

이는 다음을 의미한다:
- **텍스트 프롬프트 기반 편집**: "a photo of V* dog on the beach at sunset"처럼 다양한 맥락에 개념을 적용
- **스타일 전이**: 개념 정체성은 유지하면서 새로운 예술적 스타일 적용 가능

### 3-3. 도메인 제약 없는 오픈 도메인 일반화

임의의 개념을 개인화하기 위해 모델의 파라미터나 입력을 직접 파인튜닝하는 방식은 모든 종류의 개념에 적용 가능하지만, 파인튜닝은 개념별로 수행되어야 하며 각각에 대해 다른 파라미터를 저장해야 한다.

다른 접근법들은 특정 도메인(예: 얼굴)에 특화된 인코더를 학습시키고 확산 모델을 해당 도메인 내의 특정 개념을 재구성하는 데 사용한다. 이 방법의 장점은 모든 새 개념에 대한 재학습이 필요 없다는 것이지만, **단일 도메인에 제한**되고 인코더 학습을 위한 대규모 데이터셋이 필요하다.

본 논문의 방법은 **도메인 제약 없이(오픈 도메인)** 임의의 개념에 적용 가능하면서도, 도메인-인코더 방식의 단일 도메인 제한을 피한다. 다만 개념별 파인튜닝이 여전히 필요하다는 점에서 완전한 인코더 방식의 즉시성(instant inference)은 갖지 않는다.

### 3-4. 일반화 성능 향상을 위한 추가 가능성

다음 방향에서 일반화 성능이 더욱 향상될 수 있다:

1. **멀티스케일 어텐션 통합**: 다양한 해상도의 어텐션 맵을 통합하면 공간적 지역화 정확도 향상 가능
2. **동적 랭크 조정**: 개념의 복잡도에 따라 $r$을 동적으로 조절하여 표현력 최적화
3. **다중 개념 확장**: 여러 개념의 잔차를 공간적으로 분리하여 동시 적용

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 주요 관련 연구 비교표

| 방법 | 발표 | 학습 방식 | 파라미터 효율 | 정규화 이미지 | 학습 시간 | 특징 |
|---|---|---|---|---|---|---|
| **Textual Inversion** (Gal et al.) | 2022 | 텍스트 임베딩 최적화 | 매우 높음 | 불필요 | 수십 분 | 텍스트 토큰 학습만 수행 |
| **DreamBooth** (Ruiz et al.) | 2022/CVPR23 | 전체 모델 파인튜닝 | 낮음 | 필요 | 수십 분 | 고품질, 과적합 위험 |
| **LoRA** (Hu et al.) | 2021/ICLR22 | 저랭크 행렬 학습 | 높음 | 불필요 | 빠름 | LLM 파인튜닝에서 확장 |
| **Custom Diffusion** (Kumari et al.) | CVPR 2023 | 크로스-어텐션 K/V 학습 | 중간 | 필요 | 중간 | 다중 개념 지원 |
| **HyperDreamBooth** (Ruiz et al.) | 2023 | 하이퍼네트워크 | 높음 | 불필요 | 매우 빠름 | 얼굴에 특화 |
| **Personalized Residuals** (Ham et al.) | **CVPR 2024** | 저랭크 잔차 + LAG | **매우 높음** | **불필요** | **~3분** | 오픈 도메인, LAG 샘플링 |
| **ConceptPrism** (2026) | arXiv 2026 | 잔차 토큰 최적화 | 높음 | 불필요 | - | 개념 분리(disentanglement) |

### 4-2. 기존 방법의 문제점 vs. 본 논문의 접근

**Textual Inversion의 문제:**
Textual Inversion은 학습된 개념에 주로 집중하는 이미지를 생성하는 경향이 있어, 프롬프트의 다른 요소들을 종종 무시한다.

**DreamBooth의 문제:**
반면 DreamBooth는 학습된 개념을 간과하며, 다른 프롬프트 토큰에 더 많이 영향받는 이미지를 생성하는 경향이 있다. 이러한 문제들은 새로운 개념에 대한 임베딩 정렬의 잘못된 학습에서 비롯된다.

**ConceptPrism (후속 연구, 2026):**
개인화된 텍스트-이미지 생성은 참조 이미지의 관련 없는 잔차 정보가 포착되어 개념 충실도(concept fidelity)와 텍스트 정렬(text alignment) 간의 트레이드오프를 발생시키는 개념 얽힘(concept entanglement) 문제를 겪는다. 최근의 분리(disentanglement) 접근법들은 언어적 단서나 분할 마스크 같은 수동 가이드를 활용하여 해결하려 하지만, 이는 적용 가능성을 제한하고 대상 개념을 완전히 표현하지 못한다.

**LoRAShop (후속 연구, 2025):**
LoRAShop은 멀티-피사체 생성 및 편집을 위해 다중 LoRA 어댑터를 활용하는 새로운 훈련-불필요(training-free) 파이프라인을 제안한다. Multi-Subject Residual Blending(MSRB)으로 구성되어, 각 피사체가 나타날 공간 영역을 강조하는 피사체 prior 추출과, 서로 다른 LoRA 어댑터의 출력을 선택적으로 병합하는 잔차 특징 블렌딩 방식의 두 단계를 포함한다. 이는 추가 훈련 없이 공간적으로 구분된 특징을 결합하여 일관적이고 분리된 다중 피사체 생성 및 편집을 가능하게 한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려사항

### 5-1. 연구에 미치는 영향

#### (1) 파라미터-효율적 개인화의 새로운 표준 제시
이 논문은 개념 중심 합성을 위한 **Personalized Residuals** 방법을 소개하며, LoRA 기반 접근법을 통해 적은 수의 잔차를 학습하여 개념의 정체성을 표현함으로써, 학습 가능한 파라미터 수와 학습 시간을 줄이고 도메인 정규화 의존성을 제거하면서도 편집 유연성을 유지한다는 점을 보여준다.

이는 향후 연구에서 **정규화 없는 저비용 개인화**가 가능함을 실험적으로 증명하여, 파라미터-효율적 파인튜닝(PEFT)과 생성 모델의 결합 연구를 촉진한다.

#### (2) 어텐션 기반 공간 제어의 중요성 강조
크로스-어텐션 레이어의 어텐션 맵을 사용하여 각 타임스텝마다 생성 이미지 내 개념의 위치를 예측하고, 개인화된 잔차로 생성한 특징은 예측된 영역에만 적용하여 나머지 이미지(예: 배경 및 다른 객체)는 원본 모델에 의해 생성되도록 함으로써, 과적합으로 인한 특정 배경이나 관련 없는 객체 생성 능력의 손실을 방지할 수 있다.

이는 어텐션 맵을 단순 시각화 도구가 아닌 **추론 시 제어 신호**로 활용하는 패러다임을 제시한다.

#### (3) 다음 세대 연구로의 연결고리
- **다중 개념 동시 생성** 연구(MultiBooth, LoRAShop 등)의 기초가 됨
- **개념 분리(disentanglement)** 연구(ConceptPrism 등)의 동기 제공
- **비디오 개인화** 및 **3D 생성** 분야로의 확장 가능성

---

### 5-2. 앞으로 연구 시 고려할 점

#### ① 최신 확산 모델로의 확장
이 논문은 Stable Diffusion v1.4를 기반으로 하지만, 최신 SDXL, Stable Diffusion 3, FLUX, DiT(Diffusion Transformer) 기반 아키텍처로의 확장 연구가 필요하다. 특히 DiT 기반 모델에서는 자기-어텐션(self-attention)과 크로스-어텐션의 역할이 다르기 때문에 LAG Sampling의 설계 변경이 요구될 수 있다.

#### ② 다중 개념 동시 개인화
역방향 확산 과정에서 피사체 prior가 나타내는 토큰 위치에서만 잔차 특징을 덮어쓰고, 다른 모든 토큰은 변경하지 않음으로써, 이 연산이 지역적이고 선형적이므로 전체 노이즈 제거 경로와 전체 장면 레이아웃이 유지된다. 이와 같은 최신 연구에서 제시하는 **공간 분리 기반 다중 개념 합성**은 향후 핵심 연구 방향이다.

#### ③ 개념 충실도와 텍스트 정렬의 균형 (Fidelity-Editability Trade-off)
개인화 생성의 성공은 두 축으로 측정된다: 사용자 개념의 시각적 본질을 보존하는 **개념 충실도(concept fidelity)**와, 생성된 이미지가 주어진 텍스트 프롬프트를 충실히 따르도록 하는 **텍스트 정렬(text alignment)**이다. 이 두 요소의 균형을 자동으로 최적화하는 메커니즘 연구가 필요하다.

#### ④ 어텐션 맵의 신뢰성 향상
LAG Sampling의 성능은 크로스-어텐션 맵의 정확도에 의존한다. 어텐션 맵이 부정확하거나 개념이 이미지 전체에 분산된 경우의 강건성(robustness) 확보가 중요하다. 세그멘테이션 모델(예: SAM)과의 결합이 유망한 방향일 수 있다.

#### ⑤ 윤리적 고려사항
개인화 기술이 발전할수록 **딥페이크**, **저작권 침해**, **개인 정보 유출** 등의 위험이 증가한다. 향후 연구에서는 기술 개발과 함께 워터마킹, 생성 이미지 탐지 기술, 사용 제한 메커니즘 등의 안전 장치를 함께 고려해야 한다.

#### ⑥ 동적 랭크 선택 자동화
현재는 각 레이어의 랭크 $r_i$를 수동으로 설정하거나 휴리스틱으로 결정한다. **SVD 기반 자동 랭크 결정** 또는 **NAS(Neural Architecture Search)** 기법을 통한 최적 랭크 자동 탐색 연구가 필요하다.

---

## 📚 참고문헌 및 출처

1. **Ham, C., Fisher, M., Hays, J., Kolkin, N., Liu, Y., Zhang, R., & Hinz, T. (2024)**. *Personalized Residuals for Concept-Driven Text-to-Image Generation*. CVPR 2024, pp. 8186-8195.
   - arXiv: https://arxiv.org/abs/2405.12978
   - CVPR Open Access: https://openaccess.thecvf.com/content/CVPR2024/html/Ham_Personalized_Residuals_for_Concept-Driven_Text-to-Image_Generation_CVPR_2024_paper.html
   - Adobe Research: https://research.adobe.com/publication/personalized-residuals-for-concept-driven-text-to-image-generation/
   - HuggingFace Paper Page: https://huggingface.co/papers/2405.12978
   - OpenReview: https://openreview.net/forum?id=mHQEyXaULY
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10658090/

2. **Hu, E., et al. (2021)**. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

3. **Gal, R., et al. (2022)**. *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion*. ICLR 2023.

4. **Ruiz, N., et al. (2023)**. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. CVPR 2023.

5. **Kumari, N., et al. (2023)**. *Multi-Concept Customization of Text-to-Image Diffusion*. CVPR 2023.

6. **ConceptPrism (2026)**. *Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization*. arXiv:2602.19575. https://arxiv.org/abs/2602.19575

7. **LoRAShop (2025)**. *Training-Free Multi-Concept Image Generation and Editing with Rectified Flow Transformers*. arXiv:2505.23758. https://arxiv.org/html/2505.23758v1

8. **AttnDreamBooth (2024)**. *Towards Text-Aligned Personalized Text-to-Image Generation*. NeurIPS 2024. https://arxiv.org/html/2406.05000v1

9. **Awesome Personalized Image Generation (GitHub)**. https://github.com/csyxwei/Awesome-Personalized-Image-Generation

10. **Awesome Text-to-Image Studies (GitHub)**. https://github.com/AlonzoLeeeooo/awesome-text-to-image-studies
