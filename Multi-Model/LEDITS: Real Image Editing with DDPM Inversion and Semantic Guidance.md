
# LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance

> **논문 정보**
> - **제목**: LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance
> - **저자**: Linoy Tsaban, Apolinário Passos (HuggingFace)
> - **arXiv**: [2307.00522](https://arxiv.org/abs/2307.00522) (2023년 7월 2일)
> - **분류**: cs.CV

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

본 논문은 **LEDITS**를 제안합니다. 이는 Edit Friendly DDPM Inversion 기법과 Semantic Guidance(SEGA)를 결합한 경량 실제 이미지 편집 접근법으로, Semantic Guidance를 실제 이미지 편집으로 확장하면서 동시에 DDPM Inversion의 편집 역량도 활용합니다.

### 1.2 주요 기여 (Contributions)

| 기여 항목 | 내용 |
|---|---|
| **방법론적 통합** | Edit Friendly DDPM Inversion + SEGA의 경량 결합 |
| **실제 이미지 편집 확장** | SEGA를 합성 이미지에서 실제(real) 이미지로 확장 |
| **다양한 편집 지원** | 미세 편집부터 스타일/구도 대규모 변경까지 |
| **무(無)최적화** | 추가 학습이나 아키텍처 수정 없이 구동 |

이 접근법은 구성(composition)과 스타일의 변화를 포함한 미묘한 편집과 광범위한 편집 모두를 달성하며, **최적화나 아키텍처 확장 없이** 동작합니다.

---

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 문제점

텍스트 기반 모델에서의 이미지 편집은 원본 이미지의 특정 콘텐츠를 보존해야 하는 특성 때문에 어렵습니다. 텍스트 프롬프트를 조금만 수정해도 완전히 다른 결과가 생성되어 사용자의 의도에 정확히 맞는 원샷 생성(one-shot generation)을 달성하는 것이 매우 어렵습니다.

또한, 최첨단 도구를 사용하여 실제 이미지를 편집하려면, 먼저 이미지를 사전 학습된 모델의 도메인으로 **역변환(inversion)** 해야 하며, 이는 편집 품질과 레이턴시에 영향을 미치는 또 다른 요소를 추가합니다.

### 2.2 구체적 도전 과제

기존 대부분의 diffusion 기반 편집 연구들은 **DDIM(Denoising Diffusion Implicit Model) 방식**을 사용하는데, 이는 단일 노이즈 맵에서 이미지로의 결정론적 매핑(deterministic mapping)입니다.

반면 **Edit Friendly DDPM Inversion**은 DDPM 방식의 확산 생성 프로세스에 관련된 노이즈 맵을 새로운 방식으로 계산합니다. 이 노이즈 맵들은 일반 DDPM 샘플링에서 사용되는 것과 다르게 타임스텝 간 상관관계가 높고 분산이 더 큽니다. Edit Friendly DDPM Inversion은 텍스트 기반 편집 작업에서 최신 성능을 달성하며, DDIM 기반 방법과 달리 각 입력 이미지와 텍스트에 대해 다양한 결과를 생성할 수 있습니다.

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 DDPM 기본 프로세스

DDPM의 순방향(forward) 확산 과정은 점진적으로 가우시안 노이즈를 추가합니다:

$$q(x_t | x_{t-1}) = \mathcal{N}\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)$$

역방향(reverse) 생성 프로세스는 다음과 같이 $T$번의 스텝을 통해 이미지 $x_0$를 합성합니다:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 \mathbf{I}\right)$$

DDPM에서 생성적(역방향) 확산 프로세스는 $T+1$개의 노이즈 맵 $\{x_T, z_T, \ldots, z_1\}$을 활용하여 $T$번의 스텝으로 이미지 $x_0$를 합성하며, 이 노이즈 맵들이 생성된 이미지와 관련된 잠재 코드(latent code)로 간주됩니다.

### 3.2 Edit Friendly DDPM Inversion

이 방법은 주어진 이미지에 대한 **편집 친화적 노이즈 맵(edit-friendly noise maps)**을 추출하는 대안적 DDPM 잠재 노이즈 공간을 제안하며, 단순한 수단을 통해 다양한 편집 작업을 가능하게 합니다.

역변환(inversion) 핵심: 실제 이미지 $x_0$가 주어졌을 때, 다음 DDPM 생성 과정을 재현하는 노이즈 시퀀스 $\{x_T, z_T, \ldots, z_1\}$을 추출합니다:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0(x_t) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\, \epsilon_\theta(x_t, t) + \sigma_t\, z_t$$

여기서:
- $\hat{x}\_0(x_t) = \frac{x_t - \sqrt{1-\bar{\alpha}\_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$: 노이즈 예측 기반 $x_0$ 추정
- $z_t$: 타임스텝 간 상관 구조를 가지는 편집 친화적 노이즈 맵
- $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$: 누적 노이즈 스케일

일반 생성 프로세스에서는 노이즈 벡터들이 타임스텝 간 통계적으로 독립적이어서 연속 벡터 간 각도가 $[0°, 180°]$에 균등 분포하는 반면, Edit Friendly 방식에서는 노이즈 벡터가 더 높은 분산을 가지며 연속 타임스텝 간 음의 상관관계(negatively correlated)를 가집니다.

### 3.3 Semantic Guidance (SEGA)

SEGA는 분류기 없는 가이던스(classifier-free guidance)에서 도입된 원칙들을 확장하여, **모델의 잠재 공간에 이미 존재하는 개념들과만 상호작용**합니다. 이 계산은 진행 중인 확산 반복 내에서 이루어지며, 확산 프로세스를 다양한 방향으로 영향을 미치도록 설계됩니다.

SEGA의 편집 방향 계산 (Classifier-Free Guidance 기반 확장):

$$\tilde{\epsilon}_\theta(z_t, c_p, \{e_i\}) = \epsilon_\theta(z_t) + s_g\left(\epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t)\right) + \sum_{i} s_{e_i} \cdot \psi\!\left(\epsilon_\theta(z_t, e_i) - \epsilon_\theta(z_t)\right)$$

여기서:
- $\epsilon_\theta(z_t)$: 무조건 노이즈 추정
- $\epsilon_\theta(z_t, c_p)$: 텍스트 프롬프트 $c_p$ 조건 노이즈 추정
- $\epsilon_\theta(z_t, e_i)$: 편집 개념 $e_i$ 조건 노이즈 추정
- $s_g$: 전역 가이던스 스케일
- $s_{e_i}$: 개별 편집 개념 가이던스 스케일
- $\psi(\cdot)$: 편집 강도 조절 함수 (모노토닉 스케일링 성질 보장)

Classifier-free guidance에서 점수 추정은 다음과 같이 조정됩니다: $\tilde{\epsilon}\_\theta(z_t, c_p) := \epsilon_\theta(z_t) + s_g(\epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t))$ — 무조건 $\epsilon$-예측이 조건부 방향으로 푸시됩니다. SEGA는 이 원칙을 확장하여 여러 방향으로 확산 프로세스에 영향을 줍니다.

### 3.4 LEDITS 통합 메커니즘

LEDITS는 SEGA 방식의 확산 디노이징 프로세스를 간단히 수정한 직관적인 통합 방식을 제안합니다. 이 수정은 양쪽 방법의 편집 유연성을 유지하면서도 각 구성 요소의 편집 효과에 대한 완전한 제어권을 유지합니다. 먼저 입력 이미지에 DDPM Inversion을 적용하여 관련 잠재 코드를 추정합니다.

각 디노이징 스텝에서, SEGA 로직에 따라 노이즈 추정값을 계산하고, 사전 계산된 노이즈 맵을 사용하여 DDPM 방식에 따라 업데이트된 잠재값을 계산합니다.

**LEDITS 업데이트 규칙 (의사 수식):**

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0\!\left(\tilde{\epsilon}_\theta^{\text{SEGA}}(z_t, c_p, \{e_i\})\right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\tilde{\epsilon}_\theta^{\text{SEGA}} + \sigma_t\, n_t^{\text{inv}}$$

여기서 $n_t^{\text{inv}}$는 역변환 단계에서 사전 계산된 노이즈 맵.

### 3.5 두 가지 편집 워크플로우

**워크플로우 1**: DDPM Inversion만으로 역변환(target prompt="")하여 원본 이미지의 완벽한 재구성을 달성하고, SEGA 편집 개념을 추가하여 편집 수행. **워크플로우 2**: 원하는 편집 결과를 반영하는 타겟 프롬프트를 선택하는 두 가지 편집 작업을 SEGA 개념과 함께 동시에 수행.

---

## 4. 모델 구조

### 4.1 전체 파이프라인

```
[원본 이미지 x₀]
       ↓
┌──────────────────────────┐
│  Phase 1: DDPM Inversion │
│  - 편집 친화적 노이즈 맵  │
│    {xT, zT, ..., z1} 추출│
└──────────────────────────┘
       ↓
┌───────────────────────────────────────────────────────────┐
│  Phase 2: Reverse Diffusion with SEGA                     │
│  각 타임스텝 t에서:                                        │
│  1. SEGA 로직으로 노이즈 추정: ε̃θ(zt, cp, {eᵢ})          │
│  2. DDPM 방식으로 잠재값 업데이트 (사전계산 노이즈맵 사용) │
└───────────────────────────────────────────────────────────┘
       ↓
[편집된 이미지]
```

**상단(Top)**: 입력 이미지의 역변환 — DDPM inversion을 원본 이미지에 먼저 적용하여 역변환된 잠재값과 해당 노이즈 맵을 얻습니다. **하단(Bottom)**: 역변환된 잠재값을 사용하여 semantic guidance와 함께 역방향 확산 프로세스를 진행합니다.

이 아키텍처는 DDPM Inversion을 사용하여 입력 이미지의 잠재 코드를 추정한 후, DDPM Inversion 방식과 사전 계산된 노이즈 벡터를 사용하여 디노이징 루프를 수행합니다. 잠재값은 semantic guidance와 함께 확산 모델의 노이즈 추정을 기반으로 업데이트됩니다.

### 4.2 핵심 구성 요소

| 구성 요소 | 역할 | 특징 |
|---|---|---|
| **Edit Friendly DDPM Inversion** | 실제 이미지 → 편집 가능한 노이즈 공간 | 타임스텝 간 상관 노이즈, 고분산 |
| **SEGA** | 다방향 의미적 편집 가이던스 | 추가 학습 없음, 모노토닉 스케일링 |
| **Stable Diffusion U-Net** | 노이즈 추정 backbone | 기존 사전학습 모델 그대로 사용 |

SEGA는 추가적인 학습, 아키텍처 확장, 외부 가이던스가 전혀 필요 없으며, 단일 순방향 패스(forward pass) 내에서 계산됩니다.

---

## 5. 성능 향상 및 한계

### 5.1 성능 향상

LEDITS는 광범위한 편집 역량을 포괄하고 사용자가 편집 작업의 효과에 대해 갖는 세밀한 제어 수준을 확장합니다. LEDITS는 SEGA의 강건성(robustness), 단조성(monotonicity) 등 각 방법의 개별 강점을 일반적으로 유지합니다. 질적 실험 결과는 두 기법이 독립적인 편집 작업에 동시에 사용될 때 원본 이미지의 의미론적 충실도를 해치지 않으면서 더 다양한 출력을 산출함을 나타냅니다.

SEGA 의미 벡터는 LEDITS에서 사용될 때도 단조적 스케일링 특성을 유지합니다 — SEGA 개념의 강도를 증가/감소시킴에 따른 점진적 효과를 관찰할 수 있습니다.

실험 결과는 SEGA 가이던스 벡터와 DDPM Inversion의 결합 접근법이 편집 작업의 다양성, 다재다능함, 제어력을 향상시킴을 보여줍니다. 질적 실험은 최신 방법들과 경쟁력 있는 결과를 보여주며, 이미지 편집의 충실도와 창의성 사이의 균형을 제공합니다.

### 5.2 한계점

이 논문은 DDPM Inversion과 SEGA 기법의 결합 및 통합을 **탐색적으로(casually)** 탐구하는 것을 목표로 합니다. LEDITS는 의미론적으로 안내된 확산 생성 프로세스에 대한 간단한 수정으로 구성됩니다. 이 수정은 SEGA 기법을 실제 이미지로 확장하고 두 방법의 편집 역량을 동시에 활용하는 결합 편집 접근법을 도입하며, 최신 방법들과 **경쟁력 있는 정성적 결과**를 보입니다.

**주요 한계 요약:**

| 한계 | 설명 |
|---|---|
| **탐색적 보고서** | 논문 자체가 "exploratory report"로 엄밀한 정량적 평가 부족 |
| **DDPM 스텝 수** | DDIM 대비 추론 속도가 느릴 수 있음 |
| **Prompt Engineering 의존** | 적절한 편집 개념 텍스트 기술이 필요 |
| **기반 모델 한계 종속** | 기반 DM이 생성 불가한 포즈/구도는 편집 불가 |

---

## 6. 모델의 일반화 성능 향상 가능성 🔑

### 6.1 아키텍처 불가지론(Architecture-Agnostic) 특성

이 수정은 SEGA 기법을 실제 이미지로 확장하고, 두 방법의 편집 역량을 동시에 활용하는 결합 편집 접근법을 도입하며, 최신 방법들과 경쟁력 있는 정성적 결과를 보입니다.

후속 연구인 LEDITS++에서 이 특성이 더욱 명확히 드러납니다:
LEDITS++의 새로운 역변환 접근법은 튜닝이나 최적화가 필요 없으며, 적은 확산 스텝만으로 고충실도 결과를 생성합니다. 또한 이 방법론은 다중 동시 편집을 지원하며 **아키텍처에 구애받지 않습니다(architecture-agnostic)**.

동일한 이미지 편집 방법(LEdits++)을 적용할 때 더 강력한 SD-XL 변형이 약한 SD1.5 모델보다 뛰어납니다. 이는 아키텍처에 구애받지 않는 LEdits++가 **점점 더 강력해지는 DM으로부터 혜택을 받을 것**임을 의미합니다.

### 6.2 다양한 편집 유형 일반화

LEDITS는 원하는 편집 효과를 조정하는 데 있어 추가적인 유연성을 제공하며, 원본 이미지의 의미론적 보존과 창의적 편집 사이의 균형을 조율합니다. 여러 편집 작업이 대상 프롬프트(하나 이상의 편집 작업 반영)와 SEGA 개념(각각 편집 작업 반영)과 함께 독립적으로 동시에 적용될 수 있습니다.

### 6.3 Edit-Friendly 노이즈 공간의 범용성

편집 친화적 역변환은 제어된 편집에 적합한 노이즈 맵을 생성하고, 텍스트 조건 모델에서 구조와 의미론을 분리하며, 다양한 출력 조작을 지원합니다.

DDPM Inversion은 기존 확산 기반 편집 방법 내에서도 활용되어 품질과 다양성을 향상시킬 수 있습니다. 실제 이미지의 텍스트 기반 편집에 단독으로 또는 다른 편집 방법과 결합하여 사용될 수 있습니다. 방법의 확률론적 특성으로 인해 **다양한 출력**을 생성할 수 있으며, 이는 DDIM Inversion에 의존하는 방법들에서 자연스럽게 사용할 수 없는 기능입니다.

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 주요 방법론 비교표

| 방법 | 연도 | 역변환 방식 | 실사 이미지 | 최적화 필요 | 다중 편집 | 주요 특징 |
|---|---|---|---|---|---|---|
| **SDEdit** | 2021 | 부분 노이징 | ✅ | ❌ | 제한적 | 노이즈 추가 후 재생성 |
| **Prompt-to-Prompt** | 2022 | DDIM Inversion | ⚠️ (어려움) | ❌ | ✅ | Cross-Attention 맵 조작 |
| **DiffEdit** | 2022 | DDIM Inversion | ✅ | ❌ | 제한적 | 자동 마스크 생성 |
| **InstructPix2Pix** | 2023 | 미사용 | ✅ | ✅ (파인튜닝) | 제한적 | 자연어 지시 따르기 |
| **Null-Text Inversion** | 2023 | DDIM + 최적화 | ✅ | ✅ | ✅ | 피벗 최적화 기반 |
| **SEGA** | 2023 | DDIM | ⚠️ (한계) | ❌ | ✅ | 의미론적 방향 제어 |
| **LEDITS** | 2023 | **Edit Friendly DDPM** | ✅ | ❌ | ✅ | DDPM Inv + SEGA 결합 |
| **LEDITS++** | 2023/2024 | DPM-Solver++ | ✅ | ❌ | ✅ | 암묵적 마스킹, 속도 개선 |

### 7.2 방법론별 상세 비교

**Prompt-to-Prompt (Hertz et al., 2022):**
Prompt-to-Prompt는 모델의 크로스-어텐션 레이어의 의미론적 정보를 활용하여 픽셀과 텍스트 프롬프트 토큰을 연결합니다. 크로스-어텐션 맵 조작은 다양한 변화를 가능하게 하지만, SEGA는 토큰 기반 컨디셔닝을 필요로 하지 않으며 여러 의미론적 변화의 조합이 가능합니다.

**InstructPix2Pix (Brooks et al., 2023):**
InstructPix2Pix는 GPT-3와 Prompt-to-Prompt를 Stable Diffusion에 결합하여 얻은 합성 데이터를 기반으로 이미지 편집 지시를 따르는 네트워크를 학습합니다.

**LEDITS++ (Brack et al., 2023/2024) — LEDITS의 직접 후속:**
LEDITS++는 효율적이고 다재다능하며 정밀한 텍스트 이미지 조작 기법입니다. LEDITS++의 새로운 역변환 접근법은 튜닝이나 최적화가 필요 없으며 적은 확산 스텝으로 고충실도 결과를 생성합니다. 이 방법론은 다중 동시 편집을 지원하며 아키텍처에 구애받지 않습니다. 또한 관련 이미지 영역으로 변화를 제한하는 **새로운 암묵적 마스킹 기법**을 사용합니다.

LEDITS++는 DDPM 샘플링 방식에서 이전에 제안된 편집 친화적 노이즈 공간의 특성을, 훨씬 더 빠른 **다단계 확률적 미분 방정식(SDE) 솔버**인 DPM-Solver++로 도출합니다.

벤치마크 평가에서:
비교 측도로 CLIP과 LPIPS 점수를 사용합니다. CLIP은 편집 지시와 편집된 이미지 간의 텍스트-이미지 유사도를 측정하고, LPIPS는 실제 이미지와 편집된 이미지 간의 유사도를 측정합니다. 이로써 편집의 다재다능함(CLIP)과 정밀도(LPIPS) 간의 트레이드오프를 평가합니다.

LEdits++는 모든 경쟁 방법들을 명확히 능가합니다.

---

## 8. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 8.1 향후 연구에 미치는 영향

#### 🔬 이론적 영향

1. **편집 친화적 잠재 공간 연구의 촉진**: 이 방법은 편집 친화적 노이즈 맵 추출을 위한 대안적 DDPM 잠재 노이즈 공간을 제안하며, 단순한 수단을 통해 다양한 편집 작업을 가능하게 합니다. 이러한 "편집 친화적 공간" 개념은 이후 다양한 모달리티(음성, 비디오, 3D)로 확장 연구되고 있습니다.

2. **무최적화 편집의 패러다임 정립**: 기존 이미지-이미지 방법들이 비효율적이고 부정확하며 다재다능성이 제한적인 문제 — 시간 소모적 파인튜닝, 입력 이미지에서 불필요하게 강한 이탈, 다중 동시 편집 미지원 등 — 를 해결하는 방향으로 연구가 집중되고 있습니다.

3. **음향 도메인 확장**: DDPM Inversion은 오디오 도메인으로도 일반화되었으며, 제로샷 텍스트 기반 오디오 편집(ZETA) 및 비지도 주성분 조작(ZEUS)을 가능하게 하여 악기 참여, 리듬, 즉흥 연주에 대한 세밀한 제어를 지원합니다.

#### 🚀 실용적 영향

후속 편집 벤치마크 연구들은 InstructPix2Pix, DDPM-Inversion, **LEDITS++**, ProxEdit 등을 표준 비교 방법으로 포함하여 평가합니다.

### 8.2 향후 연구 시 고려할 점

#### ① 기반 모델 성능 종속 문제
기반 DM이 특정 포즈를 생성하지 못하는 경우(예: 앉아있는 기린) 편집에 실패하는 사례가 있습니다. 이 효과는 편집 성공률이 DM에 따라 강하게 달라진다는 점에서도 명확히 나타납니다. → **향후 연구**: 더 강력한 기반 모델(SDXL, Flux 등)과의 통합 및 검증 필요

#### ② 정량적 평가 체계 강화
현재 논문이 탐색적(exploratory) 성격을 가진 만큼, 향후 연구에서는:
새로운 TEdBench++ 벤치마크와 같은 종합적 평가 기준이 필요합니다.

#### ③ 암묵적 마스킹 및 지역화
암묵적 마스킹은 관련 부분으로의 변화를 제한하고 원본 이미지 구성과 강한 일관성을 달성하지만, 마스크 영역 내의 객체와 그 정체성은 다양한 요인에 따라 변경될 수 있습니다. → 더 정밀한 지역화(localization) 기법 연구 필요

#### ④ 다중 편집의 간섭 최소화
시스템의 일부 한계도 있습니다. 상당한 구조적 변화를 수반하는 복잡한 편집은 때때로 아티팩트를 생성할 수 있습니다. 또한 이 방법은 최적의 결과를 위해 신중한 프롬프트 엔지니어링이 필요합니다.

#### ⑤ 비디오 및 3D 편집으로의 확장
DDPM Inversion의 범위는 기하학적·광도측정 편집, 속성 전이, 구성 수정, 프롬프트 조건 변환 등을 가능하게 하는 다양한 도메인에 걸쳐 있습니다. 이는 비디오/NeRF 등으로의 확장 연구를 촉발하고 있습니다.

---

## 9. 참고 문헌 및 출처

| # | 제목 | 저자/출처 | 연도 |
|---|---|---|---|
| 1 | **LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance** | Linoy Tsaban, Apolinário Passos — arXiv:2307.00522 | 2023 |
| 2 | **LEDITS++ : Limitless Image Editing using Text-to-Image Models** | Brack et al. — arXiv:2311.16711 | 2023/2024 |
| 3 | **An Edit Friendly DDPM Noise Space: Inversion and Manipulations** | Huberman-Spiegelglas et al. — CVPR 2024 | 2023 |
| 4 | **SEGA: Instructing Text-to-Image Models using Semantic Guidance** | Brack et al. — NeurIPS 2023 | 2023 |
| 5 | **Denoising Diffusion Probabilistic Models (DDPM)** | Ho, Jain, Abbeel — NeurIPS 2020 | 2020 |
| 6 | **Prompt-to-Prompt Image Editing with Cross Attention Control** | Hertz et al. — ICLR 2023 | 2022 |
| 7 | **InstructPix2Pix: Learning to Follow Image Editing Instructions** | Brooks et al. — CVPR 2023 | 2023 |
| 8 | **DiffEdit: Diffusion-based Semantic Image Editing with Mask Guidance** | Couairon et al. | 2022 |
| 9 | **HuggingFace Research Introduces LEDITS** | MarkTechPost | 2023 |
| 10 | **LEDITS Project Page** | editing-images-project.static.hf.space | 2023 |
| 11 | **Diffusion Model** | Wikipedia | 참고 |
| 12 | **DDPM Inversion: Techniques & Applications** | EmergentMind | 2025 |
| 13 | **Image Editing with Diffusion Models: A Survey** | arXiv:2504.13226 | 2025 |
| 14 | **Replicate — LEDITS 구현** | replicate.com/cjwbw/ledits | 2023 |

---

> ⚠️ **주의**: 본 논문이 "exploratory report"로 명시되어 있어 일부 정량적 수식(특히 LEDITS 업데이트 규칙의 세부 표기)은 논문 내 상세 기술이 제한적입니다. SEGA와 Edit Friendly DDPM Inversion의 원 논문에서 보완하여 재구성하였음을 밝힙니다. 확인이 불확실한 세부 수식은 의도적으로 기재를 생략하였습니다.
