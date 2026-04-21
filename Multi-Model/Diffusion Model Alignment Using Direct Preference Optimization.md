# Diffusion Model Alignment Using Direct Preference Optimization

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **텍스트-이미지 확산 모델(Diffusion Model)을 인간 선호도(Human Preference)에 직접 정렬(Align)** 시키는 새로운 방법론인 **Diffusion-DPO**를 제안합니다. LLM 분야에서 성공적으로 활용되던 **Direct Preference Optimization(DPO)**을 확산 모델에 맞게 재정식화(re-formulate)하여, 별도의 보상 모델(Reward Model) 학습이나 강화학습(RL) 없이 인간 비교 데이터로 직접 모델을 파인튜닝합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **이론적 기여** | 확산 모델의 가능도(likelihood) 개념을 ELBO를 통해 DPO 프레임워크에 통합 |
| **방법론적 기여** | 기존 RL 기반 방법 대비 안정적이고 효율적인 선호도 학습 알고리즘 제안 |
| **실증적 기여** | SDXL-1.0 파인튜닝으로 SOTA 달성 (인간 평가 기준 69% 선호율) |
| **개방어휘(Open-vocabulary) 일반화** | 제한된 프롬프트 집합에 국한되지 않고 다양한 프롬프트에서 성능 향상 |
| **AI 피드백 활용** | 인간 레이블 없이 AI 피드백(PickScore, HPSv2, CLIP 등)으로도 유효한 정렬 가능성 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 확산 모델 정렬 방법들은 다음과 같은 한계를 가집니다:

**기존 방법의 한계:**

- **RL 기반 방법 (DDPO, DPOK):** 제한된 어휘(< 1,000 프롬프트)에서만 효과적이며 어휘가 확장될수록 성능 저하
- **보상 극대화 학습 (DRaFT, AlignProp):** 분포 보장(distributional guarantees) 없음, 과도한 학습 시 모드 붕괴(mode collapse) 발생, CLIP 기반 텍스트-이미지 정렬 개선 불가
- **고품질 데이터셋 파인튜닝 (Emu 등):** 비용이 크고 다양한 피드백 유형에 일반화 불가
- **추론 시간 최적화 (DOODL):** 추론 비용이 10배 이상 증가

Diffusion-DPO는 이 세 가지 문제(개방 어휘 일반화 / 추론 비용 동일 / 분포 발산 제어)를 **동시에** 해결하는 유일한 방법으로 제시됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 배경: 확산 모델의 기본 학습 목표

확산 모델은 다음의 ELBO(Evidence Lower Bound)를 최소화하여 학습됩니다:

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t, \mathbf{x}_t} \left[ \omega(\lambda_t) \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|_2^2 \right]$$

여기서 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $t \sim \mathcal{U}(0, T)$, $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$이며, $\lambda_t = \alpha_t^2 / \sigma_t^2$은 신호 대 잡음비(SNR)입니다.

#### 배경: 원래 DPO 목표 함수 (LLM용)

인간 선호도는 Bradley-Terry 모델로 모델링됩니다:

$$p_{\text{BT}}(\mathbf{x}_0^w \succ \mathbf{x}_0^l | \mathbf{c}) = \sigma\left(r(\mathbf{c}, \mathbf{x}_0^w) - r(\mathbf{c}, \mathbf{x}_0^l)\right)$$

RLHF의 목표는 KL-발산 패널티 하에 기대 보상을 최대화하는 것입니다:

$$\max_{p_\theta} \mathbb{E}_{\mathbf{c} \sim \mathcal{D}_c, \mathbf{x}_0 \sim p_\theta(\mathbf{x}_0|\mathbf{c})} [r(\mathbf{c}, \mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta(\mathbf{x}_0|\mathbf{c}) \| p_{\text{ref}}(\mathbf{x}_0|\mathbf{c})]$$

최적 해를 재매개변수화(reparameterization)하면 LLM용 DPO 손실 함수는:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{\mathbf{c}, \mathbf{x}_0^w, \mathbf{x}_0^l} \left[ \log \sigma \left( \beta \log \frac{p_\theta(\mathbf{x}_0^w|\mathbf{c})}{p_{\text{ref}}(\mathbf{x}_0^w|\mathbf{c})} - \beta \log \frac{p_\theta(\mathbf{x}_0^l|\mathbf{c})}{p_{\text{ref}}(\mathbf{x}_0^l|\mathbf{c})} \right) \right]$$

#### 핵심 도전 과제

확산 모델에서는 $p_\theta(\mathbf{x}_0|\mathbf{c})$가 모든 가능한 확산 경로 $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$를 주변화(marginalize)해야 하므로 직접 계산이 불가능합니다.

#### 해결책: ELBO를 활용한 확산 경로 공간으로의 확장

잠재 변수 $\mathbf{x}\_{1:T}$를 도입하고 전체 체인에 대한 보상 $R(\mathbf{c}, \mathbf{x}_{0:T})$을 정의하면:

$$r(\mathbf{c}, \mathbf{x}\_0) = \mathbb{E}_{p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0, \mathbf{c})}[R(\mathbf{c}, \mathbf{x}_{0:T})]$$

KL 발산의 상한(upper bound)으로 조인트 KL을 사용하여 목표 함수를 경로 공간으로 확장:

$$\max_{p_\theta} \mathbb{E}_{\mathbf{c}, \mathbf{x}_{0:T} \sim p_\theta(\mathbf{x}_{0:T}|\mathbf{c})} [r(\mathbf{c}, \mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta(\mathbf{x}_{0:T}|\mathbf{c}) \| p_{\text{ref}}(\mathbf{x}_{0:T}|\mathbf{c})]$$

이를 DPO 형식으로 변환하면:

$$\mathcal{L}_{\text{DPO-Diffusion}}(\theta) = -\mathbb{E}_{(\mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D}} \log \sigma \left( \beta \mathbb{E}_{\substack{\mathbf{x}_{1:T}^w \sim p_\theta(\mathbf{x}_{1:T}^w|\mathbf{x}_0^w) \\ \mathbf{x}_{1:T}^l \sim p_\theta(\mathbf{x}_{1:T}^l|\mathbf{x}_0^l)}} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T}^w)}{p_{\text{ref}}(\mathbf{x}_{0:T}^w)} - \log \frac{p_\theta(\mathbf{x}_{0:T}^l)}{p_{\text{ref}}(\mathbf{x}_{0:T}^l)} \right] \right)$$

#### Jensen 부등식 적용 후 상한 도출

역방향 과정을 전방향 과정 $q$로 근사하고 Jensen 부등식을 적용하면:

$$\mathcal{L}_{\text{DPO-Diffusion}}(\theta) \leq -\mathbb{E}_{\substack{(\mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D},\, t \sim \mathcal{U}(0,T) \\ \mathbf{x}_t^w \sim q(\mathbf{x}_t^w|\mathbf{x}_0^w),\, \mathbf{x}_t^l \sim q(\mathbf{x}_t^l|\mathbf{x}_0^l)}} \log \sigma \left( \beta T \log \frac{p_\theta(\mathbf{x}_{t-1}^w|\mathbf{x}_t^w)}{p_{\text{ref}}(\mathbf{x}_{t-1}^w|\mathbf{x}_t^w)} - \beta T \log \frac{p_\theta(\mathbf{x}_{t-1}^l|\mathbf{x}_t^l)}{p_{\text{ref}}(\mathbf{x}_{t-1}^l|\mathbf{x}_t^l)} \right)$$

#### 최종 실용적 손실 함수 (Gaussian 역방향 과정 대입)

가우시안 역방향 과정의 매개변수화(Eq. 1)를 대입하면, 최종적으로 **노이즈 예측 오차 차이** 형태의 간결한 손실이 도출됩니다:

$$\boxed{\mathcal{L}(\theta) = -\mathbb{E}_{\substack{(\mathbf{x}_0^w, \mathbf{x}_0^l) \sim \mathcal{D},\, t \sim \mathcal{U}(0,T) \\ \mathbf{x}_t^w \sim q(\mathbf{x}_t^w|\mathbf{x}_0^w),\, \mathbf{x}_t^l \sim q(\mathbf{x}_t^l|\mathbf{x}_0^l)}} \log \sigma \left( -\beta T \omega(\lambda_t) \left( \underbrace{\|\boldsymbol{\epsilon}^w - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^w, t)\|_2^2 - \|\boldsymbol{\epsilon}^w - \boldsymbol{\epsilon}_{\text{ref}}(\mathbf{x}_t^w, t)\|_2^2}_{\text{선호 이미지의 노이즈 예측 개선}} - \underbrace{\left(\|\boldsymbol{\epsilon}^l - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^l, t)\|_2^2 - \|\boldsymbol{\epsilon}^l - \boldsymbol{\epsilon}_{\text{ref}}(\mathbf{x}_t^l, t)\|_2^2\right)}_{\text{비선호 이미지의 노이즈 예측 악화}} \right) \right)}$$

여기서:
- $\mathbf{x}_t^\* = \alpha_t \mathbf{x}_0^\* + \sigma_t \boldsymbol{\epsilon}^\*$, $\boldsymbol{\epsilon}^* \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\lambda_t = \alpha_t^2 / \sigma_t^2$: 신호 대 잡음비
- $\omega(\lambda_t)$: 가중 함수 (실제로는 상수 취급)
- $\beta$: KL 발산 패널티 강도 (하이퍼파라미터)

**직관적 해석:** 이 손실 함수는 $\boldsymbol{\epsilon}_\theta$가 선호 이미지 $\mathbf{x}_0^w$를 더 잘 디노이징하고 비선호 이미지 $\mathbf{x}_0^l$를 덜 잘 디노이징하도록 유도합니다.

---

### 2.3 모델 구조

Diffusion-DPO는 별도의 새로운 아키텍처를 도입하지 않고, **기존 확산 모델의 UNet 구조(노이즈 예측 네트워크 $\boldsymbol{\epsilon}_\theta$)를 그대로 파인튜닝**합니다.

```
┌─────────────────────────────────────────────────────────┐
│                  Diffusion-DPO 학습 구조                  │
├─────────────────────────────────────────────────────────┤
│  입력: (c, x_0^w, x_0^l) - 프롬프트 + 선호/비선호 이미지 쌍 │
│                          │                               │
│         ┌────────────────┴────────────────┐             │
│         ▼                                 ▼             │
│  [학습 모델 ε_θ]                   [동결 참조 모델 ε_ref]  │
│  (파인튜닝 대상)                    (초기화 시점 고정)      │
│         │                                 │             │
│         ▼                                 ▼             │
│   노이즈 예측 오차 계산             노이즈 예측 오차 계산    │
│  ||ε^w - ε_θ(x_t^w,t)||²         ||ε^w - ε_ref(x_t^w,t)||² │
│  ||ε^l - ε_θ(x_t^l,t)||²         ||ε^l - ε_ref(x_t^l,t)||² │
│         │                                 │             │
│         └────────────────┬────────────────┘             │
│                          ▼                               │
│              DPO 손실 계산 및 역전파                       │
│   L = -log σ(-β·T·ω(λ_t)·(w_diff - l_diff))            │
└─────────────────────────────────────────────────────────┘
```

**실험 설정:**
- **기반 모델:** SD 1.5, SDXL-1.0 base
- **데이터셋:** Pick-a-Pic v2 (851,293 쌍, 58,960 고유 프롬프트)
- **옵티마이저:** AdamW (SD1.5), Adafactor (SDXL)
- **배치 크기:** 2048 쌍 (effective), 16× NVIDIA A100
- **$\beta$ 범위:** 2,000 ~ 5,000 (SD1.5: 2000, SDXL: 5000)

---

### 2.4 성능 향상

| 평가 기준 | 비교 대상 | DPO-SDXL 승률 |
|-----------|-----------|----------------|
| General Preference (PartiPrompts) | SDXL-base | **70.0%** |
| Visual Appeal (PartiPrompts) | SDXL-base | **64.3%** |
| Prompt Alignment (PartiPrompts) | SDXL-base | **64.9%** |
| General Preference (HPSv2) | SDXL-base | **64.7%** |
| General Preference (PartiPrompts) | SDXL-base + Refiner (6.6B) | **69.0%** |
| General Preference (HPSv2) | SDXL-base + Refiner (6.6B) | **64.0%** |
| Image-to-Image (TEdBench) | SDXL | **65%** |

- HPSv2 리더보드 1위 (평균 보상 28.16)
- DPO-SDXL(3.5B 파라미터)이 SDXL base+refiner(6.6B, 53% 더 많은 파라미터)를 능가

---

### 2.5 한계점

1. **오프라인(Off-policy) 알고리즘:** 정적 데이터셋에 의존하므로 온라인 탐색(exploration) 부재 → 온라인 학습 방법이 추가 성능 향상에 필요
2. **훈련 데이터 품질 의존성:** SDXL은 훈련 데이터(Pick-a-Pic, SDXL-beta 생성)보다 이미 우수하나, 여전히 데이터 품질이 성능에 영향
3. **선호도의 비보편성:** "에너제틱하고 드라마틱한" 이미지를 향한 집단 편향이 있어 개인 취향 다양성 반영 미흡
4. **SFT 미적용:** SDXL에서는 SFT가 오히려 성능 저하를 야기 (Pick-a-Pic 데이터가 SDXL보다 품질 낮음이 원인)
5. **안전성 문제:** 웹 크롤링 데이터 기반으로 유해 콘텐츠, 성적 편향 등의 윤리적 위험 존재
6. **개인화 부재:** 개인 또는 소그룹 맞춤 선호도 반영 미지원

---

## 3. 모델의 일반화 성능 향상 가능성

Diffusion-DPO는 일반화 성능 향상에 있어 다음과 같은 핵심 메커니즘을 제공합니다:

### 3.1 개방 어휘(Open-vocabulary) 일반화의 근본 원리

기존 RL 방법(DDPO)의 기울기 추정은 다음과 같습니다:

$$\nabla_\theta \mathcal{J}_{\text{DDRL}} = \mathbb{E} \sum_{t=0}^{T} \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c})}{p_{\theta_{\text{old}}}(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c})} \nabla_\theta \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) \cdot r(\mathbf{x}_0, \mathbf{c})$$

이 방식에서는 **높은 보상을 얻은 소수의 프롬프트가 기울기 방향을 지배**하여, 저보상 프롬프트는 무시되거나 억제됩니다. 반면 Diffusion-DPO는 **동일한 컨디셔닝 $\mathbf{c}$ 내에서 선호/비선호 이미지를 대조**하므로, 보상 크기에 관계없이 **모든 프롬프트 유형이 동등하게 최적화**됩니다.

### 3.2 KL 발산 제어를 통한 분포 보존

$\beta$ 파라미터가 참조 분포와의 KL 발산을 제어합니다:

$$\min_{p_\theta} \mathbb{D}_{\text{KL}}(p_\theta(\mathbf{x}_{0:T}|\mathbf{c}) \| p_{\text{ref}}(\mathbf{x}_{0:T}|\mathbf{c}) \exp(R(\mathbf{c}, \mathbf{x}_{0:T})/\beta)/Z(\mathbf{c}))$$

- **낮은 $\beta$:** 보상 극대화에 집중 → 모드 붕괴 위험
- **높은 $\beta$:** 참조 분포에 가깝게 유지 → 다양성 보존, 일반화 유리
- 논문에서 $\beta \in [2000, 5000]$이 최적 범위로 확인됨

### 3.3 암묵적 보상 모델의 일반화 능력

Diffusion-DPO는 명시적 보상 모델 없이 **암묵적 보상 차이**를 학습합니다:

$$r(\mathbf{c}, \mathbf{x}_0^A) - r(\mathbf{c}, \mathbf{x}_0^B) = \beta \left[ (SE_\theta^A - SE_{\text{ref}}^A) - (SE_\theta^B - SE_{\text{ref}}^B) \right]$$

여기서 $SE_\psi^d = \|\boldsymbol{\epsilon}_{q_t} - \psi(\mathbf{c}, t, q_t(\mathbf{x}_0^d))\|_2^2$입니다.

Table 2에 따르면 DPO-SDXL은 Pick-a-Pic v2 검증셋에서 **72.0%의 선호도 분류 정확도**를 달성하여 PickScore(64.2%), HPSv2(59.3%), CLIP(57.1%) 등 모든 명시적 보상 모델을 능가합니다. 이는 **암묵적 보상 모델이 명시적 보상 모델 이상의 표현력과 일반화 능력**을 가짐을 시사합니다.

### 3.4 AI 피드백을 통한 확장성 (일반화의 스케일링)

단순히 훈련 데이터를 AI 스코어러로 자동 레이블링하여 의사 레이블(pseudo-label) 데이터셋을 생성하면:

- PickScore 기반 AI 피드백으로 훈련 시: 일반 선호도(General Preference) 승률이 **59.8% → 63.3%** 향상
- 이는 **인간 레이블 없이도 스케일링 가능한 일반화 경로**를 제시

### 3.5 분포 외(Out-of-distribution) 데이터에서의 일반화

Fig. 7의 실험에서:
- SDXL이 훈련 데이터(SDXL-beta 생성 이미지)보다 PickScore 기준 **이미 우수함에도 불구하고**, Diffusion-DPO 파인튜닝 후 성능이 추가로 향상됨
- 이는 **훈련 데이터의 분포 밖에서도 일반화**가 가능함을 시사

### 3.6 다양한 다운스트림 태스크로의 전이 (Task Generalization)

Diffusion-DPO는 텍스트-이미지 생성뿐 아니라:
- **Image-to-Image 편집 (SDEdit):** TEdBench에서 65% 승률
- **Color layout 기반 편집**에서도 시각적 품질 향상

이 결과는 학습된 선호도 정보가 특정 태스크에 국한되지 않고 **범용적으로 전이**됨을 보여줍니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 방법 | 개방 어휘 | 추론 비용 | 분포 제어 | 주요 한계 |
|------|------|------|-----------|-----------|-----------|-----------|
| **Ho et al. (DDPM)** | 2020 | 기본 확산 학습 | ✓ | 표준 | N/A | 선호도 학습 없음 |
| **Rombach et al. (LDM/SD)** | 2022 | Latent Diffusion | ✓ | 표준 | N/A | 선호도 학습 없음 |
| **Rafailov et al. (DPO)** | 2023 | LLM용 DPO | ✓ | 표준 | ✓ | 이미지 생성에 직접 적용 불가 |
| **Black et al. (DDPO)** | 2023 | RL (Policy Gradient) | ✗ | 표준 | ✗ | 제한적 어휘, 모드 붕괴 |
| **Fan et al. (DPOK)** | 2023 | RL + KL 정규화 | ✗ | 표준 | ✓ | 어휘 확장 시 성능 저하 |
| **Clark et al. (DRaFT)** | 2023 | 미분 가능 보상 최대화 | ✓ | 표준 | ✗ | CLIP 정렬 개선 불가, 모드 붕괴 |
| **Prabhudesai et al. (AlignProp)** | 2023 | 보상 역전파 | ✓ | 표준 | ✗ | 모드 붕괴, 분포 보장 없음 |
| **Wallace et al. (DOODL)** | 2023 | 추론 시간 최적화 | ✓ | **10배 이상** | ✗ | 추론 비용 과대 |
| **Dai et al. (Emu)** | 2023 | 고품질 데이터 파인튜닝 | ✓ | 표준 | N/A | 비용 높음, 피드백 유형 제한 |
| **Wallace et al. (Diffusion-DPO)** | 2023 | ELBO 기반 DPO | **✓** | **표준** | **✓** | 오프라인, 안전성 |

### DPO vs. DDPO 기울기 추정 비교

**DDPO (RL 기반):**
$$\nabla_\theta \mathcal{J} = \mathbb{E} \sum_{t=0}^{T} \frac{p_\theta(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{c})}{p_{\theta_{\text{old}}}(\mathbf{x}\_{t-1}|\mathbf{x}\_t, \mathbf{c})} \nabla_\theta \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) \cdot r(\mathbf{x}_0, \mathbf{c})$$
→ 보상이 가중치 역할만 하므로 다양성 보장 없음, 어휘 확장 시 취약

**Diffusion-DPO (대조 기반):**

```math
\mathcal{L} = -\log \sigma\left(-\beta T \omega(\lambda_t)\left[\left(\|\boldsymbol{\epsilon}^w - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^w, t)\|^2 - \|\boldsymbol{\epsilon}^w - \boldsymbol{\epsilon}_{\text{ref}}(\mathbf{x}_t^w, t)\|^2\right) - \left(\|\boldsymbol{\epsilon}^l - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t^l, t)\|^2 - \|\boldsymbol{\epsilon}^l - \boldsymbol{\epsilon}_{\text{ref}}(\mathbf{x}_t^l, t)\|^2\right)\right]\right)
```

→ 동일 컨디셔닝 내 대조를 통해 모든 프롬프트 유형 동등 처리

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 확산 모델 정렬의 새로운 패러다임 수립**
- Diffusion-DPO는 RLHF 없이 직접 선호도 데이터로 확산 모델을 정렬하는 첫 번째 방법으로서, 이후 연구들의 기준점(baseline)이 됩니다.
- 텍스트-이미지 모델뿐 아니라 비디오 생성(Video Diffusion), 3D 생성 등 다른 확산 모델 도메인에도 동일한 프레임워크 적용이 기대됩니다.

**② LLM-이미지 통합 멀티모달 정렬 연구 촉진**
- LLM에서 성공한 DPO를 이미지 도메인으로 확장한 이 연구는, 향후 멀티모달 LLM(예: GPT-4V, LLaVA류)에서의 생성 품질 정렬 연구를 자극합니다.

**③ AI 피드백 기반 스케일링 연구 가속화**
- 인간 레이블 없이 AI 스코어러로도 경쟁력 있는 성능을 달성했으므로, 대규모 합성 선호도 데이터 구축 및 자기개선(Self-improvement) 루프 연구가 활발해질 것입니다.

**④ 온라인 DPO 연구 방향 제시**
- 현재 오프라인 방식의 한계를 인식하고 있어, **온라인 DPO**(모델이 생성한 데이터로 반복 학습)나 **반복적 DPO(iterative DPO)** 연구가 확산 모델 분야에서 중요한 연구 방향으로 부상합니다.

**⑤ 개인화 및 그룹별 선호도 학습**
- 논문이 미래 연구로 제시한 개인/소그룹 맞춤 선호도 학습은, 사용자별 맞춤형 이미지 생성 서비스의 기반이 될 수 있습니다.

### 5.2 앞으로의 연구 시 고려할 점

**① 데이터 품질 및 편향(Bias) 관리**
- Pick-a-Pic 등 크라우드소싱 데이터는 특정 스타일(드라마틱, 고대비)을 선호하는 집단 편향을 내포합니다. 미래 연구에서는 **다양한 인구통계 그룹의 균형 잡힌 어노테이터 구성**이 필수적입니다.
- 훈련 데이터 필터링 및 클리닝 파이프라인 개발이 성능 향상에 기여할 수 있습니다(논문에서 예비 실험 확인).

**② 온라인 학습 및 탐색-활용 균형**
- 오프라인 DPO의 한계를 극복하려면, 모델이 새로운 이미지를 생성하고 그에 대한 피드백을 받는 **온라인 학습 루프**가 필요합니다. 이때 탐색(exploration)과 활용(exploitation) 사이의 균형 조절이 중요합니다.

**③ $\beta$ 선택과 KL 발산 제어**
- $\beta$ 값은 성능에 매우 민감하며($\beta$가 너무 낮으면 보상 해킹, 너무 높으면 참조 모델에 고착), **적응적 $\beta$ 스케줄링** 또는 자동 조율 방법 연구가 필요합니다.

**④ 다단계 파이프라인과의 통합**
- SDXL처럼 base+refiner 구조를 가진 모델에서는 각 단계를 독립적으로 정렬할지, 공동으로 정렬할지에 대한 연구가 필요합니다.

**⑤ 평가 지표의 한계 극복**
- PickScore, HPSv2 등의 자동 지표는 훈련 데이터 편향을 반영할 수 있습니다. **선호도 평가의 황금 표준(gold standard)** 수립과 다양한 문화권/목적별 평가 프로토콜 개발이 중요합니다.

**⑥ 안전성 및 윤리적 정렬**
- 선호도 학습이 유해 콘텐츠 생성 가능성을 높일 수 있습니다. **안전성 목표(safety objective)를 선호도 목표와 함께 최적화**하는 다목적 정렬 방법론 개발이 필요합니다.

**⑦ 비디오/3D 등 다른 생성 모달리티로의 확장**
- Diffusion-DPO의 이론적 프레임워크는 시간축을 포함한 비디오 생성 모델이나 3D 확산 모델에도 적용 가능하나, 시간적 일관성(temporal consistency)과 같은 추가 도전 과제 해결이 필요합니다.

**⑧ 데이터 효율성 향상**
- 851K 쌍의 대규모 데이터가 필요했으며, **소규모 데이터로도 효과적인 정렬**이 가능한 방법(예: Active Learning, 데이터 증강)을 탐색해야 합니다. Dreamlike 모델 실험에서 15% 데이터만 사용 시 제한적 개선이 보인 점이 데이터 의존성을 시사합니다.

---

## 참고자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Wallace et al. (2023)** - "Diffusion Model Alignment Using Direct Preference Optimization" (본 논문) - arXiv:2311.12908
2. **Rafailov et al. (2023)** - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" - [DPO 원논문]
3. **Ho et al. (2020)** - "Denoising Diffusion Probabilistic Models" - NeurIPS 2020
4. **Podell et al. (2023)** - "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
5. **Rombach et al. (2022)** - "High-Resolution Image Synthesis with Latent Diffusion Models" - CVPR 2022
6. **Black et al. (2023)** - "Training Diffusion Models with Reinforcement Learning" (DDPO) - arXiv:2305.13301
7. **Fan et al. (2023)** - "DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models" - arXiv:2305.16381
8. **Clark et al. (2023)** - "Directly Fine-Tuning Diffusion Models on Differentiable Rewards" (DRaFT)
9. **Prabhudesai et al. (2023)** - "Aligning Text-to-Image Diffusion Models with Reward Backpropagation" (AlignProp)
10. **Kirstain et al. (2023)** - "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation" - arXiv:2305.01569
11. **Wu et al. (2023)** - "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis" - arXiv:2306.09341
12. **Dai et al. (2023)** - "Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack" - arXiv:2309.15807
13. **Lee et al. (2023)** - "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"
14. **Song et al. (2020)** - "Denoising Diffusion Implicit Models" - arXiv:2010.02502
15. **Kingma et al. (2021)** - "Variational Diffusion Models"
16. **Garg et al. (2021)** - "IQ-Learn: Inverse Soft-Q Learning for Imitation" - NeurIPS 2021
