# RLP: Reinforcement as a Pretraining Objective

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

RLP(Reinforcement Learning Pre-training)는 **강화학습의 핵심 정신인 탐색(exploration)을 사전학습(pretraining) 단계에 도입**하는 새로운 목적함수입니다. 기존 LLM 학습 패러다임은 방대한 데이터에 대한 다음 토큰 예측(Next-Token Prediction, NTP)으로 사전학습을 수행한 후, 포스트트레이닝 단계에서만 강화학습(RL)을 적용합니다. RLP는 이 패러다임에 근본적인 의문을 제기하며, **Chain-of-Thought(CoT)를 탐색적 행동(exploratory action)으로 취급하여 사전학습 단계에서부터 추론 능력을 주입**합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **새로운 목적함수** | 검증기(verifier) 없이 정보이득(information gain)을 보상으로 사용하는 사전학습 목적함수 제안 |
| **안정적 학습 알고리즘** | Group-relative advantage, Clipped surrogate, EMA 기준선을 결합한 실용적 학습 알고리즘 개발 |
| **이론적 보장** | 기대 보상과 교차 엔트로피 감소의 연결, 하한(lower bound) 도출 |
| **포괄적 실험 검증** | 다양한 데이터셋, 도메인, 아키텍처에서 효과 검증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 기존 사전학습의 한계

표준 NTP 목적함수:

$$\mathcal{L}_{\text{NTP}}(\eta) := \mathbb{E}_{(x_{<t}, x_t) \sim \mathcal{D}} \left[ \log q_\eta(x_t \mid x_{<t}) \right] $$

이 목적함수는:
- **장거리 추론(long-range reasoning)을 명시적으로 장려하지 않음**
- 세계 지식과의 통합을 촉진하지 않음
- 인간의 이해 방식(병렬적 맥락 통합)과 달리 선형 토큰 단위 처리에 의존
- 강화학습은 포스트트레이닝의 **마지막 단계**에만 적용되어, 사전학습에서 형성된 표현의 한계를 극복하기 어려움

#### RLP가 다루는 핵심 질문

> *"사전학습 단계에서부터 RL의 탐색 정신을 도입하면 더 강력한 추론 기반을 형성할 수 있는가?"*

---

### 2.2 제안하는 방법 (수식 포함)

#### 기본 설정

모델은 **단일 네트워크** $\theta$를 공유하는 두 역할을 수행합니다:
- **사고 정책(Thought Policy)**: $\pi_\theta(c_t \mid x_{<t})$ — CoT 생성
- **추론 예측기(Reasoned Predictor)**: $p_\theta(x_t \mid x_{<t}, c_t)$ — CoT를 조건으로 다음 토큰 예측
- **EMA 기준선(No-Think Baseline)**: $\bar{p}\_\phi(x_t \mid x_{ < t})$ — CoT 없이 예측하는 지수이동평균 교사 모델

#### EMA 기준선 업데이트

$$\phi \leftarrow \tau \phi + (1 - \tau) \theta, \quad \tau = 0.999$$

EMA가 너무 빠르게 업데이트되면 보상이 0으로 붕괴하고, 너무 느리면 의미 있는 비교가 불가능해집니다. $\tau = 0.999$는 이 균형을 최적화합니다.

#### 정보이득 보상 (Information-Gain Reward)

추론된 로그 증거(reasoned log-evidence)와 기준선 로그 증거:

```math
S_{\text{pred}}(c_t) := \log p_\theta\!\left(x_t \mid x_{ < t}, c_t\right)
```

$$S_{\text{EMA}} := \log \bar{p}_\phi\!\left(x_t \mid x_{<t}\right) $$

**정보이득 보상**은 로그 우도비(log-likelihood ratio):

$$r(c_t) := S_{\text{pred}}(c_t) - S_{\text{EMA}} $$

이 보상은:
- $r(c_t) > 0$: CoT가 다음 토큰 예측을 실제로 도움
- $r(c_t) < 0$: CoT가 도움이 되지 않음
- **검증기(verifier) 불필요** — 모델 자체의 로그 확률로 계산
- **모든 위치에서 밀집된(dense) 신호** 제공

#### RLP 최적화 목적함수

$$\max_\theta \; J(\theta) = \mathbb{E}_{x_{<t} \sim \mathcal{D}} \; \mathbb{E}_{c_t \sim \pi_\theta(\cdot \mid x_{<t})} \left[ r(c_t) \right] $$

**중요**: 기울기는 **사고 토큰에만** 적용되며, $r(c_t)$는 $\theta$에 대해 상수로 취급됩니다 (pθ나 p̄φ를 통한 역전파 없음).

#### 분산 감소를 위한 그룹-상대 이점 (Group-Relative Advantage)

각 문맥에 대해 $G \geq 2$개의 사고를 샘플링:

$$\bar{r} = \frac{1}{G} \sum_{j=1}^{G} r\!\left(c_t^{(j)}\right)$$

편향 없는(unbiased) 이점 추정:

$$A^{(i)} := \frac{G}{G-1}\!\left(r\!\left(c_t^{(i)}\right) - \bar{r}\right) $$

$\frac{G}{G-1}$ 인수는 포함평균(inclusive mean)의 $\left(1 - \frac{1}{G}\right)$ 축소를 제거합니다.

#### 클리핑된 서로게이트 손실 (Clipped Surrogate Loss)

PPO 스타일의 중요도 비율 클리핑:

$$\rho_u^{(i)} = \exp\!\left(\log \pi_\theta\!\left(\ell_u^{(i)} \mid \text{prefix}_u^{(i)}\right) - \log \pi_{\theta_{\text{old}}}\!\left(\ell_u^{(i)} \mid \text{prefix}_u^{(i)}\right)\right)$$

$$\mathcal{L}_{\text{clip}}(\theta) = -\mathbb{E}\!\left[\frac{1}{|c_t^{(i)}|} \sum_u \min\!\left(\rho_u^{(i)} \,\text{sg}(A^{(i)}),\; \text{clip}(\rho_u^{(i)};\, 1-\epsilon_\ell,\, 1+\epsilon_h)\,\text{sg}(A^{(i)})\right)\right] $$

---

### 2.3 이론적 보장

#### 명제 1 (교차 엔트로피 감소)

```math
\mathbb{E}_{x_t \sim p^*}\!\left[r(c_t)\right] = \mathrm{CE}\!\left(p^*, \bar{p}_\phi(\cdot \mid x_{ < t})\right) - \mathrm{CE}\!\left(p^*, p_\theta(\cdot \mid x_{ < t}, c_t)\right)
```

여기서 $\mathrm{CE}(p, q) \overset{\text{def}}{=} \mathbb{E}_{x \sim p}\!\left[-\log q(x)\right]$. 즉, **기대 보상 최대화 = 교차 엔트로피 감소 최대화**.

#### 명제 2 (하한 경계)

붕괴 예측기(collapsed predictor) $\tilde{p}\_\theta(x \mid x_{ < t}) = \mathbb{E}\_{z_t \sim \pi_\theta}\!\left[p_\theta(x \mid x_{ < t}, z_t)\right]$에 대해 Jensen 부등식으로:

$$J(\theta) = \mathbb{E}[r(c_t)] \leq \mathbb{E}\!\left[\log \frac{\tilde{p}_\theta(x_t \mid x_{<t})}{\bar{p}_\phi(x_t \mid x_{<t})}\right]$$

CoT-조건부 목적함수는 사고를 주변화(marginalize)했을 때 얻는 개선의 **계산 가능한 하한**.

#### 명제 3 (위치별-시퀀스 연결)

Teacher forcing 하에서 위치별 기대 보상의 평균은 시퀀스 레벨의 CE 개선과 동일:

```math
\mathbb{E}_{\boldsymbol{x}}\!\left[\frac{1}{T}\sum_{t=1}^T \mathbb{E}_{c_t \sim \pi_\theta} \mathbb{E}_{x_t \sim p^*}\!\left[r(c_t)\right]\right] = \mathrm{CE}_{\text{seq}}\!\left(p^*, \bar{p}_\phi\right) - \mathrm{CE}_{\text{seq}}\!\left(p^*, p_\theta[\pi_\theta]\right)
```

---

### 2.4 모델 구조

```
입력 문맥 x_{<t}
        │
        ▼
┌─────────────────────────────────┐
│         단일 네트워크 θ          │
│  ┌─────────────────────────┐   │
│  │  사고 정책 π_θ           │   │
│  │  c_t ~ π_θ(· | x_{<t}) │   │
│  └──────────┬──────────────┘   │
│             │ (G개 CoT 샘플링)  │
│  ┌──────────▼──────────────┐   │
│  │  추론 예측기 p_θ         │   │
│  │  p_θ(x_t | x_{<t}, c_t)│   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
        │                    │
        ▼                    ▼
   S_pred(c_t)    ←→    S_EMA (EMA 교사 p̄_φ)
        │
        ▼
   r(c_t) = S_pred - S_EMA  [정보이득 보상]
        │
        ▼
  그룹-상대 이점 A^(i) 계산
        │
        ▼
  클리핑 서로게이트로 θ 업데이트
  (사고 토큰에만 적용)
        │
        ▼
  EMA 업데이트: φ ← τφ + (1-τ)θ
```

**핵심 설계 원칙**:
- 사고를 생성하는 정책과 다음 토큰을 예측하는 예측기가 **동일한 파라미터 θ를 공유**
- EMA 교사는 별도 파라미터 $\phi$로 유지되어 "생각하지 않는" 반사실(counterfactual) 제공
- 기울기는 **사고 토큰에만** 적용 (관찰 토큰에는 적용 안 함)

---

### 2.5 성능 향상

#### QWEN3-1.7B-BASE 결과 (Table 1)

| 벤치마크 | $\mathcal{M}_{\text{base}}$ | $\mathcal{M}_{\text{CPT}}$ | $\mathcal{M}_{\text{RLP}}$ | $\mathcal{M}_{\text{RLP}}$+Post |
|---------|---------|---------|---------|---------|
| AIME25 | 2.25 | 3.96 | **5.02** | **7.05** |
| MATH500 | 48.45 | 57.52 | **58.48** | **64.30** |
| MMLU-Pro | 28.17 | 27.81 | **34.62** | **42.40** |
| Science Avg | 34.50 | 32.01 | **39.68** | **45.74** |
| **Overall** | 30.32 | 30.85 | **36.03** | **42.51** |

- $\mathcal{M}\_{\text{RLP}}$은 $\mathcal{M}\_{\text{base}}$ 대비 **평균 19%**, $\mathcal{M}_{\text{CPT}}$ 대비 **17% 향상**
- 동일한 포스트트레이닝 적용 시 $\mathcal{M}\_{\text{base}}$+Post 대비 **8%**, $\mathcal{M}_{\text{CPT}}$+Post 대비 **7% 향상**

#### NEMOTRON-NANO-12B-V2 결과 (Table 2)

| 구분 | $\mathcal{M}_{\text{base}}$ | $\mathcal{M}_{\text{RLP}}$ | $\mathcal{M}_{\text{RLP}}$+Post |
|------|---------|---------|---------|
| Overall | 42.81% | **61.32%** | **68.09%** |
| Science Avg | 34.51% | **57.26%** | **64.52%** |

- 전체 평균 **43% 상대적 향상** (42.81% → 61.32%)
- 데이터의 **0.125%만 사용**하여 달성 (250M vs 20T 토큰)

#### RPT와의 비교 (Table 3, QWEN3-1.7B-BASE)

| 설정 | 모델 | Math Avg | Science Avg | Avg |
|------|------|---------|---------|-----|
| Token-Matched | $\mathcal{M}_{\text{RPT}}$ | 47.50 | 35.88 | 41.69 |
| Token-Matched | $\mathcal{M}_{\text{RLP}}$ | **49.62** | **37.07** | **43.35** |
| Flop-Matched | $\mathcal{M}_{\text{RPT}}$ | 36.66 | 34.38 | 35.68 |
| Flop-Matched | $\mathcal{M}_{\text{RLP}}$ | **45.95** | **38.76** | **42.86** |

FLOP-matched 조건에서 RLP가 RPT 대비 **약 20% 상대적 향상**.

---

### 2.6 한계

논문에서 명시적으로 인정된 한계:

1. **계산 비용 증가**: SFT/CPT 대비 **약 2.25배 느린** 처리 속도 (롤아웃 생성 오버헤드)
   - SFT: ~92.34 samples/s vs RLP: ~41.07 samples/s
2. **롤아웃 수 포화**: G=16 이후 추가적 이득 감소 (G=32에서 미세 하락)
3. **완료 길이 의존성**: 짧은 CoT 길이(64 토큰)에서는 성능이 대폭 저하 (Overall 11.50%)
4. **단일 토큰 적용**: 실험에서는 문서당 1개의 토큰에만 RLP 적용 (이론상 모든 토큰 가능)
5. **주로 수학/과학 벤치마크 평가**: 창의성, 대화, 코드 생성 등 다른 태스크에서의 효과 미검증
6. **KL 페널티 미사용**: $\beta = 0$이 최적이지만, 이로 인한 분포 이탈 위험은 이론적으로 EMA 기준선으로 완화하나 완전히 제거되지는 않음

---

## 3. 일반화 성능 향상 가능성 (심층 분석)

### 3.1 도메인 다양성에 걸친 일반화

Table 4의 결과는 RLP의 핵심 강점을 보여줍니다:

```
SFT-style 데이터:
  OmniMath (수학):          Avg 41.43%  (+7.24% vs base)
  OpenThoughts (혼합):       Avg 41.45%  (+7.26% vs base)
  Nemotron-Crossthink (혼합): Avg 43.36%  (+9.17% vs base)  ← 최강

일반 사전학습 데이터:
  ACAD (학술 논문):          Avg 41.71%  (+7.52% vs base)
  Math-Text (수학 교재):     Avg 41.62%  (+7.43% vs base)
  Web-Crawl (웹 크롤):       Avg 42.13%  (+7.94% vs base)  ← 일반 데이터 중 최강
```

**핵심 발견**: 추론 데이터로 큐레이팅되지 않은 **웹 크롤 데이터에서도** 유의미한 추론 신호 추출 가능.

### 3.2 왜 RLP가 일반화되는가? — 메커니즘 분석

#### (a) 정보이득 보상의 데이터-앵커링 특성

보상이 **실제 다음 토큰** $x_t$에 대해 정의됩니다:

$$r(c_t) = \log p_\theta(x_t \mid x_{<t}, c_t) - \log \bar{p}_\phi(x_t \mid x_{<t})$$

Proposition 1에 의해:

```math
\mathbb{E}_{x_t \sim p^*}[r(c_t)] = \underbrace{\mathrm{CE}(p^*, \bar{p}_\phi(\cdot | x_{ < t}))}_{\text{기준선 오류}} - \underbrace{\mathrm{CE}(p^*, p_\theta(\cdot | x_{ < t}, c_t))}_{\text{추론 후 오류}}
```

이 설계는 "내부 자신감"이 아닌 **실제 데이터 분포에 대한 예측 정확도**를 최적화하므로, 도메인 불가지론적(domain-agnostic) 일반화를 가능하게 합니다.

#### (b) 다중 도메인 데이터에서의 상호 강화

Table S.2의 도메인 조합 실험:

| 학습 데이터 | Math Avg | Science Avg | Overall |
|------------|---------|------------|---------|
| 수학만 | 48.23 | 41.64 | 42.21 |
| 과학만 | 49.17 | 39.65 | 42.36 |
| **수학+과학 결합** | **49.76** | **42.54** | **43.36** |

**중요한 발견**: 다중 도메인 결합이 개별 도메인보다 **모든 지표에서** 우수합니다. 이는 RLP가 서로 다른 추론 패턴을 **상호 강화**하는 방식으로 학습함을 시사합니다.

#### (c) 모델 학습 초기 단계에서도 효과적

4T 토큰(전체 20T의 20%)만 학습한 초기 체크포인트에 RLP 적용:

| 지표 | 초기 체크포인트 | RLP 적용 후 |
|------|------------|-----------|
| Math Avg | 21.93 | **50.14** (+128%) |
| Science Avg@1[4] | 5.69 | **11.96** (+110%) |
| **Overall** | **12.05** | **24.08** (+100%) |

→ RLP는 **사전학습의 어느 단계에서도** 적용 가능하며, 초기 단계에서도 대폭적인 개선을 달성합니다.

#### (d) 퍼플렉시티 개선으로 확인된 일반화

| 모델 | Nemotron-CrossThink PPL | Wikitext-103 PPL |
|------|----------------------|----------------|
| $\mathcal{M}_{\text{base}}$ | 2.91 | 5.83 |
| $\mathcal{M}_{\text{RLP}}$ | **2.36** | **4.48** |

일반 도메인 텍스트(Wikitext-103)에서도 퍼플렉시티가 개선되어, **사고 생성이 일반 텍스트 압축에도 유용**함을 입증합니다.

#### (e) 포스트트레이닝과의 시너지가 일반화의 증거

RLP의 이점이 강력한 포스트트레이닝(SFT + RLVR) 후에도 **유지되고 증폭**됩니다:

$$\mathcal{M}_{\text{base}}\text{+Post} = 39.34\% \quad \text{vs} \quad \mathcal{M}_{\text{RLP}}\text{+Post} = 42.51\%$$

만약 RLP가 단순한 데이터 암기나 단기적 최적화라면, 강력한 포스트트레이닝 후 이점이 사라져야 합니다. 이점의 **지속성과 증폭**은 RLP가 **근본적인 표현 학습 개선**을 유발함을 보여줍니다.

### 3.3 아키텍처 독립적 일반화

| 아키텍처 | 파라미터 | 전체 평균 향상 |
|---------|---------|------------|
| QWEN3-1.7B (Pure Transformer) | 1.7B | 30.32% → 36.03% (+19%) |
| NEMOTRON-NANO-12B-V2 (Hybrid Mamba-Transformer) | 12B | 42.81% → 61.32% (+43%) |
| QWEN3-14B (Pure Transformer) | 14B | 60.66% → 65.00% (+7.2%) |

다양한 아키텍처에서 일관된 개선은 RLP가 **아키텍처-독립적(architecture-agnostic)** 임을 보여줍니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 사전학습 패러다임 관련 연구

| 연구 | 연도 | 방법 | RLP와의 차이 |
|------|------|------|------------|
| **GPT-3** (Brown et al.) | 2020 | 표준 NTP | RL 신호 없음, 암묵적 추론 |
| **T5** (Raffel et al.) | 2020 | 스팬 손상 | 생성적 추론 장려 없음 |
| **DeepSeek-R1** (Guo et al.) | 2025 | 포스트트레이닝 RLVR | RL을 사전학습이 아닌 포스트트레이닝에만 적용 |
| **QWEN3** (Yang et al.) | 2025 | 표준 사전학습 + 포스트트레이닝 | 동일한 순차적 패러다임 |
| **RPT** (Dong et al.) | 2025 | 희소 이진 보상 강화 사전학습 | 보조 모델 필터링 필요, 희소 보상 |
| **RLP** (본 논문) | 2025/2026 | 밀집 정보이득 보상 사전학습 | 검증기 불필요, 밀집 보상, 모든 텍스트 적용 가능 |

### 4.2 검증기-없는 보상(Verifier-Free Reward) 연구

| 연구 | 방법 | RLP와의 차이 |
|------|------|------------|
| **Self-Rewarding LMs** (Yuan et al., 2024) | SFT 후 자체 판단으로 선호도 쌍 생성 (반복적 DPO) | 포스트트레이닝 전용, 큐레이션된 데이터 필요 |
| **NOVER** (Liu et al., 2025b) | SFT 코퍼스에서 인센티브 RL | 포스트트레이닝 전용 |
| **Learning to Reason w/o External Rewards** (Zhao et al., 2025) | 모델 신뢰도를 보상으로 사용한 내부 피드백 RL | 포스트트레이닝 단계 |
| **RLP** | 정보이득(로그 우도비)을 보상으로 사용한 사전학습 | **사전학습 단계**, 임의 텍스트 적용 가능 |

### 4.3 CoT 및 추론 관련 연구와의 비교

| 연구 | 방법 | RLP와의 관계 |
|------|------|------------|
| **Chain-of-Thought Prompting** (Wei et al., 2022) | 추론 체인을 프롬프팅으로 유도 | RLP는 사전학습에서 CoT를 **내재화** |
| **Open-Reasoner-Zero** (Hu et al., 2025) | 기본 모델에 RL 직접 적용 | 포스트트레이닝 단계, 검증 가능한 태스크 필요 |
| **ProRL** (Liu et al., 2025a) | 연장된 RL로 추론 경계 확장 | 포스트트레이닝 전용 |
| **e3** (Setlur et al., 2025) | 탐색 학습으로 테스트 시간 계산 외삽 | 테스트 시간 추론 초점 |

### 4.4 RPT vs RLP 심층 비교

| 측면 | RPT (Dong et al., 2025) | RLP (본 논문) |
|------|------------------------|--------------|
| 보상 신호 | 희소(sparse), 이진(binary) | **밀집(dense)**, 연속(continuous) |
| 토큰 선택 | 보조 모델로 고-엔트로피 토큰 필터링 | 문서당 **무작위 1개 토큰** |
| 보조 모델 | 필요 (엔트로피 필터링용) | **불필요** |
| 기준선 | 고정 또는 증류된 체크포인트 | **EMA 동적 기준선** |
| 적용 범위 | 선선택된 토큰에만 | 임의 텍스트의 모든 위치 |
| FLOP-matched 성능 | 35.68% | **42.86%** (+20.2% 상대적) |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### (a) 패러다임 전환의 가능성

RLP는 LLM 학습의 **3단계 파이프라인** (사전학습 → SFT → RLHF/RLVR)에 대한 근본적 재고를 촉진합니다:

```
기존 패러다임:
사전학습(NTP) ──→ SFT ──→ RLVR
     ↑
  추론 능력 없음

RLP 패러다임:
사전학습(NTP + RLP) ──→ SFT ──→ RLVR
     ↑
  초기 추론 기반 형성
```

이는 **강화학습을 포스트트레이닝 전용으로 간주하는 관행에 도전**하며, 사전학습 자체를 보다 능동적인 학습 과정으로 재설계하는 방향을 제시합니다.

#### (b) 데이터 효율성 패러다임

RLP는 **극히 적은 데이터로 큰 성능 향상**을 달성:
- NEMOTRON-NANO-12B-V2에서 전체 사전학습 데이터의 **0.125%**만으로 43% 상대적 향상
- CPT 대비 35배 적은 데이터로 더 높은 성능

이는 미래 연구에서 **데이터 양보다 학습 방법론의 질**이 더 중요할 수 있음을 시사합니다.

#### (c) 사전학습-포스트트레이닝 경계 모호화

RLP의 성공은 사전학습과 포스트트레이닝의 경계를 흐리게 합니다. 앞으로 연구는:
- 어떤 RL 기법이 사전학습에 적합한가?
- 사전학습 단계의 RL이 포스트트레이닝 RL의 어떤 부분을 대체 또는 보완할 수 있는가?

#### (d) 인지과학적 영감의 구체화

RLP는 인간의 **병렬적 이해 방식** (Hagoort et al., 2004; Metzner et al., 2015)에서 영감을 받아 AI 학습에 구체적으로 구현했습니다. 이는 **신경과학 기반 AI 학습 설계**라는 새로운 연구 방향을 제시합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (a) 확장성 및 효율성

- **자동회귀 롤아웃의 벽시계 시간(wall-clock time)**: 현재 SFT 대비 2.25배 느림. **추론 캐싱(KV-cache), 투기적 디코딩(speculative decoding), 배치 롤아웃 최적화** 등이 필요합니다.
- **더 큰 모델 스케일 검증**: 1.7B, 12B, 14B에서 테스트되었지만, 70B+ 모델에서의 효과는 미검증.
- **사고 길이 최적화**: 현재 2048 토큰이 최적이지만, 태스크별 동적 길이 조정이 더 효율적일 수 있음.

#### (b) 보상 설계의 심층 연구

$$r(c_t) = \log p_\theta(x_t \mid x_{<t}, c_t) - \log \bar{p}_\phi(x_t \mid x_{<t})$$

이 보상은 단일 다음 토큰에 대한 정보이득만 측정합니다:
- **다중 토큰 정보이득**: $\sum_{k=1}^{K} \log p_\theta(x_{t+k} \mid x_{<t+k}, c_t)$와 같은 확장 가능성
- **토큰 의미론적 중요도 가중치**: 모든 토큰이 동등하게 중요하지 않을 수 있음
- **보상 해킹(reward hacking) 방지**: EMA 기준선의 $\tau$ 최적화 연구 심화

#### (c) 일반화의 이론적 이해

현재 이론은 단일 위치의 CE 감소를 보장하지만:
- **장거리 추론 효과의 이론화**: 사고가 단지 다음 토큰뿐만 아니라 장거리 예측에 어떻게 기여하는지
- **OOD(Out-of-Distribution) 일반화 보장**: 학습 분포 밖 데이터에서의 이론적 보장
- **최적 사고 길이의 이론적 유도**: 현재는 실험적으로 2048을 선택

#### (d) 다양한 태스크 및 모달리티로의 확장

- **코드 생성**: 코드 토큰에 대한 정보이득 보상 적용 가능성
- **멀티모달 모델**: 이미지, 오디오 등 다른 모달리티에서의 정보이득 정의
- **대화 및 창의적 글쓰기**: 단일 정답이 없는 생성 태스크에서의 적용 방법
- **비영어권 언어**: 다국어 코퍼스에서의 RLP 효과

#### (e) EMA 기준선의 대안 탐색

현재 $\tau = 0.999$ EMA가 최적이지만:
- **적응적 $\tau$ 스케줄**: 학습 진행에 따라 동적으로 $\tau$ 조정
- **앙상블 기준선**: 여러 EMA 모델의 앙상블을 기준선으로 사용
- **외부 소형 모델 기준선**: EMA 대신 별도의 소형 모델을 기준선으로 사용하되, 업데이트 비용 최소화

#### (f) 포스트트레이닝과의 최적 결합

- **RLP → SFT 전환 시점 최적화**: 어느 사전학습 단계에서 RLP를 중단하고 SFT로 전환할지
- **RLP와 RLVR의 교차 학습**: 사전학습 중 RLP와 RLVR를 교차 적용하는 커리큘럼 설계
- **데이터 혼합 비율 최적화**: RLP 데이터와 일반 NTP 데이터의 최적 혼합 비율

#### (g) 신뢰성 및 안전성

- **사고 추적(thought trace) 해석 가능성**: 내부 CoT가 실제로 어떤 추론을 수행하는지 더 깊은 분석 필요
- **편향(bias) 증폭 위험**: RLP가 훈련 데이터의 편향을 증폭시킬 가능성
- **분포 이탈(distribution shift) 모니터링**: $\beta = 0$ KL 설정에서 장기 학습 시 안정성

---

## 참고 자료

**논문 자체 (주요 출처)**:
- Hatamizadeh, A., Akter, S. N., Prabhumoye, S., Kautz, J., Patwary, M., Shoeybi, M., Catanzaro, B., & Choi, Y. (2026). **RLP: Reinforcement as a Pretraining Objective**. *ICLR 2026*. arXiv:2510.01265v2.

**논문에서 인용된 핵심 참고문헌**:
- Dong, Q., et al. (2025). **Reinforcement pre-training (RPT)**. arXiv:2506.08007.
- Guo, D., et al. (2025). **DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning**. arXiv:2501.12948.
- Yang, A., et al. (2025). **Qwen3 technical report**. arXiv:2505.09388.
- NVIDIA Nemotron Nano. (2025). **Efficient hybrid Mamba-Transformer reasoning model**. arXiv:2508.14444.
- Ouyang, L., et al. (2022). **Training language models to follow instructions with human feedback (InstructGPT)**. *NeurIPS 35*.
- Yuan, W., et al. (2024). **Self-rewarding language models**. arXiv:2401.10020.
- Liu, W., et al. (2025b). **NOVER: Incentive training for language models via verifier-free reinforcement learning**. arXiv:2505.16022.
- Zhao, X., et al. (2025). **Learning to reason without external rewards**. arXiv:2505.19590.
- Akter, S. N., et al. (2025). **Nemotron-Crossthink: Scaling self-learning beyond math reasoning**. arXiv:2504.13941.
- Guha, E., et al. (2025). **OpenThoughts: Data recipes for reasoning models**. arXiv:2506.04178.
- Setlur, A., et al. (2025). **e3: Learning to explore enables extrapolation of test-time compute for LLMs**. arXiv:2506.09026.
- Hendrycks, D., et al. (2021). **Measuring mathematical problem solving with the MATH dataset**. *NeurIPS*.
- Wang, Y., et al. (2024). **MMLU-Pro**. arXiv:2406.01574.
- Rein, D., et al. (2024). **GPQA: A graduate-level Google-proof Q&A benchmark**. *COLM 2024*.
- Gao, B., et al. (2024). **OmniMath: A universal Olympiad level mathematics benchmark**. arXiv:2410.07985.
- Han, S., Pari, J., Gershman, S. J., & Agrawal, P. **Position: General intelligence requires reward-based pretraining**. *ICML 2025 Position Paper Track*.
- Liu, M., et al. (2025a). **ProRL: Prolonged reinforcement learning expands reasoning boundaries**. arXiv:2505.24864.
