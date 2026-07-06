# The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장 (Counter-intuitive Claim)

이 논문의 핵심 주장은 **역설적(counter-intuitive)**이다:

> **Diffusion Language Models(dLLMs)의 임의 순서(arbitrary-order) 생성 능력이, 수학/코딩과 같은 일반 추론 과제에서는 오히려 추론 잠재력을 제한한다.**

직관적으로는 임의 순서 생성이 자동회귀(AR) 방식의 엄격한 좌→우 순서를 포함하는 더 큰 해 공간(solution space)을 제공하므로 더 나은 추론을 가능케 해야 하지만, 실험적으로는 **반대 현상**이 관찰된다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **현상 발견** | 임의 순서 생성이 "포킹 토큰(forking token)"을 우회하여 **엔트로피 저하(entropy degradation)**를 유발함을 실증 |
| **메커니즘 분석** | Pass@k 지표와 엔트로피 측정을 통해 솔루션 공간 붕괴 메커니즘 규명 |
| **방법론 제안** | **JustGRPO**: 복잡한 diffusion-specific RL 적응 없이 표준 GRPO를 dLLM에 직접 적용 |
| **실용적 가치** | 병렬 디코딩 능력을 완전히 보존하면서도 GSM8K 89.1%, MATH-500 45.1% 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 문제 ①: 임의 순서 생성의 역효과 — "유연성 함정(Flexibility Trap)"

dLLMs는 추론 과정에서 **"포킹 토큰(forking tokens)"**, 즉 "Therefore", "Since", "Thus"와 같은 논리적 분기점을 우회(bypass)하는 경향이 있다. 이 토큰들은 다음 추론 경로를 결정하는 분기점으로서 원래 높은 엔트로피를 가져야 한다.

임의 순서 생성 메커니즘에서는:
1. 모델이 **확신이 높은(low-uncertainty) 토큰**을 먼저 생성
2. 나중에 bypassed된 포킹 토큰을 채울 때, **이미 확정된 미래 맥락이 해당 토큰의 불확실성을 제거**
3. 결과적으로 **엔트로피 저하**: 포킹 토큰이 더 이상 열린 분기점이 아니라 "채우기" 역할만 수행

이를 **엔트로피 저하(entropy degradation)** 현상이라 명명한다.

#### 문제 ②: 기존 dLLM용 RL 방법의 복잡성

기존 방법들(d1, ESPO, GDPO, SPG 등)은 임의 순서를 보존하기 위해 다음과 같은 어려움을 감수한다:

- **토큰 수준 분해의 모호성**: $\pi(o_t | s_t)$ 형태의 고유한 조건부 확률 정의 불가
- **시퀀스 우도의 비가산성**:
$$\pi_\theta(o \mid q) = \sum_{\tau \in \mathcal{T}} \pi_\theta(o, \tau \mid q), \quad |\mathcal{T}| = O(N!)$$
  → 정확한 우도 계산 불가, ELBO 근사에 의존
- **샘플러-학습자 불일치(sampler-learner mismatch)**: rollout 시 신뢰도 기반 샘플링 $\pi_\theta^{\text{conf}}(o|q)$와 최적화 목표 $\pi_\theta(o|q)$ 간의 괴리

---

### 2.2 제안 방법 (JustGRPO)

#### 핵심 아이디어

RL 훈련 단계에서만 임의 순서를 포기하고 **dLLM을 AR 정책으로 취급**한다. 이는 구조적 변경 없이(causal masking 없이) 훈련 시에만 AR scaffold를 적용하는 것이다.

#### 수식: AR 정책 정의

$k$번째 토큰 $o_k$를 생성하기 위한 입력 시퀀스를 다음과 같이 구성한다:

$$\tilde{x}_k = [\underbrace{o_1, \ldots, o_{k-1}}_{\text{Observed}}, \underbrace{[\text{MASK}], \ldots, [\text{MASK}]}_{\text{Masked}}] $$

이를 바탕으로 대리 AR 정책(surrogate autoregressive policy)을 정의한다:

```math
\pi_\theta^{\text{AR}}(\cdot \mid o_{ < k}, q) \triangleq \text{Softmax}(f_{\theta,k}(\tilde{x}_k, q))
```

여기서 $f_{\theta,k}$는 position $k$에서의 모델 logit이다.

이를 통해 시퀀스 우도를 **정확히** 분해할 수 있다:

$$\pi_\theta^{\text{AR}}(o \mid q) = \prod_{k=1}^{|o|} \pi_\theta^{\text{AR}}(o_k \mid o_{ < k}, q) $$

#### Masked Diffusion Model (MDM) 기반 수식

MDM의 순전파 과정:

$$q(x_{t,k} \mid x_{0,k}) = \begin{cases} [\text{MASK}], & \text{with prob } t \\ x_{0,k}, & \text{with prob } 1-t \end{cases} $$

MDM 훈련 손실 (Negative ELBO):

$$\mathcal{L}_{\text{MDM}}(\theta) = -\mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_t \sim q(x_t|x_0)}\left[\frac{1}{t}\sum_{k=1}^{L} \mathbf{1}[x_{t,k} = [\text{MASK}]] \log p_\theta(x_{0,k} \mid x_t)\right] $$

#### GRPO 목적 함수

기본 GRPO 목적함수:

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{k=1}^{|o_i|}\left(\min\left(\rho_{i,k}\hat{A}_{i,k}, \text{clip}(\rho_{i,k}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,k}\right) - \beta\mathbb{D}_{\text{KL}}\right)\right] $$

**JustGRPO의 목적함수** (AR 정책 기반):

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}^{\text{AR}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{k=1}^{|o_i|}\left(\min\left(\rho_{i,k}\hat{A}_{i,k}, \text{clip}(\rho_{i,k}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,k}\right) - \beta\mathbb{D}_{\text{KL}}\right)\right] $$

여기서 중요도 비율(importance ratio):

$$\rho_{i,k} = \frac{\pi_\theta^{\text{AR}}(o_{i,k} \mid o_{i, < k}, q)}{\pi_{\theta_{\text{old}}}^{\text{AR}}(o_{i,k} \mid o_{i, < k}, q)}$$

어드밴티지:

$$\hat{A}_{i,k} = \frac{r(o_i) - \mu_G}{\sigma_G}$$

Pass@k 지표 (비편향 추정량):

$$\text{Pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right] $$

---

### 2.3 모델 구조

JustGRPO는 **모델 아키텍처를 변경하지 않는다**. 주요 구조적 특징:

```
훈련 단계:
  LLaDA-8B Instruct (양방향 어텐션 유지)
  + AR scaffold (causal masking 없음)
  → 표준 GRPO 적용

추론 단계:
  훈련된 모델 (양방향 어텐션 보존)
  + 병렬 디코딩 (EB-Sampler 등)
  → 기존 dLLM 추론 효율성 완전 보존
```

훈련 하이퍼파라미터:
- Base Model: LLaDA 8B Instruct
- Learning Rate: $5 \times 10^{-6}$
- Group Size $G$: 16
- Global Batch Size: 64
- Training Steps: 125
- Max Completion Length: 256
- Hardware: $16 \times$ NVIDIA H100 GPUs

---

### 2.4 성능 향상

#### 주요 벤치마크 결과 (LLaDA-Instruct 기준, Seq Len 256)

| 모델 | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| d1 (Zhao et al., 2025) | 81.1 | 38.6 | — | — |
| ESPO (Ou et al., 2026) | 82.3 | 39.0 | 42.1 | 44.6 |
| GDPO (Rojas et al., 2026) | 82.8 | 39.6 | 39.6 | 50.6 |
| SPG (Wang et al., 2026a) | 86.1 | 40.0 | — | — |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

#### 통일된 실험 환경에서의 비교 (Table 2)

| 모델 | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| d1* | 83.8 | 39.2 | — | — |
| ESPO* | 84.7 | 40.3 | 42.1 | 44.6 |
| SPG* | 86.9 | 41.8 | — | — |
| **JustGRPO** | **89.1** | **45.1** | **49.4** | **52.4** |

#### 병렬 디코딩과의 호환성

- 병렬 토큰 수 증가 시 성능 우위가 **더욱 확대**됨
- MBPP에서 1 token/step: +10.6% → ~5 tokens/step: +25.5%

---

### 2.5 한계

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **per-iteration 계산 오버헤드**: dLLM은 각 위치를 독립적으로 평가해야 하므로 단일 causal forward pass가 불가능 → 계산 비용 증가 (단, JustGRPO-Fast로 부분 완화)
2. **일반화 범위**: 수학/코딩에 집중된 실험. 창의적 글쓰기, Sudoku, Zebra puzzle 등 특정 구조화된 과제에서는 임의 순서가 여전히 유리할 수 있음
3. **단일 베이스 모델 의존**: 주로 LLaDA-Instruct에서 실험. 다른 dLLM 아키텍처에서의 일반화 여부 추가 검증 필요
4. **랜덤 순서의 실패**: 랜덤 순서(JustGRPO-Random)도 GSM8K 82.2%에 그침. 즉, 순서의 구조적 특성(인과성) 자체가 중요
5. **bidirectional refinement 미활용**: 훈련 후 모델이 기존 출력을 반복적으로 수정하는 능력 활용 가능성이 미탐구 상태

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Pass@k를 통한 잠재적 추론 공간의 확장

논문의 핵심 발견은 **AR 순서가 더 높은 Pass@k를 달성**한다는 것이다:

$$\text{Pass@k}_{\text{AR}} > \text{Pass@k}_{\text{Arbitrary}} \quad \text{for } k \geq 2$$

이는 AR 순서로 훈련된 모델이 **더 넓은 해 공간을 탐색**할 수 있음을 의미한다. 구체적으로:

- HumanEval에서 Pass@1024 기준: AR만 해결한 문제 21.3% vs. AO만 해결한 문제 0.6%
- 임의 순서로 해결되는 문제들은 AR로 해결되는 문제들의 **부분집합**에 가까움

### 3.2 엔트로피 보존과 일반화

AR 순서는 논리적 포킹 토큰에서 높은 엔트로피를 유지한다:

$$H_{\text{AR}}(\text{forking tokens}) \gg H_{\text{AO}}(\text{forking tokens})$$

이 높은 엔트로피가 **다양한 추론 경로를 샘플링**할 수 있게 하여, RL 훈련 시 더 많은 positive reward signal을 제공한다. 이는 일반화에 직접적으로 기여한다.

### 3.3 병렬 디코딩 하에서의 일반화 강건성

JustGRPO로 훈련된 모델은 병렬 디코딩(Tokens/Step 증가)에서 **더 강건한 성능**을 보인다:

- 기존 모델: 병렬 토큰 수 증가 시 성능 급락
- JustGRPO: 병렬 토큰 수 증가에도 **안정적인 성능** 유지

이는 AR scaffold 훈련이 특정 trajectory에 과적합하는 대신 **기저 모델 분포 자체를 개선**함을 시사한다. 개선된 분포는 병렬 샘플링의 근사(approximation)에 더 탄탄하여, 다양한 추론 조건에서 일반화 성능이 향상된다.

### 3.4 일반 능력(General Capability) 보존

JustGRPO 훈련 후 비추론 벤치마크 결과:

| 벤치마크 | LLaDA-Instruct | JustGRPO | 변화 |
|---|---|---|---|
| MMLU | 65.5% | 65.8% | +0.3% |
| MMLU-Pro | 37.0% | 36.7% | -0.3% |
| HellaSwag | 74.6% | 74.8% | +0.2% |
| ARC-C | 88.5% | 87.5% | -1.0% |

일반 능력이 **거의 보존**됨. 이는 AR scaffold가 모델의 지식 인코딩 방식에는 영향을 주지 않고 **추론 경로 탐색 방식만을 개선**함을 의미한다.

### 3.5 블록 크기와 임의성의 연속적 관계

반자동회귀(semi-autoregressive) 블록 크기 $B$ 실험에서:

$$\text{Pass@}k \downarrow \text{ monotonically as } B \uparrow$$

이는 **임의성이 적을수록 일관되게 추론 잠재력이 높아짐**을 보여주며, 이 관계가 단순히 두 극단 비교가 아닌 **연속적이고 일반적인 현상**임을 지지한다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 향후 연구에 미치는 영향

#### ① dLLM 설계 철학의 재검토

이 논문은 "더 많은 자유도 = 더 나은 성능"이라는 dLLM 분야의 암묵적 가정에 근본적인 의문을 제기한다. 향후 dLLM 설계에서:

- **사전학습(pre-training)**: 임의 순서 학습이 데이터 분포를 더 느슨하게 근사한다는 Du et al. (2025)의 분석과 함께, 사전학습에서도 AR 순서의 효용성을 재평가할 필요
- **아키텍처 설계**: 양방향 어텐션을 보존하면서도 생성 순서에 구조적 귀납 편향(inductive bias)을 부여하는 새로운 아키텍처 탐구 가능

#### ② RL for dLLMs 연구 방향 단순화

기존 연구들(ESPO, GDPO, SPG 등)이 복잡한 diffusion-specific 적응을 개발한 것과 달리, JustGRPO는 **단순함(simplicity)이 강력한 기준선(baseline)이 될 수 있음**을 보였다. 이는:

- 새로운 dLLM RL 방법은 JustGRPO 대비 유의미한 향상을 입증해야 함
- 복잡한 trajectory 처리의 실질적 필요성에 대한 재검토 유도

#### ③ 탐색-활용 트레이드오프(Exploration-Exploitation Tradeoff) 이해 심화

엔트로피 저하 현상은 **추론에서의 탐색-활용 트레이드오프**를 더 정밀하게 이해하는 틀을 제공한다:

- 포킹 토큰의 엔트로피 관리가 RL 훈련 성능에 핵심임을 시사
- "Low-probability tokens sustain exploration" (Wang et al., 2025; Huang et al., 2025a)과 같은 연구와 연계하여 **엔트로피 제어 기반 RL** 연구의 발전 가능

#### ④ JustGRPO-Fast의 파급 효과

상위 25% 고엔트로피 위치에서만 $\rho_{i,k}$를 계산하는 JustGRPO-Fast는:

- **희소 엔트로피 구조(sparse entropy structure)**를 활용한 효율적 RL의 가능성을 제시
- 포킹 토큰 식별 → 선택적 최적화 → 효율적 훈련의 패러다임이 AR 모델에도 적용 가능할지 탐구 여지

#### ⑤ 훈련-추론 분리(Decoupling) 원리의 확장

JustGRPO가 보여준 핵심 원리: **훈련 목적(탐색 효율)과 추론 실행(병렬 디코딩)을 분리**할 수 있다. 이 원리는:

- 다른 비자기회귀 모델(Non-AR models) 훈련 전략에도 적용 가능
- 제한적 훈련 분포로 강건한 추론 능력을 학습하는 일반적 접근으로 확장 가능

---

### 4.2 향후 연구 시 고려할 점

#### ① 임의 순서가 유리한 과제 경계 규명

이 논문은 수학/코딩에서 임의 순서의 한계를 보였으나, Ye et al. (2025a), Kim et al. (2025)은 Sudoku, Zebra puzzle에서의 우위를 보였다. 따라서:

- **어떤 과제 특성이 임의 순서의 유/불리를 결정하는가?**
- 과제별 "forking token" 밀도, 구조적 의존성, 인과성의 역할 등을 정밀 분석 필요

#### ② 더 다양한 dLLM 아키텍처에서의 검증

현재 실험은 주로 LLaDA-Instruct (8B)에 집중. 다음 환경에서의 재현성 확인 필요:
- 더 큰 모델 (예: Mercury/Inception Labs, Gemini Diffusion/DeepMind)
- 다른 마스킹 전략을 사용하는 MDM 변형

#### ③ 하이브리드 접근의 탐구

AR과 임의 순서의 이분법을 넘어서:

- **동적 블록 크기 조정**: 포킹 토큰 주변에서는 $B=1$ (AR), 나머지는 $B>1$ (병렬)
- **엔트로피 인식 스케줄링(entropy-aware scheduling)**: 생성 중 실시간 엔트로피 모니터링으로 순서 전략 적응적 전환
- **JustGRPO-Fast의 고도화**: 고엔트로피 위치 탐지를 학습 가능한 모듈로 발전

#### ④ 양방향 정제(Bidirectional Refinement) 활용

JustGRPO 이후 모델이 이미 생성된 출력을 반복적으로 수정하는 능력을 활용하는 연구:
- CDLM (Zhang et al., 2025b)의 수정적 접근
- ParallelBench (Kang et al., 2026)의 양방향 맥락 설정과의 결합

#### ⑤ 보상 함수 설계의 세분화

현재 이진(binary) 보상을 사용하지만:
- 중간 추론 단계에 대한 **과정 보상(process reward)** 도입 가능성
- 포킹 토큰의 다양성을 직접 장려하는 **엔트로피 보너스(entropy bonus)** 설계

#### ⑥ 사전학습과 RL 훈련의 상호작용

Zhang et al. (2025a)의 "사전학습-중간학습-RL 상호작용" 연구와 연계하여:
- 사전학습 단계부터 AR 순서로 훈련된 dLLM이 RL에서 더 나은 출발점을 제공하는지 검토

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 dLLM 기반 연구 계보

```
Continuous Diffusion for Text (2020-2022)
├── DDPM (Ho et al., NeurIPS 2020) — 연속 도메인 확산 기반
├── Diffusion-LM (Li et al., NeurIPS 2022) — 임베딩 공간 적용
└── DiffuSeq (Gong et al., ICLR 2023) — Seq2Seq 생성

Discrete/Masked Diffusion (2024)
├── MDLM (Lou et al., ICML 2024) — 이산 확산 비율 추정
├── MDLM (Sahoo et al., NeurIPS 2024) — 단순화된 마스크 확산
└── SMDM (Shi et al., NeurIPS 2024) — 일반화된 마스크 확산

Large-scale dLLMs (2025)
├── LLaDA (Nie et al., NeurIPS 2025) — 8B 규모 마스크 확산 LM
├── Dream (Ye et al., 2025) — 7B 확산 LM
├── LLaDA 1.5 (Zhu et al., 2025) — 선호도 최적화 개선
└── Mercury (Inception Labs, 2025) — 초고속 추론

RL for dLLMs (2025-2026)
├── d1 (Zhao et al., NeurIPS 2025) — 최초 dLLM RL 확장
├── ESPO (Ou et al., ICLR 2026) — 시퀀스 수준 관점
├── GDPO (Rojas et al., ICLR 2026) — 그룹 확산 정책 최적화
├── SPG (Wang et al., ICLR 2026a) — 샌드위치 정책 그래디언트
├── LLaDOU (Huang et al., NeurIPS 2025b) — 보조 위치 선택 모듈
└── JustGRPO [본 논문, 2026] — 단순 AR GRPO
```

### 5.2 RL 방법론별 상세 비교

| 방법 | 핵심 아이디어 | 우도 계산 | 임의 순서 보존 | GSM8K (256) | 복잡도 |
|---|---|---|---|---|---|
| **d1** (Zhao et al., 2025) | 토큰 수준 GRPO 직접 적용 | ELBO 근사 | ✓ | 81.1 | 중 |
| **ESPO** (Ou et al., 2026) | 시퀀스 수준 관점, 원리적 RL | ELBO 근사 | ✓ | 82.3 | 높음 |
| **GDPO** (Rojas et al., 2026) | 그룹 확산 정책 최적화 | 근사 | ✓ | 82.8 | 높음 |
| **SPG** (Wang et al., 2026a) | 샌드위치 정책 그래디언트 | ELBO 근사 | ✓ | 86.1 | 높음 |
| **LLaDOU** (Huang et al., 2025b) | 위치 선택 보조 모듈 | 직접 추정 | ✓ | 88.1 | 매우 높음 |
| **JustGRPO** [본 논문] | AR scaffold + 표준 GRPO | **정확한 계산** | ✗ (훈련 시) | **89.1** | **낮음** |

### 5.3 Pass@k 관점에서의 탐색 능력 비교

| 방법 | Pass@k 스케일링 | 탐색 효율 | 비고 |
|---|---|---|---|
| Arbitrary Order (dLLM 기본) | **낮음** (flat curve) | 낮음 | 포킹 토큰 우회 |
| AR Order (dLLM에 AR 적용) | **높음** (steep curve) | 높음 | 포킹 토큰 대면 |
| JustGRPO (훈련 후) | 높음 유지 | 높음 | AR 탐색 + 병렬 추론 |

### 5.4 병렬 추론 연구와의 관계

병렬 디코딩을 다루는 연구들과의 위치:

| 연구 | 방향 | JustGRPO와의 관계 |
|---|---|---|
| Fast-dLLM (Wu et al., ICLR 2026b) | KV cache + 병렬 디코딩 | **상호 보완적**: JustGRPO 훈련 후 Fast-dLLM 추론 적용 가능 |
| EB-Sampler (Ben-Hamu et al., NeurIPS 2025) | 엔트로피 경계 언마스킹 | **직접 활용**: JustGRPO는 EB-Sampler와 호환 확인 |
| ParallelBench (Kang et al., ICLR 2026) | 병렬 디코딩 트레이드오프 분석 | 향후 JustGRPO 모델 평가에 활용 권장 |
| CDLM (Zhang et al., 2025b) | 수정적 확산 언어 모델 | 양방향 정제와 결합 가능성 |

### 5.5 기존 dLLM 추론 연구와의 차별점

| 연구 | 비자기회귀 이점 주장 | 본 논문의 입장 |
|---|---|---|
| Ye et al. (ICLR 2025a) | Sudoku, Zebra puzzle에서 비순차 생성 우위 | 특정 구조화 과제에는 동의, **일반 추론에는 반론** |
| Kim et al. (ICML 2025) | 최악 순서 훈련이 일반화 향상 | 훈련 시 구조의 중요성 공유, 방향성은 차이 |
| Du et al. (2025) | 균등 순열 학습이 데이터 분포 근사를 느슨하게 함 | **pre-training에서의 AR 우위** 동일하게 관찰 → 상호 지지 |

---

## 참고 자료

본 답변은 제공된 PDF 논문 원문을 기반으로 작성되었습니다:

**주요 참고 논문 (논문 내 인용 기준)**:

1. **Ni et al. (2026)** — "The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models" *(본 논문, arXiv:2601.15165v4)*
2. **Nie et al. (NeurIPS 2025)** — "Large Language Diffusion Models" (LLaDA)
3. **Zhao et al. (NeurIPS 2025)** — "d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning"
4. **Shao et al. (2024)** — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO 원논문)
5. **Ou et al. (ICLR 2026)** — "Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective" (ESPO)
6. **Rojas et al. (ICLR 2026)** — "Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization" (GDPO)
7. **Wang et al. (ICLR 2026a)** — "SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models"
8. **Huang et al. (NeurIPS 2025b)** — "Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models" (LLaDOU)
9. **Ben-Hamu et al. (NeurIPS 2025)** — "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking" (EB-Sampler)
10. **Ye et al. (ICLR 2025a)** — "Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning"
11. **Kim et al. (ICML 2025)** — "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"
12. **Ho et al. (NeurIPS 2020)** — "Denoising Diffusion Probabilistic Models"
13. **Chen et al. (2021)** — "Evaluating Large Language Models Trained on Code" (Pass@k 지표)
14. **Yue et al. (NeurIPS 2025)** — "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
15. **Wang et al. (NeurIPS 2025)** — "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning"
16. **Du et al. (2025)** — "Understanding the Limitations of Diffusion LLMs through a Probabilistic Perspective"
17. **Zhu et al. (2025)** — "LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models"
18. **Zhang et al. (2025b)** — "Corrective Diffusion Language Models" (CDLM)
19. **Kang et al. (ICLR 2026)** — "ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs"
20. **Schulman et al. (ICML 2015)** — "Trust Region Policy Optimization" (TRPO)
