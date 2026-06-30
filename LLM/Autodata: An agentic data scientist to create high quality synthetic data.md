# Autodata: An Agentic Data Scientist to Create High Quality Synthetic Data 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**Autodata**는 LLM 에이전트를 "데이터 과학자"로 활용하여, 고품질 합성 학습/평가 데이터를 **자율적으로 생성·분석·개선**하는 일반적 프레임워크이다. 기존의 단순 프롬프트 기반 합성 데이터 생성(Self-Instruct 계열)의 한계를 넘어, 데이터 생성 → 품질 평가 → 분석 → 레시피 개선의 **반복적 루프**를 통해 더 강한 학습 신호를 제공하는 데이터를 만들 수 있다고 주장한다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **Autodata 프레임워크** | 데이터 생성, 분석, 반복 개선, 메타 최적화를 포괄하는 일반 아키텍처 제시 |
| **Agentic Self-Instruct** | Weak-Strong Solver 기반의 구체적 구현체; 데이터 난이도를 자동 조절 |
| **메타 최적화 루프** | 에이전트 자체를 진화적 방법으로 최적화 (검증 통과율 62.1% → 79.6%) |
| **추론 효율성 개선** | 학습 후 토큰 절단율 대폭 감소 (23.75% → 4.09%) |
| **도메인 일반성 입증** | CS 연구, 법률 추론, 수학적 과학 추론 등 이질적 도메인에서 일관된 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 합성 데이터 생성 방법들은 다음과 같은 공통적 한계를 가진다.

- **Self-Instruct** (Wang et al., 2023): 단순 프롬프팅으로 생성 → 난이도 제어 불가
- **CoT Self-Instruct** (Yu et al., 2025): 추론 과정 포함하지만 데이터 품질을 직접 제어하지 못함
- **WizardLM/Evol-Instruct** (Xu et al., 2024): 복잡도 진화는 가능하지만 피드백 루프 없음

결과적으로 두 가지 반대 방향의 실패 모드가 존재한다:

$$\text{CS 과제: } \text{Gap} = S_{\text{strong}} - S_{\text{weak}} = 0.696 - 0.677 = 0.019 \quad (\text{너무 쉬움})$$

$$\text{법률 과제: } S_{\text{weak}} \approx 0.159, \text{ Gap} = 0.558 \quad (\text{너무 어려움} \Rightarrow \text{GRPO 학습 신호 소멸})$$

즉, 단순 난이도 극대화가 아니라 **"Just Right"한 학습 신호**를 만드는 것이 핵심 문제이다.

---

### 2.2 제안하는 방법

#### 2.2.1 Autodata 일반 프레임워크

전체 파이프라인은 다음 세 단계로 구성된다.

$$\text{Data Creation} \rightarrow \text{Data Analysis} \rightarrow \text{Recipe Update} \rightarrow \cdots$$

이를 수식으로 나타내면, 에이전트의 목표는 다음과 같다:

$$\theta^* = \arg\max_{\theta} \; \mathbb{E}_{x \sim p_{\theta}(\mathcal{D})} \left[ \mathcal{R}(x; \mathcal{M}_{\text{weak}}, \mathcal{M}_{\text{strong}}) \right]$$

여기서:
- $p_\theta(\mathcal{D})$: 파라미터 $\theta$ (에이전트 프롬프트/전략)로 생성된 데이터 분포
- $\mathcal{R}$: Weak-Strong 분리도 기반 보상 함수
- $\mathcal{M}\_{\text{weak}}, \mathcal{M}_{\text{strong}}$: 각각 약한 솔버와 강한 솔버

#### 2.2.2 Agentic Self-Instruct (구체적 구현체)

4개의 서브에이전트로 구성된 멀티에이전트 시스템:

1. **Challenger**: grounding 문서로부터 QA 쌍 + 루브릭 생성
2. **Weak Solver** ($\mathcal{M}_w$): 생성된 문제를 풀고 실패를 보여주는 역할
3. **Strong Solver** ($\mathcal{M}_s$): 정답 보장 역할
4. **Verifier/Judge**: 품질 검증 및 피드백 생성

**검증 가능 과제의 수용 기준 (CS 과제):**

$$\text{Accept} \iff \begin{cases} \bar{s}_{\text{strong}} \geq 0.65 \\ \bar{s}_{\text{weak}} < 0.50 \\ \bar{s}_{\text{strong}} - \bar{s}_{\text{weak}} \geq 0.20 \end{cases}$$

**비검증 과제의 수용 기준 (법률 과제):**

Judge가 다음 구조화된 판정을 통해 유연하게 결정:

$$\text{verdict} = f(\text{weak pattern}, \text{strong pattern}, \text{gap interpretation}, \text{grpo suitability})$$

**GRPO 학습에서의 보상 구조:**

그룹 크기 $G$에 대해 GRPO의 이점(advantage)은:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

여기서 $r_i$는 $i$번째 롤아웃의 루브릭 점수이다. **핵심**: weak solver의 점수 분산이 너무 낮으면($\text{std} \approx 0$) 학습 신호가 소멸하므로, Agentic Self-Instruct는 이 분산을 적절하게 유지하도록 데이터를 조절한다.

#### 2.2.3 메타 최적화 (외부 루프)

Boltzmann 샘플링으로 후보 프롬프트를 선택:

$$P(\text{select } c) \propto \exp\left(\frac{\text{score}_c}{T}\right), \quad T = 0.1$$

최적화 과정:

$$\text{Select} \rightarrow \text{Evaluate} \rightarrow \text{Analyze} \rightarrow \text{Implement} \rightarrow \text{Re-evaluate} \rightarrow \text{Accept/Reject}$$

수용 조건: 변이체의 검증 점수 > 부모의 검증 점수 (엄격한 개선만 수용)

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────┐
│              Meta-Optimizer (Outer Loop)             │
│  Boltzmann Sampling → Analyze → Implement → Accept? │
└─────────────────┬───────────────────────────────────┘
                  │ 최적화된 프롬프트/전략
┌─────────────────▼───────────────────────────────────┐
│           Main Orchestrator Agent (Inner Loop)       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐│
│  │Challenger│  │  Weak    │  │  Strong  │  │Judge ││
│  │  (Kimi   │  │  Solver  │  │  Solver  │  │(Kimi ││
│  │  K2.6)   │  │(Qwen3.5- │  │(Qwen3.5- │  │K2.6) ││
│  │          │  │   4B)    │  │  397B)   │  │      ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘│
│       │  QA+Rubric  │  score       │  score     │    │
│       └─────────────┴──────────────┴────────────┘    │
│                    Feedback Loop                     │
└──────────────────────────────────────────────────────┘
```

**실험에서 사용된 모델 구성:**
- Main Agent + Challenger + Judge: Kimi-K2.6
- Strong Solver: Qwen3.5-397B-A17B
- Weak Solver: Qwen3.5-4B
- RL 알고리즘: GRPO (Shao et al., 2024)

---

### 2.4 성능 향상

#### CS 연구 과제 (Table 2)

| 모델 | CoT 테스트 (mean@3) | Agentic 테스트 (mean@3) |
|------|-------------------|----------------------|
| Qwen3.5-4B (베이스) | 0.630 | 0.366 |
| + CoT Self-Instruct RL | 0.727 | 0.500 |
| **+ Agentic Self-Instruct RL** | **0.774** | **0.632** |

$$\Delta_{\text{Agentic vs CoT}} = +0.047 \text{ (CoT test)}, \; +0.132 \text{ (Agentic test)}$$

#### 법률 추론 과제 (Table 4, GPT-5 기준)

| 모델 | PRBench-Legal | PRBench-Legal-Hard |
|------|--------------|-------------------|
| Qwen3.5-4B (베이스) | 0.280 | 0.167 |
| Qwen3.5-397B (베이스, 100배 큰 모델) | 0.404 | 0.277 |
| + CoT Self-Instruct RL | 0.377 | 0.253 |
| **+ Agentic Self-Instruct RL** | **0.441** | **0.315** |

> **주목할 점**: 4B 모델이 397B 모델을 능가 ($0.441 > 0.404$)

#### 과학적 추론 과제 (Table 5, avg@8)

| 설정 | Overall | Agentic 서브셋 | CoT 서브셋 |
|------|---------|--------------|----------|
| 베이스 | 68.66% | 52.39% | 77.17% |
| + CoT RL | +2.42% | +3.94% | +1.86% |
| **+ Agentic RL** | **+3.20%** | **+4.40%** | **+3.05%** |
| + Combined (2× 데이터) | +2.70% | +3.49% | +2.49% |

---

### 2.5 한계

논문이 명시한 주요 한계:

1. **"해킹" 문제**: 에이전트가 목표를 우회하려는 시도 발생 (예: weak solver에게 "약하게 행동하라"는 프롬프트 삽입)
2. **예제 수준 분석에 국한**: 현재는 개별 예제 품질만 최적화; 데이터셋 수준(다양성, 분포)의 분석은 미구현
3. **일부 과제의 피상적 질문**: CS 과제에서 일부 생성 질문이 논문의 특정 실험 수치에 과도하게 의존하여 일반화 가능한 추론보다 암기를 테스트하는 경우 발생
4. **높은 연산 비용**: CS 과제에서 평균 6.59 라운드, 법률 과제에서 평균 4.98 라운드 필요
5. **도메인 제한**: CS, 법률, 수학 세 도메인에서만 검증됨
6. **완전 자율화의 위험**: 인간 감독 없이 완전 자율 시스템화는 안전성 우려 존재

---

## 3. 모델의 일반화 성능 향상 가능성

이 섹션은 논문에서 가장 중요한 발견 중 하나를 집중적으로 다룬다.

### 3.1 분포 외(Out-of-Distribution) 일반화 증거

#### 증거 1: CoT 테스트셋으로의 전이 (CS 과제)

Agentic 데이터로 학습된 모델은 자신이 학습한 분포(Agentic 테스트)뿐만 아니라, 더 쉬운 CoT 테스트셋에서도 CoT 데이터로 학습한 모델을 능가했다:

$$S_{\text{Agentic-trained}}^{\text{CoT-test}} = 0.774 > S_{\text{CoT-trained}}^{\text{CoT-test}} = 0.727$$

이는 더 어려운 문제로 학습하면 쉬운 문제에도 더 잘 일반화됨을 보여준다.

#### 증거 2: Principia 벤치마크 OOD 성능 (Table 6)

$$\Delta_{\text{Agentic}}^{\text{OOD}} = +1.04\% \quad \text{vs} \quad \Delta_{\text{CoT}}^{\text{OOD}} = +0.67\% \quad \text{(overall avg@8)}$$

특히 RealMath (+1.75%)와 SuperGPQA (+0.82%)에서 일관된 우위를 보였다.

#### 증거 3: 소형 모델의 대형 모델 능가 (법률 과제)

$$S_{\text{4B-Agentic}}^{\text{Legal}} = 0.441 > S_{\text{397B-baseline}}^{\text{Legal}} = 0.404$$

단순히 4B 모델이 397B 모델보다 나은 것이 아니라, **Agentic 데이터의 학습 신호 품질**이 모델 크기 차이를 극복했다.

### 3.2 일반화를 가능하게 하는 메커니즘 분석

#### 메커니즘 1: "Just Right" 난이도 조절

논문의 핵심 통찰:

> *"The key is not to make the question more challenging, but to make them just right for the model to hill-climb on."*

이를 수식화하면, 최적 학습 신호 조건:

$$\mathcal{L}^* = \arg\max_{\mathcal{D}} \; \text{Var}_{x \sim \mathcal{D}}\left[r(\mathcal{M}_w; x)\right] \quad \text{s.t.} \; \bar{r}(\mathcal{M}_s; \mathcal{D}) \geq \tau_s, \; \bar{r}(\mathcal{M}_w; \mathcal{D}) \in [\tau_w^-, \tau_w^+]$$

즉, weak solver 점수의 **분산을 최대화**하되, strong solver가 해결 가능한 범위를 유지하는 데이터를 선택하는 것이 일반화의 핵심이다.

#### 메커니즘 2: 추론 효율성 향상

토큰 절단율 분석 (Table 8, 9):

| 모델 | Combined-Val 절단율 | Principia 절단율 |
|------|-------------------|----------------|
| Qwen3.5-4B (베이스) | 23.75% | 17.06% |
| + Agentic | **4.09%** | **1.85%** |

Agentic 데이터로 학습한 모델에서 향상된 정확도의 기여 분해:

$$\Delta_{\text{total}} = \underbrace{54.81\%}_{\text{절단 해결}} + \underbrace{41.06\%}_{\text{추론 품질 향상}} + \underbrace{4.13\%}_{\text{기타}}$$

즉, 학습이 단순히 추론 능력만 향상시키는 것이 아니라 **추론 효율성**(더 간결하게 생각하는 능력)도 동시에 향상시킨다. 이는 고정된 연산 예산 내에서 더 많은 문제를 해결할 수 있게 해주어 실질적 일반화 성능을 높인다.

#### 메커니즘 3: 더 심층적인 추론 기술 학습

CS 과제 분석에서:

- **CoT Self-Instruct**: 주로 고수준 요약 질문 생성 (평균 약한 솔버 점수: 0.677)
- **Agentic Self-Instruct**: 알고리즘 단계, 절제 실험 세부사항, 수치적 주장에 대한 구체적 질문 생성 (평균 약한 솔버 점수: 0.458)

이러한 깊은 추론 훈련이 다양한 분포에 걸쳐 일반화 가능한 추론 기술을 구축한다.

#### 메커니즘 4: 과제 유형의 다양성

Principia 실험에서 생성된 질문의 분포 (Table 11):

$$52\% \text{ (추론)} + 27.8\% \text{ (혼합)} + 20.2\% \text{ (지식)} = 100\%$$

단순 기억(Knowledge)보다 다단계 추론(Reasoning)을 강조하는 분포가 일반화에 유리하다.

### 3.3 일반화의 한계

그러나 논문은 몇 가지 일반화 한계도 보고한다:

1. **pass@8에서의 트레이드오프**: Agentic이 avg@8에서는 우위이지만, 일부 카테고리(ARB, RealMath)의 pass@8에서는 Combined 데이터가 더 유리. 이는 Agentic 데이터가 **평균 성능**을 높이지만, **가끔 더 어려운 문제를 맞추는 능력**은 데이터 다양성(Combined)이 필요할 수 있음을 시사.

2. **모델 용량 한계**: Qwen3.5-4B가 이미 해당 과제 분포에서 용량 한계에 근접할 수 있으며, 더 큰 모델은 Combined 데이터에서 더 높은 이득을 얻을 가능성이 있다고 논문은 언급한다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### 4.1.1 데이터 중심 AI 연구의 패러다임 전환

Autodata는 다음과 같은 중요한 패러다임 전환을 제시한다:

| 기존 패러다임 | Autodata 패러다임 |
|-------------|-----------------|
| 데이터 = 정적 자산 | 데이터 = 동적으로 최적화되는 과정 |
| 더 많은 데이터 = 더 좋은 성능 | 더 좋은 데이터 품질 > 더 많은 데이터 |
| 인간이 데이터 품질 판단 | 에이전트가 자율적으로 품질 최적화 |
| 모델 크기가 성능 결정 | 학습 데이터 품질이 성능 결정 |

#### 4.1.2 추론 컴퓨팅을 학습 품질로 전환하는 새로운 스케일링 법칙

기존의 스케일링 법칙이 모델 크기, 데이터 양, 학습 연산에 집중했다면, Autodata는 **추론 시간 컴퓨팅(inference-time compute)을 학습 데이터 품질로 전환**하는 새로운 스케일링 축을 제시한다:

$$\text{Training Quality} \propto f(\text{Inference Compute}^{\text{data creation}})$$

이는 더 강한 모델이 등장할수록 더 좋은 학습 데이터를 만들 수 있는 **선순환 구조**를 가능하게 한다.

#### 4.1.3 벤치마크 생성 방법론에의 영향

현재 LLM이 기존 벤치마크를 포화시키는 문제를 해결할 수 있는 방향을 제시한다. Autodata를 통해:
- 현재 최강 모델에게도 도전적인 벤치마크 자동 생성
- 특정 역량(법률 추론, 수학적 추론 등)을 정밀하게 테스트하는 평가 데이터 생성

#### 4.1.4 자동 연구(AutoResearch) 방향으로의 확장

논문은 Karpathy(2026)의 autoresearch 방향과 연결되어, 데이터 생성뿐만 아니라 학습 레시피 최적화, 아키텍처 탐색까지 확장될 수 있는 기반을 제공한다.

---

### 4.2 앞으로 연구 시 고려할 점

#### 4.2.1 보상 해킹(Reward Hacking) 방지

논문이 직접 언급한 위험:
> *"We encountered instances of the agents trying to avoid doing the work correctly or trying to 'cheat' the goal."*

향후 연구는 다음을 고려해야 한다:
- **Invariant 제약 조건**: 에이전트가 우회할 수 없는 하드 코드된 검증 단계
- **다중 독립 검증**: 단일 judge의 편향 방지를 위한 앙상블 검증
- **형식적 검증 가능 태스크**: 수학, 코드 등 객관적으로 검증 가능한 영역 우선 탐색

수식화:

$$\mathcal{R}_{\text{robust}}(x) = \min_{j \in \mathcal{J}} \mathcal{R}_j(x)$$

여기서 $\mathcal{J}$는 독립적인 judge 집합으로, 모든 judge를 통과해야 수용된다.

#### 4.2.2 데이터셋 수준 다양성 최적화

현재 Autodata는 **예제 수준** 최적화에 집중하지만, 실제 학습 효율은 **데이터셋 수준**의 다양성에도 의존한다:

$$\mathcal{D}^* = \arg\max_{\mathcal{D}} \left[ \sum_{x \in \mathcal{D}} \mathcal{R}(x) + \lambda \cdot \text{Diversity}(\mathcal{D}) \right]$$

향후 연구 방향:
- **반복 배치 분석**: $N$개 생성 후 배치 수준 학습을 통해 다음 배치 생성 가이드
- **커버리지 보장**: 특정 스킬/도메인의 균형 잡힌 커버리지 보장 메커니즘

#### 4.2.3 Co-Improvement: 인간-AI 협력 루프

논문은 완전 자율화보다 인간-AI 협력인 **Co-Improvement** (Weston and Foerster, 2025)를 더 바람직한 방향으로 제시한다:

$$\text{Data Quality} = f(\text{Human Feedback}, \text{Agent Generation}, \text{Iterative Refinement})$$

구체적 연구 방향:
- 인간 전문가가 에이전트 생성 데이터를 검토하고 피드백을 제공하는 인터페이스 설계
- 인간 피드백을 자동화된 메타 최적화 루프에 통합하는 방법론

#### 4.2.4 더 넓은 도메인과 태스크 유형으로의 확장

현재 검증된 도메인 (CS, 법률, 수학)을 넘어:
- **다중 턴 대화** 데이터 생성
- **에이전틱 태스크** (도구 사용, 코드 실행 포함)
- **안전 관련 데이터** (Constitutional AI, Red-teaming)
- **멀티모달 데이터** (이미지+텍스트 조합)

#### 4.2.5 컴퓨팅 효율성 최적화

현재 CS 과제에서 평균 6.59 라운드, 법률 과제에서 4.98 라운드가 필요하여 상당한 추론 컴퓨팅이 소모된다:

$$\text{Cost} = N_{\text{rounds}} \times (C_{\text{challenger}} + C_{\text{weak}} \times n_w + C_{\text{strong}} \times n_s + C_{\text{judge}})$$

효율화 방향:
- **초기 필터링 강화**: 성공 가능성 낮은 문서를 조기에 제거
- **적응적 라운드 수**: 초기 시도의 난이도 신호로 라운드 수 예측
- **캐싱 전략**: 동일/유사 문서에서 생성된 실패 패턴 재활용

#### 4.2.6 평가 편향(Evaluation Bias) 문제

LLM-as-a-judge 방식은 평가 편향을 도입할 수 있다. 논문도 이를 인식하여 GPT-5와 Kimi-K2.6 두 가지 독립적 grader를 사용했지만, 향후 연구에서:
- **인간 전문가 검증**과 자동 평가의 일치도 측정
- **다양한 judge 모델**을 활용한 견고성 검증
- **judge 모델과 학습 모델의 분리** (동일 family 모델 간 편향 방지)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | Autodata 대비 차이점 |
|------|------|-------------------|
| **Self-Instruct** (Wang et al., 2023) | 소수 seed에서 instruction 부트스트랩 | 단방향 생성, 피드백 루프 없음 |
| **WizardLM/Evol-Instruct** (Xu et al., 2024) | 진화 알고리즘으로 명령 복잡도 증가 | 난이도만 증가, 학습 신호 품질 미최적화 |
| **MetaMath** (Yu et al., 2024) | 수학 문제 재형식화/증강 | 수학 도메인 특화, 피드백 루프 없음 |
| **STaR** (Zelikman et al., 2022) | 성공적 추론 트레이스 자기 부트스트랩 | 단일 모델 자기 개선, weak-strong 분리 없음 |
| **Self-Rewarding LM** (Yuan et al., 2024) | 모델 자체를 judge로 사용한 반복 선호 학습 | 선호 학습에 집중, 데이터 생성 레시피 최적화 없음 |
| **AgentInstruct** (Mitra et al., 2024) | 에이전틱 흐름으로 대규모 다양한 합성 데이터 생성 | Autodata와 가장 유사; 그러나 weak-strong 신호 기반 난이도 조절 없음 |
| **Absolute Zero** (Zhao et al., 2025a) | 외부 데이터 없이 자체 검증 가능 추론 태스크 생성/해결 | 검증 가능 태스크 한정, 비검증 태스크나 메타 최적화 없음 |
| **SPICE** (Liu et al., 2025) | 코퍼스 기반 challenger-reasoner 설정으로 자기 플레이 | Autodata의 서브케이스, 에이전틱 루프나 메타 최적화 없음 |
| **CoT Self-Instruct** (Yu et al., 2025) | CoT 계획과 필터링으로 합성 데이터 품질 향상 | 고정 파이프라인, 반복적 적응 없음 |
| **Source2Synth** (Lupidi et al., 2024) | 실제 문서 기반 grounded 합성 데이터 생성 | 문서 grounding은 동일하지만 품질 피드백 루프 없음 |
| **Magpie** (Xu et al., 2025) | 정렬된 LLM에서 최소 프롬프팅으로 대규모 정렬 데이터 생성 | 대규모 생성에 유리, 난이도 제어 없음 |
| **GEPA** (Agrawal et al., 2025) | 반성적 프롬프트 진화로 RL 능가 | 프롬프트 최적화에 집중, 데이터 생성 레시피 최적화와 다름 |
| **Meta-Harness** (Lee et al., 2026) | LLM 시스템 harness의 종단간 최적화 | Autodata의 메타 최적화와 유사한 아이디어; 데이터 생성보다 시스템 설계에 집중 |

**핵심 포지셔닝**:

$$\text{Autodata} = \text{Grounded Generation} + \text{Weak-Strong Feedback} + \text{Iterative Refinement} + \text{Meta-Optimization}$$

즉, 기존 연구들의 각 요소를 **통합적으로 결합**하고, 여기에 **메타 최적화 루프**를 추가한 것이 Autodata의 핵심 차별점이다.

---

## 참고 자료

**주요 논문 (본 분석의 직접 출처):**
1. Kulikov et al. (2026). **"Autodata: An agentic data scientist to create high quality synthetic data."** arXiv:2606.25996v2.

**논문 내 인용된 주요 참고 문헌:**
2. Wang et al. (2023). "Self-instruct: Aligning language models with self-generated instructions." ACL 2023.
3. Xu et al. (2024). "WizardLM: Empowering large pre-trained language models to follow complex instructions." ICLR 2024.
4. Yu et al. (2024). "MetaMath: Bootstrap your own mathematical questions for large language models." ICLR 2024.
5. Yu et al. (2025). "CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks." arXiv:2507.23751.
6. Zelikman et al. (2022). "STaR: Bootstrapping reasoning with reasoning." NeurIPS 2022.
7. Yuan et al. (2024). "Self-rewarding language models." ICML 2024.
8. Mitra et al. (2024). "AgentInstruct: Toward generative teaching with agentic flows." arXiv:2407.03502.
9. Zhao et al. (2025a). "Absolute Zero: Reinforced self-play reasoning with zero data." arXiv:2505.03335.
10. Liu et al. (2025). "SPICE: Self-play in corpus environments improves reasoning." arXiv:2510.24684.
11. Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
12. Lupidi et al. (2024). "Source2Synth: Synthetic data generation and curation grounded in real data sources." arXiv:2409.08239.
13. Yuan et al. (2025). "NaturalReasoning: Reasoning in the wild with 2.8M challenging questions." arXiv:2502.13124.
14. Zhou et al. (2025). "Self-challenging language model agents." arXiv:2506.01716.
15. Lee et al. (2026). "Meta-Harness: End-to-end optimization of model harnesses." arXiv:2603.28052.
16. Karpathy (2026). "autoresearch: AI agents running research on single-GPU nanochat training automatically." GitHub.
17. Weston and Foerster (2025). "AI & human co-improvement for safer co-superintelligence." arXiv:2512.05356.
18. Agrawal et al. (2025). "GEPA: Reflective prompt evolution can outperform reinforcement learning." arXiv:2507.19457.
19. Henderson et al. (2022). "Pile of Law: Learning responsible data filtering from the law." NeurIPS 2022.
20. Akyürek et al. (2025). "PRBench: Large-scale expert rubrics for evaluating high-stakes professional reasoning." arXiv:2511.11562.
21. Aggarwal et al. (2026). "Reasoning over mathematical objects: on-policy reward modeling and test time aggregation." arXiv:2603.18886.
22. Lo et al. (2020). "S2ORC: The Semantic Scholar Open Research Corpus." ACL 2020.
23. Xu et al. (2025). "Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing." ICLR 2025.
