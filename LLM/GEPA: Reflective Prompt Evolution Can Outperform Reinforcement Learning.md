
# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

> **논문 정보**
> - **제목:** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
> - **arXiv ID:** [2507.19457](https://arxiv.org/abs/2507.19457) (v1: 2025.07.25, v2: 2026.02.14)
> - **학회:** ICLR 2026 **(Oral 채택)**
> - **저자:** Lakshya A. Agrawal 외 16인 (UC Berkeley, Stanford, MIT, Databricks 등)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

이 논문은 언어의 해석 가능한(interpretable) 특성이 희소(sparse)하고 스칼라(scalar)적인 보상으로 얻는 정책 경사(policy gradient)보다 LLM에게 훨씬 풍부한 학습 매체를 제공한다고 주장하며, 이를 검증하기 위해 **GEPA(Genetic-Pareto)**라는 프롬프트 옵티마이저를 제안한다.

### 🔑 핵심 주장 요약

| 주장 | 설명 |
|---|---|
| **언어 > 스칼라 보상** | RL의 스칼라 보상보다 자연어 반성(reflection)이 LLM 학습에 더 풍부한 신호 제공 |
| **샘플 효율성 우위** | RL(GRPO) 대비 최대 35배 적은 rollout으로 동등 이상의 성능 달성 |
| **SOTA 성능** | GRPO 및 기존 최고 프롬프트 옵티마이저 MIPROv2를 동시에 능가 |

### 📌 주요 기여

1. **GEPA 알고리즘 제안**: 유전적 프롬프트 진화(genetic prompt evolution), 자연어 피드백을 활용한 반성(reflection), 파레토 기반 후보 선택(Pareto-based candidate selection)의 세 가지 핵심 원리를 결합한 복합 AI 시스템용 샘플 효율적 옵티마이저를 소개한다.

2. **성능 달성**: 6개 태스크에 걸쳐 GRPO 대비 평균 6%, 최대 20% 성능 향상을 이루며 최대 35배 적은 rollout을 사용한다. 또한 선도적 프롬프트 옵티마이저인 MIPROv2를 10% 이상 능가(예: AIME-2025에서 +12% 정확도)하며, 코드 최적화를 위한 추론 시점(inference-time) 탐색 전략으로서도 유망한 결과를 보인다.

3. **새로운 학습 패러다임 제시**: 언어를 단순한 인터페이스가 아닌 학습과 반성의 주요 매체로 취급함으로써, 기존 접근보다 샘플 효율적이고 비용 효과적인 방법을 개발하였다.

---

## 2. 상세 분석: 문제 정의 → 제안 방법 → 모델 구조 → 성능 → 한계

### 2-1. 해결하고자 하는 문제

기존 RL 기반 방법(예: GRPO)은 수천 번의 보상 기반 rollout으로 가중치를 조정하지만, 풍부한 순차적 행동 정보를 단일 보상값으로 압축함으로써 언어의 자연스러운 표현력을 활용하지 못한다는 한계가 있다.

기존 옵티마이저(RL, 진화 전략)는 풍부한 실행 추적(execution trace)을 단일 스칼라 보상으로 축소하여 후보가 왜 실패했는지를 알지 못한다. GEPA는 다른 접근을 취하는데, 평가자가 실행 가능한 부가 정보(ASI: Actionable Side Information) — 오류 메시지, 프로파일링 데이터, 추론 로그 — 를 반환하면 LLM이 이 피드백을 읽어 실패를 진단하고 표적화된 수정을 제안한다.

---

### 2-2. 제안하는 방법 (수식 포함)

GEPA는 세 가지 핵심 원리를 결합한다: **(1) 유전적 프롬프트 진화**, **(2) 자연어 피드백을 활용한 반성**, **(3) 파레토 기반 후보 선택**.

#### (1) 유전적 프롬프트 진화 (Genetic Prompt Evolution)

GEPA는 새로운 rollout으로부터 도출된 자연어 피드백을 바탕으로 AI 시스템 내 모든 프롬프트를 반복적으로 변이시킨다. 각 변이에서 후보 프롬프트는 조상(ancestor)으로부터 파생되어, 관찰 및 LLM 피드백에서 추출된 고수준 교훈을 누적한다.

최적화 목표는 다음과 같이 표현할 수 있다:

$$
\pi^* = \arg\max_{\pi \in \Pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathcal{M}(f_\pi(x), y) \right]
$$

여기서:
- $\pi$: 현재 프롬프트 후보 (텍스트 공간 $\Pi$ 내)
- $\mathcal{D}$: 태스크 데이터셋
- $f_\pi(x)$: 프롬프트 $\pi$를 사용한 LLM의 출력
- $\mathcal{M}$: 평가 메트릭 (스칼라 점수)

각 반복(iteration)에서 **변이(mutation)**는 다음과 같이 이루어진다:

$$
\pi_{t+1} = \text{Mutate}(\pi_t,\; \text{Reflect}(\tau_t, \mathcal{F}_t))
$$

여기서:
- $\tau_t$: 시스템 실행 궤적(trajectory): 추론 단계, 도구 호출, 출력 등
- $\mathcal{F}_t$: ASI(Actionable Side Information) — 자연어 피드백
- $\text{Reflect}(\cdot)$: 궤적과 피드백으로부터 고수준 규칙을 도출하는 반성 단계
- $\text{Mutate}(\cdot)$: 반성 결과를 바탕으로 새 프롬프트를 생성하는 단계

#### (2) 파레토 기반 후보 선택 (Pareto-Based Candidate Selection)

탐욕적(greedy) 프롬프트 업데이트가 야기하는 지역 최적값(local optima)을 피하기 위해, GEPA는 파레토 전선(Pareto front)을 유지한다: 전역적으로 최선인 프롬프트만 진화시키는 대신, 각 문제 인스턴스별 최고 성능 프롬프트를 확률적으로 탐색하여 전략을 다양화하고 강건한 일반화를 촉진한다.

파레토 전선 $\mathcal{F}_{pareto}$는 다음과 같이 정의된다:

```math
\mathcal{F}_{pareto} = \left\{ \pi_i \in \mathcal{C} \;\middle|\; \nexists\; \pi_j \in \mathcal{C} : \forall x_k,\; \mathcal{M}(f_{\pi_j}(x_k)) \geq \mathcal{M}(f_{\pi_i}(x_k)) \text{ and } \pi_j \neq \pi_i \right\}
```

여기서:
- $\mathcal{C}$: 전체 후보 프롬프트 풀(pool)
- $\pi_i$: i번째 후보 프롬프트
- $x_k$: k번째 훈련 인스턴스

GEPA의 주요 후보 선택 전략은 모든 태스크 인스턴스에 걸쳐 파레토 전선에서 지배받지 않는(non-dominated) 후보를 찾고, 파레토 전선에서의 등장 빈도에 따라 그 중 하나를 확률적으로 선택하는 것이다.

후보 선택 확률은 다음과 같이 표현할 수 있다:

$$
P(\pi_i) = \frac{\text{freq}(\pi_i, \mathcal{F}_{pareto})}{\sum_{\pi_j \in \mathcal{F}_{pareto}} \text{freq}(\pi_j, \mathcal{F}_{pareto})}
$$

여기서 $\text{freq}(\pi_i, \mathcal{F}_{pareto})$는 파레토 전선에서 $\pi_i$가 최고 성능을 보이는 인스턴스 수이다.

#### (3) 시스템 인식 병합 (System-Aware Merge)

각 변이는 탐색 트리의 모든 조상으로부터 누적된 교훈을 상속한다. GEPA는 또한 서로 다른 태스크에서 탁월한 두 파레토 최적 후보의 강점을 결합하는 **시스템 인식 병합(system-aware merge)**을 지원한다.

병합 연산은 다음과 같이 표현된다:

$$
\pi_{merged} = \text{Merge}(\pi_A, \pi_B, \{\tau_A\}, \{\tau_B\})
$$

여기서 $\pi_A$는 인스턴스 서브셋 $S_A$에 강하고, $\pi_B$는 $S_B$에 강한 두 파레토 최적 후보이다.

---

### 2-3. 모델 구조 (Architecture)

GEPA는 세 가지 핵심 혁신을 결합한 정교한 최적화 루프를 통해 작동한다: 유전적 프롬프트 진화, 자연어 반성, 파레토 기반 후보 선택. 시스템은 후보 LLM 프로그램의 풀을 유지하며, 각각은 모듈형 컴포넌트에 대한 최적화된 프롬프트의 서로 다른 구성을 나타낸다.

```
┌─────────────────────────────────────────────────────────┐
│                   GEPA 최적화 루프                       │
│                                                         │
│  [후보 풀 C]                                            │
│       │                                                 │
│       ▼                                                 │
│  [파레토 후보 선택]  ← 파레토 전선 F_pareto             │
│       │                                                 │
│       ▼                                                 │
│  [롤아웃 실행]  → 궤적 τ (추론, 도구 호출, 출력)        │
│       │                                                 │
│       ▼                                                 │
│  [자연어 반성 + ASI 분석]                               │
│    ├─ 문제 진단                                         │
│    ├─ 개선 규칙 도출                                    │
│    └─ Feedback F_t 생성                                 │
│       │                                                 │
│       ▼                                                 │
│  [변이(Mutate) / 병합(Merge)]                           │
│       │                                                 │
│       ▼                                                 │
│  [새 후보 π_{t+1} 평가 → C 갱신]                       │
│       │                                                 │
│       └──────────────── (반복) ─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

GEPA는 LM의 DSPy 프로그램 궤적에 대한 반성 능력을 활용하여, 무엇이 잘 됐고 무엇이 안 됐으며 무엇을 개선할 수 있는지를 식별한다. 이 반성에 기반하여 GEPA는 새로운 프롬프트를 제안하고, 진화된 프롬프트 후보들의 트리를 구축하며 최적화가 진행될수록 개선 사항을 누적한다. GEPA는 스칼라 메트릭만이 아닌 도메인 특화 텍스트 피드백을 활용할 수 있기 때문에, 매우 적은 rollout 수로 고성능 프롬프트를 제안할 수 있다.

---

### 2-4. 성능 향상

#### 📊 주요 벤치마크 결과

GEPA는 HotpotQA, IFBench, HoVer, PUPA의 네 벤치마크에서 GRPO(24,000 rollouts + LoRA) 대비 최대 19%를 능가하면서, 각각 6,438 / 678(35배 적음) / 6,858 / 2,157(11배 적음) 회의 rollout만 사용하여 최적 테스트 성능을 달성한다. 특히 GEPA는 각각 402, 330, 1,179, 306회의 rollout만으로 GRPO의 최고 검증 점수를 달성하여 최대 78배 높은 샘플 효율성을 보인다.

GPT-4.1 Mini 기준 전체 태스크 집계 개선: **+14.29%** (기준 대비), Qwen3 8B 기준 **+12.44%** — MIPROv2의 개선 폭(각각 +7.04%, +6.26%)을 크게 상회한다.

| 비교 대상 | 평균 성능 개선 | 최대 성능 개선 | Rollout 비율 |
|---|---|---|---|
| vs. GRPO | +6~10% | +20% | **최대 1/35** |
| vs. MIPROv2 | +10~14% | +12% (AIME) | 동등 예산 |
| GEPA+Merge vs. GRPO | +21% | — | 동등 예산 |

**비용 효율**: GEPA는 주요 벤치마크에서 400~1,200회의 rollout만 필요한 반면, RL은 일반적으로 24,000회 이상이 필요하여 비용과 연산량을 극적으로 절감한다.

**주목할 만한 발견**: GEPA의 instruction-only 최적화가 instruction과 few-shot 예시를 동시에 최적화하는 MIPROv2의 공동 최적화를 능가한다. 이는 LLM이 복잡한 지시를 따르는 능력이 향상됨에 따라, 세밀하고 반성적인 instruction 집합을 진화시키는 것이 in-context 예시를 큐레이션하는 것보다 더 강력하고 효율적인 전략일 수 있음을 시사한다.

**코드 최적화 적용**: GEPA를 추론 시점 탐색 전략으로 사용할 때도 유망한 예비 결과를 제시한다. CUDA 및 NPU 커널 생성에 적용했을 때, GEPA는 컴파일러 피드백을 기반으로 코드를 반복적으로 개선하여 강력한 기준선 대비 유의미한 성능 향상을 달성했다.

---

### 2-5. 한계 (Limitations)

프롬프트 기반 학습과 전통적인 가중치 기반 파인튜닝 사이의 경계는 여전히 열린 문제로 남아 있으며, 데이터가 풍부한(data-abundant) 환경에서는 전체 파인튜닝이 여전히 우위를 가질 수 있다.

LoRA나 RLHF 같은 가중치 공간 적응(weight-space adaptation)의 부재는 GEPA가 깊은 행동 조정(fine-tuning)에 제한을 가질 수 있다. 또한 few-shot 예시 최적화를 지원하지 않기 때문에 패턴 시연에 의존하는 태스크에서 덜 효과적일 수 있으며, 그래디언트 기반 업데이트 없이 시간에 따른 미세 제어나 안정성이 어떻게 관리되는지의 문제도 있다.

또한 few-shot 예시 최적화를 통합하거나 시스템 추적에서 가장 가치 있는 학습 신호를 추출하기 위한 더 정교한 "피드백 엔지니어링(feedback engineering)"을 개발함으로써 GEPA를 더욱 개선할 수 있다고 논문은 제안한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 메커니즘

지역 최적값(local optima)을 피하기 위해, GEPA는 파레토 전선을 유지한다: 전역 최선 프롬프트만 진화시키는 대신 각 문제 인스턴스별 최고 성능 프롬프트를 확률적으로 탐색하여, 전략을 다양화하고 강건한 일반화를 촉진하며 지역 최솟값에 빠지는 것을 완화한다.

**검증된 일반화 우수성**: GEPA로 최적화된 프롬프트는 검증 셋과 테스트 성능 사이의 일반화 격차(generalization gap)가 더 작아, 보이지 않는 데이터로의 전이가 더 강건한 학습임을 나타낸다.

### 3-2. 일반화를 높이는 세 가지 설계 요인

#### ① 파레토 기반 다양성 유지

파레토 기반 선택은 다양성을 유지하고 일반화를 촉진하여, 도메인 외(out-of-domain) 평가에서도 GEPA가 강건하도록 만든다.

$$
\text{Generalization Gap} = \mathcal{M}_{val}(\pi^*) - \mathcal{M}_{test}(\pi^*)
$$

GEPA는 이 일반화 격차를 최소화하는 방향으로 설계되었다.

#### ② 고수준 규칙 학습

LLM의 언어 특성이 sparse한 스칼라 보상에서 도출된 정책 경사보다 훨씬 풍부한 학습 매체를 제공한다는 주장 하에, GEPA는 시행착오(trial and error)로부터 고수준 규칙을 학습하는 자연어 반성을 철저히 통합한 프롬프트 옵티마이저이다.

고수준 규칙은 특정 샘플에 과적합(overfit)되기보다 태스크의 일반적 구조를 이해함으로써 일반화 성능 향상에 기여한다.

#### ③ 누적 교훈(Accumulated Lessons)

각 변이는 탐색 트리의 모든 조상으로부터 누적된 교훈을 상속한다.

이 누적 학습 메커니즘은 다음과 같이 표현된다:

$$
\text{Lessons}_t = \bigcup_{i=0}^{t-1} \text{Reflect}(\tau_i, \mathcal{F}_i)
$$

즉, 시간이 지남에 따라 누적된 교훈이 쌓이며, 이는 일반화 가능한 지식으로 귀결된다.

### 3-3. 일반화 관련 실험 결과

논문은 Wan et al. (2024)이 제안한 검증 셋과 테스트 셋 성능 차이인 일반화 격차(generalization gap)에 대한 연구를 재수행한다.

최근의 증거들이 few-shot 예시 최적화의 효과를 지지함에도 불구하고, 연구 결과는 LLM의 instruction-following 및 자기 반성(self-reflective) 능력의 발전과 이러한 향상된 능력을 활용한 GEPA의 설계 선택 덕분에 이 트렌드에서 흥미로운 전환이 나타남을 보여준다.

GEPA는 프롬프트 길이를 최대 9.2배 줄이고 일반화를 향상시켜, 자원 제약 환경 및 추론 시점 코드 최적화 태스크에도 적합하다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

#### 🔄 패러다임 전환 신호

이 연구는 LLM 기반 에이전트를 최적화하는 방법에 있어 잠재적 패러다임 전환을 신호한다.

GEPA는 언어를 사용하는 시스템은 언어 자체를 통해 학습하고 개선해야 한다고 주장하며, 자연어 진단을 통한 반복적 프롬프트 진화를 제안한다.

#### 📌 구체적 영향 영역

1. **RL 대안 연구 촉진**: GEPA가 자연어 반성으로 프롬프트를 최적화하고 훨씬 적은 rollout으로 GRPO와 MIPROv2를 모두 능가함을 보여줌으로써, RL 없이도 강력한 에이전트를 최적화할 수 있다는 방향의 연구를 자극할 것이다.

2. **복합 AI 시스템 최적화**: GEPA는 복합 AI 시스템(compound AI systems)을 위한 반성적 프롬프트 옵티마이저임을 입증하며, 언어의 해석 가능성을 활용해 시스템 수준 궤적에서 더 풍부한 학습 신호를 추출하고 반복적 프롬프트 변이와 파레토 기반 후보 선택으로 모듈형 LLM 시스템을 최적화한다.

3. **코드 최적화 적용 확장**: GEPA를 추론 시점 탐색 전략으로 활용하는 유망한 예비 결과를 제시하며, CUDA 및 NPU 커널 생성에서 컴파일러 피드백 기반 코드 반복 개선으로 강력한 기준선 대비 유의미한 성능 향상을 달성한다.

4. **MLflow 등 프로덕션 통합**: GEPA는 MLflow의 `mlflow.genai.optimize_prompts()` API에 통합되어 평가 메트릭과 훈련 데이터를 활용한 자동 프롬프트 개선을 지원하며, 어떠한 에이전트 프레임워크와도 호환되고 다중 프롬프트 최적화를 지원한다.

---

### 4-2. 향후 연구 시 고려할 점

#### 🔬 (A) 가중치 공간 통합 (Hybrid 접근)

핵심적인 미래 연구 방향은 반성적 프롬프트 진화와 가중치 공간 적응(weight-space adaptation)의 통합이다. GEPA의 언어 기반 인사이트가 더 효율적인 RL 또는 파인튜닝 rollout을 안내하는 하이브리드 접근이 이 패러다임들을 통합하고 더 큰 성능과 효율성으로 이어질 수 있다.

수식으로 표현하면:

$$
\theta^* = \text{RL-FineTune}(\theta_0, \pi^*_{GEPA})
$$

즉, GEPA가 최적화한 프롬프트 $\pi^*_{GEPA}$를 가이드로 삼아 더 효율적인 RL 파인튜닝을 수행하는 방향이다.

#### 🔬 (B) 피드백 엔지니어링 (Feedback Engineering)

few-shot 예시 최적화 통합 또는 시스템 추적에서 가장 가치 있는 학습 신호를 추출하기 위한 더 정교한 "피드백 엔지니어링" 개발이 GEPA를 개선하는 방향으로 제안된다.

#### 🔬 (C) 데이터 풍부 환경에서의 비교

데이터가 풍부한 환경에서는 전통적인 파인튜닝이 여전히 우위를 가질 수 있으므로, 데이터 규모별 GEPA vs. LoRA/RLHF 비교 연구가 필요하다.

#### 🔬 (D) few-shot 예시와의 결합

GEPA가 few-shot 예시 최적화를 지원하지 않기 때문에 패턴 시연에 의존하는 태스크에서 덜 효과적일 수 있다는 점을 고려할 때, instruction 진화와 few-shot 예시 최적화를 결합하는 연구가 유망하다.

#### 🔬 (E) 메타 학습 및 태스크 간 전이

GEPA의 변이 로직이 태스크 및 모듈에 걸쳐 메타 학습이나 추상화를 위한 여지가 있는지 탐색하는 연구도 중요하다.

---

## 5. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 학습 신호 | 주요 특징 | 한계 |
|---|---|---|---|
| **GRPO** (Shao et al., 2024) | 스칼라 보상 | 가중치 업데이트, 강력한 태스크 적응 | 수만 rollout 필요, 비용 高 |
| **MIPROv2** (Opsahl-Ong et al., 2024) | 스칼라 메트릭 | Instruction+few-shot 공동 최적화 | GEPA 대비 10%+ 성능 열세 |
| **TextGrad** (Yuksekgonul et al., 2025) | 자연어 그래디언트 | 텍스트 역전파, 인스턴스 수준 최적화 | 시스템 수준 다중 프롬프트 최적화 어려움 |
| **OPRO** (Google DeepMind, ICLR 2024) | 스칼라 점수 | LLM 자체로 프롬프트 제안 | 단일 프롬프트 태스크, 분산 높음 |
| **APE** | 입출력 예시 | 자동 instruction 생성 | few-shot 최적화 없음 |
| **GEPA** (Agrawal et al., 2025) | 자연어(ASI) | 유전+파레토+반성 통합, 복합 시스템 최적화 | 가중치 업데이트 불가, few-shot 미지원 |

TextGrad(Stanford)는 역전파 개념을 텍스트에 적용하여, 수치 그래디언트 대신 LLM이 출력에 대한 자연어 피드백("텍스트 그래디언트")을 생성하고 이를 활용해 프롬프트나 해결책을 반복적으로 개선한다. 그러나 TextGrad는 다른 LLM으로부터 텍스트 피드백을 역전파하여 복합 시스템을 최적화하는 방식으로 시스템 수준의 다중 모듈 최적화에는 GEPA보다 제약이 있다.

OPRO(Google DeepMind, ICLR 2024)는 10라운드 × 10개 후보 방식으로 각 훈련 셋을 평가하고 점수를 LLM에 컨텍스트로 다시 피드하여 다음 라운드 제안을 받는다. 그러나 OPRO는 few-shot 최적화가 없고 단일 프롬프트 태스크에 제한되며 실행 간 분산이 높다는 약점이 있다.

---

## 참고 자료 및 출처

| # | 자료명 | URL |
|---|---|---|
| 1 | **[arXiv 논문 원문]** GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (2507.19457) | https://arxiv.org/abs/2507.19457 |
| 2 | **[arXiv HTML 전문 v1]** GEPA 논문 HTML 버전 | https://arxiv.org/html/2507.19457v1 |
| 3 | **[arXiv PDF v2 - ICLR 2026 Oral]** GEPA 최종 게재본 | https://arxiv.org/pdf/2507.19457 |
| 4 | **[HuggingFace Papers]** GEPA Paper Page | https://huggingface.co/papers/2507.19457 |
| 5 | **[OpenReview]** ICLR 2026 리뷰 페이지 | https://openreview.net/forum?id=RQm2KQTM5r |
| 6 | **[DSPy 공식 문서]** Reflective Prompt Evolution with GEPA | https://dspy.ai/tutorials/gepa_ai_program/ |
| 7 | **[DSPy API 문서]** dspy.GEPA Overview | https://dspy.ai/api/optimizers/GEPA/overview/ |
| 8 | **[GitHub 공식 레포]** gepa-ai/gepa | https://github.com/gepa-ai/gepa |
| 9 | **[GEPA 공식 프로젝트 페이지]** gepa-ai.github.io | https://gepa-ai.github.io/gepa/ |
| 10 | **[alphaXiv 개요]** GEPA 논문 분석 | https://www.alphaxiv.org/overview/2507.19457 |
| 11 | **[Morph LLM 가이드]** Prompt Optimization 비교 분석 (2026) | https://www.morphllm.com/prompt-optimization |
| 12 | **[Medium 리뷰]** LLM Prompt optimization: Genetic Pareto (GEPA) | https://medium.com/@sulbha.jindal/llm-prompt-optimization-genetic-pareto-gepa-paper-reviw-29ee52d9b5db |
| 13 | **[ArXivIQ Substack]** GEPA 논문 해설 | https://arxiviq.substack.com/p/gepa-reflective-prompt-evolution |
| 14 | **[EmergentMind]** GEPA: Reflective Prompt Evolution in Compound AI | https://www.emergentmind.com/papers/2507.19457 |
| 15 | **[Medium - Superagentic AI]** GEPA: The Game-Changing DSPy Optimizer | https://medium.com/superagentic-ai/gepa-the-game-changing-dspy-optimizer-for-agentic-ai |

> ⚠️ **정확도 참고**: 본 답변에 포함된 수식 중 일부(특히 파레토 전선 공식 및 최적화 목표 수식)는 논문의 설명을 기반으로 표준 수학적 표기법으로 재구성한 것입니다. 논문 원문의 정확한 수식 표기는 [arXiv PDF](https://arxiv.org/pdf/2507.19457)의 Figure 4(전체 GEPA 알고리즘 형식화)에서 직접 확인하시기를 권장합니다.
