# Mixture-of-Agents Enhances Large Language Model Capabilities
---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **LLM의 협력성(Collaborativeness)** 이라는 현상에 기반합니다. 즉, LLM은 다른 모델의 출력을 참조할 때 더 높은 품질의 응답을 생성하는 경향이 있으며, 이는 참조한 모델이 자신보다 낮은 성능을 보이는 경우에도 마찬가지입니다. 이를 기반으로 여러 LLM을 계층적으로 구성하는 **Mixture-of-Agents(MoA)** 프레임워크를 제안합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **새로운 프레임워크** | 다수의 LLM을 계층적으로 조합하는 MoA 프레임워크 제안 |
| **LLM 협력성 발견** | 타 모델의 출력이 저품질이어도 성능이 향상되는 협력성 현상 규명 |
| **SOTA 달성** | AlpacaEval 2.0에서 GPT-4 Omni(57.5%) 대비 65.1% 달성 (오픈소스만 사용) |
| **비용 효율성** | MoA-Lite는 GPT-4o 대비 비용 절감과 동시에 성능 향상 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 1: 단일 모델의 한계**
- 모델 스케일 확대는 수조 개의 토큰 재학습이 필요하며 비용이 매우 큼
- 개별 LLM은 특정 분야(instruction following, code generation 등)에 특화되어 범용성에 한계 존재

**문제 2: 기존 앙상블/멀티에이전트 방법의 한계**
- LLM Ranker: 단순히 가장 좋은 응답을 선택하는 방식 → MoA보다 성능이 낮음
- LLM-Blender(Jiang et al., 2023): GENFUSER 학습이 필요하여 파인튜닝 비용 발생
- 기존 멀티에이전트 토론 방식(Du et al., 2023): 구조적 유연성 부족

**핵심 질문:** *"어떻게 하면 파인튜닝 없이 여러 LLM의 집단 전문성을 활용할 수 있는가?"*

---

### 2-2. 제안하는 방법 (수식 포함)

#### MoA 공식 정의

MoA는 $l$개의 레이어로 구성되며, 각 레이어 $i$는 $n$개의 LLM $A_{i,1}, A_{i,2}, \ldots, A_{i,n}$으로 이루어집니다.

**MoA 레이어의 출력:**

$$y_i = \bigoplus_{j=1}^{n} [A_{i,j}(x_i)] + x_1, \quad x_{i+1} = y_i \tag{1}$$

여기서:
- $x_1$: 초기 입력 프롬프트
- $A_{i,j}(x_i)$: 레이어 $i$의 $j$번째 에이전트 출력
- $+$: 텍스트 연결(concatenation)
- $\bigoplus$: Table 1에 제시된 **Aggregate-and-Synthesize** 프롬프트의 적용
- $x_{i+1} = y_i$: 다음 레이어의 입력으로 전달

**최종 출력:** 마지막 $l$번째 레이어의 단일 LLM 출력 $A_{l,1}(x_l)$

#### MoE와의 비교 (수식 대비)

논문에서 참조하는 **Mixture-of-Experts(MoE)** 공식:

$$y_i = \sum_{j=1}^{n} G_{i,j}(x_i) E_{i,j}(x_i) + x_i \tag{2}$$

여기서:
- $G_{i,j}$: 게이팅 네트워크 출력 (expert $j$에 대한 가중치)
- $E_{i,j}$: expert 네트워크 $j$의 함수

**MoA vs MoE 비교:**

| 구분 | MoE | MoA |
|---|---|---|
| 작동 단위 | 서브네트워크(activation level) | 독립적 LLM(model level) |
| 게이팅 | 별도 게이팅 네트워크 | LLM 자체가 게이팅 역할 수행 |
| 파인튜닝 | 필요 | 불필요 (프롬프팅만 사용) |
| 확장성 | 아키텍처 제약 | 임의의 LLM 적용 가능 |

---

### 2-3. 모델 구조

```
[Prompt x₁]
    │
    ├── A₁,₁ ──┐
    ├── A₁,₂ ──┼── Aggregate-and-Synthesize → y₁ = x₂
    └── A₁,₃ ──┘
                │
    ├── A₂,₁ ──┐
    ├── A₂,₂ ──┼── Aggregate-and-Synthesize → y₂ = x₃
    └── A₂,₃ ──┘
                │
    ├── A₃,₁ ──┐
    ├── A₃,₂ ──┼── Aggregate-and-Synthesize → y₃ = x₄
    └── A₃,₃ ──┘
                │
             A₄,₁ → [Final Output]
```

**두 가지 역할 분류:**

- **Proposer(제안자):** 다양한 관점의 참조 응답 생성. 자체 점수는 낮아도 aggregator의 성능 향상에 기여. (예: WizardLM - proposer로서 우수, aggregator로서 부진)
- **Aggregator(집성자):** 여러 응답을 종합하여 고품질의 단일 출력 생성. (예: GPT-4o, Qwen1.5, LLaMA-3 - 양 역할 모두 우수)

**실험 설정 (Default MoA):**
- **Proposers:** Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, WizardLM-8x22B, LLaMA-3-70B-Instruct, Mixtral-8x22B-v0.1, dbrx-instruct
- **Layer 수:** 3 layers
- **Final Aggregator:** Qwen1.5-110B-Chat

---

### 2-4. 성능 향상

#### AlpacaEval 2.0 결과

| 모델 | LC Win Rate | Win Rate |
|---|---|---|
| **MoA w/ GPT-4o** | **65.7±0.7%** | 78.7±0.2% |
| **MoA (오픈소스만)** | **65.1±0.6%** | 59.8±0.3% |
| MoA-Lite | 59.3±0.2% | 57.0±0.7% |
| GPT-4 Omni (05/13) | 57.5% | 51.3% |
| GPT-4 Turbo (04/09) | 55.0% | 46.1% |

→ 오픈소스만으로 GPT-4 Omni 대비 **+7.6% 절대적 향상**

#### MT-Bench 결과

| 모델 | Avg. | 1st turn | 2nd turn |
|---|---|---|---|
| MoA w/ GPT-4o | 9.40±0.06 | 9.49 | 9.31 |
| GPT-4 Turbo | 9.31 | 9.35 | 9.28 |
| **MoA** | **9.25±0.10** | 9.44 | 9.07 |
| GPT-4 Omni | 9.19 | 9.31 | 9.07 |

#### FLASK 결과
- **우수한 영역:** robustness, correctness, factuality, insightfulness, completeness, metacognition
- **상대적 약점:** conciseness (응답이 다소 장황해지는 경향)

#### MATH 태스크 (Table 8)
레이어 증가에 따른 단조로운 정확도 향상 확인:

$$\text{Llama-3-70B: Layer1} = 0.456 \rightarrow \text{Layer2} = 0.584 \rightarrow \text{Layer3} = 0.578$$

---

### 2-5. 한계점

1. **높은 TTFT(Time to First Token):** 모든 레이어의 aggregation이 완료되어야 첫 토큰 생성 가능 → 실시간 응용에 불리
2. **계산 비용:** 다수 LLM 병렬 호출로 인한 토큰 소비 증가
3. **Verbosity 문제:** 집성 과정에서 응답이 장황해지는 경향 (FLASK conciseness 지표 하락)
4. **벤치마크 편향 가능성:** AlpacaEval 2.0의 GPT-4 기반 평가가 특정 스타일을 선호할 가능성
5. **컨텍스트 윈도우 제약:** 여러 모델의 응답을 연결(concatenate)할 때 입력 길이 제한에 도달할 가능성

---

## 3. 일반화 성능 향상 가능성

### 3-1. 다양성을 통한 일반화

논문에서 **모델 다양성(diversity)** 이 일반화에 핵심적 역할을 합니다.

**Table 3에서 도출된 핵심 결과:**

$$\text{Multiple-Proposer}(n=6) = 61.3\% > \text{Single-Proposer}(n=6) = 56.7\%$$

이는 동일 모델을 여러 번 샘플링하는 것보다, **이질적(heterogeneous) 모델** 조합이 더 높은 일반화 성능을 가져온다는 것을 의미합니다.

수학적으로, 다양한 모델 집합 $\{A_1, A_2, \ldots, A_n\}$의 오류 분포가 서로 독립적(혹은 낮은 상관관계)일 때, 집성 후 오류율은 다음과 같이 감소할 수 있습니다:

$$P(\text{aggregate error}) \ll \min_j P(A_j \text{ error})$$

이는 **앙상블 이론**의 편향-분산 트레이드오프와 일치하며, 다양한 모델이 서로 다른 오류 패턴을 보정할 때 일반화 성능이 향상됩니다.

### 3-2. 계층적 정제를 통한 일반화

각 레이어를 거치면서 이전 레이어의 최선의 응답을 보존하면서 약점을 보완합니다. BLEU 기반 Spearman 상관 분석에서:

$$\rho(\text{BLEU score}, \text{win rate}) > 0$$

즉, aggregator가 가장 높은 품질의 제안 응답을 선택적으로 통합한다는 것이 실증적으로 검증되었습니다.

### 3-3. 다양한 태스크에 대한 일반화

| 태스크 유형 | 성능 향상 증거 |
|---|---|
| 자연어 생성 (AlpacaEval 2.0) | GPT-4o 대비 +7.6% |
| 멀티턴 대화 (MT-Bench) | SOTA 달성 |
| 세분화 평가 (FLASK) | 12개 중 7개 지표 향상 |
| 수학적 추론 (MATH) | 레이어당 단조 증가 |

### 3-4. 파인튜닝 없는 일반화

MoA는 어떠한 파인튜닝도 요구하지 않으며, 새로운 LLM이 등장해도 즉시 통합 가능합니다. 이는 **도메인 외(out-of-domain) 일반화**에 유리한 특성입니다.

$$\text{MoA}(A_1, \ldots, A_n) \xrightarrow{\text{plug-in}} \text{MoA}(A_1, \ldots, A_n, A_{n+1})$$

### 3-5. 일반화의 이론적 기반

LLM의 협력성 현상은 **사회적 학습(Social Learning)** 이론과 연결됩니다. Zhang et al. (2023)의 사회심리학적 관점처럼, 다양한 에이전트가 서로의 지식을 보완할 때 개체 한계를 넘어선 집단 지성이 발현됩니다.

---

## 4. 연구에 미치는 영향 및 미래 고려사항

### 4-1. 앞으로의 연구에 미치는 영향

#### (1) 패러다임 전환: "단일 초대형 모델" → "협력적 멀티모델 시스템"

MoA는 단순히 더 큰 모델을 학습하는 대신, **기존 모델들의 조합**으로 성능을 극적으로 향상시킬 수 있음을 보였습니다. 이는 LLM 연구의 스케일링 법칙(scaling law)에 대한 새로운 관점을 제시합니다.

#### (2) 오픈소스 생태계 강화

오픈소스 모델만으로 GPT-4o를 능가한 결과는 **오픈소스 커뮤니티 중심의 연구**를 촉진할 것입니다. 모델 조합 및 오케스트레이션 기술이 독립적인 연구 분야로 발전할 가능성이 큽니다.

#### (3) AI 에이전트 시스템 설계에 영향

Proposer/Aggregator 역할 분리는 멀티에이전트 AI 시스템 설계에 직접적인 영향을 미칩니다. 향후 **역할 특화 에이전트(role-specialized agent)** 연구가 활성화될 것으로 예상됩니다.

#### (4) 해석가능성(Interpretability) 향상

중간 출력이 자연어로 표현되므로, 기존의 블랙박스 LLM에 비해 추론 과정의 해석가능성이 높습니다. 이는 **AI 안전성 및 정렬(Alignment) 연구**에도 기여할 것입니다.

---

### 4-2. 미래 연구 시 고려할 점

#### (1) MoA 아키텍처 자동 최적화
현재는 레이어 수, 에이전트 선택 등을 수동으로 결정합니다. 향후에는:

$$\arg\max_{\{A_{i,j}\}, l, n} \text{Performance}(\text{MoA}) \text{ s.t. } \text{Cost} \leq C$$

와 같은 **자동화된 아키텍처 탐색(NAS for MoA)** 이 필요합니다.

#### (2) 레이턴시 최적화
- **청크 단위 집성(chunk-wise aggregation):** 전체 응답이 완성되기 전에 부분적 집성을 시작하여 TTFT 감소
- **스트리밍 MoA:** 토큰 단위로 실시간 집성하는 방법론 개발 필요

#### (3) 동적 에이전트 선택
태스크 유형에 따라 적합한 Proposer/Aggregator를 **동적으로 라우팅**하는 시스템:

$$A^*_{i,j} = \text{Router}(x_1, \text{TaskType}) \rightarrow \{A_{i,j}\}$$

#### (4) 오류 전파 및 환각(Hallucination) 연구
집성 과정에서 여러 모델의 잘못된 정보가 강화될 가능성이 있습니다. **오류 전파 메커니즘**에 대한 심층 연구가 필요합니다.

#### (5) 도메인 특화 MoA
현재는 범용 벤치마크 중심이지만, 의료·법률·코드 등 **전문 도메인**에서의 MoA 성능 검증 및 최적화가 중요합니다.

#### (6) 윤리적·사회적 고려
다수 모델의 편향이 집성될 경우 특정 편향이 증폭될 가능성이 있으므로, **편향 탐지 및 완화** 메커니즘을 MoA 파이프라인에 통합해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 파인튜닝 필요 | 다양성 활용 | 한계 |
|---|---|---|---|---|
| **GPT-3** (Brown et al., 2020) | Few-shot prompting | ✗ | ✗ (단일 모델) | 단일 모델 한계 |
| **Chain-of-Thought** (Wei et al., 2022) | 단계적 추론 프롬프팅 | ✗ | ✗ | 단일 모델, 복잡한 추론 한계 |
| **Self-Consistency** (Wang et al., 2022) | 동일 모델 다중 샘플링 후 투표 | ✗ | △ (동일 모델) | 다양성 제한 |
| **Tree-of-Thoughts** (Yao et al., 2023) | 트리 구조 추론 탐색 | ✗ | ✗ | 탐색 비용 증가, 단일 모델 |
| **LLM-Blender** (Jiang et al., 2023) | Pairwise ranking + GenFuser | ✓ (GenFuser 학습) | ✓ | 파인튜닝 비용, 새 모델 적용 어려움 |
| **Multiagent Debate** (Du et al., 2023) | 대칭적 토론 구조 | ✗ | ✓ | 역할 고정, 수렴 보장 없음 |
| **MAD** (Liang et al., 2023) | 비대칭 토론 (debater/judge) | ✗ | ✓ | 구조 설계 복잡 |
| **FrugalGPT** (Chen et al., 2023) | 캐스케이딩 모델 사용 | △ (라우터 학습) | △ | 비용 최적화 중심, 성능 향상 제한 |
| **ReConcile** (Chen et al., 2023) | 가중 투표 기반 토론 | ✗ | ✓ | 투표 메커니즘의 한계 |
| **MoA (본 논문, 2024)** | 계층적 집성-정제 | ✗ | ✓✓ | TTFT 높음, verbosity |

### 핵심 차별점

$$\text{MoA} = \underbrace{\text{No Fine-tuning}}_{\text{vs LLM-Blender}} + \underbrace{\text{Heterogeneous Models}}_{\text{vs Self-Consistency}} + \underbrace{\text{Layered Refinement}}_{\text{vs Single-round Debate}} + \underbrace{\text{SOTA Performance}}_{\text{vs All above}}$$

---

## 참고 자료 및 출처

**주 논문:**
- Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., & Zou, J. (2024). *Mixture-of-Agents Enhances Large Language Model Capabilities*. arXiv:2406.04692v1.
- GitHub: https://github.com/togethercomputer/moa

**논문 내 인용 참고문헌:**
- Brown et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS 33.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 35.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171.
- Yao et al. (2023a). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601.
- Jiang et al. (2023). *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion*. ACL 2023.
- Du et al. (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325.
- Liang et al. (2023). *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*. arXiv:2305.19118.
- Chen et al. (2023a). *ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs*. arXiv:2309.13007.
- Chen et al. (2023b). *FrugalGPT: How to Use Large Language Models while Reducing Cost and Improving Performance*. arXiv:2305.05176.
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538.
- Dubois et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*. arXiv:2404.04475.
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685.
- Ye et al. (2023). *FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets*. arXiv:2307.10928.
- Zhang et al. (2023). *Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View*. arXiv:2310.02124.
- Hendrycks et al. (2021). *Measuring Mathematical Problem Solving with the MATH Dataset*. arXiv:2103.03874.
- Huang et al. (2024). *Enabling Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration*. arXiv:2404.12715.
- Wang et al. (2024b). *Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?* arXiv:2402.18272.
