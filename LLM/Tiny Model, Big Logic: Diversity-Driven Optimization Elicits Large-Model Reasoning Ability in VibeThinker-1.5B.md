# Tiny Model, Big Logic: Diversity-Driven Optimization Elicits Large-Model Reasoning Ability in VibeThinker-1.5B

### 1. 핵심 주장 및 기여 요약

**VibeThinker-1.5B**는 Spectrum-to-Signal Principle (SSP) 프레임워크를 기반으로 하는 혁신적인 소형 언어모델로, 기존의 확장성 중심 패러다임에 강력한 도전을 제기합니다. 이 모델의 가장 근본적인 주장은 **"파라미터 규모가 아니라 학습 전략의 효율성이 추론 성능을 좌우한다"**는 것입니다.[1]

**핵심 기여:**

- 1.5B 파라미터로 671B 규모의 DeepSeek R1을 **AIME24** (80.3 vs 79.8), **AIME25** (74.4 vs 70.0), **HMMT25** (50.4 vs 41.7)에서 상회[1]
- 총 $7,800 학습 비용으로 **200배~600배 더 큰 모델들과 경쟁 가능한 성능** 달성[1]
- 소형 모델의 추론 능력 한계를 재정의하여 "작은 모델도 강력한 추론 가능"임을 실증적으로 증명[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**주요 문제 진단:**

산업의 광범위한 인식은 소형 모델(특히 1.5B 파라미터)이 본질적으로 강력한 추론 능력을 갖지 못한다는 것입니다. DeepSeek R1 (671B), Kimi K2 (>1T)와 같은 거대 모델이 주류가 되면서, 추론 능력은 순전히 파라미터 규모에 의존한다는 가정이 지배적입니다. 이는 다음과 같은 현실적 문제들을 야기합니다:[1]

- 엄청난 훈련 비용 ($294K~$535K)과 추론 비용[1]
- 고성능 하드웨어 부족으로 인한 연구 접근성 제한[1]
- 환경 및 에너지 효율성 악화[1]

#### 2.2 핵심 제안 방법: Spectrum-to-Signal Principle (SSP)

**SSP의 이론적 기초:**

SSP는 기존의 SFT(Supervised Fine-Tuning) + RL(Reinforcement Learning) 파이프라인을 **근본적으로 재구조화**합니다. 기존 방식은 SFT에서 Pass@1(단일 정확도)을 최대화한 후 RL에서 동일 지표를 추가 개선하는 단선형 접근이었습니다. 반면 SSP는 두 단계를 상호보완적으로 설계합니다:[1]

$$\text{Spectrum Phase (SFT)}: \text{Maximize } \text{Pass@K} \rightarrow \text{Diverse Solution Candidate Space}$$

$$\text{Signal Phase (RL)}: \text{Amplify Correct Signal} \rightarrow \text{Optimal Reasoning Path Selection}$$

**이론적 근거:**

Pass@K와 다양성의 관계는 다음과 같이 정의됩니다. K개의 독립적으로 생성된 솔루션 집합 $\{y_i\}_{i=1}^K$에서:[1]

$$\text{Pass@K} = \mathbb{E}_{x \sim D, \{y_i\}_{i=1}^K \sim \pi_\theta(\cdot|x)} \left[\max\{R(x,y_1), \ldots, R(x,y_K)\}\right]$$

여기서 $R(x,y)$는 이진 보상 함수입니다. 높은 Pass@K는 모델이 다양한 정답 경로를 생성할 수 있음을 의미하며, 이는 후속 RL 단계에서 최적화할 풍부한 후보 공간을 제공합니다.[1]

#### 2.3 두 단계 방법론 상세 설명

**Stage 1: Diversity-Exploring Distillation (Spectrum Phase)**

(1) **Domain-Aware Diversity Probing**

수학 지식 공간을 N개의 서브도메인으로 분할합니다:[1]

$$S = \{S_{\text{algebra}}, S_{\text{geometry}}, S_{\text{calculus}}, S_{\text{statistics}}\}$$

각 서브도메인 $S_i$에 대해 전문가 SFT 체크포인트 $M^*_i$를 식별합니다:

$$M^*_i = \arg\max_t P_i(t)$$

여기서 $P_i(t)$는 $t$번째 스텝의 체크포인트를 프로빙셋 $D_i$에 대해 평가한 Pass@K 점수입니다.[1]

(2) **Expert Model Fusion**

도메인별 전문가 모델들을 가중 선형 결합으로 통합합니다:[1]

$$M^{\text{SFT}}_{\text{Merge}} = \sum_{i=1}^{N} w_i M^*_i, \quad \text{여기서} \sum_{i=1}^{N} w_i = 1$$

VibeThinker-1.5B 구현에서는 **균등 가중치** $w_i = \frac{1}{N}$을 사용하여 모든 서브도메인의 다양성을 공평하게 통합합니다.[1]

**Stage 2: MaxEnt-Guided Policy Optimization (MGPO) (Signal Phase)**

**(1) 최대 엔트로피 원칙:**

G개의 롤아웃에서 정답 확률을 계산합니다:[1]

$$p_c(q) = \frac{1}{G}\sum_{i=1}^{G} I(r_i = 1)$$

Information-Theoretic 원리에 따르면, 이진 분포의 엔트로피는 $p_c(q) = 0.5$일 때 최대화됩니다. 이 상태의 문제가 "모델의 학습 경계"를 정의하므로 최적의 학습 가치를 갖습니다.[1]

**(2) 엔트로피 편차 정규화:**

Max-Entropy Deviation Distance를 정의합니다:[1]

$$D_{\text{ME}}(p_c(q) \| p_0) = p_c(q) \log \frac{p_c(q)}{p_0} + (1-p_c(q)) \log \frac{1-p_c(q)}{1-p_0}$$

여기서 $p_0 = 0.5$입니다.[1]

이를 기반으로 가중치 함수를 구성합니다:[1]

$$w_{\text{ME}}(p_c(q)) = \exp(-\lambda \cdot D_{\text{ME}}(p_c(q) \| p_0))$$

$\lambda \geq 0$은 정규화 계수로서 정확도가 0.5에서 벗어나는 정도에 따라 페널티를 부과합니다. $\lambda = 0$일 때 MGPO는 표준 GRPO로 축퇴됩니다.[1]

**(3) MGPO 최적화 목표:**

Group Relative Policy Optimization (GRPO)의 장점 추정값을 다음과 같이 수정합니다:[1]

$$A'_{j}(q) = w_{\text{ME}}(p_c(q)) \cdot A_j(q)$$

최종 MGPO 목표 함수는:[1]

```math
J_{\text{MGPO}}(\theta) = \mathbb{E}_{(q,y) \sim D} \left[\mathbb{E}_{\{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left\{\min\left(r_{i,t}(\theta)A'_{i,t}(q), \text{clip}\left(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\right)A'_{i,t}(q)\right)\right\}\right]\right]
```

여기서 $r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}|q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|q, y_{i,<t})}$는 토큰 수준의 확률 비율입니다.[1]

#### 2.4 모델 구조

VibeThinker-1.5B는 **Qwen2.5-Math-1.5B**를 기반 모델로 사용합니다. 모델 아키텍처의 핵심 특성:[1]

- **파라미터**: 1.5B (Dense Model, MoE 미적용)
- **컨텍스트 윈도우**: SFT 후 16K → RL에서 32K로 확장[1]
- **토큰 수**: 3.3조 토큰 상에서 사전학습[1]

**훈련 파이프라인:**

```
Pre-trained Base Model (Qwen2.5-Math-1.5B)
         ↓
  [SFT Stage - Spectrum Phase]
  ├─ Domain-Aware Probing (4 subdomains)
  ├─ Specialist Model Training (각 도메인별)
  ├─ Expert Model Fusion (평균 가중치)
  └─ Output: Diversity-Maximized SFT Checkpoint
         ↓
  [RL Stage - Signal Phase]
  ├─ Math Reasoning (16K context)
  ├─ Math Reasoning Extended (32K context)
  ├─ Code Generation
  └─ Output: Final VibeThinker-1.5B
```

#### 2.5 성능 향상 분석

**수학 벤치마크 결과:**

| 벤치마크 | Base Model | VibeThinker-1.5B | 향상율 | DeepSeek R1 |
|---------|-----------|---------|--------|-----------|
| AIME24 | 6.7 | 80.3 | **1100%** | 79.8 |
| AIME25 | 4.3 | 74.4 | **1628%** | 70.0 |
| HMMT25 | 0.6 | 50.4 | **8300%** | 41.7 |
| MATH500 | 58.5 | 95.0 | **62%** | - |

**코딩 벤치마크 결과:**

| 벤치마크 | Base Model | VibeThinker-1.5B | Magistral Medium |
|---------|-----------|---------|----------|
| LiveCodeBench V5 | 0.0 | 55.9 | - |
| LiveCodeBench V6 | 0.0 | 51.1 | 50.3 |

**지식 벤치마크 (GPQA-Diamond):** 46.7 (기본 16.4 대비)[1]

**경쟁사 모델과의 비교:**

- Claude Opus 4: AIME24 48.2 vs VibeThinker 80.3
- GPT-4.1: AIME24 46.5 vs VibeThinker 80.3[1]
- Magistral Medium (proprietary): AIME24 73.6 vs VibeThinker 80.3

#### 2.6 구현 세부사항

**데이터 오염 제거 (Decontamination):**

10-gram 매칭을 사용한 시맨틱 데코온태미네이션으로 AIME25, HMMT25 같은 2025년 벤치마크와의 데이터 누수를 완전히 배제합니다. 이는 VibeThinker-1.5B의 우수한 벤치마크 성과가 순수 모델 능력의 결과임을 보증합니다.[1]

**훈련 비용 효율성:**

| 모델 | 파라미터 | AIME25 점수 | GPU 시간 | 총 비용 |
|-----|--------|-----------|---------|--------|
| DeepScaleR | 1.5B | 31.5 | 3.8K | $4.5K |
| VibeThinker | 1.5B | 74.4 | 3.9K | $7.8K |
| DeepSeek-R1 | 671B | 70.0 | 147K | $294K |
| MiniMax-M1 | 456B | 74.6 | 258K | $535K |

$7,800으로 대형 모델 대비 **30~70배 비용 절감**을 달성합니다.[1]

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재 일반화 성능의 한계

**GPQA-Diamond 벤치마크의 의미:**

GPQA-Diamond는 PhD 수준의 졸업 문제 198개로 구성된 **전문 지식 벤치마크**입니다. VibeThinker-1.5B의 46.7점은:[1]

- DeepSeek R1 (671B): 71.5점 대비 25점 차이[1]
- Claude Opus 4: 81.0점 대비 34점 차이[1]

이는 **소형 모델의 근본적 한계**를 시사합니다. 논문은 명시적으로 "작은 모델은 광범위한 백과사전식 지식을 다루는 능력에서 본질적 제약을 가질 수 있다"고 인정합니다.[1]

#### 3.2 일반화 향상 기제

**1) 기반 모델 개선을 통한 경로:**

현재 VibeThinker는 **수학 중심 사전학습**을 받은 Qwen2.5-Math-1.5B를 기반으로 합니다. 더 균형잡힌 기반 모델(일반 도메인 데이터 포함)로 전환하면:[1]

- 코딩 성능 향상 (현재 기반 모델은 코딩에 제한적 노출)
- 일반 지식 성능 개선 가능성

논문은 "더 강한 기초 코드 능력을 갖춘 기반 모델을 사용하면 성능을 현저히 높일 수 있다"고 제시합니다.[1]

**2) 다양성 최적화를 통한 경로:**

SSP의 Pass@K 최적화는 **모델이 다양한 풀이 경로를 습득**하도록 강제합니다. 이는:[1]

$$\text{다양한 풀이 경로} \rightarrow \text{도메인 간 전이 학습 개선} \rightarrow \text{일반화 강화}$$

의 메커니즘으로 작동합니다. 특히 도메인 간 개념 공유(예: 대수학 기법의 기하학적 응용)를 더 강력하게 활성화합니다.[1]

**3) MGPO의 동적 커리큘럼 학습:**

최대 엔트로피 원칙에 기반한 문제 선택은 자동 **하드 샘플 마이닝(Hard Sample Mining)**을 실현합니다:[1]

$$\text{높은 불확실성} \rightarrow \text{학습 경계 부근} \rightarrow \text{일반화 경계 확장}$$

이 메커니즘은 모델이 새로운 도메인 문제에 더 강건하게 대응할 수 있도록 훈련합니다.[1]

#### 3.3 향상 가능성의 정량적 추정

**최신 연구에서의 관찰:**[2]

Pass@K 정책 최적화 연구에 따르면, 다양성 기반 학습은 다음과 같은 일반화 개선을 가져옵니다:

- **분포 외(Out-of-Distribution) 일반화**: 기존 방법 대비 **15~25% 성능 향상**
- **도메인 적응성**: 새 도메인에 대한 적응 기간 **50% 단축**

VibeThinker-1.5B의 구조가 이러한 이점을 내재하고 있으므로, 적절한 기반 모델과 학습 데이터 개선으로 **GPQA 점수를 50~60대로 향상**시킬 수 있을 것으로 예상됩니다.

#### 3.4 구체적 개선 방안

**프롬프트 레벨 일반화:**

도메인별 프롬프트 템플릿 적응을 통해 새로운 도메인에 더 빠르게 적응 가능:[1]

$$\text{Template}_{i} + \text{Domain}_{\text{new}} \rightarrow \text{Few-Shot Examples} \rightarrow \text{Performance Boost}$$

**메타-학습 보완:**

소형 모델의 빠른 적응을 위해 메타-학습(Meta-Learning) 기법 추가:

$$\text{Few-Shot Adaptation Loss}: \mathcal{L}_{\text{meta}} = \sum_{i} \|f_{\text{adapted}}(x_i) - y_i\|^2$$

이는 VibeThinker의 이미 다양한 솔루션 스펙트럼을 활용하여 새 작업으로의 빠른 전이를 가능하게 합니다.

***

### 4. 한계와 도전 과제

#### 4.1 명시적 한계

**1) 지식 벤치마크 성능 부족:**

GPQA-Diamond에서 46.7점이라는 성과는 **광범위 지식의 확장성 문제**를 드러냅니다. 이는 다음을 시사합니다:[1]

- 소형 모델은 **매개변수 수 자체의 한계**로 인해 대규모 지식 저장 불가능
- Chain-of-Thought 기법이 추론 성능은 향상시키지만, 기본 지식 부족 극복 불가

**2) 코딩 성능의 상대적 약점:**

LiveCodeBench V6에서 51.1은 Magistral Medium (50.3)과 거의 동등한 수준으로:[1]

- 기반 모델의 제한된 코딩 노출로 인한 제약
- 코딩 문제의 높은 다양성과 도메인 특이성 대응 부족

**3) 컨텍스트 길이 제한:**

기본적으로 32K 토큰 컨텍스트로 제한되며, 장문서 분석이나 복잡한 멀티-턴 추론이 제약됩니다.[1]

#### 4.2 이론적 한계

**1) SSP의 수렴성 보장 부족:**

Spectrum Phase와 Signal Phase의 이론적 상호작용에 대한 수렴 증명이 제시되지 않습니다. MGPO의 최적성 보장도 이론적으로 미흡합니다.[1]

**2) 최대 엔트로피 원리의 문제:**

불확실성 = 최적 학습 가치라는 가정이 항상 성립하지 않을 수 있습니다. 특정 도메인에서는 명확한 구조의 문제가 학습에 더 유리할 수 있습니다.[3]

#### 4.3 실무적 한계

**1) 재현성과 검증:**

합성 데이터와 proprietary 데이터 혼합으로 완전 재현 불가능[1]

**2) 배포 환경의 다양성:**

1.5B 모델도 엣지 디바이스(저전력 IoT)에서는 여전히 과중할 수 있음.

***

### 5. 논문의 연구 영향과 미래 연구 고려사항

#### 5.1 패러다임 전환의 의의

**기존 스케일링 법칙의 재평가:**

Chinchilla 스케일링 법칙과 Scaling Laws for Neural Language Models 같은 기존 연구들은 **모델 크기와 학습 데이터 양의 균형**을 강조했습니다. VibeThinker는 이 패러다임에 중대한 도전을 제시합니다:[4]

- 파라미터 수 < 효율적 학습 알고리즘
- 배치 크기와 학습률 최적화 > 모델 확장

이는 향후 LLM 개발의 **우선순위 재정렬**을 의미합니다.

**연구 민주화의 실현:**

Belcak et al. (2025)의 이론적 주장을 실증적으로 입증합니다. 소형 모델이 경쟁력 있는 성능을 낼 수 있다면:[1]

- 자원이 제한된 학계 및 중소 기업의 LLM 연구 진입 가능
- 오픈소스 에코시스템의 빠른 성장
- AI 개발의 지리적 수평화

#### 5.2 향후 연구 방향

**1) Mixed-Scale Training Paradigm:**

**소형 모델의 강점 + 대형 모델의 강점 결합**

$$\text{Ensemble}: \{M_{\text{small}}(\text{빠른 추론}), M_{\text{large}}(\text{깊은 추론})\} \rightarrow \text{최적 경로 선택}$$

동적 라우팅을 통해 문제 난이도에 따라 모델 선택:

- 간단한 문제 → 소형 모델 (빠른 응답)
- 복잡한 추론 → 대형 모델 (정확성)

**2) Domain-Specific Small Models:**

특정 도메인(의료, 법률, 금융)에 최적화된 **1.5B 규모 전문 모델 개발**

$$\text{Domain Knowledge} + \text{SSP Framework} \rightarrow \text{Sub-Billion Parameter Expert Models}$$

이는 GPQA 점수 개선뿐 아니라 실무 적용성 극대화.

**3) Adaptive Entropy Weighting:**

최대 엔트로피 고정값($p_c = 0.5$) 대신 **학습 단계와 도메인에 따라 동적 조정**

$$\lambda_{\text{adaptive}}(t, \text{domain}) = f(t, \text{domain}, \text{model loss})$$

초기 학습에서는 높은 탐색(낮은 엔트로피 선호), 후기에서는 정교한 최적화(중간 엔트로피 선호).

**4) Curriculum Learning Integration:**

SSP와 커리큘럼 학습의 결합:

$$\text{쉬운 문제} \xrightarrow{\text{다양성}} \text{중간 문제} \xrightarrow{\text{MGPO}} \text{어려운 문제}$$

체계적 난이도 상승으로 일반화 성능 강화.[5]

**5) Multi-Modal Small Models:**

Vision-Language 작업에 SSP 프레임워크 적용

$$\text{Vision Diversity} + \text{Language Diversity} + \text{MGPO} \rightarrow \text{1.5B VLM}$$

현재 VLM은 대부분 7B 이상이므로, 소형 VLM 개발은 새로운 기회.

#### 5.3 이론적 개선 필요 영역

**1) SSP 수렴성 증명:**

$$\exists \delta > 0: \|J_{\text{MGPO}}(\theta^*) - J_{\text{opt}}\| \leq \delta$$

의 형태로 수렴 보장 필요.

**2) Diversity-Accuracy Trade-off 분석:**

$$\text{Pareto Frontier}: \{\text{Pass@K}, \text{Pass@1}\}$$

상에서 최적점 도출의 이론적 조건.

**3) Uncertainty Calibration:**

$p_c(q) = 0.5$가 항상 최적인지 실증적 검증. 도메인별, 모델 크기별 최적 엔트로피 수준 도출.

#### 5.4 윤리 및 사회적 영향 고려사항

**긍정적 영향:**

- 소형 모델의 민주화로 AI 연구 접근성 확대
- 에너지 효율성 개선으로 탄소배출 감소[1]
- 모바일/엣지 배포로 개인정보 보호 강화

**주의할 점:**

- 소형 모델의 낮은 일반 지식(GPQA 46.7점)으로 인한 신뢰성 문제
- 추론 데이터의 품질 의존성으로 인한 바이어스 위험
- 저비용 모델의 악의적 사용 가능성

***

### 6. 최신 관련 연구 동향

#### 6.1 동시대 소형 추론 모델 연구

**관련 모델들과의 비교:**[6][7]

| 모델 | 파라미터 | 주요 기법 | AIME25 점수 |
|-----|--------|---------|-----------|
| DeepSeek-R1-Distill-Qwen-32B | 32B | 증류 + RL | - |
| Qwen3-1.7B | 1.7B | 표준 RL | 36.8 |
| ProRL-1.5B | 1.5B | RL 최적화 | 33.3 |
| VibeThinker-1.5B | 1.5B | SSP + MGPO | **74.4** |

VibeThinker의 **성능 우위는 일반적인 다른 1.5B 모델 대비 2배 이상**.

#### 6.2 정책 최적화 최신 발전

**Pass@K Policy Optimization (PKPO) 및 확장:**[8][2]

PKPO는 Pass@K를 직접 최적화하는 저분산 편향 추정량을 제시:

$$\text{Pass@K Gradient}: \nabla_\theta \mathbb{E}[r_i] \text{의 비편향 추정}$$

이는 VibeThinker의 MGPO와 이론적으로 상호보완적이며, 향후 결합 가능성이 높습니다.[2][1]

**9가지 최신 정책 최적화 기법 (2025년 발표):**[9]

- GSPO (Group Sequence PO)
- LAPO (Length-Adaptive PO)
- HBPO (Hierarchical Budget PO)
- CISPO (Clipped Importance Sampling PO)

이들은 모두 VibeThinker의 기본 프레임워크를 보완할 수 있는 메타휴리스틱입니다.

#### 6.3 일반화 성능 향상 기술

**관련 최신 연구:**[10][5]

**PRADA (Prompt-Assisted Domain-Adversarial Fine-tuning):**

소형 모델의 Chain-of-Thought 일반화 개선을 위해:

1. 교사 모델의 다양한 CoT 생성
2. P-Tuning을 통한 프롬프트 학습
3. 도메인 적대적 미세조정

이는 VibeThinker의 도메인별 다양성 추구와 직접 호환 가능합니다.

**Efficient Reasoning Models 서베이(2025):**[5]

소형 모델 추론 개선의 세 가지 축:

1. **Shorter (추론 경로 단축)** - VibeThinker는 이미 효율적 경로 선택
2. **Smaller (모델 압축)** - VibeThinker의 기본 설계
3. **Faster (디코딩 최적화)** - 추가 개선 가능 영역

***

### 7. 결론: VibeThinker-1.5B의 전략적 중요성

**패러다임 수정 가치:**

VibeThinker-1.5B는 단순한 벤치마크 우수성을 넘어, **"스케일링 법칙 재정의"**의 증거입니다. 이는:[1]

$$\text{Model Performance} = f(\text{Parameter Size}, \text{Training Algorithm Quality}, \text{Data Distribution})$$

에서 알고리즘 품질의 중요성을 실증적으로 입증합니다.

**실무적 기여:**

- **비용 혁신**: $7,800으로 엔터프라이즈급 추론 모델 개발 가능
- **배포 혁신**: 모바일, 엣지, IoT 기기에서 고수준 추론 실행 가능
- **접근성 혁신**: 고성능 하드웨어 부족한 조직의 AI 연구 진입 가능

**향후 과제:**

여전히 해결해야 할 도전 과제들:

1. **일반 지식 기반 강화** (GPQA 성능 개선)
2. **이론적 수렴성 증명** (SSP, MGPO)
3. **멀티모달 확장** (Vision-Language 적용)
4. **도메인 특화** (의료, 법률 등 전문 분야)

***

### 참고문헌 및 인용

VibeThinker-1.5B 논문에 인용된 핵심 연구들과 최신 관련 연구를 통해, 이 모델은 **AI 연구의 민주화와 효율성 혁명**의 시작점으로 평가됩니다. 특히 2025년 현재 소형 언어 모델의 부활이라는 대세 속에서, VibeThinker의 SSP 프레임워크는 후속 연구의 강력한 기초를 제공할 것으로 기대됩니다.[11]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cdb65d72-b8ea-4fc1-9837-425a56aa8061/2511.06221v1.pdf)
[2](https://www.themoonlight.io/ko/review/passk-policy-optimization-solving-harder-reinforcement-learning-problems)
[3](https://openreview.net/forum?id=FRFuvBRueA)
[4](https://arxiv.org/pdf/2204.02311.pdf)
[5](https://www.themoonlight.io/ko/review/efficient-reasoning-models-a-survey)
[6](https://smcho1201.tistory.com/138)
[7](https://tech.hancom.com/2025-03-10-deepseek-r1-technology-analysis/)
[8](https://www.themoonlight.io/ko/review/top-pass-improve-code-generation-by-passk-maximized-code-ranking)
[9](https://turingpost.co.kr/p/9-new-policy-optimization-techniques)
[10](https://www.themoonlight.io/ko/review/enhancing-generalization-in-chain-of-thought-reasoning-for-smaller-models)
[11](https://www.jaenung.net/tree/18915)
[12](https://arxiv.org/pdf/1906.05721.pdf)
[13](https://arxiv.org/pdf/2412.06542.pdf)
[14](https://arxiv.org/html/2406.01655v1)
[15](https://aclanthology.org/2023.emnlp-main.157.pdf)
[16](https://www.mdpi.com/2073-431X/13/7/173)
[17](https://arxiv.org/pdf/2204.00827.pdf)
[18](http://arxiv.org/pdf/2405.02287.pdf)
[19](https://aclanthology.org/2023.emnlp-main.568.pdf)
[20](https://hyper.ai/kr/papers/2511.06221)
[21](https://arxiv.org/abs/2511.06221)
[22](https://do-it-ai.com/blog/?bmode=view&idx=164482502)
[23](https://tilnote.io/pages/69177c966e24323fc96c19b6)
[24](http://www.jpnt.org/wp-content/uploads/2024/03/JPNT-1004-11.pdf)
[25](https://skywork.ai/blog/ko/models/deepseek-r1-distill-qwen-32b-free-chat-online/)
[26](https://k-erc.eu/wp-content/uploads/2025/09/KERC-Issue-Report-%ED%98%B8%EB%9D%BC%EC%9D%B4%EC%A6%8C-%EC%9C%A0%EB%9F%BD-2026-2027-%EC%9B%8C%ED%81%AC%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B04-%EC%B4%88%EC%95%88-%EB%B6%84%EC%84%9D%EC%84%9C-%EC%96%91%EB%A9%B4.pdf)
[27](https://koreascience.kr/article/CFKO202404272003576.pdf)
[28](https://goddaehee.tistory.com/420)
[29](https://arxiv.org/html/2408.11294)
[30](https://arxiv.org/pdf/2008.03979.pdf)
[31](https://arxiv.org/pdf/2002.01808.pdf)
[32](http://arxiv.org/pdf/2307.14781.pdf)
[33](https://arxiv.org/pdf/2311.13784.pdf)
[34](https://arxiv.org/pdf/2307.02251.pdf)
[35](http://arxiv.org/pdf/2405.12543.pdf)
[36](http://arxiv.org/pdf/2408.05088.pdf)
[37](https://taek-l.tistory.com/30)
[38](https://littlefoxdiary.tistory.com/131)
[39](https://jeonghoonpark.com/blog/ttrl/)
[40](https://wikidocs.net/209005)
[41](https://turingpost.co.kr/p/chinese-ai-models-review)
[42](https://patents.google.com/patent/KR20180113587A/ko)
[43](https://velog.io/@foqlzm12345/Knowledge-Distillation-%EB%AA%A8%EB%8D%B8-%EC%A6%9D%EB%A5%98-%EA%B8%B0%EB%B2%95)
[44](https://metanetglobal.com/bbs/board.php?bo_table=tech&wr_id=149)
[45](http://arxiv.org/pdf/2404.11202.pdf)
[46](https://arxiv.org/pdf/2303.15647.pdf)
[47](https://arxiv.org/pdf/2402.05033.pdf)
[48](https://arxiv.org/pdf/2110.15352.pdf)
[49](http://arxiv.org/pdf/2211.12509.pdf)
[50](http://arxiv.org/pdf/2404.14219.pdf)
[51](https://arxiv.org/pdf/2401.02643.pdf)
[52](https://engineer-daddy.co.kr/entry/AI-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%EC%97%90%EC%84%9C-%EC%A7%80%EC%8B%9D-%EC%A6%9D%EB%A5%98Knowledge-Distillation%EC%9D%98-%EC%97%AD%ED%95%A0%EA%B3%BC-%ED%99%9C%EC%9A%A9%EB%B2%95)
[53](https://discuss.pytorch.kr/t/2025-11-17-23-ai-ml/8271)
[54](https://yozm.wishket.com/magazine/detail/3107/)
[55](https://blog.naver.com/PostView.naver?blogId=economic_moat&logNo=223732988544)
[56](https://blog.naver.com/simula/224071104998?fromRss=true&trackingCode=rss)
[57](https://brunch.co.kr/@@goUU/199)
[58](https://macaron.im/ko/blog/post-training-llm-techniques-2025)
[59](https://digitalbourgeois.tistory.com/2067)
[60](https://turingpost.co.kr/p/topic-30-knowledge-distillation)
