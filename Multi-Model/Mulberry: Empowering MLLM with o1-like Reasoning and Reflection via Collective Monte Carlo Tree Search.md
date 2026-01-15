
# Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search

## 1. 핵심 주장 및 주요 기여 요약

**Mulberry** 논문은 멀티모달 대형 언어 모델(MLLM)이 복잡한 추론 작업을 수행할 때 겪는 기본적인 문제를 해결하기 위해 설계되었습니다. 핵심 주장은 다음과 같습니다:

**문제의 본질**: 현재의 MLLM들은 "직접 예측(direct prediction)" 모드에서 작동하여 중간 추론 과정을 명시적으로 표현하지 않습니다. 이로 인해 복잡한 문제 해결 능력이 제한됩니다.

**해결 방안**: 리처드 파인만의 명언 "내가 만들 수 없으면, 이해하지 못한다"에서 영감을 얻어, 각 중간 단계의 추론을 학습하도록 MLLM을 훈련하는 방식을 제시합니다.

**주요 기여**는 다음 네 가지입니다:

1. **Collective Monte Carlo Tree Search (CoMCTS)**: 여러 MLLM의 집단 지식을 활용하여 효과적이고 효율적인 추론 경로를 탐색하는 새로운 학습-추론 방법론
2. **Mulberry-260k 데이터셋**: 260,000개의 질문에 대해 풍부하고 명시적이며 잘 정의된 추론 노드를 가진 트리 구조의 데이터셋
3. **Mulberry 모델**: o1 스타일의 단계별 추론 및 반성(reflection) 능력을 갖춘 MLLM 시리즈
4. **광범위한 성능 검증**: 다양한 벤치마크에서 우수한 성능 입증

***

## 2. 해결 문제, 제안 방법론, 모델 구조 상세 설명

### 2.1 해결하고자 하는 문제

MLLM이 복잡한 추론 작업에서 실패하는 두 가지 근본적인 원인:

1. **추론 깊이 부족**: 기존 MLLM은 최종 답변만 생성하며, 중간의 명시적 추론 단계가 없음
2. **직접 예측의 한계**: 짧은 생성으로 인해 복잡한 다단계 문제를 해결할 수 없음

### 2.2 제안 방법론: Collective Monte Carlo Tree Search (CoMCTS)

CoMCTS는 다음의 핵심 도전 과제를 해결합니다:

**도전 1 - 탐색 효과성(Search Effectiveness)**:
- 기존 MCTS는 단일 MLLM의 동질적인 저품질 노드에 갇히는 경향
- CoMCTS는 여러 MLLM의 다양하고 상호보완적인 추론 경로를 결합

**도전 2 - 탐색 효율성(Search Efficiency)**:
- 기존 MCTS는 반복 주기마다 하나의 추론 노드만 확장
- CoMCTS는 단일 반복에서 여러 모델의 경로를 동시에 확장

#### CoMCTS의 네 가지 반복 연산:

**(a) 확장(Expansion)**

주어진 현재 리프 노드 $s^k_m$에서, K개의 MLLM 그룹 $\{\pi_1, \pi_2, \ldots, \pi_K\}$의 집단 지식을 활용하여 다양하고 상호보완적인 후보 추론 경로를 병렬로 확장합니다:

$$S^j_{\text{candidate}} \sim \pi_j(\cdot|Q, \text{Parent}(s^k_m), s^k_m)$$

여기서:
- $Q$: 멀티모달 입력 질문
- $\text{Parent}(s^k_m)$: 현재 노드의 모든 부모 노드들
- $S_{\text{candidate}} = \cup^K_{j=1} S^j_{\text{candidate}}$: 모든 모델로부터의 후보 경로의 합집합

**(b) 시뮬레이션 및 오류 위치 파악(Simulation and Error Positioning)**

생성된 후보 노드들의 품질을 평가하고 오류 있는 노드를 필터링합니다:

$$R(s^j_i) = \frac{1}{K}\sum^K_{l=1}\pi_l(\cdot|\text{prompt}_{\text{eval}}, Q, \text{Parent}(s^j_i), s^j_i)$$

평가 점수 $R(s^j_i)$가 임계값 $t$보다 높은 노드만 유지:

$$S^*_{\text{candidate}} = \{s^j_i \in S_{\text{candidate}} | R(s^j_i) \geq t\}$$

**(c) 역전파(Backpropagation)**

새로 확장된 노드들의 통계(방문 횟수 N과 노드 값 V)를 리프에서 루트까지 역방향으로 업데이트합니다:

$$V(s) \leftarrow \frac{N(s) \cdot V(s) + \sum_{s_l \in \text{Child}(s)} R(s_l)}{N(s) + \text{CountChild}(S^*_{\text{candidate}}, s)}$$

$$N(s) \leftarrow N(s) + \text{CountChild}(S^*_{\text{candidate}}, s)$$

**(d) 선택(Selection)**

상부신뢰도(Upper Confidence Bound, UCB) 값이 가장 높은 노드를 다음 탐색의 시작점으로 선택합니다:

$$s^{k*}_m = \arg\max_{s \in S^*_{\text{candidate}}} V(s) + c \cdot \sqrt{\frac{\log N(\hat{s})}{1 + N(s)}}$$

여기서 $c$는 탐색-활용(exploration-exploitation) 균형을 조절하는 상수입니다.

### 2.3 반성 추론 경로 확장

기본 CoMCTS 외에도, 긍정적 노드와 부정적 노드 모두를 포함하는 트리를 활용하여 반성적 추론을 구성합니다:

**음수 형제 노드 식별**:

$$s_{\text{neg}} = \arg\min_{s_l \in \text{Sibling}(s)} \text{UCB}(s_l) - \text{UCB}(s), \quad \forall s \in Y$$

**반성적 추론 경로 구성**:

$$Y_{\text{reflect}} = \text{Replace}(Y, s, (s_{\text{neg}}, \text{prompt}_{\text{reflect}}, s))$$

이를 통해 모델은 오류 있는 추론 단계에서 올바른 단계로의 전환을 학습합니다.

### 2.4 모델 구조 및 학습

**집단 감독 미세 조정(Collective Supervised Fine-Tuning, CoSFT)**:

효과적인 추론 경로에 대한 표준 SFT 목표:

$$L_{\text{CoSFT}}(\pi_k) = \sum_{(Q,Y) \in D} \log \pi_k(Y|Q)$$

반성적 추론에 대한 추가 목표:

$$L_{\text{CoSFT-Re}}(\pi_k) = \sum_{(Q,Y_{\text{reflect}}) \in D} \log \pi_k(Y_{\text{reflect}}|Q)$$

최종 학습 목표는 CoMCTS가 생성한 추론 노드의 트리 $S$에서 효과적인 경로 $Y$와 반성적 경로 $Y_{\text{reflect}}$의 로그 확률을 최대화합니다.

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상

| 측정 지표 | 결과 |
|---------|------|
| **검색 성공률** | GPT-4o 직접 예측 58.2% → CoMCTS 80.2% (+22.0%) |
| **평균 검색 반복** | 42.1회 (기존 MCTS) → 12.7회 (CoMCTS) 3.3배 효율 향상 |
| **기반 모델 개선** | Qwen2-VL-7B: 58.2% → 63.1% (+4.9%) |
| | LLaMA-3.2-11B: 48.6% → 61.1% (+12.5%) |
| **일반화 성능** | Qwen2-VL-2B: 43.0% → 51.7% (+5.4%) |
| | LLaVA-NeXT-8B: 37.5% → 56.3% (+18.8%) |

**주요 벤치마크에서의 성능**:

| 벤치마크 | MathVista | MMStar | MMMU | ChartQA | 평균 |
|---------|-----------|--------|------|---------|------|
| Mulberry-7B | 63.1% | 61.3% | 55.0% | 83.9% | 58.9% |
| Qwen2-VL-7B (기준) | 58.2% | 60.7% | 54.1% | 83.0% | 54.7% |
| 개선율 | +4.9% | +0.6% | +0.9% | +0.9% | +4.2% |

### 3.2 추론 단계 분포의 유연성

CoMCTS의 주목할 만한 특성은 모델이 문제의 복잡도에 따라 추론 단계 수를 조정할 수 있다는 점입니다:

- **간단한 질문(차트 관련)**: 평균 6.8단계
- **복잡한 질문(기하학 관련)**: 평균 8.9단계
- **전체 평균**: 7.5단계

이는 모델이 "간단한 문제는 빠르게 생각하고, 복잡한 문제는 느리게 생각하는" 능력을 갖추었음을 의미합니다.

### 3.3 주요 한계

1. **데이터 규모의 제한**:
   - 260k 샘플로 제한됨 (5k만 반성 데이터)
   - 특정 도메인에 치우친 데이터 구성

2. **계산 효율성**:
   - CoMCTS는 여전히 다중 모델 평가 필요
   - 추론 시간 비용이 상당함

3. **집단 학습의 모델 의존성**:
   - 4개의 사전 훈련된 강력한 MLLM 필요
   - GPT-4o 포함으로 인한 비용/독점성 문제

4. **반성 메커니즘의 제한**:
   - 부정적 형제 노드가 없을 경우 반성 학습 불가능
   - 반성 데이터가 전체 훈련 데이터의 ~5%

5. **분포 외 일반화(Out-of-Distribution Generalization)**:
   - 모델이 훈련 분포 내의 패턴에 최적화되어 있을 수 있음
   - 완전히 새로운 추론 스타일이 필요한 문제에서는 성능 저하 가능

***

## 4. 모델의 일반화 성능 향상 가능성 중점 분석

### 4.1 일반화 성능의 현황

#### 도메인 내 성능(In-Distribution Performance)
CoMCTS-검색된 데이터로 훈련된 모델들은 8개의 주요 벤치마크에서 일관된 개선을 보여줍니다. 특히:

- **수학 추론(MathVista)**: +4.9% (Qwen2-VL-7B 기준)
- **멀티 학문적 이해(MMMU)**: 평균 +0.9%
- **차트 질문 응답(ChartQA)**: 평균 +0.9%

#### 교차 모델 일반화(Cross-Model Generalization)
CoMCTS에 포함되지 않은 모델들에 대한 일반화 성능:

| 모델 | 성능 향상 |
|-----|---------|
| Qwen2-VL-2B | +5.4% |
| LLaVA-NeXT-8B | +11.0% |
| 평균 | +8.2% |

이는 **데이터셋의 일반화 능력**이 뛰어남을 시사합니다.

### 4.2 일반화 성능 향상의 메커니즘

#### 1) 집단 학습의 다양성(Collective Knowledge Diversity)

**표: 집단 학습 기여도 분석**

| 모델 조합 | 검색 성공률 |
|---------|----------|
| GPT-4o만 사용 | 63.8% |
| + Qwen2-VL-7B | 65.6% (+1.8%) |
| + LLaMA3.2-11B | 67.3% (+1.7%) |
| + Qwen2-VL-72B | 80.2% (+12.9%) |

작은 모델(Qwen2-VL-7B)도 시스템에 +2.4%를 기여하여, **모델 크기와 무관하게 다양한 시각이 중요**함을 입증합니다.

#### 2) 반성적 학습의 효과(Reflective Learning Effect)

$$\text{Improvement} = \text{반성 데이터 없음}(50.9\%) \to \text{반성 데이터 포함}(51.7\%) = +0.8\%$$

반성적 추론 경로를 포함하면 다음을 학습합니다:
- 부정적 추론 패턴 인식
- 오류 수정 메커니즘
- 자기 교정 능력

#### 3) 명시적 추론 단계의 가치(Value of Explicit Reasoning)

기존 방식(직접 예측)과 CoMCTS의 비교:

- **직접 예측**: 답변 생성만 학습 → 분포 내 패턴에만 최적화
- **명시적 추론**: 각 단계를 학습 → 추론 과정의 일반화 가능성 ↑

### 4.3 일반화 성능의 한계와 미래 개선 방향

#### 현재 한계점:

1. **분포 내 학습(In-Distribution Bias)**
   - Mulberry-260k는 특정 도메인 분포에 기반
   - 완전히 새로운 유형의 문제에서 성능 저하 가능

2. **코드 생성 및 기호적 추론의 약점**
   - MM-Math 벤치마크: 겨우 23.7% (기존 모델 5.9% 대비 +17.8%이지만 절대값은 낮음)
   - 이는 시각-텍스트 정렬보다 기호적 정확성이 필요한 분야

3. **분포 외 정보(Out-of-Distribution Robustness)**
   - 논문에서 직접적으로 테스트되지 않음
   - 일반적으로 MLLM의 O.O.D 성능 문제는 미해결

#### 잠재적 개선 전략:

**전략 1: 다양한 추론 스타일 학습**
$$\text{Generalization} \propto \text{Diversity of Reasoning Patterns in } \mathcal{D}$$

- 다양한 도메인의 추론 경로 포함
- 다양한 오류 패턴의 명시적 학습

**전략 2: 메타-학습 적용(Meta-Learning)**
- 새로운 추론 작업에 빠르게 적응하는 능력
- 소수의 예제로부터 학습하는 능력 강화

**전략 3: 모달리티 간 정렬 개선(Cross-Modal Alignment)**
- 시각적 정보와 텍스트 추론의 더 깊은 통합
- 다중 모달리티 전이 학습

***

## 5. 2020년 이후 관련 최신 연구와의 비교 분석

### 5.1 OpenAI o1/o3 (2024)

**기본 원리**:
- 강화학습(RL)을 통해 자동으로 ChainOfThought 추론 생성
- 테스트 타임 계산 스케일링(test-time compute scaling)

**Mulberry와의 비교**:

| 측면 | OpenAI o1/o3 | Mulberry |
|-----|-------------|---------|
| 방식 | RL 기반 자동 추론 | MCTS 기반 데이터 구성 |
| 모델 접근 | 폐쇄형 (독점) | 개방형 (연구용) |
| 계산 효율성 | 우수 (추론 시 스케일) | 중간 (데이터셋 생성 시 비용) |
| 다양성 | 단일 모델 | 여러 모델의 집단 지식 |
| **o1의 강점** | 최고 성능 (MathVista 63.8%, MMMU 69.1%) | - |
| **Mulberry의 강점** | - | 여러 모델에 일반화 가능 (+11.0%) |

### 5.2 ReST-MCTS* (Zhang et al., 2024)

**핵심 메커니즘**:
- Process Reward Model (PRM)을 사용한 MCTS
- 단계별 보상 신호 학습
- 반복적 자기 훈련(iterative self-training)

**Mulberry와의 비교**:

| 측면 | ReST-MCTS* | Mulberry |
|-----|-----------|---------|
| 적용 대상 | 텍스트 기반 LLM | 멀티모달 MLLM |
| 보상 신호 | PRM 학습 필요 | 집단 평가 기반 |
| 데이터셋 | 자동 생성 | 구조화된 트리 |
| 검색 효율성 | 우수 | 우수 (12.7회 반복) |
| **성능** | GSM8K 81.8% (+5.9%) | MathVista 63.1% (+4.9%) |

### 5.3 LLaVA-CoT/LLaVA-o1 (Xu et al., 2024)

**방식**:
- 다단계 구조화된 추론 (Summary → Caption → Reasoning → Conclusion)
- 단계별 빔 서치(stage-level beam search)
- 100k의 작은 훈련 데이터셋

**Mulberry와의 비교**:

| 측면 | LLaVA-o1 | Mulberry |
|-----|---------|---------|
| 구조 | 4단계 명시적 프레임 | 동적 단계 수 조정 |
| 훈련 데이터 | 100k | 260k (5배) |
| 유연성 | 고정 구조 | 문제별 최적화 |
| 성능(MathVista) | 54.8% | 63.1% (+8.3%) |
| 추론 속도 | 빠름 (구조화) | 중간 (동적) |

### 5.4 Insight-V (2024)

**특징**:
- 긴 체인 추론 데이터셋의 중요성 강조
- Multi-agent 시스템 (Reasoner + Summary agent)
- 반복적 DPO (Direct Preference Optimization)

**Mulberry와의 비교**:

| 측면 | Insight-V | Mulberry |
|-----|----------|---------|
| 데이터 생성 | Progressive strategy | MCTS 기반 |
| 교사 모델 | GPT-4V | 여러 MLLM |
| 최적화 방법 | 반복 DPO | CoSFT + 반성 |
| 일반화 | 기반 모델 중심 | 교차 모델 일반화 |

### 5.5 Corvid (2025)

**혁신점**:
- MCoT-Instruct-287K 데이터셋 (290k 샘플)
- 하이브리드 비전 인코더
- 추론 시간 자기 검증 전략

**Mulberry와의 비교**:

| 측면 | Corvid | Mulberry |
|-----|-------|---------|
| 데이터셋 크기 | 287k | 260k |
| 데이터 특성 | CoT 형식 표준화 | 트리 구조 |
| 모델 아키텍처 | 특화된 인코더 | 기반 모델 활용 |
| 성능(MathVista) | 추론-중심 최적화 | 균형잡힌 다중 벤치마크 |

### 5.6 R3V: Vision-Language 모델 자기 개선 (2024)

**메커니즘**:
- 부트스트래핑(bootstrapping)을 통한 긍정/부정 솔루션 수집
- 반성(reflection) 메커니즘
- 반복적 자기 훈련

**Mulberry와의 비교**:

| 측면 | R3V | Mulberry |
|-----|-----|---------|
| 기반 방법 | 자기 부트스트래핑 | 집단 MCTS |
| 반성 데이터 | 부정적 솔루션 분석 | 음수 형제 노드 활용 |
| 트리 구조 | 암묵적 | 명시적 |
| 검색 메커니즘 | 없음 | 체계적 MCTS |
| **일반화** | 뛰어남 | 우수 (+5.4~+11.0%) |

### 5.7 종합 비교표

| 논문 | 연도 | 핵심 방법 | 데이터셋 | 주요 성능 | 강점 |
|-----|-----|---------|--------|---------|------|
| OpenAI o1 | 2024 | RL + CoT | 비공개 | 63.8% | 최고 성능 |
| ReST-MCTS* | 2024 | PRM + MCTS | 자동 생성 | 81.8% (GSM8K) | 효율성 |
| LLaVA-o1 | 2024 | 4단계 구조 | 100k | 54.8% | 빠른 속도 |
| Insight-V | 2024 | 다중 에이전트 | 자동 생성 | 경쟁력 있음 | 긴 사슬 |
| Corvid | 2025 | MCoT 데이터 | 287k | CoT 강화 | 표준화 |
| **Mulberry** | **2024** | **CoMCTS** | **260k** | **63.1% (7B)** | **교차 모델 일반화** |

***

## 6. 논문이 앞으로의 연구에 미치는 영향

### 6.1 방법론적 영향

#### 1) 집단 학습의 새로운 패러다임
Mulberry는 여러 모델의 강점을 통합하는 **"집단 지식(collective knowledge)"** 개념을 실증적으로 입증했습니다. 이는 다음을 의미합니다:

- **소규모 모델도 가치 있다**: Qwen2-VL-7B 포함으로 +2.4% 기여
- **다양성이 중요하다**: 동질적 오류 회피의 핵심
- **효율성 개선**: 단일 모델 MCTS 대비 3.3배 효율적

**향후 연구 방향**:
- 더 이질적인 모델 조합의 활용
- 각 모델의 강점을 활용한 적응적 가중치 학습
- 분산 학습 환경에서의 확장성

#### 2) 명시적 트리 구조의 중요성
기존의 암묵적 추론에서 명시적 트리 구조로의 전환:

$$\text{학습 신호} = f(\text{각 단계의 정확성}) > g(\text{최종 답안만})$$

**함의**:
- 단계별 감시(step-level supervision)의 실효성
- 중간 표현의 학습 가능성
- 오류 수정 메커니즘의 구성 가능성

#### 3) 동적 추론 길이 조정
모델이 문제 복잡도에 따라 추론 단계를 조정하는 능력:

**가우시안 분포 모델링 가능**:
$$P(\text{steps} | \text{problem difficulty}) \approx \mathcal{N}(\mu(\text{difficulty}), \sigma^2)$$

### 6.2 응용 분야에 미치는 영향

#### 1) 의료 및 과학 분야
- 복잡한 진단 추론 과정의 명시적 표현
- 의사 결정 투명성 향상
- 오류 가능성 감지 메커니즘

#### 2) 교육 분야
- 학생 학습을 위한 상세한 추론 과정 제시
- 오류 분석 및 수정 메커니즘
- 개별 학습 경로 최적화

#### 3) 코드 생성 및 프로그래밍
- 복잡한 알고리즘 설계의 단계별 표현
- 논리 오류의 자동 감지
- 프로그램 합성(program synthesis) 향상

### 6.3 근본적 문제들에 대한 통찰

#### 1) MLLM의 일반화 문제

Mulberry는 다음을 시사합니다:

**가설 1: 명시적 추론은 일반화를 개선한다**
- 증거: 교차 모델 적응 +8.2%
- 설명: 추론 과정의 모듈화 및 재조합 가능

**가설 2: 집단 학습은 분포 외 성능을 개선한다**
- 부분적 증거: 다양한 도메인 포함
- 향후 검증 필요: 의도적 O.O.D 시나리오

#### 2) 반성과 자기 수정의 역할

Mulberry의 반성 메커니즘:
$$\text{오류 인식} \rightarrow \text{대안 탐색} \rightarrow \text{경로 수정}$$

이는 인지 과학의 "메타인지(metacognition)" 개념과 부합:
- 자신의 오류를 인식하고
- 다양한 해결책을 고려하고
- 최선의 경로를 선택

### 6.4 연구 커뮤니티에 대한 영향

#### 1) 데이터셋 기여
- **Mulberry-260k**: 공개 자원으로 제공되어 후속 연구 가능
- 트리 구조화된 추론 데이터의 중요성 입증
- 다양한 도메인 통합의 가치 증명

#### 2) 벤치마크 재평가의 필요성
현재의 벤치마크들:
- 직접 예측 능력만 평가
- 추론 과정의 품질은 미평가

**필요한 개선**:
- 단계별 정확성 평가 메트릭
- 추론 일관성 평가
- 오류 수정 능력 평가

***

## 7. 앞으로의 연구 시 고려할 점

### 7.1 이론적 고려사항

#### 1) 수렴성 분석(Convergence Analysis)
**미해결 문제**:
- CoMCTS의 수렴 조건은?
- 검색 반복 수는 어떻게 결정할 것인가?
- 최적 모델 조합의 이론적 기준은?

**제안**:

$$\text{Convergence Rate} = O\left(\frac{1}{\sqrt{n}} + \alpha \cdot \text{model diversity}\right)$$

#### 2) 일반화 경계(Generalization Bounds)
**문제**:
- 분포 내 성능과 분포 외 성능의 갭은?
- Rademacher complexity는 어떻게 감소하는가?

**필요한 분석**:

$$\mathcal{R}_D(h) = \mathbb{E}_D\left[\sup_{h \in H}\left|\hat{L}(h) - L(h)\right|\right]$$

#### 3) 정보 이론적 분석
**질문**:
- 각 추론 단계가 제공하는 정보 게인은?
- 최적 단계 수는 얼마인가?

$$\text{Information Gain} = H(Y) - \mathbb{E}[H(Y|s_i)]$$

### 7.2 방법론적 개선사항

#### 1) 하이브리드 접근
**아이디어**: CoMCTS + 강화학습(RL)의 결합

```
Single Model:     RL → CoT 자동 생성
Multiple Models:  Collective MCTS → 다양한 경로
Hybrid:           MCTS 경로 + RL 미세 조정
```

**예상 효과**: 
- 검색 효율성 유지 + RL의 유연성
- 성능: 추정치 +3-5%

#### 2) 적응적 집단 구성
**현재**: 고정된 4개 모델

**개선**:
- 작업 유형별 최적 모델 조합 선택
- 동적 가중치 조정
- 온라인 학습 메커니즘

$$\text{Model Weight} = w_i(t) = f(\text{task type}, \text{difficulty})$$

#### 3) 반성 메커니즘의 강화
**현재**: 형제 노드 기반 반성 (데이터의 ~5%)

**개선**:
- 대조 학습(contrastive learning)
- 반례 생성(counterfactual generation)
- 다중 반성 경로

### 7.3 응용 관련 고려사항

#### 1) 도메인별 특화
**의료 도메인**:
- 전문 의료 용어의 명시적 처리
- 안전성 평가 추가
- 확인 메커니즘 강화

**코딩 도메인**:
- 구문 정확성 보장
- 복잡도 분석 추가
- 테스트 케이스 기반 검증

#### 2) 계산 효율성 최적화
**문제**: 다중 모델 평가의 오버헤드

**해결책**:
```python
# 동적 조기 종료(early stopping)
if confidence(current_node) > threshold:
    continue  # 다음 단계로
else:
    require_full_evaluation()
```

#### 3) 배포 고려사항
**현실적 제약**:
- 여러 모델의 유지 비용
- 지연시간(latency) 요구사항
- 스케일링 문제

**해결**: 
- 단일 모델로의 증류(distillation)
- 캐싱 메커니즘
- 비동기 평가

### 7.4 평가 및 벤치마킹

#### 1) 새로운 평가 지표 필요

**현재 지표의 한계**:
- 최종 답변의 정확성만 평가
- 추론 과정은 무시

**제안하는 새로운 지표**:

1. **경로 일관성 점수**:
$$\text{Consistency}(Y) = \frac{1}{n} \sum_{i=1}^{n} \text{Coherence}(s_i, s_{i+1})$$

2. **오류 감지 능력**:
$$\text{Error Detection Rate} = \frac{\text{Detected Errors}}{\text{Total Errors}}$$

3. **자기 수정 성공률**:
$$\text{Self-Correction Success} = \frac{\text{Corrected Paths}}{\text{Total Error Paths}}$$

#### 2) 분포 외 일반화 평가
**필수 평가**:
- 새로운 도메인에서의 성능
- 분포 시프트 하에서의 건강성
- 극한 케이스(edge cases) 처리

### 7.5 더 근본적인 연구 문제

#### 1) "왜" 명시적 추론이 도움이 되는가?
**가설**:
- 표현 학습(representation learning) 개선
- 오류 회복 메커니즘(error recovery mechanism)
- 추상화 수준(level of abstraction) 증가

**필요한 실험**:
- 신경망 활성화 분석(neural activation analysis)
- 프로브 기반 평가(probing-based evaluation)
- 인지과학 연계 연구

#### 2) 최적 추론 길이의 개념
**문제**: 어떤 길이가 최적인가?

**모델**:
$$\text{Optimal Depth} = \arg\min_d [\text{Error}(d) + \lambda \cdot \text{Cost}(d)]$$

#### 3) 멀티모달 추론의 본질
**독특한 측면**:
- 시각 정보 통합 방식
- 모달리티 간 충돌 해결
- 크로스 모달 참조

***

## 결론

Mulberry 논문은 MLLM의 추론 능력을 향상시키기 위한 혁신적인 접근법을 제시합니다. 특히 **집단 몬테카를로 트리 탐색(CoMCTS)**을 통해 단일 모델의 한계를 극복하고, 명시적인 트리 구조의 추론 데이터를 활용함으로써 우수한 일반화 성능을 달성했습니다.

**핵심 기여**:
1. 여러 MLLM의 집단 지식을 활용한 효율적인 추론 경로 탐색
2. 260,000개의 구조화된 추론 데이터셋 구축
3. 교차 모델 일반화 (+8.2% 평균)
4. 동적 추론 길이 조정 능력

**앞으로의 연구는** 다음에 중점을 두어야 합니다:
- 이론적 수렴성 분석
- 분포 외 일반화 성능 평가
- 도메인 특화 적용
- 계산 효율성 최적화
- 새로운 평가 지표 개발

이러한 노력을 통해 MLLM의 추론 능력이 인간 수준에 가까워질 수 있을 것으로 기대됩니다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9][^1_90][^1_91]</span>

<div align="center">⁂</div>

[^1_1]: 2412.18319v2.pdf

[^1_2]: https://ieeexplore.ieee.org/document/11094933/

[^1_3]: https://arxiv.org/abs/2507.07424

[^1_4]: https://arxiv.org/abs/2503.13360

[^1_5]: https://arxiv.org/abs/2507.22940

[^1_6]: https://ieeexplore.ieee.org/document/11028406/

[^1_7]: https://academic.oup.com/ehjdh/article/doi/10.1093/ehjdh/ztaf143.056/8422997

[^1_8]: http://arxiv.org/pdf/2308.10379.pdf

[^1_9]: https://arxiv.org/html/2503.15944

[^1_10]: https://arxiv.org/pdf/2502.06807.pdf

[^1_11]: https://arxiv.org/pdf/2412.04645.pdf

[^1_12]: https://arxiv.org/html/2503.22732v1

[^1_13]: https://arxiv.org/html/2503.21614

[^1_14]: https://arxiv.org/pdf/2502.06772.pdf

[^1_15]: https://aclanthology.org/2023.emnlp-main.936.pdf

[^1_16]: https://blog.iese.edu/artificial-intelligence-management/2024/chain-of-thought-reasoning-the-new-llm-breakthrough/

[^1_17]: https://openreview.net/pdf?id=s4OukoHwnz

[^1_18]: https://openreview.net/forum?id=h3lyFa5e1W

[^1_19]: https://openai.com/index/learning-to-reason-with-llms/

[^1_20]: https://arxiv.org/abs/2405.00451

[^1_21]: https://neurips.cc/virtual/2025/poster/116679

[^1_22]: https://leehanchung.github.io/blogs/2024/10/08/reasoning-understanding-o1/

[^1_23]: https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/

[^1_24]: https://aclanthology.org/2025.findings-emnlp.1384.pdf

[^1_25]: https://www.lesswrong.com/posts/byNYzsfFmb2TpYFPW/o1-a-technical-primer

[^1_26]: https://aiflower.tistory.com/176

[^1_27]: https://liner.com/review/visionlanguage-models-can-selfimprove-reasoning-via-reflection

[^1_28]: https://simonwillison.net/2024/Sep/12/openai-o1/

[^1_29]: https://arxiv.org/html/2405.00451

[^1_30]: https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_MMAT-1M_A_Large_Reasoning_Dataset_for_Multimodal_Agent_Tuning_ICCV_2025_paper.pdf

[^1_31]: https://arxiv.org/html/2412.18319v1

[^1_32]: https://arxiv.org/html/2503.20757v1

[^1_33]: https://arxiv.org/html/2512.02456v1

[^1_34]: https://arxiv.org/html/2506.16962v2

[^1_35]: https://arxiv.org/pdf/2406.03816.pdf

[^1_36]: https://arxiv.org/pdf/2411.00855.pdf

[^1_37]: https://arxiv.org/html/2509.19003v1

[^1_38]: https://arxiv.org/html/2405.00451v2

[^1_39]: https://arxiv.org/abs/2412.18319

[^1_40]: https://arxiv.org/html/2507.07424v1

[^1_41]: https://arxiv.org/html/2406.03816v1

[^1_42]: https://arxiv.org/html/2506.01713v3

[^1_43]: https://arxiv.org/html/2505.05315v2

[^1_44]: https://arxiv.org/html/2508.19576v1

[^1_45]: https://arxiv.org/abs/2411.00855

[^1_46]: https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf

[^1_47]: https://www.ijcai.org/proceedings/2025/0965.pdf

[^1_48]: https://aclanthology.org/2025.findings-acl.484.pdf

[^1_49]: https://www.semanticscholar.org/paper/f0e4978448c668a78af77c6dc062c940ec8c9d62

[^1_50]: https://arxiv.org/abs/2501.01904

[^1_51]: https://ieeexplore.ieee.org/document/11092605/

[^1_52]: https://arxiv.org/abs/2501.05366

[^1_53]: https://arxiv.org/abs/2502.16033

[^1_54]: https://arxiv.org/abs/2502.01081

[^1_55]: https://arxiv.org/abs/2501.05444

[^1_56]: https://arxiv.org/abs/2502.11775

[^1_57]: https://arxiv.org/pdf/2411.14432.pdf

[^1_58]: https://arxiv.org/pdf/2501.06186.pdf

[^1_59]: http://arxiv.org/pdf/2503.14674.pdf

[^1_60]: http://arxiv.org/pdf/2501.01904.pdf

[^1_61]: https://arxiv.org/html/2406.17294v3

[^1_62]: https://arxiv.org/pdf/2311.05348.pdf

[^1_63]: http://arxiv.org/pdf/2412.06263.pdf

[^1_64]: https://arxiv.org/html/2503.10615v1

[^1_65]: https://huggingface.co/papers/2411.10440

[^1_66]: https://www.promptingguide.ai/techniques/tot

[^1_67]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_On_the_Out-Of-Distribution_Generalization_of_Large_Multimodal_Models_CVPR_2025_paper.pdf

[^1_68]: https://encord.com/blog/llava-o1-explained/

[^1_69]: https://arxiv.org/abs/2305.10601

[^1_70]: https://arxiv.org/abs/2402.06599

[^1_71]: https://aclanthology.org/2025.findings-acl.1247/

[^1_72]: https://process-mining.tistory.com/218

[^1_73]: https://openreview.net/forum?id=iA3kafgHGi

[^1_74]: https://arxiv.org/html/2411.10440v1

[^1_75]: https://www.emergentmind.com/topics/tree-of-thoughts-tot

[^1_76]: https://icml.cc/virtual/2025/50644

[^1_77]: https://www.chatpaper.ai/paper/72b8745a-bbc2-4631-9557-3bc723e5b76a

[^1_78]: https://arxiv.org/abs/2409.11527

[^1_79]: https://ieeexplore.ieee.org/document/11093370/

[^1_80]: https://arxiv.org/pdf/2412.18319.pdf

[^1_81]: https://arxiv.org/abs/2406.09136

[^1_82]: https://arxiv.org/pdf/2502.00577.pdf

[^1_83]: https://arxiv.org/pdf/2505.22334.pdf

[^1_84]: https://arxiv.org/pdf/2305.10601.pdf

[^1_85]: https://openaccess.thecvf.com/content/CVPR2024W/EvGenFM/papers/Verma_Evaluating_Multimodal_Large_Language_Models_Across_Distribution_Shifts_and_Augmentations_CVPRW_2024_paper.pdf

[^1_86]: https://arxiv.org/abs/2411.10440

[^1_87]: https://arxiv.org/pdf/2406.02746.pdf

[^1_88]: https://arxiv.org/html/2502.00577v1

[^1_89]: https://arxiv.org/html/2506.01078v1

[^1_90]: https://arxiv.org/html/2401.14295v5

[^1_91]: https://arxiv.org/html/2405.12217v1
