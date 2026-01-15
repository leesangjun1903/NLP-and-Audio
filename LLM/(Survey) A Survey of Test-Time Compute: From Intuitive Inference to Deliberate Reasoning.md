
# A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning

## 1. 핵심 주장과 주요 기여

"A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning"은 **테스트 타임 컴퓨트(test-time compute)**가 인공지능 모델의 성능을 향상시키는 근본적인 새로운 패러다임임을 주장합니다. 본 논문의 핵심은 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

### 1.1 핵심 주장

이 논문은 테스트 타임 컴퓨트가 **System-1에서 System-2로의 전환**을 가능하게 한다고 주장합니다. 기존의 빠르고 직관적인 System-1 모델(ResNet, BERT 등)은 분포 변화(distribution shift)에 취약했으나, 테스트 타임에 추가 계산 자원을 할당함으로써 모델을 느리고 신중한 System-2 사고 방식으로 전환할 수 있다는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

특히, OpenAI o1/o3, DeepSeek-R1, Gemini 2.5 같은 대규모 추론 모델(Large Reasoning Models, LRMs)의 성공은 **테스트 타임 컴퓨트 스케일링 효과**를 명확히 입증합니다. 추론 단계에서 더 많은 계산량을 할당할수록 성능이 지속적으로 향상되는 현상이 관찰되고 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

### 1.2 주요 기여

본 논문의 주요 기여는 다음과 같습니다:

1. **최초의 포괄적 체계화**: 테스트 타임 컴퓨트 방법을 System-1과 System-2라는 심리학적 프레임워크에 따라 **최초로 체계적으로 분류**한 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

2. **이중 단계 진화 모델**: 모델의 발전을 다음과 같이 구조화합니다:
   - **System-1 모델 → Test-Time Adaptation**: 파라미터 업데이트, 입력 수정, 표현 편집, 출력 보정
   - **System-2 모델 → Test-Time Reasoning**: 반복 샘플링, 자체 수정, 트리 탐색 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

3. **미래 방향 제시**: 일반화 능력, 멀티모달, 효율성, 스케일링 법칙 등 5가지 고급 주제를 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 해결하는 문제

#### 문제 1: 훈련 단계 스케일링의 한계

대규모 언어 모델(LLM)의 성능 향상은 전통적으로 다음에 의존했습니다:
- 모델 파라미터 수 증가
- 훈련 데이터 규모 확대
- 훈련 시간 연장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

그러나 이 접근법은 데이터 부족과 계산 자원 제약에 직면했습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

#### 문제 2: System-1 모델의 견고성 결여

기존의 심화 학습 모델들(ResNet, BERT, Transformer 등)은:
- 분포 변화(distribution shift)에 취약
- 대적 사례(adversarial examples)에 민감
- 복잡한 추론 작업에 어려움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

#### 문제 3: System-2 사고의 제약

Chain-of-Thought(CoT) 프롬프팅으로도 달성되는 System-2 사고는:
- 오류 누적(error accumulation)에 의한 성능 저하
- 선형적 추론 패턴에 국한됨
- 인간의 비선형적 인지 과정(브레인스토밍, 성찰, 역추적)을 완전히 재현하지 못함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

### 2.2 제안하는 방법

#### 2.2.1 System-1을 위한 Test-Time Adaptation (TTA)

**A. 파라미터 업데이트(Parameter Updating)**

모델의 파라미터를 테스트 단계에서 업데이트하여 테스트 분포에 적응시킵니다.

**학습 신호 설계:**

$$\mathcal{L}_{auxiliary} = \alpha \mathcal{L}_{main} + (1-\alpha) \mathcal{L}_{aux}$$

여기서 보조 작업(auxiliary task)으로는 다음을 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- 회전 예측(rotation prediction)
- 마스크 자동 인코딩(masked autoencoding)
- 대조 학습(contrastive learning)

특히 엔트로피 최소화는 불확실성을 학습 신호로 활용합니다:

$$\mathcal{L}_{entropy} = -\sum_c p_\theta(c|x) \log p_\theta(c|x)$$

**파라미터 효율화:**

정규화 층(normalization layers), 소프트 프롬프트(soft prompts), 저순위 모듈(low-rank modules), 어댑터 모듈(adapter modules)만 업데이트하여 효율성을 향상시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**B. 입력 수정(Input Modification)**

In-Context Learning(ICL) 능력을 활용하여 시연(demonstrations)을 최적화합니다.

**시연 선택(Demonstration Selection):**

$$\text{relevance}(d, x) = \text{similarity}(\text{embed}(d), \text{embed}(x))$$

다음과 같은 고급 선택 전략을 제안합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- 정보 이론 기반: 최대 국소 엔트로피, 최소 기술 길이(MDL)
- 강화 학습 기반: 시연 선택을 순차 결정 문제로 모델링

**시연 생성(Demonstration Creation):**

모델의 생성 능력을 활용하여 시연을 직접 생성합니다:
- DAIL: 시연 메모리 구축
- DAWN-ICL: 테스트 샘플 순회 순서를 MCTS로 최적화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**C. 표현 편집(Representation Editing)**

중간 층의 표현(intermediate representation)을 직접 조정합니다.

**대조 프롬프트 기반 방법:**

$$\mathbf{v}_{steer} = \text{embed}(P_{positive}) - \text{embed}(P_{negative})$$

편집된 표현:

$$\mathbf{h}_{edited} = \mathbf{h}_{original} + \lambda \mathbf{v}_{steer}$$

여기서 $\lambda$는 조종 강도(steering intensity)입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**D. 출력 보정(Output Calibration)**

외부 정보로 모델의 출력 분포를 보정합니다.

**kNN 기계 번역(kNN-MT) 방식:**

$$p_{calibrated}(y|x) = \frac{\alpha p_{model}(y|x) + (1-\alpha) p_{kNN}(y|x)}{\sum_y [\alpha p_{model}(y|x) + (1-\alpha) p_{kNN}(y|x)]}$$

데이터 저장소에서 검색한 $k$-최근접 이웃의 확률과 모델 확률을 융합합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

#### 2.2.2 System-2를 위한 Test-Time Reasoning (TTR)

**A. 피드백 모델링(Feedback Modeling)**

**점수 기반 피드백 (Score-based Feedback):**

1. **결과 기반 검증자(ORM: Outcome Reward Model)**
   - 최종 답변의 정확성만 평가
   - 학습 신호: $r = \mathbb{1}[\text{answer} = \text{ground truth}]$

2. **과정 기반 검증자(PRM: Process Reward Model)**
   - 각 추론 단계의 정확성 평가
   - 학습 신호: $r_t = \mathbb{1}[\text{step}_t \text{ is correct}]$

**생성 기반 피드백(Generative-based Feedback):**

LLM-as-a-Judge 방식으로 자연어 비판을 생성합니다:

$$\text{critique} = \text{LLM}(\text{prompt}, \text{response})$$

특히 Prometheus는 구조화된 평가 기준을 사용합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

$$p(\text{score}) = \text{softmax}(\text{LLM}(\text{criteria}, \text{response}))$$

**B. 탐색 전략(Search Strategies)**

**1. 반복 샘플링(Repeated Sampling)**

다중 후보를 병렬로 샘플링합니다:

$$\{y_1, y_2, ..., y_K\} \sim p_\theta(\cdot|x)$$

**검증 전략:**

- **Majority Voting**: 최빈 답변 선택
  
  $$\hat{y} = \arg\max_y \sum_{i=1}^K \mathbb{1}[y_i = y]$$

- **Best-of-N (BoN) Sampling**: 검증자 점수로 최상 답변 선택
  
  $$\hat{y} = \arg\max_y r(y)$$

Self-Consistency CoT는 이 방법으로 **18% 성능 향상**을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**2. 자체 수정(Self-Correction)**

순차적으로 답변을 반복 개선합니다.

**피드백 소스:**

- 도구 확인(tool checking): 컴파일러 피드백(코드 생성)
- 외부 모델 평가(external model evaluation): 비판 모델 사용
- 내재 피드백(intrinsic feedback): 자체 비판 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**자체 비판 메커니즘:**

$$y_{refined} = \text{LLM}(\text{"Let's think again..."}, y_{initial}, \text{critique}(y_{initial}))$$

**3. 트리 탐색(Tree Search)**

복합적인 계획 문제를 위해 계층적 탐색을 수행합니다.

**선택된 탐색 알고리즘:**

$$\text{Value}(s) = r(s) + \gamma \mathbb{E}[V(s')]$$

**무정보 탐색(Uninformed Search):**
- Tree-of-Thought (ToT): BFS/DFS 기반 탐색
- Beam Search: 상위 K개 경로만 유지

**휴리스틱 탐색(Heuristic Search):**

**Monte Carlo Tree Search (MCTS):**
1. 선택(Selection): UCB 공식으로 경로 선택
   
   $$\text{UCB}(s) = \frac{Q(s)}{N(s)} + C\sqrt{\frac{\ln N(parent)}{N(s)}}$$

2. 전개(Expansion): 새로운 노드 추가
3. 시뮬레이션(Simulation): 랜덤 롤아웃
4. 역전파(Backpropagation): 결과를 거슬러 올라가며 업데이트 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**값 함수(Value Function):**

$$V(s) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s]$$

RAP는 다중 휴리스틱 값 함수를 결합합니다:
- 행동의 우도(likelihood)
- 상태의 신뢰도(confidence)
- 자체 평가 결과
- 작업별 보상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

***

## 3. 모델 구조

### 3.1 System-1에서 System-2로의 전환

논문은 다음의 계층적 모델 구조를 제시합니다:

```
System-1 모델
├─ 직접 답변 생성 (낮은 지연시간)
└─ 분포 변화에 취약

   ↓ Test-Time Adaptation 적용

약한 System-2 모델
├─ CoT 프롬프팅 활성화
├─ 암묵적 느린 사고
└─ 선형 추론 패턴

   ↓ Test-Time Reasoning 적용 (RL 훈련)

강한 System-2 모델
├─ 반복 샘플링/자체 수정/트리 탐색
├─ 명시적 느린 사고 (Chain-of-Thought)
└─ 비선형 추론: 브레인스토밍, 성찰, 역추적
```

### 3.2 피드백 모델링 아키텍처

```
피드백 모델링
├─ 점수 기반 피드백 (Scoring)
│  ├─ ORM: 이진 분류 (정답/오답)
│  └─ PRM: 단계별 분류
└─ 생성 기반 피드백 (Critique)
   ├─ 폐쇄형: GPT-4, Claude (비용 높음)
   ├─ 오픈소스: Prometheus (SFT 훈련)
   └─ 자체 비판: 모델 자신이 생성
```

### 3.3 탐색 전략 조합

논문은 세 가지 탐색 전략의 특성을 대조합니다:

| 전략 | 병렬성 | 구현 복잡도 | 계산 효율 | 적합 작업 |
|------|--------|----------|---------|---------|
| **반복 샘플링** | 높음 | 낮음 | 중간 | 검증 가능한 작업 (수학, 코드) |
| **자체 수정** | 낮음 | 중간 | 중간 | 쉽게 검증 가능한 작업 |
| **트리 탐색** | 중간 | 높음 | 낮음 | 복잡한 계획 문제 | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상 결과

#### 4.1.1 System-1 TTA 방법의 성능

**파라미터 업데이트:**
- Tent: ImageNet-C에서 분포 변화 시 성능 유지
- MEMO: 단일 샘플 TTA에서 Tent보다 안정적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**입력 수정:**
- EPR, UDR: 시연 검색 기반 선택
- DAIL: 시연 메모리로 순환 샘플에 대한 성능 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**출력 보정:**
- kNN-MT: 기계 번역의 교차 도메인 및 다중언어 작업에서 우수한 전달성
- Bi-kNN: 성능 및 효율성 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

#### 4.1.2 System-2 TTR 방법의 성능

**반복 샘플링:**
- Self-Consistency CoT: 기본 CoT 대비 **18% 정확도 향상** (수학 추론) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)
- DeepSeek-R1-Zero: AIME 2024에서 **15.6% → 71.0%** (다수결 투표로 86.7%) [nature](https://www.nature.com/articles/s41586-025-09422-z)

**자체 수정:**
- Self-Refine, RCI: 반복적 프롬프팅으로 상당한 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)
- Reflexion: QA 작업에서 의미 있는 성능 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

**트리 탐색:**
- AlphaMath: MCTS 기반 탐색으로 수학 벤치마크 성능 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)
- ReST-MCTS*: 고품질 추론 궤적 수집으로 정책 모델 및 보상 모델 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

#### 4.1.3 최신 추론 모델의 성과 (2020년 이후)

| 모델 | 출시 | AIME 성능 | MATH-500 | 주요 특징 |
|------|------|----------|----------|---------|
| **GPT-3** | 2020 | - | - | CoT 기초 확립 |
| **Self-Consistency CoT** | 2022 | - | - | 다중 샘플링 및 다수결 투표 |
| **OpenAI o1-preview** | 2024.9 | ~83% | 96%+ | Chain-of-Thought 스케일링 |
| **OpenAI o3-mini** | 2024.12 | 96%+ | 99%+ | 향상된 테스트-타임 스케일링 |
| **DeepSeek-R1-Zero** | 2025.1 | 71% → 86.7%\* | 97.3% | 순수 RL 기반, 추론 길이 자동 증가 |
| **DeepSeek-R1-Distill-32B** | 2025.1 | 72.6% | 94.3% | R1 증류, o1-mini 수준 |
| **QwQ-32B** | 2024.12 | 80.3% | - | 모듈식 추론 구조 | [arxiv](https://arxiv.org/pdf/2501.12948.pdf)

### 4.2 일반화 성능 향상 분석

#### 4.2.1 도메인 간 일반화 가능성

최신 연구(2024-2025)는 흥미로운 결과를 보여줍니다: [jjeongil.tistory](https://jjeongil.tistory.com/3268)

**RL 훈련의 강력한 전달성:**

수학 데이터로만 훈련한 RL 모델이 다른 도메인으로 일반화됨:
- 예: 논리 퍼즐 데이터로 훈련 후 AIME에서 **125% 성능 향상**, AMC에서 **38% 향상** [reddit](https://www.reddit.com/r/ArtificialInteligence/comments/1jwvhng/research_shows_that_reasoning_models_generalize/)

**SFT vs RL의 차이:**

| 메서드 | 수학 성능 | 타 도메인 전달 | 메커니즘 |
|--------|----------|--------------|---------|
| **SFT** | 높음 | 낮음 | 표현 드리프트 발생 |
| **RL** | 높음 | 높음 | 일반적 추론 능력 습득 | [arxiv](https://arxiv.org/pdf/2507.00432.pdf)

**PRM의 교차 도메인 일반화:**

- 수학 데이터로 훈련된 PRM이 코드 생성에서도 우수한 성능 발휘
- 최근 ContextPRM: 논리적 흐름에 초점을 맞춰 MMLU-Pro의 9개 비수학 도메인에서 **6.5% 평균 정확도 향상** 달성 [arxiv](https://arxiv.org/abs/2509.24460)

**다언어 추론 스케일링:**

- 영어 중심 추론 모델도 다른 언어로 일반화 가능
- 일부 저자원 언어에서는 여전히 한계 존재 [arxiv](https://arxiv.org/abs/2505.05408)

#### 4.2.2 한계 분석

**2021 요약(Summary 1):** System-1 TTA 방법의 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- **파라미터 업데이트**: LLM에서 훈련 불안정성과 비효율성에 시달림
- **출력 보정**: 대상 도메인 정보 의존 및 지식 유출 위험
- **입력 수정**: ICL 능력에만 의존 → 제한적 적용성
- **표현 편집**: 수동 사전 지식 필요 → 확장성 제한

**2021 요약(Summary 2):** System-2 TTR 방법의 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- **반복 샘플링**: 구현은 쉽지만 계산 비효율 (실제 적용 어려움)
- **자체 수정**: 정밀한 세밀 피드백 필요; 피드백 불량 또는 약한 추론 능력 시 성능 저하
- **트리 탐색**: 복잡한 계획 문제에 최적화되나 구현 복잡도 높음

### 4.3 근본적 한계 및 미해결 문제

#### 4.3.1 역스케일링(Inverse Scaling) 현상 [discuss.pytorch](https://discuss.pytorch.kr/t/test-time-computing-inverse-scaling-in-test-time-compute/7338)

최근 발견: **더 많이 생각할수록 오히려 성능이 나빠지는 현상**

- o1 계열 모델: CoT 길이가 과도하면 "과도한 사고(overthinking)"로 정확한 답을 틀린 답으로 수정
- 이를 극복하기 위한 추론 길이 예산 제어(L1, Elastic Reasoning)가 등장 [arxiv](https://arxiv.org/abs/2408.03314)

#### 4.3.2 검증자의 도메인 특이성 [arxiv](https://arxiv.org/abs/2507.09884)

- 수학용으로 훈련된 검증자가 다른 도메인에서 높은 민감도를 보임
- 교차 도메인 일반화에 구조적 한계 존재

#### 4.3.3 의료 도메인의 한계 [arxiv](https://arxiv.org/abs/2504.00869)

- 의료 추론은 수학과 근본적으로 다름 (지식 표현, 의사결정 방식)
- 약 4K 토큰 예산까지만 성능 향상, 초과하면 오히려 성능 저하

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 성능 향상의 주요 경로

#### 경로 1: 다도메인 RL 훈련

**General-Reasoner 접근:** [arxiv](https://arxiv.org/pdf/2505.14652.pdf)

웹 크롤링으로 수집한 다양한 도메인의 검증 가능한 질문으로 RL 훈련:
- 수학, 물리, 화학, 금융, 인문학 등 포괄
- 생성 기반 검증자로 다양한 답변 형식 지원

**결과:** 단일 도메인 훈련보다 우수한 일반화, 특히 수학 벤치마크에서 경쟁력 유지 [arxiv](https://arxiv.org/pdf/2505.14652.pdf)

#### 경로 2: 추상적 추론 프로토타입 활용

**ProtoReasoning 접근:** [xugj520](https://www.xugj520.cn/en/archives/cross-domain-reasoning-llms-abstract-prototypes.html)

Prolog/PDDL 기반 추상 추론 구조로 도메인 노이즈 제거:
- 표면적 차이(수학 기호 vs 자연어)를 초월한 보편적 논리 구조 포착
- 10배 이상 표본 효율성 향상

#### 경로 3: 도메인별 최적화된 검증자

**ContextPRM:** [arxiv](https://arxiv.org/abs/2509.24460)

도메인 특정 지식 대신 단계 간의 맥락적 일관성에 초점:
- 논리적 흐름이 모든 도메인에서 공통적
- MMLU-Pro 비수학 도메인에서 6.5% 평균 향상

### 5.2 일반화 성능의 한계와 과제

#### 한계 1: 개념적 추론 작업 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

현재 LRM들은 명확한 답변 형식이 필요한 작업에 편향:
- STEM 문제: 잘 수행 (검증 가능)
- 문화적 상식: 성능 저하 (도메인 외 일반화 약함) [arxiv](https://arxiv.org/abs/2505.05408)

#### 한계 2: 약한 모델의 한계

크기 7B 미만 모델:
- 기본 추론 능력 부족으로 RL의 이점 제한적
- 미세 조정(fine-tuning)으로만 부분적 개선 가능 [arxiv](https://arxiv.org/abs/2504.00869)

#### 한계 3: 검증자 설계의 복잡성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- 수학: 규칙 기반 검증자 (단순, 정확)
- 의료/법률: 모델 기반 검증자 필요 (복잡, 불완전)
- 고도의 해석이 필요한 작업: 검증 불가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

### 5.3 미래 일반화 개선 방향

#### 방향 1: 약한-강 일반화(Weak-to-Strong Generalization) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

인간이 완벽한 피드백을 제공할 수 없는 미지의 영역(수학 추측 증명, 과학적 발견)에서도 모델이 학습 가능하도록:
- 약한 감시 신호로부터 강한 능력 유도
- 과학적 문제 해결의 새로운 경로 열림

#### 방향 2: 멀티모달 일반화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- MM-CoT, VisualPRM: 시각 정보 통합
- OpenVLThinker, Vision-R1: 이미지-텍스트 일관성
- 각 모달리티별 검증자 개발 필요

#### 방향 3: 검색 보강 추론(RAG + Test-Time Reasoning) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

- Search-o1, Search-R1: 외부 도구 활용
- Deep Research: 웹 검색, 코드 실행 통합
- **일반 도메인 작업의 성능 대폭 향상 가능**

#### 방향 4: 스케일링 법칙의 보편화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

현재 미해결: 테스트-타임 컴퓨트의 보편적 스케일링 법칙

- Brown et al.: 반복 샘플링은 대략 로그-선형 관계
  $$\text{accuracy} \approx a + b \log(N)$$
  
- Chen et al.: 다중 샘플링 실패율이 거듭제곱 법칙 따름
  $$P(\text{failure}) \propto N^{-\alpha}$$

**미해결 과제:**
- 여러 전략의 통합 프레임워크 부재
- 표본 어려움, 피드백 신호 정확도, 디코딩 하이퍼파라미터의 영향 정량화 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf)

***

## 6. 논문의 영향과 향후 연구 시 고려사항

### 6.1 학문적 영향

#### 영향 1: 패러다임 전환 [arxiv](https://arxiv.org/html/2501.12948v1)

**전통적 AI 스케일링:**
$$\text{성능} = f(\text{모델 크기}, \text{훈련 데이터}, \text{훈련 시간})$$

**테스트-타임 컴퓨트 스케일링 (새로운 패러다임):**
$$\text{성능} = f(\text{추론 시 계산량}, \text{추론 깊이}, \text{검색 전략})$$

이는 AI의 능력 향상을 위한 **완전히 새로운 축**을 제시합니다. [cameronrwolfe.substack](https://cameronrwolfe.substack.com/p/llm-scaling-laws)

#### 영향 2: 작은 모델의 가능성 [arxiv](https://arxiv.org/abs/2502.06703)

테스트-타임 스케일링을 통해:
- 0.5B 모델이 GPT-4o 능가 (compute-optimal TTS 전략)
- 3B 모델이 405B 모델 초과
- 경제성과 효율성 혁신 [arxiv](https://arxiv.org/abs/2502.06703)

#### 영향 3: 강화 학습의 재조명 [magazine.sebastianraschka](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)

DeepSeek-R1의 성공으로 **순수 RL(사전 학습만으로)의 가능성** 입증:
- SFT 데이터 없이도 강력한 추론 능력 획득 가능
- 산업계에 새로운 훈련 패러다임 제시

### 6.2 향후 연구 시 고려사항

#### 고려사항 1: 검증자의 일반화 능력 강화

**필요한 작업:**
- 도메인별 특화 검증자 개발
- 또는 도메인 불가지론적(domain-agnostic) 검증 신호 발견
- 특히 **맥락 일관성** 기반 접근이 유망 [arxiv](https://arxiv.org/abs/2509.24460)

#### 고려사항 2: 테스트-타임 연산 효율성

**문제점:**
- o1 같은 모델은 단순 산술(2+3)도 "과도하게 생각"
- 이는 비용 낭비 및 실제 배포 어려움 [newsletter.semianalysis](https://newsletter.semianalysis.com/p/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures)

**해결 방안:**
- 문제 난이도 예측 모듈 개발
- 적응형 연산 예산 할당
- 초기 종료(early exit) 메커니즘

#### 고려사항 3: 크로스 도메인 스케일링 법칙

**미개척 영역:**
- 각 도메인-모델 쌍에 따른 최적 테스트-타임 전략 정의
- 통합 프레임워크 개발

**접근법:**
- 포괄적인 벤치마크 구축 (현재 MMLU-Pro, ProcessBench 등이 초기 단계)
- 메타 학습으로 새로운 도메인에 신속 적응

#### 고려사항 4: 해석 가능성과 신뢰성

**현안:**
- 장시간 추론 과정에서 왜 그렇게 결정했는가? (블랙박스 문제)
- 중간 단계의 오류를 정확히 감지하는가?

**해법:**
- 단계별 중요도 분석 (feature attribution)
- 회로 기반 분석(CRV)으로 추론 실패의 근본 원인 파악 [arxiv](https://arxiv.org/html/2510.09312v1)

#### 고려사항 5: 윤리 및 안전

**위험:**
- 더 오래 추론할수록 할루시네이션 누적 가능성
- 편향된 시선의 강화

**필요:**
- 장시간 추론 시 안전성 평가 메트릭
- 윤리적 추론 경로 유도 메커니즘

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 시간 전개에 따른 연구 진화

#### Phase 1 (2020-2022): CoT 기초 확립

| 연구 | 연도 | 주요 기여 |
|------|------|----------|
| **Chain-of-Thought Prompting** | 2022 | 단계별 추론의 유효성 입증 |
| **Self-Consistency** | 2022 | 다중 샘플링 + 다수결 투표로 18% 향상 | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **In-Context Learning** | 2020-2022 | 데모 선택의 중요성 | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |

#### Phase 2 (2023-2024 초): 검증자 등장

| 연구 | 연도 | 주요 기여 |
|------|------|----------|
| **Outcome Reward Models** | 2021 | 최종 정답 기반 학습 신호 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **Process Reward Models** | 2023 | 단계별 감독 신호 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **Math-Shepherd** | 2023 | MCTS로 자동 과정 감독 데이터 수집 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **ORM 암묵적 PRM** | 2024 | 결과 라벨로 과정 감독 데이터 자동 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |

#### Phase 3 (2024 후반-2025): 대규모 추론 모델 시대

| 모델 | 릴리스 | 특징 |
|------|--------|------|
| **OpenAI o1** | 2024.9 | 테스트-타임 CoT 스케일링의 성공 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **DeepSeek-R1-Zero** | 2025.1 | 순수 RL로 강력한 추론 능력 습득 [arxiv](https://arxiv.org/html/2501.12948v1) |
| **DeepSeek-R1** | 2025.1 | 콜드스타트 + RL로 o1 수준 달성 [arxiv](https://arxiv.org/html/2501.12948v1) |
| **OpenAI o3** | 2025 | 96% AIME 달성 (o1 대비 향상) [nature](https://www.nature.com/articles/s41586-025-09422-z) |
| **QwQ-32B** | 2024.12 | 모듈식 추론 구조로 80% AIME [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e01eea41-34fd-4231-81c9-b06a15f7766f/2501.02497v3.pdf) |
| **Kimi k1.5** | 2025.1 | 확장 가능 규모(최대 32,768토큰) |

### 7.2 핵심 기술 비교

#### 테스트-타임 전략의 진화

```
시간 →

2022-2023: Self-Consistency (병렬 샘플링)
            └─ 단순하지만 비효율

2023-2024: Tree-of-Thought + MCTS (휴리스틱 탐색)
            └─ 효율 향상, 구현 복잡

2024: 강화 학습 기반 자동 추론 스케일링 (o1/R1)
      └─ 모델이 자체적으로 최적 사고 길이 학습
      └─ 최고 효율성
```

#### 검증자 진화

```
시간 →

2021: ORM (이진 분류, 단순)
      └─ 정확하지만 오류 진단 불가

2023: PRM (단계별 분류)
      └─ 정확하고 진단 가능, 높은 비용

2024: 암묵적 PRM (결과 라벨로 과정 추론)
      └─ 주석 비용 대폭 감소, 효율성 향상

2024-2025: 생성 기반 비평가 (LLM-as-Critic)
           └─ 자연어 설명으로 해석성 향상
           └─ 추가 학습 데이터 활용 가능
```

### 7.3 성능 비교 (벤치마크)

#### AIME 2024 성능 추이

```
달성도 (%)
  ↑
100 │                          ╔═══════════╗
    │                          ║ OpenAI o3 │ (96%+)
 80 │           ╔══════════════╗
    │           ║ DeepSeek-R1  │ (79.8%)
 60 │           ║             ║
    │    ╔══════╝             ║
 40 │    ║ o1-preview (83%)   ║
    │    ║                    ║
 20 │ ◎─╫─────────────────────╫──── 인간 평균 (AIME 우수자)
    │  base model
  0 └────┴────┴────┴────┴────┴──→ 시간
       2023 2024-초 2024-말 2025
```

#### 다양 도메인 성능 비교 (2025)

| 벤치마크 | o1 | o3 | DeepSeek-R1 | QwQ-32B | 설명 |
|---------|-----|-----|--------------|---------|------|
| **AIME 2024** | 83% | 96%+ | 79.8% | 80.3% | 수학 경쟁 문제 |
| **MATH-500** | 96%+ | 99%+ | 97.3% | - | 수학 교과서 문제 |
| **LiveCodeBench** | 88.5% (o1-high) | - | - | 57.2% | 코드 생성 |
| **GPQA (과학)** | 95%+ | - | 82% | - | 대학원 과학 |
| **GSM8K** | 99%+ | - | - | - | 초등학교 수학 |

**관찰:**
1. 수학에서 모든 추론 모델이 인간 능력 초과
2. 과학에서 아직 개선 여지 있음
3. 코드에서 o1이 여전히 우수 (RL 훈련 강점)

### 7.4 핵심 발견: 2020년 이후 진전 사항

#### 발견 1: RL의 우월성 [arxiv](https://arxiv.org/pdf/2507.00432.pdf)

**연구:** "Does Math Reasoning Improve General LLM Capabilities?" [arxiv](https://arxiv.org/pdf/2507.00432.pdf)

- **SFT 모델**: 수학 성능은 향상하나 타 도메인 능력 소실
- **RL 모델**: 수학 성능 향상 + 타 도메인 일반화 우수

**메커니즘:** 
- SFT는 표현 드리프트 유발
- RL은 일반적 추론 능력을 학습

#### 발견 2: 검증자의 교차 도메인 일반화 [arxiv](https://arxiv.org/abs/2509.24460)

**연구:** ContextPRM [arxiv](https://arxiv.org/abs/2509.24460)

- 논리적 일관성에 초점 → 모든 도메인 적용 가능
- MMLU-Pro 비수학 도메인: 평균 6.5% 향상

#### 발견 3: 약한 모델도 가능 [arxiv](https://arxiv.org/abs/2502.06703)

**연구:** "Can 1B LLM Surpass 405B LLM?" [arxiv](https://arxiv.org/abs/2502.06703)

- compute-optimal TTS 전략으로 극적인 성능 역전
- 크기는 문제 난이도마다 최적값 상이

#### 발견 4: 역스케일링 현상 [discuss.pytorch](https://discuss.pytorch.kr/t/test-time-computing-inverse-scaling-in-test-time-compute/7338)

**문제:** 추론을 너무 오래 하면 성능 저하

**해결책:**
- 길이 기반 보상 (L1, Elastic Reasoning)
- 조기 종료 메커니즘 (DEER)
- 적응형 예산 할당 [arxiv](http://arxiv.org/pdf/2502.12215.pdf)

***

## 결론

이 논문은 **테스트-타임 컴퓨트가 AI의 미래를 결정하는 핵심 축**이 될 것임을 강력하게 주장합니다. 2020년부터 2025년까지의 진전을 보면:

1. **개념적 진화**: Chain-of-Thought → 다중 샘플링 → 검증자 기반 탐색 → RL 기반 자동 스케일링
2. **성능 비약**: 기본 모델의 40%에서 추론 모델의 96% AIME까지
3. **효율성 개선**: 큰 모델 불필요 → 작은 모델도 충분한 테스트-타임 컴퓨트로 경쟁 가능

**향후 과제:**
- 보편적 스케일링 법칙 정립
- 도메인 일반화 능력 강화
- 검증자 설계의 표준화
- 효율과 성능의 균형점 발견

이 논문은 AI 연구의 **새로운 지평**을 열었으며, 업계는 이미 이를 실제 제품(o1, o3, R1)으로 구현하고 있습니다.

***

## 참고문헌
<span style="display:none">[^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88]</span>

<div align="center">⁂</div>

[^1_1]: 2501.02497v3.pdf

[^1_2]: https://www.nature.com/articles/s41586-025-09422-z

[^1_3]: https://arxiv.org/pdf/2501.12948.pdf

[^1_4]: https://arxiv.org/html/2501.12948v1

[^1_5]: https://jjeongil.tistory.com/3268

[^1_6]: https://openreview.net/pdf?id=8PItTWQ5Qa

[^1_7]: https://arxiv.org/pdf/2507.00432.pdf

[^1_8]: https://www.reddit.com/r/ArtificialInteligence/comments/1jwvhng/research_shows_that_reasoning_models_generalize/

[^1_9]: https://arxiv.org/abs/2509.24460

[^1_10]: https://arxiv.org/abs/2505.05408

[^1_11]: https://discuss.pytorch.kr/t/test-time-computing-inverse-scaling-in-test-time-compute/7338

[^1_12]: https://arxiv.org/abs/2408.03314

[^1_13]: https://arxiv.org/abs/2507.09884

[^1_14]: https://arxiv.org/abs/2504.00869

[^1_15]: https://arxiv.org/pdf/2505.14652.pdf

[^1_16]: https://arxiv.org/html/2505.14652v1

[^1_17]: https://www.xugj520.cn/en/archives/cross-domain-reasoning-llms-abstract-prototypes.html

[^1_18]: https://cameronrwolfe.substack.com/p/llm-scaling-laws

[^1_19]: https://arxiv.org/abs/2502.06703

[^1_20]: https://arxiv.org/abs/2502.12215

[^1_21]: https://arxiv.org/abs/2502.14382

[^1_22]: https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

[^1_23]: https://newsletter.semianalysis.com/p/scaling-laws-o1-pro-architecture-reasoning-training-infrastructure-orion-and-claude-3-5-opus-failures

[^1_24]: https://arxiv.org/html/2510.09312v1

[^1_25]: https://arxiv.org/abs/2504.00891

[^1_26]: http://arxiv.org/pdf/2502.12215.pdf

[^1_27]: https://arxiv.org/html/2512.02008v1

[^1_28]: http://proceedings.mlr.press/v119/sun20b/sun20b.pdf

[^1_29]: https://www.youtube.com/watch?v=CiKuTVx9R_o

[^1_30]: https://pdfs.semanticscholar.org/0360/c5a5691c530e040c8230200f6dfbcb1a2c51.pdf

[^1_31]: https://arxiv.org/abs/2503.19855

[^1_32]: https://arxiv.org/abs/2502.18418

[^1_33]: https://aclanthology.org/2025.findings-emnlp.742

[^1_34]: https://arxiv.org/abs/2503.16040

[^1_35]: https://arxiv.org/abs/2504.01317

[^1_36]: https://arxiv.org/abs/2503.23803

[^1_37]: https://arxiv.org/abs/2504.10449

[^1_38]: http://arxiv.org/pdf/2502.14382.pdf

[^1_39]: https://arxiv.org/pdf/2502.18080.pdf

[^1_40]: https://arxiv.org/html/2504.01317v1

[^1_41]: http://arxiv.org/pdf/2501.19393.pdf

[^1_42]: https://arxiv.org/pdf/2504.00810v1.pdf

[^1_43]: https://arxiv.org/pdf/2410.13639.pdf

[^1_44]: https://blogs.nvidia.com/blog/ai-scaling-laws/

[^1_45]: https://www.reddit.com/r/MachineLearning/comments/1g3dqof/d_the_difference_between_outcomesupervised_reward/

[^1_46]: https://arxiv.org/abs/2505.14069

[^1_47]: https://magazine.sebastianraschka.com/p/understanding-reasoning-llms

[^1_48]: https://www.stephendiehl.com/posts/process_reward/

[^1_49]: https://aclanthology.org/2025.findings-emnlp.742.pdf

[^1_50]: https://www.emergentmind.com/topics/test-time-scaling-law

[^1_51]: https://rlhfbook.com/c/07-reward-models.html

[^1_52]: https://discuss.pytorch.kr/t/deep-research-test-time-compute-test-time-scaling/6153

[^1_53]: https://www.tanayj.com/p/openais-o-1-and-inference-time-scaling

[^1_54]: https://rlhfbook.com/c/07-reward-models

[^1_55]: https://arxiv.org/html/2507.14419v1

[^1_56]: https://arxiv.org/html/2504.00294v1

[^1_57]: https://arxiv.org/html/2507.01551v2

[^1_58]: https://arxiv.org/pdf/2505.11484.pdf

[^1_59]: https://arxiv.org/pdf/2501.07301.pdf

[^1_60]: https://arxiv.org/html/2502.11514v1

[^1_61]: https://www.arxiv.org/pdf/2510.08049.pdf

[^1_62]: https://arxiv.org/html/2502.12215v1

[^1_63]: https://arxiv.org/html/2505.21825v2

[^1_64]: https://arxiv.org/pdf/2305.20050.pdf

[^1_65]: https://arxiv.org/html/2503.19855v1

[^1_66]: https://arxiv.org/html/2505.11484v1

[^1_67]: https://arxiv.org/abs/2501.07301

[^1_68]: https://arxiv.org/abs/2502.11514v2

[^1_69]: https://arxiv.org/abs/2503.19948

[^1_70]: https://ieeexplore.ieee.org/document/10981139/

[^1_71]: https://arxiv.org/abs/2505.24863

[^1_72]: https://arxiv.org/abs/2506.00027

[^1_73]: https://arxiv.org/abs/2507.06999

[^1_74]: https://arxiv.org/abs/2505.18283

[^1_75]: https://www.semanticscholar.org/paper/55ff230e8af72f757d17fcade0655b02c51b4be7

[^1_76]: https://arxiv.org/pdf/2411.17869.pdf

[^1_77]: http://arxiv.org/pdf/2304.04494.pdf

[^1_78]: http://arxiv.org/pdf/1710.03463.pdf

[^1_79]: https://arxiv.org/html/2503.06288v1

[^1_80]: https://arxiv.org/pdf/2203.04600.pdf

[^1_81]: http://arxiv.org/pdf/2408.09138.pdf

[^1_82]: http://arxiv.org/pdf/2308.09931.pdf

[^1_83]: https://arxiv.org/pdf/2302.02609.pdf

[^1_84]: https://seohyun00.tistory.com/44

[^1_85]: https://tiger-ai-lab.github.io/General-Reasoner/

[^1_86]: https://openaccess.thecvf.com/content/WACV2025/papers/Sui_Just_Shift_It_Test-Time_Prototype_Shifting_for_Zero-Shot_Generalization_with_WACV_2025_paper.pdf

[^1_87]: https://aclanthology.org/2025.acl-long.699.pdf

[^1_88]: https://openreview.net/forum?id=xUBgfvyip3
