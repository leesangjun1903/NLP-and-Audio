
# Gato : A Generalist Agent

## 1. 핵심 주장 및 주요 기여

**Gato** 논문(Reed et al., 2022)의 핵심 주장은 **단일 신경망이 언어 모델 스케일링의 원리를 따라 구축될 경우, 다양한 도메인과 구현(embodiment)에 걸쳐 높은 성능의 일반화 에이전트가 될 수 있다**는 것입니다. 이는 기계학습 커뮤니티에서 오랫동안 주류였던 도메인별 특화 모델 접근 방식에 대한 도전입니다.

**주요 기여:**
- **멀티모달, 멀티태스크, 멀티실체화 정책**: 동일한 가중치를 가진 단일 신경망으로 604개의 서로 다른 작업을 수행하는 최초의 대규모 일반화 에이전트 제시
- **통합 토큰화 방식**: 텍스트, 이미지, 로봇 제어 신호 등 이질적인 데이터를 단일 토큰 시퀀스로 표현하는 체계적 방법 개발
- **프롬프트 기반 조건화**: 자연어 프롬프트와 시연(demonstration)을 통해 과제를 구분하고 제어하는 메커니즘 제안

***

## 2. 문제 정의, 방법론, 모델 구조

### 2.1 문제 정의

기존의 강화학습과 로봇 제어 연구는 주로 다음의 문제를 가지고 있었습니다:

- **도메인 특화성**: 각 작업별로 다른 모델 아키텍처와 가중치 필요
- **데이터 활용 비효율**: 유사한 환경의 데이터도 독립적으로 학습
- **확장성 제한**: 새로운 작업 추가 시 처음부터 학습 필요
- **인간의 인지**: 인간은 단일 뇌로 다양한 작업 수행 가능

Gato의 연구진은 **대규모 언어 모델의 성공에서 영감을 받아**, 이러한 문제를 해결하기 위해 광범위한 데이터와 모델 규모를 통해 일반화 능력을 획득할 수 있다고 가설을 제시합니다.

### 2.2 방법론: 토큰화 및 시퀀스 모델링

#### 2.2.1 통합 토큰화 방식

Gato의 핵심 혁신은 모든 데이터 유형을 단일 토큰 시퀀스로 변환하는 것입니다:

**텍스트:**
$$\text{Text} \rightarrow \text{SentencePiece}(32000 \text{ subwords}) \rightarrow [0, 32000)$$

**이미지:**
- 16×16 패치로 분할 (Vision Transformer 방식)
- 픽셀 정규화: $\text{pixel} \in [-1, 1]$ 후 $\sqrt{\text{patch size}} = 4$ 로 나눔
- 각 패치는 ResNet 블록으로 임베딩

**이산값 (예: Atari 버튼):**
$$\text{Action} \rightarrow \text{Flatten (row-major)} \rightarrow [0, 1024)$$

**연속값 (예: 관절 토크, 로봇 제어):**

```math
\text{continuous\_value} \rightarrow \text{Mu-law encoding} \rightarrow [-1, 1] \rightarrow \text{Discretization (1024 bins)} \rightarrow [32000, 33024)
```

Mu-law 인코딩은 다음 수식으로 표현됩니다:

$$f_{\mu}(x) = \text{sign}(x) \frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)}$$

여기서 $\mu = 255$는 압축 파라미터입니다. 이 방식은 연속 값의 범위를 비선형적으로 압축하여 낮은 범위의 값에 더 많은 표현 능력을 할당합니다.

#### 2.2.2 훈련 목적 함수

Gato는 **인과적 언어 모델 방식**의 자기회귀 학습을 사용합니다:

$$\log p_\theta(s_1, \ldots, s_L) = \sum_{l=1}^{L} \log p_\theta(s_l|s_1, \ldots, s_{l-1}) \quad \cdots (1)$$

배치 $B$에 대한 손실 함수는:

$$L(\theta, B) = -\sum_{b=1}^{|B|} \sum_{l=1}^{L} m(b,l) \log p_\theta\left(s_l^{(b)} | s_1^{(b)}, \ldots, s_{l-1}^{(b)}\right) \quad \cdots (2)$$

여기서 **마스킹 함수** $m(b,l)$은 다음과 같이 정의됩니다:

$$m(b,l) = \begin{cases} 1 & \text{if } s_l \text{ is from text or logged action} \\ 0 & \text{otherwise (observations, image tokens)} \end{cases}$$

이 마스킹은 모델이 관찰(observation)은 예측하지 않고, 텍스트와 행동만을 예측하도록 제한합니다.

### 2.3 모델 구조

#### 2.3.1 아키텍처 세부사항

Gato는 **1.2B 파라미터 디코더 전용 트랜스포머**를 사용합니다:

| 구성 요소 | 사양 |
|---------|------|
| **파라미터 수** | 1.2 billion (1.2B) |
| **레이어 수** | 24 layers |
| **임베딩 차원** | 2048 |
| **피드포워드 은닉층** | 8192 |
| **어텐션 헤드** | 32 heads |
| **컨텍스트 길이** | 1024 tokens |
| **배치 크기** | 512 |
| **훈련 시간** | 1M steps (~4 days on 16×16 TPU v3) |

#### 2.3.2 임베딩 함수

매개변수화된 임베딩 함수 $f(\cdot; \theta_e)$는 모달리티별로 다르게 작동합니다:

**텍스트/이산/연속값 임베딩:**

$$\mathbf{e}_i = \text{Lookup}(\text{token}_i) + \text{PosEnc}_{\text{local}}(i)$$

**이미지 패치 임베딩:**

```math
\mathbf{e}_{patch} = \text{ResNet\_Block}(\text{patch}) + \text{PosEnc}_{\text{within-image}}
```

#### 2.3.3 프롬프트 컨디셔닝

모델의 목표 명확성을 위해 **프롬프트 기반 컨디셔닝**을 사용합니다:

- 배치 내 시퀀스의 25%에 대해 프롬프트 시퀀스 선행
- 프롬프트는 같은 출처 에이전트의 같은 과제 에피소드에서 추출
- 50%는 에피소드 끝에서 추출 (목표 컨디셔닝)
- 50%는 균등 샘플링

### 2.4 배포 전략

Gato의 배포는 **자기회귀 생성** 방식을 따릅니다:

$$\text{Deployment Process:}$$
$$1. \text{Prompt} \rightarrow \text{Tokenize}$$
$$2. \text{Observation} \rightarrow \text{Tokenize} \rightarrow \text{Append to sequence}$$
$$3. \text{Sample Action tokens autoregressively:} \quad a_t = \arg\max_{\text{token}} p_\theta(\cdot | \text{context})$$
$$4. \text{Action} \rightarrow \text{Detokenize} \rightarrow \text{Execute}$$
$$5. \text{Repeat steps 2-4}$$

***

## 3. 성능 향상 및 일반화 능력

### 3.1 시뮬레이션 제어 작업 성능

Gato는 604개 작업에서 다음과 같은 성능을 달성했습니다:

| 도메인 | 성능 (% 전문가 점수) |
|--------|------------------|
| **BabyAI** | 93.2% (80% 이상 달성: 거의 모든 레벨) |
| **DM Lab** | 91.4% |
| **Meta-World** | 87.0% (50/50 과제에서 >50% 달성) |
| **DM Control Suite** | 63.6% |
| **Atari** | 30.9% (23개 게임에서 인간 수준 이상) |

**특히 주목할 점:**
- **Atari 성능**: 23개 게임에서 평균 인간 수준 이상 달성, 11개 게임에서 인간 성능의 2배 초과
- **Meta-World**: 50개 과제 중 44개에서 50% 이상 달성, 35개에서 80% 이상

### 3.2 로봇 작업 성능

#### Skill Generalization (새로운 물체 형태에 대한 일반화)

| 에이전트 | 성공률 |
|---------|-------|
| **Gato (일반화 에이전트)** | 50.2% |
| **BC-IMP (단일 작업 기준)** | 49.0% |

Gato는 단일 목적 에이전트와 비교하여 **경쟁력 있는 성능**을 달성합니다.

#### Skill Mastery (학습 데이터에 포함된 물체)

| 에이전트 | 성공률 |
|---------|-------|
| **Gato** | 75.6% |
| **BC-IMP** | 74.6% |

#### 새로운 목표 적응 능력

연구팀은 훈련 데이터에 없는 **"파란색을 초록색 위에 쌓기"** 작업을 평가했습니다:

- Gato: **60% 성공률** (500개 시연 + 10% 시뮬레이션 데이터 섞음)
- Behavior Cloning 기반라인: **0.5% 성공률**

이는 Gato의 **뛰어난 재프로그래밍 능력**을 보여줍니다.

### 3.3 스케일링 법칙

Gato 연구진은 세 가지 모델 크기에서 성능을 비교했습니다:

$$\text{Model sizes: } 79M, 364M, 1.18B \text{ parameters}$$

**발견:**
- 동일한 토큰 수에 대해 **더 큰 모델이 일관되게 더 높은 성능** 달성
- 성능이 토큰 수 증가에 따라 **평활한 곡선**을 따름
- 더 큰 모델이 **더 적은 파인튜닝 데이터로 빠르게 적응**

### 3.4 분포 외 태스크 학습 (Few-shot Adaptation)

4개의 숨겨진 테스트 과제에서 파인튜닝 성능 평가:

#### **Cartpole Swing-up** (상태 기반 제어)
$$\text{Performance ranking:} \text{All data} > \text{Same domain} > \text{Scratch} \approx \text{No control}$$

- 모든 데이터로 사전학습: **최고 성능**
- 같은 도메인만: **중간 성능**
- 처음부터: **낮은 성능**

#### **Assembly-v2** (Meta-World)
- 모든 데이터 사전학습이 최고 성능
- 비제어 데이터만 사용 시 **음성 전이** 관찰

#### **DM Lab Order_of_apples** (이미지 기반)
- DM Lab 데이터만으로도 충분 (전문가 점수 달성)
- 이미지 캡셔닝 데이터에서 **긍정적 전이** 가능

#### **Boxing** (Atari)
- 모든 변형에서 랜덤 초기화가 더 우수
- **부정적 전이**: 시각적 차이로 인해 사전학습 방해

***

## 4. 모델의 일반화 성능 향상 메커니즘

### 4.1 주요 일반화 능력

#### 1) 스케일 기반 일반화

$$\text{Performance} \propto \log(\text{Model size} \times \text{Data quantity})$$

Gato는 대규모 언어 모델의 **스케일 법칙**을 따릅니다. 더 큰 모델과 더 많은 데이터는 새로운 작업에 대한 일반화 능력을 향상시킵니다.

#### 2) 데이터 다양성 효과

파인튜닝 실험에서 사전학습 데이터 구성의 영향:

$$\text{Fine-tuning Performance} = f(\text{domain diversity}, \text{data quantity}, \text{model scale})$$

**발견:**
- 광범위한 다양한 데이터로 사전학습 → **최고의 새로운 작업 학습**
- 이미지 기반 작업: 비전 언어 데이터에서 긍정적 전이
- Atari 게임: 시각적 차이로 인해 전이 어려움

#### 3) 프롬프트 기반 조건화의 역할

프롬프트는 다음 함수로 모델을 조건화합니다:

$$p(\text{action}_t | \text{observation}_t, \text{prompt}, \text{history})$$

이는 모델이 **과제별 행동 정책을 학습**하도록 돕습니다.

#### 4) 어텐션 메커니즘의 일반화

어텐션 시각화 분석(Figure 12)에서:
- 첫 번째 레이어의 특정 헤드들이 **과제 관련 물체와 영역**에 집중
- Atari Breakout: 공의 움직임 추적
- RGB Stacking: 블록의 위치와 색상 추적
- 이는 **자동적인 특징 추출**을 시사합니다.

### 4.2 한계점

#### 1) 컨텍스트 길이 제한

**문제:**
$$\text{Max context} = 1024 \text{ tokens} \quad \text{(< full episode)}$$

이미지 패치가 많은 바이트를 차지하므로, 에피소드의 일부만 모델이 볼 수 있습니다.

**영향:**
- 프롬프트 구조의 효과 제한
- In-context 학습 능력 저해
- 장기 의존성 학습 불가

#### 2) 행동만 예측 가능

$$\text{Prediction target} = \{\text{text}, \text{actions}\} \quad \text{(not observations)}$$

관찰(이미지, 로봇 상태)는 학습 목표에서 제외되어, 모델이 미래 상태를 예측하는 능력이 제한됩니다.

#### 3) 음성 전이 현상

특정 도메인 간 전이가 **성능을 저하**시키는 경우:
- Atari Boxing: 시각적으로 독특한 환경
- Assembly vs. Cartpole: 불연속적인 행동 공간

$$\text{Negative Transfer:} \quad \text{Performance}_{\text{pretrained}} < \text{Performance}_{\text{scratch}}$$

#### 4) 자기기만(Self-delusion) 편향

Gato 논문에서 지적하는 **인과적 편향**:

아토레그레시브 샘플링으로 인해 혼동 변수가 있으면 잘못된 과제를 풀 수 있습니다.

예: 행동 시퀀스가 비슷한 두 과제에서 모델이 잘못된 과제로 잘못 해석.

**완화 방법:** 프롬프트 엔지니어링으로 혼동 변수 차단

***

## 5. 최신 연구 영향 및 미래 연구 고려사항 (2023-2025)

### 5.1 Gato 이후의 발전 방향

#### 1) **RoboCat** (2023)
다중 실체화 로봇 학습을 위한 **목표 조건화 Decision Transformer**
- Gato의 로봇 능력을 확장
- 시뮬레이션과 실제 로봇 간 직접 전이 학습

#### 2) **LEO: Embodied Generalist Agent in 3D World** (2024)
- 3D 공간에서의 **지각, 이해, 행동** 통합
- 3D 비전-언어 정렬 단계 도입
- Gato의 이미지 토크화를 3D 포인트 클라우드로 확장

#### 3) **REGENT** (2025)
**Retrieval-Augmented Generalist Agent**
- 검색 메커니즘 통합으로 **빠른 적응** 달성
- 1-NN 기반 에이전트도 최신 일반화 에이전트와 경쟁력 있음을 보여줌

#### 4) **Agent S2** (2025)
**Compositional Generalist-Specialist Framework**
- 단순 일반화 에이전트의 한계 해결
- GUI 에이전트를 위한 **혼합 그라운딩 기법**
- 계층적 계획 수립 도입

### 5.2 주요 연구 트렌드

#### A. 멀티모달 통합의 심화

**현황:**
- 비전, 언어, 행동의 **통합 표현 학습**
- Flamingo, GPT-4V 등 기반 모델의 확산

**영향:**
$$\text{Task Performance} = f(\text{vision encoding}, \text{language understanding}, \text{action grounding})$$

더 강력한 시각-언어 기초 모델이 제어 성능 향상

#### B. 컨텍스트 길이 확장 기술

**해결책:**
- **Transformer-XL**, **Perceiver IO** 등 효율적 주의 메커니즘
- **롱 컨텍스트 언어 모델** (Claude 200K, Llama 100K tokens)

$$\text{Context Length Problem} \rightarrow \text{Potential Solution: Efficient Attention}$$

#### C. 오프라인 강화학습과의 통합

**논문 지적:** Gato는 순감독 학습(supervised) 기반

**발전:** 
- Decision Transformer + Offline RL 결합
- 정책 학습과 가치 함수 동시 학습

#### D. 현실 세계 적응성

**도전:**
- 시뮬레이션-현실 격차(Sim2Real)
- 도메인 변이(domain shift)

**해결 방향:**
- 더 많은 현실 로봇 데이터 수집
- 자기 개선 에이전트 (Self-improving Agents)
- 온라인 적응 메커니즘

### 5.3 앞으로의 연구 고려 사항

#### 1) **데이터 수집의 현실성**
$$\text{Challenge:} \quad \text{Web-scale control data} < \text{Web-scale language/vision data}$$

**해결 방안:**
- 관찰만 있는 데이터(YouTube, 로봇 동영상) 활용
- 시뮬레이션 환경 고도화
- 인간-로봇 협력 데이터 수집

#### 2) **프롬프트 길이와 컨텍스트 관리**

현재 제한:
$$\text{Context window} = 1024 \text{ tokens}$$

개선:
$$\text{Efficient Attention} \rightarrow \text{Extended Context} \rightarrow \text{Better In-context Learning}$$

#### 3) **음성 전이 현상 이해 및 해결**

필요한 연구:
- 도메인 유사성 측정 지표 개발
- 선택적 다중 작업 학습
- 적응적 데이터 혼합 전략

#### 4) **안전성 및 정렬 문제**

Gato 논문의 윤리 섹션 강조:
- 다중 행동체(embodiment)의 안전 보장
- 기대 정렬(Value Alignment)
- 의도하지 않은 행동 전이 방지

$$\text{Safety Concern:} \quad \text{Knowledge Transfer} \not\equiv \text{Desirable Behavior Transfer}$$

#### 5) **해석 가능성 및 검증**

개선 필요:
- 어텐션 시각화 이상의 심화 분석
- 태스크별 활성화 패턴 연구
- 실패 케이스 분석

***

## 6. 결론

**Gato는 AI 연구에 미친 영향:**

1. **패러다임 전환**: 도메인별 특화 모델 → 범용 일반화 에이전트
2. **스케일 가설 확증**: LLM의 스케일 법칙이 제어 도메인에도 적용 가능
3. **기초 모델(Foundation Model) 개념 확대**: 텍스트 넘어 제어 정책으로 확장

**2023-2025년 발전:**
- 3D 환경, 검색 기반 적응, 컴포지셔널 구조로 발전
- 멀티모달 기초 모델의 통합 심화
- 현실 로봇 학습 데이터 수집 가속화

**향후 핵심 과제:**
$$\text{Generalization} = f(\text{scale}, \text{data diversity}, \text{architecture innovation}, \text{safety mechanism})$$

Gato의 성과와 한계를 통해, 진정한 의미의 일반화 에이전트는 **더 큰 모델, 더 다양한 데이터, 더 효율적인 아키텍처, 그리고 견고한 안전 메커니즘**의 조화를 요구함을 알 수 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/d15995ce-1532-492e-be2f-ec3d886ee24d/2205.06175v3.pdf)
[2](https://arxiv.org/html/2504.00906v1)
[3](https://arxiv.org/html/2412.04759)
[4](https://arxiv.org/pdf/2503.01861.pdf)
[5](https://arxiv.org/pdf/2312.15536.pdf)
[6](https://arxiv.org/html/2306.11706)
[7](https://arxiv.org/html/2311.12871v2)
[8](https://arxiv.org/html/2410.18603)
[9](http://arxiv.org/pdf/2406.14228.pdf)
[10](https://bboyseiok.com/ko/posts/generalist-agents/)
[11](https://discuss.pytorch.kr/t/2025-07-07-13-ai-ml/7248)
[12](https://seo.goover.ai/report/202503/go-public-report-ko-74a861c0-f086-48c2-9333-bd324650c343-0-0.html)
[13](https://eair.tistory.com/49)
[14](https://liner.com/ko/review/xtreme-massively-multilingual-multitask-benchmark-for-evaluating-crosslingual-generalization)
[15](https://www.ultralytics.com/ko/blog/multi-modal-models-and-multi-modal-learning-expanding-ais-capabilities)
[16](https://hyeok1235.tistory.com/96)
[17](https://www.themoonlight.io/ko/review/skyrl-agent-efficient-rl-training-for-multi-turn-llm-agent)
[18](https://research4lab.tistory.com/entry/Report-%EB%A9%80%ED%8B%B0%EB%AA%A8%EB%8B%AC-AI%EC%99%80-%ED%86%B5%ED%95%A9%EC%A0%81-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%B2%98%EB%A6%AC-%EC%82%B0%EC%97%85-%EB%B3%80%ED%99%94%EC%9D%98-%EC%83%88%EB%A1%9C%EC%9A%B4-%ED%8C%A8%EB%9F%AC%EB%8B%A4%EC%9E%84)
[19](https://www.scribd.com/document/789464118/2024%EB%85%84-KISTEP-%EB%AF%B8%EB%9E%98%EC%9C%A0%EB%A7%9D%EA%B8%B0%EC%88%A0-%EC%84%A0%EC%A0%95%EC%97%90-%EA%B4%80%ED%95%9C-%EC%97%B0%EA%B5%AC-%EC%83%9D%EC%84%B1%ED%98%95-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EC%8B%9C%EB%8C%80%EC%9D%98-%EB%AF%B8%EB%9E%98%EC%9C%A0%EB%A7%9D%EA%B8%B0%EC%88%A0-3%EA%B5%90)
