
# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

## 1. 논문의 핵심 주장과 주요 기여

**GLM-4.5**는 Z.ai(Zhipu AI) 연구팀이 2025년 7월에 발표한 오픈소스 대규모 언어 모델로, **에이전트 기능(Agentic)**과 **고급 추론 능력(Reasoning)**, 그리고 **코딩 성능(Coding)**을 모두 갖춘 통합 파운데이션 모델입니다.[1][2][3]

### 핵심 주장

본 논문의 중심 주장은 **Mixture-of-Experts(MoE) 아키텍처와 다단계 강화학습을 결합하면, 비용 효율적이면서도 상용 모델 수준의 성능을 달성할 수 있다**는 것입니다. 특히, 파라미터 효율성과 추론 능력의 균형을 이루어 오픈소스 모델 중에서 경쟁력 있는 성능을 입증했습니다.[2][4][1]

### 주요 기여

1. **혼합 전문가(MoE) 구조의 최적화**: 355B 총 파라미터 중 32B만 활성화하여 연산 효율성을 극대화했습니다.[3][1]

2. **하이브리드 추론 메서드**: 깊이 있는 사고(Thinking Mode)와 직접 응답(Direct Response Mode)을 모두 지원하는 유연한 추론 방식을 제시했습니다.[5][6]

3. **다단계 훈련 파이프라인**: 23조 토큰의 사전훈련과 전문가 모델 반복학습, 강화학습을 통합한 포괄적인 훈련 방식을 제안했습니다.[1]

4. **자동화된 에이전트 데이터 생성**: 4단계 LLM 기반 파이프라인으로 고품질 에이전트 훈련 데이터를 효율적으로 생성하는 방법론을 개발했습니다.[7]

***

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

### 문제 정의

현대 LLM의 세 가지 주요 도전 과제:

1. **파라미터 효율성**: 초거대 모델의 학습과 배포에 막대한 자원 소요
2. **능력 통합**: 추론, 에이전트, 코딩 능력을 하나의 모델에서 균형있게 달성하기 어려움
3. **일반화 성능**: 학습 데이터와 다른 분포의 새로운 환경에서의 일반화 능력 부족

### 제안 방법

#### 2.1 아키텍처 설계

**Mixture-of-Experts 구조**를 핵심으로 다음과 같이 설계했습니다:[8][4]

$$\text{Output} = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

여기서:
- $$G(x)$$: 게이팅 네트워크 (Sigmoid gating 함수 적용)
- $$E_i(x)$$: $i$번째 전문가의 출력
- $$N$$: 전체 전문가 수

**핵심 특징**:[4][8]
- **Loss-free balance routing**: 보조 손실 함수 없이 전문가 편향값만으로 부하 분산
- **Grouped-Query Attention (GQA)**: Key-Value 캐시 크기 감소로 추론 속도 향상
- **Partial RoPE**: 회전 위치 임베딩의 선택적 적용
- **QK-Norm**: 주의 메커니즘 안정성 강화
- **Multi-Token Prediction (MTP)**: Speculative decoding 지원

#### 2.2 훈련 파이프라인

**다단계 훈련 전략**:[7][1]

$$\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{SFT}} + \lambda_1 \mathcal{L}\_{\text{reasoning-RL}} + \lambda_2 \mathcal{L}_{\text{agent-RL}}$$

여기서 각 손실 함수는:

1. **감독된 미세조정 (SFT)**:

$$\mathcal{L}\_{\text{SFT}} = -\sum_{t=1}^{T} \log P(y_t | x, y_{ < t}; \theta)$$
   
   - 고품질 추론 궤적으로 미세조정
   - 합성 데이터로 추론 능력 강화
   - 64K 컨텍스트에서 훈련

2. **추론 강화학습 (Reasoning RL)**:

$$\mathcal{L}_{\text{reasoning-RL}} = \mathbb{E}[\min(r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}})]$$
   
   - 단일 단계 강화학습 (전체 64K 컨텍스트)
   - 어려움 기반 커리큘럼 (Difficulty-based Curriculum)
   - 동적 샘플링 온도로 탐색-활용 균형
   - 적응형 클리핑으로 STEM 문제에서 안정성 보장

3. **에이전트 강화학습 (Agent RL)**:

$$\mathcal{L}\_{\text{agent-RL}} = \mathbb{E}\_{q \sim D, \tau \sim \pi_\theta} [r(\tau) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}})]$$
   
   - 검증 가능한 보상 (Verifiable Rewards) 활용:
     * 웹 검색 (정보 정확성 검증)
     * 코드 실행 (실행 결과 검증)
   - 자동화된 다중홉 추론 파이프라인
   - 지식 그래프 기반 자동 합성 데이터

#### 2.3 에이전트 데이터 생성 자동화

4단계 자동 파이프라인:[7]

$$\text{Pipeline} = \text{ToolCollection} \rightarrow \text{TaskSynthesis} \rightarrow \text{TrajectoryGeneration} \rightarrow \text{QualityFiltering}$$

각 단계:
1. **도구 수집**: 다양한 API와 함수 자동 수집
2. **작업 합성**: LLM으로 현실적 작업 생성
3. **궤적 생성**: 도구 호출 시퀀스 자동 생성
4. **품질 필터링**: LLM 판단자로 고품질 데이터만 선별

### 모델 구조 상세 분석

#### 2.4 MoE 라우팅 메커니즘

**Sigmoid Gating 함수**:

$$G(x)_i = \text{Sigmoid}(W_{\text{gate}} x + b_i)$$

- 각 전문가마다 학습 가능한 편향 $$b_i$$ 보유
- 손실 함수 없이 편향값 조정으로 부하 분산 (Loss-free balance)

#### 2.5 어텐션 메커니즘

**Grouped-Query Attention의 연산**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- 96개의 attention head 구성
- 히든 차원 5120에 대해 2.5배 다중성
- 부분 RoPE: 위치 임베딩을 선택적으로 적용

#### 2.6 Multi-Token Prediction

동시에 여러 토큰 예측:

$$P(y_t, y_{t+1}, ..., y_{t+k} | x) = \prod_{i=0}^{k} P(y_{t+i} | x, y_{t: < t+i})$$

- Speculative decoding으로 최대 200토큰/초 생성 속도 달성[9]

***

## 3. 성능 향상 및 일반화 성능

### 성능 향상 결과

#### 3.1 벤치마크 성능

| 벤치마크 | GLM-4.5 | 경쟁 모델 | 의의 |
|---------|---------|----------|------|
| TAU-Bench (에이전트) | **70.1%** | Claude 4 Sonnet (동등) | 복잡한 작업 지시 수행 능력[1][7] |
| AIME 2024 (수학) | **91.0%** | o3/o3-mini (동등) | 고난도 수학 올림피아드 문제 해결[1] |
| SWE-bench Verified (코딩) | **64.2%** | GPT-4.1보다 우위 | 실제 소프트웨어 엔지니어링 작업[1] |
| Terminal-Bench (터미널) | **37.5%** | Gemini 2.5-pro 우위 | 명령어 기반 작업 수행[8] |
| 도구 호출 성공률 | **90.6%** | Claude 3.5 Sonnet (89.5%) | Function calling 정확도[9] |

#### 3.2 전체 모델 순위

- **12개 주요 벤치마크 평균**: 63.2점[2]
- **전체 모델 순위**: 3위 (o3 65.0점, Grok 4 63.6점 뒤)
- **오픈소스 모델**: **1위**[2]
- **코딩 부문**: 68.8점 - 모든 테스트 모델 중 **최고 성능**[2]

#### 3.3 경량화 버전 성능

GLM-4.5-Air (106B 파라미터):
- 12개 벤치마크 평균: **59.8점**
- 100B급 모델 중 **최고 성능**[4]
- 소형 GPU 환경에서도 배포 가능[9]

### 일반화 성능 향상

#### 3.4 일반화 능력의 핵심 메커니즘

**메타강화학습(Meta-RL) 원리**:

$$\theta^* = \arg\min_\theta \mathbb{E}_{T \sim p(T)} [ \mathcal{L}_T(\theta) ]$$

다양한 작업 분포 $$p(T)$$에서의 손실 최소화[10][11]

#### 3.5 구체적인 일반화 향상 방법

1. **다중 도메인 학습**:
   - 수학, 코딩, 웹 검색, 다국어 등 다양한 도메인에서 훈련
   - 도메인 간 시너지로 일반화 성능 증진

2. **검증 가능한 보상 활용**:

$$r_{\text{verified}} = \begin{cases} 1 & \text{if } \text{execute}(\tau) = y^* \\ 0 & \text{otherwise} \end{cases}$$
   
   - 실제 환경(웹, 코드 실행)에서만 보상 제공
   - 분포 외(Out-of-Distribution) 상황에서도 일반화

3. **어려움 기반 커리큘럼**:

$$\text{Difficulty}(\tau_t) = \alpha \cdot \text{Complexity} + \beta \cdot \text{LossValue}$$
   
   - 단순한 작업부터 점진적으로 복잡한 작업으로 학습
   - 과적합 방지 및 강건성 증진

4. **동적 환경 적응**:
   - 샘플링 온도 동적 조정
   - 보상 신호에 따른 정책 업데이트
   - 다양한 환경 변화에 대한 강건성

#### 3.6 교차 도메인 일반화

**언어 간 성능 일반화**:
- 영어 외 언어에서도 추론 능력 유지
- 한국어 포함 다국어 벤치마크에서 강력한 성능[12]

**Zero-shot 일반화**:
- 학습하지 않은 새로운 도구에 대한 함수 호출 성능
- 유연한 에이전트 적응력

***

## 4. 논문의 한계

### 4.1 기술적 한계

1. **구체적 수식 부재**: 
   - 정확한 손실 함수 가중치 ($$\lambda_1, \lambda_2, \beta$$) 미공개
   - 정확한 라우팅 알고리즘 세부 사항 제한적 공개[13]

2. **메모리 및 분산 훈련**:
   - MoE의 고질적 문제인 높은 VRAM 요구사항
   - All-to-all 통신 오버헤드[14]

3. **라우팅 최적화**:
   - 부하 불균형 문제 완전히 해결되지 않음
   - 특정 토큰에 대한 전문가 편향 가능성[15]

4. **추론 비용**:
   - 부호화된 추론으로 인한 비용 증가
   - 검증 가능한 보상 획득 비용[16]

### 4.2 평가상 한계

1. **벤치마크 의존성**:
   - 학술 벤치마크와 실제 응용의 차이
   - 특정 벤치마크에 특화될 가능성

2. **인간 평가 부족**:
   - 일부 평가가 자동화 메트릭에만 의존
   - 질적 분석 상세도 제한[5]

3. **도메인 특화 한계**:
   - 특정 산업 도메인(의료, 법률 등)에서의 성능 미검증
   - 장기 시간대의 안정성 평가 부족

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 MoE 아키텍처 진화

| 모델 | 발표연도 | 파라미터 | 특징 | 성능 |
|------|---------|---------|------|------|
| **Switch Transformer** | 2021 | 1.6T | 단일 전문가 라우팅[17] | 기준선 대비 7% 향상 |
| **Mixtral 8x7B** | 2023 | 56B | Top-K(k=2) 라우팅 | 70B 모델 대비 경쟁력[14] |
| **GLM-4.5** | 2025 | 355B | Loss-free balance, Sigmoid gating | 오픈소스 1위 (63.2점)[2][4] |
| **DeepSeek-V3** | 2024 | 685B | 미세 전문가 분할, 공유 전문가 | 전체 3위 성능[17] |
| **Qwen3-Coder** | 2025 | 110B | 도메인 특화 | 코딩에서 GLM-4.5 대비 81.2% 우세 vs GLM-4.5 80.8% 우세[2] |

### 5.2 추론 능력 강화 연구

#### Chain-of-Thought (CoT) 기반 발전

**OpenAI o1/o3 시리즈** (2024-2025):
- CoT 프롬프팅을 통한 추론 능력 강화
- o1: 초기 추론 모델, AIME 85% 달성
- o3: 더 깊은 추론, AIME 96% 달성
- o3-mini: 경량화 버전[18]

**DeepSeek-R1** (2025):

$$\mathcal{L}\_{\text{RL}} = \mathbb{E}[\text{reward}(\text{trace})] - \beta D_{KL}(\pi || \pi_{\text{ref}})]$$

- 순수 강화학습으로 추론 능력 유도 (인간 라벨 불필요)[19]
- GLM-4.5와 유사한 강화학습 철학
- 자기 반성, 검증, 동적 전략 적응 능력 발현

**GLM-4.5와의 비교**:
- GLM-4.5: SFT + RL 하이브리드 (감독 신호 활용)
- DeepSeek-R1: 순수 RL (감독 신호 최소화)
- AIME 성능: GLM-4.5 91.0% vs DeepSeek-R1 96%+

### 5.3 에이전트 AI 연구 진화

#### 도구 활용 능력 개선

**Function Calling 성능 추이**:
- Claude 3.5 Sonnet: 89.5%[9]
- Kimi K2: 86.2%[9]
- **GLM-4.5: 90.6%** (최고)[9]

**자동 궤적 생성**:
- **GLM-4.5**: 4단계 LLM 기반 파이프라인[7]
- **Tongyi-DeepResearch**: 마ル티 에이전트 시스템 (질문자, 해결자, 검사자)[20]
- **DeepAnalyze**: 5가지 행동 유형으로 데이터 과학 작업 해결[21]

#### 강화학습 에이전트 훈련

**RLVR (Reinforcement Learning from Verifiable Rewards)**:

$$r(o) = \begin{cases} 1 & \text{if } f(o) \text{ executes successfully} \\ 0 & \text{otherwise} \end{cases}$$

- GLM-4.5: 웹 검색, 코드 실행 기반 보상[7]
- DeepSeek: 자동화 도구 호출 기반 보상[13]
- 비용 효율성: 기존 SFT 대비 50-70% 적게 소요[16]

### 5.4 일반화 성능 향상 연구

#### 메타강화학습 (Meta-RL) 접근

**기본 원리**:
$$P(\text{새 작업에서 성공}) = \sum_{T} p(T) \mathcal{P}_{\theta}(\text{성공} | T)$$

**주요 연구**:
1. **메모리 기반 메타-RL**: RNN 히든 상태로 다양한 작업 적응[22]
2. **목표 생성 메타-RL**: GAN으로 자동 목표 생성 (ICML 2018)[22]
3. **TAVT (자동 가상 임무 예습)** (UNIST, 2025): AI가 스스로 다양한 임무 생성[23]

**GLM-4.5의 일반화 전략**:
- 다양한 도메인에서 감독 + 강화학습 결합
- 어려움 기반 커리큘럼으로 점진적 강화
- 실제 환경 보상으로 분포 외 강건성 확보

### 5.5 이미지 기반 강화학습의 일반화

**자기지도 학습 (Self-Supervised Learning) 활용**:
- 랜덤 증강과 대조 학습 결합[11]
- 데이터 효율성과 일반화 간 트레이드오프 해결
- 정적/동적 환경 변화에 강건[11]

**GLM-4.5에의 시사점**:
- 텍스트 도메인에서도 자기지도 신호 활용 가능
- 증강 강도 동적 조정으로 강건성 향상

### 5.6 MoE 최적화의 최신 동향

#### 전문가 특화도(Expert Specialization) 강화

**문제점**:
- 표준 MoE의 보조 부하 분산 손실: 전문가 중복 유발
- 라우팅 균일화로 인한 성능 저하

**GLM-4.5 해결책**:
- Loss-free balance routing
- Sigmoid gating으로 선택적 활성화[4]

**최신 연구 (2025)**:
- **직교성 손실**: 서로 다른 토큰 처리 전문가 유도[15]
- **분산 손실**: 판별적 라우팅 결정 강화[15]
- 성과: 기준선 대비 최대 23.79% 성능 향상[15]

#### 매개변수 효율 개선

**MoLAE (Mixture of Latent Experts)** (2025):

$$E_i(x) = A_i^{\text{down}} \cdot B_{\text{down}} \cdot A_i^{\text{gate}} \cdot B_{\text{gate}} \cdot B_{\text{up}} \cdot x$$

- 공유 대차원 축소 + 전문가별 변환[24]
- 40% 매개변수 감소 (성능 유지)[24]

**GLM-4.5와의 비교**:
- GLM-4.5: 표준 MoE (큰 파라미터지만 효율적 활성화)
- MoLAE: 파라미터 자체 감소

***

## 6. 향후 연구에 미치는 영향과 고려 사항

### 6.1 학술 연구에 미치는 영향

#### 1) **추론 중심 패러다임 정착**

GLM-4.5의 성공은 2025년 AI 분야의 대세를 명확히 합니다:[25]

- **모델 크기 경쟁에서 추론 능력 경쟁으로**: 파라미터 수 증가보다 **RL 기반 추론 강화**가 주요 과제
- **Test-time Scaling**: 추론 단계에서 계산량 증가로 성능 향상 (Compute Budget Forcing)[13]
  - GLM-4.5: 19% → 27% (Single application)
  - Pass@32로 67% 달성 가능[13]

#### 2) **오픈소스 AI의 경쟁력 증진**

- 상용 모델과의 성능 격차 해소: 전체 3위 달성
- 오픈소스 개발자 커뮤니티 활성화
- 민주적 AI 개발 추진력 강화

#### 3) **강화학습의 재조명**

$$\mathcal{J}_{\text{policy}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[r(\tau) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}})]$$

- 인간 라벨 의존도 감소
- 자동화된 보상 신호 활용의 중요성
- RLVR의 비용 효율성 입증

### 6.2 산업 응용에의 영향

#### 1) **에이전트 AI 상용화**

- 도구 호출 성공률 90.6%: 신뢰할 수 있는 자동화 가능
- 웹 검색, 코드 실행 등 복합 작업 처리
- 기업용 자동화 솔루션 개발 가속

#### 2) **비용 효율성 개선**

| 측면 | 영향 |
|------|------|
| 훈련 비용 | MoE로 50% 이상 절감 가능[14] |
| 추론 비용 | Test-time scaling으로 필요시에만 계산 증가 |
| 운영 비용 | 경량화 버전(Air) 소형 GPU에서 배포 가능[9] |

#### 3) **도메인 특화 모델 개발**

- 오픈소스 베이스 모델로 빠른 특화 모델 개발 가능
- 의료, 법률, 금융 등 전문 분야 적용 용이

### 6.3 향후 연구 방향

#### Phase 1: 기술 개선 (단기, 1-2년)

**아키텍처 최적화**:
1. **더 효율적인 라우팅**:
   - 세밀한 전문가 분할 (Fine-grained MoE)
   - 적응형 라우팅 정책: 입력 특성에 따른 전문가 수 동적 조정

2. **전문가 특화도 증진**:

$$\mathcal{L}\_{\text{spec}} = \mathcal{L}_{\text{orthogonality}} + \mathcal{L}_{\text{variance}}$$
   
   - 직교성 손실로 서로 다른 토큰 처리
   - 분산 손실로 판별적 라우팅[15]

3. **다중 모달리티 확장**:
   - GLM-4.5V (비전): 106B 파라미터, 12B 활성[26][8]
   - 음성, 비디오 등 다양한 모달리티 통합

**훈련 효율성**:
1. **분산 훈련 최적화**:
   - All-to-all 통신 오버헤드 감소
   - 비동기 롤아웃 (Asynchronous Rollout)을 통한 병렬 RL[20]

2. **메모리 효율성**:
   - MoLAE 같은 매개변수 효율 아키텍처 통합[24]
   - KV 캐시 최적화 (70% 감소)[27]

#### Phase 2: 일반화 능력 강화 (중기, 2-3년)

**도메인 적응 학습**:
1. **전이 학습 (Transfer Learning)**:
   $$\theta_{\text{target}} = \theta_{\text{source}} + \Delta\theta_{\text{adaptation}}$$
   
   - 기본 모델에서 특정 도메인으로 효율적 전이

2. **Zero-shot/Few-shot 일반화**:
   - 메타강화학습으로 빠른 적응
   - 프롬프트 학습 (In-context Learning) 강화

3. **동적 환경 적응**:
   - 온라인 강화학습: 배포 후 지속적 학습
   - 비분포(Out-of-distribution) 탐지 및 대응

**신뢰성 강화**:
1. **불확실성 정량화**:
   - 예측 신뢰도 추정
   - 불확실한 경우 인간 개입 요청

2. **편향 완화**:
   - 공정성 평가 및 개선
   - 다양한 인구통계 집단에서의 성능 균형

#### Phase 3: 근본적 혁신 (장기, 3-5년)

**새로운 훈련 패러다임**:
1. **자기개선 (Self-Improvement)**:
   - 모델이 자체 생성 데이터로 학습
   - 순환 강화학습 (Curriculum Learning at Scale)

2. **다중 에이전트 상호작용**:
   - 여러 LLM 에이전트 간 협력 및 경쟁
   - 창발적 능력 개발

3. **신경-상징 통합 (Neuro-Symbolic Integration)**:
   - LLM의 직관적 추론 + 기호 논리의 정확성
   - 설명 가능성 향상

### 6.4 구체적 연구 과제별 고려사항

#### 1) **추론 능력 강화 연구**

**해결할 과제**:
- 추론 깊이와 비용의 트레이드오프
- 검증 불가능 문제에서의 추론

**고려사항**:
$$\text{성능} = f(\text{계산량}) + \text{상수}$$

추론 곡선의 수렴점 파악 필수[25]

#### 2) **에이전트 신뢰성 연구**

**필수 요소**:
- 도구 호출 신뢰도: 90%+ 목표
- 에러 복구 능력
- 장시간 작업에서의 안정성

**평가 지표**:
- Pass@K: K번 재시도 중 성공률
- Maj@K: 다수결(Majority Voting) 정확도[13]

#### 3) **일반화 성능 평가**

**벤치마크 다양화**:
- 학술 벤치마크 (MMLU, AIME 등)
- 실제 작업 벤치마크 (SWE-bench, BrowseComp 등)
- 도메인 특화 벤치마크

**평가 방법론**:
- 인간 평가 (Human Evaluation)
- 자동 메트릭 (BLEU, ROUGE 등)
- 하이브리드 평가

### 6.5 교육 및 지식 공유 차원

#### 1) **재현성 강화**

GLM-4.5의 오픈소스 공개로:[7]
- 완전한 모델 가중치 공개
- 훈련 코드 및 데이터 파이프라인 공개
- 커뮤니티 기반 개선 가능

#### 2) **벤치마크 표준화**

- TAU-Bench, SWE-bench 등 표준 벤치마크 확대
- 공정한 모델 비교를 위한 평가 프로토콜 통일

#### 3) **인력 양성**

- 오픈소스 기반 AI 개발 교육 확산
- LLM 에이전트, 강화학습 실무 교육 필요성 증대

***

## 7. 결론

**GLM-4.5**는 단순한 성능 최고 모델을 넘어, 2025년 AI 연구의 **패러다임 전환**을 상징합니다.[25]

### 핵심 요약

1. **기술적 혁신**: MoE의 효율성과 다단계 강화학습으로 파라미터 효율성과 성능을 동시에 달성[4][2]

2. **성능 입증**: 오픈소스 모델 중 최고 (63.2점), 전체 3위 달성으로 상용 모델과의 격차 해소[2]

3. **일반화 능력**: 어려움 기반 커리큘럼, 검증 가능한 보상, 다중 도메인 학습으로 강건한 일반화 성능[17][7]

4. **실용성**: 도구 호출 90.6%, 코딩 성능 64.2%로 실제 응용 가능성 입증[1][9]

### 향후 전망

$$\text{AI의 미래} = \text{효율성(MoE)} + \text{추론(RL)} + \text{신뢰성(Verification)}$$

앞으로의 AI 연구는 다음 세 축을 중심으로 진행될 것입니다:[25]

- **추론 중심**: 계산 효율적인 추론 능력 강화
- **에이전트**: 신뢰할 수 있는 자동화 도구
- **일반화**: 새로운 환경과 작업에 빠르게 적응

GLM-4.5는 이 세 가지 목표를 통합적으로 달성하는 첫 번째 사례이며, 이는 향후 5년간 AI 연구의 명확한 방향을 제시합니다.

***

**주요 참고문헌**:

[1](https://huggingface.co/papers/2508.06471)
[2](http://makersjournal.co.kr/View.aspx?No=3728778)
[3](https://arxiv.org/abs/2508.06471)
[4](https://news.hada.io/topic?id=22487)
[5](https://www.youtube.com/watch?v=CEiplK31RI8)
[6](https://llm-stats.com/models/glm-4.5)
[7](https://z.ai/blog/glm-4.5)
[8](https://digitalbourgeois.tistory.com/1787)
[9](https://digitalbourgeois.tistory.com/1714)
[10](https://cse.snu.ac.kr/community/news/1116)
[11](https://www.dbpia.co.kr/journal/detail?nodeId=T16938288)
[12](http://arxiv.org/pdf/2501.02448.pdf)
[13](https://arxiv.org/pdf/2510.06135.pdf)
[14](https://developer.nvidia.com/ko-kr/blog/applying-mixture-of-experts-in-llm-architectures/)
[15](https://arxiv.org/abs/2505.22323)
[16](https://reviewinsight.blog/2025/12/15/2025%EB%85%84-llm-ai-%ED%8A%B8%EB%A0%8C%EB%93%9C-%EB%B3%80%ED%99%94-%ED%9D%90%EB%A6%84-gpt-5-2-%EC%B6%9C%EC%8B%9C%EC%99%80-%ED%95%A8%EA%BB%98-%EC%82%B4%ED%8E%B4%EB%B3%B8/)
[17](https://wikidocs.net/275230)
[18](https://botpress.com/ko/blog/best-large-language-models)
[19](https://arxiv.org/abs/2501.12948)
[20](https://arxiv.org/html/2510.24701v1)
[21](https://arxiv.org/html/2510.16872v1)
[22](https://dmqa.korea.ac.kr/uploads/seminar/20210312_%EC%A1%B0%EC%96%B5_Reinforcement%20Learning%20for%20Generalization.pdf)
[23](https://www.mstoday.co.kr/news/articleView.html?idxno=98560)
[24](https://www.arxiv.org/pdf/2503.23100.pdf)
[25](https://news.hada.io/topic?id=25486)
[26](https://arxiv.org/pdf/2510.10991.pdf)
[27](https://arxiv.org/pdf/2508.19667.pdf)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/24ca346d-315b-4bb1-ad68-89c3adad6477/2508.06471v1.pdf)
[29](https://arxiv.org/abs/2309.11657)
[30](https://aclanthology.org/2022.acl-long.26.pdf)
[31](https://arxiv.org/pdf/2112.06905.pdf)
[32](https://arxiv.org/pdf/2304.13793.pdf)
[33](http://arxiv.org/pdf/2403.00013.pdf)
[34](https://pmc.ncbi.nlm.nih.gov/articles/PMC10991518/)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC6431156/)
[36](https://pmc.ncbi.nlm.nih.gov/articles/PMC10925287/)
[37](https://liner.com/review/glm45-agentic-reasoning-and-coding-arc-foundation-models)
[38](https://velog.io/@lhj/GLM-4.5-Agentic-Reasoning-and-Coding-ARC-Foundation-Models)
[39](https://www.youtube.com/watch?v=W2c18ig8xUY)
[40](https://digitalbourgeois.tistory.com/1767)
[41](https://news.ycombinator.com/item?id=44871337)
[42](https://skywork.ai/blog/llm/z-ai-glm-4-5-guide-free-chat-ko/)
[43](https://arxiv.org/pdf/2411.02959.pdf)
[44](https://arxiv.org/pdf/2511.02805.pdf)
[45](https://arxiv.org/pdf/2510.01782.pdf)
[46](https://arxiv.org/pdf/2510.17498.pdf)
[47](https://arxiv.org/html/2509.18883v1)
[48](https://www.arxiv.org/pdf/2512.23647.pdf)
[49](https://www.semanticscholar.org/paper/GLM-4.5:-Agentic,-Reasoning,-and-Coding-(ARC)-Zeng-Lv/e05795cdb5b0c11e72016c79122777c5b3a7ed48)
[50](https://arxiv.org/pdf/2512.20491.pdf)
[51](https://www.semanticscholar.org/author/GLM-4.5-Team-Aohan-Zeng/2375326375)
[52](https://arxiv.org/abs/2109.04650)
[53](https://arxiv.org/pdf/2309.10305.pdf)
[54](https://arxiv.org/pdf/2403.10882.pdf)
[55](https://arxiv.org/pdf/2306.13549v2.pdf)
[56](https://aclanthology.org/2021.emnlp-main.274.pdf)
[57](https://arxiv.org/pdf/2307.06435v3.pdf)
[58](http://arxiv.org/pdf/2404.14219.pdf)
[59](https://ksp.etri.re.kr/ksp/article/file/70691.pdf)
[60](https://royzero.tistory.com/entry/sparse-moe-architecture-llm-efficiency)
[61](https://blog.naver.com/simula/224130139179?fromRss=true&trackingCode=rss)
[62](https://www.allibee.ai/blog/llm-to-agent-term)
[63](https://www.ibm.com/kr-ko/think/topics/mixture-of-experts)
[64](https://m.hanbit.co.kr/channel/view.html?cmscode=CMS0545719107)
[65](https://pdfs.semanticscholar.org/5a57/ef489e3d05c4b9f62bf694876a7997b8c979.pdf)
[66](https://pdfs.semanticscholar.org/ead4/75248fc37818f8ea68bbcb031b8594b6afef.pdf)
[67](https://arxiv.org/abs/2507.11181)
[68](https://arxiv.org/abs/2409.02060)
[69](https://arxiv.org/html/2507.07818v2)
[70](https://arxiv.org/abs/2405.16039)
[71](https://arxiv.org/html/2408.10681v1)
[72](https://arxiv.org/html/2407.12709v1)
[73](http://arxiv.org/abs/2506.08356)
[74](https://arxiv.org/abs/2404.10237)
