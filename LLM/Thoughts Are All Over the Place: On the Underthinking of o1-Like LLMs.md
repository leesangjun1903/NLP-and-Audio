
# Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs

## 1. 논문의 핵심 주장과 기여

### 1.1 주요 문제 정의: Underthinking 현상

본 논문은 OpenAI의 o1 및 그 복제 모델(QwQ, DeepSeek-R1 등)에서 발견되는 **"Underthinking"** 현상을 최초로 체계적으로 분석합니다. Underthinking은 o1 유사 모델들이 유망한 추론 경로를 충분히 탐색하지 못한 채 조기에 포기하고 빈번하게 다른 사고로 전환하는 현상을 의미합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

**핵심 발견:**
- o1 유사 모델은 잘못된 응답에서 정답 응답보다 **225% 더 많은 토큰**을 생성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)
- 이러한 차이는 **418% 더 빈번한 사고 전환**으로 인해 발생합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)
- 부정확한 응답의 **70% 이상이 적어도 하나의 올바른 생각**을 포함하고 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

### 1.2 세 가지 주요 기여

**첫째, Underthinking 현상의 형식적 정의와 특성화**
- 어려운 문제에서 더 자주 발생하는 현상을 체계적으로 분석
- 정확한 초기 사고가 충분히 탐색되지 않고 포기되는 패턴 규명
- 잦은 사고 전환이 성능 저하와 강한 상관관계 입증

**둘째, 새로운 Underthinking 메트릭 개발**
본 논문은 토큰 효율성을 기반으로 하는 Underthinking 스코어(ξUT)를 도입했습니다:

$$\xi_{UT} = \frac{1}{N} \sum_{i=1}^{N} \left(1 - \frac{\hat{T}_i}{T_i}\right)$$

여기서:
- $N$ : 부정확한 응답의 개수
- $T_i$ : i번째 부정확한 응답의 총 토큰 수
- $\(\hat{T}_i\)$ : i번째 응답에서 첫 번째 정확한 생각까지의 토큰 수

낮은 ξUT 값은 높은 토큰 효율성을 의미합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

**셋째, Thought Switching Penalty (TIP) 디코딩 전략**
- 모델 파인튜닝 없이 디코딩 단계에서만 작용
- 각 추론 경로를 더 깊이 있게 탐색하도록 격려
- 즉시 적용 가능한 실용적 해결책

## 2. 문제 해결 방법: TIP 메커니즘

### 2.1 제안 방법의 수식 및 원리

**표준 디코딩:**
$$P(x_t = v | x_{ < t}) = \frac{\exp(z_{t,v})}{\sum_{v' \in V} \exp(z_{t,v'})}$$

**사고 전환 패널티 적용:**

$$\hat{z}\_{t,v} = \begin{cases} z_{t,v} - \alpha, & \text{if } v \in \hat{V} \text{ and } t < \Psi + \beta \\ z_{t,v}, & \text{otherwise} \end{cases}$$

**수정된 확률 분포:**

$$\hat{P}(x_t = v | x_{ < t}) = \frac{\exp(\hat{z}\_{t,v})}{\sum_{v' \in V} \exp(\hat{z}_{t,v'})}$$

### 2.2 핵심 파라미터

| 파라미터 | 의미 | 최적값 | 범위 |
|---------|------|--------|------|
| **α** | Penalty Strength - 사고 전환 토큰 로짓 감소 정도 | 3 |  [aclanthology](https://aclanthology.org/2025.naacl-long.375) |
| **β** | Penalty Duration - 사고 시작 후 패널티 활성 위치 수 | 600 |  |
| **Ψ** | 사고 시작 위치 | - | 동적 |
| **\(\hat{V}\)** | 사고 전환 관련 토큰 집합 (예: "alternatively") | - | 수동 정의 |

### 2.3 하이퍼파라미터 최적화 결과

AIME 2022-2023 개발 집합에서의 그리드 서치 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

| α | β=300 | β=400 | β=500 | β=600 | β=700 |
|---|-------|-------|-------|-------|-------|
| 3 | 35.2% | 39.3% | 38.5% | **39.8%** | 37.1% |
| 5 | 37.0% | 37.1% | 38.7% | 39.4% | 39.4% |
| 10 | 39.0% | 37.1% | 39.1% | 38.0% | 39.0% |
| 20 | 39.4% | 38.4% | 39.2% | 38.0% | 38.3% |

최적값 α=3, β=600에서 39.8% 정확도 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

## 3. 모델 구조 및 평가 방법

### 3.1 평가 대상 모델

| 모델 | 파라미터 | 특징 |
|-----|---------|------|
| **QwQ-32B-Preview** | 32B | Qwen 오픈소스 추론 모델 |
| **DeepSeek-R1-Preview** | - | DeepSeek 추론 모델 (조기 버전) |
| **DeepSeek-R1-671B** | 671B | DeepSeek 최대 규모 추론 모델 |

비교 대상: Qwen-Math-72B, Llama3.3-70B (전통 LLM)

### 3.2 사고(Thought) 정의 및 추출 방법

**정의:** o1 유사 모델이 "alternatively" 같은 표현으로 표시되는 인지적 단계 간 전환 포인트 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

**추출 방법:**
1. QwQ-32B-Preview 응답에서 수동으로 사고 전환 표현식 수집
2. Llama-3.3-70B 모델로 전체 응답 스캔하여 전환 표현식 식별
3. 모델의 지시 추종 능력으로 진정한 사고 전환 여부 검증

**검증:** Llama-3.3-70B 평가 정확도
- 정확한 생각: 82.9%
- 부정확한 생각: 81.8%

### 3.3 테스트 데이터셋

| 데이터셋 | 특성 | 샘플 수 |
|----------|------|--------|
| **MATH500-Hard** | 고등학교 수학 경시대회 (Level 5) | 107문제 |
| **GPQA Diamond** | 대학원 수준 객관식 문제 | 198문제 |
| **AIME2024** | 미국 수학 경시대회 2024년도 | 30문제 |
| **AIME2022-2023** | 하이퍼파라미터 튜닝용 | 120문제 |

## 4. 성능 향상 결과

### 4.1 Pass@1 정확도 개선

| 모델 | 데이터셋 | 베이스라인 | TIP 적용 | 개선도 |
|-----|---------|----------|---------|--------|
| **QwQ-32B-Preview** | MATH500-Hard | 83.1% | 83.7% | +0.6% |
| | GPQA Diamond | 57.6% | 59.1% | +1.5% |
| | AIME2024 | 38.3% | 44.1% | **+5.8%** |
| **R1-Distill-Qwen-32B** | AIME2024 | 61.4% | 64.1% | +2.7% |
| **DeepSeek-R1** | AIME2024 | 73.8% | 74.8% | +1.0% |

가장 큰 개선은 AIME2024에서 **5.8%** 향상. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

### 4.2 Underthinking 스코어 개선

가중 평균 Underthinking 스코어: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

$$\xi_{wUT} = \frac{1}{32} \sum_{i=1}^{32} \xi_{UT}(s_i)$$

| 모델 | 데이터셋 | 베이스라인 | TIP 적용 | 개선도 |
|-----|---------|----------|---------|--------|
| **QwQ-32B-Preview** | MATH500-Hard | 11.7±20.5 | 11.0±19.5 | -0.7 |
| | GPQA Diamond | 25.1±23.9 | 23.2±23.2 | -1.9 |
| | AIME2024 | 40.6±28.4 | 35.8±27.8 | **-4.8** |
| **R1-Distill-Qwen-32B** | AIME2024 | 19.6±20.6 | 17.7±20.6 | -1.9 |
| **DeepSeek-R1** | AIME2024 | 14.6±19.1 | 13.0±18.0 | -1.6 |

낮은 값이 더 좋으므로, TIP 적용으로 토큰 효율성 향상. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

### 4.3 사고 전환 패턴 변화

| 메트릭 | 베이스라인 | TIP 적용 | 변화 |
|--------|-----------|---------|------|
| 사고 전환 토큰 수 (AIME2024) | 13.8 | 5.7 | -58.7% ↓ |
| 토큰 간 평균 간격 | 580.1 | 941.6 | +62.3% ↑ |
| 생성된 전체 토큰 | 유사 | 감소 | 효율성 ↑ |

## 5. 모델 일반화 성능 분석

### 5.1 도메인별 성능 차이

**MATH500-Hard (수학 경시대회):**
- UT 스코어 감소: 11.7 → 11.0
- 정확도 개선: 소폭 (83.1% → 83.7%)
- **해석:** 이미 높은 정확도 달성, 한계 개선

**GPQA Diamond (대학원 수준 과학):**
- UT 스코어 감소: 25.1 → 23.2
- 정확도 개선: 1.5% (57.6% → 59.1%)
- **해석:** 중간 수준의 개선, 도메인 특이성 존재

**AIME2024 (수학 경시대회):**
- UT 스코어 감소: 40.6 → 35.8 (가장 큼)
- 정확도 개선: 5.8% (38.3% → 44.1%) (가장 큼)
- **해석:** TIP 방법의 효과가 가장 크게 나타남

### 5.2 모델 규모별 일반화

**모델 간 일반화 성능:**

| 측면 | 관찰 |
|-----|------|
| **소형 모델** | QwQ-32B에서 일관된 개선 |
| **중형 모델** | R1-Distill-Qwen-32B도 개선 |
| **대형 모델** | DeepSeek-R1-671B도 개선 |
| **하이퍼파라미터** | α=3, β=600이 모든 모델에 적용 가능 |

**결론:** 제안된 TIP 방법이 모델 크기와 무관하게 일반화 가능함을 시사. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

### 5.3 Task 난이도별 성능

문제 난이도별 MATH500 분석: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

| 난이도 | QwQ 토큰 (K) | QwQ 사고 수 | 추세 |
|--------|-------------|-----------|------|
| Level 1 | 11.2 | 4.0 | 낮음 |
| Level 2 | 6.2 | 2.3 | 감소 |
| Level 3 | 4.8 | 1.9 | 감소 |
| Level 4 | 4.2 | 1.5 | 감소 |
| Level 5 | 3.5 | 1.1 | 최저 |

**역설적 발견:** 어려운 문제일수록 생성 토큰이 적음 (모델이 조기에 포기하기 때문). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

## 6. 최신 관련 연구 비교 분석 (2020년 이후)

### 6.1 Overthinking vs Underthinking 구분

| 특성 | Overthinking | Underthinking |
|-----|-------------|---------------|
| **정의** | 단순 문제에 불필요한 계산 | 어려운 문제에서 불충분한 탐색 |
| **주요 논문** | Chen et al. 2024 [arxiv](https://arxiv.org/pdf/2412.21187.pdf) | Wang et al. 2025 [arxiv](https://arxiv.org/abs/2501.18585) (본 논문) |
| **영향** | 높은 지연 및 비용 | 낮은 정확도 |
| **해결책** | 조기 종료, 길이 제한 | 더 깊은 탐색 격려 |
| **심각성** | 효율성 문제 | 성능 문제 |

**Chen et al. (2024) "Do NOT Think That Much for 2+3=?"** [arxiv](https://arxiv.org/pdf/2412.21187.pdf)
- o1 유사 모델이 간단한 산술에서 과도한 시간 사용
- 추론 효율성 메트릭 제안
- 단순 문제에서의 자제를 강조

### 6.2 Test-Time Compute 스케일링 최적화 연구

**Yang et al. (2025) - "Thinking-Optimal Scaling of Test-Time Compute"** [arxiv](https://arxiv.org/abs/2502.18080)

핵심 발견:
- 긴 Chain-of-Thought(CoT)가 항상 성능 개선을 보장하지 않음
- 일부 도메인에서는 오히려 성능 저하 가능
- 최적의 추론 길이 분포가 도메인마다 다름

제안 방법 (TOPS):
1. **Format Imitation:** 다양한 추론 길이 학습
2. **Reasoning Effort-Conditioned Generation:** 문제별 차등 추론
3. **Self-Improvement:** 최적 응답으로 재훈련

성과: 동일 정확도 유지하면서 추론 토큰 30-40% 감소. [themoonlight](https://www.themoonlight.io/ko/review/towards-thinking-optimal-scaling-of-test-time-compute-for-llm-reasoning)

### 6.3 효율적 추론을 위한 조기 종료 연구

**Liu et al. (2025) - "Mitigating LLM Overthinking via RCP Detection"** [arxiv](https://arxiv.org/html/2508.17627v1)

Reasoning Completion Point (RCP) 개념:
- 모델이 첫 완전 추론 사이클 완료 지점
- 이후 추론은 중복적이고 역효과

3단계 추론 프로세스:
1. **Insufficient Exploration Stage:** 얕은 분석
2. **Compensatory Reasoning Stage:** 자기 수정 행동 (최적)
3. **Reasoning Convergence Stage:** 중복적 반복 (문제)

결과: 과도한 반복에서 일부 AIME 문제는 무한 루프 발생 [arxiv](https://arxiv.org/html/2508.17627v1)

### 6.4 강화학습 기반 효율 개선

**Han et al. (2025) - "Just-Enough Thinking (JET)"** [arxiv](https://arxiv.org/abs/2509.23392)

Evidence Accumulation 원리:
- 모델이 추론 초기에 충분한 정보 축적
- 이후 추론 단계는 대부분 중복

JET 방법:
- Trajectory Truncation: 롤아웃 중 짧은 경로 노출
- Quality-Controlled Length Reward: 간결성과 정확성 균형

결과: DeepSeek-Distill-Qwen-1.5B에서 [arxiv](https://arxiv.org/abs/2509.23392)
- 정확도: +4.6%
- 출력 길이: -46.3%
- Olympiad 벤치마크에서 뛰어난 성능

### 6.5 Best-of-N 샘플링 전략

**Self-Consistency (SC) 및 Laconic Decoding** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

Self-Consistency (Wang et al. 2023):
- 다중 추론 경로 샘플링
- 가장 일관된 답변 선택

Laconic Decoding (Raoof & Dimakis):
- 짧은 응답이 정확할 확률 높음
- N회 샘플 중 최단 응답 선택

**TIP + SC 결과 (Pass@4):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)
- QwQ-32B: 43.7% → 51.4% (+7.7%)
- R1-Distill-Qwen: 67.0% → 69.9% (+2.9%)
- DeepSeek-R1: 79.3% → 81.3% (+2.0%)

**TIP + Laconic Decoding (Pass@4):** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)
- QwQ-32B: 47.0% → 50.3% (+3.3%)
- R1-Distill-Qwen: 71.1% → 75.4% (+4.3%)
- DeepSeek-R1: 81.4% → 83.1% (+1.7%)

## 7. 모델 일반화 성능 향상 가능성

### 7.1 구조적 개선 가능성

**현재의 한계:**
- 디코딩 단계에서만 작동 (파라미터 수정 불가)
- 모델의 내재적 사고 선택 메커니즘은 미변화
- 하이퍼파라미터 α, β의 도메인 의존성 존재 가능

**향상 경로:**
1. **적응형 패널티 (Adaptive TIP):** 문제 난이도별 동적 α, β
2. **내재적 메커니즘 학습:** 모델이 자체적으로 사고 전환 규제하는 RL 학습
3. **혼합 전략 (Hybrid Approach):** 디코딩 패널티 + 프롬프트 엔지니어링

### 7.2 도메인 적응성 개선

**관찰:**
- AIME2024에서 5.8% 개선 (최고)
- MATH500-Hard에서 0.6% 개선 (최저)
- 도메인별 최적 하이퍼파라미터 존재 가능

**개선 방안:**
1. 각 도메인별 개발 집합으로 α, β 재튜닝
2. 다중 작업 학습으로 도메인 불변 표현 학습
3. Meta-learning으로 새로운 도메인 빠른 적응

### 7.3 지식 이전 (Transfer Learning)

**긍정적 신호:**
- 소형 모델(QwQ-32B)에서 대형 모델(R1-671B)까지 일관된 개선
- 서로 다른 데이터셋 구성(수학, 과학)에서도 효과 입증
- 오픈소스 모델 간 일반화 가능

**추가 실험 필요:**
- 폐쇄형 모델(GPT-4o, Claude 등)에 적용 가능성
- 다국어 추론 작업에의 확장
- 비수학 도메인(법률, 의료) 평가

### 7.4 스케일 효과 분석

**모델 규모별 응답 분석:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c617e8f-77f4-4cdd-bda5-474db6e3ffe2/2501.18585v2.pdf)

| 모델 | 파라미터 | 정확도 | UT 스코어 |
|-----|---------|--------|----------|
| QwQ-32B | 32B | 83.1% | 11.7 |
| R1-Preview | 미상 | 79.2% | - |
| R1-671B | 671B | 92.5% | 65.4 |

**해석:**
- R1-671B의 높은 정확도 (92.5%)
- 역설적으로 높은 UT 스코어 (65.4) - 더 복잡한 추론 시도
- **가설:** 대형 모델이 더 도전적인 경로를 탐색하므로 조기 전환 빈도 증가

## 8. 논문의 영향 및 향후 연구 방향

### 8.1 핵심 기여의 의미

1. **새로운 관점의 제시**
   - "Overthinking" 관점에서 벗어나 "Underthinking" 문제 제기
   - 높은 정확도 모델도 내부적 비효율 존재 입증

2. **실질적 해결책 제공**
   - 모델 수정 없이 적용 가능한 디코딩 기법
   - 다양한 모델과 데이터셋에서 검증됨

3. **평가 프레임워크 확장**
   - 단순 정확도를 넘어 추론 효율성 메트릭 도입
   - 토큰 효율성 기반의 질적 평가 가능

### 8.2 앞으로의 연구 고려사항

**단기 (6-12개월):**
1. 적응형 패널티 메커니즘 (Adaptive TIP)
   - 문제별 복잡도 감지
   - 동적 α, β 조정

2. 프롬프트 엔지니어링과의 통합
   - "사고 지속" 유도 프롬프트 + TIP
   - 최적 조합 탐색

3. 더 다양한 도메인 평가
   - 코딩, 물리학, 법률 문제
   - 비영어 언어 추론

**중기 (1-2년):**
1. 모델 학습 단계에서의 통합
   - 강화학습 보상 함수에 UT 메트릭 포함
   - 자체 조절 능력 있는 모델 학습

2. 멀티모달 추론 확장
   - 이미지 포함 수학 문제
   - 시각-언어 추론

3. 하이브리드 시스템 개발
   - 분류기로 문제 난이도 판별
   - 동적으로 추론 깊이 할당

**장기 (2년 이상):**
1. 메타인지적 추론 모델
   - 자신의 추론 상태 모니터링
   - 주의 자원 동적 할당

2. 효율성-정확도 파레토 최적화
   - 계산 예산 제약 하에서 최적 성능
   - 비용-편익 트레이드오프 이론

3. 인간-유사 추론 프레임워크
   - 시스템 1 vs 시스템 2 동적 선택
   - 메타인지적 판단 능력 구현

### 8.3 실무 적용 시 고려사항

**배포 최적화:**
1. 하이퍼파라미터 선택
   - 지연시간 vs 정확도 트레이드오프
   - 도메인별 최적값 설정

2. 기존 시스템 통합
   - API 수준의 변수 수정만 필요
   - 빠른 배포 가능

3. 비용-효율성
   - 토큰 감소로 인한 비용 절감
   - 정확도 향상으로 인한 이용자 만족도 증대

## 결론

"Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs"는 o1 유사 추론 모델들의 근본적인 비효율을 새로운 관점에서 분석한 중요한 연구입니다. **단순 정확도 증진에서 벗어나 추론 깊이와 토큰 효율성의 균형**을 강조합니다.

**주요 성과:**
- Underthinking 현상 최초 체계적 정의
- 토큰 효율성 기반 정량적 메트릭 개발
- 파라미터 수정 없는 실용적 해결책 제시
- 최대 5.8% 정확도 개선 달성

**중대한 의미:**
본 연구는 단순히 또 다른 미세 조정 기법이 아니라, o1 유사 모델들의 **내재적 추론 구조 문제**를 지적하고 이를 해결하는 첫 발걸음입니다. 이는 향후 더 효율적이고 적응적인 추론 모델 개발의 토대가 될 것으로 기대됩니다.

***

## 참고 자료 인덱스

<span style="display:none">[^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2501.18585v2.pdf

[^1_2]: https://aclanthology.org/2025.naacl-long.375

[^1_3]: https://arxiv.org/abs/2509.23392

[^1_4]: https://arxiv.org/pdf/2412.21187.pdf

[^1_5]: https://arxiv.org/abs/2501.18585

[^1_6]: https://arxiv.org/abs/2502.18080

[^1_7]: https://www.themoonlight.io/ko/review/towards-thinking-optimal-scaling-of-test-time-compute-for-llm-reasoning

[^1_8]: https://arxiv.org/html/2508.17627v1

[^1_9]: https://arxiv.org/abs/2503.19855

[^1_10]: https://arxiv.org/abs/2505.19914

[^1_11]: https://arxiv.org/html/2503.21614

[^1_12]: https://www.amazon.science/blog/the-overthinking-problem-in-ai

[^1_13]: https://arxiv.org/abs/2509.25808

[^1_14]: https://arxiv.org/abs/2508.07629

[^1_15]: https://ieeexplore.ieee.org/document/11028406/

[^1_16]: https://www.frontiersin.org/articles/10.3389/frai.2025.1614874/full

[^1_17]: https://dergipark.org.tr/en/doi/10.25282/ted.1779377

[^1_18]: https://academic.oup.com/jes/article/doi/10.1210/jendso/bvaf149.1794/8299452

[^1_19]: https://arxiv.org/html/2503.22732v1

[^1_20]: https://arxiv.org/html/2502.01081

[^1_21]: https://arxiv.org/pdf/2501.18585.pdf

[^1_22]: http://arxiv.org/pdf/2502.10867.pdf

[^1_23]: http://arxiv.org/pdf/2502.01584.pdf

[^1_24]: https://arxiv.org/pdf/2503.16040.pdf

[^1_25]: https://kimjy99.github.io/논문리뷰/huggin/

[^1_26]: https://openreview.net/forum?id=ufozo2Wc9e

[^1_27]: https://huggingface.co/papers/2501.18585

[^1_28]: https://aclanthology.org/2025.findings-emnlp.742.pdf

[^1_29]: https://openreview.net/pdf?id=bNJyu7JHUO

[^1_30]: https://discuss.pytorch.kr/t/deep-research-test-time-compute-test-time-scaling/6153

[^1_31]: https://pub.towardsai.net/stop-overthinking-a-survey-on-efficient-reasoning-for-large-language-models-paper-review-f25191bf9d3b

[^1_32]: https://openai.com/index/learning-to-reason-with-llms/

[^1_33]: https://arxiv.org/abs/2412.09078

[^1_34]: https://arxiv.org/html/2501.18585v1

[^1_35]: https://turingpost.co.kr/p/topic-26-test-time-compute

[^1_36]: https://www.reddit.com/r/LocalLLaMA/comments/1nfqe2c/whats_with_the_obsession_with_reasoning_models/

[^1_37]: https://arxiv.org/html/2503.21614v2

[^1_38]: https://arxiv.org/html/2502.18080v1

[^1_39]: https://arxiv.org/html/2503.04472v3

[^1_40]: https://arxiv.org/pdf/2511.03408.pdf

[^1_41]: https://arxiv.org/pdf/2505.11484.pdf

[^1_42]: https://arxiv.org/html/2512.02008v1

[^1_43]: https://arxiv.org/html/2504.06514v1

[^1_44]: https://arxiv.org/html/2505.22017v2

[^1_45]: https://arxiv.org/html/2509.23392v1

[^1_46]: https://neurips.cc/virtual/2025/109595
