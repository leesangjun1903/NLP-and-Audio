# Inside-Out: Hidden Factual Knowledge in LLMs

### 1. 핵심 주장 및 주요 기여 요약

본 논문(Gekhman et al., 2025)은 **LLM이 외부에 표현하는 지식보다 내부 매개변수에 더 많은 사실적 지식을 인코딩하고 있다**는 현상인 "**숨겨진 지식(Hidden Knowledge)**"을 체계적으로 정의하고 실증한다.[1]

주요 기여는 다음과 같다:

- **공식적 정의 제시**: 지식을 정의하는 명확한 계산 절차 제공
- **실증적 증거**: Llama-3, Mistral-7B, Gemma-2 등 3개 모델에서 평균 **40% 상대 격차** 발견
- **극단적 사례 발견**: 모델이 1,000회 샘플링 후에도 생성하지 못하면서도 완벽하게 알고 있는 답변 **9%의 질문에서 발견**
- **성능 향상 가능성 제시**: 테스트 타임 컴퓨트를 통한 **52% 향상 잠재력** 제시

***

### 2. 문제 정의 및 제안 방법

#### 2.1 해결하려는 문제

기존 연구는 다음의 문제점이 있었다:
- LLM의 "지식"에 대한 명확한 정의 부재
- 단일 생성된 답변으로만 평가 (여러 가능한 올바른 표현 무시)
- 내부 지식과 외부 표현의 차이를 정량화하지 않음

#### 2.2 지식의 형식적 정의 (Definition 1)

주어진 모델 $M$과 사실 $(s, r, o)$ (예: France, capital, Paris)에 대해:

$$K_q(s, r, o; S_M) = \frac{1}{|\Omega(s,r,o)|} \sum_{(a,\tilde{a}) \in \Omega(s,r,o)} \mathbb{I}[S_M(q,a) > S_M(q,\tilde{a})]$$ [1]

여기서:
- $Q(s,r)$: 질문의 모든 표현 (예: "What is the capital of France?")
- $\tilde{A}(o)$: 그럴듯한 답변들 (같은 타입의 모든 도시명)
- $A(o) \subseteq \tilde{A}(o)$: 올바른 답변들
- $\Omega(s,r,o)$: 올바른 답변과 그릇된 답변의 모든 쌍

전체 지식 정도:

$$K(s, r, o; S_M) = \frac{1}{|Q(s,r)|} \sum_{q \in Q(s,r)} K_q(s, r, o; S_M)$$ [1]

완전한 지식:

$$K^*(s, r, o; S_M) = \mathbb{I}[K(s, r, o; S_M) = 1]$$[1]

#### 2.3 숨겨진 지식의 정의 (Definition 2)

내부 함수 $T_M$이 외부 함수들의 집합 $S^E_M$보다 더 높은 지식을 측정할 때:

$$\frac{1}{n} \sum_{i=1}^{n} K(s_i, r_i, o_i; T_M) > \max_{S_M \in S^E_M} \left( \frac{1}{n} \sum_{i=1}^{n} K(s_i, r_i, o_i; S_M) \right) + \Delta$$[1]

**외부 함수 (Observable signals):**
- 생성 가능도: $P(a|q) = \prod_{i=1}^n P(a_i | q, a_{<i})$
- 길이 정규화: $P_{norm}(a|q) = \exp\left(\frac{1}{n} \sum_{i=1}^n \log P(a_i | q, a_{<i})\right)$
- 검증: $P(\text{True}|q,a)$ - 모델이 답변 정확성 판단 확률

**내부 함수 (Hidden representations):**
- 프로빙 분류기: 숨겨진 상태 $h_M(q,a)$에 로지스틱 회귀 적용

#### 2.4 모델 구조 및 구현

**프로빙 분류기:**
$$T_M(q,a) = \sigma(\mathbf{w}^T h_M(q,a) + b)$$ 

여기서 $h_M(q,a)$는 Q-A 쌍을 인코딩한 LLM의 숨겨진 상태, $\sigma$는 시그모이드

**지식 인식 프로빙 (Knowledge-aware Probing):**
- 훈련 데이터: 모델이 그리디 디코딩으로 올바른 답변을 생성한 질문만 선택
- 긍정 예제: 올바른 답변
- 부정 예제: 고온 샘플링으로 유도된 오답 (모델이 실제 알고 있지만 생성하지 않는 경우 모사)

이는 프로브가 텍스트 콘텐츠 암기보다는 실제 내부 지식을 학습하도록 보장

***

### 3. 실험 설계 및 결과

#### 3.1 실험 설정

**데이터셋:**
- EntityQuestions의 Wikidata 기반 사실 삼중항
- 4개 관계 선택: P26 (배우자), P176 (제조사), P264 (음반사), P50 (저자)
- 총 1,700개 QA 질문
- 관계당 500개 질문 (개발: 10%, 테스트: 90%)

**모델:**
- Llama-3-8B-Instruct
- Mistral-7B-Instruct  
- Gemma-2-9B-Instruct
- (추가) Qwen3-32B (더 큰 모델 검증용)

**평가 방법:**
- 질문당 1,000개 답변 생성
- LLM 판사 (Qwen2.5-14B)로 정확성 레이블링
- 프로브 훈련 데이터: 모델이 생성한 답변

#### 3.2 주요 결과

**결과 1: 숨겨진 지식의 일관적 존재**[1]

| 모델 | P26 | P264 | P176 | P50 | 평균 |
|------|-----|------|------|-----|------|
| Llama | +14% | +23% | +8% | +12% | **+14%** |
| Mistral | +52% | +60% | +52% | +28% | **+48%** |
| Gemma | +62% | +64% | +42% | +60% | **+57%** |

프로브(내부)와 최고 외부 함수(P(True)) 간의 **상대 격차 비교**

**결과 2: 극도로 숨겨진 지식 (Tip of the Tongue)**[1]

- 1,000회 샘플링 후 모든 답변이 오답: **56%의 질문**
- 완벽하게 알면서 생성하지 못함 ($K^*=1$ but $P(a|q) < 0.01$): **9%의 질문**
- 추가 40% 향상 가능성이 접근 불가능

**사례 연구: Volvo B58 질문**[1]

질문: "Volvo B58은 어느 회사가 생산하는가?"

| 점수 함수 | BMW | Volvo | Volvo Buses | BMW Group | K 값 |
|----------|------|-------|-------------|-----------|------|
| $P(a\|q)$ | 0.761 | 0.012 | ≈0 | 0.001 | 0.375 |
| $P_{norm}$ | 0.873 | 0.114 | ≈0 | 0.041 | 0.25 |
| $P(\text{True})$ | 0.980 | 0.110 | 0.926 | 0.941 | 0.625 |
| 프로브 | 0.465 | 0.080 | **0.980** | 0.065 | **1.0** |

프로브만 완벽하게 정렬, "Volvo Buses"가 1,000회 샘플링에서 한 번도 나타나지 않음

**결과 3: 테스트 타임 컴퓨트 개선**[1]

| 방법 | Llama | Mistral | Gemma | 평균 | 상대 개선 |
|------|-------|---------|--------|-------|----------|
| 그리디 디코딩 | 22.1 | 18.8 | 22.7 | 21.2 | - |
| 프로브 | 25.4 | 22.0 | 23.7 | 23.7 | **+12.1%** |
| 오라클 | 44.2 | 38.9 | 49.8 | 44.3 | +108.8% |
| 프로브 + 금 답변 | 34.5 | 33.9 | 27.6 | 32.0 | **+52.7%** |

프로브+금 답변 간격은 **접근 불가능한 40% 상대 향상 가능성** 시사

#### 3.3 일반화 성능 분석

**더 큰 모델에서의 일관성:**[1]

Qwen3-32B (32B 파라미터, 더 작은 1,000 → 200 샘플로 평가)
- 평균 $K$ 격차: **12.5%** (Llama 14% vs Gemma 57%와 중간)
- 격차가 **축소되지 않음**: 단순 스케일링만으로는 충분하지 않음
- 7-32B 범위 여전히 추가 완화 전략 필요

**모델 간 편차 분석:**
- Gemma는 생성과 검증 간 큰 간격 (57%)
- Llama는 상대적으로 작은 간격 (14%)
- 이는 **훈련 방식, 데이터, 아키텍처의 차이** 시사

***

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 일반화 성능 제약 요인

**1. 생성 한계 (Generation Bottleneck):**
- 모델이 내부에는 알지만 생성 확률이 극히 낮은 답변
- 오토리그레시브 생성은 $P(a|q)$ 분포에 의존
- $P(a|q)$가 낮으면 생성 가능성은 적음

**2. Post-training Sharpening 가설:**[1]
- Post-training이 확률 분포를 "날카롭게" 함
- 이전에 그럴듯하던 올바른 답변의 확률 감소
- 모델은 여전히 정확성을 알지만 생성 가능성 낮음

**3. Style vs Factuality Trade-off:**[1]
- Post-training이 유창성(style)을 강조
- 더 자연스러운 답변이 생성되지만 사실성은 낮을 수 있음

#### 4.2 일반화 향상 전략 (논문에서 제시)

**전략 1: 내부 신호 활용 디코딩**
- 현재: 토큰 확률에만 의존
- 제안: 숨겨진 표현의 내부 신호를 디코딩에 포함
- 사례: Rimsky et al. (2024)의 대조 활성화 추가 (Contrastive Activation Addition)

**전략 2: 훈련 중 Robustness 향상**

$$\mathcal{L}\_{robust} = \mathbb{E}_{q \sim \text{paraphrases}} \left[ -\log P_M(a^*|q) \right]$$

- 다양한 표현(paraphrase)에 동일한 올바른 답변 노출
- 데이터셋 레이블과 최적화 목표 동시 수정 필요

**전략 3: 강화학습(RL) 기반 개선**
- Post-training 단계에서 사실성 보상 설계
- 스타일 유창성과 사실성 간의 균형 필요

#### 4.3 최신 관련 연구 (2025년 기준)

**선행 연구들의 시사:**

1. **"Self-Improvement in Language Models: The Sharpening Mechanism"** (Huang et al., 2025)[2]
   - 모델 자신의 검증 능력이 생성보다 우수함 실증
   - Sharpening: 높은 품질 시퀀스에 높은 확률 할당 학습
   - SFT vs RLHF 접근 비교 분석

2. **"Enhancing LLM Knowledge Learning through Generalization"** (Zhu et al., 2025)[3]
   - 서로 다른 표현(paraphrase)에 동일 토큰 예측 능력과 QA 성능 상관
   - 포맷 기반 데이터 증강 (텍스트 변경 없이 형식만 다양화)
   - SAM (Sharpness-Aware Minimization) 옵티마이저 적용

3. **"LLM Knowledge is Brittle: Truthfulness Representations..."** (Haller et al., 2025)[4]
   - 내부 표현이 입력 표면 변화(typo, 표현 변경)에 매우 민감
   - OOD 샘플에서 진실성 표현의 분리가 붕괴
   - 낮은 일반화와 Shallow, Non-robust 지식 학습 시사

4. **"Do I Know This Entity?"** (Ferrando et al., 2025)[5]
   - 희소 자동인코더(SAE)로 엔티티 인식 메커니즘 발견
   - 모델의 자기-지식: 자신이 무엇을 알고 모르는지 내부에 표현
   - 할루시네이션과 거부 행동 간 인과 관계

5. **"Scaling LLM Test-Time Compute Optimally..."** (Snell et al., 2024)[6]
   - 테스트 타임 컴퓨트 스케일링은 모델 파라미터 스케일링보다 효율적
   - 검증(verifier) 모델 활용 시 4배 효율 개선

***

### 5. 논문의 한계 및 고려사항

#### 5.1 명시된 한계

1. **계산 비용:**
   - 질문당 1,000개 답변 생성, LLM 판사 평가, 다중 점수 함수 계산
   - 7-9B 모델에 집중 (더 큰 모델은 200 샘플만 사용)
   - 스케일 확장의 어려움

2. **지식 정의의 제약:**
   - 관련 사실 검증 미포함 (예: Paris가 France의 수도임을 알지만, Paris가 France의 도시임을 모르는 경우)
   - 단일 관계만 고려

3. **K* 메트릭의 민감성:**
   - 레이블 오류에 매우 민감 (0 ↔ 1 전환)
   - LLM 판사의 정확성에 의존 (99%+ 달성했으나 여전히 위험)
   - 연속 메트릭 K를 주요 평가 지표로 사용

#### 5.2 미해결 질문

1. **Post-training 메커니즘의 상세 분석 부족**
   - Sharpening 가설은 제시되었으나 인과 검증 없음
   - 구체적 훈련 절차 분석 필요

2. **모델 간 격차 원인**
   - Llama 14% vs Gemma 57% 격차의 근본 원인 불명확
   - 훈련 데이터? 아키텍처? 알고리즘?

3. **내부 함수의 최적성**
   - 선택된 선형 프로브가 최적인지 불명확
   - 더 복잡한 내부 함수 미탐색 (논문: "lower bound"로 해석)

***

### 6. 향후 연구 시사점 및 고려사항

#### 6.1 직접적 후속 연구

1. **디코딩 메커니즘 개선:**
   - 내부 신호 기반 새 디코딩 알고리즘
   - 활성화 조작(Activation Steering) 기법 적용
   - 기존 연구: Rimsky et al. (2024), 최신: Adaptive Activation Steering (Wang et al., 2025)[7]

2. **프로빙 방법론 확장:**
   - 비선형 프로브 실험
   - 희소 자동인코더(SAE) 기반 특성 분석
   - 최신: SSAEs로 희소 개념 해석[8]

3. **모델 활성화 분석:**
   - 계층별 표현 분석
   - 회로(circuit) 발견을 통한 지식 전파 메커니즘
   - 최신: ModCirc - 모듈식 회로 어휘 발견[9]

#### 6.2 일반화 개선 전략

1. **훈련 시 개선:**
   - 다양한 표현 노출 (formatting-based augmentation)
   - SAM 최적화 적용으로 일반화 능력 향상
   - 사실성 보상 기반 RL (TruthRL - 2025)[10]

2. **추론 시 개선:**
   - 검증 기반 모델 선택 (12% 상대 개선)
   - 더 나은 샘플링 전략으로 다양성 증대
   - 테스트-타임 컴퓨트 최적 할당

3. **표현 수준 개입:**
   - 진실성 방향 탐지 및 조작
   - 엔티티 인식 특성 활성화 (최신 SAE 연구)
   - 거짓과 참 표현 공간 분리

#### 6.3 실질적 응용

1. **할루시네이션 감지 및 완화:**
   - 내부 표현에서 불확실성 감지
   - 진실성 기반 거절 학습
   - 최신: SCIURus - 불확실성 회로 분석[11]

2. **신뢰성 개선:**
   - 자기-지식 활용 안전장치
   - 다중 속성 제어 (Truthfulness, Toxicity, Bias)
   - 최신: MAT-STEER - 다중 속성 스티어링[12]

3. **지식 편집 및 업데이트:**
   - 회로-인식 편집(Circuit-aware Editing)
   - 새로운 지식 통합 시 추론 경로 보존
   - 최신: CaKE - 일반화 가능한 학습자[13]

#### 6.4 기초 연구

1. **숨겨진 지식의 원인 규명:**
   - Post-training 분석 (SAE, 회로 분석)
   - 생성 vs 검증 능력 차이의 메커니즘
   - 최신: Self-Improvement와 Sharpening[2]

2. **표현 견고성:**
   - OOD 일반화 메커니즘
   - 얕은 vs 깊은 지식 학습 구분
   - 최신: Brittleness 분석[4]

3. **모델 간 비교:**
   - 아키텍처, 훈련 절차, 데이터의 영향
   - 확장 법칙 (Scaling Laws)
   - 최신: 회로 일반화 연구[14]

***

### 7. 결론

**Inside-Out** 논문은 LLM의 **지식 표현과 생성 능력 간의 근본적 간격**을 처음으로 체계적으로 정의하고 정량화했다. 평균 40%의 숨겨진 지식 격차와 극단적 사례(완벽한 내부 지식이지만 1,000회 샘플링 후에도 생성 불가)는 다음을 시사한다:

1. **LLM 이해의 새로운 관점:** 성능 한계는 지식 부족이 아닌 **표현 메커니즘의 제약**
2. **성능 향상의 기회:** 기존 테스트-타임 컴퓨트 접근의 **근본적 한계** 드러냄
3. **안전 및 신뢰성:** 숨겨진 정보의 우발적 노출 위험성

최신 2025년 연구들은 **활성화 조작, 희소 자동인코더, 회로 분석** 등을 통해 이 격차를 좁히려 시도 중이며, 특히 **일반화 성능 향상**을 위해서는:
- 훈련 시 다양한 표현에 대한 견고성
- 추론 시 내부 신호 활용
- Post-training에서의 사실성-유창성 균형

이 세 영역의 동시적 개선이 필수적임을 시사한다.

***

### 주요 수식 정리

**지식의 기본 메트릭:**
$$K_q(s,r,o;S_M) = \frac{1}{|\Omega(s,r,o)|} \sum_{(a,\tilde{a}) \in \Omega} \mathbb{I}[S_M(q,a) > S_M(q,\tilde{a})]$$

**생성 확률:**
$$P(a|q) = \prod_{i=1}^n P(a_i | q, a_{<i})$$

**길이 정규화:**
$$P_{norm}(a|q) = \exp\left(\frac{1}{n} \sum_{i=1}^n \log P(a_i | q, a_{<i})\right)$$

**프로빙 분류기:**
$$T_M(q,a) = \sigma(\mathbf{w}^T h_M(q,a) + b)$$

**숨겨진 지식 증거:**
$$\frac{1}{n} \sum_{i=1}^{n} K(s_i, r_i, o_i; T_M) > \max_{S_M \in S^E_M} \left( \frac{1}{n} \sum_{i=1}^{n} K(s_i, r_i, o_i; S_M) \right) + \Delta$$

**테스트-타임 성공 확률:**
$$\text{Pr[Success]} = \sum_{i=0}^{n} \binom{n}{i} p^i (1-p)^{n-i} \left[1 - K^{n-i}\right]^i$$

***

### 참고 문헌 (ID 기준)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ee638e58-d59f-4b6b-835a-33433e845c56/2503.15299v4.pdf)
[2](https://openreview.net/forum?id=WJaUkwci9o)
[3](https://aclanthology.org/2025.findings-emnlp.469/)
[4](https://arxiv.org/html/2510.11905v1)
[5](https://openreview.net/forum?id=WCRQFlji2q)
[6](https://arxiv.org/abs/2408.03314)
[7](https://openreview.net/forum?id=NBHOdQJ1VE)
[8](https://aclanthology.org/2025.findings-naacl.87.pdf)
[9](https://openreview.net/forum?id=do5vVfKEXZ)
[10](https://chatpaper.com/paper/194257)
[11](https://aclanthology.org/2025.naacl-long.618/)
[12](https://aclanthology.org/2025.acl-long.1007.pdf)
[13](https://arxiv.org/html/2503.16356)
[14](https://arxiv.org/pdf/2411.16105.pdf)
[15](https://arxiv.org/pdf/2503.15299.pdf)
[16](https://arxiv.org/pdf/2503.03705.pdf)
[17](https://arxiv.org/html/2310.09725v3)
[18](https://arxiv.org/pdf/2306.08302.pdf)
[19](https://arxiv.org/pdf/2305.04757.pdf)
[20](https://arxiv.org/pdf/2502.12598.pdf)
[21](http://arxiv.org/pdf/2410.08255.pdf)
[22](https://openreview.net/forum?id=f7GG1MbsSM)
[23](https://openreview.net/forum?id=R7qRUFHGTx)
[24](https://arxiv.org/abs/2503.03705)
[25](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-23-analyzing-model-behavior/probing-internal-representations)
[26](https://www.ijcai.org/proceedings/2025/0687.pdf)
[27](https://arxiv.org/abs/2510.11905)
[28](https://aipaper.tistory.com/59)
[29](https://arxiv.org/html/2510.01070v1)
[30](http://arxiv.org/pdf/2503.02078.pdf)
[31](http://arxiv.org/pdf/2410.20488.pdf)
[32](http://arxiv.org/pdf/2412.11043.pdf)
[33](http://arxiv.org/pdf/2502.03199.pdf)
[34](https://arxiv.org/html/2501.14082v1)
[35](https://arxiv.org/pdf/2402.14433.pdf)
[36](https://arxiv.org/html/2501.17994v1)
[37](http://arxiv.org/pdf/2503.00491.pdf)
[38](https://arxiv.org/html/2511.06571v1)
[39](https://dejan.ai/blog/advanced-interpretability-techniques-for-tracing-llm-activations/)
[40](https://arxiv.org/html/2508.06030v1)
[41](https://www.rohan-paul.com/p/self-improvement-in-language-models)
[42](https://pair.withgoogle.com/explorables/patchscopes/)
[43](https://aclanthology.org/2025.emnlp-main.826.pdf)
[44](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models)
[45](https://aclanthology.org/2025.findings-emnlp.401.pdf)
[46](https://arxiv.org/html/2502.12203v1)
[47](http://arxiv.org/pdf/2406.16985.pdf)
[48](https://arxiv.org/pdf/2406.16033.pdf)
[49](https://arxiv.org/pdf/2401.03646.pdf)
[50](https://arxiv.org/html/2408.08590)
[51](https://arxiv.org/abs/2407.04307)
[52](https://proceedings.mlr.press/v267/he25x.html)
[53](https://www.pnas.org/doi/10.1073/pnas.2506316122)
[54](https://arxiv.org/abs/2406.11944)
[55](https://aclanthology.org/2024.findings-acl.737.pdf)
