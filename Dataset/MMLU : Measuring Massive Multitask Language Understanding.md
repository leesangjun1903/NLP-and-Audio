# Measuring Massive Multitask Language Understanding

## 1. 핵심 주장과 주요 기여 (간결한 요약)

본 논문은 기존 NLP 벤치마크(GLUE, SuperGLUE 등)들이 모델의 실제 광범위한 지식과 추론 능력을 측정하지 못한다고 주장한다. GLUE나 SuperGLUE에서 모델들이 빠르게 인간 수준 성능에 도달했지만, 이는 진정한 언어 이해를 반영하지 못한다는 것이다.[1]

**주요 기여:**

1. **광범위한 멀티태스크 벤치마크**: 57개 과목, 15,908개 다중선택형 질문으로 구성된 MMLU 벤치마크 개발. 이는 인문학, 사회과학, STEM 등 초등학교부터 전문가 수준까지의 난이도를 포함.[1]

2. **초기 모델 성능 실증**: GPT-3 (175B)는 43.9% (소수샷)를 달성하며 무작위 수준(25%)을 초과했으나 전문가 수준(약 90%)에 미치지 못함을 보여줌.[1]

3. **모델 구조적 한계 규명**: 선언적 지식과 절차적 지식의 불균형, 도메인별 편향된 성능, 신뢰도-정확도 불일치(캘리브레이션 문제)를 규명.[1]

4. **새로운 평가 방법론**: 미세조정 없이 제로샷 및 소수샷 학습만으로 평가하여 모델의 실제 일반화 능력을 측정.[1]

***

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하는 문제

벤치마크 포화 현상이 가장 근본적인 문제였다. 기존 벤치마크에서 모델들이 빠르게 인간 수준에 도달하면서 모델 간 구분이 어려워졌다. 더 중요한 것은 트랜스포머 모델들이 인터넷, 책, 웹사이트를 포함한 거대한 텍스트 코퍼스에서 학습한 광범위한 지식이 기존 벤치마크로는 제대로 평가되지 않는다는 점이었다.[1]

### 2.2 제안 방법론

#### 벤치마크 구성

MMLU는 다음과 같이 구조화됨:
- 총 15,908개의 다중선택형 질문
- **57개 과목**: Abstract Algebra, Anatomy, Astronomy부터 World Religions까지
- **4개의 주요 범주**: 인문학, 사회과학, STEM, 기타
- **난이도 레벨**: 초등학교, 고등학교, 대학, 전문가 수준
- **데이터셋 분할**:
  - 개발 세트(Dev): 과목당 5개 질문
  - 검증 세트: 1,540개 질문
  - 테스트 세트: 14,079개 질문 (과목당 최소 100개)[1]

#### 평가 방법론 (수식 포함)

**정확도 계산:**

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Questions}}$$

**소수샷 프롬프트 구성:**

$$\text{Prompt}_k = \text{"The following are multiple choice questions about [subject]."} + \sum_{i=1}^{k} (\text{Question}_i, \text{Answer}_i) + \text{"Answer: "}$$

여기서 $k \in \{0, 1, 2, 3, 4, 5\}$ (0은 제로샷, 1-5는 소수샷)[1]

**신뢰도(Confidence) 계산:**

$$\text{Confidence}(x) = \max_{c \in \{A,B,C,D\}} P(c | x)$$

모델이 각 선택지에 할당하는 확률 중 최댓값[1]

**기대 캘리브레이션 오차 (Expected Calibration Error, ECE):**

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{accuracy}(B_m) - \text{confidence}(B_m) \right|$$

여기서 $B_m$은 신뢰도가 $(\frac{m-1}{M}, \frac{m}{M}]$ 범위의 샘플 집합, $M$은 구간 개수(일반적으로 10)[1]

### 2.3 평가 대상 모델

1. **GPT-3 (크기별)**:
   - Small (2.7B): 25.9% (소수샷)
   - Medium (6.7B): 24.9%
   - Large (13B): 26.0%
   - X-Large (175B): **43.9%** (주요 결과)[1]

2. **UnifiedQA** (T5 기반, 미세조정): 48.9%

3. **기타 모델** (미세조정):
   - RoBERTa-base: 27.9%
   - ALBERT-xxlarge: 27.1%
   - GPT-2: 32.4%[1]

### 2.4 성능 분석

#### 도메인별 성능 (GPT-3, 소수샷)

| 도메인 | 인문학 | 사회과학 | STEM | 기타 | 평균 |
|--------|--------|---------|------|------|------|
| 정확도(%) | 40.8 | 50.4 | 36.7 | 48.8 | 43.9 |

**극단적 성능 편차**:
- 최고: US Foreign Policy (69%)
- 최저: College Chemistry (26%)
- 범위: 43% 차이[1]

#### 핵심 성능 특성

**1. 선언적 vs 절차적 지식 불균형**

모델은 개념을 알지만 적용하지 못하는 현상 발견. 예: PEMDAS(연산 순서) 정의에 대한 질문에는 정답하지만, $(1 + 1) \times 2$ 계산에서는 오답 제시.[1]

**2. 편향된 성능 분포**

특정 과목에서 전문가 수준 성능을 보이지 못함. 인간과 달리 어떤 분야에서도 90% 이상 정확도 미달성.[1]

**3. 도메인별 강약점**

| 강점 분야 | 약점 분야 |
|---------|---------|
| 문화·정책 (70% 대) | 계산 집약적 STEM (26-30%) |
| 언어 기반 과목 | 법률·윤리 (32% 대) |

**4. 비정형 학습 패턴**

대학 의학(47.4%) > 대학 수학(35.0%) > 초등 수학(29.9%) 
- 인간의 교육 순서와 역행.[1]

### 2.5 캘리브레이션 분석

**신뢰도-정확도 불일치 심각:**

$$\text{Calibration Gap} = |\text{Mean Confidence} - \text{Actual Accuracy}|$$

- 제로샷: 최대 24% 차이 (Formal Logic에서 신뢰도 60% vs 정확도 36%)
- 소수샷: 최대 14% 차이 (개선되나 여전히 신뢰할 수 없음)
- 특정 과목 RMS 오차 (Elementary Mathematics): 19.4%[1]

**해석**: 모델이 자신의 예측 신뢰성을 제대로 판단하지 못하므로 고위험 응용에 부적합.

### 2.6 모델의 주요 한계

**1. 도메인 특화 성능 부재**
- 모든 57개 과목에서 전문가 수준(90%) 미달성[1]

**2. 법률·윤리의 극심한 취약성**
- Professional Law: 32%
- Moral Scenarios: 유사 수준
- **사회적 함의**: 미래 AI의 법적·윤리적 신뢰성 확보 필수.[1]

**3. 계산 능력 부족**
- 초등수학: 29.9%
- College Chemistry: 26%[1]

**4. 학습 데이터 오염 문제**
- 추가 법률 학습 데이터(2,000개) 미세조정: 32.8%
- Harvard 법률 코퍼스 사전학습(160만 문서) 후 미세조정: 36.1%
- **결론**: 고품질 데이터만으로도 성능 개선 한계.[1]

***

## 3. 모델 일반화 성능 향상 가능성 (중점 분석)

### 3.1 스케일링의 한계와 임계값

**현재 문제:**

$$\text{Performance Gap: } 43.9\% - 25\% = 18.9\% \text{ (175B 대비 13B)}$$

크기 13B까지는 무작위 수준이고, 175B에서 비로소 유의미한 진전이 발생. 이는 규모만으로 일반화 능력을 단순 증진할 수 없음을 시사.[1]

**최신 스케일링 법칙 (Chinchilla, 2022):**

$$\text{Loss}(N, D) = E N^{-\alpha} + B D^{-\beta} + C$$

여기서 $\alpha \approx 0.07$, $\beta \approx 0.10$[2]

**실무 적용:**
- 10배 모델 확대 시 약 5배 데이터 증대 필요
- 기존 모델들이 크기 대비 과소 학습 상태[2]

### 3.2 일반화 능력 향상 전략

#### 1. 체인-오브-쏘트 프롬프팅

**최신 연구 발견 (2024-2025):**
- MMLU-Pro에서 GPT-4 Turbo: 직접 응답 대비 **15.3% 정확도 향상** (CoT 사용 시)[3]
- MMLU 원본에서는 CoT 효과 작음 (질문이 쉬워 명시적 추론 불필요)[4]

**의미**: 더 어려운 질문에서 추론 증진이 효과적.[3]

#### 2. 미세조정과 사전학습의 상호작용

**UnifiedQA 사례:**
- T5 기반 (11B): 48.9% (GPT-3 X-Large 43.9% 초과)
- GPT-2 (1.5B): 32.4%
- **결론**: 규모 외에도 사전학습 데이터셋과 미세조정 품질이 중요[1]

#### 3. 불확실성 추정 기반 선택적 분류

**원리:**
모델이 신뢰도 낮은 예측을 거부할 때:

$$\text{Risk} = P(\text{Error} | \text{Prediction})$$

정확도와 신뢰도를 개선[5]

**최신 발견 (2025):**
언어적 불확실성(Linguistic Verbal Uncertainty, LVU)이 토큰 확률(TPU) 기반 방법보다 우수[5]

$$\text{AUROC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

여기서 더 높은 AUROC는 정확한 신뢰도 판단 의미[5]

#### 4. Mixture of Experts 구조의 잠재성

**발견:**
- MoE 모델이 조밀한 모델보다 불확실성 추정 신뢰성 높음[6]
- 구조적 다양성이 도메인 특화 성능 개선에 기여[5]

### 3.3 새로운 벤치마크를 통한 일반화 검증

#### MMLU-Pro (2024)

**개선사항:**
- 질문 난이도 증가
- 선택지 4개 → 10개 (삼중 검증 포함)

**성능 변화:**
- GPT-4: 86.4% (MMLU) → 72.6% (MMLU-Pro)
- **14% 이상 저하**, 더 정교한 추론 요구[7]
- Chain-of-Thought 필수적[4]

#### MMLU-ProX (2025) - 다언어 확장

**발견:**
- 고리소스 언어(영어 등): 70% 이상
- 저리소스 언어(스와힐리어): 약 40%
- **의미**: 일반화 능력이 언어-문화 맥락에 크게 의존[8]

#### MMLU-Redux: 품질 관리

**발견:**
약 6.49% 질문에 오류나 모호함[9]
- 공개 데이터셋 때문에 학습 데이터 포함 가능
- 성능 메트릭에 영향[9]

### 3.4 일반화 실패 원인과 개선 방향

#### 문제 1: 표면 수준 휴리스틱 학습

**증거:**
- 선택지 변경 시 25-30% 정확도 저하[10]
- 의미 있는 프롬프트를 무의미 토큰으로 교체해도 94% 성능 유지[11]

**원인**: 문제 구조 이해 없이 패턴 매칭 수행[11]

#### 문제 2: 절차적 지식 부족

**현상**: PEMDAS 알지만 적용 못함[1]

**개선 방안:**
$$\text{Procedural Learning} = \text{Declarative Knowledge} \otimes \text{Action Policy} \otimes \text{Feedback}$$
- 중간 추론 단계의 명시적 표현 학습
- 강화 학습 기반 정책 학습

#### 문제 3: 도메인 특화 지식 부족

**법률 분야 사례 (논문 실험):**
- RoBERTa-base + 커스텀 법률 데이터 (2K): 32.8%
- Harvard 법률 코퍼스 사전학습 (1.6M 문서): 36.1%
- **한계**: 추가 데이터만으로 불충분[1]

**개선 방안:**
- 법률 구조화된 온톨로지 통합
- 신경-기호 하이브리드 모델
- 도메인 전문가 협력 기반 강화학습

***

## 4. 미래 연구에 미치는 영향과 고려사항

### 4.1 MMLU가 미친 학계·산업 영향

#### 1. 평가 표준의 확립
MMLU는 AI 평가의 사실상 표준(de facto standard)으로 확립. 주요 모델(GPT-4, Claude, PaLM 2, LLaMA 등)의 개발사들이 MMLU 점수를 필수 보고. 2023년 GPT-4 (86.4%), 최신 모델들이 88-90% 근처에 위치.[12][7]

#### 2. 벤치마크 연쇄 개발 촉발

| 벤치마크 | 개선 사항 |
|----------|----------|
| MMLU-Pro (2024) | 난이도 상향, 10선택지 |
| MMLU-Pro+ (2024) | 단축 학습 감지 |
| MMLU-CF (2024) | 오염 제거 |
| MMLU-ProX (2025) | 43개 언어 확장 |
| MMLU-Redux (2024) | 6.49% 오류 제거 |

#### 3. 연구 방향 촉발

**스케일링 연구:**
MMLU 성능을 기반으로 스케일링 법칙 정합성 검증[2]

**추론 능력 강화:**
Chain-of-Thought, 자기 반성 프롬프팅, reasoning 모드 개발[3]

**정렬(Alignment) 연구:**
법률·윤리 과목의 낮은 성능이 RLHF 기반 정렬 연구 동기 제공[1]

#### 4. 산업 응용
의료, 법률, 금융 등 고위험 영역에서 모델 신뢰성 평가 기준[12]

### 4.2 앞으로의 연구 시 고려사항

#### 1. 도메인 적응 연구 강화

**필요성:**
법률·윤리 분야의 낮은 성능(32%)은 특수 전략 필수.

**방향:**
$$\text{Transfer Learning} = \text{Pre-training} + \text{Domain Adaptation} + \text{Few-Shot Tuning}$$

- 도메인 특화 사전학습 강화
- 온톨로지/규칙 기반 시스템과의 신경-기호 통합[13]
- 전문가 피드백 기반 강화학습

#### 2. 절차적 지식 획득 메커니즘 규명

**핵심 질문:**
왜 "PEMDAS 정의"는 맞지만 계산은 틀릴까?

**연구 방향:**
- 중간 표현 단계 명시화
- 연쇄 추론(Chain-of-Thought) 자동화
- 오류 피드백 기반 학습[14]

#### 3. 모델 캘리브레이션 신뢰도 개선

**최신 진전 (2025):**
LVU 기반 불확실성 추정이 ECE 개선[5]

**구현 전략:**
$$\text{Confidence}_{\text{calibrated}} = f(\text{Token Probability}, \text{Linguistic Signals}, \text{Ensemble Votes})$$

- 다양한 불확실성 지표 조합
- 작업 특화 캘리브레이션

#### 4. 데이터 오염 문제 근본 해결

**현재 상황:**
MMLU의 ~6.5% 오류로 인한 점수 부풀음[9]

**대응:**
- MMLU-CF (Contamination Free) 별도 관리
- 시간 기반 분리 (벤치마크 발표 후 데이터만 사용)
- 질문 다양화로 중복 최소화

#### 5. 다언어 및 다문화 일반화

**발견:**
고리소스 언어 70% vs 저리소스 언어 40% (MMLU-ProX)[8]

**개선 방향:**
- 저리소스 언어 학습 데이터 강화
- 문화 간 지식 차이 모델링
- 번역 품질 개선

#### 6. 추론 강화 기반 성능 개선

**최신 경향 (o1, o3 모델):**
추론 단계에서 더 많은 계산 사용

**적용:**
$$\text{Quality}_{\text{output}} = \int_0^{T} Q(\text{Step}_t) \, dt$$

더 많은 중간 단계 생성 → 더 나은 최종 답변[2]

#### 7. 멀티모달 확장

**현재 한계:**
MMLU는 텍스트만 포함. 이미지, 다이어그램 필요한 과목 평가 불가.

**미래 방향:**
- MMLU-Vision 개발
- 과학·의학·공학의 시각적 문제 포함

***

## 결론

MMLU 벤치마크는 단순한 평가 도구를 넘어 **LLM 개발의 방향성을 결정하는 역할**을 수행했다.[1]

**핵심 성과:**
- 43.9% GPT-3 성능이 대부분 모델의 무작위 수준을 드러냄[1]
- 도메인별 편향 성능이 특정 분야의 근본적 약점 규명[1]
- 캘리브레이션 문제가 신뢰할 수 있는 AI 시스템의 핵심 과제임을 제시[1]

**미래 연구의 우선순위:**
1. **절차적 지식 습득 메커니즘** 규명
2. **도메인 특화 성능** 개선 (특히 법률·윤리)
3. **모델 불확실성 추정** 신뢰성 강화
4. **다언어 및 다문화** 일반화 능력 개발
5. **데이터 오염** 문제의 근본적 해결

MMLU-Pro, MMLU-ProX, MMLU-CF 등의 파생 벤치마크들은 평가 방법론의 더욱 정교한 발전을 이끌고 있으며, 이는 AI 모델의 신뢰성 있는 평가와 체계적 개선으로 이어질 것으로 예상된다.[7][8][4][9]

***

## 참고자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/83e0d496-8284-4b9f-88ec-f9b390515102/2009.03300v3.pdf)
[2](https://cameronrwolfe.substack.com/p/llm-scaling-laws)
[3](https://www.rohan-paul.com/p/zero-shot-and-few-shot-learning-techniques)
[4](http://arxiv.org/pdf/2406.01574v4.pdf)
[5](https://arxiv.org/abs/2505.23854)
[6](http://arxiv.org/pdf/2502.17187.pdf)
[7](https://www.linkedin.com/pulse/latest-evaluation-benchmarks-large-language-models-jin-sdfse)
[8](https://arxiv.org/pdf/2503.10497.pdf)
[9](https://arxiv.org/pdf/2406.04127.pdf)
[10](http://arxiv.org/pdf/2502.12896.pdf)
[11](https://arxiv.org/html/2502.14318v1)
[12](https://www.datacamp.com/blog/what-is-mmlu)
[13](http://arxiv.org/pdf/2501.11599.pdf)
[14](https://arxiv.org/html/2504.02181v1)
[15](https://arxiv.org/pdf/2406.01382.pdf)
[16](https://arxiv.org/pdf/2410.22368.pdf)
[17](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/exsy.13243)
[18](https://arxiv.org/pdf/2406.01574.pdf)
[19](http://arxiv.org/pdf/2409.02257v2.pdf)
[20](https://arxiv.org/pdf/2401.04757.pdf)
[21](https://academic.oup.com/bib/article/25/5/bbae354/7739674)
[22](https://arxiv.org/html/2510.20460v1)
[23](https://aclanthology.org/2025.naacl-industry.77/)
[24](https://arxiv.org/pdf/2502.11830.pdf)
[25](https://openreview.net/pdf/d91aa9a8e3eb5236b22b1e010d2fcf734989adec.pdf)
[26](https://arxiv.org/html/2509.04013v1)
[27](https://arxiv.org/pdf/2305.18395.pdf)
[28](https://arxiv.org/pdf/2205.00445.pdf)
[29](https://aclanthology.org/2023.emnlp-main.228.pdf)
[30](https://www.ijcai.org/proceedings/2022/0232.pdf)
[31](https://mbrenndoerfer.com/writing/big-bench-mmlu-comprehensive-evaluation-benchmarks-large-language-models)
[32](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06583.pdf)
[33](https://www.projectpro.io/article/mmlu-benchmark/1162)
[34](https://www.coursera.org/articles/scaling-laws-for-neural-language-models)
[35](https://arxiv.org/html/2504.04017v1)
