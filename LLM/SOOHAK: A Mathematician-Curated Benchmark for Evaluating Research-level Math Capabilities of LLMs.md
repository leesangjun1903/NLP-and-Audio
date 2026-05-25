# SOOHAK: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs

> **참고 자료 (출처)**
> - **주 논문**: Son, G., Kim, S., et al. (2026). *SOOHAK: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs*. arXiv:2605.09063v3 [cs.CL], 19 May 2026.
> - **관련 인용 논문** (논문 내 참고문헌 기준):
>   - Glazer et al. (2024). *FrontierMath*. arXiv:2411.04872
>   - Hendrycks et al. (2021). *MATH dataset*. NeurIPS
>   - Cobbe et al. (2021). *GSM8K*. arXiv:2110.14168
>   - Phan et al. (2025). *Humanity's Last Exam*. arXiv:2501.14249
>   - Garre et al. (2026). *Riemann-Bench*. arXiv:2604.06802
>   - Zhang et al. (2025). *RealMath*. arXiv:2505.12575
>   - Schmitt et al. (2025). *ImProofBench*. arXiv:2509.26076
>   - Guo et al. (2025). *DeepSeek-R1*. arXiv:2501.12948
>   - Yang et al. (2025). *Qwen3 Technical Report*. arXiv:2505.09388
>   - Singh et al. (2025). *OpenAI GPT-5 System Card*. arXiv:2601.03267

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SOOHAK은 **IMO 금메달 수준 이후(post-IMO)의 다음 평가 목표**로서, 연구 수준(research-level)의 수학 문제를 통해 LLM의 진정한 수학적 추론 역량을 측정해야 한다고 주장한다. 기존 올림피아드 스타일 벤치마크가 포화 상태에 접어드는 가운데, 수학자가 직접 창작한 대학원·연구 수준 문제만이 프론티어 모델을 의미 있게 변별할 수 있다고 강조한다.

### 주요 기여 (5가지)

| 기여 항목 | 내용 |
|---|---|
| **① 고품질 벤치마크 구축** | 64명의 수학자가 처음부터 직접 작성한 439문제 (Challenge 340 + Refusal 99) |
| **② Refusal 서브셋 신설** | 비정형(ill-posed) 문제를 인식·거부하는 능력 측정 (어떤 모델도 50% 미달) |
| **③ SOOHAK-Mini 공개** | 702문항의 고교~대학원 초기 수준 서브셋으로 소형 모델 추적 가능 |
| **④ 오염 방지 설계** | 2026년 말 공개 예정의 임시 엠바고, 다단계 검증 파이프라인 |
| **⑤ 인간 기준선 제시** | 25명(5팀)의 수학자·올림피아드 메달리스트 참여 인간 기준선 ($50.6\%$ 커버리지) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**세 가지 구조적 문제**를 동시에 해결하고자 한다:

1. **벤치마크 포화(Saturation)**: MATH, GSM8K 등 기존 벤치마크는 프론티어 모델에 더 이상 변별력이 없음
2. **학습 데이터 오염(Contamination)**: 공개 시험·교재 기반 데이터셋은 훈련 데이터와 중복될 위험이 높음
3. **범위 협소성(Narrowness)**: 기존 연구 수준 벤치마크(Riemann-Bench: 25문제, FrontierMath Tier-4: 50문제)는 규모가 너무 작아 신뢰할 수 있는 평가가 어려움

### 2.2 제안하는 방법 (데이터 수집 파이프라인)

#### 5단계 파이프라인

```
[1단계] 개별 제출 + 동의 → [2단계] LLM 자동 검사 → [3단계] 결과 반환 + Opt-in
→ (의심 항목: 수동 검토) → [4단계] 최종 제출 풀 (검증된 데이터셋)
```

#### 3단계 모델 게이트 기반 난이도 분류

| 게이트 | 실패 조건 모델 | 배정 서브셋 |
|---|---|---|
| Gate 1 | Qwen3-8B, OpenThinker3-7B | SOOHAK-Mini (일부) |
| Gate 2 | gpt-oss-20B, Qwen3-32B | SOOHAK-Mini |
| Gate 3 (Challenge) | gpt-oss-120B, Qwen3-235B, DeepSeek-R1 **모두 실패** | SOOHAK Challenge |

#### 평가 메트릭 (수식)

각 모델-문제 쌍에 대해 3번 독립 샘플링 후 다음 두 지표를 보고한다:

$$\text{avg@3} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{3} \sum_{j=1}^{3} c_{i,j} \right)$$

$$\text{pass@3} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left[\max_j c_{i,j} = 1\right]$$

여기서 $c_{i,j} \in \{0, 1\}$은 문제 $i$의 $j$번째 샘플에 대한 정답 여부, $N$은 전체 문제 수이다.

#### 복합 점수 (Carefulness-adjusted)

모델의 능력과 신중함(refusal 능력)을 동시에 측정하는 복합 지표:

$$\text{Capability} = \frac{1}{2}\left(\text{SOOHAK-Mini} + \text{Challenge}\right)$$

$$\text{Avg-R} = \frac{1}{3}\left(\text{SOOHAK-Mini} + \text{Challenge} + \text{Refusal}\right)$$

$$\text{SOOHAK-R} = \frac{1}{2}\left(\text{Challenge} + \text{Refusal}\right)$$

이를 패널티 형태로 재표현하면:

$$\text{Avg-R} = \text{Capability} - \frac{1}{3}\left(\text{Capability} - \text{Refusal}\right)$$

즉, 추론 능력과 거부 성능의 격차가 클수록 $\text{Avg-R}$이 낮아지는 구조이다.

#### 답안 판정 방법

GPT-5-Mini를 **LLM judge**로 사용하여 수학적 동치 여부(mathematical equivalence)를 판단. Judge는 문제 텍스트나 풀이 없이 금 정답과 파싱된 답만 비교하여 이진 레이블 출력.

### 2.3 모델 구조 (평가 대상 모델)

SOOHAK은 새로운 모델을 제안하는 논문이 아니라 **평가 벤치마크** 논문이므로, 평가 대상 모델의 설정을 다룬다.

| 구분 | 모델 | 온도 | 추론 방식 |
|---|---|---|---|
| Closed | Gemini-3-Pro, GPT-5, Claude-Opus-4.5, Grok-4.1-Fast 등 | 1.0 또는 0.6 | Medium/Extended Thinking |
| Open-weight | Qwen3-235B, GPT-OSS-120B, Kimi-2.5, GLM-5 | 0.6 또는 1.0 | Thinking variant |

### 2.4 성능 결과

#### SOOHAK Challenge (avg@3 기준)

| 모델 | Avg@3 (%) | Pass@3 (%) |
|---|---|---|
| **Gemini-3-Pro** | **30.39** | **44.12** |
| GPT-5 | 26.37 | 40.88 |
| GPT-5-Mini | 18.82 | 28.82 |
| Grok-4.1-Fast | 18.43 | 30.88 |
| Gemini-3-Flash | 15.69 | 25.59 |
| **Kimi-2.5** (최고 오픈) | **13.87** | **20.00** |
| GPT-OSS-120B | 11.27 | 18.53 |
| Claude-Opus-4.5 | 10.39 | 18.82 |
| Qwen3-235B | 8.04 | 15.00 |
| Claude-Sonnet-4.5 | 5.69 | 10.29 |

#### SOOHAK Refusal (avg@3 기준)

| 모델 | Avg@3 (%) | 비고 |
|---|---|---|
| **GLM-5** | **49.49** | 오픈웨이트 최고 |
| GPT-5 | 43.09 | |
| Gemini-3-Flash | 43.10 | |
| GPT-OSS-120B | 43.77 | |
| Qwen3-235B | 2.69 | **최하위** |

#### Test-time Scaling 효과

$$\text{GPT-OSS-120B}: 18.53 \xrightarrow{\text{hard}} 26.47 \xrightarrow{\text{hard+81920 ctx}} 29.71$$

$$\text{Qwen3-235B}: 15.00 \xrightarrow{\text{81920 ctx}} 22.35$$

### 2.5 한계점

1. **시간 제약**: 4개월의 압축된 스케줄로 인해 리뷰 인프라 조기 구축 불가
2. **고유 정수 답 형식의 한계**: 증명, 구성적 반례 등을 다루는 고급 수학 영역을 평가하기 어려움
3. **지리적 편향**: 초기 국내(한국) 중심 모집으로 수학 세부 분야 커버리지 제한
4. **품질 검증의 한계**: 최대 약 $5\%$의 문제에 실질적 오류 존재 가능 (상한 추정)
5. **보상 설계 문제**: 난이도 기반 보상이 벤치마크 품질과 항상 일치하지 않음
6. **인간-모델 비교의 불공정성**: 인간은 4.5시간 제약, 모델은 토큰 예산 제약으로 직접 비교 부적절

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 스케일링 법칙과 일반화

SOOHAK은 LLM의 일반화 성능을 **세 가지 축**에서 분석한다:

#### (1) 파라미터 스케일링 (Train-time Compute)

Qwen3 패밀리 기준:

$$\text{Pass@3}(\text{Challenge}): 2.94\% \xrightarrow{\text{0.6B} \to \text{32B}} 15.29\%$$

약 파라미터 체크포인트마다 $+3$ 포인트 선형 증가 패턴이 관찰된다. 이는 **Challenge가 파라미터 수에 대해 아직 포화되지 않았음**을 시사한다.

#### (2) 테스트 타임 스케일링 (Test-time Compute)

토큰 예산을 $16,384 \to 81,920$ 토큰 ($5\times$)으로 확장 시:

| 모델 | 기본 Pass@3 | 81920 ctx Pass@3 | 향상 |
|---|---|---|---|
| GPT-OSS-120B (medium) | 18.53 | 29.71 | $+11.18$ pp |
| Qwen3-235B | 15.00 | 22.35 | $+7.35$ pp |

이는 현재 모델들이 **충분한 추론 시간이 주어지면 더 잘 일반화**할 수 있음을 보여준다.

#### (3) Refusal은 스케일링에 반응하지 않는다

$$\text{Challenge}: \text{스케일 증가} \propto \text{성능 향상}$$

$$\text{Refusal}: \text{스케일 증가} \not\propto \text{성능 향상}$$

이는 **수학적 문제 풀이 능력과 메타인지(자기 인식) 능력이 서로 다른 메커니즘**으로 학습됨을 강력히 시사한다. Refusal 능력의 향상을 위해서는 별도의 훈련 신호(training signal)나 RLHF 전략이 필요하다.

### 3.2 오픈 vs 클로즈드 모델의 일반화 격차

$$\Delta_{\text{일반화 격차}} = \underbrace{30.39\%}_{\text{Gemini-3-Pro}} - \underbrace{13.87\%}_{\text{Kimi-2.5 (최고 오픈)}} = 16.52\%\text{pp}$$

이 격차는 SOOHAK-Mini에서:

$$\underbrace{72.22\%}_{\text{GPT-5}} - \underbrace{66.07\%}_{\text{Kimi-2.5}} = 6.15\%\text{pp}$$

에 비해 훨씬 크다. 즉, **연구 수준의 수학으로 갈수록 오픈웨이트 모델의 일반화 열세가 급격히 심화**된다. 논문은 이를 "오픈웨이트 모델이 미발표·연구 인접 수학으로 덜 안정적으로 전이한다"고 해석하며, 이는 학습 데이터 접근성의 차이(비공개 논문, 페이월 뒤 자료 등)에 기인할 가능성을 제기한다.

### 3.3 서브필드별 일반화 특성

모델은 서브필드별로 강점이 달라지며(no universally best model), 이는 현재 LLM이 **수학의 특정 영역에만 과특화(over-specialized)**되어 있고 범용 수학적 일반화에는 미달함을 보여준다:

| MSC 영역 | 선두 모델 | 평균 (%) | 범위 (pp) |
|---|---|---|---|
| MSC 40 (Series/Summability) | Grok-4.1-Fast | 69.8 | 52.8 |
| MSC 16 (Rings/Algebras) | Gemini-3-Pro | 14.6 | 47.2 |
| MSC 60 (Probability) | Grok-4.1-Fast | 45.7 | 55.6 |
| MSC 15 (Linear Algebra) | GPT-OSS-120B | 38.7 | 42.4 |

### 3.4 일반화 향상을 위한 시사점

논문이 직접 명시하거나 결과로부터 도출할 수 있는 일반화 향상 방향:

1. **Folklore-level 추론 학습**: 기발표 정리를 결합하는 능력보다, 수학자 공동체의 휴리스틱을 활용하는 비공식 추론 능력 개발 필요
2. **다중 논문 합성 훈련**: 단일 논문 기반 문제는 이미 LLM이 해결 가능해지고 있으며, 여러 전문 논문을 교차 합성하는 능력이 다음 목표
3. **Refusal-aware 훈련**: 부정형 문제 감지 능력은 별도의 최적화 목표로 명시적으로 훈련되어야 함
4. **컨텍스트 확장**: 81,920 토큰 이상의 Extended thinking이 연구 수준 문제에 큰 효과 → 더 긴 chain-of-thought 학습 유망

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### (1) 벤치마크 설계 패러다임의 전환

SOOHAK은 "공개 자료 스크래핑 → 사람이 직접 창작"으로의 전환을 구체적으로 실현한 사례로, 향후 벤치마크 제작의 표준 워크플로우로 자리잡을 가능성이 높다. 특히 **모델 게이트 기반 동적 난이도 분류** 방식은 벤치마크 갱신 비용을 줄이면서 난이도를 유지하는 실용적 전략이다.

#### (2) Refusal/메타인지 연구의 새로운 방향

"어떤 모델도 50%를 넘지 못했다"는 결과는 **LLM의 자기 인식(self-awareness)과 불확실성 정량화** 연구를 위한 새로운 동기를 제공한다. 향후 RLHF, RLAIF, Constitutional AI 등의 방법론이 Refusal 능력 향상에 집중될 것으로 예상된다.

#### (3) 수학 AI 연구의 난이도 기준선 재설정

IMO 금메달 달성 이후 커뮤니티가 찾는 다음 마일스톤을 제시했으며, Gemini-3-Pro의 $30.39\%$가 현재 SOTA임을 확인했다. 이는 **AI 수학 연구 로드맵의 중간 목표점**으로 기능할 것이다.

#### (4) 오픈웨이트 모델 격차 연구

클로즈드-오픈 모델의 $16.52\%$ pp 격차는 **오픈소스 수학 학습 데이터의 한계**에 대한 연구를 촉진할 것이다. 특히 페이월 뒤의 고급 수학 논문에 대한 접근성 문제가 중요한 연구 주제로 부상할 것이다.

#### (5) 인간-AI 협력 수학 연구

Gemini-3-Pro가 인간 팀 통합 커버리지($50.6\%$)를 초과($60.8\%$)했다는 결과는 **AI를 수학 연구 보조 도구로 활용하는 방향의 연구**를 가속화할 것이다 (cf. Alexeev et al. 2026a,b의 단축 증명 연구).

### 4.2 향후 연구 시 반드시 고려할 점

#### (A) 평가 형식의 다양화

논문 자신이 인정하듯, **고유 정수 답 형식(unique integer answer format)**은 이미 한계에 도달하고 있다. 앞으로의 연구는:

- **증명 기반 평가**: Lean, Coq 등 proof assistant 활용
- **구성적 출력 평가**: 반례, 알고리즘, 등가류(equivalence class) 출력 검증
- **전문가 선별 채점**: 정수 채점이 불가능한 소수의 문제에 대한 전문가 평가 병행

을 통합해야 한다.

#### (B) 오염 방지의 장기적 지속 가능성

2026년 말 공개 시점 이후에는 SOOHAK도 오염 위험에 노출된다. **Living benchmark** 방식 (EternalMath, 2026) 또는 **지속적 갱신 파이프라인**이 필요하다.

#### (C) 보상 설계의 재고

난이도만을 기준으로 한 보상은 "난이도는 높지만 정보량은 낮은 문제"를 양산할 위험이 있다. 향후 벤치마크는 **난이도 × 다양성 × 진단 가치**를 함께 고려한 다차원 보상 체계가 필요하다.

#### (D) Refusal 능력의 훈련 방법론 개발

현재 어떤 모델도 Refusal에서 $50\%$를 초과하지 못했으며, 스케일링 법칙도 적용되지 않는다. 이는 **Refusal이 명시적 훈련 목표(optimization target)**로 설정되어야 함을 의미하며, 관련 데이터셋과 훈련 신호 설계가 시급한 연구 과제다.

#### (E) 전 지구적 수학자 네트워크 구축

서브필드 커버리지의 불균형을 해소하려면 특정 지역에 국한되지 않는 글로벌 수학자 커뮤니티 기반의 벤치마크 제작 인프라가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 벤치마크 | 연도 | 문제 수 | 난이도 | 오염 방지 | 인간 창작 | 특징 |
|---|---|---|---|---|---|---|
| **MATH** (Hendrycks et al.) | 2021 | 12,500 | 고교~AMC | ❌ (공개) | ❌ | 초기 표준, 프론티어 모델에 포화 |
| **GSM8K** (Cobbe et al.) | 2021 | 8,500 | 초등 수준 | ❌ | 부분 | 단계별 추론 연구 기반 |
| **OmniMATH** (Gao et al.) | 2025 | 대규모 | 올림피아드 | ❌ | ❌ | 올림피아드 통합 스위트 |
| **FrontierMath** (Glazer et al.) | 2024 | Tier4: 50 | 연구 수준 | ✅ (엑세스 제한) | ✅ | 접근 제한으로 투명성 희생 |
| **Humanity's Last Exam** (Phan et al.) | 2025 | 3,000+ | 극난이도 | ✅ | ✅ | 다분야, 약 30% 오류 논란 |
| **BeyondAIME** (ByteDance) | 2025 | - | AIME 이상 | ✅ | ✅ | AIME 이후 올림피아드 추적 |
| **Riemann-Bench** (Garre et al.) | 2026 | 25 | 연구 최첨단 | ✅ | ✅ | 규모 너무 작음 |
| **RealMath** (Zhang et al.) | 2025 | 연속형 | 연구 수준 | ✅ | ✅ | 지속적 갱신 방식 |
| **ImProofBench** (Schmitt et al.) | 2025 | - | 연구 수준 | ✅ | ✅ | 증명 생성 평가 |
| **AMO-Bench** (An et al.) | 2025 | - | 고교 올림피아드 | 부분 | ✅ | 단일 분야 한정 |
| **SOOHAK** (Son et al.) | 2026 | **439+702** | **연구 수준** | ✅ (임시 엠바고) | ✅ (105명) | **규모+Refusal+이중언어+인간기준선** |

### 차별점 분석

SOOHAK은 기존 연구들의 한계를 다음과 같이 극복한다:

$$\text{SOOHAK} = \underbrace{\text{FrontierMath의 난이도}}_{\text{연구 수준}} + \underbrace{\text{OmniMATH의 규모}}_{\text{439문제}} + \underbrace{\text{새로운 Refusal 차원}}_{\text{최초 도입}} + \underbrace{\text{이중언어 지원}}_{\text{영어·한국어}}$$

특히 **Refusal 서브셋**은 기존 어떤 벤치마크도 체계적으로 다루지 않았던 "메타인지적 수학 능력"을 최초로 측정한다는 점에서 독창적인 기여이다.

---

> **⚠️ 정확도 관련 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2605.09063v3)에만 근거합니다. 논문에 명시되지 않은 내용(예: 다른 논문의 상세 내용)은 논문 내 인용 정보를 기반으로 서술했으며, 해당 논문들의 전문을 직접 검토하지 않은 부분은 논문의 요약·인용 수준에서만 기술했음을 밝힙니다.
