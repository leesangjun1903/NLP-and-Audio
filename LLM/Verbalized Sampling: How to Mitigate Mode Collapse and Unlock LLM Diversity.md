# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity

### 1. 핵심 주장 및 주요 기여

본 논문의 핵심은 **모드 붕괴(mode collapse)가 알고리즘 제한이 아닌 데이터 수준의 문제**라는 주장에 있습니다. 연구자들은 인지심리학 원리에 기반한 **타이피컬리티 바이어스(typicality bias)**—즉, 주석자들이 체계적으로 친숙한 텍스트를 선호하는 경향—이 모드 붕괴의 근본 원인임을 식별했습니다.[1]

이를 해결하기 위해 **Verbalized Sampling(VS)**이라는 훈련 없는 프롬프팅 전략을 제안했습니다. VS는 모델에게 명시적으로 단일 응답이 아닌 확률 분포를 생성하도록 요청합니다. 예를 들어, "커피 농담 5개와 각각의 확률을 생성하세요"와 같은 방식입니다.

**주요 기여:**
- 모드 붕괴의 새로운 이론적 프레임워크 제시 및 타이피컬리티 바이어스의 실증적 검증
- 훈련 불필요한 추론 시간 솔루션 제공
- 다양한 작업에서 1.6-2.1배 다양성 증가를 보여주는 포괄적 실험[1]

### 2. 해결 문제와 제안 방법

#### 2.1 해결하고자 하는 문제

포스트 트레이닝 정렬(post-training alignment) 방법, 특히 RLHF는 의도하지 않게 LLM의 다양성을 크게 감소시킵니다. 이는 창의적 쓰기, 사회 시뮬레이션, 합성 데이터 생성과 같은 다양성이 중요한 응용 분야의 효과를 제한합니다.[1]

기존 연구들은 이 현상을 알고리즘 원인으로 돌렸습니다:
- 불충분한 보상 모델(inadequate reward models)[1]
- KL 정규화 최적화가 다수파 응답을 증폭[1]
- 감독된 미세 조정(SFT)의 교차 엔트로피 손실 함수[1]

그러나 본 논문은 이러한 알고리즘적 원인들이 타이피컬리티 바이어스라는 데이터 수준의 문제에 의해 악화된다고 주장합니다.

#### 2.2 타이피컬리티 바이어스의 이론적 모델링

논문은 Bradley-Terry 모델 기반으로 보상 함수를 다음과 같이 공식화합니다:

$$ r(x, y) = r_{true}(x, y) + \alpha \log \pi_{ref}(y | x) + \epsilon(x) $$

여기서:
- $$r_{true}$$: 진정한 작업 효용
- $$\alpha$$: 타이피컬리티 바이어스 가중치
- $$\log \pi_{ref}(y|x)$$: 사전학습된 기본 모델의 로그 우도(텍스트 타이피컬리티 프록시)
- $$\epsilon(x)$$: 노이즈항

RLHF 최적화 목표:

$$ \max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ r(x, y) - \beta \text{KL}(\pi(\cdot | x) \| \pi_{ref}(\cdot | x)) \right] $$

#### 2.3 모드 붕괴 메커니즘

Rafailov et al.(2024)의 폐쇄형 해(closed-form solution)를 식(1)에 대입하면:[1]

$$ \pi^{*}(y | x) \propto \pi_{ref}(y | x)^{\gamma} \exp\left(\frac{r_{true}(x, y)}{\beta}\right) $$

여기서 **γ := 1 + α/β > 1** (α > 0일 때)

이 γ > 1 조건이 핵심입니다. 이는 기본 모델의 분포를 날카롭게 만드는 **온도 스케일링 효과**를 유발합니다.[1]

**모드 붕괴의 조건**: 응답 집합 S의 모든 쌍 (y, y') ∈ S에 대해 $$r_{true}(x, y) = r_{true}(x, y')$$인 경우 (즉, 진정한 효용이 동일한 경우), 최적 정책은:

$$ \pi^{*}(\cdot | x) \propto \pi_{ref}(\cdot | x)^{\gamma} \text{ on } S, \quad \gamma > 1 $$

γ가 매우 클 때: $$y^* \in \arg\max_y \pi_{ref}(y | x)$$

이는 **창의적 쓰기, 대화 시뮬레이션 등 다양한 고품질 답변이 존재하는 작업에서 기본 모델의 최빈값으로 확률이 집중**됨을 의미합니다.

#### 2.4 타이피컬리티 바이어스의 실증적 검증

HELPSTEER 데이터셋에서 정확성이 일치하는 6,874개 쌍을 분석한 결과:
- Llama 3.1 405B Base: $$\hat{\alpha} = 0.57 \pm 0.07$$ (p < 10^{-14})
- GLM 4.5 Base: $$\hat{\alpha} = 0.65 \pm 0.07$$ (p < 10^{-14})

이는 인간 평가자들이 정확성과 무관하게 타이피컬한 응답을 선호함을 증명합니다.[1]

더 광범위한 검증:
- 4개 선호도 데이터셋(OpenAI TL;DR, UltraFeedback, NVIDIA HelpSteer-v2, Skywork Preference)
- 5개 기본 모델에서 4-12 퍼센트 포인트의 타이피컬리티 바이어스 확인[1]

#### 2.5 제안 방법: Verbalized Sampling

##### 프롬프트 수준의 모드 분류

논문은 세 가지 프롬프팅 전략이 서로 다른 모드로 붕괴됨을 보여줍니다:

| 프롬프트 유형 | 예시 | 모드 특성 |
|-----------|------|---------|
| **인스턴스 수준** | "커피 농담을 말해봐" | 기본 모델의 최빈값 농담 |
| **리스트 수준** | "커피 농담 5개를 말해봐" | 균일 분포 (기본 모델 학습) |
| **분포 수준(VS)** | "커피 농담 5개와 확률을 말해봐" | 기본 모델의 분포 근사 |

이론적 근거: 분포 수준 프롬프트는 모델이 사전학습 중 학습한 온전한 분포에 접근하도록 명시적으로 지시합니다.

##### VS 구현

**VS-Standard**: 기본 형태
```
"Generate 5 responses with their corresponding probabilities.
Responses should each include a <text> and a numeric <probability>."
```

**VS-CoT**: 체계적 사고 포함
```
"Think step-by-step, then generate 5 responses with probabilities."
```

**VS-Multi**: 다중 턴
```
Turn 1: "Generate 5 responses with probabilities"
Turn 2+: "Generate 5 more responses with probabilities"
```

### 3. 모델 구조 및 성능 향상

#### 3.1 실험 설정

**모델 테스트:**
- 폐쇄형: GPT-4.1-mini, GPT-4.1, Gemini-2.5-Flash, Gemini-2.5-Pro, Claude-3.7-Sonnet, Claude-4-Sonnet
- 오픈소스: Llama-3.1-70B-Instruct, Qwen3-235B
- 추론 모델: OpenAI o3, DeepSeek R1[1]

**작업 범위:**
1. 창의적 쓰기: 시 계속하기, 이야기 생성, 농담 작성
2. 대화 시뮬레이션: PersuasionForGood 데이터셋
3. 개방형 QA: CoverageQA 벤치마크 (50개 미국 주 이름 지정 등)
4. 합성 데이터 생성: 수학 문제 생성 및 미세 조정

#### 3.2 창의적 쓰기 성능

**의미론적 다양성 점수 (%):**

| 방법 | 시 | 이야기 | 농담 |
|------|------|---------|---------|
| **Direct** | 11.4 | 22.2 | 30.0 |
| **CoT** | 12.2 | 23.2 | 39.9 |
| **Sequence** | 18.3 | 29.6 | 58.8 |
| **Multi-turn** | 14.9 | 26.0 | 57.6 |
| **VS-Standard** | 21.9 | 34.7 | 62.5 |
| **VS-CoT** | 25.8 | 38.2 | 62.9 |
| **VS-Multi** | 23.2 | 36.0 | 62.8 |

**결과:** VS 방법들이 모든 작업에서 기준선을 초과합니다. 농담에서 1.6-2.1배 향상, VS-CoT가 이야기 생성에서 최고 성능을 보입니다.[1]

#### 3.3 품질-다양성 트레이드오프

![다양성-품질 파레토 프론트(Figure 4d)]

- VS-CoT와 VS-Multi은 파레토 프론트에 접근
- 품질이 유지되거나 개선되면서 다양성 획득
- 더 큰 모델(GPT-4.1, Gemini-2.5-Pro)일수록 더 큰 이득 (1.5-2배)[1]

#### 3.4 대화 시뮬레이션 성능

PersuasionForGood 데이터셋 (939개 유효한 인스턴스):

**기부 금액 분포 정렬:**
- 더 큰 모델과 추론 모델이 VS에서 더 나은 성과
- GPT-4.1 + VS는 미세 조정된 Llama-3.1-8B과 동등
- DeepSeek-R1은 미세 조정 모델을 초과[1]

**언어학적 정렬:**
- Distinct-1/2/3 다양성 향상
- 의미론적 다양성 증가
- 가독성은 여전히 개선 여지 있음[1]

#### 3.5 개방형 QA 성능

40개 질문 (평균 20개 이상의 정답):

| 메트릭 | Direct | Sequence | VS-Standard | VS-CoT | VS-Multi |
|--------|--------|----------|-------------|--------|----------|
| **KL Divergence** ↓ | 14.43 | 4.27 | 3.50 | 3.23 | 3.22 |
| **Coverage-N** ↑ | 0.10 | 0.64 | 0.67 | 0.68 | 0.71 |
| **Precision** ↑ | 1.00 | 0.96 | 0.96 | 0.96 | 0.96 |

**해석:** VS가 사전학습 분포에 더 잘 정렬되며 (KL=0.12), 정확성을 희생하지 않으면서 답변 범위 확대.[1]

#### 3.6 합성 데이터 생성 성능

1,000개 합성 수학 문제 생성 후 미세 조정:

| 데이터셋 | Baseline | Direct | Sequence | **VS-Multi** |
|---------|----------|--------|----------|------------|
| **MATH500** | 27.2 | 26.1 | 30.5 | 34.8 |
| **OlympiadBench** | 27.2 | 24.9 | 28.2 | 31.7 |
| **Minerva Math** | 40.7 | 36.9 | 42.5 | 43.6 |
| **평균** | 32.8 | 30.6 | 34.3 | **37.5** |

더 다양한 합성 데이터가 직접 프롬프팅(모드 붕괴 유발)보다 우수한 다운스트림 성능 생성.[1]

### 4. 일반화 성능 향상 가능성

#### 4.1 포스트 트레이닝 단계별 다양성 유지

Tulu-3 모델 시리즈 (SFT → DPO → RLVR):

- **Direct 프롬프팅:** SFT 후 20.8% → DPO 후 10.8% (47% 감소)
- **VS-Standard:** 전 단계에서 ~30% 유지, DPO 후 182.6% 향상
- **기본 모델 다양성 복구:** 66.8% 복구 (Direct는 23.8%)[1]

VS는 정렬 훈련의 부작용에 탄력적임을 보여줍니다.

#### 4.2 온도와의 상호작용

온도 $$t \in \{0.4, 0.6, 0.8, 1.0, 1.2, 1.4\}$$:

- VS는 온도와 **직교적**(orthogonal): 두 가지 함께 사용 시 파레토 프론트 개선
- 온도 조정만으로는 제한적 다양성 향상
- VS + 온도 조정이 최적 트레이드오프 달성[1]

#### 4.3 복호화 전략과의 호환성

top-p, min-p 샘플링과 결합 가능:
- VS는 다양한 복호화 방법과 상충하지 않음
- 각 방법과의 조합이 기준선을 개선[1]

#### 4.4 모델 크기와의 상호작용 (창발 트렌드)

더 큰 모델이 VS에서 더 큰 이득:

| 모델 크기 | 다양성 향상 | 품질 변화 |
|---------|-----------|---------|
| **소형** (GPT-4.1-mini, Gemini-2.5-Flash) | +1.4-5.4 | -4.3-+1.1 |
| **대형** (GPT-4.1, Gemini-2.5-Pro) | +6.4-15.4 | -0.1-+5.0 |

**의미:** VS-CoT/Multi은 복잡한 프롬프트의 "인지 부담"을 더 큰 모델이 더 잘 처리함을 시사합니다.[1]

#### 4.5 다양성 튜닝

확률 임계값 조정으로 다양성 제어:

- 임계값 1.0 (표준 VS) → 0.001로 감소할 때 다양성 증가
- 기준선 방법들은 이 튜닝 옵션 없음[1]

### 5. 한계점

#### 5.1 방법론적 한계

1. **품질 비용 (작은 모델)**: VS-Standard와 Sequence는 작은 모델에서 품질 저하 초래. 이는 복잡한 프롬프트 이해의 어려움 반영.

2. **가독성 문제 (대화 시뮬레이션)**: VS는 여전히 인간보다 복잡한 응답 생성 (Flesch-Kincaid 점수 높음).

3. **계산 비용**: N개 응답 생성을 위해 최소 ⌈N/k⌉번의 LLM 호출 필요 (Direct는 N번).

#### 5.2 이론적 한계

1. **타이피컬리티 바이어스 모델**: 로그 우도를 타이피컬리티의 프록시로 사용하는 것이 모든 작업에 일반화되는가?

2. **분포 수준 프롬프트의 가정**: 모델이 정말로 올바른 확률을 생성하는가? (이 확률의 보정 문제는 다루지 않음)

3. **케이스 제한**: 분석이 S에서 $$r_{true}$$가 평탄한 경우만 다루지만, 실제는 부분적 평탄성.

#### 5.3 평가 한계

1. **합성 지표의 한계**: 의미론적 다양성(cosine 유사성)이 인간이 인식하는 다양성을 완벽히 포착하지 못함. 인간 평가는 제한적.

2. **일반화 범위**: 영어 중심, 폐쇄형 모델 대부분. 다국어 및 개방형 모델 평가 필요.

3. **안전성 평가**: 약간의 사실성 유지 테스트 있지만 (§G.7-G.8), 깊이 있는 안전성 분석 부족.

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 모드 붕괴 문제 (2024-2025)

**비전 모드 붕괴 (Shumailov et al., 2024)**[2]
- 합성 데이터의 재귀적 훈련이 분포의 꼬리 손실을 유발
- LLM, VAE, GMM 모두 영향받음
- **VS와의 차이:** 사전학습 기본 분포를 명시적으로 활용하여 완화

**개념적 다양성 감소 (Harvard Kempner Institute, 2025)**[3]
- RLHF/RLAIF가 개념 다양성 감소
- 온도와 맥락 조정만으로 부족함
- **VS와의 차이:** 분포 명시화가 더 근본적 해결

**다양성 붕괴 측정 (Shypula et al., 2025)**[4]
- 형식적 다양성과 의미론적 다양성의 트레이드오프 실증
- SFT, DPO, PPO, GRPO 효과 비교
- **VS와의 차이:** 훈련 단계에서 개입하지 않고 추론 시 해결

#### 6.2 정렬 방법의 다양성 관점 개선 (2024-2025)

**DPO (Direct Preference Optimization) 변형들:**

| 방법 | 핵심 기여 | 다양성 관련 |
|------|---------|----------|
| **β-DPO (2024)** | 동적 β 조정 | 최적화 수렴 개선 |
| **DPOP/DPO-Positive (2024)** | 선호 우도 유지 항 추가 | 과최적화 방지 |
| **DPH-RL (2025)** | Forward-KL/JS-divergence (reverse-KL 대신) | 다양성 보존 초점 |
| **GAPO (2025)** | 그룹 레벨 보상 | 유효한 완성의 균일 샘플링 |

DPH-RL의 접근과 VS의 유사점:
- 둘 다 **기본 정책 참조의 중요성 인식**
- DPH-RL은 훈련 시 포함, VS는 추론 시 활용[5]

#### 6.3 프롬프트 엔지니어링 다양성 전략 (2024-2025)

**NoveltyBench (2025)**: 다양성 평가 벤치마크[6]
- 20개 선도 LLM 평가 → 인간보다 훨씬 낮은 다양성
- 큰 모델 ≠ 높은 다양성 (오히려 역상관 보고)
- **VS의 관찰과 일치:** 더 큰 모델이 VS에서 더 큰 이득 → 기본 다양성 회복

**Template-Induced Diversity Collapse (2025)**[7]
- 구조화된 채팅 템플릿이 다양성 감소 유발
- 최소 구조의 프롬프트가 최고 다양성
- **VS와의 관계:** VS는 단순 프롬프트 개선이 아닌 분포 명시화

**DiVeRSe Prompting (2024)**[8]
- 다양한 프롬프트와 검증 단계 조합
- vs. VS의 단일 명확한 프롬프트 개선

#### 6.4 합성 데이터 품질 (2024-2025)

**합성 데이터 다양성의 영향 (2024)**[9]
- 다양성 부족 문제: 58%의 문제가 동일한 추론 전략 사용
- **RPD 메트릭** 제안: 다양하지만 유효한 솔루션 평가
- VS와의 상호작용: 다양한 합성 데이터가 다운스트림 성능 향상[1]

**모델 붕괴 방지 (2024)**[2]
- 합성 콘텐츠 중복 문제
- VS의 다양성 증대가 장기적 합성 데이터 품질 유지에 기여할 수 있음

#### 6.5 다양성-품질 트레이드오프 연구 (2024-2025)

**생성 공간 크기(GSS) 측정 (2025)**[10]
- 모델이 생성할 수 있는 의미론적으로 구분되는 출력 집합 정의
- 창의적 과제는 과다 균질화, 사실적 과제는 과다 할루시네이션
- **VS와의 연결:** 다양성 튜닝으로 GSS 조정 가능

**선호도 학습의 다양성 측면 (2025)**[11]
- RLHF/DPO가 다양성 감소 메커니즘 실증
- 다양한 인간 선호도 캡처 문제
- **VS의 공헌:** 훈련 재설계 없이 기본 모델의 다양성 회복

#### 6.6 모드 붕괴 메커니즘의 이해 (2024-2025)

**모드 붕괴의 특성화 (2024)**[12]
- 의미론적 네트워크와 다음 토큰 확률로 분석
- SFT만으로도 모드 붕괴 발생

**타이피컬리티의 새로운 이해 (2025: VS 논문)**[1]
- **처음으로 인지 심리학 기반의 데이터 수준 원인 식별**
- 타이피컬리티 바이어스를 정량화 (α = 0.57-0.65)
- 모든 주요 선호도 데이터셋에서 실증

이는 기존의 알고리즘 중심 설명보다 **더 근본적인 원인 제시**.

#### 6.7 일반화 성능과의 관계 (2024-2025)

**정렬과 일반화 (Kirk et al., 2024)**[3]
- RLHF가 어떻게 부정적 일반화 영향을 미치는지 실증
- OOD(분포 외) 성능 저하 관찰

**강화 학습에서 다양성 (2024)**[13]
- 다양성이 탐색 능력 증대 → 강화 학습 성능 향상 가능

**VS의 기여:**
- 추론 시 다양성 복구 → 강화 학습에서 더 나은 탐색 공간 제공 가능
- 합성 데이터 다양성 → 다운스트림 미세 조정 성능 향상 (37.5% 수학 정확도)[1]

#### 6.8 비교 요약 표

| 연구 | 문제 | 솔루션 | 적용 | 한계 |
|------|------|--------|------|------|
| **VS (2025)** | 타이피컬리티 바이어스 | 분포 명시화 | 추론 시 | 확률 보정, 계산 비용 |
| **DPH-RL (2025)** | KL 정규화 문제 | Forward-KL 사용 | 훈련 시 | 훈련 비용 |
| **GAPO (2025)** | 그룹 다양성 | 그룹 보상 | 훈련 시 | 복잡한 설정 |
| **Template Collapse** | 프롬프트 구조 | 최소화 프롬프트 | 추론 시 | 성능 가능성 |
| **NoveltyBench** | 평가 방법 부재 | 다양성 벤치마크 | 평가 | 인간-모델 간극 |
| **GSS (2025)** | 다양성 정량화 | 생성 공간 크기 | 분석 | 실시간 조정 불가 |

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 7.1 패러다임 변화: 데이터 중심 관점

기존 연구들이 알고리즘(RLHF, DPO, PPO)의 구조 개선에 초점을 맞춘 반면, VS는 **선호도 데이터 자체의 편향**을 근본 원인으로 제시합니다.

**미래 연구 방향:**
1. 선호도 데이터의 다양성 향상 (타이피컬리티 바이어스 감소)
2. 선호도 데이터 수집 시 의도적으로 다양한 고품질 응답 포함
3. 보상 모델의 타이피컬리티 편향 감지 및 보정[1]

#### 7.2 추론 시 개입의 새로운 가능성

VS는 훈련 없이 추론 시에만 개입하므로:
- 기존 정렬 모델 즉시 개선 가능
- 폐쇄형 모델(API 접근만)에도 적용 가능
- 계산 비용 없이 기존 모델의 내재적 다양성 회복[1]

**관련 연구:**
- Test-time scaling (Snell et al., 2024): VS는 반복 샘플링의 대안[14]
- 추론 시 정렬 (training-free alignment, 2025)[15]

#### 7.3 강화 학습에서의 응용

VS의 다양한 생성이 RL 탐색을 향상시킬 가능성:

$$ \pi_{RL}(\text{diverse action space from VS}) \rightarrow \text{better exploration} \rightarrow \text{improved rewards} $$

논문에서 언급: "RL 훈련에서 더 나은 탐색을 위한 가능성 있는 방향"[1]

#### 7.4 합성 데이터 생성의 질 향상

다양한 합성 데이터 → 더 강건한 모델:
- VS-Multi로 생성한 합성 데이터가 37.5% 수학 정확도
- 기존 방법의 32.8% 대비 14% 향상[1]

**장기 영향:** 모델 붕괴(recursive training on synthetic data) 완화[16]

#### 7.5 다양성-품질 트레이드오프의 해결 공식

VS-CoT가 다양성을 향상시키면서 품질도 유지/개선:
- 파레토 프론트 개선 (Figure 4d)
- 이전 대안들(높은 온도 = 품질 저하)과 달리[1]

**시사점:** 적절한 프롬프트 설계로 근본적 트레이드오프 완화 가능

#### 7.6 모델 규모와 다양성의 관계 재정의

관찰: 큰 모델이 VS에서 더 큰 이득 (1.5-2배)

**대안적 해석:**
- 큰 모델일수록 더 많은 다양한 분포를 학습
- RLHF가 더 심각하게 이를 억제
- VS가 이 억압된 다양성을 더 효과적으로 해제[1]

**부시사:** 모델 규모 증가가 항상 다양성 감소를 초래하지는 않음 (적절한 방법 사용 시)

### 8. 앞으로 연구 시 고려할 점

#### 8.1 이론적 개선

1. **타이피컬리티 바이어스 일반화**
   - 다국어, 다양한 도메인에서 α 값의 변동성 조사
   - 다양한 보상 함수 형태(Bradley-Terry 외)에서의 일반화 증명

2. **확률 보정 문제**
   - 모델이 생성한 확률이 실제 신뢰도를 반영하는가?
   - 온도 등의 스케일링 문제 해결

3. **분포 수렴성**
   - 어떤 조건 하에서 VS가 진정한 사전학습 분포에 수렴하는가?
   - k(후보 수)의 최적값 이론적 도출

#### 8.2 방법론적 개선

1. **작은 모델의 품질 문제**
   - VS-CoT/Multi의 추가 추론 이점을 작은 모델에 전이
   - 프롬프트 단순화 기법

2. **계산 효율성**
   - k 감소 시 성능 곡선 분석
   - 배치 처리를 통한 효율화
   - 근사 방법 개발

3. **다양성 튜닝의 자동화**
   - 작업별 최적 확률 임계값 자동 선택
   - 사용자 의도 반영

#### 8.3 평가 방법론 개선

1. **더 깊이 있는 인간 평가**
   - 다양성뿐만 아니라 관련성, 창의성, 참신성 분리 평가
   - 작업별 맞춤형 루브릭

2. **안전성 평가 강화**
   - 해로운 콘텐츠 생성 위험 평가
   - 사실성과 다양성의 상호작용 분석

3. **도메인 확장**
   - 기술 작성, 대화, 분류 등 다양한 작업
   - 특정 산업(의료, 법률) 안전성 평가

#### 8.4 응용 개발

1. **RL 통합**
   - VS와 강화 학습의 구체적 결합
   - 탐색-착취 균형에 미치는 영향 측정

2. **사용자 제어 인터페이스**
   - 확률 임계값 조정 UI
   - 실시간 다양성 피드백

3. **멀티모달 확장**
   - 이미지, 비디오 생성 모델에 VS 개념 적용
   - 비전-언어 모델에서의 효과

#### 8.5 윤리 및 사회 영향

1. **다양성 vs. 일관성**
   - 높은 다양성이 사용자에게 혼란을 야기할 수 있음
   - 안전성 관련 다양성 제한 필요

2. **편향과 다양성**
   - 타이피컬리티 바이어스 제거가 역편향 유발할 수 있음
   - 공정한 다양성 표현 정의

3. **변수 서빙(serving)**
   - 같은 모델이 맥락에 따라 다양한 vs. 일관된 응답 조정 필요

### 결론

"Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"는 **LLM 정렬의 숨겨진 원인을 밝혀내고 이를 우아하고 실행 가능한 해결책으로 제시**한 중요한 논문입니다. 

**핵심 공헌:**
1. 인지 심리학 원리를 활용한 **타이피컬리티 바이어스의 발견과 정량화**
2. 이를 **공식적으로 모델링**하여 RLHF 최적화 공식 유도
3. 훈련 없이도 **추론 시에 적용 가능한 실용적 해결책** 제시
4. **포괄적 실험**을 통해 창의성부터 합성 데이터 생성까지 광범위한 이득 입증

2020년 이후의 다양성 관련 연구들이 주로 **알고리즘 최적화**(DPO 개선, RL 변형)에 초점을 맞춘 반면, 이 논문은 **데이터 수준의 근본 문제**를 제시함으로써 연구 패러다임을 전환합니다. 특히 GAPO, DPH-RL 등 최신 훈련 방법들과 상호보완적이며, template collapse나 GSS와 같은 다양성 측정 연구들과 통합될 수 있습니다.

앞으로의 연구는 이 발견을 바탕으로 선호도 데이터 수집 개선, 추론 시 다양성 제어의 자동화, RL과의 통합 등으로 발전할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c7569d52-58b3-475a-90c6-a2dced63693e/2510.01171v3.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11269175/)
[3](https://kempnerinstitute.harvard.edu/research/deeper-learning/alignment-reduces-conceptual-diversity-of-language-models/)
[4](https://openreview.net/pdf?id=40uDwtrbd3)
[5](https://arxiv.org/abs/2509.07430)
[6](https://arxiv.org/pdf/2504.05228.pdf)
[7](https://arxiv.org/html/2505.18949v1)
[8](https://latitude-blog.ghost.io/blog/how-to-reduce-bias-in-ai-with-prompt-engineering/)
[9](https://arxiv.org/html/2510.26122v2)
[10](https://arxiv.org/abs/2510.12699)
[11](https://arxiv.org/html/2511.08594v1)
[12](http://arxiv.org/pdf/2410.12341.pdf)
[13](https://aclanthology.org/2025.emnlp-main.1649)
[14](https://arxiv.org/html/2510.01171v1)
[15](https://aclanthology.org/2025.findings-emnlp.238.pdf)
[16](https://arxiv.org/abs/2511.05535)
[17](https://arxiv.org/abs/2510.01171)
[18](https://dl.acm.org/doi/10.1145/3746252.3760863)
[19](https://www.semanticscholar.org/paper/80d86aea613a91272a27c81c2d6a43763da0bece)
[20](https://link.aps.org/doi/10.1103/PhysRevD.111.103029)
[21](https://biss.pensoft.net/article/182910/)
[22](https://link.springer.com/10.1007/s00146-024-02173-x)
[23](https://arxiv.org/html/2502.11266v1)
[24](http://arxiv.org/pdf/2412.10271.pdf)
[25](https://arxiv.org/pdf/2305.17493.pdf)
[26](http://arxiv.org/pdf/2406.15951.pdf)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446866/)
[28](https://arxiv.org/html/2510.01171v2)
[29](https://aclanthology.org/2025.findings-emnlp.836.pdf)
[30](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa69e968b7319fd42524febd41475fb3-Paper-Conference.pdf)
[31](https://aimatters.co.kr/news-report/ai-report/32921/)
[32](https://www.veratai.co.uk/blog/alignment-post-training-for-llms-lessons-learned-making-it-work)
[33](https://alignmentsurvey.com/uploads/pair_lab/rlhf/RLHF_slide_6.pdf)
[34](https://blog.naver.com/rainbow-brain/224048426217)
[35](https://arxiv.org/html/2509.04784v2)
[36](https://datasciocean.com/en/paper-intro/verbalized-sampling/)
[37](https://pytorch.org/blog/a-primer-on-llm-post-training/)
[38](https://openreview.net/forum?id=PXD3FAVHJT)
[39](https://arxiv.org/pdf/2509.04419.pdf)
[40](https://arxiv.org/pdf/2502.18770.pdf)
[41](https://arxiv.org/pdf/2502.21321.pdf)
[42](https://arxiv.org/html/2512.24146v1)
[43](https://arxiv.org/html/2511.15573v1)
[44](https://arxiv.org/html/2511.00379v1)
[45](https://arxiv.org/html/2502.18770v3)
[46](https://arxiv.org/html/2411.04427v3)
[47](https://arxiv.org/html/2502.18770v1)
[48](https://arxiv.org/html/2504.05228v2)
[49](https://arxiv.org/html/2505.18126v1)
[50](https://dl.acm.org/doi/10.1145/3746027.3755499)
[51](https://www.semanticscholar.org/paper/c074789156dd9f29aa09e7055ccdb26bfdf856c0)
[52](https://arxiv.org/abs/2407.08639)
[53](https://www.semanticscholar.org/paper/0d1c76d45afa012ded7ab741194baf142117c495)
[54](https://link.springer.com/10.1007/978-3-032-04971-1_52)
[55](https://arxiv.org/abs/2501.16629)
[56](https://arxiv.org/abs/2503.11701)
[57](https://arxiv.org/abs/2505.20065)
[58](https://arxiv.org/abs/2502.13146)
[59](https://arxiv.org/abs/2503.08619)
[60](https://arxiv.org/html/2402.13228v2)
[61](https://arxiv.org/pdf/2410.20187.pdf)
[62](https://arxiv.org/pdf/2412.20299.pdf)
[63](http://arxiv.org/pdf/2405.17956.pdf)
[64](http://arxiv.org/pdf/2406.07327.pdf)
[65](https://arxiv.org/pdf/2410.08458v1.pdf)
[66](http://arxiv.org/pdf/2503.01233.pdf)
[67](https://arxiv.org/pdf/2312.16430.pdf)
[68](https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf)
[69](https://openreview.net/pdf?id=fwCoLe3TAX)
[70](https://aiflower.tistory.com/150)
[71](https://www.labellerr.com/blog/tuning-the-power-strategies-for-enhancing-language-models-through-instruction-tuning/)
[72](https://relevanceai.com/prompt-engineering/use-diverse-prompting-to-improve-ai-responses)
[73](https://arxiv.org/pdf/2305.18290.pdf)
[74](https://proceedings.iclr.cc/paper_files/paper/2024/file/c2d82a425af4c18a35049899fea5ee82-Paper-Conference.pdf)
[75](https://infomineo.com/artificial-intelligence/prompt-engineering-techniques-examples-best-practices-guide/)
[76](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffusion-dpo/)
[77](https://jingfengyang.github.io/alignment)
[78](https://arxiv.org/abs/2402.01727)
[79](https://blog.outta.ai/307)
[80](https://arxiv.org/html/2406.15178v1)
[81](https://www.promptingguide.ai/applications/generating_textbooks)
[82](https://arxiv.org/pdf/2402.13228.pdf)
[83](https://arxiv.org/html/2505.02666v2)
[84](https://arxiv.org/html/2507.18638v2)
[85](https://arxiv.org/html/2511.07419v1)
[86](https://arxiv.org/html/2402.07927v1)
[87](https://arxiv.org/html/2404.05868v1)
[88](https://arxiv.org/html/2406.19032v1)
[89](https://arxiv.org/html/2506.05614v1)
[90](https://arxiv.org/html/2411.04712v2)
[91](https://arxiv.org/html/2505.18644v1)
[92](https://arxiv.org/html/2502.07599v1)
[93](https://arxiv.org/html/2510.03519v1)
[94](https://arxiv.org/html/2506.18199v1)
[95](https://www.alignmentforum.org/posts/YhQr36yGkhe6x8Fyn/learning-the-prior-and-generalization)
[96](https://openai.com/index/deliberative-alignment/)
