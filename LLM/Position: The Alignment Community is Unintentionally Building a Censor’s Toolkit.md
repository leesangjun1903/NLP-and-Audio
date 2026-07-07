# Position: The Alignment Community is Unintentionally Building a Censor's Toolkit

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Sarah Ball & Phil Hackemann (LMU Munich / MCML, ICML 2026)의 이 포지션 페이퍼는 다음과 같은 **핵심 명제**를 제시합니다:

> **"AI 정렬(Alignment) 기법은 목적 중립적(purpose-agnostic) 도구이며, 선의로 개발된 정렬 방법이 악의적 행위자에 의해 검열(censorship)과 여론 조작(manipulation)의 도구로 전용될 수 있다."**

즉, 정렬 연구는 필요하지만, 그것이 동시에 **이중 용도(dual-use) 기술**임을 인식하지 못한 채 개선에만 집중하는 커뮤니티의 태도를 비판합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 체계적 매핑 | 정렬 기술 3계층(사전학습 필터링, 사후훈련 정렬, 추론시간 제어)을 이중 용도 위험성과 매핑 |
| 실증적 증거 | DeepSeek, Ernie Bot, Grok 등 실제 무기화 사례 제시 |
| 생태계 분석 | 사회·경제·정치적 맥락에서 오남용 가능성 분석 |
| 완화 전략 제안 | 검증 가능한 정렬(verifiable alignment), 모델 다원주의, 감사 메커니즘 등 제안 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

이 논문은 기술 논문이 아닌 **포지션 페이퍼(Position Paper)**이므로 새로운 알고리즘이나 모델을 제안하지 않습니다. 대신 다음 문제를 제기합니다:

$$\text{문제} = \underbrace{\text{정렬 기술의 목적 중립성}}_{\text{기술적 요인}} \times \underbrace{\text{권위주의 확산}}_{\text{정치적 요인}} \times \underbrace{\text{LLM 과점 구조}}_{\text{경제적 요인}}$$

핵심은 정렬 기술이 "누가 그 가치를 정의하느냐"에 따라 완전히 다른 결과를 낳는다는 것입니다. Ji et al.(2025)의 정의를 인용하면:

> *"AI alignment aims to make AI systems behave in line with **human intentions and values**."*

여기서 **'whose intentions and values?'** 가 핵심 문제입니다.

---

### 2.2 정렬 기술의 3계층 구조 및 이중 용도 분석

논문은 현대 LLM의 제어 스택을 세 계층으로 분류합니다:

#### 계층 1: 사전학습 데이터 필터링 (Pre-Training Data Filtering)

사전학습 데이터 필터링은 다음과 같은 파이프라인으로 구성됩니다:

$$\mathcal{D}_{\text{filtered}} = \mathcal{D}_{\text{raw}} \setminus \{x \in \mathcal{D}_{\text{raw}} : f_{\text{heuristic}}(x) = 1 \lor f_{\text{classifier}}(x) \geq \tau\}$$

여기서:
- $\mathcal{D}_{\text{raw}}$: 원본 학습 데이터
- $f_{\text{heuristic}}$: 키워드 기반 휴리스틱 필터 (예: "dirty word" counting, Raffel et al., 2020)
- $f_{\text{classifier}}$: 모델 기반 분류기 (품질 및 안전성 평가)
- $\tau$: 분류 임계값

**이중 용도 위험**: $f_{\text{heuristic}}$의 키워드 목록이나 $f_{\text{classifier}}$의 학습 데이터를 교체하면 정치적 정보를 선택적으로 제거 가능. 중국 당국의 사례: "socialist core values" 위반 키워드 필터링 (McMorrow & Hu, 2024).

**특징**:

| 특성 | 수준 |
|------|------|
| 접근 요구사항 | 전체 사전학습 파이프라인 |
| 컴퓨팅 자원 | 매우 높음 |
| 기술 전문성 | 높음 |
| 수정 용이성 | 보통~어려움 |
| 변경 심도 | **근본적(fundamental)** |

---

#### 계층 2: 사후훈련 선호도 정렬 (Post-Training Preference Alignment)

**RLHF (Reinforcement Learning from Human Feedback)** 프레임워크:

**Step 1 - 보상 모델 학습:**

$$r_\phi(x, y) = \arg\max_\phi \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

여기서:
- $x$: 입력 프롬프트
- $y_w$: 선호되는 응답 (winner)
- $y_l$: 비선호 응답 (loser)
- $r_\phi$: 보상 모델 파라미터

**Step 2 - PPO를 이용한 정책 최적화 (Schulman et al., 2017):**

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

여기서:
- $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$: 확률 비율
- $\hat{A}_t$: 어드밴티지 추정값
- $\epsilon$: 클리핑 파라미터

**KL 정규화를 포함한 전체 목적함수:**

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y)\right] - \beta \cdot D_{\text{KL}}\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]$$

여기서 $\beta$는 참조 정책 $\pi_{\text{ref}}$로부터의 편차를 제어하는 하이퍼파라미터입니다.

**이중 용도 위험**: 선호 데이터 $\mathcal{D}$를 특정 이데올로기에 부합하게 큐레이션하거나, 주석자 풀(annotator pool)을 편향적으로 선발하면 보상 모델 $r_\phi$가 검열 목적으로 훈련됩니다.

**가이드라인 기반 정렬의 사례:**
- **Constitutional AI** (Bai et al., 2022b): 원칙(constitution) 집합이 $r_\phi$ 대체
- **Deliberative Alignment** (Guan et al., 2024): 정책 집합 $\mathcal{P}$에 대해 추론 후 응답

$$y^* = \arg\max_{y} P(y|x, \mathcal{P})$$

**특징**:

| 특성 | 수준 |
|------|------|
| 접근 요구사항 | 모델 가중치 |
| 컴퓨팅 자원 | 보통~높음 |
| 기술 전문성 | 보통~높음 |
| 수정 용이성 | 보통 |
| 변경 심도 | **지속적(persistent)** |

---

#### 계층 3: 추론 시간 제어 (Inference-Time Control)

**시스템 프롬프트 기반 제어:**

$$y = \text{LLM}(\underbrace{s}_{\text{system prompt}} \oplus \underbrace{u}_{\text{user input}})$$

시스템 프롬프트 $s$는 모델의 역할, 제약 조건, 우선순위를 정의합니다. 이는 파라미터 수정 없이 즉각 적용 가능합니다.

**안전 분류기 기반 필터링:**

$$\hat{y} = \begin{cases} y & \text{if } g_\psi(x, y) < \tau_{\text{safe}} \\ \text{[REFUSED]} & \text{if } g_\psi(x, y) \geq \tau_{\text{safe}} \end{cases}$$

여기서 $g_\psi$는 안전 분류기(예: Llama Guard, Inan et al., 2023), $\tau_{\text{safe}}$는 거부 임계값입니다.

**특징**:

| 특성 | 수준 |
|------|------|
| 접근 요구사항 | 런타임 접근 |
| 컴퓨팅 자원 | 무시할 수준~보통 |
| 기술 전문성 | 낮음~보통 |
| 수정 용이성 | **쉬움** |
| 변경 심도 | 표면적(superficial) |

---

### 2.3 세 계층 비교 요약

$$\underbrace{\text{사전학습 필터링}}_{\substack{\text{높은 비용} \\ \text{근본적 변경}}} \xrightarrow{\text{접근성 증가}} \underbrace{\text{사후훈련 정렬}}_{\substack{\text{중간 비용} \\ \text{지속적 변경}}} \xrightarrow{\text{접근성 증가}} \underbrace{\text{추론시간 제어}}_{\substack{\text{낮은 비용} \\ \text{즉각 적용}}}$$

---

### 2.4 성능 향상 및 한계

이 논문은 새 모델의 성능 향상을 측정하지 않으며, 대신 **정렬 기술의 이중 용도 위험성**을 다음 측면에서 분석합니다:

**주요 실증 사례:**
- **DeepSeek / Ernie Bot**: 천안문 사태, 대만 문제 등 정치 민감 주제 거부 (Huang et al., 2025; McDonell, 2023; Naseh et al., 2025)
- **Grok (xAI)**: 시스템 프롬프트 변경으로 정치적 편향 주입 → 반유대주의 발언, 홀로코스트 부정 (Conger, 2025)
- **Yi-large**: 생성 후 키워드/모델 필터로 시진핑 비판 응답을 거부 응답으로 교체 (McMorrow & Hu, 2024)
- **서구 LLM의 중국어 자기검열**: 훈련 코퍼스가 중국 정부 영향 하에 있어 Simplified Chinese 프롬프트 시 자기검열 발생 (Ahmed & Knockel, 2024; Waight et al., 2026)

**한계:**

1. **실증적 측정의 부재**: 이중 용도 위험의 정량적 측정 척도를 제시하지 않음
2. **표준화된 벤치마크 부재**: 기존 검열 벤치마크가 협소한 맥락(예: 중국 검열)에 국한됨
3. **정상적 정렬과 오남용의 경계 불명확**: 문화적 맥락에 따라 검열과 적법한 콘텐츠 모더레이션의 구분이 주관적
4. **파인튜너의 공격에 취약한 사후훈련**: 포스트-트레이닝 정렬은 파인튜닝 공격으로 우회 가능 (Qi et al., 2024)

---

## 3. 모델의 일반화 성능 향상 가능성과의 관련성

이 논문은 정렬 기술이 **일반화 성능(generalization)** 에 미치는 영향을 두 가지 방향에서 중요하게 다룹니다.

### 3.1 정렬이 일반화를 저해하는 경로

Kirk et al. (2024b)의 연구 **"Understanding the effects of RLHF on LLM Generalisation and Diversity"** 를 인용하며, RLHF가 모델의 다양성과 일반화를 저해할 수 있음을 지적합니다:

$$\text{RLHF 목적함수} = \underbrace{\mathbb{E}[r_\phi(x,y)]}_{\text{선호도 최적화}} - \underbrace{\beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]}_{\text{일반화 정규화}}$$

- $\beta$가 너무 작으면 보상 해킹(reward hacking) → 특정 표현 패턴 과적합
- $\beta$가 너무 크면 참조 모델 종속 → 창의성과 다양성 감소

**검열 관점의 일반화 문제:**
- 특정 토픽을 배제한 훈련 데이터로 학습된 모델은 해당 도메인에서 **체계적으로 실패(systematic failure)**
- Wu et al. (2025)의 "Generative Monoculture" 연구: RLHF 기반 정렬이 출력의 단조로움(monoculture)을 야기 → 다양한 인구 집단에 대한 일반화 성능 저하

### 3.2 다원적 정렬(Pluralistic Alignment)과 일반화

논문이 인용하는 다원적 정렬 연구들은 일반화 성능 향상을 직접 목표로 합니다:

**MaxMin-RLHF (Chakraborty et al., 2024):**

$$\max_{\pi_\theta} \min_{g \in \mathcal{G}} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}[r_g(x, y)]$$

여기서 $\mathcal{G}$는 다양한 인구 집단의 선호도 집합. 최소 집단의 효용을 최대화함으로써 편향 없는 일반화를 추구합니다.

**SharedRep-RLHF (Mukherjee et al., 2025):** 다양한 선호도 간 공유 표현을 학습하여 일반화 향상:

$$r_\phi(x, y) = f_{\text{shared}}(x, y) + f_{\text{group}}(x, y, g)$$

**이중 용도 관점에서의 일반화 함의:**

| 상황 | 일반화에 미치는 영향 |
|------|---------------------|
| 정상적 다원적 정렬 | 다양한 집단에 대한 일반화 성능 향상 |
| 단일 행위자 정렬(오남용) | 특정 이념에 과적합 → 일반화 성능 저하 |
| 검열 목적 데이터 필터링 | 필터링된 도메인에서 완전한 지식 부재 |

### 3.3 언어 간 일반화 문제

서구 LLM이 Simplified Chinese로 프롬프트될 때 자기검열이 발생하는 현상(Ahmed & Knockel, 2024; Waight et al., 2026)은 **교차 언어 일반화(cross-lingual generalization)**의 심각한 실패를 보여줍니다:

$$P(y_{\text{censored}} | x_{\text{CN}}) > P(y_{\text{censored}} | x_{\text{EN}})$$

이는 훈련 코퍼스의 언어별 편향이 모델의 언어 간 일반화에 직접적으로 영향을 미침을 보여주며, 단순히 검열 문제를 넘어 **다국어 NLP의 공정성과 일반화** 문제임을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 미래 연구에 미치는 영향

#### 4.1.1 검증 가능한 정렬(Verifiable Alignment) 연구

논문은 다음을 촉구합니다:

$$\text{Verifiable Alignment} = \text{투명한 정렬 정책} + \text{표준화 벤치마크} + \text{독립 감사}$$

이는 다음 연구 방향을 열어줍니다:
- **정렬 감사 프레임워크** 개발 (블랙박스 모델에 대한 외부 평가)
- **정보 억압 벤치마크** 구축 (다양한 언어, 문화, 정치 맥락을 포함)
- **정치적 편향 측정 지표** 표준화 (Bang et al., 2024; Rettenberger et al., 2025 등의 한계 극복)

#### 4.1.2 다원적 정렬(Pluralistic Alignment) 연구의 가속화

Sorensen et al. (2024)의 로드맵을 기반으로:
- 단일 행위자 정의 가치가 아닌 **다원적 가치 집합**을 학습하는 알고리즘
- 문화적 맥락에 민감한 정렬 데이터 수집 방법론
- PRISM 데이터셋 (Kirk et al., 2024a)과 같은 참여적·대표적 피드백 데이터 확장

#### 4.1.3 정렬 기술의 적대적 강인성 연구

정렬 기술이 오남용될 때 이를 탐지하거나 우회하는 연구:
- **정렬 우회 탐지**: 악의적 정렬 변경을 감지하는 메커니즘 (Lee et al., 2024)
- **역방향 정렬 공격**: 사후훈련 정렬을 파인튜닝으로 제거하는 공격 분석 (Qi et al., 2024)
- **Jailbreak의 이중적 역할**: 검열 우회 수단으로서의 긍정적 기능 연구

#### 4.1.4 정렬 윤리의 제도화

ICML 등 주요 학회의 윤리 성명(ethics statement) 문화를 실질적으로 강화하는 방향:
- 정렬 연구의 이중 용도 위험 평가를 의무화
- AI 감사(auditing) 방법론의 표준화 (EU AI Act의 투명성 요건 확장)

---

### 4.2 미래 연구 시 고려 사항

#### 고려 사항 1: 이중 용도 위험의 사전 평가 의무화

모든 정렬 연구에서 다음 질문을 명시적으로 다루어야 합니다:

> *"이 기술이 악의적 행위자에게 어떻게 전용될 수 있는가?"*

특히 다음 기술 영역에서 주의가 필요합니다:
- 더 정밀한 콘텐츠 필터링 분류기 개발
- 효율적인 파인튜닝 기법 (LoRA 등) - 낮은 비용으로 정렬 변경 가능
- 가이드라인 기반 정렬 (Constitutional AI, Deliberative Alignment) - 손쉬운 지침 교체

#### 고려 사항 2: 모델 다원주의 생태계 설계

$$\text{정보 건전성} \propto \text{모델 다양성} \times \text{접근 가능성}$$

- 독점 방지를 위한 오픈소스 모델 생태계 유지
- "Neutrality Through Diversity" (Fisher et al., 2025) 원칙 적용
- 특정 국가나 기업에 의한 일방적 정렬 표준화 저지

#### 고려 사항 3: 표준화된 글로벌 검열 벤치마크 필요

기존 벤치마크의 한계:

| 기존 벤치마크 | 한계 |
|--------------|------|
| Ahmed & Knockel (2024) | 중국 검열에만 집중 |
| Noels et al. (2025) | 역사적 인물 정보 억압에 국한 |
| Rozado (2023), Bang et al. (2024) | 좌-우 정치 스펙트럼에만 집중, 특정 국가 맥락 |

**필요한 벤치마크 특성:**
- 전 세계 정치 맥락 포함
- 좌-우 외에 권위주의적 성향 측정
- 동적으로 업데이트되어 변화하는 현실 반영

#### 고려 사항 4: 법·정책 연구와의 학제간 협력

- EU AI Act, DSA의 정렬 투명성 조항 강화 방향 연구
- 국제적 집행력이 부재한 상황에서의 대안 메커니즘
- 민주주의 국가에서도 발생하는 정렬 오남용(트럼프 행정부, Musk/Grok 사례) 대응

#### 고려 사항 5: 연구 문화 개혁

Do et al. (2023)과 Ashurst et al. (2022)이 지적한 바와 같이:
- 단순한 형식적 윤리 성명을 넘어 진정한 비판적 성찰 필요
- 빠른 발전을 우선시하는 학계 문화의 구조적 개혁
- 이중 용도 위험 논의를 포함하지 않은 논문에 대한 리뷰어 피드백 강화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 정렬 기법의 발전과 이중 용도 위험 확대

| 연구 | 핵심 기여 | 이중 용도 관련성 |
|------|----------|-----------------|
| Ouyang et al. (2022) - InstructGPT | RLHF로 지시 따르기 학습 | RLHF 선호 데이터 조작으로 편향 주입 가능 |
| Bai et al. (2022b) - Constitutional AI | 원칙 기반 자기비판 정렬 | 원칙 교체만으로 정렬 방향 전환 용이 |
| Guan et al. (2024) - Deliberative Alignment | 추론 기반 안전 정렬 | 정책 집합 교체로 검열 구현 가능 |
| Shao et al. (2024) - GRPO | 그룹 상대 정책 최적화 | DeepSeek 등 중국 모델에 적용, 검열과 결합 |
| Qi et al. (2024) - Fine-tuning Attack | 파인튜닝으로 정렬 우회 | 역으로 악의적 정렬을 파인튜닝으로 강화 가능 |
| Chakraborty et al. (2024) - MaxMin-RLHF | 다양한 선호도 포용 | 이중 용도 완화 가능성 있는 다원적 접근 |

### 5.2 검열 탐지 및 평가 연구

| 연구 | 기여 | 한계 |
|------|------|------|
| Noels et al. (2025) - ECML | LLM의 검열 관행 실증 연구 | 특정 역사적 인물에 국한 |
| Naseh et al. (2025) - R1dacted | DeepSeek R1의 로컬 검열 분석 | 단일 모델, 단일 지역 |
| Qiu et al. (2026) - Information Sciences | DeepSeek 검열 정량화 | 중국 모델에만 집중 |
| Ahmed & Knockel (2024) - FOCI | 온라인 검열의 LLM 영향 | 중국 검열에 국한 |
| Huang et al. (2025) | DeepSeek vs GPT 편향 비교 | 미-중 이분법적 시각 |

### 5.3 다원적 정렬(Pluralistic Alignment) 연구 흐름

$$\text{단일 보상 모델} \xrightarrow{\text{한계 인식}} \text{다원적 정렬} \xrightarrow{\text{이중 용도 관점}} \text{검증 가능한 정렬}$$

| 연구 | 기여 |
|------|------|
| Santurkar et al. (2023) - ICML | LLM이 반영하는 의견의 대표성 분석 |
| Kirk et al. (2024a) - PRISM NeurIPS | 참여적·대표적 피드백 데이터셋 |
| Sorensen et al. (2024) - ICML | 다원적 정렬 로드맵 |
| Ryan et al. (2024) | 글로벌 대표성에 대한 정렬의 의도치 않은 영향 |
| Wu et al. (2025) - ICLR | 생성적 단조로움(generative monoculture) 문제 |
| Fisher et al. (2025) - ICML Position | AI의 정치적 중립성 불가능성, 다양성을 통한 중립 |

### 5.4 본 논문의 차별점

| 기준 | 기존 연구 | 본 논문 |
|------|----------|---------|
| 분석 대상 | 일반 AI 시스템 위험 | **기술적 정렬 방법 자체** |
| 관점 | 가상적(hypothetical) | 실증적(documented) |
| 범위 | 개별 모델/지역 | **체계적 매핑, 전 지구적 맥락** |
| 제안 | 규제 일반론 | 구체적 완화 전략 4가지 |
| 프레이밍 | 사고 실험 | **현재 진행 중인 위협** |

---

## 참고자료

**[주 논문]**
- Ball, S. & Hackemann, P. (2026). *Position: The Alignment Community is Unintentionally Building a Censor's Toolkit*. ICML 2026 Position Paper Track.

**[논문 내 주요 인용 문헌]**
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*, 35.
- Bai, Y. et al. (2022a). Training a helpful and harmless assistant with RLHF. *arXiv:2204.05862*.
- Bai, Y. et al. (2022b). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Christiano, P. et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS*, 30.
- Guan, M.Y. et al. (2024). Deliberative alignment: Reasoning enables safer language models. *arXiv:2412.16339*.
- Kirk, H.R. et al. (2024a). The PRISM alignment dataset. *NeurIPS*, 37.
- Kirk, R. et al. (2024b). Understanding the effects of RLHF on LLM generalisation and diversity. *ICLR 2024*.
- Chakraborty, S. et al. (2024). MaxMin-RLHF. *ICML 2024*.
- Qi, X. et al. (2024). Fine-tuning aligned language models compromises safety. *ICLR 2024*.
- Sorensen, T. et al. (2024). A roadmap to pluralistic alignment. *ICML 2024*.
- Wu, F., Black, E., & Chandrasekaran, V. (2025). Generative monoculture in large language models. *ICLR 2025*.
- Noels, S. et al. (2025). What large language models do not talk about. *ECML-PKDD 2025*.
- Naseh, A. et al. (2025). R1dacted: Investigating local censorship in DeepSeek's R1. *arXiv:2505.12625*.
- Qiu, P., Zhou, S., & Ferrara, E. (2026). Information suppression in LLMs. *Information Sciences*, 724.
- Ahmed, M. & Knockel, J. (2024). The impact of online censorship on LLMs. *FOCI*.
- Huang, P. et al. (2025). Analysis of LLM bias in DeepSeek-R1 vs. ChatGPT. *arXiv:2506.01814*.
- Fisher, J. et al. (2025). Political neutrality in AI is impossible. *ICML 2025 Position Paper*.
- Santurkar, S. et al. (2023). Whose opinions do language models reflect? *ICML 2023*.
- Ryan, M.J., Held, W., & Yang, D. (2024). Unintended impacts of LLM alignment. *arXiv:2402.15018*.
- Shao, Z. et al. (2024). DeepSeekMath: GRPO. *arXiv:2402.03300*.
- Inan, H. et al. (2023). Llama Guard. *arXiv:2312.06674*.
- Mukherjee, A. et al. (2025). SharedRep-RLHF. *arXiv:2509.03672*.
- Waight, H. et al. (2026). State media control influences large language models. *Nature*.
- Vesteinsson, K. et al. (2025). *Freedom on the Net 2025*. Freedom House.
- Nord, M. et al. (2025). State of the world 2024. *Democratization*, 32(4).
- Ji, J. et al. (2025). AI alignment: A comprehensive survey. *arXiv:2310.19852*.
- Raffel, C. et al. (2020). T5: Exploring limits of transfer learning. *JMLR*, 21(140).
- Grattafiori, A. et al. (2024). The Llama 3 herd of models. *arXiv:2407.21783*.

> **⚠️ 정확도 주의사항**: 이 논문은 ICML 2026 게재 예정 포지션 페이퍼로, 제공된 PDF 원문에 근거하여 분석하였습니다. 수식의 경우 논문이 명시적 수식을 제공하지 않으므로, 논문이 인용하는 원본 방법론(RLHF, PPO, Constitutional AI 등)의 표준 수식을 재구성하여 제시하였습니다. 모델 성능 수치는 논문이 포지션 페이퍼 특성상 제공하지 않으므로 포함하지 않았습니다.
