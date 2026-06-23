# Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**Prompt-Level Distillation (PLD)**는 대형 Teacher 모델의 추론 능력을 소형 Student 모델의 **시스템 프롬프트(System Prompt)**에 직접 증류(distill)함으로써, 파라미터 업데이트 없이 Chain-of-Thought(CoT) 수준의 추론 성능을 Zero-shot 추론 속도로 달성하는 **비파라메트릭(Non-Parametric) 프레임워크**입니다.

> **핵심 명제:** 지식은 모델의 가중치(weights)가 아닌 **컨텍스트 창(context window)**에 증류될 수 있다.

### 주요 기여 (3가지)

| 기여 영역 | 내용 |
|---|---|
| **프레임워크** | 파인튜닝 없이 Teacher 로직을 System Prompt로 이전하는 PLD 제안 |
| **방법론** | 지도 명령 추출 → 클러스터링 → 충돌 해결의 모듈형 파이프라인 |
| **성능** | Gemma-3 4B, Mistral Small 3.1이 프론티어 모델 수준 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문은 세 가지 핵심 문제를 동시에 해결하려 합니다:

```
[문제 삼각형]
CoT 추론의 높은 지연(latency)
        ↕
파인튜닝의 높은 유지보수 비용
        ↕
소형 모델의 낮은 추론 능력
```

**기존 방법의 한계:**

1. **CoT 프롬프팅**: 수백 토큰의 추론 트레이스 생성 → 추론 지연 선형 증가
2. **지식 증류(KD) / 파인튜닝**:
   - 파라미터 업데이트 필요 → 유지보수 부채(maintenance debt) 발생
   - 파인튜닝은 스타일 모방에는 성공하나 **추론 능력 이전에는 실패** (Gudibande et al., 2023)
   - Teacher 모델 업그레이드 시마다 Student 재학습 필요
   - Catastrophic forgetting 위험 (Luo et al., 2025)
3. **RAG (Retrieval-Augmented Generation)**: 복잡한 운영 패러다임 도입 (Shah et al., 2026a)

---

### 2.2 제안 방법 (수식 포함)

PLD는 **4단계 파이프라인**으로 구성됩니다:

```
Phase 1          Phase 2          Phase 3          Phase 4
[명령 추출] → [클러스터링] → [충돌 해결] → [추론(Inference)]
```

#### Phase 1: 지도 명령 추출 (Supervised Instruction Extraction)

레이블된 훈련 데이터셋 $T = \{(x_i, y_i)\}$가 주어질 때, Teacher 모델은 각 예제에 대해 두 가지 작업을 동시 수행합니다:

1. **지도 문제 풀기**: 입력 $x_i$의 논리적 제약을 CoT로 분석하여 Ground-truth 레이블 $y_i$를 정당화
2. **명령 추상화**: 추론 과정을 특정 엔티티 이름 없이 인과 메커니즘을 보존한 일반화된 자연어 명령 $I_i$로 추상화

이 과정의 결과물은 확장된 데이터셋:

$$D = \{(x_i, y_i, I_i)\}$$

여기서 $I_i$는 훈련 예제 $i$로부터 파생된 추상 명령(abstract instruction)입니다.

#### Phase 2: 클러스터링 기반 논리 합성 (Clustering Logic Synthesis)

각 마이크로 명령 $I_i$를 고밀도 벡터로 표현:

$$\mathbf{v}_i = \text{Embed}(I_i) \in \mathbb{R}^{768}$$

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)을 코사인 거리 메트릭으로 적용:

$$d_{\cos}(\mathbf{v}_i, \mathbf{v}_j) = 1 - \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$

DBSCAN 파라미터:
- 이웃 반경: $\epsilon = 0.4$
- 최소 샘플 수: $\text{min samples} = 6$

이 설정은 Contract-NLI에서 **17개 클러스터**, StereoSet에서 **4개 클러스터**를 생성합니다.

**DBSCAN의 클러스터 $C_k$ 정의:**

$$C_k = \{I_i : \exists I_j \in C_k, d_{\cos}(\mathbf{v}_i, \mathbf{v}_j) \leq \epsilon\}$$

**K-Means 대비 DBSCAN의 장점:**
- 사전에 클러스터 수 $K$를 지정할 필요 없음
- 노이즈 포인트(비일반화 명령)를 자동으로 제거(outlier 처리)

각 클러스터 내 마이크로 룰들은 Teacher 모델(Gemini 3 Pro)에 의해 단일 통합 명령으로 합성됩니다.

#### Phase 3: 폐루프 충돌 해결 (Closed-Loop Conflict Resolution)

충돌 해결 루프는 다음을 반복합니다:

**Step 1 - 추론 및 오류 분석:**

$$\mathcal{E} = \{(x_i, y_i) : \hat{y}_i \neq y_i, \hat{y}_i = \text{Student}(x_i \mid \mathcal{P}^{(t)})\}$$

여기서 $\mathcal{P}^{(t)}$는 $t$번째 반복의 현재 시스템 프롬프트입니다.

**Step 2 - 적대적 정제(Adversarial Refinement):**

실패 샘플 $\mathcal{E}$와 성공 샘플을 Conflict Resolution Model(Teacher)에 제공하여 개선된 프롬프트 생성:

$$\mathcal{P}^{(t+1)} = \text{ConflictModel}(\mathcal{P}^{(t)}, \mathcal{E}, \mathcal{S})$$

여기서 $\mathcal{S}$는 성공 예제 집합입니다.

**Step 3 - 수렴 조건:**

$$\text{수렴} \iff |\mathcal{E}^{(t+1)}| \approx |\mathcal{E}^{(t)}| \quad \text{또는} \quad |\mathcal{E}^{(t)}| < \delta$$

실험에서 StereoSet은 **1번**, Contract-NLI는 **2번** 반복으로 수렴했습니다.

#### Phase 4: 추론 (Inference)

최종 정제된 시스템 프롬프트 $\mathcal{P}^*$를 Student 모델에 주입하여 Zero-shot 추론 수행:

$$\hat{y} = \text{Student}(x \mid \mathcal{P}^*)$$

런타임에서 중간 추론 토큰 생성이 불필요합니다.

---

### 2.3 모델 구조 및 실험 설정

#### Teacher 모델
- **명령 추출**: Gemini 3 Flash (thinking mode 활성화)
- **클러스터 합성 & 충돌 해결**: Gemini 3 Pro (thinking mode 활성화)
- **임베딩**: Gemini Embedding (768차원, Lee et al., 2025)

#### Student 모델 (평가 대상)
| 모델 | 파라미터 | 특징 |
|---|---|---|
| Gemma-3 4B | 4B | 소형, 고효율 추론 |
| Mistral Small 3.1 | 24B | 교차 아키텍처 검증 |
| Gemini 2 Flash | - | 저비용 추론 |

#### Baseline 비교 대상
- Zero-shot 프롬프팅
- Few-shot 프롬프팅 ($k=5$, 무작위 샘플링)
- TextGrad (Yuksekgonul et al., 2024) - APO 방법론
- LoRA 파인튜닝 (Hu et al., 2022) - Gemma-3 4B 대상

---

### 2.4 성능 향상

#### 주요 실험 결과 (Table 1)

**StereoSet (Macro-F1):**

| 방법 | Gemma-3 4B | Gemini 2 Flash | Mistral Small 3.1 |
|---|---|---|---|
| Zero-shot | 0.57 | 0.53 | 0.65 |
| Few-shot | 0.75 | 0.83 | 0.77 |
| Fine-tuning | 0.89 | - | 0.96 |
| TextGrad | 0.87 | 0.90 | 0.96 |
| **PLD (Post-Conflict)** | **0.90** | **0.93** | **0.97** |

**Contract-NLI (Macro-F1):**

| 방법 | Gemma-3 4B | Gemini 2 Flash | Mistral Small 3.1 |
|---|---|---|---|
| Zero-shot | 0.67 | 0.73 | 0.71 |
| Fine-tuning | 0.76 | - | 0.77 |
| TextGrad | 0.74 | 0.77 | 0.73 |
| **PLD (Post-Conflict)** | **0.83** | **0.83** | **0.78** |

**LogiQA (Accuracy):**

| 방법 | Gemma-3 4B | Gemini 2 Flash |
|---|---|---|
| Zero-shot | 0.67 | 0.64 |
| Fine-tuning | 0.67 | - |
| **PLD (Post-Conflict)** | **0.70** | **0.67** |

#### 비용 및 속도 효율성

- Gemma-3 4B는 Gemini 3 Flash 대비 **25배 저렴**, **80배 빠름**
- Gemma-3 4B의 평균 지연: ~0.046초 vs Gemini 3 Flash: ~3.768초

---

### 2.5 한계점

논문에서 명시적으로 인정한 한계:

1. **동적 추론 한계**: 복잡한 산술, 기호 증명 등 중간 토큰 생성이 필수적인 작업에는 적용 어려움
2. **컨텍스트 창 확장 문제**: 태스크 복잡도 증가 시 통합 명령 집합이 컨텍스트 창을 초과할 수 있음
3. **평가 범위**: 현재 추론 집약적 분류 태스크에 국한
4. **편향 위험**: Teacher 모델이 훈련 데이터의 편향을 증폭시킬 가능성 (StereoSet 관련 윤리적 우려)
5. **DBSCAN 하이퍼파라미터 민감도**: $\epsilon$, $\text{min samples}$ 설정에 따른 성능 변동 (Appendix C.2)

---

## 3. 일반화 성능 향상 가능성

이 논문에서 가장 중요한 함의 중 하나는 **일반화 성능 향상** 메커니즘입니다.

### 3.1 교차 아키텍처 일반화 (Cross-Architecture Generalizability)

PLD의 핵심 강점은 **아키텍처 독립성**입니다:

$$\mathcal{P}^* \xrightarrow{\text{주입}} \text{Gemma-3 4B}, \text{ Mistral Small 3.1}, \text{ Gemini 2 Flash}$$

동일한 정제 프롬프트를 다른 아키텍처에 적용했을 때 유사한 성능 향상이 관찰되었습니다. 이는 추출된 논리 규칙이 **모델 아키텍처에 종속되지 않는 보편적 지식 표현**임을 시사합니다.

### 3.2 소수 레이블 및 엣지 케이스 보존

충돌 해결 단계는 단순 정확도 향상뿐 아니라 **소수 레이블(minority label)에 대한 논리적 커버리지 보존**에 핵심적 역할을 합니다:

```
클러스터링 합성 단계의 문제:
  - 빈도가 낮은 엣지 케이스 관련 규칙이 노이즈로 처리되어 버려짐
  - 소수 레이블에 대한 충돌 규칙이 무작위 병합됨

충돌 해결 루프의 해결:
  - 오류 샘플 집중 분석 → 누락된 엣지 케이스 규칙 복원
  - 전체 정확도 ↑ + 소수 클래스 재현율 ↑
```

### 3.3 확장성 일반화 (Scalability Ablation)

Appendix D의 스케일 실험에서 중요한 일반화 특성이 관찰됩니다:

| 데이터셋 크기 | 총 클러스터 수 | 프롬프트 길이(토큰) | F1 Macro |
|---|---|---|---|
| 1,030 | 16 | 4,062 | 0.77 |
| 3,090 | 18 | 4,410 | 0.80 |
| 7,190 | 18 | 4,630 | 0.83 |

**핵심 관찰:** 데이터 증가에도 클러스터 수가 **18에서 포화(plateau)**되어 프롬프트 길이가 안정적으로 유지됩니다. 추가 데이터는 새 주제를 생성하는 것이 아니라 **기존 클러스터를 정제**하여 성능을 향상시킵니다.

수학적으로:

$$\lim_{|T| \to \infty} |C| \approx C_{\max}$$

이는 도메인의 논리 구조가 유한한 수의 핵심 패턴으로 수렴함을 의미합니다. 이 특성은 데이터 스케일에 대한 **강건한 일반화**를 보장합니다.

### 3.4 의미론적 압축을 통한 일반화

PLD는 특정 엔티티 이름을 제거하고 **인과 메커니즘(causal mechanism)**만을 보존하는 방식으로 명령을 추상화합니다:

$$I_i = \text{Abstraction}(\text{reasoning}(x_i, y_i)) \setminus \{\text{entity names}\}$$

이러한 추상화는 훈련 데이터에 나타나지 않은 새로운 입력에도 적용 가능한 **도메인 일반 규칙**을 생성합니다.

### 3.5 타 도메인 이전 가능성

PLD가 법률(Contract-NLI), 편향 측정(StereoSet), 논리 추론(LogiQA) 등 **이질적인 도메인**에서 모두 성능 향상을 보인다는 점은:

- 의료 기록 분류
- 금융 규정 준수 검사
- 콘텐츠 모더레이션
- 엣지 디바이스 배포

등 다양한 도메인으로의 일반화 가능성을 강하게 시사합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 지식 증류 계열과의 비교

| 연구 | 방법 | 파라미터 업데이트 | 추론 지연 | 해석 가능성 | 유지보수 비용 |
|---|---|---|---|---|---|
| Hinton et al. (2015) | Soft label KD | 필요 | 낮음 | 낮음 | 높음 |
| Hsieh et al. (2023) **Distilling Step-by-Step** | 교사 트레이스로 SFT | 필요 | 낮음 | 낮음 | 높음 |
| Ho et al. (2023) **Fine-tune-CoT** | CoT 트레이스 파인튜닝 | 필요 | 낮음 | 낮음 | 높음 |
| Magister et al. (2023) | 소형 모델 추론 교육 | 필요 | 낮음 | 낮음 | 높음 |
| **PLD (본 논문)** | 프롬프트 증류 | **불필요** | **매우 낮음** | **높음** | **매우 낮음** |

**Distilling Step-by-Step** (Hsieh et al., 2023)과 비교 시, PLD는 더 적은 데이터로 유사하거나 더 나은 성능을 달성하면서 재학습 비용이 전혀 없습니다.

### 4.2 자동 프롬프트 최적화(APO)와의 비교

| 연구 | 목적 | 추론 방식 | 오프라인 추론 캐싱 |
|---|---|---|---|
| Zhou et al. (2023b) **APE** | 최적 프롬프트 탐색 | 모델 내재 | ✗ |
| Yang et al. (2024) **OPRO** | LLM을 최적화기로 사용 | 모델 내재 | ✗ |
| Khattab et al. (2024) **DSPy** | 파이프라인 최적화 | 모델 내재 | ✗ |
| Yuksekgonul et al. (2024) **TextGrad** | 텍스트 그래디언트 최적화 | 모델 내재 | ✗ |
| **PLD (본 논문)** | **논리 휴리스틱 합성** | **외부화** | **✓** |

핵심 차이: APO 방법들은 **프롬프트 표현(wording)** 최적화에 집중하는 반면, PLD는 **논리 패턴 합성**을 수행합니다. TextGrad 대비 Contract-NLI에서 +9% Macro-F1 달성.

### 4.3 CoT 효율화 연구와의 비교

| 연구 | 방법 | 런타임 추론 비용 |
|---|---|---|
| Kang et al. (2025) **C3oT** | CoT 압축 | 여전히 필요 |
| Shah et al. (2026b) **CROP** | 토큰 효율적 추론 | 여전히 필요 |
| Shi et al. (2025) **SpecCoT** | 추론 추측적 디코딩 | 감소되나 필요 |
| Struppek et al. (2025) **Focused CoT** | 구조적 입력 정보 | 여전히 필요 |
| **PLD (본 논문)** | **오프라인 추론 캐싱** | **제로(Zero)** |

### 4.4 블랙박스 증류 연구와의 비교

| 연구 | 접근 방법 | API 접근 필요 |
|---|---|---|
| Gudibande et al. (2023) | 스타일 모방 (한계 지적) | 필요 |
| Ye et al. (2025) | 블랙박스 온정책 증류 | 필요 |
| Li et al. (2025) | 위원회 학습 | 필요 |
| **PLD (본 논문)** | **프롬프트 주입** | **선택적** |

### 4.5 소형 모델 추론 강화 연구와의 비교

| 연구 | 방법 | 파인튜닝 필요 |
|---|---|---|
| Fu et al. (2023) | 멀티스텝 추론 특화 | ✓ |
| Magister et al. (2023) | 소형 모델 추론 교육 | ✓ |
| Webb et al. (2023) | 창발적 유추 추론 분석 | - |
| **PLD (본 논문)** | **프롬프트 주입** | **✗** |

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 패러다임 전환: "가중치 증류" → "컨텍스트 증류"

PLD는 AI 시스템 설계의 근본적 관점 전환을 제시합니다:

$$\underbrace{\theta_{\text{student}} = f(\theta_{\text{teacher}})}_{\text{파라메트릭 증류}} \quad \longrightarrow \quad \underbrace{\mathcal{P}^* = g(T, \theta_{\text{teacher}})}_{\text{비파라메트릭 증류}}$$

이 패러다임 전환은 다음 연구 방향을 촉진할 것입니다:

**1) 도메인 특화 추론 라이브러리 구축**
- 법률, 의료, 금융 등 각 도메인의 PLD 명령 집합을 "추론 지식 베이스"로 구축
- 소형 모델들이 공유 가능한 포터블 추론 패턴 라이브러리

**2) Closed-Weight 모델 활용 확대**
- API 형태로만 접근 가능한 모델에도 적용 가능한 증류 방법론 개발
- 독점 LLM API의 논리를 오픈소스 소형 모델로 이전

**3) 설명 가능한 AI(XAI) 연구와의 융합**
- PLD로 생성된 자연어 규칙은 모델 결정의 완전한 인간 검증 가능
- 규제 산업(법률, 의료, 금융)의 AI 감사(audit) 프레임워크로 활용

#### 5.1.2 에지 컴퓨팅 및 온디바이스 AI

Gemma-3 4B의 경우:
- 지연: **0.046초** (Gemini 3 Flash 대비 80배 빠름)
- 비용: **25배 저렴**

이는 스마트폰, IoT 디바이스, 오프라인 환경에서의 고품질 추론 가능성을 열어줍니다.

#### 5.1.3 LLM 유지보수 전략의 혁신

$$\text{기존}: \text{Teacher 업데이트} \Rightarrow \text{Student 재학습} \Rightarrow \text{배포}$$

$$\text{PLD}: \text{Teacher 업데이트} \Rightarrow \text{PLD 재실행} \Rightarrow \text{프롬프트 업데이트}$$

재학습 없이 Teacher 모델 업그레이드의 혜택을 즉시 반영할 수 있어 MLOps 비용을 대폭 절감합니다.

---

### 5.2 향후 연구 시 고려사항

#### 5.2.1 방법론적 개선 방향

**① 인컨텍스트 클러스터링 (In-Context Clustering) 적용**

논문 자체에서 제안한 미래 방향으로, Wang et al. (2025)의 인컨텍스트 클러스터링을 DBSCAN 대신 활용하면 더 의미론적으로 일관성 있는 클러스터 생성이 가능합니다:

$$\text{DBSCAN}(\epsilon, \text{min samples}) \rightarrow \text{LLM-based In-Context Clustering}$$

**② 충돌 해결 루프의 실패 샘플 샘플링 전략**

현재 단순 샘플링에서 더 정교한 전략으로 발전:
- 불확실성 기반 샘플링 (Uncertainty Sampling)
- 다양성 기반 샘플링 (Diversity-based Sampling)
- 커리큘럼 학습 기반 단계적 난이도 증가

**③ 프롬프트 압축과 PLD의 결합**

컨텍스트 창 한계 문제 해결:

```math
\mathcal{P}^*_{\text{compressed}} = \text{LLMLingua}(\mathcal{P}^*) \text{ (Jiang et al., 2023)}
```

PLD로 생성된 장문의 명령 집합을 프롬프트 압축 기법으로 토큰 효율화.

#### 5.2.2 평가 범위 확장

**① 생성 태스크로의 확장**

현재 분류 태스크에 국한된 평가를 다음으로 확장:
- 요약 (Summarization)
- 질의응답 (Open-domain QA)
- 코드 생성 (Code Generation)

**② 도메인 드리프트 강건성 평가**

훈련 도메인과 다른 도메인에서의 성능 측정:

$$\text{일반화 갭} = \text{F1}_{\text{in-domain}} - \text{F1}_{\text{out-of-domain}}$$

PLD가 학습한 규칙이 도메인 변화에 얼마나 강건한지 체계적 분석 필요.

**③ 연속 학습(Continual Learning) 시나리오**

도메인 로직이 시간에 따라 변화하는 경우:
- 규칙의 점진적 업데이트 메커니즘
- 기존 규칙의 보존과 새 규칙 통합 간의 균형

#### 5.2.3 이론적 기반 강화

**① 프롬프트 용량 이론 개발**

$$\text{정보 용량} = f(\text{컨텍스트 창 크기}, \text{명령 복잡도}, \text{모델 파라미터 수})$$

PLD가 전달할 수 있는 최대 추론 복잡도의 이론적 상한선을 정립해야 합니다.

**② 클러스터 수와 성능의 최적 관계 이론화**

실험에서 관찰된 클러스터 수 포화 현상에 대한 이론적 설명:

$$|C^*| \propto \log(|\mathcal{L}|) \cdot H(\text{task})$$

여기서 $|\mathcal{L}|$은 레이블 수, $H(\text{task})$는 태스크의 논리적 복잡도입니다.

#### 5.2.4 윤리 및 안전성 고려사항

**① 편향 전파 방지 메커니즘**

Teacher 모델이 훈련 데이터의 편향을 명령에 인코딩할 위험:

```
훈련 데이터 편향 → Teacher 명령 추출 → Consolidated Prompt → Student 편향 행동
```

**해결 방향:**
- 명령 추출 전 훈련 데이터 편향 감사
- 충돌 해결 단계에서 공정성 제약 추가
- 인간 검토자(Human-in-the-loop) 통합

**② 폐쇄형 모델 의존성**

현재 구현이 Gemini 3 Flash/Pro에 의존 → 벤더 종속(vendor lock-in) 위험

오픈소스 Teacher 모델(Llama 3.1, Qwen 등)로의 이전 가능성 검증 필요.

#### 5.2.5 산업 적용 시 고려사항

**① 규제 환경 적합성**

PLD의 투명성 특성은 EU AI Act, GDPR 등 규제 요구사항에 부합하나:
- 자동 생성된 명령의 법적 책임 소재
- 규제 기관을 위한 명령 집합 감사 프로세스 수립

**② 멀티모달 확장**

텍스트 기반 규칙 추출을 이미지, 테이블, 코드 등 멀티모달 데이터로 확장 시:
- 모달리티별 임베딩 및 클러스터링 전략 개발
- 교차 모달 논리 규칙의 표현 방법

---

## 참고문헌 (본 답변에서 인용된 자료)

**주요 논문 (제공된 PDF):**
- Badhe, S., & Shah, D. (2026). *Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning*. arXiv:2602.21103v2

**논문 내 참조 문헌 (PDF 참고문헌 섹션에서 확인된 것만 포함):**
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531
- Hsieh, C.-Y. et al. (2023). *Distilling step-by-step! Outperforming larger language models with less training data*. ACL 2023
- Ho, N. et al. (2023). *Large language models are reasoning teachers*. ACL 2023
- Wei, J. et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. NeurIPS 2022
- Wang, X. et al. (2023). *Self-consistency improves chain of thought reasoning in language models*. ICLR 2023
- Yuksekgonul, M. et al. (2024). *TextGrad: Automatic "differentiation" via text*. arXiv:2406.07496
- Hu, E. J. et al. (2022). *LoRA: Low-rank adaptation of large language models*. ICLR 2022
- Gudibande, A. et al. (2023). *The false promise of imitating proprietary LLMs*. arXiv:2305.15717
- Ester, M. et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD 1996
- Khattab, O. et al. (2024). *DSPy: Compiling declarative language model calls into state-of-the-art pipelines*. ICLR 2024
- Zhou, Y. et al. (2023b). *Large language models are human-level prompt engineers*. ICLR 2023
- Yang, C. et al. (2024). *Large language models as optimizers*. ICLR 2024
- Koreeda, Y., & Manning, C. D. (2021). *ContractNLI: A dataset for document-level NLI for contracts*. arXiv:2110.01799
- Nadeem, M. et al. (2021). *StereoSet: Measuring stereotypical bias in pretrained language models*. ACL 2021
- Liu, J. et al. (2020). *LogiQA: A challenge dataset for machine reading comprehension with logical reasoning*. IJCAI 2020
- Lee, J. et al. (2025). *Gemini embedding: Generalizable embeddings from Gemini*. arXiv:2503.07891
- Luo, Y. et al. (2025). *An empirical study of catastrophic forgetting in LLMs*. IEEE TASLP
- Jiang, H. et al. (2023). *LLMLingua: Compressing prompts for accelerated inference*. EMNLP 2023
- Wang, Y., Ren, M., & Wilson, A. G. (2025). *In-context clustering with large language models*. arXiv:2510.08466
- Kang, Y. et al. (2025). *C3oT: Generating shorter chain-of-thought*. AAAI 2025
- Shah, D. et al. (2026b). *CROP: Token-efficient reasoning via regularized prompt optimization*. ICLR 2026 Workshop
- Shi, J. et al. (2025). *SpecCoT: Accelerating chain-of-thought reasoning through speculative exploration*. EMNLP 2025
- Webb, T., Holyoak, K. J., & Lu, H. (2023). *Emergent analogical reasoning in large language models*. Nature Human Behaviour
