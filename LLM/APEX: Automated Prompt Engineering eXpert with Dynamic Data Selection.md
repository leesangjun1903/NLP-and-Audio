# APEX: Automated Prompt Engineering eXpert with Dynamic Data Selection 

---

## 📌 참고 자료 (출처)

> **본 답변은 제공된 논문 PDF 원문에만 기반하여 작성되었습니다.**
>
> - **주요 논문**: Wang, F., Si, S., Hsieh, C.-J., & Dhillon, I. S. (2026). *APEX: Automated Prompt Engineering eXpert with Dynamic Data Selection*. arXiv:2606.11459v1 [cs.CL], 9 Jun 2026.
>
> **논문 내 인용 참고문헌 (비교 분석용)**:
> - Pryzant et al. (2023). *Automatic prompt optimization with "gradient descent" and beam search*. EMNLP 2023.
> - Agrawal et al. (2025). *GEPA: Reflective prompt evolution can outperform reinforcement learning*. arXiv:2507.19457.
> - Zhou et al. (2023b). *Large language models are human-level prompt engineers*. ICLR 2023.
> - Hsieh et al. (2024). *Automatic engineering of long prompts*. ACL Findings 2024.
> - Fernando et al. (2024). *Promptbreeder: self-referential self-improvement via prompt evolution*. ICML 2024.
> - Yang et al. (2024). *Large language models as optimizers*. ICLR 2024.
> - Xia et al. (2024). *LESS: Selecting influential data for targeted instruction tuning*. ICML 2024.
> - Zhou et al. (2023a). *LIMA: Less is more for alignment*. NeurIPS 2023.
> - Wan et al. (2024). *Teach better or show smarter? on instructions and exemplars in automatic prompt optimization*. NeurIPS 2024.
> - Diao et al. (2024). *Active prompting with chain-of-thought for large language models*. ACL 2024.
> - Dong et al. (2025). *Model performance-guided evaluation data selection for effective prompt optimization*. ACL Findings 2025.
> - Shin et al. (2020). *AutoPrompt: Eliciting knowledge from language models with automatically generated prompts*. arXiv:2010.15980.
> - Khattab et al. (2024). *DSPy: Compiling declarative language model calls into self-improving pipelines*.
> - Novikov et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv:2506.13131.
> - Snell et al. (2025). *Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning*. ICLR 2025.
> - Yuksekgonul et al. (2025). *Optimizing generative AI by backpropagating language model feedback*. Nature.

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장 (Core Claim)

APEX의 핵심 주장은 다음 한 문장으로 요약됩니다:

> **"자동 프롬프트 최적화의 병목은 탐색 알고리즘(search algorithm)이 아니라 데이터 활용 방식(data usage)에 있으며, 데이터를 동적으로 분류하고 정보가 높은 데이터에 집중하면 동일한 계산 예산 하에서 훨씬 높은 성능을 달성할 수 있다."**

기존의 유전 알고리즘 기반 프롬프트 최적화 방법(APO, GEPA 등)은 개발 데이터셋을 **정적(static) 벤치마크**로 취급하여, 최적화 과정에서 정보 가치가 변하는 데이터 포인트들을 무분별하게 사용해왔습니다. APEX는 이 문제를 **데이터 중심적(data-centric) 관점**에서 해결합니다.

### 🏆 주요 기여 (Contributions)

논문이 명시하는 세 가지 기여는 다음과 같습니다:

| # | 기여 | 내용 |
|---|------|------|
| 1 | **병목 문제 식별** | 기존 유전적 프롬프트 최적화의 핵심 병목이 "데이터 효율성 부재"임을 최초로 명확히 규명 |
| 2 | **APEX 알고리즘 제안** | 데이터를 Easy/Hard/Mixed 세 계층으로 동적 분류하고, "Addressable Frontier"와 "Rank-Sensitive Frontier"를 타겟팅하는 새로운 프레임워크 제안 |
| 3 | **실증적 성능 향상** | 5,000회 평가 호출의 고정 예산 하에서, Gemini 2.5 Flash에서 **+11.2%**, Gemma 3 27B에서 **+6.8%** 성능 향상 달성 |

---

## 2. 상세 분석: 문제 정의, 방법론, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제 (Problem Statement)

#### 2.1.1 프롬프트 최적화의 공식 정의

최적 프롬프트 $P^*$는 다음과 같이 정의됩니다:

$$P^* = \arg\max_{P} \mathbb{E}_{x \sim \mathcal{D}^*}[f(x, \text{LLM}(P, x))] $$

여기서 $\mathcal{D}^*$는 오라클 데이터 분포이고, $f(\cdot)$는 평가 함수(문자열 매칭 정확도 또는 모델 기반 채점)입니다. 이진 평가 결과는 다음과 같이 정의됩니다:

$$s(P, x) \in \{0, 1\} $$

#### 2.1.2 기존 방법의 두 단계

**Mutation (변이):**

$$P_{\text{new}} \sim \text{LLM}_{\text{meta}}(P_{\text{curr}}, \mathcal{E}_{\text{curr}}) $$

여기서 $\mathcal{E}\_{\text{curr}} \subset \{x \in \mathcal{D} \mid s(P_{\text{curr}}, x) = 0\}$은 현재 프롬프트의 실패 사례 집합입니다.

**Selection (선택):**

$$\mathcal{P}_{t+1} = \text{Top-}k\left(\mathcal{P}_t \cup \{P_{\text{new}}\} \mid \mathcal{D}\right) $$

#### 2.1.3 데이터 효율성 문제의 공식화

전체 최적화 상태를 **희소 점수 행렬(sparse score matrix)** $\mathbf{S} \in \{0, 1, \emptyset\}^{|\mathcal{P}| \times |\mathcal{D}|}$로 표현합니다:

$$S_{j,i} = s(P_j, x_i) \in \{0, 1\}, \quad (\text{또는 } \emptyset \text{ if not applicable}) $$

**변이(Mutation)의 비효율:**
- 랜덤 샘플링으로 인해 최적화 궤적(trajectory)을 고려하지 못함
- "해결 불가능한(Hard)" 오류 케이스에 집중하여 $\text{LLM}_{\text{meta}}$를 오도(mislead)
- 오류 해결 가능성의 계층 구조(fixability hierarchy)를 무시: $\mathcal{E}_a \to \mathcal{E}_b$ 순서가 중요하나 이를 역전시키거나 무작위 샘플링하면 고분산 업데이트 발생

**선택(Selection)의 비효율:**
- 전체 평가는 예산의 **90% 이상**을 소모하여 반복 횟수 $T$ 제한
- 무작위 서브샘플링은 랭크 역전(rank inversion) 유발
- 진정으로 필요한 것은 **판별적 데이터(discriminative data)**:

$$\mathcal{D}_{\text{disc}} = \{x_i \in \mathcal{D} \mid \exists P_a, P_b \in \mathcal{P}_t : s(P_a, x_i) \neq s(P_b, x_i)\} $$

$x_i \notin \mathcal{D}_{\text{disc}}$인 데이터 포인트 평가는 선택에 **zero information**을 제공

---

### 2.2 제안하는 방법 (Proposed Method: APEX)

#### 2.2.1 동적 데이터 계층화 (Dynamic Data Stratification)

**핵심 개념: 로컬 히스토리 $\mathcal{R}_i$**

데이터 포인트 $x_i \in \mathcal{D}$에 대해 $\mathcal{H}_{\text{valid}}^{(i)}$를 $x_i$를 실제 평가한 과거 프롬프트의 부분 시퀀스라 할 때, **최근 $k$개 프롬프트의 결과 집합**으로 정의합니다:


```math
\mathcal{R}_i = \left\{s(P, x_i) \mid P \in \text{last}_k\left(\mathcal{H}_{\text{valid}}^{(i)}\right)\right\}
```

**세 계층(Tier) 분류:**

$$\text{Tier}(i) = \begin{cases} \text{E (Easy)} & \text{if } \text{Set}(\mathcal{R}_i) \equiv \{1\} \\ \text{H (Hard)} & \text{if } \text{Set}(\mathcal{R}_i) \equiv \{0\} \\ \text{M (Mixed)} & \text{if } \text{Set}(\mathcal{R}_i) \equiv \{0, 1\} \end{cases} $$

| 계층 | 의미 | 전략적 역할 |
|------|------|-------------|
| **Easy** | 최근 모든 프롬프트가 해결 | 재평가 가치 최소 → 스킵 |
| **Hard** | 최근 모든 프롬프트가 실패 | 현재로선 풀 수 없는 노이즈 |
| **Mixed** | 혼합 성능(일부 성공, 일부 실패) | 정보가 가장 풍부한 핵심 데이터 |

역사적 계층 $T \in \{E, H, M\}$과 현재 프롬프트 평가 결과 $s \in \{1, 0, \emptyset\}$의 교차로 데이터를 **9개의 세부 버킷 $\mathcal{B}\_{T,s}$ **으로 분류합니다. 예: $\mathcal{B}_{M,0}$ = Mixed이지만 현재 실패 중인 데이터.

#### 2.2.2 궤적 유도 변이 (Trajectory-Guided Mutation)

**Addressable Frontier 타겟팅:**

변이를 위한 오류 사례는 주로 $\mathcal{B}_{M,0}$("Mixed-Fail" 버킷)에서 선택합니다. 이 "소프트 실패(soft failures)"는 모델이 최근 프롬프트 계통에서 해결 능력을 보였지만 현재 변형에서 회귀(regressed)한 케이스입니다.

**커버리지 극대화를 위한 사용 이력(usage history) $\mathcal{U}$ 관리:**

$$e \in \{x_i \mid x_i \in (\mathcal{B}_{M,0} \cup \mathcal{B}_{H,0}),\ x_i \notin \mathcal{U}\} $$

방문하지 않은 실패 사례만을 변이에 활용하여 동일 실패 케이스 반복 과적합(overfitting) 방지. $\mathcal{U}$가 소진되면 리셋하여 광범위한 오류 표면 탐색.

#### 2.2.3 랭크 민감 평가 (Rank-Sensitive Evaluation)

**평가 예산 $N$의 구성:**

1. **필수 기준선 (Required baseline)**:

$$\mathcal{D}_{\text{req}} = \mathcal{B}_{M,\emptyset} \quad \text{(역사적으로 Mixed이지만 현재 미평가 데이터)}$$

2. **나머지 예산 계산**:

$$R = N - |\mathcal{D}_{\text{req}}|$$

3. **앵커 비율 $\alpha_t$ (Anchor ratio)**를 활용한 계층적 샘플링:

$$k_{\text{pos}} = \lfloor \min(\alpha_t, \rho_{\text{mix}}, \rho_{\text{all}}) \cdot R \rfloor, \quad k_{\text{neg}} = R - k_{\text{pos}} $$

여기서 $\rho_{\text{mix}} = \text{PassRate}(\mathcal{B}\_M)$, $\rho_{\text{all}} = \text{PassRate}(\mathcal{D})$

- $\mathcal{D}\_{\text{pos}}$: $\mathcal{B}\_{M,1}$ 우선 샘플링 (회귀 탐지) → $\mathcal{B}_{E,\emptyset}$ 순
- $\mathcal{D}\_{\text{neg}}$: $\mathcal{B}\_{M,0}$ 우선 샘플링 (수정 확인) → $\mathcal{B}_{H,\emptyset}$ 순
- **최종 평가 집합**: $\mathcal{D}\_{\text{eval}} = \mathcal{D}\_{\text{req}} \cup \mathcal{D}\_{\text{pos}} \cup \mathcal{D}_{\text{neg}}$

4. **앵커 어닐링 스케줄 (Anchor Annealing Schedule)**:

프롬프트 업데이트가 성공할 때마다 앵커 비율 증가:

$$\alpha_{t+1} = \alpha_t + \beta \cdot \mathbb{I}(P_{\text{new}} \succ P_{\text{curr}}) $$

프롬프트가 개선될수록 $\alpha$가 증가하여 "숙달된 지식(mastered logic)"을 보호하기 위한 평가 예산을 더 많이 할당. $\alpha_0 = 0.2$, $\beta = 0.03$으로 초기화.

5. **증분 평가 (Incremental Evaluation)**: 이미 이력 $\mathcal{H}$에 기록된 $s(P_{\text{curr}}, x_i)$는 메모리에서 직접 조회하여 불필요한 API 호출 최소화.

---

### 2.3 모델 구조 (Model Architecture & Algorithm Flow)

```
[최적화 시작]
    │
    ▼
[초기 프롬프트 P₀ 전체 평가] → H (이력) 초기화
    │
    ▼ (반복 t = 1...T)
┌───────────────────────────────────────────────────────┐
│  Step 1: 동적 데이터 계층화 (Algorithm 2)              │
│    - 각 x_i에 대해 최근 k개 결과 R_i 계산              │
│    - Easy / Hard / Mixed 분류                         │
│    - 현재 프롬프트 평가 결과 s로 9개 버킷 B_{T,s} 생성  │
└───────────────────────────┬───────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────┐
│  Step 2: 궤적 유도 변이 (Trajectory-Guided Mutation)   │
│    - B_{M,0} ∪ B_{H,0}에서 미방문 오류 m개 샘플링      │
│    - LLM_meta → Critique(P_curr, E) → 비평 C 생성     │
│    - LLM_meta → Mutate(P_curr, C) → P_new 생성        │
└───────────────────────────┬───────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────┐
│  Step 3: 랭크 민감 선택 (Rank-Sensitive Selection)     │
│    - D_req = B_{M,∅} (필수 평가 집합)                 │
│    - 앵커 비율 α_t로 D_pos, D_neg 계층적 샘플링         │
│    - P_new와 P_curr를 D_eval에서 평가                  │
│    - P_new > P_curr이면 업데이트 & α_t 증가            │
└───────────────────────────┬───────────────────────────┘
                            │
                     [H 업데이트]
                            │
                    [다음 반복 or 종료]
                            │
                     [최적 P* 반환]
```

**구현 하이퍼파라미터 요약:**

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| $k$ (Lookback window) | 5 | 계층 분류에 활용하는 최근 이력 수 |
| $m$ (Mutation batch size) | 5 | 변이에 사용하는 오류 케이스 수 |
| $N$ (Per-iteration budget) | 100 | 반복당 평가 예산 |
| $\alpha_0$ (Initial anchor ratio) | 0.2 | 초기 앵커 비율 |
| $\beta$ (Anchor increment) | 0.03 | 성공 시 앵커 증가량 |
| Total budget | 5,000 | 전체 평가 호출 횟수 제한 |

---

### 2.4 성능 향상 (Performance Results)

#### 2.4.1 메인 결과 (Table 1)

**Gemini 2.5 Flash 기준:**

| 방법 | IFBench | SimpleQA Verified | FACTS Grounding | 평균 |
|------|---------|-------------------|-----------------|------|
| Initial Prompt | 38.5 ± 1.2 | 23.6 ± 0.6 | 85.8 ± 1.1 | 49.3 |
| GEPA | 41.2 ± 0.9 (+2.7) | 28.8 ± 0.8 (+5.2) | 93.5 ± 0.0 (+7.7) | 54.5 (+5.2) |
| APO (\|D\|) | 43.7 ± 1.2 (+5.2) | 27.4 ± 2.6 (+3.8) | 89.7 ± 1.9 (+3.9) | 53.6 (+4.3) |
| APO (\|D\|/2) | 43.5 ± 1.9 (+5.0) | 25.0 ± 3.5 (+1.4) | 89.9 ± 0.1 (+4.1) | 52.8 (+3.5) |
| **APEX (Ours)** | **52.3 ± 1.4 (+13.8)** | **35.0 ± 2.3 (+11.4)** | **94.1 ± 0.4 (+8.3)** | **60.5 (+11.2)** |

**Gemma 3 27B 기준:**

| 방법 | IFBench | SimpleQA Verified | FACTS Grounding | 평균 |
|------|---------|-------------------|-----------------|------|
| Initial Prompt | 33.4 ± 0.7 | 9.4 ± 0.5 | 80.7 ± 1.6 | 41.2 |
| GEPA | 34.1 ± 0.5 (+0.7) | 9.4 ± 0.5 (+0.0) | 91.7 ± 0.5 (+11.0) | 45.1 (+3.9) |
| APO (\|D\|) | 35.7 ± 2.4 (+2.3) | 11.5 ± 0.1 (+2.1) | 88.5 ± 0.7 (+7.8) | 45.2 (+4.0) |
| **APEX (Ours)** | **39.3 ± 2.0 (+5.9)** | **11.5 ± 1.2 (+2.1)** | **93.3 ± 0.1 (+12.6)** | **48.0 (+6.8)** |

#### 2.4.2 어블레이션 연구 결과 (Table 2 & 3)

**컴포넌트별 기여도 (IFBench, Gemini 2.5 Flash):**

| Trajectory-Guided Mutation | Rank-Sensitive Selection | Score |
|:-:|:-:|:-----:|
| ✓ | ✓ | **52.3** |
| ✗ | ✓ | 50.2 |
| ✓ | ✗ | 48.3 |
| ✗ | ✗ | 42.9 |

→ **두 컴포넌트의 시너지 효과**: 개별 기여보다 결합 시 큰 성능 향상

**데이터 계층 샘플링 전략별 비교 (Table 3):**

| 방법 | Score |
|------|-------|
| APEX | **52.3** |
| Hard + Mixed에서 무작위 샘플링 | 47.3 |
| Hard만 무작위 샘플링 | 30.3 |
| 전체 데이터에서 무작위 | 42.9 |

→ Hard 계층만 사용 시 **심각한 과적합** 발생 (30.3), Mixed 계층 우선화의 중요성 입증

#### 2.4.3 예산 효율성 및 프롬프트 간결성

**Lookback window 민감도 분석 (Table 4):**

| 설정 | Score |
|------|-------|
| lookback = 3 | 50.3 |
| **lookback = 5 (default)** | **52.3** |
| lookback = 10 | 50.6 |

→ 너무 좁으면 맥락 부족, 너무 넓으면 오래된 신호 재유입. **$k=5$가 최적 균형**

**프롬프트 길이 비교 (Table 5):**

| 방법 | 프롬프트 길이 |
|------|-------------|
| **APEX** | **257 (가장 짧음)** |
| APO (\|D\|) | 275 |
| APO (\|D\|/2) | 342 |
| GEPA | 662 |

→ APEX는 더 짧고 정밀한 프롬프트로 최고 성능 달성. 성능 향상이 단순 텍스트 증가에서 비롯되지 않음을 입증.

---

### 2.5 한계점 (Limitations)

논문이 명시한 한계:

1. **대표적 데이터셋 및 신뢰할 수 있는 평가 함수 의존성**: 주관적이거나 개방형(open-ended) 태스크에서는 프로그래밍적 평가 함수 확보가 어려움

2. **타겟 모델의 내재적 역량에 의한 상한선**: APEX는 모델이 합리적인 기준 능력을 보유하고 있다고 가정. 근본적인 지식/추론 결함은 극복 불가 (Gemma 3 27B의 SimpleQA Verified 사례)

3. **텍스트 기반 LLM에 한정된 현재 평가**: 멀티모달, 에이전틱 워크플로우로의 확장은 미래 과제

4. **이진 평가 가정의 제약**: $s(P, x) \in \{0, 1\}$의 이진화로 부분 점수(partial credit) 정보 손실

5. **블랙박스 API 설정의 전제**: 내부 모델 신호(logits, gradients) 불가 환경에서 설계되어, 화이트박스 접근 가능 환경에서의 비교 연구 부재

---

## 3. 모델의 일반화 성능 향상 가능성 (Generalization Performance)

이 항목은 APEX 논문에서 가장 중요한 발견 중 하나와 직결됩니다.

### 3.1 일반화를 지원하는 핵심 메커니즘

#### 3.1.1 Mixed 계층 우선화의 일반화 효과

논문의 핵심 발견 중 하나는 **Hard 계층만 사용하면 과적합(overfitting)이 심각하게 발생**한다는 것입니다. Table 3의 결과에서:

$$\text{Score}(\text{Hard only}) = 30.3 \ll \text{Score}(\text{APEX}) = 52.3$$

Hard 계층은 현재 완전히 풀 수 없는 엣지 케이스(edge cases)에 해당합니다. 이런 케이스에만 집중하면 프롬프트가 좁은 극단적 케이스에 과적합되어 일반 성능이 저하됩니다. 반면 Mixed 계층은 "경계(boundary)" 데이터로서, 프롬프트의 일반적 성능 개선에 직접 기여하는 데이터를 포함합니다.

$$\text{Generalization} \propto \text{Mixed Tier Priority}$$

논문은 이를 명시적으로 확인합니다:

> *"This confirms that prioritizing mixed-tier examples provides a vital grounding signal, preventing catastrophic overfitting and ensuring the prompt maintains broad generalization."* (Section 5.3)

#### 3.1.2 주소지정 가능한 프론티어(Addressable Frontier)와 점진적 일반화

APEX는 오류 계층 구조의 "fixability hierarchy"를 고려합니다:

$$\mathcal{E}_a \to \mathcal{E}_b \text{ (올바른 순서)} \quad \text{vs} \quad \mathcal{E}_b \to \mathcal{E}_a \text{ (잘못된 순서)}$$

현재 해결 가능한 오류($\mathcal{B}_{M,0}$)부터 순차적으로 해결함으로써:
- 기초적 오류 패턴부터 고차원 오류까지 **단계적 일반화** 달성
- 해결 불가능한 케이스를 시도하여 발생하는 "망각(forgetting)" 방지

#### 3.1.3 앵커링 메커니즘을 통한 회귀 방지

앵커 어닐링 스케줄은 프롬프트가 개선될수록 기존 성능을 유지하는 데 더 많은 예산을 할당합니다:

$$\alpha_{t+1} = \alpha_t + \beta \cdot \mathbb{I}(P_{\text{new}} \succ P_{\text{curr}})$$

이는 새로운 케이스를 해결하면서 이미 해결한 케이스의 성능이 회귀하지 않도록 보장합니다. $\mathcal{B}_{M,1}$ (Mixed이지만 현재 성공 중인 케이스)에 대한 모니터링을 통해 **성능 단조 증가(monotonic improvement)** 경향을 달성합니다.

#### 3.1.4 사용 이력 $\mathcal{U}$를 통한 탐색 다양성 보장

$$e \in \{x_i \mid x_i \in (\mathcal{B}_{M,0} \cup \mathcal{B}_{H,0}),\ x_i \notin \mathcal{U}\}$$

이미 사용한 오류 케이스를 제외함으로써 **반복적 과적합** 방지. $\mathcal{U}$ 소진 시 리셋으로 광범위한 오류 표면 탐색 → 특정 서브셋에 과적합되지 않는 일반화된 프롬프트 생성.

#### 3.1.5 테스트 셋 일반화의 실증적 증거

IFBench 실험 설계에서:

> *"Notably, the constraints in the development set and the test set have no overlap."* (Section 5.1)

개발 셋과 테스트 셋의 제약(constraints) 유형이 전혀 겹치지 않음에도 APEX가 최고 성능을 달성했다는 것은 **분포 외(out-of-distribution) 일반화** 능력을 실증합니다.

IFBench에서의 APEX 성능: **52.3%** (초기 프롬프트 대비 +13.8%) — 이는 학습/개발셋에서 본 적 없는 제약 유형에도 유효한 프롬프트를 생성했음을 의미합니다.

#### 3.1.6 프롬프트 질적 진화와 일반화

Figure 4의 질적 분석에서, APEX가 최적화한 프롬프트의 진화 패턴:

1. **초기 단계**: 입력 분석 (constraint 정의 명시)
2. **중간 단계**: 프로세스 제어 (plan-and-verify 메커니즘)
3. **최종 단계**: 엄격한 출력 아키텍처 (Constraints → Plan → Execution → Answer)

이러한 **메타인지적 스캐폴딩(meta-cognitive scaffolding)** 구조는 특정 케이스가 아닌 **범용적 추론 프레임워크**를 프롬프트에 내장시킵니다. 이것이 일반화 성능 향상의 질적 근거입니다.

### 3.2 일반화의 한계와 조건

그러나 논문은 일반화의 한계도 명확히 합니다:

> *"Prompt optimization yields the most significant returns in a 'promising region' where the model possesses latent knowledge but lacks the instruction alignment to express it."* (Section 5.2)

즉, 일반화 성능 향상은 다음 조건을 전제로 합니다:
- 모델이 해당 태스크에 대한 **잠재적 지식(latent knowledge)**을 보유해야 함
- 성능 저하가 지식 부족이 아닌 **지시 정렬(instruction alignment) 부재**에서 기인해야 함

Gemma 3 27B의 SimpleQA Verified에서 APEX의 개선폭이 상대적으로 작은 이유가 바로 이 때문입니다: 모델의 근본적 지식 결핍은 프롬프트 최적화만으로 극복 불가.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 자동 프롬프트 최적화 연구 계보

```
[기울기 기반 시대]                    [LLM 기반 시대]                 [데이터 중심 시대]
─────────────────────────────────────────────────────────────────────────────────────────
AutoPrompt (2020)    →    APE (2023b)    →    APO (2023)    →    AELP (2024)    →    APEX (2026)
Shin et al.               Zhou et al.         Pryzant et al.     Hsieh et al.        Wang et al.
(토큰 레벨 최적화)         (LLM as PE)        (유전 알고리즘)    (긴 프롬프트)       (동적 데이터)
```

### 4.2 주요 방법별 상세 비교

| 방법 | 연도 | 핵심 아이디어 | 데이터 전략 | 한계 | APEX와의 차이 |
|------|------|-------------|------------|------|--------------|
| **AutoPrompt** (Shin et al.) | 2020 | 기울기 기반 토큰 삽입 | 전체 데이터셋 | 기울기 필요, 해석 불가 | APEX는 블랙박스 환경 |
| **APE** (Zhou et al., 2023b) | 2023 | LLM이 후보 프롬프트 생성 | 랜덤 샘플링 | 정적 데이터, 단일 반복 | APEX는 동적 반복 최적화 |
| **APO** (Pryzant et al., 2023) | 2023 | 텍스트 피드백 기반 유전 알고리즘 | 랜덤 샘플링 또는 전체 데이터 | 랭크 불안정 또는 예산 소모 | APEX는 계층적 동적 선택 |
| **AELP** (Hsieh et al., 2024) | 2024 | 긴 프롬프트 최적화, 이력 안내 | 제한적 변이 범위 | 긴 프롬프트 특화 | APEX는 범용 |
| **Promptbreeder** (Fernando et al., 2024) | 2024 | 자기 참조적 자기 개선, 진화 | 전체 데이터 | 계산 비용 높음 | APEX는 계산 효율적 |
| **OPRO** (Yang et al., 2024) | 2024 | LLM as optimizer | 전체 데이터 | 정적 평가 | APEX는 동적 계층화 |
| **GEPA** (Agrawal et al., 2025) | 2025 | Pareto 기반 다목적 최적화, 에이전트 | 전체 개발셋 | 탐색 깊이 제한 (90%+ 예산 소모) | APEX는 선택적 평가 |
| **DSPy** (Khattab et al., 2024) | 2024 | 선언적 LLM 파이프라인 최적화 | 퓨샷 예시 선택 | 특정 프레임워크 의존 | APEX는 플러그인 가능 |
| **Active Prompting** (Diao et al., 2024) | 2024 | 능동 학습 기반 예시 선택 | 불확실성 기반 | 인간 주석 필요, 분류 한정 | APEX는 완전 자동화 |
| **Dong et al.** (2025) | 2025 | 모델 성능 유도 데이터 선택 | 유사도+신뢰도 휴리스틱 | 내부 신호(logits) 필요 | APEX는 블랙박스 |
| **APEX** (Wang et al., 2026) | 2026 | 동적 데이터 계층화 + 궤적 유도 변이 + 랭크 민감 선택 | **동적 Easy/Hard/Mixed 분류** | 평가 함수 및 대표 데이터 필요 | — |

### 4.3 데이터 선택 방법론 비교 (LLM Alignment 관점)

| 방법 | 연도 | 데이터 선택 기준 | 적용 단계 | APEX와의 관계 |
|------|------|----------------|----------|--------------|
| **LIMA** (Zhou et al., 2023a) | 2023 | 품질 > 양 (수동 큐레이션) | 파인튜닝 | APEX의 철학적 근거 |
| **AlpaGasus** (Chen et al., 2024b) | 2024 | LLM 품질 점수 기반 필터링 | 파인튜닝 | 정적 선택 vs APEX 동적 |
| **LESS** (Xia et al., 2024) | 2024 | 기울기 유사도 기반 | 파인튜닝 | 기울기 필요 vs APEX 블랙박스 |
| **Data Advisor** (Wang et al., 2024a) | 2024 | 동적 안전 정렬 데이터 큐레이션 | 파인튜닝 | 동적 접근의 유사성 |
| **APEX** | 2026 | 최적화 궤적 기반 동적 계층화 | **프롬프트 최적화** | 프롬프트 영역으로 데이터 선택 확장 |

### 4.4 핵심 차별성 분석

**APEX의 독창적 위치:**

```
기존 방법들의 한계                        APEX의 해결책
─────────────────────────────────────────────────────────────
정적 데이터 취급         →    동적 계층화 (Easy/Hard/Mixed)
랜덤 오류 샘플링         →    궤적 유도 Addressable Frontier
전체/랜덤 평가           →    Rank-Sensitive 계층적 샘플링
인간 주석 의존           →    완전 자동화 블랙박스 솔루션
내부 신호(logits) 의존   →    외부 평가 함수만 활용
분류 태스크 한정         →    생성 태스크 포함 범용 적용
```

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 5.1.1 데이터 중심 최적화 패러다임의 확립

APEX는 프롬프트 최적화 연구에서 **"알고리즘 중심"에서 "데이터 중심"으로의 패러다임 전환**을 제안합니다. 이는 다음 연구 방향에 영향을 미칩니다:

- **블랙박스 LLM 최적화 전반**: 후보 솔루션이 진화함에 따라 데이터도 함께 진화해야 한다는 원칙은 프롬프트 최적화를 넘어 더 광범위한 블랙박스 최적화에 적용 가능

- **에이전트 워크플로우 최적화**: 논문이 언급하듯, APEX는 멀티-스텝 에이전트 최적화(GEPA와 유사한 방향)와 결합 가능. 에이전트의 의사결정 궤적에서도 Easy/Hard/Mixed 분류 원칙 적용 가능성 존재

- **인퍼런스 타임 스케일링**: AlphaEvolve (Novikov et al., 2025), 테스트 타임 컴퓨트 스케일링 (Snell et al., 2025)과 결합하여 동적 데이터 가이드 기반의 인퍼런스 최적화 가능

#### 5.1.2 LLM 프로그래밍 프레임워크와의 통합

논문이 직접 언급하는 방향:
- **DSPy** (Khattab et al., 2024): APEX의 데이터 선택 메커니즘을 DSPy의 최적화 레이어로 통합
- **TextGrad** (Yuksekgonul et al., 2025): 언어 모델 피드백 역전파와 APEX의 동적 데이터 선택 결합

#### 5.1.3 평가 방법론 연구에 대한 시사점

- **동적 평가 프로토콜**: 고정 벤치마크 대신, 최적화 과정에서 데이터의 "가치"가 변화한다는 인식이 평가 방법론 연구를 자극할 것

- **계산 효율성 지표**: "평가 호출당 성능 향상(performance gain per evaluation call)"이 새로운 표준 지표로 부상할 가능성

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 기술적 고려사항

**① 연속적 평가 함수로의 확장**

현재 APEX는 $s(P, x) \in \{0, 1\}$의 이진 평가를 가정합니다. 실제 태스크에서는 연속적 점수가 더 적합한 경우가 많습니다:

$$s(P, x) \in [0, 1] \quad \text{(연속적 부분 점수 허용)}$$

이 경우 Easy/Hard/Mixed의 이진 분류 기준을 재설계해야 합니다. 예를 들어 분산(variance) 기반 분류나 임계값(threshold) 기반 분류가 필요합니다.

**② 멀티모달 및 에이전틱 확장**

논문이 한계로 언급한 텍스트 기반 LLM 한정 평가를 극복하기 위해:
- 멀티모달 태스크에서 이미지-텍스트 쌍의 계층적 분류 방법 설계
- 에이전트 워크플로우에서 다단계 행동 궤적(trajectory)의 난이도 분류 방법 개발

**③ 메타-러닝(Meta-Learning)과의 결합**

초기 앵커 비율 $\alpha_0$, 증가량 $\beta$, Lookback window $k$ 등의 하이퍼파라미터가 태스크마다 최적값이 다를 수 있습니다. 메타러닝을 통해 이를 태스크 특성에 맞게 자동 적응시키는 연구가 필요합니다.

**④ 분포 변이(Distribution Shift) 고려**

현재 APEX는 개발셋이 테스트 분포를 충분히 대표한다고 가정합니다. 실제 배포 환경에서 분포 변이가 발생하는 경우 Mixed 계층의 정의와 의미가 달라질 수 있습니다. **온라인 분포 적응(online distribution adaptation)** 메커니즘이 필요합니다.

**⑤ 다중 목적 최적화**

GEPA가 Pareto 기반 다목적 최적화를 시도했듯이, APEX도 단순 정확도 외에 **안전성, 편향 감소, 효율성** 등 다목적 목표를 동시에 고려하는 방향으로 확장 가능합니다:

$$P^* = \arg\max_{P} \mathbb{E}_{x \sim \mathcal{D}^*}\left[\alpha_1 f_{\text{accuracy}}(x, \text{LLM}(P,x)) + \alpha_2 f_{\text{safety}}(x, \text{LLM}(P,x)) + \cdots\right]$$

#### 5.2.2 방법론적 고려사항

**⑥ Cold Start 문제**

APEX는 초기에 전체 데이터셋을 평가($P_0$의 전체 평가)해야 합니다. 데이터셋이 매우 클 경우 이 초기화 비용이 상당합니다. 초기화 비용을 줄이는 **효율적 콜드 스타트 전략**이 필요합니다.

**⑦ 변이 다양성(Mutation Diversity) 보장**

현재 APEX의 변이는 단일 크리틱(critique)에서 단일 프롬프트를 생성합니다. 집단(population) 기반 다양한 변이를 병렬로 탐색하는 **빔 서치(beam search)** 또는 **앙상블 변이** 전략과의 결합을 고려해야 합니다.

**⑧ 최적화 안정성 이론화**

APEX가 경험적으로 보여주는 단조적 성능 향상 경향을 이론적으로 보장하는 수렴 분석(convergence analysis)이 부재합니다. 미래 연구에서는 APEX의 최적화 안정성에 대한 이론적 근거를 마련해야 합니다.

#### 5.2.3 실용적 고려사항

**⑨ 평가 함수 신뢰성**

APEX의 성능은 평가 함수 $f$의 신뢰성에 크게 의존합니다. 모델 기반 채점(model-based grading)을 사용하는 경우 채점 LLM 자체의 편향이 최적화 방향을 왜곡할 수 있습니다. **평가 함수 불확실성(uncertainty)을 고려한 강건한 최적화** 연구가 필요합니다.

**⑩ 이중 사용(Dual-Use) 우려**

논문이 명시하듯, 자동 프롬프트 최적화는 유용한 용도와 악의적 용도 모두에 사용될 수 있습니다. 특히 정교한 탈옥(jailbreak) 프롬프트나 허위 정보 생성에 악용될 가능성이 있습니다. 미래 연구에서는 **윤리적 가드레일(ethical guardrails)**을 최적화 프레임워크에 내장하는 방법을 고려해야 합니다.

**⑪ 다국어(Multilingual) 및 교차 언어 일반화**

현재 실험은 영어 중심 벤치마크에 한정됩니다. 프롬프트 최적화가 다른 언어로 일반화될 수 있는지, 혹은 언어별 별도 최적화가 필요한지에 대한 연구가 필요합니다.

---

## 📊 종합 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                         APEX 핵심 요약                           │
├─────────────────────────────────────────────────────────────────┤
│ 문제: 정적 데이터 취급 → 데이터 효율성 병목                        │
│ 해결: 동적 계층화 (Easy/Hard/Mixed) + 프론티어 타겟팅             │
│ 혁신: 데이터 자체를 최적화 변수로 간주                             │
│ 성과: +11.2% (Gemini 2.5 Flash), +6.8% (Gemma 3 27B)          │
│ 의의: 데이터 중심 프롬프트 최적화 패러다임 확립                     │
│ 한계: 이진 평가 가정, 대표 데이터셋 의존, 텍스트 한정              │
│ 미래: 멀티모달, 에이전트, 인퍼런스 스케일링과의 결합               │
└─────────────────────────────────────────────────────────────────┘
```

APEX의 가장 근본적인 기여는 단순히 새로운 알고리즘을 제안한 것이 아니라, **"최적화 과정에서 데이터의 가치는 고정되어 있지 않다"**는 관점 전환을 이루었다는 것입니다. 이 통찰은 프롬프트 최적화를 넘어 블랙박스 LLM 최적화 전반에 걸쳐 중요한 설계 원칙으로 자리잡을 것으로 예상됩니다.
