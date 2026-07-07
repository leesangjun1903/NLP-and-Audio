# Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 LLM 출력물 평가의 병목 문제를 해결하기 위해 **BINEVAL(Binary Evaluation)** 프레임워크를 제안합니다. 핵심 주장은 다음과 같습니다:

> "하나의 총체적 판단을 요청하는 대신, 모델에게 작고 검증 가능한 이진(yes/no) 질문들의 집합을 물어라."

기존 평가 방식의 세 가지 문제점을 지적합니다:

1. **인간 평가**: 비용이 높고 느림
2. **어휘적 지표** (ROUGE, BLEU, BERTScore): 개방형 생성에서 인간 판단과 상관관계 낮음
3. **전체론적 LLM 판사** (G-Eval, AlpacaEval 등): 불투명한 점수로 디버깅 어려움

### 주요 기여 4가지

| 기여 | 내용 |
|------|------|
| 해석 가능한 평가 프레임워크 | 평가 기준을 원자적 yes/no 질문으로 분해하는 태스크 무관 모듈형 방법 |
| 훈련 없이 강력한 성능 | SummEval, Topical-Chat, QAGS에서 학습된 평가기와 동등하거나 우월한 성능 |
| 반복적 프롬프트 개선 | 질문 수준 피드백을 통한 평가기 및 생성기 프롬프트 양방향 최적화 |
| 디버깅 가능한 점수 | 각 점수가 개별 판정과 설명에 근거하여 평가기 동작 진단 용이 |

---

## 2. 해결 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**평가 병목(Evaluation Bottleneck)**: 현대 LLM은 유창하고 맥락에 맞는 텍스트를 잘 생성하지만, 그 출력물을 정확하고 해석 가능하게 평가하는 것이 어렵습니다. 특히 반복적 개발(iterative development) 과정에서 단일 스칼라 점수는 실패 원인을 특정하지 못하는 문제가 있습니다.

**Ceiling Effect 문제**: G-Eval, UniEval 같은 기존 방법들은 표면적으로 그럴듯한 텍스트에 높은 점수를 부여하는 경향이 있어, 사실적 오류가 있는 요약에도 만점을 부여하는 사례가 발생합니다.

---

### 2.2 제안 방법 및 수식

BINEVAL은 세 단계로 구성됩니다.

#### 단계 1: 이진 질문 생성 (Binary Question Generation)

태스크 프롬프트 $T$를 받아 원자적 이진 질문 집합을 생성하는 분해 함수를 정의합니다:

$$\mathcal{Q} = \mathcal{F}_{\text{LLM}}(T; M) = \{q_1, q_2, \ldots, q_N\}$$

여기서 $M$은 메타-프롬프트(meta-prompt)이며, 두 단계의 분해를 수행합니다:

- **Step 1 – 요약(Summarize)**: 태스크 프롬프트 $T$를 명시적 요구사항 집합으로 변환:
$$\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$$

- **Step 2 – 분해(Decompose)**: 각 요구사항 $r_k$에 대해 "yes"가 충족, "no"가 위반을 나타내는 이진 질문 생성

질문들은 평가 차원 $\mathcal{D}$에 따라 분할됩니다:

$$\mathcal{Q} = \bigcup_{d \in \mathcal{D}} \mathcal{Q}_d$$

여기서 $\mathcal{Q}_d$는 차원 $d$에 특화된 질문들의 집합입니다.

#### 단계 2: 이진 평가 및 점수화 (Binary Evaluation and Scoring)

평가 LLM $E$, 입력 $x$, 출력 $y$, 이진 질문 $q_i$에 대해 이진 평가 함수를 정의합니다:

$$f_E(x, y, q_i) \in \{0, 1\}$$

여기서 $f_E(x, y, q_i) = 1$이면 "yes" (기준 충족), $0$이면 "no" (위반)입니다.

**차원별 점수**:

$$S_d(x, y) = \frac{1}{|\mathcal{Q}_d|} \sum_{q_i \in \mathcal{Q}_d} f_E(x, y, q_i)$$

**전체 점수**:

$$S(x, y) = \frac{1}{N} \sum_{i=1}^{N} f_E(x, y, q_i)$$

두 점수 모두 $[0, 1]$ 범위에 속하며, 기존 평가 척도와의 비교를 위해 $[a, b]$로의 어파인 스케일링이 가능합니다:

$$S'(x, y) = S(x, y) \cdot (b - a) + a$$

#### 단계 3-A: 교차 모델 프롬프트 업데이트 (Cross-Model Prompt Update)

소스 평가기 $E_{\text{src}}$와 타겟 평가기 $E_{\text{tgt}}$ 간의 이진 질문 불일치를 활용합니다. 반복 $t$에서:

**평가**:

$$A_j^{\text{src}} = \{f_{E_{\text{src}}}(x_j, y_j, q_i)\}_{i=1}^N$$

$$A_j^{\text{tgt}} = \{f_{E_{\text{tgt}}}(x_j, y_j, q_i; P_E^{(t-1)})\}_{i=1}^N$$

**불일치 식별**:

$$\Delta_j = \{q_i \in \mathcal{Q} : A_j^{\text{src}}(q_i) \neq A_j^{\text{tgt}}(q_i)\}$$

**레슨 추출**:

$$\mathcal{L}_j = L_{\text{note}}(x_j, y_j, A_j^{\text{src}}, A_j^{\text{tgt}}, \Delta_j)$$

**의미론적 중복 제거**:

$$\text{Dedup}(\ell_{\text{new}}, \mathcal{M}) = \begin{cases} \text{merge}(\ell_{\text{new}}, \ell_k), & \text{if } \ell_{\text{new}} \sim \ell_k \\ \text{add}(\ell_{\text{new}}), & \text{otherwise} \end{cases}$$

**프롬프트 업데이트**:

$$P_E^{(t)} \leftarrow P_E^{(t)}\text{.replace}(s_k, s'_k)$$

**수렴 조건** (모든 차원 $d \in \mathcal{D}$에 대해):

$$|S_d^{\text{tgt},(t)} - S_d^{\text{src}}| < \epsilon \quad \forall d \in \mathcal{D}$$

#### 단계 3-B: 자기 프롬프트 업데이트 (Self Prompt Update)

생성기 LLM $L_G$의 프롬프트 $P_G^{(t)}$를 반복적으로 개선합니다:

**생성**:
$$y_j^{(t)} = L_G(x_j; P_G^{(t)})$$

**평가 및 실패 질문 수집**:
$$\mathcal{E}_j = \{(q_i, e_i) : f_E(x_j, y_j^{(t)}, q_i) = 0\}$$

여기서 $e_i$는 평가기의 실패 설명입니다.

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                      BINEVAL Framework                   │
├─────────────────────────────────────────────────────────┤
│  [Phase 1] Binary Question Generation                    │
│    Task Prompt T → Meta-Prompt M → Q = {q₁,...,qₙ}     │
│    Step 1: T → Requirements R = {r₁,...,rₖ}            │
│    Step 2: R → Binary Questions Q (with violation ex.)  │
├─────────────────────────────────────────────────────────┤
│  [Phase 2] Binary Evaluation & Scoring                   │
│    For each (x, y, qᵢ): fE(x,y,qᵢ) ∈ {0,1} + eᵢ      │
│    Per-dimension: Sd(x,y) = mean over Qd                │
│    Overall: S(x,y) = mean over all N questions          │
├─────────────────────────────────────────────────────────┤
│  [Phase 3] Iterative Prompt Optimization                 │
│    ┌─────────────────┬─────────────────────────────┐   │
│    │  Self-Update    │  Cross-Model Update          │   │
│    │  (Generator PG) │  (Evaluator Esrc → Etgt)    │   │
│    └─────────────────┴─────────────────────────────┘   │
│    Note-taker → Lesson Extraction → Dedup → Rewrite     │
└─────────────────────────────────────────────────────────┘
```

**사용 모델**: gpt-oss-120b, Claude Sonnet 4 (temperature=0, 2회 평균)

---

### 2.4 성능 향상

#### SummEval (Spearman $\rho$ / Kendall $\tau$)

| 방법 | 일관성(Consistency) | 평균(Average) |
|------|-------------------|--------------|
| ROUGE-1 | 0.160 / 0.130 | 0.192 / 0.150 |
| BERTScore | 0.110 / 0.090 | 0.225 / 0.175 |
| UniEval (T5) | 0.446 / 0.371 | 0.474 / 0.377 |
| G-Eval (GPT-4) | 0.507 / 0.425 | 0.514 / 0.418 |
| **BINEVAL (Claude)** | **0.655 / 0.615** | **0.563 / 0.491** |

#### Topical-Chat: BINEVAL (Claude) 평균 Spearman $\rho = 0.632$로 최고 성능

#### QAGS (사실적 일관성): BINEVAL (Claude) 평균 $r/\rho/\tau = 0.604/0.620/0.534$

#### 반복적 프롬프트 업데이트 결과 (SummEval)

| 차원 | 기준선 | 자기-업데이트 최고 | 향상 | 교차-모델 최고 | 향상 |
|------|--------|-------------------|------|--------------|------|
| 일관성 | .477 | .568 | +.091 | .637 | **+.136** |
| 유창성 | .255 | .375 | **+.119** | .318 | +.072 |
| 평균 | .440 | .515 | +.075 | .520 | +.070 |

#### 분해가 효과적인 세 가지 메커니즘

1. **복잡성 감소 (Complexity Reduction)**: 단일 다면적 판단을 단순한 하위 문제들로 분해
2. **집계를 통한 분산 감소 (Variance Reduction via Aggregation)**: $N$개의 약하게 상관된 이진 분류기를 집계하면 분산이 $1/N$에 비례하여 감소
3. **실패 모드 커버리지 (Coverage of Failure Modes)**: 기준의 명시적 열거를 강제하여 전체론적 판단 대비 재현율 향상

---

### 2.5 한계

| 한계 | 설명 |
|------|------|
| **질문 품질 의존성** | 중요한 기준이 생성된 질문에서 누락되면 최종 점수도 이를 놓침 |
| **선형성 가정** | 충족된 질문 비율이 전체 품질에 선형적으로 매핑된다는 가정이 항상 성립하지 않음 |
| **관련성(Relevance) 차원의 어려움** | 지나친 분해가 인간의 전체론적 판단과 괴리를 초래할 수 있음 (Spearman $\rho$가 .505→.357로 하락한 실패 사례 보고) |
| **계산 비용 증가** | 단일 전체론적 판단 대비 모델 호출 수와 처리 텍스트 양이 증가 |
| **모델 능력 한계** | 계산적 제약(카운팅, 비율 추적)에서는 프롬프트 개선이 능력 한계를 극복하지 못함 (IFBench count 정확도: 63%→31% 하락) |
| **프롬프트 과부하(Prompt Bloat)** | 반복적 업데이트 시 레슨 누적으로 인한 지시 충돌 발생 가능성 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 태스크 무관성(Task-Agnosticism)을 통한 일반화

BINEVAL의 가장 강력한 일반화 근거는 **메타-프롬프트 $M$이 태스크 무관하다**는 점입니다:

$$\mathcal{Q} = \mathcal{F}_{\text{LLM}}(T; M)$$

동일한 메타-프롬프트 $M$이 태스크 프롬프트 $T$만 변경하여 요약, 대화, 지시 따르기 등 어떤 태스크에서도 적절한 이진 질문을 자동 생성합니다. 이는 **추가 훈련이나 태스크별 파인튜닝 없이** 새로운 도메인에 즉시 적용 가능함을 의미합니다.

### 3.2 다양한 벤치마크에 걸친 일반화 증거

| 벤치마크 | 태스크 유형 | BINEVAL 성능 |
|---------|------------|-------------|
| **SummEval** | 요약 평가 (4차원) | 평균 Spearman 0.563으로 최고 |
| **Topical-Chat** | 대화 평가 (4차원) | 평균 Spearman 0.632로 최고 |
| **QAGS** | 사실적 일관성 평가 | 평균 Spearman 0.620으로 최고 |
| **IFBench** | 지시 따르기 생성 | Format +17pp, Sentence +17pp 향상 |

SummEval → Topical-Chat으로의 전이는 요약 도메인의 분해 전략이 대화 도메인에서도 효과적임을 보여줍니다.

### 3.3 분산 감소를 통한 강건성

$N$개의 약하게 상관된 이진 분류기를 집계할 때 분산은 $1/N$에 비례하여 감소합니다. 논문의 phi-coefficient 분석에서:

- **관련성(Relevance)**: 평균 $\phi = 0.20$ (쌍의 80%가 $|\phi| < 0.3$)
- **일관성(Coherence)**: 평균 $\phi = 0.28$ (쌍의 64%가 $|\phi| < 0.3$)

이 낮은 상호 상관관계는 각 질문이 서로 다른 측면을 독립적으로 포착함을 의미하며, 집계 시 분산이 크게 감소하여 **더 안정적이고 일반화된 평가 신호**를 제공합니다.

### 3.4 백본 모델 교체에 대한 강건성

같은 백본(gpt-oss)으로 비교할 때:

$$\text{BINEVAL(gpt-oss)} > \text{G-Eval(gpt-oss)} > \text{UniEval(gpt-oss)}$$

이는 분해 전략 자체가 특정 모델에 의존하지 않고 일반화됨을 보여줍니다. 더 강력한 모델(Claude)과 결합 시 성능이 추가로 향상되므로, **평가기 모델이 발전할수록 BINEVAL의 성능도 자연스럽게 향상**됩니다.

### 3.5 교차 모델 업데이트를 통한 모델 마이그레이션 일반화

교차 모델 업데이트는 **모델 마이그레이션 시나리오**에서의 일반화에 특히 유용합니다. 더 강력한 소스 모델($E_{\text{src}}$)의 판단 기준을 약한 타겟 모델($E_{\text{tgt}}$)로 전이할 때, 이진 질문 불일치 $\Delta_j$가 정확히 어떤 기준이 일관성 없이 판단되는지 식별합니다. 일관성(Consistency) 차원에서 +0.136의 가장 큰 향상을 달성하며, 이는 **시스템적 평가 편향의 교차-모델 전이 교정** 가능성을 시사합니다.

### 3.6 한계: 일반화가 어려운 경우

- **주관적 판단이 본질적으로 전체론적인 경우** (예: 관련성): 이진 분해가 지나치게 엄격해져 인간 판단과 괴리 발생
- **계산 능력 한계가 있는 경우** (예: 정확한 카운팅, 비율 추적): 프롬프트 개선이 모델의 실행 능력 한계를 극복하지 못함
- **프롬프트 과부하**: 반복 횟수가 증가할수록 레슨이 축적되어 이전에 효과적이었던 지시들과 충돌 발생

---

## 4. 2020년 이후 최신 연구 비교 분석

### 4.1 평가 방법론 비교

| 방법 | 연도 | 접근법 | 장점 | 단점 |
|------|------|--------|------|------|
| **BERTScore** (Zhang et al.) | 2020 | 임베딩 기반 의미 매칭 | 어휘 중복보다 강건 | 사실적 정확성 포착 어려움 |
| **UniEval** (Zhong et al.) | 2022 | Boolean QA 형태로 재형식화 + T5 파인튜닝 | 다차원 평가 | 태스크별 훈련 필요 |
| **G-Eval** (Liu et al.) | 2023 | CoT + Likert 척도 | GPT-4 활용 강력한 성능 | 불투명한 점수, Ceiling Effect |
| **FActScore** (Min et al.) | 2023 | 원자적 사실 분해 후 개별 검증 | 세밀한 사실적 정확성 | 생성 콘텐츠를 분해 (평가 기준 X) |
| **Prometheus 2** (Kim et al.) | 2024 | 오픈소스 전문 평가 LLM | 독점 모델 수준의 판단 | 특정 평가 형식에 의존 |
| **ARES** (Saad-Falcon et al.) | 2024 | RAG 평가를 위한 분해 | RAG 특화 | 일반 평가에 직접 적용 어려움 |
| **RAGAS** (Es et al.) | 2024 | RAG 자동 평가 | 검색 증강 생성 특화 | 도메인 특수성 |
| **BINEVAL** (Cho et al.) | 2026 | 평가 기준 자체를 이진 질문으로 분해 | 태스크 무관, 훈련 불필요, 해석 가능, 반복 개선 지원 | 계산 비용, 질문 품질 의존 |

### 4.2 BINEVAL vs. FActScore: 분해 패러다임의 차이

FActScore는 **생성된 콘텐츠**를 원자적 사실들로 분해한 후 검증하는 반면, BINEVAL은 **평가 기준(criteria) 자체**를 이진 질문으로 분해합니다. 이 차이는 중요합니다:

- FActScore: "이 출력에서 사실 $f_i$가 지지되는가?" (콘텐츠 중심 분해)
- BINEVAL: "기준 $r_k$가 충족되는가?" (평가 기준 중심 분해)

BINEVAL의 접근법은 임의의 평가 차원(유창성, 일관성, 참여도 등)에 적용 가능하므로 더 범용적입니다.

### 4.3 BINEVAL vs. DSPy/OPRO: 프롬프트 최적화 관점

| 방법 | 최적화 신호 | 방법론 |
|------|------------|--------|
| **OPRO** (Yang et al., 2023) | 전체 정확도 | LLM을 최적화기로 사용하여 프롬프트 반복 생성 |
| **DSPy** (Khattab et al., 2023) | 태스크 메트릭 | 선언적 파이프라인 자체 개선 |
| **MIPRO** (Opsahl-Ong et al., 2024) | 검증 메트릭 | 지시 및 예시에 대한 베이지안 탐색 |
| **BINEVAL** | 이진 질문 불일치 | 불일치 기반 세밀한 레슨 추출 및 프롬프트 수정 |

BINEVAL의 차별점은 **불일치 신호가 어떤 기준이 일관성 없이 판단되는지를 정확히 식별**하여, 전체 점수 차이보다 훨씬 세밀하고 실행 가능한 최적화 신호를 제공한다는 것입니다.

### 4.4 JudgeBiasBench와의 관련성

JudgeBiasBench (Zhou et al., 2026)는 LLM 판사의 편향(위치, 장황함, 자기 향상 편향)을 분류하고 디바이어싱 전략을 제안합니다. BINEVAL은 각 이진 질문을 독립적으로 평가함으로써 이러한 편향의 일부를 구조적으로 완화합니다. 단, 이진 질문 자체가 기반 LLM의 편향을 상속할 수 있으므로 완전한 해결책은 아닙니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

#### (1) 평가 패러다임의 전환

BINEVAL은 **"판단(Judgment)" 패러다임에서 "질문(Questioning)" 패러다임으로의 전환**을 촉진합니다. 향후 연구는 단일 점수 대신 원자적 기준들의 집합으로 평가를 설계하는 방향으로 이동할 가능성이 높습니다.

#### (2) 자동화된 평가-개선 루프

평가기와 생성기 프롬프트를 동시에 개선하는 **폐쇄 루프(closed-loop) 자동화**의 가능성을 입증합니다. 이는 다음 연구 방향을 열어줍니다:

- 에이전틱(agentic) 시스템의 자기 진단 및 개선
- 멀티-턴 대화 시스템의 자동 품질 관리
- 지속적 배포 환경에서의 모델 모니터링

#### (3) 해석 가능한 AI 평가의 표준화

각 실패가 구체적인 질문에 매핑되는 BINEVAL의 구조는 **AI 시스템 감사(auditing)** 에 직접 활용 가능한 프레임워크를 제공합니다. 고위험(high-stakes) 애플리케이션에서의 LLM 신뢰성 평가 연구에 중요한 기반이 됩니다.

#### (4) 분해 기반 평가의 이론적 기반 강화

분산 감소($1/N$), 커버리지 향상, 복잡성 감소의 세 메커니즘 분석은 **왜 분해가 효과적인가**에 대한 이론적 기반을 제공하며, 향후 최적 질문 수, 질문 다양성 설계에 관한 연구를 자극할 것입니다.

---

### 5.2 향후 연구 시 고려할 점

#### (1) 질문 품질 최적화 연구

현재 BINEVAL은 질문 생성 자체를 LLM에 위임하지만, **어떤 질문이 더 좋은 평가 신호를 제공하는가**에 대한 연구가 필요합니다. 고려 사항:

- **Yes-rate 분산**: 너무 쉽거나(yes-rate ≈ 1.0) 너무 어려운(yes-rate ≈ 0.0) 질문은 판별력이 낮음
- **phi-계수($\phi$)**: 질문 간 상관관계가 낮을수록 독립적인 측면을 포착
- **질문의 최적 개수**: 너무 많으면 계산 비용 증가, 너무 적으면 커버리지 감소

#### (2) 주관적 품질 차원에 대한 적응적 분해

관련성(Relevance) 차원의 실패 사례는 **분해가 항상 유익하지 않음**을 보여줍니다. 향후 연구에서는:

- 차원의 "전체론적 정도"를 사전에 측정하는 방법 개발
- 주관적 차원에는 소프트(soft) 분해, 객관적 차원에는 엄격한 분해를 적용하는 **적응적 전략** 필요

#### (3) 프롬프트 업데이트의 안정성

IFBench 실험에서 반복 4회차에 성능이 급격히 하락한 사례는 **프롬프트 최적화의 안정성 문제**를 제기합니다:

- Early stopping 전략의 정교화
- 레슨 중요도 가중치 도입
- 상충하는 레슨 간의 충돌 해소 메커니즘

#### (4) 계산 비용 효율화

현재 질문별 평가를 병렬화하거나 배치 처리하는 방법이 제한적으로 논의되었습니다. 향후 연구에서:

- 질문 답변 배치화(batching) 최적화
- 중요한 질문만 선택적으로 평가하는 **적응적 평가** 전략
- 작은 모델로 이진 질문을 답변하고 강력한 모델로 검증하는 **계층적 평가**

#### (5) 에이전틱 및 멀티-턴 환경으로의 확장

논문은 단일 출력 평가에 초점을 맞추지만, 실제 응용에서는:

- **에이전틱 시스템**: 다단계 작업 실행에서 어느 단계가 실패했는지 추적
- **멀티-턴 대화**: 대화 히스토리 전반에 걸친 일관성 및 목표 달성도 평가
- **도구 호출(Tool Calling)**: 함수 호출의 정확성과 결과 활용 평가

#### (6) 이진 판정의 불확실성 모델링

현재 BINEVAL은 $f_E(x, y, q_i) \in \{0, 1\}$의 확정적 이진 판정을 사용합니다. 향후에는:

$$f_E(x, y, q_i) \in [0, 1]$$

로 확장하여 판정의 불확실성(예: "부분적으로 만족")을 모델링하는 **퍼지 이진 평가** 연구가 의미 있을 것입니다.

#### (7) 인간-AI 협업 평가 파이프라인

BINEVAL의 질문 수준 피드백은 인간 검토자가 특정 질문에만 집중할 수 있게 하는 **능동 학습(active learning)** 기반의 인간-AI 협업 평가 파이프라인 설계에 활용 가능합니다.

#### (8) 편향 분석 및 공정성

LLM 기반 평가기의 편향이 이진 질문 형식에서도 여전히 존재할 수 있으므로, JudgeBiasBench (Zhou et al., 2026)의 방법론을 BINEVAL에 적용하여 **이진 질문 판정에서의 편향 패턴** 분석이 필요합니다.

---

## 참고문헌

**논문 원본**
- Cho, S., Chawla, K., Cai, P., Liu, Z., Zhu, C., Zhang, S.-X., & Sahu, S. (2026). *Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement*. arXiv:2606.27226v1. Accepted to the 2nd Workshop on Compositional Learning at ICML 2026.

**논문 내 인용 주요 참고문헌**
- Zhong, M., et al. (2022). *Towards a unified multi-dimensional evaluator for text generation*. EMNLP 2022. [UniEval]
- Liu, Y., et al. (2023). *G-Eval: NLG evaluation using GPT-4 with better human alignment*. EMNLP 2023. [G-Eval]
- Min, S., et al. (2023). *FActScore: Fine-grained atomic evaluation of factual precision in long form text generation*. EMNLP 2023.
- Kim, S., et al. (2024). *Prometheus 2: An open source language model specialized in evaluating other language models*. EMNLP 2024.
- Fabbri, A. R., et al. (2021). *SummEval: Re-evaluating summarization evaluation*. TACL 9:391–409. [평가 벤치마크]
- Wang, A., Cho, K., & Lewis, M. (2020). *Asking and answering questions to evaluate the factual consistency of summaries*. ACL 2020. [QAGS]
- Mehri, S., & Eskenazi, M. (2020). *USR: An unsupervised and reference free evaluation metric for dialog generation*. ACL 2020. [Topical-Chat]
- Khattab, O., et al. (2023). *DSPy: Compiling declarative language model calls into self-improving pipelines*. arXiv:2310.03714.
- Yang, C., et al. (2023). *Large language models as optimizers*. arXiv:2309.03409. [OPRO]
- Zhou, H., et al. (2026). *Toward robust LLM-based judges: Taxonomic bias evaluation and debiasing optimization*. arXiv:2603.08091. [JudgeBiasBench]
- Zhang, T., et al. (2020). *BERTScore: Evaluating text generation with BERT*. ICLR 2020.
- Saad-Falcon, J., et al. (2024). *ARES: An automated evaluation framework for retrieval-augmented generation systems*. arXiv:2311.09476.
- Es, S., et al. (2024). *RAGAS: Automated evaluation of retrieval augmented generation*. EACL 2024.
- Opsahl-Ong, K., et al. (2024). *Optimizing instructions and demonstrations for multi-stage language model programs*. EMNLP 2024. [MIPRO]
- Pyatkin, V., et al. (2025). *Generalizing verifiable instruction following*. arXiv:2507.02833. [IFBench]
