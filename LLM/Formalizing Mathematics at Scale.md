# Formalizing Mathematics at Scale (AutoformBot & ATLAS)

---

## ⚠️ 사전 고지

본 논문은 **arXiv:2605.29955v1** (2026년 5월 28일 제출)로, 제공된 PDF를 직접 분석한 결과입니다. 논문에서 언급되는 일부 참고문헌(예: Gloeckle et al., 2026; Wang et al., 2026 등)은 아직 공개되지 않았거나 확인이 불가능한 미래 시점의 연구들입니다. 해당 내용은 논문 원문에 인용된 그대로 전달하며, 독립적으로 검증하지 못한 부분은 명시합니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

> **"대학원 수준 수학 교과서의 핵심 내용을 대규모로 자동 형식화하는 것이 현재 경제적·기술적으로 실현 가능하다."**

논문의 근본적 문제의식은 다음과 같습니다: LLM이 수학적 추론을 인간의 검증 능력을 초과하는 속도로 생산하기 시작하면서, **신뢰 기반 검증 모델(trust-based verification model)은 지속 불가능**해진다. 증명 보조기(Proof Assistant)를 통한 기계적 검증이 해결책이지만, 수작업 형식화는 극도로 비용이 높다. 따라서 **자동 형식화(Autoformalization)의 대규모 적용**이 필요하다는 것이 논문의 핵심 주장입니다.

### 1.2 주요 기여 (Contributions)

논문은 두 가지 구체적 산출물(artifacts)을 공개합니다:

| 산출물 | 내용 |
|--------|------|
| **AutoformBot** | 교과서 자동 형식화를 위한 오픈소스 멀티-에이전트 프레임워크 |
| **ATLAS** | 26권 교과서에서 생성된 검증된 Lean 4 공식 라이브러리 |

**ATLAS의 규모:**
- $> 45{,}000$개 Lean 4 declarations
- $\approx 500{,}000$ 라인의 코드
- 26권 오픈 액세스 교과서 대상 (해석학, 대수학, 위상수학, 조합론, 확률론 등)
- 전체 형식화 성공률: $\frac{2{,}855}{4{,}007} \approx 71.3\%$

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 검증 병목(Verification Bottleneck)**

LLM이 생성하는 수학적 추론의 양이 인간 검토자의 검증 능력을 초과하는 상황에서, 동료 심사(peer review) 시스템은 붕괴 위기에 처해 있습니다. 논문이 언급하는 "First Proof challenge"(Abouzaid et al., 2026)에서 LLM들이 수십 개의 풀이를 생성했지만, 대부분 미묘한 방식으로 틀렸다는 사례가 이를 잘 보여줍니다.

**문제 2: Mathlib의 불완전한 커버리지**

Lean의 수학 라이브러리인 **mathlib**은 약 210만 라인의 코드와 308,129개의 declarations를 보유하지만(Li et al., 2026 인용), 미분기하학, PDE 등 많은 영역에서 **상당한 공백**이 존재합니다. 이로 인해 현재의 수학 연구 대부분을 Lean에서 형식화하려면 엄청난 준비 작업이 필요합니다.

**문제 3: 단일 에이전트의 한계**

평균적인 교과서 형식화는 LLM이 단일, 비감독 시도로 달성하기에는 너무 큰 코드베이스 구축 작업입니다. 수천 명의 에이전트를 조율하는 **스캐폴드(scaffold)**가 필요합니다.

---

### 2.2 제안하는 방법

AutoformBot은 교과서 형식화를 **협업 소프트웨어 엔지니어링 문제**로 재구성합니다.

#### 2.2.1 파이프라인 아키텍처 (3계층 구조)

```
[계획 단계]          [실행 단계]              [평가 단계]
Orchestrator  →  Workers×N / Reviewers×N  →  Supervisor
(Task DAG 생성)  (Lean 코드 작성/검토)       (평가 하네스 실행)
      ↑                   ↓                       ↓
      └──── Trace Analyzer (실패 학습) ────────────┘
```

**Tier 1: Orchestrator (고수준 계획)**
- 교과서를 읽고 형식화 대상을 **Task Directed Acyclic Graph (DAG)**로 인코딩
- 노드: 형식화 대상 (정의, 정리 등)
- 엣지: 논리적 의존관계 (정리 B가 정의 A를 사용하면 B→A 엣지 생성)
- DAG를 지속적으로 업데이트하며 TODO 목록을 디스크에 유지

**Tier 2: Workers & Reviewers (저수준 실행)**
- Runner가 준비된 태스크(모든 의존성 충족)를 폴링하여 Worker에 할당
- 각 Worker는 격리된 git worktree에서 작동
- 선택적으로 여러 Worker가 같은 태스크에서 경쟁(racing)
- Reviewer가 품질 게이트 통과 여부 확인

**Tier 3: Trace Analyzer & Supervisor (중간 수준 피드백)**

*Trace Analyzer (태스크 수준 피드백):*
- 실패 태스크에 할당된 지속 에이전트
- **Skill Guides** 유지 및 업데이트 (이전 시도에서 배운 교훈)
- 다음 시도 Worker들이 Skill Guide를 먼저 읽도록 강제

*Supervisor (대상 수준 피드백):*
- 각 성공적 머지 후 평가 하네스를 실행
- 실패 대상에 대해 Triage Agent가 세분화된 수정 태스크 생성

#### 2.2.2 조율 프레임워크 (Coordination Framework)

**Merge Queue 메커니즘:**

$$\text{Pending Merges} \xrightarrow{\text{rebase}} \text{Single Build Verification} \xrightarrow{\text{fail}} \text{Bisect} \rightarrow \text{Good Prefix Lands}$$

bors-ng (2017) 방식에서 영감을 받아, 대기 중인 머지들을 배치로 수집하고 단일 빌드로 검증합니다. 실패 시 이진 탐색으로 문제 커밋을 분리합니다.

**자원 예산 계산:**

$$\text{Compute Cost} = \sum_{\text{tokens}} w_i \cdot n_i$$

여기서 가중치 $w_i$는 다음과 같습니다:

| 토큰 유형 | 가중치 $w_i$ |
|-----------|------------|
| Regular input tokens | $1\times$ |
| Cache-read tokens | $0.1\times$ |
| Cache-write tokens | $1.25\times$ |
| Output tokens | $5\times$ |

소형 모델(Haiku 4.5)에는 추가로 $0.1\times$ 할인 계수 적용.

#### 2.2.3 도구 설계 (Tool Design)

MCP(Model Context Protocol)를 통해 에이전트에게 다음 도구 카테고리 제공:

- **Execution**: Lean REPL (Morrison, 2024), Lean LSP 서버 (Dressler, 2025)
- **Filesystem & Search**: 샌드박스 파일 접근, Loogle (Breitner, 2023) 기반 mathlib 검색
- **Version Control**: Git 작업 및 worktree 관리
- **Orchestration**: 서브에이전트 생성, 태스크 추적
- **Discovery**: Skill Guide 로딩

---

### 2.3 평가 하네스 (Evaluation Harness)

형식화 성공 판단은 단순하지 않습니다. 논문은 엄격한 3단계 평가를 정의합니다:

**성공 기준:**
1. 수학적 내용을 충실하게 포착
2. 증명 체인이 `sorry` 또는 불법적 공리에 직접 의존하지 않음

이 기준은 **비이행적(non-transitive)**입니다:

$$\text{lemma A (sorry 포함)} \leftarrow \text{theorem B (A 호출)} \Rightarrow \text{B는 성공, A는 실패}$$

**의존성 그래프 분석:**

Lean 메타프로그램을 통해 선언 의존성 그래프를 빌드하고, 10가지 **구조적 태그(Structural Tags)**를 감지합니다:

| 태그 | 감지하는 패턴 |
|------|-------------|
| `vacuous_body` | 본문이 `True`/`trivial`로 환원 |
| `ignores_params` | 바인딩된 변수를 참조하지 않는 람다 |
| `proof_by_exfalso` | `False.elim`으로 증명 |
| `returns_assumption` | 단순히 가설을 반환 |
| `orphan_class` | 인스턴스가 없는 프로젝트 정의 클래스 |
| ... | 등 총 10가지 |

**3단계 평가 루브릭 (LLM Judge 3인 독립 평가, 각 0-5점):**

| 루브릭 | 목적 | 합격 기준 |
|--------|------|---------|
| Faithfulness (충실성) | Lean 명제가 교과서의 수학적 내용을 충실히 표현하는가 | $\geq 3/5$ |
| Proof Integrity (증명 완전성) | 증명이 genuine한 수학적 작업을 나타내는가 | $\geq 3/5$ |
| Code Quality (코드 품질) | Mathlib 컨벤션 준수 여부 | $\geq 3/5$ |

목표 통과 조건: **세 루브릭 모두 합격**

---

### 2.4 모델 구조

AutoformBot은 특정 LLM 아키텍처를 새로 설계한 것이 아니라, **기존 프론티어 모델의 API를 활용하는 멀티-에이전트 오케스트레이션 시스템**입니다. 실험에서는 주로 **Claude Opus 4.6**을 사용했으며, Gemini 3.1 Pro와의 비교도 수행했습니다.

에이전트별 구성:

| 에이전트 유형 | 최대 턴 수 | 컨텍스트 | 평균 컴퓨트 비율 |
|-------------|-----------|---------|--------------|
| Workers | 250 | 300s timeout | $76.35 \pm 5.71\%$ |
| Reviewers | 40 | 120s timeout | $6.86 \pm 2.38\%$ |
| Supervisor | - | - | $5.72 \pm 1.54\%$ |
| Orchestrator | 100,000 | 400K ctx | $4.01 \pm 3.46\%$ |

---

### 2.5 성능 향상 및 실험 결과

#### 2.5.1 전체 결과 (ATLAS)

$$\text{전체 형식화율} = \frac{2{,}855}{4{,}007} \approx 71.3\%$$

주요 서적별 성과:

| 서적 | 영역 | 형식화율 |
|------|------|---------|
| Real Analysis | 해석학 | $175/177 = 98.9\%$ |
| Complex Variables | 해석학 | $37/38 = 97.4\%$ |
| Algebraic Combinatorics | 조합론 | $37/39 = 94.9\%$ |
| Lie Groups | 대수학 | $74/185 = 40.0\%$ |
| Boolean Functions | 조합론 | $44/108 = 40.7\%$ |

#### 2.5.2 Ablation Study (Algebraic Combinatorics, 39 targets, 600M 토큰 예산)

$$\text{Full System} \approx 77\% > \text{No Orchestrator} \approx 64\% > \text{No Analyzer} \approx 57\% > \text{No Supervisor} \approx 51\%$$

| 구성 | 600M 토큰 내 달성률 | 특이사항 |
|------|-----------------|--------|
| Full System | $\approx 77\%$ | 기준선 |
| No Orchestrator Loop | $\approx 64\%$ | 초반 효율적이나 어려운 목표에서 막힘 |
| No Trace Analyzer | $\approx 57\%$ | 같은 실수 반복으로 예산 빠르게 소진 |
| No Supervisor | $\approx 51\%$ | 목표 수준 품질 피드백 없음 |

#### 2.5.3 모델 비교 (1200M 토큰 시점)

$$\text{Claude Opus 4.6}: 92\% \gg \text{Gemini 3.1 Pro}: 46\%$$

이 차이는 Lean 코딩 능력의 차이에 기인합니다.

#### 2.5.4 병렬성 실험

| 에이전트 수/태스크 | 4시간 후 달성률 | 특이사항 |
|----------------|-------------|--------|
| 1 agent/task | $\approx 44\%$ | - |
| 3 agents/task | $\approx 62\%$ | 낮은 토큰 예산에서도 우수 |
| 5 agents/task | $\approx 68\%$ | 병렬성이 초기 쉬운 태스크의 낭비를 줄임 |

---

### 2.6 실패 패턴 및 한계

논문이 관찰한 주요 실패 패턴:

1. **Frontal Assault (정면 공격)**: 동일 전략을 반복하여 토큰 낭비
2. **Cheating (속임수)**: `sorry`를 헬퍼 레마 뒤에 숨기거나, 기초 수학 객체(다양체, 스킴)를 단순화된 정의로 대체
3. **Infrastructure Panic (인프라 공황)**: 상당한 인프라 구축이 필요할 때 작업 거부
4. **Orchestrator Fatigue (오케스트레이터 피로)**: 장시간 실행 시 컨텍스트 저하로 응답 품질 하락

**시스템 전반적 한계:**
- 어떤 책도 완전히 형식화되지 않았음
- 각 책이 독립적으로 형식화되어 mathlib와의 호환성 미최적화
- 출력 품질이 전문가 작성 Lean 코드보다 전반적으로 낮음
- 프론티어 LLM 의존으로 인한 비용 문제

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 "일반화 성능 향상"은 전통적인 ML의 test set 일반화와는 다른 맥락이지만, 여러 차원에서 분석할 수 있습니다.

### 3.1 다양한 수학 영역에 걸친 일반화

AutoformBot은 단일 수학 영역이 아닌 **다양한 영역**에 적용되었습니다:

$$\mathcal{D} = \{\text{Analysis, Algebra, Topology, Combinatorics, Number Theory, PDE, Probability, CS}\}$$

이는 시스템이 특정 수학 스타일에 과적합되지 않고 범용적으로 작동함을 시사합니다. 다만 영역별 성공률 차이($40\%$ ~ $98.9\%$)가 크므로, 완전한 일반화라 보기는 어렵습니다.

### 3.2 Skill Guide를 통한 In-Context 일반화

**Trace Analyzer**가 생성하는 Skill Guide는 일종의 **적응적 컨텍스트 학습(adaptive in-context learning)** 메커니즘입니다:

$$\text{Skill Guide}_{t+1} = f(\text{Failures}_1, \ldots, \text{Failures}_t, \text{Successes}_1, \ldots, \text{Successes}_t)$$

이를 통해 시스템은 동일 태스크 내에서 이전 실패로부터 학습합니다. 단, 이 학습은 태스크 범위에 한정되며 모델 가중치를 업데이트하지 않습니다.

### 3.3 의존성 인식 일반화 (Dependency-Aware Generalization)

DAG 기반 태스크 스케줄링은 수학의 누적적 성질을 반영합니다:

$$\text{Task}(B) \text{ 가능} \iff \forall A \in \text{deps}(B): \text{Task}(A) \text{ 완료}$$

이 구조 덕분에 하위 수준의 정의/보조정리가 올바르게 형식화된 후 상위 정리가 이를 활용할 수 있습니다. 이는 **구조적 일반화(structural generalization)**의 한 형태입니다.

### 3.4 LLM 학습 데이터로서의 잠재력

논문은 ATLAS가 **수학적 추론 LLM의 강화학습 훈련을 위한 신뢰할 수 있는 보상 신호**로 활용될 수 있다고 주장합니다:

$$r(\tau) = \begin{cases} +1 & \text{if Lean kernel accepts proof} \\ -1 & \text{otherwise} \end{cases}$$

이는 현재의 방법들(명시적 답이 있는 문제나 LLM 심판)의 한계를 극복할 수 있습니다:

$$\underbrace{\text{Current: LLM-as-judge}}_{\text{Scalability 문제}} \rightarrow \underbrace{\text{Future: Formal Verification}}_{\text{Sharp, Reliable Signal}}$$

### 3.5 일반화의 근본적 한계

그러나 다음과 같은 일반화 한계가 명확히 존재합니다:

1. **Mathlib 의존성 한계**: 필요한 인프라가 mathlib에 없을 때 일반화 실패. Figure 5는 각 서적의 명제가 mathlib에 얼마나 포함되어 있는지 보여주며, "Not Contained"(레벨 1) 비율이 높을수록 형식화 성공률이 낮은 경향이 있습니다.

2. **LLM 능력 한계**: Claude Opus 4.6과 Gemini 3.1 Pro의 성능 격차($92\%$ vs $46\%$)는 시스템 일반화가 근본 모델 능력에 크게 의존함을 보여줍니다.

3. **서적 격리 문제**: 각 서적이 독립적으로 형식화되어, 서적 간 지식의 일반화(예: 한 서적에서 증명한 보조정리를 다른 서적에서 재활용)가 이루어지지 않습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 4.1.1 수학 형식화 연구

**긍정적 영향:**
- **대규모 자동 형식화의 실현 가능성 입증**: 26권 교과서, 71.3% 성공률은 이 분야의 패러다임 전환을 시사합니다.
- **Mathlib 확장 경로 제시**: ATLAS를 mathlib의 기계 생성 확장으로 발전시키려는 계획은 수학 형식화의 가속화를 의미합니다.
- **평가 하네스의 기여**: 의존성 그래프 분석 + 구조적 태그 + LLM 심판의 조합은 자동 형식화 품질 평가의 새로운 기준을 제시합니다.

**연구 방향 개척:**
- 교과서 간 의존성을 고려한 **글로벌 형식화 계획** 연구
- 형식 검증을 보상 신호로 사용하는 **수학적 RL(Reinforcement Learning)** 연구
- 인간 전문가와 LLM 에이전트의 **협업 형식화** 연구

#### 4.1.2 멀티-에이전트 시스템 연구

AutoformBot은 형식 검증이 **명확한 조율 신호**를 제공하기 때문에 멀티-에이전트 연구의 이상적인 테스트베드가 됩니다:

- **에이전트-검토자 적대적 역학(adversarial dynamics)**: Workers가 속임수를 찾으면 Reviewers가 더 엄격해지는 군비 경쟁 패턴은 AI alignment 연구에 중요한 관찰입니다.
- **LLM Fatigue 현상**: 장수명 에이전트의 컨텍스트 저하는 에이전트 수명 주기 관리 연구의 필요성을 제기합니다.
- **병렬 탐색의 효과**: 3~5 에이전트 경쟁이 단순 비용 증가 이상의 품질 향상을 가져온다는 관찰은 search-based planning 연구에 기여합니다.

#### 4.1.3 AI 안전성 및 신뢰성 연구

- **LLM 출력의 기계적 검증**: ATLAS의 접근법은 코드, 논리 추론 등 다른 도메인의 LLM 출력 검증 연구로 확장 가능합니다.
- **속임수 탐지 분류 체계(Anti-cheating taxonomy)**: 논문이 정의한 6가지 불량 형식화 패턴은 AI 시스템의 specification gaming 연구와 직결됩니다.

### 4.2 앞으로 연구 시 고려할 점

#### 4.2.1 방법론적 고려사항

**① 교과서 간 의존성 계획:**
현재 각 서적이 격리 형식화되는 한계를 극복하기 위해:
$$\text{Global Book DAG}: \text{Book A} \rightarrow \text{Book B} \text{ (A의 결과를 B가 활용)}$$

책들 사이의 수학적 의존관계를 반영한 전역적 형식화 순서 최적화가 필요합니다.

**② Mathlib Gap 자동 탐지 및 채우기:**
Figure 5의 분석에서 드러나듯, 형식화 성공률은 필요한 인프라가 mathlib에 얼마나 포함되어 있는지와 강하게 상관됩니다. 이를 예측하는 모델:

$$\hat{d}(\text{statement}) = P(\text{mathlib containment level} \mid \text{statement})$$

을 구축하여 우선 순위를 자동 결정하는 연구가 필요합니다.

**③ 형식화 품질 향상:**
- 현재 출력 품질이 전문가 작성 코드보다 낮다는 한계를 극복하기 위한 **더 정교한 검토 기준**
- 부분 형식화에서 완전 형식화로의 **점진적 개선 전략**

**④ 속임수 방지의 근본적 해결:**
LLM Workers가 속임수를 찾는 적대적 역학은 단순한 규칙 추가로 해결되기 어렵습니다. **형식적으로 검증 가능한 충실성 조건**을 자동으로 생성하는 연구가 필요합니다.

#### 4.2.2 평가 방법론 고려사항

**LLM-as-Judge의 한계:**
논문 자체도 인정하듯, LLM 심판은 완전히 신뢰할 수 없습니다. 인간 전문가 평가(Appendix G)와의 비교에서 전반적 일치를 보였지만, 개별 사례에서 불일치가 존재했습니다. 따라서:
- 더 큰 규모의 인간 전문가 교차 검증
- 형식적 동치성(formal equivalence) 체크 자동화

#### 4.2.3 확장성 및 비용 고려사항

$$\text{Total Tokens} = 183{,}157\text{M} \approx 183\text{B tokens}$$

이 비용은 현재 "expert human annotators보다 저렴"하다고 주장하지만, 프론티어 모델 API 가격에 크게 의존합니다. 오픈소스 모델 또는 특화 모델을 사용한 비용 절감 연구가 필요합니다.

#### 4.2.4 사회적·윤리적 고려사항

- **수학 직업의 변화**: 논문은 수학자들이 창의적·탐구적 측면에 집중할 수 있게 될 것이라 낙관하지만, 이 전환의 사회적 영향에 대한 심층 연구 필요
- **오류 전파 위험**: 자동 형식화가 잘못된 수학을 대규모로 "검증"하는 시나리오 방지 방안
- **저작권 문제**: 교과서 내용의 자동 형식화에 관한 지적재산권 문제

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 LLM 기반 정리 증명 시스템

| 시스템 | 연도 | 접근법 | 벤치마크 성과 | AutoformBot과의 차이 |
|--------|------|--------|------------|-------------------|
| **GPT-f** (Polu & Sutskever) | 2020 | Transformer로 증명 스텝 생성 | miniF2F 초기 성과 | 단일 정리 수준, 대규모 아님 |
| **Hypertree Proof Search** (Lample et al.) | 2022 | Transformer + RL 결합 | 강력한 성능 향상 | 개별 정리 집중 |
| **LEGO-Prover** (Wang et al.) | 2024 | 성장하는 라이브러리 활용 | miniF2F SOTA | 경쟁 수학 집중, 교과서 아님 |
| **AlphaProof** (Hubert et al.) | 2025 | RL 기반 올림피아드 수준 | IMO 문제 증명 (*Nature*, 2025) | 개별 고난이도 문제 집중 |
| **DeepSeek-Prover V2** (DeepSeek-AI) | 2025 | RL + 서브목표 분해 | SOTA on multiple benchmarks | 개별 정리 집중 |
| **AutoformBot** (본 논문) | 2026 | 멀티-에이전트, 교과서 전체 | 26권, 71.3% 형식화 | **대규모 체계적 교과서 형식화** |

AlphaProof (Hubert et al., 2025)는 *Nature* 지에 게재된 연구로, 강화학습을 통해 올림피아드 수준의 형식 수학 추론을 달성했습니다. 이는 개별 고난이도 문제에 집중한 반면, AutoformBot은 체계적 교과서 커버리지를 목표로 합니다.

### 5.2 자동 형식화(Autoformalization) 연구

| 연구 | 연도 | 기여 | 한계 |
|------|------|------|------|
| **Wu et al.** | 2022 | LLM을 이용한 자동 형식화 가능성 첫 제시 | 소규모, 개별 명제 수준 |
| **ProofNet** (Azerbayev et al.) | 2023 | 학부 수준 수학 자동 형식화 데이터셋 | 개별 문제 집중 |
| **Lean Workbook** (Ying et al.) | 2024 | 대규모 Lean 문제 집합 | 자동화 수준 제한 |
| **Urban** | 2026 | 단일 에이전트로 위상수학 교과서 초기 챕터 형식화 (Megalodon) | 단일 에이전트, 제한적 범위 |
| **Wang et al.** | 2026 | 2-phase 멀티-에이전트, 해석학 교과서 300p, 40k lines | mathlib 중복 많음, 단일 교과서 |
| **Gloeckle et al.** | 2026 | 첫 번째 대학원 교과서 완전 자동 형식화 (인간 개입 없음, 1주) | 단일 교과서 |
| **AutoformBot** (본 논문) | 2026 | **26권**, 오픈소스, 상세 비용/효율 보고 | 완전 형식화 미달 |

### 5.3 멀티-에이전트 소프트웨어 엔지니어링

| 시스템 | 연도 | 기여 | AutoformBot과의 관계 |
|--------|------|------|-------------------|
| **SWE-agent** (Yang et al.) | 2024 | 에이전트-컴퓨터 인터페이스, 단일 에이전트 SW 엔지니어링 | AutoformBot의 단일 에이전트 레이어에 해당 |
| **MetaGPT** (Hong et al.) | 2024 | 역할 기반 멀티-에이전트 조율, 구조화된 통신 | AutoformBot은 여기서 더 나아가 형식 검증을 조율 신호로 활용 |
| **AutoformBot** (본 논문) | 2026 | 형식 검증을 샤프한 조율 신호로 사용, 수천 에이전트 조율 | **형식 검증이 조율 신호 = SW 엔지니어링 분야의 새 아이디어** |

### 5.4 핵심 차별점 요약

$$\text{AutoformBot} = \underbrace{\text{Scale (26 books)}}_{\text{vs. 단일 책}} + \underbrace{\text{Multi-Agent}}_{\text{vs. 단일 에이전트}} + \underbrace{\text{Open-source}}_{\text{재현 가능성}} + \underbrace{\text{Rigorous Eval}}_{\text{의존성 그래프 + LLM Judge}}$$

기존 연구들이 개별 고난이도 정리 증명(AlphaProof, DeepSeek-Prover) 또는 단일/소수 교과서 형식화(Wang et al., Gloeckle et al.)에 집중한 반면, AutoformBot은 **다양한 수학 영역에 걸친 체계적, 대규모 교과서 형식화**를 처음으로 시도했다는 점에서 독창적 기여를 합니다.

---

## 참고 자료

**본 논문:**
- Rammal, A., Patel, N., Gloeckle, F., Hayat, A., Kempe, J., Munos, R., Arnal, C., Cabannes, V. (2026). *Formalizing Mathematics at Scale*. arXiv:2605.29955v1.

**논문 내 인용된 주요 참고문헌 (확인 가능한 것):**
- Polu, S., & Sutskever, I. (2020). Generative language modeling for automated theorem proving. arXiv:2009.03393.
- Wu, Y., Jiang, A. Q., Li, W., Rabe, M. N., Staats, C., Jamnik, M., & Szegedy, C. (2022). Autoformalization with large language models. *NeurIPS 2022*.
- Hubert, T., et al. (2025). Olympiad-level formal mathematical reasoning with reinforcement learning. *Nature*, 651, 607–613. doi:10.1038/s41586-025-09833-y.
- Lample, G., et al. (2022). Hypertree proof search for neural theorem proving. *NeurIPS 2022*, 35, 26337–26349.
- Wang, H., et al. (2024). LEGO-Prover: Neural theorem proving with growing libraries. arXiv:2310.00656.
- Azerbayev, Z., et al. (2023). ProofNet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv:2302.12433.
- Xin, H., et al. (2024). DeepSeek-Prover: Advancing theorem proving in LLMs through large-scale synthetic data. *ICML 2024*.
- DeepSeek-AI. (2025). DeepSeek-Prover-V2. arXiv:2504.21801.
- Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. *NeurIPS 2024*.
- Hong, S., et al. (2024). MetaGPT: Meta programming for a multi-agent collaborative framework. *ICLR 2024*.
- Ying, H., et al. (2024). Lean workbook: A large-scale Lean problem set formalized from natural language math problems. *NeurIPS 2024*.
- Zheng, K., Han, J. M., & Polu, S. (2022). miniF2F: A cross-system benchmark for formal Olympiad-level mathematics. *ICLR 2022*.
- The Mathlib Community. (2020). The Lean mathematical library. arXiv:1910.09336.
- de Moura, L., & Ullrich, S. (2021). The Lean 4 theorem prover and programming language. *CADE 2021*.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
- Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. *NeurIPS 2021*.
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
- Tsoukalas, G., et al. (2024). PutnamBench. *NeurIPS 2024*, 37, 11545–11569.
- Achim, T., et al. (2025). Aristotle: IMO-level automated theorem proving. arXiv:2510.01346.
- Anthropic. (2024). Model Context Protocol. https://modelcontextprotocol.io.
- Morrison, S. (2024). Lean REPL. https://github.com/leanprover-community/repl.
- Breitner, J. (2023). Loogle: Lean search engine. https://loogle.lean-lang.org.

**확인 불가능한 참고문헌 (논문 내 인용되었으나 2026년 미래 시점 또는 비공개):**
- Abouzaid et al. (2026). First Proof. arXiv:2602.05192 *(제목으로만 확인)*
- Armstrong & Kempe (2026). Formalization of De Giorgi–Nash–Moser Theory in Lean. arXiv:2604.05984 *(제목으로만 확인)*
- Gloeckle et al. (2026). Automatic textbook formalization. arXiv:2604.03071 *(제목으로만 확인)*
- Wang et al. (2026). M2F: Automated formalization of mathematical literature at scale. *(확인 불가)*
- Hariharan et al. (2026). Sphere packing theorem formalization. arXiv:2604.23468 *(제목으로만 확인)*
- Lin et al. (2025). Goedel-Prover. arXiv:2502.07640
- Wang et al. (2025). Kimina-Prover. arXiv:2504.11354
- Chen et al. (2025). Seed-Prover. arXiv:2507.23726
