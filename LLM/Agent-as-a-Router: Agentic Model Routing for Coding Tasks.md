# Agent-as-a-Router: Agentic Model Routing for Coding Tasks 

---

## 📌 참고 자료

- **주요 논문**: Zhou, Pengfei, et al. "Agent-as-a-Router: Agentic Model Routing for Coding Tasks." arXiv:2606.22902v3 [cs.AI], 26 Jun 2026. (본문에 첨부된 PDF 전체)
- **관련 인용 논문들** (논문 내 References 기반):
  - Ong et al., "RouteLLM: Learning to route LLMs from preference data." ICLR 2025. [26]
  - Li et al., "LLMRouterBench: A massive benchmark and unified framework for LLM routing." arXiv:2601.07206, 2026. [21]
  - Li, Lihong et al., "A contextual-bandit approach to personalized news article recommendation." WWW 2010. [22]
  - Aggarwal et al., "Automix: Automatically mixing language models." NeurIPS 2024. [1]
  - Chen et al., "FrugalGPT: How to use large language models while reducing cost and improving performance." TMLR. [10]
  - Ding et al., "Hybrid LLM: Cost-efficient and quality-aware query routing." arXiv:2404.14618, 2024. [12]
  - Šakota et al., "Fly-swat or cannon? cost-effective language model choice via meta-modeling." WSDM 2024. [33]
  - Shnitzer et al., "Large language model routing with benchmark datasets." CoLM 2024. [34]
  - Yue et al., "MasRouter: Learning to route LLMs for multi-agent systems." ACL 2025. [42]
  - Jimenez et al., "SWE-Bench." ICLR 2024. [18]
  - Hu et al., "LoRA." ICLR 2022. [16]
  - Lattimore & Szepesvári, "Bandit Algorithms." Cambridge University Press, 2020. [20]

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

이 논문의 핵심 주장은 **"LLM 라우터의 성능 병목은 추론 능력(reasoning)의 부재가 아니라 정보 결핍(information deficit)이다"** 는 것이다.

기존 정적(static) 라우터들은 배포 시점에 정보 상태가 고정되어 있어, 실행 기반(execution-grounded) 피드백을 축적할 수 없다. 이를 해결하기 위해 논문은 라우팅을 **Context→Action→Feedback(C-A-F)** 루프로 형식화한 **Agent-as-a-Router** 프레임워크를 제안한다.

### 🏆 주요 기여 3가지

| 기여 유형 | 내용 |
|-----------|------|
| **Framework** | C-A-F 루프 기반 Agent-as-a-Router 프레임워크 제안, 누적 후회(Cumulative Regret)를 스트리밍 평가 지표로 형식화 |
| **Artifacts** | ACRouter 구현체(Orchestrator + Verifier + Memory) 및 CodeRouterBench(~10K 태스크, 8 LLMs, 실행 검증) 공개 |
| **Findings** | 정보 제공만으로 +15.3% 상대적 성능 향상; ACRouter가 ID/OOD 모두에서 최저 누적 후회 달성; 정적 학습 라우터의 OOD 일반화 실패 확인 |

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

**문제 배경**: 실사용 환경에서 사용자는 여러 LLM 제공자를 동시에 구독하며, 각 모델은 서로 다른 코딩 차원에서 강점을 지닌다. 따라서 "어떤 모델이 이 태스크를 처리해야 하는가?"라는 자동 라우팅 문제가 중요해진다.

**기존 방법의 한계**:
- 기존 라우터들은 정적(static) 분류 문제로 취급 → 배포 중 경험 축적 불가
- Vanilla LLM-as-a-Router(Claude Sonnet 4.6): AvgPerf% **41.41** → Oracle: **57.00** (큰 격차)
- 진단 실험에서 +Perf stats 추가만으로 **47.74** 달성 → **+15.3%** 상대적 향상

### 2.2 제안 방법 및 수식

#### C-A-F 루프 형식화

라우터는 모델 풀 $\mathcal{M} = \{m_1, \ldots, m_M\}$과 태스크 스트림 $\mathcal{T} = (t_1, \ldots, t_N)$에 대해 다음 루프를 반복한다:

$$c_i \xrightarrow{\text{Decide}} a_i \xrightarrow{\text{Execute}} f_i \xrightarrow{\text{Memorize}} c_{i+1} \tag{1}$$

- **Context** $c_i = (p_i, d_i, \mathcal{H}\_{ < i})$: 현재 태스크의 프롬프트 $p_i$, 메타데이터 $d_i$, 이전 루프들에서 축적된 메모리 상태 $\mathcal{H}_{<i}$
- **Action** $a_i \in [M]$: 선택된 모델의 인덱스
- **Feedback** $f_i = (\hat{s}_i, \hat{\kappa}_i)$: 검증된 성능 점수 $\hat{s}_i \in [0,1]$와 비용 $\hat{\kappa}_i$

#### 보상 함수

사용자 지정 가중치 $\epsilon_1 > 0$, $\epsilon_2 < 0$ 하에서 태스크별 보상:

$$r_i(a_i) = \epsilon_1 s_i(a_i) + \epsilon_2 \kappa_i(a_i) \tag{2}$$

정책 $\pi$의 평균 보상:

$$V(\pi) = \frac{1}{N}\sum_{i=1}^{N} r_i(a_i) = \epsilon_1 \frac{1}{N}\sum_{i=1}^{N} s_i(a_i) + \epsilon_2 \frac{1}{N}\sum_{i=1}^{N} \kappa_i(a_i) \tag{3}$$

#### 결과 행렬 및 보상 행렬

전체 결과 행렬 $O \in \mathbb{R}^{N \times M \times 2}$, $O_{ij} = (s_{ij}, \kappa_{ij})$로부터:

$$R_{ij} = \epsilon_1 s_{ij} + \epsilon_2 \kappa_{ij} \quad \text{for } i \in [N],\ j \in [M] \tag{4}$$

#### 태스크별 Oracle 및 누적 후회

$$a_i^* = \arg\max_{j \in [M]} R_{ij}, \quad r_i^* = \max_{j \in [M]} R_{ij}, \quad \forall i = 1,\ldots,N \tag{5}$$

$$V^* = \frac{1}{N}\sum_{i=1}^{N} r_i^* = \frac{1}{N}\sum_{i=1}^{N} \max_{j \in [M]} R_{ij} \tag{6}$$

$$\text{CumReg}_N(\pi) = \sum_{i=1}^{N} \delta_i = N\bigl(V^* - V(\pi)\bigr) \tag{7}$$

여기서 $\delta_i = r_i^* - r_i(a_i) \geq 0$는 단일 태스크 후회.

#### Verifier 점수 집계

$$u_i = \sum_{k \in \mathcal{K}_{d(t_i)}} w_{d(t_i),k} \cdot \hat{s}_k(a_i, t_i) \tag{8}$$

여기서 $\mathcal{K}\_{d(t_i)}$는 검증 도구 집합(AST 파싱, 샌드박스 실행 등), $\sum_{k} w_{d(t_i),k} = 1$.

### 2.3 ACRouter 모델 구조

```
┌─────────────────────────────────────────────┐
│              C-A-F Loop                     │
│                                             │
│  ┌──────────────┐    ┌──────────────┐       │
│  │ Orchestrator │    │   Verifier   │       │
│  │(C → A)       │    │ (A → F)      │       │
│  │              │    │              │       │
│  │ Qwen3.5-0.8B │    │ AST parser   │       │
│  │ (fine-tuned) │    │ Sandbox exec │       │
│  │ + kNN top-10 │    │ LLM-as-Judge │       │
│  │ + DimBest    │    │ Prompt tests │       │
│  │ + weighted   │    │              │       │
│  │   voting     │    │ → score u_i  │       │
│  └──────┬───────┘    └──────┬───────┘       │
│         │ Action            │ Feedback      │
│  ┌──────▼───────────────────▼──────────┐    │
│  │           Memory (F → C)            │    │
│  │  Online vector store (FIFO, 20K)    │    │
│  │  Key: task embedding (voyage-code-3 │    │
│  │       or BGE-large)                 │    │
│  │  Value: model, perf, cost, traces   │    │
│  │  Retrieval: cosine kNN, k=10        │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
         ↕ Tool Layer
┌─────────────────────────────────────────────┐
│  Candidate Models (8 LLMs)                  │
│  Routing Policy | Retrieval Support         │
│  Evaluation Tools | Sandbox/Infra           │
└─────────────────────────────────────────────┘
```

**3개 핵심 모듈**:

| 모듈 | 역할 | 구현 |
|------|------|------|
| **Orchestrator** | 정보 통합 → 라우팅 결정 | Qwen3.5-0.8B (fine-tuned) + 휴리스틱 가중치 투표 |
| **Verifier** | 실행 기반 피드백 생성 | AST 파싱 + 샌드박스 실행 + LLM-as-Judge |
| **Memory** | 경험 누적 | FIFO bounded 온라인 벡터 스토어 (코사인 kNN) |

### 2.4 성능 향상

#### In-Distribution (n=2,919)

| Router | AvgPerf%↑ | CumReg↓ | Perf/$↑ |
|--------|-----------|---------|---------|
| Oracle | 57.00 | 0 | 8.20 |
| **ACRouter (ours)** | **49.98** | **205.5** | 3.79 |
| LinUCB | 46.84 | 296.9 | 4.38 |
| DimensionBest | 47.50 | 277.4 | 3.69 |
| LogReg | 47.26 | 284.4 | 6.27 |
| Always-Opus 4.6 | 43.83 | 387.1 | 1.29 |

#### OOD Test (n=176, Agentic Programming)

| Router | AvgPerf%↑ | CumReg↓ | Perf/$↑ |
|--------|-----------|---------|---------|
| Oracle | 75.89 | 0 | 2.32 |
| **ACRouter (ours)** | **62.50** | **17.0** | 1.18 |
| Qwen3.5-0.8B-FT | 55.36 | 27.2 | 0.74 |
| Always-Opus 4.6 | 57.14 | 26.7 | 0.64 |
| LogReg | 19.64 | 61.8 | 1.17 |
| RouteLLM-MF | 8.93 | 72.7 | 0.94 |

**주요 발견**: 정적 학습 라우터(LogReg, RouteLLM 등)는 OOD에서 AvgPerf가 **8.93%~21.43%**로 급락하여 Random(31.25%)보다도 낮음.

### 2.5 한계

1. **비용 추정의 불확실성**: 공급자 측 캐시 히트율이 관측 불가하여 금전적 비용은 공식 토큰 가격 기준의 근사치
2. **OOD 평가 제한**: agentic programming 평가 시 표준 250스텝 대신 40스텝 제한 적용 (예산 제약)
3. **Memory 구현의 한계**: 현재 LLM 정책 + kNN 앙상블 방식; 파라미터 수준의 고급 메모리 기법은 미탐색
4. **OOD 규모**: OOD 테스트가 176개 태스크로 비교적 소규모

---

## 3. 일반화 성능 향상 가능성 (심층 분석)

이 논문에서 **일반화**는 핵심 차별화 요소이다. 정적 라우터들이 OOD에서 붕괴하는 반면, ACRouter는 강건성을 유지한다.

### 3.1 일반화 실패의 원인 분석

**정적 라우터의 OOD 붕괴 메커니즘**:

$$\underbrace{\text{훈련 분포 (9개 단일 턴 차원)}}_{\text{단일 파일, 짧은 태스크}} \xrightarrow{\text{Distribution Shift}} \underbrace{\text{OOD (Agentic Programming)}}_{\text{다단계 계획, 파일 탐색, 반복 디버깅}}$$

TF-IDF, LogReg, RouteLLM 등은 프롬프트 피처에 조건화되어 있는데, 이 피처들이 OOD 분포에서 더 이상 유효한 라우팅 시그널을 전달하지 못한다.

### 3.2 ACRouter의 일반화 메커니즘

**핵심: 배포 중 정보 갭을 능동적으로 닫는 구조**

```
배포 초기 (OOD 진입 시)
├── Memory: 프로빙 셋에서 워밍업된 초기 상태
├── Orchestrator: DimBest prior + kNN으로 초기 결정
└── Verifier: 실행 결과 검증 → Memory 업데이트

배포 진행 (태스크 스트림 진행)
├── Memory: OOD 태스크들의 경험 누적
├── kNN 검색: OOD 태스크 내 유사 태스크 참조
└── 결정 품질: 경험 증가와 함께 지속적 향상
```

**분산 분해 분석** (논문 §D.3):
차원 정체성(dimension identity)은 Oracle 결정의 엔트로피 $H(y_t)$의 약 **27%**만 설명한다. 나머지 73%는 태스크별 콘텐츠 정보에 있으며, 이는 ACRouter의 **태스크 임베딩 기반 kNN**이 포착하는 세밀한 정보다.

$$I(y_t;\ d(t)) \approx 0.27 \cdot H(y_t)$$

DimensionBest가 ~47.5% AvgPerf를 달성하는 반면, Oracle이 57%에 도달하는 이유가 바로 이 73% 정보 차이에 있다.

### 3.3 온라인 밴딧과의 비교

온라인 밴딧(LinUCB, LinTS)도 OOD에서 일정 수준 생존(49.82% / 46.43%)하지만 ACRouter(62.50%)에 미치지 못하는 이유:

- **밴딧**: 팔(arm)별 선형 사후 분포 → **문맥 인식 추론 부재**
- **ACRouter**: Orchestrator(LLM 정책) + Memory(임베딩 kNN) → **태스크 콘텐츠의 세밀한 문맥 파악**

### 3.4 스케일링과 일반화의 관계 (Qwen 라우터 스케일링)

| 파라미터 크기 | AvgPerf%↑ | Gap%↓ |
|--------------|-----------|-------|
| 0.8B | 46.41 | 18.6 |
| 2B | 46.69 | 18.1 |
| 4B | 46.21 | 18.9 |
| 9B | 46.56 | 18.3 |
| 27B | 46.74 | 18.0 |

**~30배 파라미터 범위에서 AvgPerf 변화 ~0.5%포인트** → 라우터 용량(capacity)이 병목이 아님을 재확인. 일반화를 위해서는 더 큰 모델이 아닌 **동적 정보 메커니즘**(Memory + Verifier)이 필요함.

### 3.5 일반화 성능 향상의 미래 가능성

논문이 시사하는 일반화 향상 방향:

1. **콜드 스타트 개선**: 초기 워밍업 전략 최적화 (현재 200 val 태스크)
2. **임베딩 모델 선택**: voyage-code-3 vs BGE-large의 코드 도메인 특화 성능 비교 탐색
3. **계층적 Memory**: 차원 수준 + 태스크 수준의 이중 메모리 구조
4. **전이 가능한 Verifier**: OOD 태스크 유형에 맞는 검증 도구 자동 선택

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 LLM 라우팅 연구 계보

```
2021        2022        2023        2024               2025~2026
 │           │           │           │                    │
FrugalGPT  HybridLLM  LLM Routing  RouteLLM    LLMRouterBench  Agent-as-a-Router
 [Chen+]    [Ding+]    Benchmark    [Ong+]      [Li+,2026]      [Zhou+,2026]
             │          [Shnitzer]   │                           │
          비용 최적화   벤치마크 기반  선호도 학습 기반          C-A-F 루프
          이진 라우팅   정적 분류     정적 분류                  동적 적응
```

### 4.2 핵심 연구 비교표

| 연구 | 방법론 | 라우팅 유형 | 도메인 | 평가 지표 | 한계 |
|------|--------|------------|--------|----------|------|
| **FrugalGPT** (Chen+, TMLR) | 캐스케이드 라우팅, 비용 최적화 | 정적 | 일반 NLP | 비용-성능 | 코딩 미적용, 정적 |
| **RouteLLM** (Ong+, ICLR 2025) | 선호도 데이터 기반 분류기 학습 | 정적 훈련 정책 | 일반 NLP | 단일 정확도 | OOD 일반화 취약 |
| **LLM Routing with Benchmark Datasets** (Shnitzer+, CoLM 2024) | 벤치마크 성능 기반 메타모델 | 정적 | 일반 NLP | 단일 정확도 | 스트리밍 미지원 |
| **Automix** (Aggarwal+, NeurIPS 2024) | 자동 LLM 혼합 | 동적(제한적) | 일반 NLP | 성능-비용 | 코딩 미적용 |
| **LLMRouterBench** (Li+, arXiv 2026) | 21개 NLU 데이터셋, 33개 모델 평가 | 정적 비교 | 일반 NLP | 단일 정확도 | 코딩/OOD 미포함 |
| **MasRouter** (Yue+, ACL 2025) | 멀티에이전트 LLM 라우팅 | 정적 훈련 정책 | 멀티에이전트 | 태스크 성능 | 스트리밍 미지원 |
| **Hybrid LLM** (Ding+, 2024) | 품질-비용 쿼리 라우팅 | 정적 이진 | 일반 NLP | 비용-품질 | 이진 결정만 가능 |
| **Agent-as-a-Router (본 논문)** | **C-A-F 루프, 실행 기반 피드백** | **동적 적응** | **코딩 특화** | **누적 후회** | OOD 규모, 비용 추정 한계 |

### 4.3 핵심 차별점

**Agent-as-a-Router가 기존 연구와 다른 세 가지 핵심 차원**:

1. **평가 지표의 혁신**: 기존 연구들은 단일 정확도(single-shot accuracy)를 사용하지만, 본 논문은 **스트리밍 환경에서의 누적 후회(cumulative regret)**를 도입하여 라우터의 시간적 성능을 포착

2. **도메인 특화**: 기존 라우팅 연구들이 일반 NLP에 집중하는 반면, 본 논문은 **코딩 도메인에 특화된 9개 차원 + 1개 OOD 차원** 구성

3. **자기 진화 루프**: 기존 정적 라우터들과 달리 **배포 중 실행 검증 피드백을 통한 지속적 학습**을 구현

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구 영향

#### 패러다임 전환
이 논문은 모델 라우팅을 단순한 **분류 문제에서 지속적 학습 문제**로 재정의한다. C-A-F 루프는 AI 에이전트 시스템 설계의 일반적 원리로 확장 가능하다:

$$\text{Model Routing} \rightarrow \text{Tool Selection} \rightarrow \text{API Endpoint Selection} \rightarrow \text{Prompt Strategy Selection} \rightarrow \text{Effort Allocation}$$

#### 벤치마크 기여
CodeRouterBench는 향후 라우팅 연구의 **표준 평가 환경**이 될 가능성이 높다:
- 10개 차원, ~10K 태스크, 8개 프론티어 모델
- 실행 검증된 per-task per-model 결과 행렬 $O \in \mathbb{R}^{N \times M \times 2}$
- 스트리밍 환경에서의 누적 후회 평가 지원

#### LLM 에이전트 시스템 설계에 미치는 영향
- **단일 모델 에이전트의 한계** 공식화: 5개의 서로 다른 모델이 9개 차원에서 dimension-best를 차지함
- **정보 결핍 진단 프레임워크**: 라우터 성능 분석의 새 방법론 제시

### 5.2 앞으로의 연구에서 고려할 점

#### (A) 기술적 확장 방향

**1. 고급 Memory 아키텍처**
```
현재: FIFO bounded 온라인 벡터 스토어 (코사인 kNN)
미래:
  ├── 파라미터 수준 메모리 (e.g., 지속적 학습)
  ├── 계층적 메모리 (차원 수준 + 태스크 수준)
  ├── 망각 메커니즘 (오래된 경험의 가중치 감소)
  └── 교차 태스크 전이 학습
```

**2. Verifier 강화**

현재 Verifier는 AST 파싱, 샌드박스 실행, LLM-as-Judge를 조합하지만, 다음이 추가적으로 탐색 가능하다:

$$u_i^{\text{enhanced}} = \sum_{k \in \mathcal{K}_{d(t_i)}} w_{d(t_i),k} \cdot \hat{s}_k(a_i, t_i) + \lambda \cdot \text{Self-consistency}(a_i, t_i)$$

**3. 멀티 모달 라우팅**
코딩 이외 도메인(vision-language, multimodal coding)으로의 C-A-F 적용

**4. 계층적 라우팅**
단일 모델 선택이 아닌 **모델 앙상블 구성** 라우팅:

$$\pi(a_i | h_i) \in \Delta^{M-1} \rightarrow \pi(\{a_i^1, a_i^2, \ldots\} | h_i)$$

#### (B) 평가 프레임워크 개선

**OOD 테스트 확장**: 현재 176개 태스크는 통계적 신뢰성에 한계가 있음. 다양한 OOD 시나리오(도메인 이동, 시간적 이동, 언어 이동) 포함 필요.

**비용 모델 정교화**: 현재 공식 API 가격 기준이나, 실제 배포 환경에서는:
- 캐시 히트율 (provider-side)
- 배치 처리 할인
- 지역별 가격 차이
를 반영한 현실적 비용 모델 필요

#### (C) 연구 윤리 및 실용적 고려사항

**1. 모델 풀 변동성 문제**
LLM 생태계는 빠르게 변화하며, 새 모델 등장 시 과거 경험의 유효성이 저하될 수 있다. **온라인 망각(online forgetting)** 또는 **개념 이동(concept drift) 감지** 메커니즘이 필요하다.

**2. 공정성(Fairness) 문제**
Memory 기반 라우팅은 특정 모델에 대한 **편향된 경험**을 증폭시킬 위험이 있다. 예를 들어, 초기 탐색 불충분으로 인한 과소평가된 모델이 영구적으로 선택되지 않는 문제.

**3. 탐색-활용 균형 (Exploration-Exploitation Trade-off)**
현재 ACRouter는 주로 활용(exploitation) 중심이다. 밴딧 이론의 **UCB (Upper Confidence Bound)** 원리를 Orchestrator 수준에서 통합하면:

$$a_i = \arg\max_{j \in [M]} \left[ \hat{\mu}_j(h_i) + \beta \cdot \sigma_j(h_i) \right]$$

여기서 $\hat{\mu}_j$는 모델 $j$의 예상 성능, $\sigma_j$는 불확실성 추정치.

**4. 프라이버시 및 보안**
Memory에 저장되는 태스크 프롬프트와 실행 결과에는 민감한 코드가 포함될 수 있으므로, **차분 프라이버시(differential privacy)** 적용 또는 **연합 학습(federated learning)** 기반 분산 Memory 구조 탐색이 필요하다.

**5. 다중 사용자 환경에서의 Memory 공유**
현재 논문은 단일 사용자 스트리밍을 가정하지만, 다중 사용자 환경에서의 **공유 Memory vs. 개인화 Memory**의 트레이드오프 연구가 필요하다.

#### (D) 이론적 보완 필요 사항

**수렴 보장**: C-A-F 루프의 누적 후회가 태스크 수 $N$에 대해 sub-linear하게 증가함을 이론적으로 증명하는 연구 필요:

$$\mathbb{E}[\text{CumReg}_N(\pi_{\text{ACRouter}})] = O(N^{\alpha}), \quad \alpha < 1$$

현재는 실험적으로만 검증됨.

**정보 이론적 하한**: 어떤 온라인 라우터도 달성할 수 없는 최소 누적 후회의 하한(minimax lower bound) 도출 필요.

---

## 결론 요약

**Agent-as-a-Router**는 LLM 라우팅 문제를 정적 분류에서 **동적 적응 루프**로 패러다임을 전환한 중요한 연구다. 핵심 통찰인 "정보 결핍이 성능 병목"이라는 진단은 단순하지만 강력하며, C-A-F 루프라는 형식화는 라우팅을 넘어 일반적인 에이전트 의사결정 프레임워크로 확장 가능하다. 특히 OOD 일반화에서 정적 라우터들이 완전히 붕괴하는 반면 ACRouter가 강건함을 유지하는 결과는, 미래의 에이전트 시스템이 **배포 중 자기 진화(self-evolution) 능력**을 반드시 갖춰야 함을 강력히 시사한다.
