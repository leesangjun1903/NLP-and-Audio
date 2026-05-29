# EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

EverMemOS는 LLM 기반 장기 에이전트를 위한 **자기조직화 메모리 운영체제(Memory Operating System)** 로, 기존 메모리 시스템의 근본적 한계인 **단편적 경험의 고수준 의미 구조 통합 부재** 문제를 해결합니다.

> **핵심 테제:** 기존 시스템의 실패는 정보 부재가 아닌 **통합 부재(poor integration)** 에서 비롯된다.

### 주요 기여

| 기여 영역 | 내용 |
|-----------|------|
| **시스템 설계** | 메모리를 수동 저장에서 구조화된 경험 조직으로 재개념화한 Memory OS |
| **혁신적 방법론** | 단편적 에피소드 → 일관된 지식 구조로 변환하는 3단계 라이프사이클 |
| **실증 검증** | LoCoMo, LongMemEval, PersonaMem-v2 3개 벤치마크 SOTA 달성 |

**성능 향상 수치:**
- LoCoMo: 최강 기준선(Zep) 대비 **+9.2%** (GPT-4.1-mini 기준)
- LongMemEval: 최강 기준선(MemOS) 대비 **+6.7%**

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 정의:** LLM의 제한된 컨텍스트 윈도우와 기존 메모리 시스템의 구조적 한계

```
기존 시스템의 한계:
┌─────────────────────────────────────────┐
│ 단편적 레코드 저장 → 고립된 검색        │
│ 갈등 감지 실패 → 일관성 없는 추론       │
│ 사용자 상태 모델 유지 불가              │
│ "Lost-in-the-Middle" 현상               │
└─────────────────────────────────────────┘
```

**구체적 동기 사례 (Figure 2):**
- 사용자가 IPA 맥주를 좋아한다는 단편 정보 + 항생제 복용 중이라는 새 정보가 통합되지 않으면, 알코올 음료를 추천하는 실수 발생
- EverMemOS는 이를 통합하여 논알코올 대안을 안전하게 추천

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 메모리 기본 단위: MemCell

MemCell $c$는 다음과 같이 정의된 튜플입니다:

$$c = (E, \mathcal{F}, P, M)$$

- $E$ **(Episode):** 이벤트의 3인칭 서술 내러티브 (의미론적 앵커)
- $\mathcal{F} = \{f_1, \ldots, f_n\}$ **(Atomic Facts):** $E$에서 도출된 검증 가능한 원자적 사실들
- $P$ **(Foresight):** 유효 구간 $[t_{start}, t_{end}]$이 부여된 시간 한정 예측/계획
- $M$ **(Metadata):** 타임스탬프 및 소스 포인터

#### 2.2.2 Phase I: Episodic Trace Formation

상호작용 이력 스트림을 다음과 같이 정의합니다:

$$\mathcal{D} = \{d_1, \ldots, d_T\}$$

3단계 파이프라인으로 MemCell을 생성합니다:

**Step 1 - Contextual Segmentation:**
$$\text{Boundary Detector}: \mathcal{D} \xrightarrow{\text{sliding window}} \text{Raw Episode History}$$

**Step 2 - Narrative Synthesis:**
$$\text{Episode History} \xrightarrow{\text{LLM rewriting}} E \text{ (3인칭, 코참조 해소 완료)}$$

**Step 3 - Structural Derivation:**
$$E \xrightarrow{\text{LLM prompting}} \left(\mathcal{F}, P \text{ with } [t_{start}, t_{end}]\right) \rightarrow c = (E, \mathcal{F}, P, M)$$

#### 2.2.3 Phase II: Semantic Consolidation

**Incremental Semantic Clustering:**

새로운 MemCell $c$가 도착하면:

$$\text{sim}(c, \text{MemScene}_k) = \text{cosine}(\text{emb}(c), \mu_k)$$

$$\text{결정 규칙:} \begin{cases} c \to \text{MemScene}_k & \text{if } \max_k \text{sim}(c, \text{MemScene}_k) \geq \tau \\ \text{새 MemScene 생성} & \text{otherwise} \end{cases}$$

여기서 $\tau$는 클러스터링 임계값 (LoCoMo: $\tau = 0.70$, LongMemEval: $\tau = 0.50$)

**Scene-Driven Profile Evolution:**

$$\text{UserProfile} \xleftarrow{\text{online update}} \text{Scene Summaries} \quad \text{(개별 턴이 아닌 씬 요약 기반)}$$

프로파일은 다음 두 필드를 유지합니다:
- **Explicit facts:** 검증 가능한 속성 (시간 가변 측정값 포함)
- **Implicit traits:** 선호도 및 습관

#### 2.2.4 Phase III: Reconstructive Recollection

**MemScene Selection:**

쿼리 $q$에 대해 Reciprocal Rank Fusion(RRF)을 통한 하이브리드 검색:

$$\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}$$

MemScene 스코어링:

$$\text{score}(\text{MemScene}_j) = \max_{c_i \in \text{MemScene}_j} \text{relevance}(q, c_i)$$

**Foresight Filtering:**

시간 유효성 조건:

```math
P_{\text{valid}} = \left\{ p \in P \mid t_{now} \in [t_{start}, t_{end}] \right\}
```

**Agentic Verification and Query Rewriting:**

```math
\text{Context}^* = \begin{cases} \text{Retrieved Context} & \text{if Verifier}(q, \text{Context}) = \texttt{sufficient} \\ \text{Retrieve}(\text{RewrittenQueries}(q)) & \text{otherwise} \end{cases}
```

---

### 2.3 모델 구조

```
┌────────────────────────────────────────────────────────┐
│                    EverMemOS 아키텍처                   │
├────────────────────────────────────────────────────────┤
│  Phase I: Episodic Trace Formation                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 대화 스트림 D → [Semantic Boundary Detector]     │  │
│  │ → Raw Episode History → [Narrative Synthesis]   │  │
│  │ → Episode E → [Structural Derivation]           │  │
│  │ → MemCell c = (E, F, P, M)                      │  │
│  └──────────────────────────────────────────────────┘  │
│                          ↓                             │
│  Phase II: Semantic Consolidation                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ MemCells → [Incremental Clustering (τ)]         │  │
│  │ → MemScenes (주제별 군집)                        │  │
│  │ → [Scene-Driven Profile Evolution]              │  │
│  │ → UserProfile {Explicit, Implicit}              │  │
│  └──────────────────────────────────────────────────┘  │
│                          ↓                             │
│  Phase III: Reconstructive Recollection                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Query q → [Dense+BM25 RRF] → MemScene Selection │  │
│  │ → [Episode Rerank] → [Foresight Filter]         │  │
│  │ → [Sufficiency Check] → (충분? Yes:답변 생성)   │  │
│  │                        (No: Query Rewrite → 재검색)│  │
│  └──────────────────────────────────────────────────┘  │
│                          ↓                             │
│         Memory-Augmented Reasoning / Chat              │
└────────────────────────────────────────────────────────┘
```

**구현 세부사항:**
- **LLM 백본:** GPT-4.1-mini (또는 GPT-4o-mini)
- **임베딩:** Qwen3-Embedding-4B
- **희소 검색:** BM25
- **융합:** Reciprocal Rank Fusion (RRF)
- **재랭킹:** Qwen3-Reranker-4B
- **기본 하이퍼파라미터:** $N=10$ MemScenes, $K=10$ Episodes

---

### 2.4 성능 향상

#### LoCoMo 결과 (Table 1)

| 방법 | Single Hop | Multi Hop | Temporal | Overall |
|------|-----------|-----------|----------|---------|
| Zep | 90.84 | 81.91 | 77.26 | 85.22 |
| **EverMemOS** | **96.67** (+6.4%) | **91.84** (+12.1%) | **89.72** (+16.1%) | **93.05** (+9.2%) |

#### LongMemEval 결과 (Table 2)

| 방법 | SS-Asst | Know. Upd | Overall |
|------|---------|-----------|---------|
| MemOS | 67.86 | 74.26 | 77.80 |
| **EverMemOS** | **85.71** (+14.3%) | **89.74** (+20.6%) | **83.00** (+6.7%) |

#### Ablation Study (Figure 4)

| 구성 | LoCoMo | LongMemEval |
|------|--------|-------------|
| w/o EverMemOS | 0.52 | 5.00 |
| w/o MemCell | 81.82 | 71.20 |
| w/o MemScene | 89.16 | 79.60 |
| **EverMemOS (Full)** | **93.05** | **83.00** |

---

### 2.5 한계

논문에서 명시적으로 인정한 한계점들:

1. **단일 모달리티:** 텍스트 전용 평가 (MemCell/MemScene 추상화는 모달리티 독립적이나, 멀티모달/구현 설정 확장은 미구현)
2. **계산 비용:** LLM 매개 연산으로 인한 레이턴시 증가 (Phase I: ~9.42M 토큰, Phase III: ~10.27M 토큰 on LoCoMo)
3. **벤치마크 한계:** 초장기 타임라인 스트레스 테스트 프로토콜 부재
4. **End-to-end 효율성:** 캐싱/배치/비동기 처리 가능하나 개선 여지 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지지하는 구조적 특성

#### (1) 모달리티 독립적 추상화
MemCell $(E, \mathcal{F}, P, M)$ 구조는 특정 데이터 형식에 종속되지 않습니다. 텍스트를 넘어 이미지 캡션, 음성 전사, 센서 데이터로 확장 가능한 일반 프레임워크를 제공합니다.

#### (2) 백본 독립성
Table 3에서 GPT-4.1-mini와 Qwen3-4B 간 경계 감지 성능 차이가 $\leq 0.7$ 포인트로 매우 작습니다:

$$\Delta_{\text{accuracy}} \leq 0.7 \text{ points across backbone variants}$$

이는 시스템이 특정 LLM에 과적합되지 않음을 시사합니다.

#### (3) 의미론적 세그멘테이션의 우월성
Table 3에서 의미론적 세그멘테이션이 고정 휴리스틱보다 일관되게 우수하고, 오라클 세션 분할(Session Oracle)보다도 우수합니다:

| 세그멘테이션 방법 | GPT-4.1-mini | Qwen3-4B |
|----------------|-------------|---------|
| Fixed-Token-1024 | 84.52 | 75.19 |
| Session (Oracle) | 87.66 | 80.63 |
| **EverMemOS** | **89.16~89.78** | **82.73~83.07** |

이는 **데이터셋의 인위적 경계에 의존하지 않는 방법론**이 실제 배포 환경에서 더 잘 일반화됨을 보여줍니다.

#### (4) 하이퍼파라미터 견고성
Figure 5의 민감도 분석에서 $N=10$ 부근에서 성능이 포화되며, 이는 다양한 설정에서 안정적 성능을 보장합니다:

$$\text{Performance}(N) \approx \text{Performance}(N^*), \quad N \geq 10$$

#### (5) 정보 중복성 활용
Table 10에서 재현율이 0인 경우에도 12~20%의 질문에 정확히 답변:

$$P(\text{correct} \mid \text{zero recall}) \in [12\%, 20\%]$$

이는 관련 콘텐츠로부터의 추론 능력을 보여주며, 분포 외 쿼리에 대한 견고성을 시사합니다.

### 3.2 일반화 한계 및 개선 가능성

#### 현재 한계
- LoCoMo(10개 대화) 및 LongMemEval(500개 대화)에만 평가 → **데이터셋 다양성 제한**
- 클러스터링 임계값 $\tau$가 데이터셋별로 수동 조정 필요 ($\tau_{LoCoMo}=0.70$, $\tau_{LME}=0.50$)
- 사용자 수가 극적으로 증가하거나 도메인이 크게 변화하는 경우 미검증

#### 일반화 향상을 위한 제안
1. **적응형 임계값 학습:** $\tau$를 데이터 분포에 따라 자동 조정하는 메타학습 접근
2. **멀티모달 MemCell:** 이미지/음성을 포함한 멀티모달 에피소드 표현
3. **도메인 전이 평가:** 의료, 법률, 교육 등 전문 도메인에서의 추가 검증

---

## 4. 연구에 미치는 영향 및 향후 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 메모리 시스템 설계 패러다임 전환

EverMemOS는 **메모리 = 라이프사이클** 이라는 새로운 관점을 확립했습니다. 이전의 "저장-검색" 이분법에서 "형성-통합-재구성" 삼단계 모델로의 전환은 향후 모든 장기 에이전트 시스템 설계에 영향을 미칠 것입니다.

```
기존 패러다임:    [저장] ←→ [검색]
EverMemOS 패러다임: [형성] → [통합] → [재구성]
```

#### (2) 생물학적 인지과학과 AI의 융합 촉진

Engram 라이프사이클, 시스템 통합(McGaugh, 2000), 재구성적 기억(Schacter, 2008) 등 인지과학 원리의 계산적 구현이 실제로 성능 향상을 가져옴을 증명했습니다. 이는 **신경과학-AI 경계 연구**를 촉진할 것입니다.

#### (3) 벤치마크 개발 방향 제시

현재 벤치마크가 포착하지 못하는 능력들을 명시적으로 지적했습니다:
- 갈등 감지(conflict detection)
- 프로파일 안정성(profile stability)
- 경험 기반 예지(experience-grounded foresight)

이는 **새로운 평가 프로토콜 개발**을 촉진할 것입니다.

#### (4) RAG 패러다임의 고도화

단순 Dense/Sparse 검색을 넘어 **계층적 의미 구조 기반 검색** 패러다임을 제시했습니다. 이는 RAG 연구의 새로운 방향성을 제공합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (1) 효율성 최적화
LLM 호출이 많아 레이턴시와 비용이 높습니다. 향후 연구에서는:
- **비동기/병렬 처리** 파이프라인 설계
- **소형 모델로의 증류:** Phase I/II 작업을 경량 모델로 위임
- **캐싱 전략:** 빈번히 접근되는 MemScene의 사전 계산

$$\text{Cost}_{\text{total}} = \underbrace{\text{Phase I}}_{\sim 9.42M \text{ tokens}} + \underbrace{\text{Phase III}}_{\sim 10.27M \text{ tokens}} \quad \text{(LoCoMo, 1,540 질문)}$$

#### (2) 멀티모달 확장
MemCell 구조는 모달리티 독립적이나 실제 구현은 텍스트 전용입니다. 향후 연구는:
- 이미지 캡션의 $E$ (Episode) 통합
- 음성 전사의 원자적 사실 추출
- 구현 에이전트의 감각 데이터 통합

#### (3) 온라인 학습 및 적응
현재 시스템은 LLM 프롬프팅에 의존하며, 사용자별 적응이 제한적입니다:
- **개인화된 클러스터링 임계값 $\tau$** 자동 조정
- **연속 학습(Continual Learning)** 통합으로 치명적 망각 방지
- **피드백 루프:** 사용자 수정을 통한 프로파일 업데이트

#### (4) 초장기 타임라인 평가
현재 벤치마크는 수개월 수준의 대화를 다루지만, 실제 배포에서는 수년간의 상호작용이 필요합니다:

$$\text{TimeScale}_{\text{current}} \approx \text{months} \ll \text{TimeScale}_{\text{real}} \approx \text{years}$$

**초장기 메모리 벤치마크** 개발이 필수적입니다.

#### (5) 갈등 해결 메커니즘 강화
현재 시스템은 재현율 우선(recency-aware) 갈등 추적을 수행하지만, 더 정교한 갈등 해결이 필요합니다:
- 시간 가중 신뢰도 계산
- 출처 신뢰도(source credibility) 통합
- 명시적 갈등 그래프 관리

---

## 5. 2020년 이후 최신 연구 비교 분석

| 연구 | 연도 | 방법론 | 핵심 특징 | EverMemOS 대비 차이점 |
|------|------|--------|----------|----------------------|
| **RAG** (Lewis et al.) | 2020 | 검색 증강 생성 | 외부 지식베이스 활용 | 단편적 검색, 통합 메커니즘 없음 |
| **MemGPT** (Packer et al.) | 2024 | 계층적 컨텍스트 관리 | OS처럼 메모리 관리 | 단편 저장, 의미론적 통합 미비 |
| **MemoryBank** (Zhong et al.) | 2024 | 장기 메모리 강화 | 에빙하우스 망각 곡선 적용 | 의미 구조화 부재 |
| **Zep** (Rasmussen et al.) | 2025 | 시간적 지식 그래프 | 지식 그래프 기반 사실 유지 | 구조화되나 에피소드 통합 한계 |
| **Mem0** (Chhikara et al.) | 2025 | 계층적 장기 메모리 | 프로덕션급 스케일 | 지식 그래프 중심, 라이프사이클 없음 |
| **MemOS** (Li et al.) | 2025 | 통합 메모리 OS | 메모리 유형 통합 스케줄링 | 저장 최적화 중심 |
| **MemoryOS** (Kang et al.) | 2025 | 계층적 메모리 제어 | 생명주기 및 용량 관리 | 의미론적 통합 한계 |
| **Nemori** (Nan et al.) | 2025 | 인지과학 기반 자기조직화 | 예측 기반 업데이트 | EverMemOS와 가장 유사하나 3단계 라이프사이클 없음 |
| **EverMemOS** (본 논문) | 2026 | 엔그램 라이프사이클 | 형성-통합-재구성 3단계 | **본 논문의 방법론** |

**핵심 차별점 요약:**

$$\text{EverMemOS} = \underbrace{\text{Episodic Formation}}_{\text{MemGPT 부재}} + \underbrace{\text{Semantic Consolidation}}_{\text{Zep/Mem0 미비}} + \underbrace{\text{Reconstructive Recollection}}_{\text{모든 기존 시스템 미구현}}$$

---

## 참고자료

**본 논문 (직접 분석 대상):**
- Hu, C., Gao, X., Zhou, Z., et al. (2026). *EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning*. arXiv:2601.02163v2

**논문 내 인용 주요 참고문헌:**
- Maharana, A., et al. (2024). *Evaluating very long-term conversational memory of LLM agents*. arXiv:2402.17753 [LoCoMo 벤치마크]
- Wu, D., et al. (2025). *LongMemEval: Benchmarking chat assistants on long-term interactive memory*. arXiv:2410.10813
- Jiang, B., et al. (2025). *PersonaMem-v2*. arXiv:2512.06688
- Li, Z., et al. (2025). *MemOS: A Memory OS for AI System*. arXiv:2507.03724
- Rasmussen, P., et al. (2025). *Zep: A temporal knowledge graph architecture for agent memory*. arXiv:2501.13956
- Chhikara, P., et al. (2025). *Mem0: Building production-ready AI agents with scalable long-term memory*. arXiv:2504.19413
- Kang, J., et al. (2025). *Memory OS of AI Agent*. arXiv:2506.06326
- Nan, J., et al. (2025). *Nemori: Self-organizing agent memory inspired by cognitive science*. arXiv:2508.03341
- Packer, C., et al. (2024). *MemGPT: Towards LLMs as operating systems*. NeurIPS 2024
- Liu, N.F., et al. (2024). *Lost in the middle: How language models use long contexts*. TACL
- Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks*. NeurIPS 2020
- Josselyn, S.A., et al. (2015). *Finding the engram*. Nature Reviews Neuroscience
- McGaugh, J.L. (2000). *Memory–a century of consolidation*. Science
- Schacter, D.L. (2008). *Searching for memory: The brain, the mind, and the past*. Basic Books
- Zhang, Y., et al. (2025). *Qwen3 Embedding*. arXiv:2506.05176
