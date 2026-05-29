# Zep: A Temporal Knowledge Graph Architecture for Agent Memory

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Zep은 **LLM 기반 AI 에이전트를 위한 동적 메모리 레이어 서비스**로, 기존 RAG(Retrieval-Augmented Generation) 프레임워크가 정적 문서 검색에 국한된 한계를 극복하고자 한다. 핵심 엔진인 **Graphiti**는 시간 인식(temporally-aware) 지식 그래프로, 비구조화 대화 데이터와 구조화 비즈니스 데이터를 동시에 통합·관리한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **Graphiti 엔진** | 시간 인식 동적 지식 그래프 엔진: 사실(fact)의 유효기간 추적 |
| **계층적 메모리 구조** | 에피소드 → 의미 엔티티 → 커뮤니티 3계층 서브그래프 |
| **바이-템포럴(bi-temporal) 모델** | 사건 발생 시간($T$)과 데이터 수집 시간($T'$) 이중 추적 |
| **하이브리드 검색** | 코사인 유사도 + BM25 전문 검색 + 너비우선탐색(BFS) 결합 |
| **성능** | DMR: 94.8% (MemGPT 93.4% 대비), LongMemEval: 최대 18.5% 정확도 향상, 응답 지연 90% 감소 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 RAG의 한계:**

1. **정적 코퍼스 의존**: 문서가 추가되면 변경되지 않는 정적 저장소 기반
2. **컨텍스트 윈도우 제약**: LLM의 컨텍스트 윈도우에 전체 대화 이력을 담을 수 없음
3. **시간적 추론 부재**: 정보의 시간적 유효성 관리 불가
4. **단순 검색**: 복잡한 다중 세션 정보 합성 불가
5. **동적 지식 통합 불가**: 변화하는 사실 관계 추적 불가

---

### 2.2 제안하는 방법

#### 2.2.1 지식 그래프 구조 정의

Zep의 메모리는 시간 인식 동적 지식 그래프로 정의된다:

$$\mathcal{G} = (\mathcal{N}, \mathcal{E}, \phi)$$

여기서:
- $\mathcal{N}$: 노드 집합
- $\mathcal{E}$: 엣지 집합
- $\phi: \mathcal{E} \rightarrow \mathcal{N} \times \mathcal{N}$: 형식적 입사 함수(incidence function)

**3계층 서브그래프:**

$$\mathcal{G} = \mathcal{G}_e \cup \mathcal{G}_s \cup \mathcal{G}_c$$

- **에피소드 서브그래프** $\mathcal{G}_e$: 원시 입력 데이터 저장
  - 에피소드 노드: $n_i \in \mathcal{N}_e$ (메시지, 텍스트, JSON)
  - 에피소드 엣지: $e_i \in \mathcal{E}_e \subseteq \phi^*(\mathcal{N}_e \times \mathcal{N}_s)$

- **의미 엔티티 서브그래프** $\mathcal{G}_s$: 엔티티 및 관계 저장
  - 엔티티 노드: $n_i \in \mathcal{N}_s$
  - 의미 엣지: $e_i \in \mathcal{E}_s \subseteq \phi^*(\mathcal{N}_s \times \mathcal{N}_s)$

- **커뮤니티 서브그래프** $\mathcal{G}_c$: 고수준 클러스터 요약
  - 커뮤니티 노드: $n_i \in \mathcal{N}_c$
  - 커뮤니티 엣지: $e_i \in \mathcal{E}_c \subseteq \phi^*(\mathcal{N}_c \times \mathcal{N}_s)$

#### 2.2.2 바이-템포럴(Bi-Temporal) 모델

Zep은 두 개의 독립적 타임라인을 유지한다:

$$T: \text{사건의 실제 발생 순서 (chronological timeline)}$$
$$T': \text{데이터 수집/처리 순서 (transactional timeline)}$$

각 엣지(사실)에 저장되는 4개의 타임스탬프:

$$\{t'_{\text{created}},\ t'_{\text{expired}}\} \in T' \quad \text{(시스템 내 생성/만료 시각)}$$
$$\{t_{\text{valid}},\ t_{\text{invalid}}\} \in T \quad \text{(사실의 실제 유효 기간)}$$

**엣지 무효화(Edge Invalidation) 로직:**

새 엣지 $e_{\text{new}}$가 기존 엣지 $e_{\text{old}}$와 모순될 때:

$$t_{\text{invalid}}(e_{\text{old}}) \leftarrow t_{\text{valid}}(e_{\text{new}})$$

#### 2.2.3 메모리 검색 파이프라인

검색 시스템은 함수 합성으로 표현된다:

$$f: \mathcal{S} \rightarrow \mathcal{S}$$

$$f(\alpha) = \chi(\rho(\varphi(\alpha))) = \beta$$

**각 단계의 형식적 정의:**

- **검색 함수** $\varphi$:

$$\varphi: \mathcal{S} \rightarrow \mathcal{E}_s^n \times \mathcal{N}_s^n \times \mathcal{N}_c^n$$

쿼리 문자열 $\alpha$를 의미 엣지, 엔티티 노드, 커뮤니티 노드의 3-튜플로 변환

- **리랭커** $\rho$:

$$\rho: \varphi(\alpha), \ldots \rightarrow \mathcal{E}_s^n \times \mathcal{N}_s^n \times \mathcal{N}_c^n$$

- **컨스트럭터** $\chi$:

$$\chi: \mathcal{E}_s^n \times \mathcal{N}_s^n \times \mathcal{N}_c^n \rightarrow \mathcal{S}$$

#### 2.2.4 세 가지 검색 방법

$$\varphi = \{\varphi_{\text{cos}},\ \varphi_{\text{bm25}},\ \varphi_{\text{bfs}}\}$$

| 검색 방법 | 수식/기법 | 특성 |
|---|---|---|
| **코사인 유사도** $\varphi_{\text{cos}}$ | $\text{sim}(q, d) = \frac{q \cdot d}{\|q\|\|d\|}$ | 의미적 유사성 |
| **BM25** $\varphi_{\text{bm25}}$ | Okapi BM25 | 어휘적 유사성 |
| **너비우선탐색** $\varphi_{\text{bfs}}$ | $n$-hop 그래프 탐색 | 맥락적 유사성 |

엔티티는 **1024차원 벡터 공간**에 임베딩된다.

#### 2.2.5 리랭킹 방법

- **Reciprocal Rank Fusion (RRF)**
- **Maximal Marginal Relevance (MMR)**
- **에피소드 언급 빈도 기반 리랭킹**
- **노드 거리 기반 리랭킹**
- **크로스 인코더(Cross-encoder)**: LLM 기반 교차주의(cross-attention) 관련도 점수 생성

#### 2.2.6 커뮤니티 탐지

GraphRAG의 Leiden 알고리즘 대신 **레이블 전파 알고리즘(Label Propagation)**을 채택:

- 새 엔티티 노드 $n_i \in \mathcal{N}_s$ 추가 시, 이웃 노드들의 커뮤니티를 조사
- 다수결(plurality) 커뮤니티에 신규 노드 배정
- 완전한 레이블 전파 재실행 없이 동적으로 커뮤니티 갱신 → **지연시간 및 LLM 추론 비용 절감**

---

### 2.3 모델 구조

```
[Raw Data Input]
      │
      ▼
[Episode Subgraph Ge]          ← 원시 데이터 비손실 저장
      │ Entity Extraction
      │ (n=4 context window, Reflexion 기법)
      ▼
[Semantic Entity Subgraph Gs]  ← 엔티티 + 사실(Facts) + 시간 정보
      │ Community Detection
      │ (Label Propagation)
      ▼
[Community Subgraph Gc]        ← 고수준 클러스터 요약

[Memory Retrieval]
Query α → φ(cos + bm25 + bfs) → ρ(RRF/MMR/Cross-encoder) → χ → Context β
```

---

### 2.4 성능 향상

#### DMR 벤치마크 (Deep Memory Retrieval)

| 메모리 방식 | 모델 | 점수 |
|---|---|---|
| Recursive Summarization† | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT† | gpt-4-turbo | 93.4% |
| Full-conversation | gpt-4-turbo | 94.4% |
| **Zep** | **gpt-4-turbo** | **94.8%** |
| Conversation Summaries | gpt-4o-mini | 88.0% |
| Full-conversation | gpt-4o-mini | 98.0% |
| **Zep** | **gpt-4o-mini** | **98.2%** |

#### LongMemEval 벤치마크

| 메모리 방식 | 모델 | 점수 | 지연시간 | 평균 컨텍스트 토큰 |
|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 31.3s | 115k |
| **Zep** | **gpt-4o-mini** | **63.8%** | **3.20s** | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9s | 115k |
| **Zep** | **gpt-4o** | **71.2%** | **2.58s** | **1.6k** |

**질문 유형별 주요 성과 (gpt-4o 기준):**
- single-session-preference: $+184\%$↑
- temporal-reasoning: $+38.4\%$↑
- multi-session: $+30.7\%$↑

---

### 2.5 한계

1. **single-session-assistant 성능 저하**: gpt-4o에서 $-17.7\%$, gpt-4o-mini에서 $-9.06\%$ 하락
   - Zep의 시간적 데이터 표현이 어시스턴트 발화 기반 단일 세션 질문에 적합하지 않을 수 있음
2. **DMR 벤치마크 한계**: 단순 사실 검색 질문 위주, 실제 엔터프라이즈 시나리오 미반영
3. **MemGPT와의 직접 비교 불가**: LongMemEval에서 MemGPT 평가 실패 (기존 메시지 이력 직접 수집 미지원)
4. **커뮤니티 표류(Community Drift)**: 동적 업데이트 시 완전한 레이블 전파와 점진적으로 차이 발생 → 주기적 갱신 필요
5. **LLM 의존성**: 그래프 구축 전반에 LLM 활용 → 비용·지연시간·환각(hallucination) 위험
6. **벤치마크 부족**: 구조화 비즈니스 데이터와 대화 이력 통합 능력을 평가하는 벤치마크 전무

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지원하는 구조적 특성

#### (1) 비손실(Non-lossy) 에피소드 저장

모든 원시 데이터가 에피소드 서브그래프 $\mathcal{G}_e$에 보존되므로, 향후 더 발전된 추출 알고리즘이나 다른 도메인에도 재처리(re-processing) 적용 가능:

$$\mathcal{G}_e \xrightarrow{\text{재처리}} \mathcal{G}_s^{\text{new domain}}$$

#### (2) 바이-템포럴 모델의 도메인 독립성

$T$와 $T'$ 타임라인 분리는 특정 도메인에 종속되지 않으므로, 의료·법률·금융 등 시간적 맥락이 중요한 모든 도메인에 일반화 가능하다.

#### (3) 하이브리드 검색의 강건성

$$\varphi_{\text{hybrid}} = \text{combine}(\varphi_{\text{cos}},\ \varphi_{\text{bm25}},\ \varphi_{\text{bfs}})$$

어휘적·의미적·구조적 유사도를 동시에 활용하므로, 특정 임베딩 모델이나 어휘에 과적합되지 않고 다양한 도메인에 적응 가능하다.

#### (4) 커뮤니티 기반 전역 이해

GraphRAG 스타일의 커뮤니티 요약은 **도메인별 개념 클러스터**를 자동 형성하여, 새로운 도메인 진입 시 고수준 구조 파악을 가속화한다.

### 3.2 일반화 향상을 위한 미래 방향

논문이 명시적으로 제안하는 방향:

**① 파인튜닝된 추출 모델 도입:**

> "Research has already demonstrated the value of fine-tuned models for LLM-based entity and edge extraction within the GraphRAG paradigm, improving accuracy while reducing costs and latency." (Distill-SynthKG [19], Triplex [25] 참조)

도메인별 파인튜닝 시 엔티티·엣지 추출 정확도 향상 → 도메인 특화 일반화 성능 제고

**② 도메인 특화 온톨로지(Ontology) 통합:**

$$\mathcal{G}_{\text{domain}} = \mathcal{G} \cup \mathcal{O}_{\text{domain}}$$

여기서 $\mathcal{O}_{\text{domain}}$은 도메인 온톨로지. 사전 정의된 스키마는 지식 추출의 일관성을 높여 OOD(out-of-distribution) 시나리오에서의 일반화를 지원한다.

**③ LightRAG 방법론과의 통합:**

LightRAG의 고수준 키워드 검색과 Graphiti의 그래프 기반 시스템을 결합하면:
- 키워드 매칭 기반 검색의 어휘적 일반화
- 그래프 구조 기반 맥락적 일반화

**④ 구조화 데이터와 비구조화 데이터의 통합 처리:**

JSON 에피소드 타입을 통한 비즈니스 데이터 처리는 텍스트 이외의 모달리티로의 일반화 가능성을 시사한다.

### 3.3 일반화 성능의 현재 근거

LongMemEval에서의 성과가 일반화 가능성을 실증:

$$\Delta_{\text{accuracy}}(\text{gpt-4o}) = 71.2\% - 60.2\% = +11.0\%p \quad \text{(전체 평균)}$$

$$\Delta_{\text{latency}} \approx -90\% \quad \text{(컨텍스트 토큰: } 115k \rightarrow 1.6k\text{)}$$

특히 **cross-session 정보 합성**과 **시간적 추론** 항목에서의 향상은, Zep이 단순 사실 검색을 넘어 복잡한 추론 시나리오에 일반화될 수 있음을 보여준다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 향후 연구에 미치는 영향

#### (1) LLM 에이전트 메모리 아키텍처의 새로운 패러다임 제시

Zep은 단순한 벡터 스토어나 요약 기반 메모리를 넘어, **시간 인식 지식 그래프**를 에이전트 메모리의 핵심으로 삼는 새로운 아키텍처를 확립했다. 이는 향후 에이전트 메모리 연구의 기준점이 될 것이다.

#### (2) 벤치마크 설계의 재고

논문은 DMR의 한계를 명확히 지적한다:
- 단순 단일 사실 검색 위주
- 실제 엔터프라이즈 시나리오 미반영

이는 **더 현실적인 메모리 평가 벤치마크 개발**을 촉구하며, 특히 다음을 포함해야 함:
- 다중 세션 복합 추론
- 시간적 관계 추론
- 구조화 데이터 + 대화 이력 통합 평가

#### (3) 프로덕션 시스템 성능 지표의 중요성 부각

지연시간·비용 등 실용적 지표를 함께 보고함으로써, 학술 연구가 프로덕션 적용 가능성을 함께 고려해야 함을 강조했다.

#### (4) GraphRAG 계열 연구 활성화

GraphRAG [4], AriGraph [9], LightRAG [17], Zep의 계보를 잇는 연구들이 더욱 활발해질 것으로 예상:

$$\text{GraphRAG} \rightarrow \text{AriGraph} \rightarrow \text{LightRAG} \rightarrow \text{Zep} \rightarrow \text{?}$$

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 더 강건한 벤치마크 구축
- 고객 서비스, 의료 상담, 법률 자문 등 실제 엔터프라이즈 시나리오 반영
- 구조화 데이터와 대화 이력의 통합 처리 능력 평가 포함

#### (2) 도메인 특화 온톨로지 설계
- 금융, 의료 등 도메인별 사전 정의 스키마 통합
- 온톨로지 없는 현재 접근법(open-schema KG)의 한계 보완

#### (3) 파인튜닝된 소형 모델 활용
- GPT-4 수준 모델에 의존하는 현재 구조의 비용 문제 해결
- Distill-SynthKG [19], Triplex [25]처럼 특화된 경량 모델로 엔티티·엣지 추출 대체

#### (4) Single-session-assistant 성능 저하 원인 분석
- Zep의 시간적 데이터 표현이 어시스턴트 응답 기반 질문에 왜 적합하지 않은지 규명
- 응답자(assistant) 발화를 별도로 처리하는 메커니즘 연구

#### (5) 커뮤니티 표류(Community Drift) 해결
- 동적 레이블 전파와 완전 재실행 간의 최적 균형점 탐색
- 적응형 커뮤니티 갱신 트리거 메커니즘 개발

#### (6) 멀티모달 에피소드 처리
- 현재 텍스트·JSON 중심 → 이미지, 음성 등 멀티모달 데이터 통합

#### (7) 개인정보 및 보안 고려
- 개인 대화 데이터가 지식 그래프에 장기 보존될 때의 프라이버시 보호 메커니즘
- 민감 정보의 선택적 삭제(edge invalidation 활용) 정책 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기법 | 시간 인식 | 동적 업데이트 | 주요 특징 |
|---|---|---|---|---|---|
| **MemGPT** [3] | 2024 | 계층적 메모리 관리 | ✗ | 제한적 | LLM을 OS처럼 활용, 컨텍스트 페이징 |
| **GraphRAG** [4] | 2024 | 커뮤니티 기반 그래프 RAG | ✗ | ✗ | Leiden 알고리즘, 전역 요약 |
| **AriGraph** [9] | 2024 | 지식 그래프 + 에피소드 메모리 | 부분적 | 부분적 | 에피소드·의미 메모리 이중 구조 |
| **LightRAG** [17] | 2024 | 경량 그래프 RAG | ✗ | 부분적 | 고수준 키워드 검색, 저지연 |
| **Distill-SynthKG** [19] | 2024 | KG 구축 워크플로우 증류 | ✗ | ✗ | 파인튜닝 모델로 비용·지연 감소 |
| **GraphReader** [26] | 2024 | 그래프 기반 에이전트 | ✗ | ✗ | 장문 컨텍스트 처리 |
| **Zep (Graphiti)** | 2025 | 시간 인식 동적 KG | ✓ (bi-temporal) | ✓ (실시간) | 바이-템포럴 모델, 엣지 무효화 |

### 핵심 차별점 분석

**Zep vs. MemGPT:**
- MemGPT는 LLM의 컨텍스트 윈도우를 OS 메모리처럼 관리하는 방식으로, 정보의 시간적 유효성을 직접 추적하지 않음
- Zep은 $\{t_{\text{valid}}, t_{\text{invalid}}\}$ 타임스탬프로 모든 사실의 유효기간을 명시적으로 관리

**Zep vs. GraphRAG:**
- GraphRAG는 정적 문서 코퍼스 대상 오프라인 처리에 최적화
- Zep은 실시간 대화 스트림에 동적으로 업데이트되는 온라인 처리 시스템

**Zep vs. LightRAG:**
- LightRAG는 경량·고속을 우선시하지만 시간적 관계 추적 미흡
- Zep은 바이-템포럴 모델로 시간적 추론 능력을 강화하면서도 90% 지연 감소 달성

---

## 참고자료

**논문 원문:**
- Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956v1 [cs.CL].

**논문 내 인용 참고문헌:**
- [3] Packer et al. *MemGPT: Towards LLMs as Operating Systems*, 2024.
- [4] Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024.
- [7] Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, 2024.
- [9] Anokhin et al. *AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents*, 2024.
- [17] Guo et al. *LightRAG: Simple and Fast Retrieval-Augmented Generation*, 2024.
- [19] Choubey et al. *Distill-SynthKG: Distilling Knowledge Graph Synthesis Workflow for Improved Coverage and Efficiency*, 2024.
- [25] Pimpalgaonkar et al. *Triplex: A SOTA LLM for Knowledge Graph Construction*, 2024.
- [26] Li et al. *GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models*, 2024.

**기타 참조:**
- Zep GitHub: https://github.com/getzep/graphiti
- Zep 공식 사이트: https://www.getzep.com

> **주의**: 본 답변은 제공된 논문 PDF(arXiv:2501.13956v1) 원문에 기반하여 작성되었으며, 논문에 명시되지 않은 내용은 포함하지 않았습니다. 2020년 이후 최신 연구 비교는 해당 논문의 참고문헌 목록에 수록된 연구들을 중심으로 분석하였습니다.
