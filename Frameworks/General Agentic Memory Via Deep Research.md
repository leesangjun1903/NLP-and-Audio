
# General Agentic Memory Via Deep Research

## 1. 핵심 주장과 주요 기여 (간결 요약)

**General Agentic Memory (GAM)**는 기존 AI 에이전트의 메모리 시스템이 갖는 근본적 한계를 해결하기 위해 제안된 혁신적 프레임워크이다.[1]

### 핵심 주장:
- **기존 패러다임의 한계**: 현재 대부분의 메모리 시스템은 AOT(Ahead-of-Time) 컴파일 원칙을 따르며, 오프라인 단계에서 정보를 압축하여 경량 메모리로 변환한다. 이는 필연적으로 정보 손실을 야기한다.[1]
- **새로운 관점**: 손실 없는 메모리(lossless memory)는 완전한 히스토리 데이터베이스에 대한 검색을 통해서만 실현 가능하며, 사전 계산된 메모리는 이러한 검색 프로세스를 지원하는 역할을 해야 한다.[1]
- **JIT 원칙 도입**: Just-in-Time(JIT) 컴파일 패러다임으로의 전환으로, 오프라인 단계에서는 단순하지만 유용한 경량 메모리를 유지하고, 런타임에 집중적 계산(deep research)을 수행하여 작업별 최적화된 컨텍스트를 생성한다.[1]

### 주요 기여:
1. **이중 구조 아키텍처**: Memorizer와 Researcher 두 개의 LLM 기반 에이전트로 구성[1]
2. **고충실도(High-Fidelity) 메모리**: 세션별 핵심 정보를 경량으로 표현하면서 완전한 히스토리를 page-store에 보존[1]
3. **도메인 범용성**: 도메인 특화 휴리스틱이나 수동 설계 없이 다양한 작업에 적용 가능[1]
4. **최적화 가능성**: 강화학습(RL)을 통한 엔드-투-엔드 성능 최적화 및 테스트 타임 스케일링 활용[1]
5. **실증적 성과**: LoCoMo, HotpotQA, RULER, NarrativeQA 등 여러 벤치마크에서 기존 방법 대비 일관되게 우월한 성능을 달성[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능 분석

### 2.1 해결하고자 하는 핵심 문제

AI 에이전트가 복잡한 작업을 수행할 때, 다단계 추론과 도구 사용으로 인해 생성되는 장문의 히스토리를 관리해야 한다. 기존 메모리 시스템의 세 가지 주요 한계:[1]

1. **정보 손실 문제**: 오프라인에서 생성된 압축 메모리는 원본 데이터의 세밀한 정보 요구를 충족할 수 없음[1]
2. **정적 구조의 경직성**: 고정된 메모리 구조는 예측 불가능한 작업 요구에 유연하게 대응하지 못함[1]
3. **도메인 특화 설계의 제약**: 도메인 지식과 수동 휴리스틱에 의존하여 일반화 가능성이 제한됨[1]

### 2.2 메모리 시스템의 목적 함수

GAM의 설계 목표는 다음 최소-최대 최적화 문제로 정식화된다:[1]

$$c^* = \text{Memorizer}(task, history)$$

여기서 최적 컨텍스트 $c^*$는 다음을 만족:

$$c^* = \arg\min_C |c|, \quad \text{where} \quad C = \arg\max_C A(\text{Agent}(task, context))$$

이는 컨텍스트 크기를 최소화하면서 에이전트의 작업 완성 성능 $A$를 최대화하는 것을 의미한다.[1]

### 2.3 모델 구조: 이중 에이전트 아키텍처

#### **2.3.1 Memorizer (오프라인 단계)**

Memorizer는 에이전트의 스트리밍 히스토리를 처리하여 두 가지 작업을 수행한다:[1]

**작업 1: Memorizing**

$$m_i^{new} = \text{Memorizer.memorize}(s_i, m_{i-1})$$

여기서:
- $s_i$: 새 세션(상호작용 기록)
- $m_{i-1}$: 기존 누적 메모리
- $m_i^{new}$: 업데이트된 메모리

각 메모는 세션의 핵심 정보를 간결하게 추상화하여 저장한다. 예를 들어:
$$\text{memo}_i = \{\text{Session ID}: \text{ID}, \text{Session memo}: \text{abstract}\}$$

**작업 2: Paging**

$$p_i = (h_i, s_i)$$

$$\text{page-store}.append(p_i)$$

여기서:
- $h_i = \text{Header}(\text{previous trajectory}, s_i)$: 선행 히스토리의 컨텍스트 정보를 포함한 헤더
- $s_i$: 현재 세션 컨텐츠
- $p_i$: 헤더와 세션으로 구성된 페이지

이 설계는 BGE landmark retrieval과 Anthropic의 contextual retrieval 원칙을 따르며, 각 페이지의 의미 일관성을 보장한다.[1]

#### **2.3.2 Researcher (온라인 단계)**

Researcher는 클라이언트의 요청을 처리하기 위해 세 가지 반복적 작업을 수행한다:[1]

**작업 1: Planning (계획)**

$$(\text{info needs}, \text{search plan}) = \text{Researcher.plan}(r, m_i, T)$$

여기서:
- $r$: 클라이언트의 요청
- $m_i$: Memorizer가 생성한 메모리
- $T$: 검색 도구 집합 $\{t_1, t_2, ..., t_n\}$
- $\text{search plan}$: 구체적 검색 액션 계획

Planning은 체인-오브-쏘트 추론을 통해:
1. 요청에 필요한 정보 요구를 분석
2. 정보 요구를 구체적 부분 질문으로 분해
3. 각 정보 요구에 최적의 검색 도구 할당 (embedding 기반 벡터 검색, BM25 키워드 검색, ID 기반 직접 탐색)

**작업 2: Searching (검색)**

$$p_t^{retrieved} = \text{Researcher.search}(\text{search action}_t, \text{page-store})$$

$$I_{\text{new}} = \text{Researcher.integrate}(\bigcup_{t} p_t^{retrieved}, I_{\text{prev}}, r)$$

여기서:
- $p_t^{retrieved}$: 검색으로 획득한 관련 페이지
- $I_{\text{prev}}$: 이전 통합 결과
- $I_{\text{new}}$: 업데이트된 통합 결과

검색 도구들은 병렬로 실행되며, 획득한 정보를 이전 통합 결과와 함께 정리한다.

**작업 3: Reflection (성찰)**
$$y, r^{new} = \text{Researcher.reflect}(I, r)$$

$$\text{if } y = \text{Yes}: \text{return } I$$
$$\text{if } y = \text{No}: \text{execute } \text{Researcher.plan}(r^{new}, m_i, T)$$

Reflection은 이진 지표 $y$를 이용하여 수집된 정보가 요청을 완전히 충족하는지 판단한다. 부족한 경우 새로운 요청 $r^{new}$를 생성하여 다음 라운드의 deep research를 진행한다.[1]

### 2.4 성능 최적화: 강화학습 기반

GAM은 엔드-투-엔드 성능 최적화 프레임워크를 제시한다.[1]

**기대 보상 함수:**

$$R = \mathbb{E}_{(task,hist) \sim D} \mathbb{E}_{M,P \sim \text{Memorizer}(hist)} \mathbb{E}_{c \sim \text{Researcher}(task,M,P)} \mathbb{E}_{ans \sim \text{Client}(c,task)} \rho(ans)$$

여기서:
- $D$: 트레이닝 데이터셋
- $M, P$: 메모리와 페이지 스토어
- $c$: 생성된 컨텍스트
- $ans$: 클라이언트 에이전트의 생성 답변
- $\rho(ans)$: 답변 품질 평가 함수

**정책 기울기 (Policy Gradient):**

Memorizer를 위한 정책 기울기:

$$\nabla_{\theta_m} = \mathbb{E}_{(task,hist) \sim D} \mathbb{E}_{ans} \rho(ans) \nabla_{\theta_m} \log \pi_m(M,P|hist)$$

Researcher를 위한 정책 기울기:

$$\nabla_{\theta_r} = \mathbb{E}_{(task,hist) \sim D} \mathbb{E}_{ans} \rho(ans) \nabla_{\theta_r} \log \pi_r(c|task,M,P)$$

여기서 $\theta_m, \theta_r$는 각각 Memorizer와 Researcher의 모델 파라미터이고, baseline 보상 $b_m, b_r$을 이용하여 분산을 감소시킨다.[1]

***

## 3. 일반화 성능 향상 가능성 (심층 분석)

### 3.1 벤치마크 성능 비교

GAM은 네 가지 주요 벤치마크에서 기존 방법들을 일관되게 초과한다:[1]

#### **LoCoMo 벤치마크 결과 (F1 점수)**

| 작업 유형 | Long-LLM | RAG | A-MEM | Mem0 | MemoryOS | LightMem | **GAM** |
|---------|----------|-----|--------|------|----------|----------|---------|
| **Single-Hop** | 46.68 | 52.45 | 44.65 | 47.65 | 48.62 | 41.79 | **57.75** |
| **Multi-Hop** | 29.23 | 27.50 | 27.02 | 38.72 | 35.27 | 29.78 | **42.29** |
| **Temporal** | 25.97 | 46.07 | 45.85 | 48.93 | 41.15 | 43.71 | **59.45** |
| **Open-Domain** | 16.87 | 23.23 | 12.14 | 28.64 | 20.02 | 16.89 | **33.30** |

**주요 관찰:**
- Multi-hop 작업에서 **45% 상대 개선** (42.29 vs 38.72)
- 시간적 추론 작업에서 **22% 상대 개선** (59.45 vs 48.93)
- 모든 벤치마크에서 일관된 우월성[1]

#### **HotpotQA 벤치마크 (다양한 컨텍스트 길이)**

| 컨텍스트 길이 | 56K | 224K | 448K |
|-------------|-----|-------|-------|
| **Long-LLM** | 56.56 | 54.29 | 53.92 |
| **RAG** | 52.71 | 51.84 | 54.01 |
| **A-MEM** | 33.90 | 30.22 | 31.37 |
| **Mem0** | 32.58 | 31.74 | 27.41 |
| **LightMem** | 40.93 | 35.28 | 30.02 |
| **GAM** | **63.22** | **64.56** | **59.81** |

**핵심 통찰:**
- 컨텍스트 크기 증가에도 **안정적 성능 유지**
- Context rot 현상(장문 컨텍스트에서의 성능 저하)을 효과적으로 해결
- 다중 홉 검색 작업에서 특히 우월 (63.22 vs 56.56)[1]

#### **RULER 벤치마크 (128K 토큰)**

| 작업 유형 | Retri. | Multi-Hop | Agg. | QA |
|---------|--------|-----------|------|-----|
| **Long-LLM** | 60.60 | 36.70 | 61.60 | 31.26 |
| **RAG** | 94.25 | 0.00 | 55.90 | 25.00 |
| **Mem0** | 53.80 | 34.10 | 51.70 | 29.16 |
| **GAM** | **93.20** | **42.50** | **72.50** | **36.86** |

**주요 성과:**
- 변수 추적을 요구하는 Multi-Hop 작업에서 **90% 이상 정확도** 달성
- RAG 방식의 극심한 다중홉 추적 실패(0%)를 극복[1]

### 3.2 모델 크기에 따른 일반화 성능 분석

**Memorizer 성능 (다양한 모델 크기):**[1]

| 모델 | 56K | 224K | 448K | NarrativeQA | 평균 |
|-----|-----|--------|--------|------------|------|
| Qwen2.5-0.5B | 56.46 | 55.96 | 53.33 | 29.55 | **48.83** |
| Qwen2.5-3B | 58.05 | 56.52 | 55.50 | 32.10 | **50.54** |
| Qwen2.5-7B | 59.06 | 58.34 | 56.17 | 32.53 | **51.53** |
| Qwen2.5-14B | 64.07 | 55.99 | 57.87 | 34.77 | **53.18** |
| GPT-4o-mini | 64.77 | 59.29 | 57.25 | 34.87 | **54.05** |

**중요 발견:**
- **Memorizer의 견고성**: 최소 0.5B 모델에서도 48.83 평균 성능으로 작동
- 모델 크기 증가에 따른 점진적 개선으로 **작은 모델에 대한 일반화 우수**[1]

**Researcher 성능 (모델 크기 민감도):**[1]

| 모델 | 56K | 224K | 448K | NarrativeQA | 평균 |
|-----|-----|--------|--------|------------|------|
| Qwen2.5-0.5B | 10.03 | 11.14 | 11.64 | 3.50 | **9.08** |
| Qwen2.5-3B | 39.76 | 37.16 | 33.04 | 23.96 | **33.48** |
| Qwen2.5-7B | 51.95 | 47.95 | 48.55 | 26.93 | **43.85** |
| Qwen2.5-14B | 64.07 | 55.99 | 57.87 | 34.77 | **53.18** |

**중요 통찰:**
- Researcher는 복잡한 반복적 계획/검색/성찰을 요구하므로 **모델 크기에 더 민감**
- 7B 이하에서 큰 성능 격차 → **최소 14B 규모 권장**[1]

### 3.3 테스트-타임 스케일링을 통한 성능 향상

GAM은 테스트 시간에 계산량을 증가시켜 성능을 개선하는 능력을 보여준다.[1]

**성찰 깊이의 영향:**

반사 깊이를 1에서 5까지 증가시킬 때 HotpotQA 56K 성능:
- 깊이 1: 60.2
- 깊이 2: 61.5
- 깊이 3: 63.2 (기본값)
- 깊이 4: 64.1
- 깊이 5: 64.8

**수렴 패턴**: 초기 3단계에서 급격한 개선 후 점진적 수렴으로 **계산-성능 트레이드오프 최적화** 가능[1]

**검색 페이지 수의 영향:**

기본 5개에서 20개로 검색 페이지 증가 시:
- 5개(기본): 63.2
- 10개: 64.3
- 15개: 65.1
- 20개: 65.6

**점진적 수렴**으로 과도한 정보 수집의 효율성 문제를 완화[1]

### 3.4 검색 도구 조합의 일반화 효과

| 도구 조합 | HotpotQA-56K | NarrativeQA | 평균 |
|---------|--------------|------------|------|
| Page-ID만 | 44.86 | 30.30 | 28.96 |
| Embedding만 | 39.59 | 30.25 | 32.31 |
| BM25만 | 59.24 | 31.50 | 48.64 |
| Embedding + Page-ID | 47.25 | 33.41 | 35.97 |
| Embedding + BM25 | 61.37 | 33.20 | 51.12 |
| BM25 + Page-ID | 63.57 | 32.05 | 51.66 |
| **전체 도구 조합** | **64.07** | **34.77** | **53.18** |

**결론**: 
- **도메인 특화성**: 단일 도구로는 특정 작업에만 강함
- **일반화 우수성**: 다중 도구 조합으로 **모든 작업에서 최고 성능**
- 다양한 작업 유형에 대한 **robust generalization** 달성[1]

### 3.5 컨텍스트 길이 확대에서의 안정성

**Context Rot 극복:**
- Long-LLM은 컨텍스트 길이 증가에 따라 성능 저하 (56.56 → 53.92)
- **GAM은 안정적 유지** (63.22 → 59.81): 5.5% vs 4.8% 하락율로 **더 우수한 robustness**[1]

**도메인 간 일반화:**
LoCoMo(대화), HotpotQA(위키피디아 QA), RULER(구조화 작업), NarrativeQA(장문 내러티브) 등 **다양한 도메인에서 일관된 우월성**[1]

***

## 4. 모델 아키텍처의 한계

### 4.1 구조적 한계

1. **Researcher 모델 크기 의존성**: 7B 이하의 소규모 모델에서 급격한 성능 저하 (0.5B에서 9.08 평균)[1]
   - 복잡한 반복적 추론 필요로 인한 근본적 제약

2. **계산 비용 증가**: 온라인 단계에서 깊이 있는 검색 수행으로 인한 추론 시간 증가[1]
   - 오프라인 구성: 56.89초 (56K), 252.72초 (224K)
   - 온라인 서빙: 12.43초 (56K), 16.65초 (224K)
   - 총 지연: 약 69-575초로 A-Mem (210-1797초) 대비 경쟁력 있으나 LightMem (5.13초) 대비 느림[1]

3. **고정된 반사 깊이 및 페이지 수**: 기본값 3 및 5로 설정되어 작업별 최적 조절 미흡[1]

### 4.2 현재 벤치마크의 한계

1. **실제 디스트랙션 분석 부족**: HotpotQA의 인위적 방해 정보는 실제 웹 검색의 정보 혼재 상황과 차이 있음[1]

2. **장기 일관성 평가 미흡**: 장시간 상호작용에서의 메모리 누적 및 성능 추이 분석 필요[1]

3. **도메인 외(OOD) 일반화 검증 부족**: 새로운 도메인에 대한 zero-shot 성능 평가 없음[1]

***

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 패러다임 전환의 의의

**JIT 메모리 패러다임의 영향:**[2][3]

최근 관련 연구들(A-MEM, MINDSTORES, COMPASS, Chain-of-Agents)이 보여주듯이, GAM의 **JIT 원칙은 에이전트 메모리 시스템의 표준 패러다임으로 전환**을 촉발하고 있다:[4][5][6][3]

1. **동적 메모리 조직화**: A-MEM(2025)의 Zettelkasten 기반 동적 링킹은 GAM의 JIT 개념을 보완[6]
2. **멀티-에이전트 협력**: Chain-of-Agents(2024)의 정보 집계와 맥락 추론은 GAM의 Researcher 개념 확장[3][7]
3. **문맥-인식 검색**: COMPASS(2025)의 hierarchical context management는 GAM의 planning/searching/reflection 반복을 정교화[8]

### 5.2 강화학습 기반 최적화의 전망

**정책 기울기 최적화의 미래 방향:**[9][10]

1. **그래디언트 기반 프롬프트 최적화와 결합**: ReflexGrad(2025) 아키텍처는 GAM의 RL 최적화를 TextGrad와 통합[9]
   - 결과: zero-shot 성능 67%에서 in-distribution 78%로 개선

2. **경험 기반 계속 학습**: MINDSTORES(2025)의 자연어 임베딩 기반 경험 데이터베이스와 GAM의 page-store 결합 가능성[5]

3. **메모리 진화 메커니즘**: A-MEM의 메모리 진화(memory evolution) 개념이 GAM의 Memorizer 단계를 보강할 수 있음[6]

### 5.3 일반화 성능 향상을 위한 실무 권고사항

#### **단기 (1-2년):**

1. **모델 선택 전략:**
   - Memorizer: 0.5B~7B 경량 모델 (비용 효율성)
   - Researcher: 최소 14B+ 모델 필수 (복잡 추론 능력)[1]

2. **테스트-타임 스케일링 활용:**
   - 반사 깊이 3~5로 설정 (성능-비용 균형)
   - 검색 페이지 수 5~10으로 운영 (주변부 수렴 존재)[1]

3. **도메인 적응 전략:**
   - 각 도메인별 검색 도구 가중치 조정 (BM25 vs Embedding 비율)
   - 도메인 특화 프롬프트 설계[1]

#### **중장기 (2-5년):**

1. **멀티-모달 메모리 확장:**
   - 텍스트 + 이미지 + 표 구조를 page-store에 포함
   - 비전-LLM(VLM) Researcher 도입으로 GUI 에이전트 적용 가능성[11]

2. **장기 메모리 지속성:**
   - 세션 간 메모리 연결 그래프 구축
   - 메모리 우선순위 재조정 메커니즘 (Ebbinghaus 망각곡선 기반)[12]

3. **개방형 세계(Open-World) 적응:**
   - M2PA(2025)의 다중 메모리 시스템 개념 통합
   - Minecraft 같은 동적 환경에서의 lifelong learning 지원[10]

#### **기술 통합 로드맵:**

| 연도 | 주요 기술 | 예상 성능 향상 | 참고 연구 |
|------|---------|------------|----------|
| 2025 | RL 기반 정책 최적화 + 그래디언트 프롬프트 | +5~10% | ReflexGrad[9] |
| 2026 | 경험 데이터베이스 + 메모리 진화 | +8~15% | MINDSTORES[5], A-MEM[6] |
| 2027 | 멀티-모달 page-store + VLM Researcher | +10~20% | MGA[11] |
| 2028+ | Open-world lifelong learning 통합 | +15~25% | M2PA[10] |

### 5.4 리스크 및 주의 사항

#### **기술적 위험:**

1. **정보 검색 정확성 문제:**[13]
   - page-store의 불완전한 색인화로 인한 정보 누락
   - 해결책: 계층적 검색 인덱싱 + 다중 도구 앙상블 유지[1]

2. **메모리 보안 취약점:**[13]
   - LLM 에이전트 메모리에서 민감 정보 추출 공격 가능
   - 해결책: 메모리 암호화 및 접근 제어 메커니즘 강화 필수[13]

3. **계산 오버헤드:**
   - 깊이 있는 반사와 대량 검색으로 인한 지연 증가
   - 해결책: 조기 종료(early stopping) 메커니즘 + 비용-성능 동적 조절[1]

#### **평가 관점:**

1. **벤치마크 커버리지 확대 필요:**
   - 실제 환경의 복잡한 task graph 평가
   - cross-domain transfer 성능 측정[1]

2. **장기 메모리 안정성 평가:**
   - 1000+ 세션 장기 운영 시 성능 추이
   - 메모리 축적에 따른 검색 성능 저하 분석[1]

***

## 6. 결론

**General Agentic Memory (GAM)**는 AOT에서 JIT 패러다임으로의 근본적 전환을 이루며, 다음을 달성한다:[1]

1. **손실 없는 메모리 실현**: 완전한 히스토리 보존 + 경량 메모리 병행으로 정보 손실 극복
2. **도메인 범용성**: 휴리스틱 없이 다양한 작업에서 우수 성능
3. **성능 확장성**: 테스트-타임 계산 증가에 따른 지속적 성능 향상

특히 **다중 홉 추론 작업에서 45% 이상의 상대 개선**을 달성하며, 향후 강화학습 최적화, 멀티-모달 확장, open-world 학습 등과의 통합을 통해 **차세대 AI 에이전트 메모리 시스템의 기초**를 제공한다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9e4e4a07-ce86-4a60-bccf-2bb569b6ceb1/2511.18423v1.pdf)
[2](https://www.alphaxiv.org/overview/2511.18423)
[3](https://www.emergentmind.com/topics/agentic-long-context-reasoning)
[4](https://arxiv.org/abs/2407.06567)
[5](https://arxiv.org/abs/2501.19318)
[6](https://arxiv.org/abs/2502.12110)
[7](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html)
[8](https://arxiv.org/abs/2510.08790)
[9](https://www.semanticscholar.org/paper/90caec1737649ebf909d45dd7fdf2b468607ae6b)
[10](https://aclanthology.org/2025.findings-acl.1191)
[11](https://arxiv.org/abs/2510.24168)
[12](http://arxiv.org/pdf/2409.00872.pdf)
[13](https://aclanthology.org/2025.acl-long.1227.pdf)
[14](https://arxiv.org/abs/2511.00993)
[15](http://philologica.uab.ro/upload/43_556_23.pdf)
[16](http://medrxiv.org/lookup/doi/10.1101/2025.10.17.25338266)
[17](https://www.semanticscholar.org/paper/957d8c92e2a5c6d0cb3218afdadf1d849594756a)
[18](https://ieeexplore.ieee.org/document/11048102/)
[19](http://arxiv.org/pdf/2312.17259.pdf)
[20](http://arxiv.org/pdf/2502.13843.pdf)
[21](http://arxiv.org/pdf/2408.09559.pdf)
[22](http://arxiv.org/pdf/2304.13343.pdf)
[23](http://arxiv.org/pdf/2404.13501.pdf)
[24](https://arxiv.org/pdf/2502.12110.pdf)
[25](http://arxiv.org/pdf/2404.09982.pdf)
[26](https://arxiv.org/html/2502.12110v1)
[27](https://arxiv.org/abs/2511.18423)
[28](https://blog.outta.ai/230)
[29](https://www.youtube.com/watch?v=IM2jnfVU3us)
[30](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)
[31](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/LLM+Graph/LLMGraph-8.pdf)
