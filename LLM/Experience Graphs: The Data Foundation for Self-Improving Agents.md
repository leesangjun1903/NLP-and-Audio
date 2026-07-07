# Experience Graphs: The Data Foundation for Self-Improving Agents 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음 한 문장으로 요약됩니다:

> **"Search over experience graphs is a database access pattern."**

장기 수평(long-horizon) 에이전트 작업이 생성하는 탐색 기록—즉 **Experience Graph**—은 일회성 로그가 아니라, **지속 가능하고(queryable), 거버넌스가 적용된 데이터베이스 객체**로 취급되어야 한다는 것입니다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **①** 자기개선 에이전트 시스템 정의 | Inner Loop (에이전트 세션) + Outer Loop (탐색 오케스트레이션) + 영구 데이터 기반의 **두 루프 아키텍처** 공식화 |
| **②** Trellis 데이터 파운데이션 설계 | 기존 3계층 메모리(declarative, procedural, episodic)가 불완전함을 증명하고, 리워드를 포함하는 experience graph를 통합하는 **통합 쿼리 레이어** 제안 |
| **③** 새로운 DB 연구 기회 발굴 | 멀티모달 쿼리 플래닝, 동시 트리 탐색 일관성, 물리적 설계, 양방향 시간 메모리, 멀티에이전트 기관 의미론 등 **7개의 개방형 연구 문제** 식별 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 에이전트 프레임워크의 문제를 구체적으로 다음과 같이 진단합니다:

```
문제 1: 크래시 복구 불가 → Python 객체/JSON 체크포인트에 의존
문제 2: 세션 간 재사용 불가 → 이전 탐색 결과를 새 세션에서 활용 불가
문제 3: 유사도 검색 불가 → 에피소드를 벡터로 인덱싱하지 않음
문제 4: 학습 데이터 추출 불가 → 사후 스크래핑에 의존
문제 5: 거버넌스 부재 → 크로스유저 쿼리 및 접근 정책 관리 불가
```

이를 종합하면, **"Experience Graph가 데이터베이스 객체가 아니기 때문에 발생하는 문제들"** 입니다.

### 2.2 Experience Graph의 정의와 데이터 모델

**Experience Graph** $G = (V, E, R)$ 으로 비공식적으로 표현하면:

$$G = \{v_i \mid v_i = (\text{artifact}_i,\ \text{tool output}_i,\ r_i,\ \text{parent}(v_i),\ \text{siblings}(v_i),\ \text{stats}_i)\}$$

여기서:
- $r_i \in \mathbb{R}$: 객관적 보상(fitness score, speedup 등)
- $\text{stats}_i$: MCTS의 경우 방문 횟수 $n_i$와 누적 보상 $Q_i$
- $\text{parent}(v_i)$: 인과적 선조
- $\text{siblings}(v_i)$: 비교 대상 형제 노드

**논리적 스키마(4계층)**:

```
Tasks → Sessions → Nodes → Prompt Histories
         ↓              ↓
    Embeddings      Artifacts (object storage)
```

### 2.3 탐색 알고리즘과 프런티어 선택

**MCTS UCB 점수** 기반 프런티어 선택:

$$\text{UCB}(v_i) = \frac{Q_i}{n_i} + C \sqrt{\frac{\ln N}{n_i}}$$

여기서:
- $Q_i$: 노드 $v_i$의 누적 보상
- $n_i$: 노드 $v_i$의 방문 횟수
- $N$: 부모 노드의 방문 횟수
- $C$: 탐색/활용 균형 상수

Trellis에서 이 프런티어 선택은 아래와 같은 **SQL 쿼리**로 표현됩니다:

```sql
-- Greedy 전략 예시
SELECT node_id FROM exploration_nodes
WHERE session_id = $sid AND is_leaf = true
ORDER BY fitness_score DESC
LIMIT 1;

-- MCTS 전략 예시
SELECT node_id FROM exploration_nodes
WHERE session_id = $sid
ORDER BY (cumulative_reward / visit_count 
         + $C * SQRT(LN(parent_visits) / visit_count)) DESC
LIMIT 1;
```

### 2.4 벡터 시드 그래프 확장 (핵심 쿼리 패턴)

Cross-session 재사용의 핵심인 **벡터 시드 그래프 확장(vector-seeded graph expansion)**:

$$\text{score}(t_j, q) = \cos(\mathbf{e}_{t_j}, \mathbf{e}_q) > \theta$$

여기서 $\mathbf{e}_{t_j}$는 이전 태스크 $t_j$의 임베딩, $\mathbf{e}_q$는 새로운 태스크 $q$의 임베딩, $\theta = 0.8$은 유사도 임계값.

이를 단일 Cypher+SQL 쿼리로 표현:

```cypher
MATCH (t:tasks)
WHERE t.embedding <~> $q > 0.8
MATCH (t)<-[:BELONGS_TO]-(s:sessions)
MATCH (s)<-[:IN_SESSION]-(n:nodes)
WHERE n.is_buggy = false
  AND policy_allows($user, n)
RETURN t.task_id, score(t), n.node_id,
       n.fitness_score
ORDER BY score(t) DESC, n.fitness_score DESC
LIMIT 10
```

### 2.5 학습 데이터 추출 (From Memory to Training)

#### SFT 궤적

루트에서 리프까지의 경로를 AS-OF 재구성으로 추출:

$$\tau_{\text{SFT}} = (s_0, a_0, s_1, a_1, \ldots, s_T) \mid \forall t:\ r_t \geq r_{\min},\ \neg \text{buggy}(s_t)$$

각 스텝의 상태는 변경 로그의 논리적 스텝 번호 $\ell$를 기준으로 재구성:

$$s_t^{(\ell)} = s_t^{(\text{final})} \ominus \Delta_{(\ell, \text{final}]}$$

#### DPO 선호 쌍

같은 부모를 가진 형제 노드 쌍:

$$\mathcal{D}_{\text{DPO}} = \{(a_i, b_i) \mid \text{parent}(a_i) = \text{parent}(b_i),\ r_{a_i} > r_{b_i} + \delta\}$$

Cypher 쿼리:
```cypher
MATCH (p)-[:HAS_CHILD]->(a), (p)-[:HAS_CHILD]->(b)
WHERE a.fitness_score > b.fitness_score + $m
  AND a.is_buggy = false
RETURN a AS chosen, b AS rejected
```

#### GRPO 그룹

동일 상태 $s$에서 $N$개의 자식 후보 생성 후 그룹 정규화 이점 계산:

$$\hat{A}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}, \quad \mu_{\text{group}} = \frac{1}{N}\sum_{j=1}^N r_j$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{i}\left[\min\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

여기서 $\rho_i = \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)}$.

### 2.6 모델 구조

**Trellis의 4계층 물리적 아키텍처**:

```
┌──────────────────────────────────────────────────────────┐
│              Outer Loop (Search Orchestration)           │
│         MCTS | Evo | Greedy | Linear | One-Shot         │
├──────────────────────────────────────────────────────────┤
│              Inner Loop (Agent Session)                  │
│    Skill → Meta LLM → Sandbox → Evaluate → Persist      │
├──────────────────────────────────────────────────────────┤
│              Trellis — Data Foundation                   │
│  Tasks | Sessions | Nodes | Prompts | Embeddings | Artifacts │
│         SQL      |    Cypher    |    Vector              │
│  Axiom Optimizer · Velox Engine · XDB/MySQL · Hive       │
├──────────────────────────────────────────────────────────┤
│              Training Views (Materialized)               │
│    SFT Trajectories | DPO Pairs | GRPO Groups           │
└──────────────────────────────────────────────────────────┘
```

**핵심 설계 원칙**: 컴퓨트와 상태의 분리 (disaggregation of compute from state)

$$\text{Agent Process} \leftarrow \text{stateless},\quad \text{State} \rightarrow \text{Trellis DB}$$

이는 클라우드 스토리지 분리(Snowflake), 서버리스 함수(AWS Lambda)와 동일한 원칙을 에이전트 시스템에 적용한 것입니다.

### 2.7 성능 향상 (KernelEvolve 실험 결과)

Meta의 KernelEvolve 시스템에서 측정한 결과 (3회 독립 세션 평균):

| 지표 | Baseline (메모리 없음) | CS $p=0.1$ | CS $p=0.5$ |
|------|----------------------|-----------|-----------|
| 버기 노드 비율 | 55% | 34% | 21% |
| 유효 노드 비율 (기준 speedup 충족) | 79.5% | 90.8% | 100% |
| 1.2× speedup 도달 스텝 | 51 스텝 | ~5 스텝 | ~5 스텝 |
| **수렴 가속도** | 기준 | **10×** | **10×** |
| 토큰 비용 절감 | 기준 | **52% 절감** | 52% 절감 |
| 최고 성능 (단일 최적값) | **1.49×** | 1.35× | 1.36× |

여기서 injection rate $p$는:

$$p = P(\text{prior node가 현재 스텝에 컨텍스트로 주입될 확률})$$

### 2.8 한계

논문이 명시적으로 인정하는 한계:

1. **탐색-앵커링 트레이드오프**: $p=0.5$일 때 전략 다양성이 20가지 → 8가지로 감소, 최고 성능이 오히려 낮아짐 (1.36× vs 1.49×)
2. **실험 범위 제한**: KernelEvolve 단일 도메인에서의 측정이며 "a fuller evaluation is left to future work"
3. **일관성 의미론 미해결**: 동시 MCTS 백프로파게이션에서의 정확한 isolation level 미정의
4. **멀티모달 쿼리 비용 모델 부재**: 벡터 인덱스 선택도 + 조인 팬아웃 + 그래프 순회 비용의 통합 최적화 미해결
5. **소규모 실험**: 세션당 ~100 노드, 세션당 3회 반복으로 통계적 유의성 제한

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 가장 직접적으로 연결되는 메커니즘은 **자기개선 플라이휠(Self-Improving Flywheel)** 입니다.

### 3.1 일반화의 구조적 메커니즘

**플라이휠 순환 공식**:

$$\text{Better Models} \xrightarrow{\text{Better Exploration}} \text{Better Data} \xrightarrow{\text{SFT/DPO/GRPO}} \text{Better Models}$$

더 구체적으로:

$$\mathcal{M}_{t+1} = \text{Train}(\mathcal{M}_t,\ \mathcal{D}_t),\quad \mathcal{D}_{t+1} = \text{Query}(G_{t+1}),\quad G_{t+1} = \text{Explore}(\mathcal{M}_{t+1})$$

이 순환이 컴파운딩되면서 도메인 내 일반화가 개선됩니다.

### 3.2 Cross-Session 재사용이 일반화에 기여하는 방식

**벡터 유사도 기반 지식 전이**:

$$\text{sim}(q, t_j) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{t_j}}{\|\mathbf{e}_q\| \|\mathbf{e}_{t_j}\|} \geq \theta$$

이 조건을 만족하는 이전 태스크의 고품질 서브트리를 새 세션의 시작점으로 활용함으로써:
- 유사하지만 다른 태스크로의 **지식 전이(knowledge transfer)** 가 가능
- 각 세션이 처음부터 탐색을 시작하지 않아도 됨 → **탐색 효율성 개선**

### 3.3 도메인 일반화 증거: KernelEvolve → MTIA 실리콘 검증

논문은 **단일 infrastructure 재사용**으로 도메인 일반화를 실증합니다:

```
KernelEvolve (GPU 커널 최적화)
    ↓ fitness function + skills만 변경
MTIA 실리콘 하드웨어 검증
    - 보상: speedup → 버그 발견
    - ISA 커버리지 매트릭스로 세션 간 탐색 공백 관리
    - experience graph, 쿼리 레이어, 학습 뷰 → 변경 없이 재사용
```

이는 **Trellis의 infrastructure 자체가 태스크 무관(task-agnostic)** 함을 증명합니다.

### 3.4 Value Model을 통한 외부 루프 일반화

**그래프 피처 기반 가치 모델**:

$$V(v_i) = f(\text{ancestor rewards},\ \text{sibling diversity},\ \text{failure signatures},\ \text{artifact diffs},\ \text{visit counts},\ \text{budget remaining})$$

이 가치 모델이 누적되면:
- 어떤 프런티어 노드를 확장할지 학습
- 어떤 이전 서브트리를 재사용할지 학습
- 검증자(verifier) 호출 시점 학습

→ 고정된 탐색 휴리스틱(UCB 등)에서 **학습된 탐색 정책**으로의 전환이 가능해져 새로운 도메인에서의 탐색 효율을 높일 수 있습니다.

### 3.5 일반화의 한계: 탐색-앵커링 트레이드오프

일반화 성능 향상에 있어 근본적인 긴장 관계:

$$\underbrace{\text{높은 } p}_{\text{빠른 수렴, 낮은 다양성}} \xleftrightarrow{\text{tradeoff}} \underbrace{\text{낮은 } p}_{\text{느린 수렴, 높은 다양성}}$$

실험적으로: $p=0.5$에서 탐색 전략이 20개 → 8개로 붕괴, 단일 최고 성능은 오히려 no-memory (1.49×) 조건이 우세.

이 트레이드오프는 **retrieval policy를 적응적(adaptive)으로** 설계해야 함을 시사합니다:

$$p^*(t) = \arg\max_p \mathbb{E}[\text{quality}(G_T) \mid p,\ \text{session context at step } t]$$

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 데이터베이스 연구에 미치는 영향

#### 새로운 워크로드 패러다임

기존 워크로드와 비교한 Experience Graph의 독특한 접근 패턴:

| 접근 패턴 | 기존 DB | Trellis의 요구 |
|-----------|---------|---------------|
| 쓰기 | 단순 CRUD | Append + 경로 업데이트 (MCTS 역전파) 혼합 |
| 읽기 | 단일 모달 | 순서 스캔 + 멀티홉 그래프 + 벡터 + 풀 스캔 동시 |
| 일관성 | ACID 또는 최종 일관성 | **혼합**: 노드 삽입은 Durability 필요, 통계는 최종 일관성 허용 |
| 시간 | 단일 시간 축 | **Bi-temporal** (유효 시간 + 트랜잭션 시간) |
| 거버넌스 | 테이블 수준 | **View propagation**: 소스 노드 철회 시 파생 학습 예제도 철회 |

#### 개방형 연구 문제들

논문이 제시하는 7가지 연구 문제:

1. **멀티모달 쿼리 플래닝**: 벡터 인덱스 선택도 → 조인 팬아웃 → 그래프 순회 비용의 통합 비용 모델
2. **동시 트리 탐색 일관성**: MCTS 백프로파게이션의 적절한 isolation level 형식화
3. **물리적 설계 최적화**: 위의 혼합 접근 패턴에 최적화된 스토리지 레이아웃
4. **거버넌스 뷰 유지보수**: 소스 철회 → 파생 예제 무효화 연쇄
5. **검색 정책 및 메모리 품질**: 주입 비율의 비용 기반, 품질 인식 최적화
6. **양방향 시간 메모리**: 유효 시간 + 트랜잭션 시간의 쿼리 레이어 통합
7. **멀티에이전트 기관 의미론**: 과학 사회 시뮬레이션을 위한 트랜잭션 의미론

### 4.2 AI/ML 연구에 미치는 영향

#### Inference-Time Compute 패러다임 변화

논문의 핵심 기여 중 하나는 추론 시간 탐색을 **일회성 계산에서 지속 가능한 기관 자산으로** 전환시킨다는 것입니다:

$$\underbrace{\text{Inference-time Search}}_{\text{기존: 소모성}} \rightarrow \underbrace{\text{Durable Institutional Asset}}_{\text{Trellis: 누적적}}$$

이는 2020년 이후 활발히 연구되는 Test-Time Compute(TTC) 연구 방향과 직접적으로 연결됩니다.

#### 강화학습 데이터 파이프라인 혁신

기존 RLHF/RLAIF 파이프라인의 병목:

```
기존: 탐색 → 로그 → 스크래핑 → 정제 → 학습
Trellis: 탐색 = 학습 데이터 생성 (materialized view로 직접 추출)
```

이는 특히 GRPO와 같은 그룹 기반 방법에서 **롤아웃 비용을 단일 스텝 확장으로 축소**시켜 학습 효율을 획기적으로 개선할 수 있습니다.

### 4.3 앞으로 연구 시 고려할 점

#### 기술적 고려 사항

**① 임베딩 공간의 드리프트 문제**

모델이 업데이트될 때마다 태스크 임베딩 공간이 변화하므로:

$$\mathbf{e}_{t}^{(\mathcal{M}_{t+1})} \neq \mathbf{e}_{t}^{(\mathcal{M}_t)}$$

이전 버전 모델로 생성된 임베딩과 새 모델 임베딩 간의 호환성 문제를 해결해야 합니다. **임베딩 마이그레이션 전략** 또는 **모델 버전 인식 벡터 인덱스**가 필요합니다.

**② 메모리 품질 저하 (Memory Decay)**

시간이 지남에 따라:
- 하드웨어 환경 변화 (새 GPU 세대)
- 컴파일러/런타임 버전 변경
- 기준선(baseline) 변화

로 인해 과거 high-fitness 노드가 현재 환경에서는 낮은 품질일 수 있습니다. **유효 시간(valid time) 기반 메모리 품질 스코어**:

$$\text{quality}(v_i, t) = r_i \cdot \exp\left(-\lambda \cdot \Delta t\right) \cdot \mathbb{1}[\text{env}(v_i) \approx \text{env}(t)]$$

**③ 확장성 병목**

경험 그래프가 기하급수적으로 증가할 때:
- 벡터 인덱스 재구축 비용
- 멀티홉 그래프 순회의 깊이 제한 필요
- 학습 뷰 유지보수 비용

**distributed graph partitioning** 전략이 중요한 연구 과제가 됩니다.

**④ 보상 해킹(Reward Hacking) 위험**

자기개선 루프에서 에이전트가 fitness 함수를 최적화하는 방향으로 편향될 수 있습니다. 이 경우:

$$r_{\text{hacked}} \gg r_{\text{true}},\quad \text{but quality degrades}$$

**거버넌스 레이어에서의 보상 검증** 메커니즘이 필수적입니다.

#### 사회적/윤리적 고려 사항

**⑤ 거버넌스의 복잡성**

- 생성된 코드, 하드웨어 신호 등이 포함된 trace의 접근 정책
- 소스 노드 철회 시 파생 학습 예제의 연쇄 무효화
- 멀티에이전트 환경에서의 지식 귀속(attribution) 문제

**⑥ 중앙화 vs. 분산화**

Trellis는 중앙집중식 데이터 파운데이션을 가정하지만, 실제 조직 경계를 넘는 협업에서는 **연합 학습(federated learning) 스타일의 분산 experience graph**가 필요할 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 에이전트 메모리 시스템

| 연구 | 연도 | 핵심 기여 | Trellis와의 차이 |
|------|------|-----------|----------------|
| **MemGPT** (Packer et al.) | 2023 | OS 메모리 계층 유추: fast(context) + slow(external) | transport layer; Trellis는 storage engine. 보상 그래프 없음 |
| **Mem0** (Chhikara et al.) | 2025 | 프로덕션용 장기 메모리 서비스 | 구조화 메모리 추가, 그러나 reward-bearing graph, 시간 여행 없음 |
| **Graphiti** (Rasmussen et al.) | 2025 | 실시간 시간 지식 그래프 합성 | 기업 메모리 지향, RSI experience graph 없음, CDC/학습 뷰 없음 |
| **Trellis** (본 논문) | 2026 | Experience graph를 DB 객체로 | reward + causal graph + 학습 뷰 + 시간 여행 통합 |

### 5.2 재귀적 자기개선(RSI) 시스템

| 연구 | 연도 | 방법 | 저장 방식 | Trellis 대비 |
|------|------|------|-----------|-------------|
| **FunSearch** (Romera-Paredes et al.) | 2024 | 진화적 프로그램 탐색 + LLM | Python 객체 | 데이터 기반 없음, 세션 간 재사용 불가 |
| **AlphaEvolve** (Novikov et al.) | 2025 | 코딩 에이전트 + 진화 탐색 | JSON 체크포인트 | 크래시 복구, 크로스 세션 불가 |
| **AI Scientist v2** (Yamada et al.) | 2025 | 에이전트 트리 탐색 기반 과학 발견 | 프로세스 로컬 상태 | Trellis가 해결하는 문제들 존재 |
| **AIDE** (Jiang et al.) | 2025 | ML 엔지니어 에이전트 | 파일 기반 | 동일 |
| **KernelEvolve + Trellis** | 2026 | MCTS + 진화 탐색 + DB 파운데이션 | **Trellis DB** | 크래시 복구, 10× 수렴 가속, 52% 비용 절감 |

### 5.3 Test-Time Compute (TTC) 연구와의 관계

최근 TTC 연구들(OpenAI o1/o3, DeepSeek-R1)은 추론 시간에 더 많은 계산을 투입해 성능을 향상시키는 방향을 탐구합니다:

$$\text{성능} \propto f(\text{추론 시간 계산량})$$

Trellis는 이를 **데이터 인프라 관점에서 보완**합니다:
- TTC 연구: 단일 세션 내 탐색 개선
- Trellis: **세션 간 탐색 결과를 누적**하여 점진적 개선

$$\underbrace{\text{TTC (단일 세션)}}_{\text{일회성}} + \underbrace{\text{Trellis (세션 간 누적)}}_{\text{지속적}} = \text{진정한 자기개선}$$

### 5.4 에이전트 오케스트레이션 프레임워크

| 연구 | 연도 | 핵심 | Trellis 대비 |
|------|------|------|-------------|
| **Omnigent** (Zaharia et al.) | 2026 | 멀티에이전트 메타 하네스, 정책 거버넌스 | Control plane. Data plane(지속/쿼리/버전/학습)은 미해결 |
| **Sakana Fugu** (Fugu Team) | 2026 | 오케스트레이터 학습, GRPO 기반 학습 | 공유 상태 필요성 증명, 그러나 data foundation 미설계 |
| **AIRA-Compose** (Pepe et al.) | 2026 | 에이전트 신경 아키텍처 발견 | 특정 도메인에 한정 |

### 5.5 학습 데이터 인프라

| 연구 | 연도 | 핵심 | Trellis 대비 |
|------|------|------|-------------|
| **DeepSeek-R1** (Guo et al.) | 2025 | GRPO 기반 RL로 추론 능력 개선 | 학습 데이터 생성 파이프라인이 탐색과 분리 |
| **MLflow** (Zaharia et al.) | 2018 | ML 실험 추적 | 오프라인 사이드카; 온라인 프런티어 쿼리, 시간 여행, 그래프 순회 없음 |
| **Trellis** | 2026 | 탐색 = 학습 데이터 생성 (materialized view) | 운영 스토어와 학습 파이프라인 통합 |

---

## 참고 자료

본 논문에서 직접 인용된 주요 참고문헌:

1. **Liao et al. (2026)** — "Experience Graphs: The Data Foundation for Self-Improving Agents" *(본 논문)*, arXiv:2606.29823v1
2. **Novikov et al. (2025)** — "AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery", arXiv:2506.13131
3. **Romera-Paredes et al. (2024)** — "Mathematical Discoveries from Program Search with Large Language Models", *Nature*, Vol. 625
4. **Packer et al. (2023)** — "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560
5. **Chhikara et al. (2025)** — "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory", arXiv:2504.19413
6. **Rasmussen et al. (2025)** — "Graphiti: Building Real-Time Knowledge Graphs for AI Agents", arXiv:2501.13956
7. **Yamada et al. (2025)** — "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search", arXiv:2504.08066
8. **Jiang et al. (2025)** — "AIDE: The Machine Learning Engineer Agent", arXiv:2502.13138
9. **Guo et al. (2025)** — "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948
10. **Rafailov et al. (2023)** — "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model", *NeurIPS*
11. **Shao et al. (2024)** — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" *(GRPO)*, arXiv:2402.03300
12. **Pedreira et al. (2022)** — "Velox: Meta's Unified Execution Engine", *VLDB Endowment* 15(12)
13. **Francis et al. (2018)** — "Cypher: An Evolving Query Language for Property Graphs", *SIGMOD*
14. **Browne et al. (2012)** — "A Survey of Monte Carlo Tree Search Methods", *IEEE Transactions on Computational Intelligence and AI in Games*
15. **Kocsis & Szepesvári (2006)** — "Bandit Based Monte-Carlo Planning", *ECML*
16. **Hu et al. (2025)** — "Memory in the Age of AI Agents: A Survey", arXiv:2512.13564
17. **Zaharia et al. (2026)** — "Introducing Omnigent: A Meta-Harness to Combine, Control and Share Your Agents", Databricks Blog
18. **Fugu Team, Sakana AI (2026)** — "Sakana Fugu Technical Report"
19. **Park et al. (2023)** — "Generative Agents: Interactive Simulacra of Human Behavior", *UIST*
20. **Stonebraker & Çetintemel (2005)** — "'One Size Fits All': An Idea Whose Time Has Come and Gone", *ICDE*

> **⚠️ 투명성 고지**: 본 답변은 제공된 PDF 원문(arXiv:2606.29823v1)에 근거하여 작성되었으며, 논문 외부의 정보(예: 다른 시스템의 세부 구현 사항)는 논문이 인용한 내용 범위 내에서만 기술하였습니다. 논문이 "a fuller evaluation is left to future work"라고 명시한 부분은 그대로 한계로 기술하였으며, 확인되지 않은 수치는 포함하지 않았습니다.
