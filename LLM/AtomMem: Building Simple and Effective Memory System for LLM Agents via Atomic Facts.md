# AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts

> **참고 자료**: Yao et al. (2026). "AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts." arXiv:2606.19847v1 [cs.CL], 18 Jun 2026. (제공된 PDF 전문을 1차 자료로 사용)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

AtomMem은 LLM 에이전트의 **장기 메모리(Long-term Memory) 시스템**이 직면한 세 가지 근본 문제를 동시에 해결하고자 한다:

| 기존 문제 | AtomMem의 해결 방향 |
|---|---|
| 비효율적 메모리 표현 (원본 대화 저장 시 노이즈 과부하 또는 요약 시 정보 손실) | **원자적 사실(Atomic Fact)** 기반 고밀도 표현 |
| 불안정한 메모리 갱신 (LLM 기반 무제약 수정으로 인한 환각 누적) | **제약적·점진적 갱신 메커니즘** |
| 평면적(Flat) 검색의 한계 (세션 간 연관 증거 복구 실패) | **연관 메모리 그래프(Associative Memory Graph)** 기반 계층적 검색 |

### 1.2 주요 기여 (3가지)

1. **AtomMem 프레임워크 제안**: 원자적 사실 중심의 장기 메모리 시스템으로, 그래프 기반 연상 회상(Associative Recall)을 통해 메모리 인식 응답 생성
2. **Fact Executor 및 고품질 데이터셋 공개**: SFT(Supervised Fine-Tuning)로 훈련된 경량 LLM이 대화에서 자기 완결적(self-contained) 원자적 사실을 추출; 4,352개의 고품질 훈련 샘플 공개
3. **LoCoMo 벤치마크 SOTA 달성**: 단순화 변형(AtomMem-Flat)도 최소 토큰(722K)으로 경쟁력 있는 성능 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

LLM 기반 에이전트는 고정 컨텍스트 윈도우로 인해 장기 다중 세션에서 정보를 축적·재사용하기 어렵다. 기존 접근법의 딜레마는 다음과 같다:

$$\text{원본 대화 저장} \xrightarrow{\text{정보 과부하}} \text{RAG 성능 저하 (노이즈)}$$
$$\text{압축 요약 저장} \xrightarrow{\text{정보 손실}} \text{세밀한 사실 소실 + 오류 누적}$$

이 두 극단 사이의 균형점으로서 **원자적 사실(Atomic Fact)** 이라는 중간 표현을 제안한다.

---

### 2.2 제안 방법 및 수식

#### 2.2.1 원자적 사실 추출기 (Atomic Fact Extractor) 훈련

SFT 기반 경량 LLM을 훈련하여 대화 컨텍스트 $C$와 지시 $I$로부터 목표 원자적 사실 집합 $F$를 생성한다:

$$\max_{\theta} \sum_{(I, C, F) \in \mathcal{D}} \log P_{\theta}(F \mid I, C) \tag{1}$$

- $\theta$: 모델 파라미터 (Qwen3-14B + LoRA, rank=128)
- $\mathcal{D}$: 4,352개 고품질 샘플로 구성된 전용 데이터셋

#### 2.2.2 구조화된 원자적 사실 정의

각 원자적 사실 $F$는 다음과 같이 정의된다:

$$F = \{id, \; c, \; \mathbf{v}, \; \mathcal{P}, \; \mathcal{K}, \; \mathcal{T}, \; \mathcal{E}\}$$

| 필드 | 의미 |
|---|---|
| $id$ | 사실 식별자 |
| $c$ | 자기 완결적 텍스트 |
| $\mathbf{v}$ | 밀집 의미 임베딩 벡터 |
| $\mathcal{P}$ | 관련 참여자 |
| $\mathcal{K}$ | 주제 키워드 집합 |
| $\mathcal{T}$ | 시간 정보 (타임스탬프/구간) |
| $\mathcal{E}$ | 연관 이벤트 ID 목록 |

#### 2.2.3 하이브리드 유사도 메트릭

두 입력 $x$, $y$ 간의 보편적 하이브리드 유사도:

$$S_h(x, y) = \alpha \cdot \text{sim}_e(\mathbf{v}_x, \mathbf{v}_y) + \beta \cdot \text{Jac}(\mathcal{K}_x, \mathcal{K}_y) \tag{2}$$

- $\alpha = 0.7$: 의미 임베딩 유사도 가중치
- $\beta = 0.3$: 키워드 Jaccard 유사도 가중치
- $\text{sim}_e$: 코사인 유사도 (all-MiniLM-L6-v2 임베딩 사용)

#### 2.2.4 사실 검증 (Fact Verification)

새로운 사실 $F_{new}$와 검색된 후보 $C_{ret}$ 간의 관계를 LLM이 분석:

$$(c'_{new}, \mathcal{U}) \leftarrow \text{LLM}(c_{new} \| C_{ret})$$

- $c'_{new}$: 기존 컨텍스트에 포함되지 않은 잔여 새 정보
- $\mathcal{U}$: 논리적 충돌 감지 시 기존 사실에 대한 갱신 튜플 집합

#### 2.2.5 이벤트 메모리 구조

관련 사실들을 응집적 에피소딕 블록으로 집계:

$$E = \{id, \; \mathcal{S}, \; \mathcal{F}_{ids}, \; \mathcal{P}_e, \; \mathcal{K}_e, \; \mathcal{T}_e\}$$

#### 2.2.6 시간 프로파일 구조

사용자 상태 진화를 추적하는 장기 속성:

$$P = \{id, \; u, \; c, \; v_p, \; \mathcal{K}_p, \; \mathcal{E}_{evi}, \; t_{from}, \; \mathcal{H}\}$$

- $\mathcal{H}$: 과거 버전 이력 저장소

#### 2.2.7 연관 메모리 그래프의 세 가지 엣지

**(a) 엔티티 엣지 (Entity Edge)**: IDF 가중 키워드 중복

$$w_{kw}(F_i, F_j) = \frac{\sum_{k \in \mathcal{K}_i \cap \mathcal{K}_j} \omega(k)}{\sqrt{\sum_{k \in \mathcal{K}_i} \omega(k) \cdot \sum_{k \in \mathcal{K}_j} \omega(k)} + \epsilon} \tag{3}$$

쿼리 인식 가중치:

$$\omega(k) = \text{IDF}(k) \cdot \beta_q(k) \cdot \pi(k) \tag{6}$$

**(b) 이벤트 엣지 (Event Edge)**: 이벤트 크기에 따른 패널티 포함

$$w_{event}(F_i, F_j) = \sum_{e \in \mathcal{E}_i \cap \mathcal{E}_j} \frac{1}{(|\mathcal{F}_e| - 1)^{\gamma_e}} \tag{4}$$

**(c) 시간 엣지 (Temporal Edge)**: 대화 턴 거리 기반 지수 감쇠

$$w_{turn}(F_i, F_j) = \exp\left(-\frac{|\text{pos}_i - \text{pos}_j|}{\tau}\right) \tag{5}$$

#### 2.2.8 멀티채널 전이 행렬

각 채널 $c \in \{kw, event, turn\}$의 전이 확률을 동적으로 융합:

$$\mathbf{P}_{i,j} = \sum_{c \in \mathcal{C}_i} \bar{\rho}_{i,c} (\mathbf{P}_c)_{i,j} \tag{7}$$

$$\bar{\rho}_{i,c} = \frac{\rho_c}{\sum_{c' \in \mathcal{C}_i} \rho_{c'}} \tag{8}$$

#### 2.2.9 Personalized PageRank (PPR) 기반 연상 회상

시드 사실의 점수로 재시작 분포를 초기화:

$$p_i = \frac{\tilde{s}_i^{\gamma_s}}{\sum_{F_j \in \mathcal{R}_{seed}} \tilde{s}_j^{\gamma_s}} \tag{9}$$

Random Walk with Restart (RWR) 반복:

$$\mathbf{r}^{(t+1)} = \eta \mathbf{p} + (1 - \eta) \mathbf{P}^T \mathbf{r}^{(t)} \tag{10}$$

수렴 조건:

$$\|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 < \epsilon_c \tag{11}$$

---

### 2.3 모델 구조 (전체 아키텍처)

```
[원시 대화 스트림]
        ↓
┌─────────────────────────────────────┐
│        메모리 구성 단계               │
│  ① Atomic Fact Extractor (SFT)      │
│     - 공참조 해소, 시간 앵커링         │
│     - 저가치 내용 필터링              │
│  ② Structured Fact Construction    │
│     - {id, c, v, P, K, T, E} 구조화  │
│  ③ Fact Verification               │
│     - 중복/충돌 감지 및 처리          │
│  ④ Event Memory (에피소딕 통합)      │
│     - 관련 사실을 이벤트 블록으로 집계  │
│  ⑤ Temporal Profile (상태 진화)     │
│     - 세션 배치 방식 프로파일 갱신     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│        메모리 검색 단계               │
│  ① Query Intent Analysis           │
│     - Qparsed = {Iprof, Pq, Kq, Tq} │
│  ② Hierarchical Hybrid Retrieval   │
│     - Primary Recall (직접 사실)     │
│     - Compensatory Recall (이벤트)   │
│     - Associative Recall (PPR 그래프) │
│  ③ Profile Augmentation           │
│     - 사용자 프로파일 보강            │
│  ④ Response Generation            │
└─────────────────────────────────────┘
        ↓
    [최종 응답]
```

---

### 2.4 성능 향상

#### LoCoMo 벤치마크 결과 (Table 1)

| 방법 | Single-Hop J↑ | Multi-Hop J↑ | Temporal J↑ | Open Domain J↑ | Tokens(K)↓ |
|---|---|---|---|---|---|
| LoCoMo | 57.07 | 48.58 | 38.01 | 41.67 | 827.20 |
| A-MEM | 54.10 | 43.26 | 34.89 | 32.29 | 11,687.58 |
| MEM0 | 78.00 | 62.41 | 30.53 | 54.17 | 55,300.30 |
| MemoryOS | 66.47 | 60.99 | 34.58 | 51.04 | 19,207.67 |
| LightMem | 68.97 | 64.89 | 51.09 | 43.75 | 5,021.56 |
| **AtomMem-Flat** | 67.66 | 55.67 | 59.50 | 52.08 | **722.75** |
| **AtomMem** | **78.48** | **68.44** | **66.98** | **64.58** | 21,357.06 |

**핵심 성능 하이라이트:**
- Temporal 태스크: 최강 기존 베이스라인(LightMem) 대비 **J-score 31.1% 향상**
- Multi-Hop 태스크: LightMem 대비 **J-score 5.5% 향상**
- Open Domain: MEM0 대비 **J-score 19.2% 향상** (54.17 → 64.58)
- MEM0 대비 **토큰 소비 61.4% 절감** (55,300K → 21,357K)

#### Ablation 분석 (Table 2)

| 변형 | Multi-Hop J | Temporal J | Open Domain J |
|---|---|---|---|
| AtomMem-Flat (계층 없음) | 55.67 | 59.50 | 52.08 |
| w/o Profile | 59.22 | 62.93 | 54.17 |
| w/o Graph | 62.76 | 62.93 | 60.42 |
| **Full AtomMem** | **68.44** | **66.98** | **64.58** |

---

### 2.5 한계

1. **LLM 의존성**: 여러 단계가 기반 LLM의 생성 안정성에 민감하여, 기반 모델의 환각이 시스템 전체에 영향
2. **텍스트 모달리티 한정**: 이미지, 오디오 등 멀티모달 입력 미지원
3. **토큰 효율성의 추가 개선 여지**: Full AtomMem은 21,357K 토큰 소비로 아직 최적화 가능
4. **하이퍼파라미터 민감도**: 수많은 하이퍼파라미터(Table 5에 약 30개 이상)가 성능에 영향을 미쳐 도메인별 재조정 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지지하는 설계 요소

#### (1) 도메인 독립적 원자적 사실 표현

SFT 훈련 데이터를 **LoCoMo 평가셋과 완전히 분리된 캐릭터와 시나리오**로 구성했다:
- 훈련셋: Elena (요리사), Kenji (소설가) 등 커스텀 캐릭터
- 평가셋: Caroline, Melanie 등 별도 캐릭터

이 제로-오버랩 설계는 Fact Executor의 **도메인 전이 능력**을 검증하는 간접 증거가 된다.

#### (2) LongMemEval 교차 검증

LoCoMo와 완전히 독립적인 LongMemEval 벤치마크에서도 강건한 성능을 보였다:

$$\text{SSU: } F_1 = 80.70, \quad \text{KU: } F_1 = 66.35, \quad \text{SSP: } J = 80.00$$

Knowledge Update(KU) 카테고리에서의 높은 F1(66.35)은 **동적 지식 변화 추적 능력**의 일반화를 시사한다.

#### (3) 구조적 일반화 능력

- **Temporal Profile의 버전 이력 관리**: 사용자 선호가 시간에 따라 변화하더라도 과거 상태를 보존하며 적응 → 다양한 사용자 행동 패턴에 일반화 가능
- **PPR 기반 연상 그래프**: 사전에 명시적 관계를 정의하지 않아도 엔티티/이벤트/시간 엣지를 통해 **새로운 연결 패턴을 유연하게 발견**

#### (4) AtomMem-Flat의 의의

계층 구조 없이 원자적 사실만으로도 강력한 성능을 달성:
- Multi-Hop F1: 원본 LoCoMo 20.97 → AtomMem-Flat **37.03** (**76.6% 상대적 향상**)
- 이는 **원자적 사실이라는 핵심 표현이 특정 시스템 구조에 의존하지 않고 일반화**됨을 보여줌

### 3.2 일반화를 제한하는 요소

```
일반화 위협 요인:
├── LLM 백본 의존성: GPT-4o-mini 기준으로 최적화 → 다른 LLM에서 성능 변동 가능
├── 영어 중심 데이터: LoCoMo/LongMemEval 모두 영어 기반 → 다국어 일반화 미검증
├── 대화 도메인 특화: 개인 대화 중심으로 전문 도메인(의료, 법률 등)에서의 일반화 불명확
└── 하이퍼파라미터 민감도: α, β, we, ρ 등 30+개 파라미터가 벤치마크에 최적화됨
```

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 대상 연구 분류

#### 카테고리 1: RAG 기반 접근

| 연구 | 방법 | AtomMem과의 차이 |
|---|---|---|
| Lewis et al. (2020) - RAG | 검색-읽기 파이프라인 기반 | AtomMem은 메모리 구조 자체를 고품질화하여 RAG 노이즈 문제 근본 해결 |
| Guu et al. (2020) - REALM | 검색 결합 언어모델 사전훈련 | AtomMem은 추론 시점의 검색에 집중, 사전훈련 비용 불필요 |
| Asai et al. (2024) - Self-RAG | 자기 반성적 동적 검색 | AtomMem은 메모리 표현 품질을 우선시, 검색 타이밍보다 정보 밀도에 집중 |

#### 카테고리 2: 메모리 표현 방식 비교

| 연구 | 메모리 표현 | 업데이트 방식 | 검색 방식 | 한계 |
|---|---|---|---|---|
| MemoryBank (Zhong et al., 2023) | 원본 텍스트 경험 | 에빙하우스 망각 곡선 | 평면 검색 | 노이즈 과부하 |
| Think-in-Memory (Liu et al., 2023) | 진화하는 사고 | LLM 재작성 | 평면 검색 | 무제약 업데이트 불안정 |
| RET-LLM (Modarressi et al., 2023) | 삼중항(Triple) | 규칙 기반 | 구조적 검색 | 세밀한 맥락 손실 |
| Mem0 (Chhikara et al., 2025) | 지식 그래프 | LLM 재작성 | 그래프 검색 | 토큰 소비 과다(55,300K) |
| A-MEM (Xu et al., 2025) | 동적 노트 | 자기 조직화 | 동적 검색 | 무제약 진화 불안정 |
| MemGPT (Packer et al., 2023) | 계층적 페이지 | OS 방식 관리 | 계층적 | 구현 복잡도 높음 |
| **AtomMem** | **원자적 사실** | **제약적 점진 갱신** | **PPR 그래프** | **LLM 의존, 단일 모달** |

#### 카테고리 3: 최근 학습 기반 메모리

| 연구 | 특징 | AtomMem과의 관계 |
|---|---|---|
| MEM1 (Zhou et al., 2025) | 메모리와 추론 시너지 학습 | AtomMem은 추론을 검색 단계에서 처리, MEM1은 강화학습 기반 |
| RMM (Tan et al., 2025) | 다중 세분화 대화 요약 | AtomMem이 더 세밀한 원자 단위 표현 사용 |
| LightMem (Fang et al., 2026) | 경량 효율적 메모리 증강 | AtomMem-Flat보다 토큰 효율 낮음(5,021K vs 722K) |

### 4.2 포지셔닝 분석

```
정보 밀도
    ↑
High│    AtomMem ★
    │         LightMem
    │    MemoryOS
    │         Mem0
    │
Low │ LoCoMo(원본) MemoryBank A-MEM
    └──────────────────────────────→
       낮음      검색 복잡도      높음
```

AtomMem은 높은 정보 밀도와 높은 검색 복잡도를 동시에 추구하되, **원자적 사실이라는 고품질 기반 표현**으로 두 목표를 균형 있게 달성한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 메모리 표현 패러다임의 재정의

AtomMem은 **"메모리 시스템의 성능은 알고리즘보다 표현 품질이 결정적"** 이라는 명제를 실험적으로 입증했다. AtomMem-Flat의 결과는 앞으로의 메모리 연구가 복잡한 시스템 설계보다 **기반 표현의 충실도**를 우선적으로 고려해야 함을 시사한다.

#### (2) SFT 기반 정보 추출의 표준화 가능성

대화에서 고품질 원자적 사실을 추출하는 SFT 파이프라인과 데이터셋이 공개되어, **후속 연구의 기반 인프라**로 활용될 수 있다. 특히 다국어·멀티모달 확장 연구의 출발점이 될 것이다.

#### (3) 그래프 기반 에피소딕 메모리 연구 촉진

PPR + 멀티채널 그래프의 조합은 **신경과학의 연상 기억(Associative Memory) 이론**을 AI 시스템에 구현한 사례로, 관련 분야(인지 컴퓨팅, 뇌-컴퓨터 인터페이스)에도 영향을 미칠 수 있다.

#### (4) 개인화 에이전트 평가 프레임워크의 발전

LoCoMo, LongMemEval을 넘어 PERMA(Liu et al., 2026)와 같은 동적 사용자 선호 추적 벤치마크의 중요성이 부각될 것이다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 멀티모달 원자적 사실 확장

현재는 텍스트만 처리하나, 실제 대화는 이미지·오디오·비디오를 포함한다. 향후 연구는 다음을 고려해야 한다:

$$F_{multi} = \{id, \; c_{text}, \; c_{visual}, \; c_{audio}, \; \mathbf{v}_{fused}, \; \mathcal{P}, \; \mathcal{K}, \; \mathcal{T}, \; \mathcal{E}\}$$

멀티모달 임베딩 공간에서의 하이브리드 유사도 재설계가 필요하다.

#### (2) 온라인 학습 및 적응적 Fact Executor

현재 Fact Executor는 정적 SFT 모델이다. 사용자의 대화 스타일이나 도메인에 **온라인으로 적응**하는 메커니즘이 필요하다:
- Continual Learning 기법 적용
- RLHF(Reinforcement Learning from Human Feedback)를 통한 사실 품질 피드백

#### (3) 하이퍼파라미터 자동 최적화

약 30개 이상의 하이퍼파라미터가 수동으로 설정되었다. 향후 연구에서는:
- **AutoML/Bayesian Optimization** 기반 자동 튜닝
- 쿼리 유형별 동적 하이퍼파라미터 조정 메커니즘

#### (4) 메모리 망각 메커니즘 설계

현재 AtomMem은 명시적 망각(Forgetting) 메커니즘이 없다. 시간이 지남에 따라 메모리가 무한히 축적될 경우의 확장성 문제를 해결하기 위해 **에빙하우스 망각 곡선** 또는 **중요도 기반 가지치기**를 도입할 필요가 있다.

#### (5) 메모리 프라이버시 및 보안

개인화 에이전트가 사용자의 장기 정보를 저장함에 따라 **차등 프라이버시(Differential Privacy)**, **연합 학습(Federated Learning)** 등의 프라이버시 보호 메커니즘 연구가 병행되어야 한다.

#### (6) 다국어 및 문화 다양성 일반화

현재 영어 중심 벤치마크에서만 검증되었다. 한국어, 중국어 등 형태론적으로 다른 언어에서의 원자적 사실 추출 품질 및 키워드 Jaccard 유사도의 유효성을 별도로 검증해야 한다.

#### (7) LLM 백본 독립성 검증

GPT-4o-mini 기반 실험만 수행되었다. 오픈소스 모델(Llama, Mistral 등)이나 소형 모델에서의 성능 변동 폭을 분석하여 **시스템 로버스트니스**를 입증할 필요가 있다.

---

## 참고 자료 목록

**1차 자료 (논문 PDF 직접 분석)**
- Yao, Y., Li, S., Zheng, Z., Zheng, H., Liu, Q., Xu, T., & Chen, E. (2026). *AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts.* arXiv:2606.19847v1 [cs.CL]

**논문 내 인용 참고 자료 (검증 목적)**
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Guu, K. et al. (2020). *Retrieval Augmented Language Model Pre-Training.* ICML 2020.
- Maharana, A. et al. (2024). *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL 2024.
- Wu, D. et al. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025.
- Chhikara, P. et al. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.* arXiv:2504.19413.
- Xu, W. et al. (2025). *A-MEM: Agentic Memory for LLM Agents.* NeurIPS 2025.
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.
- Zhou, Z. et al. (2025). *MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents.* arXiv:2506.15841.
- Tan, Z. et al. (2025). *In Prospect and Retrospect: Reflective Memory Management for Long-Term Personalized Dialogue Agents.* ACL 2025.
- Liu, S. et al. (2026). *PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments.* arXiv:2603.23231.
- Asai, A. et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection.* ICLR 2024.
- Kang, J. et al. (2025). *Memory OS of AI Agent.* EMNLP 2025.
- Fang, J. et al. (2026). *LightMem: Lightweight and Efficient Memory-Augmented Generation.* ICLR 2026.
- Modarressi, A. et al. (2023). *RET-LLM: Towards a General Read-Write Memory for Large Language Models.* arXiv:2305.14322.
- Park, J.S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.

> ⚠️ **정확도 고지**: 본 답변은 제공된 PDF 원문에 기반하며, 논문 내 인용 자료는 원문 레퍼런스 섹션을 그대로 참조했습니다. 제공된 PDF에 포함되지 않은 외부 정보는 추측하지 않았습니다.
