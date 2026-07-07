# Generative Skill Composition for LLM Agents

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **LLM 에이전트의 스킬 라이브러리 활용 문제를 단순한 검색(retrieval) 문제가 아닌, 구조화된 시퀀스 예측(structured sequence prediction) 문제로 재정의**해야 한다는 것입니다.

기존 방법들은 두 가지 패러다임으로 분류됩니다:
1. **직접 추론(Direct reasoning)**: 에이전트가 전체 스킬 라이브러리에 노출되어 추론 → 명시적 계획 없음
2. **임베딩/LLM 기반 검색(Retrieval)**: 순위 기반 스킬 선택 → 순서 없는 부분집합만 반환

이 두 접근 모두 스킬 구성의 **구조적 특성(structural nature)**, 즉 **(1) 어떤 스킬을, (2) 몇 개를, (3) 어떤 순서로** 사용할지에 대한 결합적 결정(joint decision)을 놓치고 있다고 주장합니다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **문제 정식화** | 스킬 사용을 task-conditioned skill sequence prediction으로 정식화 |
| **데이터셋 구축** | 실제 인간 큐레이션 스킬 라이브러리 기반 task-skill 구성 쌍 데이터셋 구축 |
| **SkillComposer 제안** | 제약된 자기회귀 디코더 기반 생성적 스킬 구성 프레임워크 |
| **실험적 검증** | SkillsBench에서 No-skill 대비 {+23.1, +18.2} pp pass rate 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

스킬 라이브러리가 대규모화됨에 따라, **inference-time bottleneck**이 "스킬 획득"에서 "올바른 스킬 구성"으로 이동했습니다. 예를 들어 "deprecated API를 찾아 코드베이스 전체에서 리팩토링하고 회귀 테스트를 실행하라"는 태스크에서, 단순 검색은 개별적으로 관련 스킬은 찾을 수 있지만 실행 순서(검색 → 수정 → 테스트)를 명시하지 못합니다.

**스킬의 정식 정의:**

$$s_i = (m_i, C_i, \pi_i, T_i, R_i)$$

- $m_i$: 메타데이터 (이름 + 한 줄 설명)
- $C_i$: 적용 가능 조건
- $\pi_i$: 절차적 정책
- $T_i$: 종료 조건
- $R_i$: 선택적 호출 가능 인터페이스

**스킬 라이브러리:**

$$\mathcal{S} = (s_1, \ldots, s_K), \quad K = 196$$

### 2.2 제안 방법

#### 핵심 정식화 (Task-Conditioned Skill Sequence Prediction)

$$\hat{\mathbf{z}} = (\hat{z}_1, \hat{z}_2, \ldots, \hat{z}_{\hat{n}}, \text{STOP}) = f_\theta(x, c, \mathcal{S}) $$

- $x \in \mathcal{X}$: 자연어 태스크 설명
- $c \in \mathcal{C}$: 환경 컨텍스트
- $\hat{z}_t \in \{1, \ldots, K\}$: 각 스킬 인덱스
- STOP: 종료 심볼

예측된 스킬 시퀀스는 인덱스 조회로 복원:

$$\hat{\mathbf{s}} = (s_{\hat{z}_1}, s_{\hat{z}_2}, \ldots, s_{\hat{z}_{\hat{n}}})$$

#### 자기회귀 디코더 (Autoregressive Decoder)

```math
p_\theta(\mathbf{z} \mid x, c, \mathcal{S}) = \prod_{t=1}^{n+1} p_\theta(z_t \mid \mathbf{h}, \mathbf{z}_{ < t})
```

- $\mathbf{h} = W_{\text{proj}}\mathbf{h}_x \in \mathbb{R}^d$: 태스크 벡터 (Qwen3-Embedding-0.6B로 생성, $d=256$)
- $D_\theta$: 3-layer, 256-dim transformer, 4 attention heads
- 출력 어휘: $\mathcal{V} = \{1, \ldots, K\} \cup \{\text{STOP}\}$

#### Cardinality Head (How Many)

$$p_\psi(\hat{n} \mid x, c) = \text{softmax}(W_n \mathbf{h}) $$

- 선형 분류기로 스킬 개수 $\hat{n} \in \{1, \ldots, N_{\max}\}$ 직접 예측 ($N_{\max} = 8$)
- STOP 위치에 의존하지 않는 독립적인 길이 신호

#### Set Head (Which Skills)

$$\sigma_i = g_\xi(\mathbf{h}, \mathbf{e}_i) = \text{MLP}_\xi([\mathbf{h}; \mathbf{e}_i; \mathbf{h} \odot \mathbf{e}_i; |\mathbf{h} - \mathbf{e}_i|]) $$

- $\mathbf{e}\_i = W_m E_\phi(m_i) \in \mathbb{R}^d$: 스킬 $s_i$의 메타데이터 임베딩
- 4가지 항: identity, interaction, element-wise product, L1 distance
- 2-layer MLP (hidden width 256), binary cross-entropy 지도학습
- **gold membership** $\mathbb{1}[s_i \in \hat{\mathbf{s}}]$에 대한 직접적 그래디언트 제공 (순서 독립)

#### Retrieval-Augmented Decoding - Logit Fusion

$$\underbrace{\tilde{\ell}_t(i)}_{\text{fused logit}} = \underbrace{\ell_t(i)}_{\text{contextual}} + \alpha \cdot \underbrace{\bar{r}_i}_{\text{relevance}} + \beta \cdot \underbrace{\sigma_i}_{\text{set}}, \quad i \in \{1, \ldots, K\} $$

- $\ell_t(i)$: AR 디코더의 컨텍스트 기반 logit
- $\bar{r}_i$: TF-IDF cosine similarity 기반 검색 점수
- $\sigma_i$: set head의 task-aware membership prior
- 하이퍼파라미터: $\alpha = 1.0$, $\beta = 0.5$ (validation Set F1으로 튜닝)
- **STOP 토큰에는 검색/membership prior 미적용** → 종료는 AR head가 제어

### 2.3 모델 구조

```
입력: Task x, Environment c, Skill library S = {s₁,...,s₁₉₆}
         ↓
[Task Encoder] Eφ (Frozen Qwen3-Embedding-0.6B)
   → 태스크 벡터 h ∈ R²⁵⁶
         ↓
[Autoregressive Decoder] Dθ (3-layer Transformer, 256-dim, 4 heads)
   - cross-attention to skill metadata embeddings
   - AR head: ordering
   - Cardinality head: how many skills
   - Set head: which skills
         ↓
[Retrieval-Augmented Decoding]
   - TF-IDF retrieval prior fusion (α=1.0)
   - Set membership prior fusion (β=0.5)
   - Width-4 beam search, duplicate-skill constraint
         ↓
출력: Ordered skill index sequence → Agent Context
```

**훈련 가능 파라미터**: ~3.9M (SFT Qwen3-0.6B-Base의 600M 대비 ~154× 적음)

### 2.4 데이터셋 구성

총 **9,872개** task-skill-sequence 훈련 레코드:

| 데이터 종류 | 수량 | 특징 |
|-------------|------|------|
| Real anchors | 65 | SkillBench의 인간 저작 SW 엔지니어링 태스크 |
| Single-skill synthetic | 2,880 | Gemini 2.5 Flash로 생성, 196개 스킬 균일 커버 |
| Multi-skill synthetic | 6,927 | Gemini 2.5 Pro로 생성, 2~5개 스킬 구성 |

스킬 의존성 그래프 (196 nodes, 924 edges):
- **Dependency edges** (658개): I/O 타입 중첩 기반 데이터 흐름 순서
- **Workflow edges** (266개): 실제 에이전트 궤적의 스킬 공동 출현

### 2.5 성능 향상

#### 스킬 구성 품질 (Table 1)

| 방법 | Synthetic Set F1 | Real-task Set F1 | 파라미터 |
|------|-----------------|------------------|---------|
| BM25 (best-k) | 33.0 | 47.0 | - |
| TF-IDF (best-k) | 52.5 | 60.6 | - |
| Qwen3-Emb (best-k) | 43.9 | 58.5 | - |
| Gemini-2.5-flash (LLM-judge) | 61.0 | 59.9 | ~수십억 |
| SFT Qwen3-0.6B-Base | 71.1 | 43.6 | 600M |
| **SkillComposer** | **73.9** | **62.9** | **3.9M** |

#### 다운스트림 태스크 성능 (Table 2, SkillsBench)

| 스킬 조건 | Codex Pass(%) | Gemini Pass(%) | Codex Tok. |
|-----------|---------------|----------------|------------|
| No Skills | 22.2 | 25.8 | 0.94M |
| All Skills | 29.3 | 38.7 | 1.27M |
| Retrieval (top-3) | 44.0 | 41.8 | 1.09M |
| Retrieval (oracle) | 44.0 | 42.2 | 1.13M |
| **SkillComposer** | **45.3** | **44.0** | **1.03M** |
| Gold Skills (상한) | 51.1 | 48.4 | 1.12M |

→ No-skill 대비 **+23.1 pp (Codex), +18.2 pp (Gemini)** 향상, gold-skill retrieval 상한에 근접하면서 **프롬프트 토큰 비용 최소화**

#### Ablation Study (Table 3)

| Variant | Set F1 |
|---------|--------|
| AR-only | 69.3 |
| + set head | 71.8 |
| + cardinality head | 69.6 |
| **SkillComposer (full)** | **73.9** |
| − decode set-fusion ($\beta=0$) | 65.0 (−8.9) |
| − decode retrieval prior ($\alpha=0$) | 67.5 (−6.4) |

### 2.6 한계점

논문 Appendix A에서 명시된 한계:

1. **텍스트 전용 태스크**: 멀티모달 태스크 명세(스크린샷, 음성 등) 미지원
2. **코드 지향 스킬 라이브러리**: 과학적 워크플로우, 로봇공학, 구체화된 에이전트 미검증
3. **단기 태스크 편향**: 합성 코퍼스가 ≤3개 스킬 구성에 집중 → 장기 체인 태스크에서 under-emit 경향 (Case 3)
4. **고정 스킬 라이브러리**: 온라인 스킬 업데이트 시나리오 미지원
5. **소규모 인코더/디코더**: 더 강력한 백본 사용 시 추가 성능 향상 가능성

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 핵심 발견: SFT vs SkillComposer의 분포 이동 대응

논문에서 가장 중요한 일반화 관련 발견은 **분포 이동(distribution shift) 하에서의 성능 차이**입니다:

$$\Delta_{\text{SFT}} = 71.1 - 43.6 = -27.5 \text{ pp Set F1 (합성 → 실제 태스크)}$$
$$\Delta_{\text{SkillComposer}} = 73.9 - 62.9 = -11.0 \text{ pp Set F1 (합성 → 실제 태스크)}$$
$$\text{Gap} = 27.5 - 11.0 = +19.3 \text{ pp 우위}$$

이는 **동일한 훈련 데이터와 동일한 스킬 라이브러리에서** SkillComposer가 SFT보다 훨씬 나은 OOD(Out-of-Distribution) 일반화를 보임을 의미합니다.

### 3.2 일반화 성능 향상의 메커니즘

#### (1) Frozen Retrieval-Tuned Encoder의 전이 편향 (Transfer Bias)

```
Frozen Qwen3-Embedding-0.6B
→ 사전 훈련된 의미 표현 (대규모 텍스트 코퍼스)
→ 실제 태스크 언어와의 의미적 근접성 포착
→ 합성 템플릿 분포를 암기하지 않음
```

SFT는 합성 템플릿 분포를 암기(memorize)하여 실제 태스크의 자연스러운 언어 표현에서 실패하는 반면, SkillComposer의 동결된 인코더는 사전 훈련된 의미 표현을 유지합니다.

**논문의 직접적 언급:**
> "The frozen retrieval-tuned encoder paired with a small specialist decoder is what supplies SkillComposer with this transfer bias."
> "Retrieval baselines actually improve from synthetic to real tasks because real-task phrasing is closer to the skill descriptions, whereas SFT has memorised the synthetic template distribution and has no robust prior to fall back on."

#### (2) 구조적 분리 (Structural Decoupling)

순서(ordering)와 집합 선택(set selection), 크기(cardinality)를 분리된 supervision channel로 학습:
- **AR head** → 순서 (위치-의존)
- **Set head** → 관련성 (위치-독립, 모든 gold 스킬에 직접 gradient)
- **Cardinality head** → 개수 (STOP 위치와 독립)

이 factorized supervision은 희소한 훈련 신호를 보완하여 일반화를 향상시킵니다. 특히 set head는 순서에 독립적으로 모든 gold 스킬에 직접 gradient를 제공하므로, 새로운 태스크 표현에서도 개별 스킬의 관련성을 판단할 수 있습니다.

#### (3) TF-IDF 기반 Retrieval Prior의 역할

```
Position-independent channel:
r(x, s_i) = TF-IDF cosine similarity
→ 어휘적(lexical) 중첩으로 스킬 식별
→ 학습 없이도 임의의 태스크에 적용 가능
→ heavy-tail 스킬 (훈련 데이터 1~2개)에 특히 효과적
```

**스파스 > 덴스** (Table 4):
- TF-IDF: 73.9 Set F1
- BM25: 70.0
- Qwen3-Embedding: 68.8
- No prior: 67.5

폐쇄된 196개 스킬 라이브러리에서 토큰 수준 어휘 중첩이 고정밀 구별 신호를 제공합니다.

#### (4) 카디널리티 견고성 (Cardinality Robustness, Figure 4)

| 금 카디널리티 k | Gemini Set F1 | SFT Set F1 | SkillComposer Set F1 |
|----------------|---------------|------------|----------------------|
| k=1 | 50 | 65 | **80** |
| k=2 | 65 | 80 | **80** |
| k=3 | 65 | 75 | **76** |
| k≥4 | 66 | 72 | **71** |
| mean | 62 | 71 | **74** |

k=1 버킷에서 SkillComposer의 우위가 가장 큰데, 이는 과잉 방출(over-emission)이 가장 패널티를 받는 상황에서 cardinality head가 효과적으로 작동함을 보여줍니다.

#### (5) Compute-Accuracy 파레토 최적성

SkillComposer는 **~154× 적은 훈련 가능 파라미터**와 **~25× 적은 훈련 연산량**으로 SFT를 능가합니다. 이는 소규모 특화 모델이 대규모 일반 LM보다 폐쇄된 스킬 라이브러리에서 더 신뢰할 수 있는 구성자임을 시사합니다:

> "The result suggests that, for closed agentic skill libraries, a small specialist that exploits the structure of the library is a more reliable composer than scaling up a generalist LM."

#### (6) 사례 분석에서 드러난 일반화 메커니즘

**Case 1 (adaptive-cruise-control)**: SkillComposer가 gold skill set과 다르게 `imc-tuning-rules`를 포함하고 I/O 래퍼 스킬을 제외 → **실제 유용한 스킬을 의미론적으로 파악** (reward 1.00 vs gold 0.33)

**Case 2 (exoplanet-detection-period)**: Gold Skills보다 3개 더 적은 최소 스킬 집합으로 완벽 해결 → **불필요한 스킬 제외 능력** (reward 1.00 vs gold 0.00)

**Case 3 (lean4-proof)**: 장기 체인에서 under-emit → **합성 데이터의 ≤3 스킬 편향 반영, 훈련 데이터 개선 필요**

### 3.3 일반화 향상을 위한 향후 방향

1. **장기 시퀀스 구성**: 4개 이상 스킬이 필요한 태스크를 위한 합성 데이터 확충
2. **멀티모달 입력**: 스크린샷, 음성, 스케치 등 다양한 태스크 명세 지원
3. **온라인 스킬 업데이트**: 라이브러리가 동적으로 성장하는 시나리오
4. **도메인 확장**: 과학적 워크플로우, 로봇공학, 구체화된 에이전트
5. **강력한 백본**: 더 큰 인코더/디코더 사용 시 추가 성능 향상 가능

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 연구 계보

```
VOYAGER (2023) → CRAFTEX/SkillFlow (2023-2025) → SkillBench (2026) → SkillComposer (2026)
Toolformer (2023) → HuggingGPT (2023) → TaskBench (2024) → SkillComposer (2026)
ReAct (2023) ──────────────────────────────────────────────────────→ SkillComposer (2026)
```

### 4.2 주요 관련 연구 비교

| 연구 | 연도 | 스킬/도구 선택 방식 | 순서 결정 | 개수 결정 | 주요 한계 |
|------|------|---------------------|-----------|-----------|-----------|
| **Toolformer** (Schick et al.) | 2023 | LM 자가 학습 | 암묵적 | 암묵적 | 단일 도구 콜 수준 |
| **VOYAGER** (Wang et al.) | 2023 | 임베딩 기반 검색 | 없음 | 없음 | 순서·개수 미결정 |
| **HuggingGPT** (Shen et al.) | 2023 | LLM 계획 | 명시적 | 명시적 | 전체 라이브러리 노출 필요 |
| **CREATOR** (Qian et al.) | 2023 | 도구 창조 | 없음 | 없음 | 새 도구 생성에 집중 |
| **ToolkenGPT** (Hao et al.) | 2023 | 임베딩 기반 선택 | 없음 | 없음 | 원자적 API 수준 |
| **ToolChain*** (Zhuang et al.) | 2023 | A* 탐색 | 명시적 | 명시적 | 타입 시그니처 필요 |
| **TaskBench** (Shen et al.) | 2024 | 그래프 기반 계획 | 명시적 | 명시적 | API 수준, 타입 추론 |
| **Graph Learning for Planning** (Wu et al.) | 2024 | 그래프 구조 계획 | 명시적 | 암묵적 | 도메인 특화 |
| **SkillFlow** (Li et al.) | 2025 | 검색+재랭킹 | 없음 | 없음 | 구성 순서 미고려 |
| **SkillRL** (Xia et al.) | 2026 | RL 기반 스킬 구성 | 암묵적 | 암묵적 | 개방형 스킬 생성 |
| **SkillRouter** (Zheng et al.) | 2026 | LLM-as-a-judge | 없음 | 없음 | 대규모 라이브러리 비효율 |
| **Graph of Skills** (Liu et al.) | 2026 | 의존성 그래프 검색 | 부분적 | 없음 | 순서·개수 jointly 미결정 |
| **Skill Retrieval Augmentation** (Su et al.) | 2026 | RAG 기반 검색 | 없음 | 없음 | 구성 구조 미고려 |
| **SkillComposer** (Zhao et al.) | 2026 | 생성적 시퀀스 예측 | **명시적·jointly** | **명시적·jointly** | 고정 라이브러리, 텍스트 전용 |

### 4.3 차별화 포인트

**검색 기반 방법 대비:**
- 기존: 독립적 스킬 점수 → 순서 없는 집합
- SkillComposer: prefix-conditioned 생성 → 순서 있는 실행 계획

**SFT 기반 방법 대비:**
- 기존: 전체 LLM 파인튜닝 → 합성 분포 암기, 분포 이동에 취약
- SkillComposer: Frozen encoder + 소규모 특화 디코더 → 강력한 전이 편향

**도구 계획 방법 대비 (TaskBench, ToolChain*):**
- 기존: API 수준, 타입 시그니처가 강한 구조적 신호 제공
- SkillComposer: 스킬 수준, 잠재적·태스크-논리적 의존성 처리

**핵심 혁신**: 스킬 구성을 **폐쇄 어휘 시퀀스 생성(closed-vocabulary sequence generation)**으로 명시적 카디널리티 및 순서 결정과 함께 모델링한 최초의 연구

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 스킬/도구 선택의 패러다임 전환

SkillComposer는 스킬 선택 문제를 **구조화된 예측 문제(structured prediction)**로 재정의함으로써, 향후 LLM 에이전트 연구에서 "어떤 도구를 몇 개, 어떤 순서로 사용할지"를 명시적으로 모델링하는 방향성을 제시합니다. 이는 단순 검색이나 전체 라이브러리 노출 패러다임에서 벗어나는 중요한 전환점입니다.

#### (2) 소규모 특화 모델의 효용성 입증

~3.9M 파라미터로 600M 파라미터 SFT를 능가하고 frontier API (Gemini-2.5-flash)를 압도한다는 결과는, **구조적 귀납 편향(structural inductive bias)이 파라미터 스케일보다 중요할 수 있음**을 시사합니다. 이는 에이전트 시스템의 모듈화 설계에 대한 새로운 시각을 제공합니다.

#### (3) Frozen Encoder + 소규모 Specialist Decoder 아키텍처

이 아키텍처 패턴은 다른 구조화된 예측 문제(도구 선택, 워크플로우 계획, API 구성)에도 적용 가능한 일반적 프레임워크를 제공합니다. 특히 사전 훈련된 임베딩 모델의 전이 편향을 활용하면서 소규모 특화 디코더를 훈련하는 접근법은 효율적인 에이전트 컴포넌트 개발의 모범 사례가 될 수 있습니다.

#### (4) 스킬 의존성 그래프 기반 데이터 합성

skill dependency graph를 사용한 데이터 합성 방법론은 다른 도메인의 에이전트 훈련 데이터 구축에 활용 가능합니다. 특히 의존성 엣지(I/O 중첩)와 워크플로우 엣지(공동 출현)의 조합은 현실적인 다단계 태스크 데이터 생성에 유용한 템플릿을 제공합니다.

#### (5) SkillsBench 벤치마크 영향

SkillsBench를 활용한 평가 방법론—구성 품질(composition quality)과 다운스트림 태스크 성공(downstream task success)을 분리하여 측정—은 향후 에이전트 스킬 연구의 표준 평가 프로토콜로 발전할 가능성이 있습니다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 장기 스킬 체인 데이터 구축

**현재 문제**: 합성 데이터가 ≤3개 스킬 구성에 치우쳐 4개 이상의 장기 체인에서 under-emit 발생

**고려 사항**:
- 4~8개 이상의 스킬을 포함하는 복잡한 다단계 태스크 합성 데이터 확충
- 장기 의존성 체인을 보존하는 합성 프롬프트 설계
- 장기 시퀀스에서의 cardinality head 보정 방법 연구

#### (2) 동적 스킬 라이브러리 지원

**현재 문제**: 스킬 라이브러리가 훈련 시 고정되어 새로운 스킬 추가 시 재훈련 필요

**고려 사항**:
- Zero-shot 또는 few-shot 스킬 통합 메커니즘 연구
- 스킬 메타데이터만으로 새 스킬을 처리할 수 있는 메타 학습 접근법
- 점진적 학습(continual learning)을 통한 라이브러리 확장 지원

#### (3) 멀티모달 태스크 명세 처리

**현재 문제**: 텍스트 전용 태스크 설명만 처리 가능

**고려 사항**:
- 비전-언어 모델(VLM)을 인코더로 활용하는 멀티모달 SkillComposer
- 스크린샷, 음성, 스케치 등 다양한 태스크 명세 형식 지원
- 멀티모달 태스크에서의 스킬 구성 벤치마크 필요

#### (4) 스킬 순서의 피드백 루프 (Interactive Composition)

**현재 문제**: 단일 패스 구성, 실행 중 피드백 미반영

**고려 사항**:
- 에이전트 실행 중간 상태를 반영한 적응적 스킬 구성 재계획
- 실패한 스킬 실행 후 대안 스킬 선택 메커니즘
- RLHF 또는 온라인 RL을 통한 실행 피드백 통합

#### (5) 평가 메트릭의 확장

**현재 문제**: Set F1이 순서 고려를 충분히 반영하지 못할 수 있음

**고려 사항**:
- 실행 순서의 중요도를 가중하는 순서-민감 메트릭 개발
- 부분 크레딧(partial credit)을 허용하는 소프트 매칭 메트릭
- 실제 에이전트 성능과의 상관관계 분석 심화

#### (6) 적대적 및 보안 취약성 분석

**현재 문제**: 논문(Liu et al., 2026b)에서 에이전트 스킬의 보안 취약성이 지적됨

**고려 사항**:
- 악의적으로 설계된 스킬이 라이브러리에 포함될 경우의 견고성
- 스킬 구성 예측에 대한 적대적 태스크 설명의 영향
- 스킬 활성화 감사(auditing) 메커니즘 연구

#### (7) 도메인 특화 스킬 라이브러리

**현재 문제**: 코드/SW 엔지니어링 도메인에 한정

**고려 사항**:
- 의료, 법률, 금융 등 전문 도메인 스킬 라이브러리 적용 연구
- 로봇공학, 구체화된 에이전트(embodied agents)에서의 물리적 행동 스킬 구성
- 이종(heterogeneous) 도구와 물리적 액츄에이터를 포함하는 복합 그래프

#### (8) 스킬 라이브러리 규모 확장성

**현재 문제**: K=196개의 소규모 스킬 라이브러리에서만 검증

**고려 사항**:
- K=1,000, 10,000 등 대규모 라이브러리에서의 확장성 연구
- 대규모 라이브러리에서 계층적 스킬 조직화 필요성
- 희소(sparse) 스킬 대상 검색 prior의 중요성이 더욱 증가할 가능성

---

## 참고 자료

### 주 논문
- **Zhao, X., Tan, Z., Tadiparthi, V., Agarwal, N., Lee, K., Moradi Pari, E., Nourkhiz Mahjoub, H., & Chen, T. (2026). Generative Skill Composition for LLM Agents. arXiv:2606.32025v1 [cs.CL].** *(본 분석의 주 대상 논문)*

### 논문 내 인용 참고문헌 (관련 최신 연구)
- Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291
- Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS 36.
- Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. ICLR 2023.
- Shen, Y., et al. (2023). HuggingGPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. NeurIPS 36.
- Hao, S., et al. (2023). ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings. NeurIPS 36.
- Zhuang, Y., et al. (2023). ToolChain*: Efficient action space navigation in large language models with A* search. arXiv:2310.13227
- Rajput, S., et al. (2023). Recommender systems with generative retrieval. NeurIPS 36.
- Qian, C., et al. (2023). CREATOR: Tool creation for disentangling abstract and concrete reasoning of LLMs. EMNLP 2023.
- Yuan, L., et al. (2023). CRAFT: Customizing LLMs by creating and retrieving from specialized toolsets. arXiv:2309.17428
- Shen, Y., et al. (2024). TaskBench: Benchmarking large language models for task automation. NeurIPS 37.
- Wu, X., et al. (2024). Can graph learning improve planning in LLM-based agents? NeurIPS 37.
- Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? ICLR 2024.
- Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. NeurIPS 2024.
- Xie, T., et al. (2024). OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. NeurIPS 2024.
- Li, F., Tagkopoulos, P., & Tagkopoulos, I. (2025). SkillFlow: Scalable and efficient agent skill retrieval system. arXiv:2504.06188
- Ma, Z., et al. (2025). Automated creation of reusable and diverse toolsets for enhancing LLM reasoning. AAAI 2025.
- Yue, M., et al. (2025). ToolLibGen: Scalable automatic tool creation and aggregation for LLM reasoning. arXiv:2510.07768
- Anthropic. (2025). Equipping agents for the real world with agent skills.
- Li, X., et al. (2026). SkillsBench: Benchmarking how well agent skills work across diverse tasks. arXiv:2602.12670
- Jiang, Y., et al. (2026). SoK: Agentic Skills – Beyond tool use in LLM agents. arXiv:2602.20867
- Xu, R. & Yan, Y. (2026). Agent skills for large language models: Architecture, acquisition, security, and the path forward. arXiv:2602.12430
- Liu, D., et al. (2026a). Graph of skills: Dependency-aware structural retrieval for massive agent skills. arXiv:2604.05333
- Liu, Y., et al. (2026b). Agent skills in the wild: An empirical study of security vulnerabilities at scale. arXiv:2601.10338
- Zheng, Y., et al. (2026). SkillRouter: Skill routing for LLM agents at scale. arXiv:2603.22455
- Su, W., et al. (2026). Skill retrieval augmentation for agentic AI. arXiv:2604.24594
- Xia, P., et al. (2026). SkillRL: Evolving agents via recursive skill-augmented reinforcement learning. arXiv:2602.08234
- Wang, C., et al. (2026). SkillX: Automatically constructing skill knowledge bases for agents. arXiv:2604.04804
- Jiao, Z., et al. (2026). Agentic proposing: Enhancing LLM reasoning via compositional skill synthesis. arXiv:2602.03279
- Harbor Framework Team. (2026). Harbor: A framework for evaluating and optimizing agents and models in container environments. GitHub.
