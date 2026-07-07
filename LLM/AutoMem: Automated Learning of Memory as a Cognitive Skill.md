# AutoMem: Automated Learning of Memory as a Cognitive Skill 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

AUTOMEM의 근본적인 주장은 다음과 같습니다:

> **"메모리 관리는 고정된 아키텍처 모듈이 아니라, LLM 에이전트가 스스로 학습할 수 있는 독립적인 인지 기술(cognitive skill)이다."**

이는 인지과학의 **메타메모리(metamemory)** 개념 — 자신의 기억 과정을 모니터링하고 조절하는 능력 (Flavell, 1979; Nelson, 1990) — 을 LLM 에이전트 설계에 직접 적용한 것입니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **(i) 메모리의 재정의** | 파일 시스템 연산(read, write, search, append, create)을 태스크 액션과 동등한 **1급 메모리 액션**으로 승격 |
| **(ii) AUTOMEM 프레임워크** | 두 개의 자동화된 외부 루프를 통해 메모리 스캐폴드(구조)와 메모리 숙련도(능력)를 동시에 최적화 |
| **(iii) 실증적 성과** | 32B 오픈 가중치 모델로 $\sim 2\times$ – $4\times$ 성능 향상, Claude Opus 4.5 및 Gemini 3.1 Pro Thinking 수준에 도달 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: LLM의 컨텍스트 윈도우 한계**

LLM의 컨텍스트 윈도우는 인간의 작업 기억(working memory)과 유사한 **고정 크기 버퍼**입니다. 장기 수평(long-horizon) 태스크에서는 이 용량을 초과하는 정보가 발생합니다.

$$\text{Context Window Capacity} \ll \text{Information required for long-horizon tasks}$$

**문제 2: 기존 접근법의 한계**

기존 외부 메모리 연구들(RAG, MemGPT, Generative Agents 등)은 메모리를 **고정된 아키텍처 모듈**로 취급했습니다. 즉, 시스템에 설계되어 들어가는 것이지, 모델이 스스로 학습하는 것이 아니었습니다.

**문제 3: 장기 궤적 최적화의 어려움**

에피소드가 수만 스텝에 달하고, 메모리 실수의 결과가 수백 스텝 후에야 나타나기 때문에 인간이 전체 궤적을 검토하는 것은 **사실상 불가능**합니다.

---

### 2.2 제안하는 방법

#### 전체 프레임워크 개요

AUTOMEM은 두 축에서 메모리 기술을 향상시킵니다:

```math
\text{Memory Skill} = \underbrace{\text{Structure}}_{\text{Outer-loop \#1}} + \underbrace{\text{Proficiency}}_{\text{Outer-loop \#2}}
```

각 루프의 파라미터와 업데이트 신호:

$$\theta_1 = \text{agent scaffold}, \quad \nabla L_1 = \text{code revision by meta-LLM}$$

$$\theta_2 = \text{memory model weights (LoRA)}, \quad \nabla L_2 = \text{supervised training on curated traces}$$

---

#### 내부 루프 에이전트: 파일 시스템 기반 메모리

에이전트는 각 스텝에서 두 가지 루틴을 실행합니다:

**LOG 루틴** (기록 결정):

$$\text{LOG: "What is worth recording about what just happened?"}$$

$$\rightarrow \text{APPEND} \mid \text{WRITE} \mid \text{CREATE} \mid \text{UPSERT MAP}$$

**PLAN 루틴** (검색 및 행동 결정):

$$\text{PLAN: "What do I need to recall to act now?"}$$

$$\rightarrow \text{READ} \mid \text{SEARCH} \mid \text{GAMEPLAY ACTION}$$

통합 액션 공간은 다음과 같이 정의됩니다:

$$\mathcal{A} = \mathcal{A}_{\text{memory}} \cup \mathcal{A}_{\text{task}}$$

$$\mathcal{A}_{\text{memory}} = \{\texttt{READ}, \texttt{WRITE}, \texttt{APPEND}, \texttt{SEARCH}, \texttt{CREATE}, \texttt{UPSERT MAP}\}$$

$$\mathcal{A}_{\text{task}} = \{\text{GO NORTH, GO SOUTH, CRAFT, ..., environment-specific actions}\}$$

---

#### Outer-Loop #1: 메모리 스캐폴드 최적화

메타-LLM이 완전한 에피소드 궤적을 검토하고 에이전트 스캐폴드를 반복적으로 수정합니다.

**최적화 프로세스:**

$$\text{scaffold}^{(t+1)} = \text{meta-LLM-Revise}(\text{scaffold}^{(t)}, \tau^{(t)})$$

여기서 $\tau^{(t)}$는 스텝 $t$에서의 완전한 에피소드 궤적입니다.

**개선 게이팅 조건:**

$$\text{Accept revision} \iff \bar{\rho}(\text{scaffold}^{(t+1)}) > \bar{\rho}(\text{scaffold}^{(t)})$$

여기서 $\bar{\rho}$는 고정 평가 시드에 대한 평균 진행률(progression rate)입니다.

**NetHack 예시 — UPSERT_MAP 연산:**

기존 v0 방식 (중복 누적):
```
(x:44,y:19): gold
(x:44,y:19): wall to north
(x:44,y:19): gold  ← 중복!
```

개선된 v1 방식 (좌표 키 기반 중복 제거):
```
<|UPSERT_MAP|>44,19|floor, '>' south
← 기존 (44,19) 항목을 대체
```

이로 인해 NetHack의 per-step 메모리 증가량이 **138자 → 6자 (95% 감소)**했습니다.

---

#### Outer-Loop #2: 메모리 숙련도 학습

스캐폴드 최적화가 수렴한 후, 모델의 파라메트릭 메모리 능력을 직접 학습시킵니다.

**핵심 구조 — 분리된 두 모델:**

$$\pi_{\text{memory}} = \text{base model} + \text{LoRA adapter} \quad (\text{memory specialist, trainable})$$

$$\pi_{\text{task}} = \text{base model} \quad (\text{gameplay model, frozen})$$

**LoRA 파인튜닝 목적 함수:**

$$\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in \mathcal{D}_{\text{mem}}} \log p_{\theta_{\text{LoRA}}}(y \mid x)$$

여기서 $\mathcal{D}_{\text{mem}}$은 메타-LLM이 에이전트의 자체 경험에서 선별한 **고품질 메모리 결정 트레이스**입니다.

**데이터 수집 규모:**

$$|\mathcal{D}_{\text{mem}}|: \text{Crafter: 1597 examples, MiniHack: 444 examples, NetHack: 800 examples}$$

**훈련 데이터 수집 에피소드:**

$$N_{\text{train}}: \text{Crafter: 100 episodes, MiniHack: } 50 \times 8 = 400 \text{ episodes, NetHack: 50 episodes}$$

**LoRA 하이퍼파라미터 (예: Crafter):**

$$r = 256, \quad \alpha = 512, \quad \text{dropout} = 0.0, \quad \text{lr} = 5 \times 10^{-5}, \quad \text{epochs} = 4$$

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                    OUTER-LOOP #1                         │
│         Meta-LLM (Claude Opus 4.6) - Scaffold Optimizer  │
│    trajectory review → failure diagnosis → code revision │
└────────────────────────┬────────────────────────────────┘
                         │ deploys & iterates
┌────────────────────────▼────────────────────────────────┐
│              INNER-LOOP AGENT                            │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │  Memory Specialist│    │     Task Model           │   │
│  │ (LoRA-finetuned) │    │   (Frozen base model)    │   │
│  │                  │    │                          │   │
│  │  LOG routine     │    │  GAMEPLAY action         │   │
│  │  PLAN/SEARCH     │───▶│  commitment              │   │
│  └────────┬─────────┘    └──────────────────────────┘   │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────────────────────────────┐            │
│  │           File System Memory            │            │
│  │  game_rules.txt  │  dungeon_map.txt     │            │
│  │  inventory.txt   │  strategy.txt        │            │
│  │  monster_encounters.txt  │  ...         │            │
│  └─────────────────────────────────────────┘            │
└────────────────────────┬────────────────────────────────┘
                         │ feeds traces
┌────────────────────────▼────────────────────────────────┐
│                    OUTER-LOOP #2                         │
│      Meta-LLM (Claude Opus 4.7) - Training Engine        │
│  data curation → LoRA config → train memory specialist   │
└─────────────────────────────────────────────────────────┘
```

---

### 2.4 성능 향상

#### 주요 성능 결과 (Table 1)

| 에이전트 | Crafter (%) | MiniHack (%) | NetHack (%) |
|---------|-------------|--------------|-------------|
| **Frontier 모델** | | | |
| Gemini-3-Pro | $57.3 \pm 4.4$ | $40.0 \pm 7.7$ | $6.8 \pm 3.2$ |
| Gemini-3.1-Pro-Thinking | $55.0 \pm 6.4$ | $27.5 \pm 7.1$ | $2.6 \pm 0.3$ |
| Claude-Opus-4.5 | $49.5 \pm 3.1$ | $27.5 \pm 7.1$ | $2.0 \pm 0.5$ |
| **Open-weight 기준** | | | |
| Qwen2.5-72B-Instruct | $27.3 \pm 3.6$ | $5.0 \pm 3.4$ | $0.3 \pm 0.3$ |
| **AUTOMEM (Qwen2.5-32B 기반)** | | | |
| v0 (기본 파일 시스템) | $25.0 \pm 5.5$ | $7.5 \pm 4.2$ | $0.42 \pm 0.37$ |
| + scaffold opt. (Loop #1) | $47.27 \pm 2.05$ | $27.50 \pm 7.06$ | $1.57 \pm 0.35$ |
| + memory training (Loop #2) | $\mathbf{51.36 \pm 3.81}$ | $\mathbf{30.00 \pm 7.25}$ | $\mathbf{1.85 \pm 0.44}$ |

#### 향상 배수

$$\text{Crafter}: \frac{51.36}{25.0} \approx 2.05\times, \quad \text{MiniHack}: \frac{30.0}{7.5} = 4.0\times, \quad \text{NetHack}: \frac{1.85}{0.42} \approx 4.4\times$$

#### 행동 지표 개선 (Figure 4)

| 지표 | 개선 방향 | Crafter | MiniHack | NetHack |
|-----|----------|---------|----------|---------|
| 비생산적 행동률 (stuck/oscillating) | ↓ | $-37\%$ | $-65\%$ | $-32\%$ |
| 반복 WRITE 비율 | ↓ | $-83\%$ | $-68\%$ | $-68\%$ |
| 빈 SEARCH 비율 | ↓ | $-20\%$ | $-13\%$ | $-50\%$ |
| 스텝당 입력 토큰 (k) | ↓ | $-30\%$ | $-3\%$ | $-25\%$ |

#### 훈련 후 메모리 규율 지표 (Table 2)

$$\frac{\text{writes}}{\text{searches}}: \text{Crafter: } 0.84 \to 0.39\ (-54\%), \quad \text{MiniHack: } 2.89 \to 0.82\ (-72\%), \quad \text{NetHack: } 4.66 \to 1.31\ (-72\%)$$

---

### 2.5 한계점

논문이 명시한 한계:

1. **에피소딕 메모리만 다룸**: 파일 시스템이 에피소드마다 초기화되므로, 에피소드 간 지식 이전(cross-episode persistent memory)이 없음
2. **게임 환경에 국한**: 실제 세계의 메모리 집약적 태스크(문서 분석, 코드 작성, 다중 세션 대화 등)에 대한 검증 부재
3. **환경별 별도 스캐폴드**: Crafter/MiniHack/NetHack 각각에 대해 별도 스캐폴드와 메모리 전문가를 훈련하므로, **단일 범용 에이전트**로의 통합 여부 미검증
4. **메타-LLM 의존성**: 두 외부 루프 모두 Claude Opus 4.6/4.7이라는 강력한 독점 LLM에 의존함 — 메타-LLM의 품질이 전체 프레임워크 성능의 상한을 결정
5. **평가 규모 제한**: 10개 고정 시드, 환경당 적은 에피소드 수로 통계적 불확실성 존재 (특히 NetHack의 SE 범위가 넓음)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지지하는 근거

#### (a) 메모리 기술의 분리 가능성 (Separability)

AUTOMEM의 가장 중요한 통찰은 메모리 관리를 **분리 가능한 기술(separable skill)**로 취급한다는 점입니다.

$$\pi_{\text{agent}} = \pi_{\text{task}}(\text{frozen}) \otimes \pi_{\text{memory}}(\text{trainable})$$

이 분리는 두 가지 일반화 이점을 제공합니다:

- **태스크 능력 보존**: 게임플레이 모델의 가중치를 전혀 수정하지 않으므로, 기존 태스크 능력이 훼손되지 않음
- **메모리 기술의 전이 가능성**: 메모리 결정 패턴(언제 기록할지, 어떻게 검색할지)은 도메인 독립적일 수 있음

#### (b) 절차적 생성 환경에서의 검증

세 게임 환경 모두 **매 에피소드마다 세계가 재생성**되므로, 사전학습 지식의 단순 암기가 아닌 진정한 메모리 기술 학습을 측정합니다:

$$P(\text{세계 구성}_{\text{에피소드}_i} = \text{세계 구성}_{\text{에피소드}_j}) \approx 0 \quad \forall i \neq j$$

#### (c) 모델 규모를 초월하는 성능

$$\text{Qwen2.5-32B + AUTOMEM} \gg \text{Qwen2.5-72B-Instruct}$$

이는 메모리 기술이 모델 파라미터 수와 독립적인 일반화 능력을 제공함을 시사합니다. 즉:

$$\text{성능} \propto f(\text{메모리 구조}, \text{메모리 숙련도}) \quad \text{not only } f(\text{모델 규모})$$

#### (d) 훈련 데이터와 평가 데이터의 시드 분리

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{eval}} = \emptyset, \quad \mathcal{S}_{\text{eval}} = \{42, 43, ..., 51\}$$

이는 메모리 전문가가 특정 에피소드를 암기한 것이 아님을 보장합니다.

### 3.2 일반화의 구체적 메커니즘

#### 메모리 스캐폴드의 일반화

스캐폴드 최적화는 **코드, 프롬프트, 파일 스키마**를 수정하므로, 동일한 환경 내 임의의 새로운 에피소드에 적용 가능합니다. 예를 들어 NetHack의 `UPSERT_MAP` 연산은 어떤 던전 구성에서도 중복 제거를 보장합니다.

#### "쓰기 전 검색" 규율의 내재화

$$\text{consult-before-write ratio}: \frac{\text{SEARCHes}}{\text{WRITEs}} \uparrow \quad \text{(일반적 메모리 규율)}$$

이 행동 패턴은 특정 환경 구성에 의존하지 않으므로, 새로운 에피소드에서도 동일하게 적용됩니다.

### 3.3 일반화의 한계 및 불확실성

반면, 다음과 같은 일반화 제약도 존재합니다:

- **환경 특수성**: 각 환경에 대해 별도의 스캐폴드와 LoRA 어댑터를 훈련 — 단일 메모리 전문가의 **도메인 간 전이** 미검증
- **에피소드 간 비연속성**: 에피소드 시작 시 메모리 초기화로 인해 장기적 지식 축적이 불가능
- **메타-LLM 병목**: 일반화를 위해 강력한 메타-LLM이 필요하며, 이는 계산 비용과 접근성 문제를 야기

---

## 4. 미래 연구에 미치는 영향 및 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (a) 에이전트 능력의 모듈화 패러다임

AUTOMEM은 **에이전트 능력을 독립적으로 학습 가능한 기술들로 분해**할 수 있음을 보여줍니다:

$$\pi_{\text{agent}} = \pi_{\text{memory}} \oplus \pi_{\text{planning}} \oplus \pi_{\text{reasoning}} \oplus \pi_{\text{task}}$$

향후 연구에서는 각 능력(reasoning, planning, tool use 등)을 유사한 방식으로 분리하여 최적화하는 방향이 탐색될 것입니다.

#### (b) 메타-LLM 기반 자동화 루프의 확장

논문의 핵심 발견 중 하나는 **강력한 메타-LLM이 수만 스텝의 궤적을 분석하고 의미 있는 개선을 제안할 수 있다**는 것입니다. 이는 다음 연구 방향에 영향을 줄 것입니다:

- **자동화된 에이전트 시스템 설계 (ADAS)**: 메모리뿐 아니라 전체 에이전트 아키텍처의 자동 최적화
- **자동화된 데이터 큐레이션**: 메타-LLM을 교사가 아닌 필터로 사용하는 방식의 확산
- **자기 개선 시스템**: 에이전트가 자신의 과거 경험에서 훈련 데이터를 생성하는 패러다임

#### (c) LLM 에이전트 벤치마킹 재정의

기존 벤치마크(BALROG 등)에서 오픈 가중치 모델과 프론티어 모델의 성능 격차가 **메모리 관리만으로 상당 부분 해소**될 수 있음을 보여줬습니다. 이는 향후 벤치마킹에서:

- 컨텍스트 관리 전략을 명시적 변수로 통제해야 함
- "순수 모델 능력"과 "메모리 관리 능력"을 분리하여 측정하는 프로토콜이 필요함

#### (d) 인지과학과 AI의 통합

메타메모리 개념의 AI 적용 성공은 인지과학의 다른 개념들 — 예를 들어 실행 기능(executive function), 인지 부하 이론(cognitive load theory) — 을 AI 에이전트 설계에 적용하는 연구를 자극할 것입니다.

---

### 4.2 향후 연구 시 고려할 점

#### 고려사항 1: 에피소드 간 지속적 메모리 (Cross-Episode Persistent Memory)

현재 AUTOMEM의 파일 시스템은 에피소드마다 초기화됩니다. 향후 연구는:

$$\mathcal{M}_{\text{episode}_t} \xrightarrow{\text{knowledge transfer}} \mathcal{M}_{\text{episode}_{t+1}}$$

즉, 에피소드 간 지식 이전(meta-learning과 유사한 구조)을 가능하게 하는 영속적 메모리 설계가 필요합니다.

#### 고려사항 2: 단일 범용 메모리 전문가

현재 각 환경에 별도의 LoRA 어댑터를 훈련합니다. 미래 연구는:

$$\pi_{\text{memory}}^{\text{universal}} = \arg\min_\theta \sum_{e \in \mathcal{E}} \mathcal{L}_{\text{SFT}}(\theta; \mathcal{D}_{\text{mem}}^{(e)})$$

즉, 다양한 환경과 도메인을 아우르는 범용 메모리 전문가 훈련 방법론이 필요합니다.

#### 고려사항 3: 메타-LLM 의존성 감소

현재 두 외부 루프 모두 Claude Opus 수준의 강력한 독점 LLM에 의존합니다. 향후 연구는:

- 더 소형의 오픈 가중치 메타-LLM으로 동일한 효과를 달성하는 방법 탐구
- 메타-LLM의 역할을 학습된 비평 모델(learned critic model)로 대체하는 방안

#### 고려사항 4: 강화학습과의 통합

현재 Outer-loop #2는 지도 학습(SFT)을 사용합니다. 향후에는:

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\tau \sim \pi_{\text{memory}}} [R(\tau)] + \lambda \cdot \mathcal{L}_{\text{SFT}}$$

즉, 에피소드 결과를 직접 보상으로 사용하는 강화학습(예: Memory-R1, MemSearcher 방식)과의 결합이 유망합니다.

#### 고려사항 5: 실제 세계 태스크로의 확장

게임 환경의 특성(절차적 생성, 명확한 진행률 메트릭)은 연구에 유리하지만, 실제 적용을 위해서는:

- 의료 기록 관리, 코드 리뷰, 장기 연구 과제 등 **실제 메모리 집약적 태스크**로의 검증
- 안전성 검토 및 신뢰성 있는 메모리 관리 보장 메커니즘

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 메모리 방식 | 구조 최적화 | 능력 학습 | AUTOMEM과의 차이 |
|------|------|------------|------------|----------|-----------------|
| **RAG** (Lewis et al.) | 2020 | 벡터 검색 DB | ✗ | ✗ | 고정 아키텍처, 메모리 관리 학습 없음 |
| **MemGPT** (Packer et al.) | 2023 | OS 페이징 방식 | ✗ | ✗ | 고정 메커니즘, 규칙 기반 |
| **Generative Agents** (Park et al.) | 2023 | 타임스탬프 메모리 스트림 | ✗ | ✗ | 고정 검색/반영 구조 |
| **ReAct** (Yao et al.) | 2022 | 컨텍스트 내 추론 | ✗ | ✗ | 외부 메모리 없음 |
| **Reflexion** (Shinn et al.) | 2023 | 언어적 자기 반영 | △ (언어 수준) | ✗ | 파라메트릭 학습 없음 |
| **ExpeL** (Zhao et al.) | 2024 | 경험 인사이트 추출 | △ | ✗ | 파라미터 업데이트 없음 |
| **A-MEM** (Xu et al.) | 2025 | 능동적 보존/망각 결정 | ✗ | △ | 스캐폴드 자동 최적화 없음 |
| **MemAct** (Zhang et al.) | 2025 | 컨텍스트 윈도우 기반 액션 | ✗ | △ | 외부 파일 시스템 아닌 컨텍스트 기반 |
| **MemEvolve** (Zhang et al.) | 2025 | 모듈식 메모리 아키텍처 진화 | ✓ | ✗ | 능력 훈련(LoRA) 없음 |
| **MeMo** (Quek et al.) | 2026 | 전용 메모리 모델 | ✗ | ✓ | 정적 문서 QA에 특화, 장기 행동 에이전트 아님 |
| **Memory-R1** (Yan et al.) | 2025 | RL 기반 메모리 관리 | ✗ | ✓ | SFT 대신 RL, 스캐폴드 최적화 없음 |
| **MemSkill** (Zhang et al.) | 2026 | 실패 사례 검토 기반 기술 진화 | △ | ✗ | 파라미터 업데이트 없음 |
| **MetaMem** (Xin et al.) | 2026 | 자기반영적 지식 검색 최적화 | △ | △ | 게임/장기 행동 에이전트 적용 미검증 |
| **Mem1** (Zhou et al.) | 2025 | 메모리-추론 시너지 | ✗ | ✓ | RL 기반, 스캐폴드 자동화 없음 |
| **AUTOMEM** (본 논문) | 2026 | 파일 시스템 기반 자율 메모리 | **✓✓** | **✓✓** | **두 축 모두 자동화, 장기 궤적 분석 기반** |

### 핵심 차별점 요약

$$\text{AUTOMEM} = \underbrace{\text{Scaffold Auto-Optimization}}_{\text{기존 연구 대비 차별화}} + \underbrace{\text{Parametric Proficiency Training}}_{\text{기존 연구 대비 차별화}} + \underbrace{\text{Trajectory-level Meta-LLM Review}}_{\text{새로운 기여}}$$

AUTOMEM이 이전 연구들과 구별되는 가장 중요한 점은, **두 최적화 축을 모두 자동화**하고, **최대 $10^5$ 스텝에 달하는 완전한 궤적을 분석 신호로 사용**한다는 것입니다.

---

## 참고 자료

**주 논문:**
- Wu, S., Zhu, H., Zhang, Y., Wang, X., & Yeung-Levy, S. (2026). *AutoMem: Automated Learning of Memory as a Cognitive Skill*. arXiv:2607.01224v1.

**논문 내 인용 참고문헌 (주요):**
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10):906.
- Nelson, T. O. (1990). Metamemory: A theoretical framework and new findings. *Psychology of Learning and Motivation*, 26:125–173.
- Clark, A. & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1):7–19.
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 33*:9459–9474.
- Packer, C. et al. (2023). MemGPT: Towards LLMs as operating systems. arXiv:2310.08560.
- Park, J. S. et al. (2023). Generative agents. *UIST 2023*.
- Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Yao, S. et al. (2022). ReAct: Synergizing reasoning and acting in language models. arXiv:2210.03629.
- Shinn, N. et al. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS 36*.
- Paglieri, D. et al. (2024). BALROG: Benchmarking agentic LLM and VLM reasoning on games. arXiv:2411.13543.
- Hafner, D. (2021). Benchmarking the spectrum of agent capabilities. arXiv:2109.06780.
- Samvelyan, M. et al. (2021). MiniHack the planet. arXiv:2109.13202.
- Küttler, H. et al. (2020). The NetHack learning environment. *NeurIPS 33*:7671–7684.
- Zhao, A. et al. (2024). ExpeL: LLM agents are experiential learners. *AAAI 38*.
- Xu, W. et al. (2025). A-MEM: Agentic memory for LLM agents. arXiv:2502.12110.
- Yan, S. et al. (2025). Memory-R1. arXiv:2508.19828.
- Zhang, G. et al. (2025a). MemEvolve. arXiv:2512.18746.
- Quek, R. W. H. et al. (2026). MeMo: Memory as a model. arXiv:2605.15156.
- Zhang, H. et al. (2026a). MemSkill. arXiv:2602.02474.
- Zhou, Z. et al. (2025). Mem1. arXiv:2506.15841.
- Liu, J. et al. (2026). EvolveMem. arXiv:2605.13941.
- Sumers, T. et al. (2023). Cognitive architectures for Language Agents. *TMLR*.
- Wang, G. et al. (2023a). Voyager. arXiv:2305.16291.
- Hu, S., Lu, C., & Clune, J. (2024). ADAS. arXiv:2408.08435.

**프로젝트 페이지:** https://autolearnmem.github.io/
**코드베이스:** https://github.com/autoLearnMem/AutoMem
