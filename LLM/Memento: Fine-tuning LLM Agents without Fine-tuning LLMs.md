# Memento: Fine-tuning LLM Agents without Fine-tuning LLMs

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Memento는 **LLM 자체의 파라미터를 업데이트(fine-tuning)하지 않고도**, 외부 에피소딕 메모리(Case Bank)와 온라인 강화학습(Online RL)을 통해 LLM 에이전트가 **지속적으로 적응·학습**할 수 있음을 보입니다.

> **"How can we build LLM agents that learn continuously from a changing environment without the prohibitive cost of fine-tuning the underlying LLMs?"**

### 주요 기여

| 기여 | 설명 |
|------|------|
| **새로운 학습 패러다임** | LLM 파라미터 고정 + 메모리 기반 온라인 강화학습 |
| **M-MDP 형식화** | Memory-augmented Markov Decision Process 수학적 정의 |
| **이중 메모리 설계** | Non-Parametric CBR + Parametric CBR (Q-function 기반) |
| **GAIA Top-1 달성** | Validation 87.88% Pass@3, Test 79.40% |
| **OOD 일반화 증명** | 비학습 데이터셋에서 4.7%~9.6% 절대 성능 향상 |
| **오픈소스 공개** | https://github.com/Agent-on-the-Fly/Memento |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 LLM 에이전트의 두 가지 한계:

1. **정적 워크플로우 기반 접근**: 하드코딩된 반사(reflection) 워크플로우에 의존 → 유연성 부족, 새로운 상황 적응 불가
2. **파라미터 업데이트 기반 접근**: Supervised Fine-tuning(SFT) 또는 강화학습(RL)으로 LLM 자체를 업데이트 → 막대한 계산 비용, catastrophic forgetting 위험

**핵심 연구 문제**: 그래디언트 업데이트 없이 LLM 에이전트가 동적 환경에서 지속적으로 학습할 수 있는가?

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Memory-augmented Markov Decision Process (M-MDP)

**Definition 3.1**: M-MDP는 튜플 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{M} \rangle$로 정의되며, 기존 MDP에 메모리 공간 $\mathcal{M} = (\mathcal{S} \times \mathcal{A} \times \mathbb{R})^*$가 추가됩니다.

**Case-Based Reasoning Agent의 전체 정책** (Eq. 1):

```math
\pi(a \mid s, M) = \sum_{c \in M} \mu(c \mid s, M) \, p_{\text{LLM}}(a \mid s, c)
```

여기서:
- $\mu(c \mid s, M)$: 케이스 검색 정책 (case retrieval policy)
- $p_{\text{LLM}}(a \mid s, c)$: 검색된 케이스 $c$를 조건으로 하는 LLM의 행동 확률

**CBR 에이전트 궤적의 확률** (Eq. 2):

$$p(\tau) = \prod_{t=0}^{T-1} \underbrace{\mu(c_t \mid s_t, M_t)}_{\text{(1) Retrieve}} \underbrace{p_{\text{LLM}}(a_t \mid s_t, c_t)}_{\text{(2) Reuse, Revise}} \underbrace{\mathbb{I}[r_t = \mathcal{R}(s_t, a_t)]}_{\text{(3) Evaluation}} \underbrace{\mathbb{I}[M_{t+1} = M_t \cup (s_t, a_t, r_t)]}_{\text{(4) Retain}} \underbrace{\mathcal{P}(s_{t+1} \mid s_t, a_t)}_{\text{(5) Transition}}$$

#### 2.2.2 Soft Q-Learning for CBR Agent

최대 엔트로피 강화학습 프레임워크(Haarnoja et al., 2018)를 적용한 최적화 목표 (Eq. 3):

$$J(\pi) = \mathbb{E}_{\tau \sim p} \left[ \sum_{t=0}^{T-1} \left[ \mathcal{R}(s_t, a_t) + \alpha \mathcal{H}\left(\mu(\cdot \mid s_t, M_t)\right) \right] \right]$$

**가치 함수** (Eq. 4):

$$V^\pi(s_t, M_t) = \sum_{c \in M_t} \mu(c \mid s_t, M_t) \left[ Q^\pi(s_t, M_t, c) - \alpha \log \mu(c \mid s_t, M_t) \right]$$

**Q 가치 함수** (Eq. 5):

$$Q^\pi(s_t, M_t, c_t) = \mathbb{E}_{a \sim p_{\text{LLM}}(\cdot \mid s_t, c_t),\, s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a_t)} \left[ \mathcal{R}(s_t, a_t) + \gamma V^\pi(s_{t+1}, M_{t+1}) \right]$$

**최적 검색 정책 (Closed-form solution)** (Eq. 7):

```math
\mu^*(c \mid s, M) = \frac{\exp(Q^*(s, M, c) / \alpha)}{\sum_{c' \in M} \exp(Q^*(s, M, c') / \alpha)}
```

**TD 학습 업데이트** (Eq. 8):

$$Q(s_t, M_t, c_t) \leftarrow Q(s_t, M_t, c_t) + \eta \left[ r_t + \gamma \alpha \log \sum_{c' \in M_{t+1}} \exp\left(Q(s_{t+1}, M_{t+1}, c_{t+1})\right) - Q(s_t, M_t, c_t) \right]$$

#### 2.2.3 Kernel 기반 Q-function 추정 (Neural Episodic Control 방식)

에피소딕 메모리 $\mathcal{D} = \{(s, c, Q)\}$를 활용한 커널 기반 Q-value 추정 (Eq. 9):

$$Q_{\text{EC}}(s, M, c; \theta) = \frac{\sum_{(s', c', Q') \in \mathcal{D}_c} k_\theta(s, s') Q'}{\sum_{(\hat{s}, \hat{c}, \hat{Q}) \in \mathcal{D}_c} k_\theta(s, \hat{s})}$$

**TD 학습 손실** (Eq. 10):

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,c,r,s',M,M')} \left[ \left( Q_{\text{EC}}(s, M, c; \theta) - \left[ r + \gamma \alpha \log \sum_{c' \in M'} \exp\left(Q_{\text{EC}}(s', M', c'; \bar{\theta})\right) \right] \right)^2 \right]$$

#### 2.2.4 Non-Parametric Memory Retrieval

(Eq. 13):

$$\text{Read}_{\text{NP}}(s_t, M_t) = \text{TopK}_{(s_i, a_i, r_i) \in M_t} \; \text{sim}\left(\text{enc}(s_t),\, \text{enc}(s_i)\right)$$

#### 2.2.5 Parametric Memory (단일 스텝 Q-learning, CE Loss)

이진 보상 $r \in \{0, 1\}$에 대한 Cross-Entropy 손실 (Eq. 15):

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,c,r)} \left[ -r \log Q(s, c; \theta) - (1-r) \log\left(1 - Q(s, c; \theta)\right) \right]$$

**Parametric Read** (Eq. 16):

$$\text{Read}_{\text{P}}(s_t, M_t) = \text{TopK}_{c_i \in M_t} \; Q(s_t, c_i; \theta)$$

**Parametric 메모리의 CE 그래디언트** (Eq. 26):

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{(s,c,r)} \left[ \frac{Q(s, c; \theta) - r}{Q(s, c; \theta)(1 - Q(s, c; \theta))} \nabla_\theta Q(s, c; \theta) \right]$$

---

### 2.3 모델 구조

Memento는 **Planner–Executor** 아키텍처로 구성됩니다:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Stage 1: Case-Based Planning │
│                                     │
│  ┌──────────┐    ┌───────────────┐  │
│  │Case Memory│◄──►│ LLM Planner  │  │
│  │(Write/Read│    │  (GPT-4.1)   │  │
│  └──────────┘    └───────┬───────┘  │
│                          │Plan      │
└──────────────────────────┼──────────┘
                           │
┌──────────────────────────┼──────────┐
│    Stage 2: Tool-Based Execution    │
│                          │          │
│  ┌──────────┐    ┌───────▼───────┐  │
│  │Tool Memory│◄──►│LLM Executor  │  │
│  └──────────┘    │(o3/o4-mini)  │  │
│  ┌──────────┐    └───────┬───────┘  │
│  │Subtask   │            │MCP       │
│  │Memory    │    ┌───────▼───────┐  │
│  └──────────┘    │  MCP Server   │  │
│                  │Search/Crawl/  │  │
│                  │Code/Math/...  │  │
└──────────────────┴───────────────┘
```

**세 가지 메모리 모듈**:
1. **Case Memory**: 과거 성공/실패 궤적을 벡터화하여 저장 (고수준 계획용)
2. **Subtask Memory**: 현재 실행 중인 서브태스크와 결과를 텍스트로 저장
3. **Tool Memory**: 각 서브태스크에서의 도구 호출 로그 저장

**모델 구성**:
- Planner: GPT-4.1
- Executor: o3 (GAIA), o4-mini (기타 벤치마크)
- Image: GPT-4o / Video: Gemini 2.5 Pro / Audio: Assembly AI
- Non-Parametric CBR 인코더: SimCSE (코사인 유사도 기반)
- Parametric CBR Q-function: SimCSE + 2-layer MLP

---

### 2.4 성능 향상

#### GAIA 벤치마크 (Validation)

| 에이전트 | 평균 정확도 | Level 1 | Level 2 | Level 3 |
|---------|-----------|---------|---------|---------|
| **Memento (Pass@3)** | **87.88%** | 96.23% | 90.70% | 61.54% |
| Alita | 87.27% | 88.68% | 89.53% | 76.92% |
| AWorld | 77.58% | 88.68% | 77.91% | 53.85% |
| OpenAI DeepResearch | 67.40% | 74.30% | 69.10% | 47.60% |

#### DeepResearcher 데이터셋 (7개 QA 벤치마크 평균)

| 방법 | F1 | PM |
|------|----|----|
| CoT + RAG | 37.7% | 43.2% |
| DeepResearcher (훈련 기반) | 51.8% | 60.5% |
| **Memento (GPT-4.1 + o4-mini)** | **66.6%** | **80.4%** |

#### SimpleQA 및 HLE

| 벤치마크 | Memento | 차순위 |
|---------|---------|-------|
| SimpleQA PM | **95.0%** | WebSailor 93.5% |
| HLE PM | **24.4%** | GPT-5 25.32% (2위) |

---

### 2.5 한계점

1. **Level 3 태스크**: 50단계 이상의 장기 추론이 필요한 고난도 문제에서 상대적으로 낮은 성능 (61.54%)
2. **Case Bank 포화 문제**: 약 3k 데이터 규모에서 빠르게 수렴하여, 이후 반복에서 한계 수익 체감(diminishing returns)
3. **데이터 오염(data contamination)**: DeepResearcher 일부 벤치마크에서 온라인 검색 도구가 오히려 성능을 저하시키는 현상 발견 (외부 지식과 내부 LLM 지식의 충돌)
4. **계산 비용**: Level 3 태스크에서 평균 121k 입력 토큰 소모
5. **환경 의존성**: 실험에 GPT-4.1, o3 등 상용 API 모델 사용 → 재현성 및 비용 문제
6. **메모리 스웜핑(swamping)**: K=4 이상에서 성능 정체 또는 하락 → 케이스 수가 많아질수록 노이즈 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 OOD(Out-of-Distribution) 일반화 실험 설계

논문은 **학습 데이터셋**(NQ, TQ, HotpotQA, 2Wiki)에서 케이스를 수집하고, **비학습 데이터셋**(MusiQue, Bamboogle, PopQA)에서 평가하는 방식으로 OOD 일반화를 측정합니다.

### 3.2 OOD 성능 향상 수치

| 데이터셋 | 지표 | Memento w/o CBR | Memento (with CBR) | 향상 |
|---------|------|----------------|-------------------|------|
| MusiQue | F1 | 기준 | +4.7% | **+4.7%p** |
| MusiQue | PM | 기준 | +8.0% | **+8.0%p** |
| Bamboogle | F1 | 기준 | +7.0% | **+7.0%p** |
| Bamboogle | PM | 기준 | +9.6% | **+9.6%p** |
| PopQA | F1 | 기준 | +5.3% | **+5.3%p** |
| PopQA | PM | 기준 | +5.5% | **+5.5%p** |

### 3.3 일반화 향상의 메커니즘

#### (a) 유사도 기반 케이스 전이 (Analogical Transfer)

CBR의 핵심 가정: **유사한 문제는 유사한 해결책을 가진다**. 비학습 도메인의 쿼리도 학습 도메인의 유사 케이스로부터 계획 전략을 전이받을 수 있습니다.

$$\text{Read}_{\text{NP}}(s_t, M_t) = \text{TopK}_{(s_i,a_i,r_i) \in M_t} \; \text{sim}\left(\text{enc}(s_t), \text{enc}(s_i)\right)$$

이 비파라메트릭 검색은 **도메인 레이블 없이 의미론적 유사성만으로** 케이스를 전이하므로, 도메인 간 일반화가 가능합니다.

#### (b) 실패 사례를 통한 학습 (Failure-aware Planning)

Case Bank는 성공뿐 아니라 **실패 궤적도 저장**합니다. 이를 통해:
- 과거에 실패했던 전략 회피
- 성공 전략의 재활용

$$M_{t+1} = M_t \cup \{(s_t, a_t, r_t)\} \quad \text{where } r_t \in \{0, 1\}$$

#### (c) Parametric Q-function의 패턴 학습

Parametric CBR는 상태-케이스 쌍에 대한 Q-value를 신경망으로 학습하여, **단순 유사도를 넘어 태스크 성공 가능성을 예측**:

$$Q(s, c; \theta) \approx p(r=1 \mid s, c; \theta)$$

이는 특정 케이스가 현재 태스크에 얼마나 유용할지를 직접 예측함으로써, 낯선 도메인에서도 유용한 케이스를 선별할 수 있습니다.

#### (d) 지속 학습 곡선 (Continual Learning Curves)

Table 4에서 확인된 5회 반복 학습 결과:

| 방법 | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|------|--------|--------|--------|--------|--------|
| Memento w/o CBR | 78.65 | 80.93 | 82.62 | 83.53 | 84.47 |
| w/ Non-Parametric CBR | 79.84 | 81.87 | 83.09 | 84.03 | 84.85 |
| **w/ Parametric CBR** | **80.46** | **82.84** | **84.10** | **84.85** | **85.44** |

Parametric CBR가 매 반복마다 일관되게 높은 성능을 보이며, **파라미터 업데이트 없이도 지속적 성능 향상**이 가능함을 입증합니다.

#### (e) HLE에서의 일반화: 장기 꼬리 도메인

HLE(전문 학술 영역)에서 Memento는 **24.4% PM**으로 GPT-5(25.32%) 다음 2위를 기록. CBR이 **롱테일(long-tail) 도메인**에서도 에피소딕 경험을 재사용 가능한 지식으로 전환하는 보완적 일반화 경로를 제공함을 보여줍니다.

---

## 4. 연구에 미치는 영향과 향후 연구 시 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (a) 새로운 에이전트 학습 패러다임 정립

기존의 "Fine-tuning LLM = 성능 향상"이라는 통념에 도전합니다. Memento는 **메모리 기반 온라인 RL**이 파라미터 업데이트를 대체할 수 있음을 보여, 이후 연구에서 비파라메트릭 에이전트 학습이 주류로 부상할 가능성을 열었습니다.

#### (b) M-MDP 이론적 프레임워크의 확장 가능성

M-MDP는 다중 에이전트 시스템, 멀티모달 에이전트, 실시간 의사결정 등 다양한 설정으로 확장될 수 있는 일반적 수학적 프레임워크를 제공합니다.

#### (c) CBR과 현대 LLM의 융합 연구

Aamodt & Plaza (1994)의 고전 CBR 패러다임을 현대 LLM에 성공적으로 접목함으로써, **인지과학 기반 AI 설계**의 실용적 가능성을 증명하였습니다.

#### (d) 오픈소스 딥리서치 에이전트 벤치마크

GAIA Top-1 달성으로 오픈소스 딥리서치 에이전트 연구의 새로운 기준점(baseline)을 제시합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 메모리 관리 및 스케일링

- **Case Bank 포화 문제**: 현재 약 3k 케이스에서 빠르게 수렴. 대규모 케이스 뱅크에서의 효율적 관리 방법 필요
- **메모리 압축(Memory Compression)**: 유사한 케이스를 병합하거나, 중요도에 따라 선택적으로 유지하는 메커니즘 연구
- **Ebbinghaus 망각 곡선 적용**: MemoryBank(Zhong et al., 2024)처럼 오래된 낮은 유용성 케이스를 자동으로 decay하는 방식 고려

#### (b) 케이스 유사도 측정의 고도화

현재 SimCSE + 코사인 유사도 사용. 더 정교한 유사도 측정 방법 고려:
- 태스크 구조 유사도 (structural similarity)
- 도구 사용 패턴 유사도
- 다중 모달 유사도 (이미지, 코드 포함)

#### (c) 보상 신호 설계

현재 이진 보상 $r \in \{0, 1\}$ 사용. 더 세밀한 보상 함수 설계가 필요:
- 부분 성공에 대한 연속적 보상
- 과정(process) 보상과 결과(outcome) 보상의 조합
- 불확실성을 반영한 확률적 보상

#### (d) 데이터 오염 및 신뢰성 문제

논문 자체에서 지적된 **데이터 오염(data contamination)** 문제:
- 온라인 검색 도구가 오히려 내부 LLM 지식을 방해하는 경우 발생
- 외부 검색 결과의 신뢰성 필터링 메커니즘 필요 (TrustRAG 등 참고, Zhou et al., 2025)

#### (e) 확률적 보상 및 메모리 업데이트

논문은 현재 결정론적 보상 함수와 메모리 업데이트만 다루며, 확률적 케이스는 미래 연구로 남겨두었습니다.

#### (f) 계산 비용 최적화

Level 3 태스크에서 121k 입력 토큰 소모 → 컨텍스트 압축 및 계층적 메모리 구조 탐색 필요

#### (g) 적대적 환경 및 보안

실제 배포 환경에서의 **프롬프트 인젝션**, **유해 케이스 삽입** 등의 보안 위협에 대한 강건성 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법론 | LLM 파라미터 업데이트 | 메모리 | 지속 학습 | 주요 특징 |
|------|--------|----------------------|--------|-----------|-----------|
| **RAG** (Lewis et al., 2020) | 검색-생성 결합 | ❌ | 정적 문서 DB | ❌ | 정적 지식 검색 |
| **WebGPT** (Nakano et al., 2021) | 브라우저 + LLM | ✅ (SFT+RLHF) | ❌ | ❌ | 웹 검색 특화 |
| **ReAct** (Yao et al., 2023) | 추론+행동 통합 | ❌ | ❌ | ❌ | 프롬프트 엔지니어링 |
| **Reflexion** (Shinn et al., 2023) | 언어 반성 루프 | ❌ | 언어 피드백 | 제한적 | 하드코딩된 반성 |
| **ExpeL** (Zhao et al., 2024) | 경험 증류 | ❌ | 오프라인 궤적 | 제한적 | 오프라인 경험 → 규칙 |
| **DS-Agent** (Guo et al., 2024) | CBR + LLM | ❌ | Kaggle 솔루션 | ❌ | 데이터 과학 특화 |
| **Search-R1** (Jin et al., 2025) | GRPO 기반 RL | ✅ (RL) | ❌ | ❌ | 검색-추론 공동 훈련 |
| **DeepResearcher** (Zheng et al., 2025) | RL + 실시간 웹 | ✅ (RL) | ❌ | ❌ | 웹 환경 RL 훈련 |
| **Agent-K** (Grosnit et al., 2024) | 구조적 메모리 | ❌ | 구조적 메모리 | 부분적 | Kaggle 자동화 |
| **MemoryBank** (Zhong et al., 2024) | 망각 곡선 + 검색 | ❌ | Ebbinghaus 스케줄 | 부분적 | 개인화 대화 |
| **Mem0** (Chhikara et al., 2025) | ADD/UPDATE/DELETE | ❌ | 구조화 메모리 | 부분적 | 프로덕션 에이전트 |
| **Agent-KB** (Tang et al., 2025) | 공유 지식 베이스 | ❌ | 도메인 간 공유 KB | 부분적 | 교차 도메인 경험 |
| **START** (Li et al., 2025) | 자기 훈련 추론 | ✅ (SFT) | ❌ | ❌ | 도구 사용 SFT |
| **Alita** (Qiu et al., 2025) | 자기 진화 에이전트 | ❌ | 공유 KB | 부분적 | 최소 사전 정의 |
| **Memento (본 논문)** | **M-MDP + CBR + Online RL** | **❌** | **에피소딕 Case Bank** | **✅ (온라인)** | **그래디언트 없는 지속 학습** |

### 핵심 차별점 요약

1. **vs. RAG/ExpeL**: 정적 데이터베이스가 아닌 **온라인으로 성장하는 Case Bank** 사용
2. **vs. Search-R1/DeepResearcher**: LLM 파라미터 업데이트 **불필요** → 계산 비용 대폭 절감
3. **vs. Reflexion/ReAct**: 하드코딩된 반성 워크플로우가 아닌 **학습 가능한 Q-function** 기반 케이스 선택
4. **vs. Agent-K/Agent-KB**: **수학적 형식화(M-MDP)** + **온라인 강화학습**으로 체계적 정책 최적화
5. **vs. MemoryBank/Mem0**: 단순 메모리 관리를 넘어 **강화학습 기반 정책 학습** 통합

---

## 참고 자료

**주요 논문 (본문에서 인용된 문헌)**:

1. Zhou, H., Chen, Y., et al. (2025). **"Memento: Fine-tuning LLM Agents without Fine-tuning LLMs"**. arXiv:2508.16153v2.
2. Haarnoja, T., et al. (2018). **"Soft Actor-Critic Algorithms and Applications"**. arXiv:1812.05905.
3. Haarnoja, T., et al. (2017). **"Reinforcement Learning with Deep Energy-Based Policies"**. ICML 2017.
4. Pritzel, A., et al. (2017). **"Neural Episodic Control"**. ICML 2017.
5. Aamodt, A. & Plaza, E. (1994). **"Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches"**. AI Communications, 7(1):39–59.
6. Lewis, P., et al. (2020). **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**. NeurIPS 2020.
7. Mialon, G., et al. (2023). **"GAIA: A Benchmark for General AI Assistants"**. ICLR 2024.
8. Zheng, Y., et al. (2025). **"DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-World Environments"**. arXiv:2504.03160.
9. Jin, B., et al. (2025). **"Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning"**. arXiv:2503.09516.
10. Guo, S., et al. (2024). **"DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning"**. ICML 2024.
11. Grosnit, A., et al. (2024). **"Large Language Models Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Level"**. arXiv:2411.03562.
12. Shinn, N., et al. (2023). **"Reflexion: Language Agents with Verbal Reinforcement Learning"**. NeurIPS 2023.
13. Yao, S., et al. (2023). **"ReAct: Synergizing Reasoning and Acting in Language Models"**. ICLR 2023.
14. Zhao, A., et al. (2024). **"ExpeL: LLM Agents are Experiential Learners"**. AAAI 2024.
15. Zhong, W., et al. (2024). **"MemoryBank: Enhancing Large Language Models with Long-Term Memory"**. AAAI 2024.
16. Phan, L., et al. (2025). **"Humanity's Last Exam"**. arXiv:2501.14249.
17. Wei, J., et al. (2024). **"Measuring Short-Form Factuality in Large Language Models"**. arXiv:2411.04368.
18. Tang, X., et al. (2025). **"Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving"**. arXiv:2507.06229.
19. Chhikara, P., et al. (2025). **"Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"**. arXiv:2504.19413.
20. Zhou, H., et al. (2025). **"TrustRAG: Enhancing Robustness and Trustworthiness in RAG"**. arXiv:2501.00879.
