
# Critique of Agent Model 

> **논문 정보**
> - **제목:** Critique of Agent Model
> - **저자:** Eric Xing, Mingkai Deng, Jinyu Hou
> - **소속:** SAILING Lab (CMU & MBZUAI)
> - **arXiv:** [2606.23991](https://arxiv.org/abs/2606.23991) (2026년 6월 22일 공개)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

LLM 기반 시스템들이 "coding agent", "AI co-scientist" 등으로 마케팅되며 생산성 향상을 약속하는 동시에, AI가 인간의 통제를 벗어날 수 있다는 실존적 우려도 커지고 있다. 이에 따라 **자동화(automation)와 에이전시(agency)의 경계**를 명확히 규명하는 것이 필수적이다.

이 논문은 **진정한 에이전시(genuine agency)는 외부 스캐폴딩(external scaffolding)이 아닌 시스템 자체 내부에 내재화(internalized)되어야 한다**고 주장하며, 이를 통해 규정된 태스크를 위해 설계된 "agentic" 시스템과 진정한 자율성으로 열린 세계에서 작동 가능한 "agentive" 시스템을 구분한다.

### 📌 핵심 이분법: Agentic vs. Agentive

| 구분 | Agentic System | Agentive System |
|------|---------------|----------------|
| 역량의 소재 | 외부 엔지니어링 워크플로우 | 모델 내부의 내생적 구조 |
| 목표 관리 | 외부에서 주입 | 장기 목표를 스스로 유지 |
| 학습 방식 | 수동 훈련-배포-재훈련 주기 | 자기 주도적 학습 |
| 계획 | 고정된 계획-실행 워크플로우 | 내부 세계 모델을 통한 시뮬레이션 |

**Agentic 시스템**은 오케스트레이션된 도구와 워크플로우를 통해 자율적으로 태스크를 완수하며 역량이 주로 LLM 주변 엔지니어링에 있는 반면, **Agentive 시스템**은 장기 목표 유지, 자기 정체성 진화, 미래 가능성 시뮬레이션, 추론 방식의 자기 조절, 더 나은 행동 학습 등 역량을 내생적으로 도출한다. **현재 AI 시스템은 대부분 agentic이지만 agentive는 아니며**, 역량의 대부분이 모델 자체가 아닌 워크플로우와 하네스에 존재한다.

### 🏆 주요 기여

1. **개념적 명확화:** "에이전트"와 "에이전시"의 정의를 데카르트 철학 및 SF 문학에서 도출하여 체계화
2. AI 에이전트의 현황을 조사하고, **목표(goal), 정체성(identity), 의사결정(decision-making), 자기조절(self-regulation), 학습(learning)** 5가지 차원에서 에이전트 아키텍처를 분석
3. 이 분석을 기반으로 **GIC(Goal-Identity-Configurator) 아키텍처**를 제안—계층적 목표 분해, 정체성 진화, 별도로 훈련된 세계 모델 기반 시뮬레이티브 추론, 학습된 자기조절, 실제 및 시뮬레이션 경험 기반 자기 주도 학습을 결합
4. 더 큰 자율성을 가진 agentive 시스템의 **감사 가능성(auditability), 제어 가능성(controllability), 안전성(safety)**에 대한 인사이트 제공

---

## 2. 논문의 세부 분석

### 2-1. 해결하고자 하는 문제

기존 연구들은 에이전트 모델의 일부 측면을 다루는 제안들을 제시했지만, **단일 프레임워크로 모든 측면을 체계적으로 다루면서 구현 가능한 처리 방법이 여전히 부재**하다. 이 논문은 이러한 접근법들을 분류하고 확장 가능한 범용 에이전시를 향한 한계를 분석한다.

구체적으로 논문은 다음과 같은 핵심 문제들을 지적한다:

- 에이전트 모델과 세계 모델(world model)을 명확하고 기능적으로 구분하는 정의가 부재하여, 행동 생성이 기본적으로 세계 모델 프레임워크에 흡수되어 왔다
- **목표 분해(goal decomposition), 정체성 진화(identity evolution), 자기조절 심의(self-regulated deliberation), 자기 주도 학습(self-directed learning)** 같은 에이전시의 핵심 요소들이 현재 시스템에서 결여되어 있다

---

### 2-2. 제안하는 방법: GIC 아키텍처

GIC(Goal-Identity-Configurator)는 **Belief Encoder, Goal Decomposer, Identity Evolver, Configurator, Simulative Planner, Actor**를 결합하여 에이전시를 내재화하는 범용 에이전트 아키텍처이며, 별도로 훈련된 World Model과 Critic과 함께 동작한다.

#### GIC의 5가지 핵심 구성요소

GIC 아키텍처는 다음을 결합한다:
1) **계층적 목표 분해(hierarchical goal decomposition)** — 지속적 목표 유지
2) **진화하는 정체성(evolving identity)** — 재훈련 없이 적응
3) **시뮬레이티브 플래닝(System II)** — 내부 세계 모델을 통한 계획 + 반응적 행동(System I)
4) **자기조절(learned configurator, System III)** — 언제, 얼마나 깊이 심의할지 학습
5) **자기 주도 학습** — 실제 및 시뮬레이션 경험 모두로부터

#### 주요 수식

논문은 에이전트 모델을 **잠재 변수(latent variables)**로 구성하는 형식화를 도입한다.

에이전트 모델은 **목표(goals), 정체성(identity), 계획(plans), 조절 메커니즘(regulation mechanisms)**을 잠재 변수로 도입하여 내생적 에이전시의 속성을 형식화함으로써 구성된다.

> **에이전트의 상태 표현 (잠재 변수 기반):**
>
> $$s_t = (b_t, g_t, \theta_t)$$
>
> 여기서:
> - $b_t$: Belief Encoder로부터 얻은 현재 환경에 대한 신념(belief) 상태
> - $g_t$: 계층적으로 분해된 목표(goal) — Goal Decomposer 출력
> - $\theta_t$: 에이전트의 정체성(identity) 파라미터 — Identity Evolver 출력

> **세계 모델(World Model) $f$와 에이전트 행동의 분리:**
>
> $$s_{t+1} \approx f(s_t, a_t)$$
>
> 에이전트는 별도로 훈련된 세계 모델 $f$를 활용하여 잠재적 미래를 시뮬레이션하고 최적 행동 시퀀스를 계산한다(Simulative Reasoning, System II).

> **Configurator (System III) — 자기조절 메커니즘:**
>
> $$c_t = \text{Configurator}(b_t, g_t, \theta_t)$$
>
> $$a_t = \begin{cases} \text{System I: 반응적 행동} & \text{if } c_t = \text{fast} \\ \text{System II: 시뮬레이티브 플래닝} & \text{if } c_t = \text{deliberate} \end{cases}$$
>
> 자기조절은 단순한 계산 스케줄러가 아니라 상황 평가를 기반으로 에이전트의 우선순위와 행동 레퍼토리를 구성하는 **인간의 감정과 유사한 기능**을 한다. 또한 Configurator는 에이전트가 언제 어떻게 경험으로부터 학습해야 하는지를 결정하는 역할도 담당한다.

> **자기 주도 학습 (Self-Directed Learning):**
>
> $$\mathcal{L}_{\text{agent}} = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] + \mathcal{L}_{\text{identity}}(\theta_t)$$
> $$\mathcal{L}_{\text{world}} = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}[\| f(s,a) - s' \|^2]$$
>
> 세계 모델 $f$와 에이전트 정책 $\pi_\theta$는 **서로 다른 신호로부터 학습**한다.

---

### 2-3. 모델 구조 다이어그램

```
┌────────────────────────────── GIC Architecture ──────────────────────────────┐
│                                                                              │
│  Environment ──► Belief Encoder (b_t)                                        │
│                        │                                                    │
│                  ┌─────▼──────┐    ┌──────────────────┐                     │
│                  │Goal Decomposer│   │Identity Evolver  │                    │
│                  │   (g_t)    │   │    (θ_t)          │                     │
│                  └─────┬──────┘   └────────┬─────────┘                      │
│                        └──────────┬─────────┘                               │
│                                   ▼                                         │
│                          Configurator (System III)                          │
│                          c_t = fast / deliberate                            │
│                         /                      \                            │
│               System I (fast)          System II (deliberate)               │
│               Reactive Actor       Simulative Planner                       │
│                                   + World Model f                           │
│                         \                      /                            │
│                          ──────── Action a_t ──────────► Environment        │
│                                                                              │
│  Self-Directed Learning ◄──── Real + Simulated Experience ◄────────────────  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 2-4. 성능 향상 및 한계

#### ✅ 강점

GIC는 에이전트가 자신의 심의 모드, 목표 분해, 정체성 진화를 자율적으로 관리하도록 함으로써 **영속적이고 자기 주도적인 운용**을 가능하게 하며, 단일체 LLM 기반 에이전트 설계에 비해 감사 가능성과 안전성을 향상시킨다.

세계 행동 모델(WAM: World Action Models)과 같은 시스템들은 세계 모델 측면을 정책 자체에 통합하여, 대규모 데이터로부터 물리적 사전 지식(physical priors)을 습득하고 **미지의 태스크 및 환경에 대한 일반화(generalization to unseen tasks and environments)**를 시연하고 있다.

#### ⚠️ 한계

현재 시스템들은 **감각 레퍼토리(sensory repertoire)가 제한적**(예: 힘, 질감, 경도, 온도 인식 부재)이며, 목표 분해, 정체성 진화, 자기조절 심의, 자기 주도 학습 등 에이전시의 중요한 측면들이 여전히 부재하다.

현재 접근법들은 에이전트를 인간이 설계한 파이프라인(예: 시뮬레이터에서의 RL, 지도 학습 데모)을 통해 훈련하고 고정된 체크포인트를 배포하는 방식으로, **에이전트 스스로 학습을 주도하지 못한다**.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화(Generalization)는 이 논문의 가장 핵심적인 동기 중 하나이다.

### 3-1. 외부 스캐폴딩 탈피 → 오픈 월드 일반화

**Agentic 시스템**의 역량은 엔지니어링된 워크플로우에 존재하고, **Agentive 시스템**의 역량(사회적 상호작용 포함)은 내생적으로 발생한다. 이것이 규정된 태스크를 위해 설계된 시스템과 **진정한 자율성으로 오픈 월드에서 운용 가능한 시스템** 사이의 경계를 정의한다.

→ 즉, **내생적 구조를 내재화한 모델은 사전에 규정되지 않은 환경에서도 일반화**할 수 있다는 것이 논문의 핵심 주장이다.

### 3-2. 계층적 목표 분해 → 새로운 태스크로의 전이

$$G = \{g^{(0)}, g^{(1)}, \ldots, g^{(L)}\}$$

여기서 $g^{(0)}$는 최상위 장기 목표, $g^{(L)}$는 즉시 실행 가능한 하위 목표이다.

계층적 목표 분해는 새로운 태스크를 접했을 때 **기존 하위 목표 구조를 재조합**함으로써 제로샷(zero-shot) 수준의 일반화를 지원한다.

### 3-3. 정체성 진화 → 재훈련 없는 도메인 적응

**진화하는 정체성(evolving identity)**은 재훈련 없이 적응할 수 있으며, 내부 세계 모델(System II)을 통한 시뮬레이티브 계획이 반응적 행동(System I)과 함께 제공된다.

정체성 파라미터 $\theta_t$는 새로운 도메인에서 경험이 쌓이면서 점진적으로 업데이트:

$$\theta_{t+1} = \theta_t + \alpha \cdot \nabla_\theta \mathcal{L}_{\text{identity}}(\theta_t, \text{experience}_t)$$

이를 통해 Fine-tuning 없이도 **도메인 시프트(domain shift)**에 적응 가능하다.

### 3-4. 세계 모델 기반 시뮬레이션 → 분포 외(Out-of-Distribution) 일반화

내부 세계 모델 $f$를 통해 에이전트는 실제로 경험하지 않은 상황을 사전에 시뮬레이션할 수 있다:

$$a^* = \arg\max_{a} \mathbb{E}_{f}[R(s_{t+1}, \ldots, s_{t+H}) \mid s_t, a]$$

이는 **모델 기반 강화학습(Model-Based RL)**의 일반화 이점을 극대화한다.

### 3-5. 자기 주도 학습 → 지속적 일반화 능력 향상

에이전트는 실제 환경과 시뮬레이션 모두에서 학습하며 일반화 성능을 지속적으로 개선한다:

$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{real}} \cup \mathcal{D}_{\text{simulated}}$$

이 논문은 기존 agentic 시스템들의 **확장 가능하고 범용적인 에이전시를 향한 한계**를 분석한다. 이는 일반화가 단순히 모델 크기를 키우는 것이 아니라, 내생적 구조의 설계를 통해 달성되어야 함을 강조한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 접근법 | GIC와의 관계 |
|------|--------|------------|
| **ReAct (Yao et al., 2022)** | LLM + 추론-행동 교차 | 외부 스캐폴딩 의존 (agentic) |
| **Reflexion (Shinn et al., 2023)** | 언어적 강화학습 | 자기조절 일부 구현, but 재훈련 필요 |
| **AutoGPT / BabyAGI (2023)** | 외부 플래너 + LLM | 전형적인 agentic 시스템 |
| **DreamerV3 (2023)** | 세계 모델 + 강화학습 | GIC의 System II와 유사 |
| **VLA 모델들 (2024~)** | 비전-언어-행동 통합 | 물리적 사전지식 일반화 추구 |

최근 Vision-Language-Action(VLA) 아키텍처들이 데모, 모방 학습, 대규모 시뮬레이션으로 훈련되고 있으며, World Action Models(WAMs)은 공유 아키텍처 내에서 미래 상태와 행동을 함께 예측하면서 세계 모델 측면을 정책 자체에 통합하고 있다.

인지 과학에서 영감을 받은 모델, 계층적 강화학습 프레임워크, LLM 기반 추론의 인사이트를 종합하는 방향으로 연구가 진행되고 있으며, 실세계 시나리오에 에이전트를 배포할 때 발생하는 윤리, 안전성, 해석 가능성 문제들이 핵심 과제로 부상하고 있다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① 패러다임 전환의 기준점 제공**

"더 나은 에이전트는 더 나은 하네스(harness)에서 오는 것이 아니라, **스스로를 하네스할 수 있는 모델**에서 온다." — 이 명제는 향후 에이전트 연구의 설계 철학을 근본적으로 재정의할 것이다.

**② GIC 5축 프레임워크의 평가 기준화**

진정한 인공 에이전시는 **목표, 정체성, 의사결정, 자기조절, 학습에 대한 내재화된 구조**를 요구하며, 이는 자율 시스템과 태스크 특화 시스템을 구분한다는 주장은 새로운 에이전트 시스템 평가의 표준 축이 될 수 있다.

**③ 안전성 연구와의 연계**

감사 가능성(auditability), 제어 가능성(controllability), 안전성에 대한 통찰 제공은 에이전트 정렬(AI alignment) 연구와 직접 연결되어, 해석 가능한 에이전트 설계 연구를 촉진할 것이다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### 🔬 기술적 고려사항

1. **세계 모델과 에이전트 모델의 훈련 신호 분리**
   - 두 모델은 서로 다른 최적화 목표를 가지므로, 공동 훈련 시 목표 충돌(objective conflict) 방지 전략 필요

   $$\min_\phi \mathcal{L}\_{\text{world}}(\phi) \quad \text{and} \quad \max_\theta \mathcal{L}_{\text{agent}}(\theta)$$

   (단, $\phi$와 $\theta$가 상호 의존적일 때 안정적 학습 보장 방법 연구 필요)

2. **정체성(Identity) 파라미터의 연속적 업데이트와 Catastrophic Forgetting 방지**
   
   - Continual Learning 기법 (EWC, LoRA 기반 점진적 업데이트 등)과의 결합 연구 필요

3. **Configurator(System III)의 메타-학습**
   
   - "언제 심의할 것인가"를 결정하는 컨피규레이터 자체를 학습하기 위한 메타-강화학습(Meta-RL) 프레임워크 연구 필요

   $$c^* = \arg\max_c \mathbb{E}[R_{\text{total}} - \lambda \cdot C_{\text{compute}}(c)]$$
   
   여기서 $C_{\text{compute}}$는 심의(deliberation) 비용.

4. **감각 레퍼토리 확장**

   현재 시스템들은 힘, 질감, 경도, 온도 등의 감각 정보가 없는 제한된 감각 레퍼토리를 가지고 있다는 점에서 멀티모달 감각 통합(haptic, tactile 등) 연구가 필요하다.

#### 🛡️ 안전성 및 제어 관련 고려사항

5. **정체성 진화의 안전 경계 설정**
   
   - 에이전트가 스스로 정체성을 진화시킬 때 인간이 정의한 가치와 규범을 벗어나지 않도록 하는 **Constitutional AI** 또는 **RLHF와의 결합** 필요

6. **목표 계층의 투명성 (Auditability)**
   
   - 계층적 목표 분해 과정을 외부에서 감사(audit)할 수 있는 해석 가능한 표현 방식 연구 필요

#### 📐 평가 및 벤치마크 고려사항

7. **5축 에이전시 평가 벤치마크 부재**
   
   - 현재 기존 벤치마크(GAIA, WebArena 등)는 주로 태스크 완수 성능만 측정. GIC가 제안하는 5가지 축(goal, identity, decision-making, self-regulation, learning)을 종합적으로 평가하는 **새로운 벤치마크 설계**가 시급하다.

8. **장기(Long-Horizon) 자율성 평가**
   
   - 단기 태스크가 아닌 수일~수개월 단위의 장기 목표 달성 능력을 평가하는 실험 프로토콜 개발 필요

---

## 📚 참고 자료 및 출처

| # | 자료명 | URL |
|---|--------|-----|
| 1 | **Critique of Agent Model** (원문, arXiv 2606.23991) | https://arxiv.org/abs/2606.23991 |
| 2 | **Critique of Agent Model** (HTML 전문) | https://arxiv.org/html/2606.23991 |
| 3 | **Critique of Agent Model** (PDF) | https://arxiv.org/pdf/2606.23991 |
| 4 | Hugging Face Paper Page | https://huggingface.co/papers/2606.23991 |
| 5 | alphaXiv Abstract | https://www.alphaxiv.org/abs/2606.23991 |
| 6 | Moonlight Literature Review | https://www.themoonlight.io/en/review/critique-of-agent-model |
| 7 | GitHub - Autonomous-Agents (tmgthb) | https://github.com/tmgthb/Autonomous-Agents |
| 8 | A Comprehensive Review of AI Agents (arXiv 2508.11957) | https://arxiv.org/abs/2508.11957 |
| 9 | World Model for Robot Learning Survey (arXiv 2605.00080) | https://arxiv.org/html/2605.00080v1 |
| 10 | MUSE: Competence-Aware AI Agents (arXiv 2411.13537) | https://arxiv.org/pdf/2411.13537 |

---

> **⚠️ 주의:** 이 논문은 2026년 6월 22일에 공개된 **매우 최신 논문**으로, 아직 peer review를 거치지 않은 arXiv 프리프린트입니다. GIC 아키텍처는 이론적 제안 단계이며, 실험적 벤치마크 비교나 정량적 성능 수치는 현재 공개된 정보에 포함되어 있지 않습니다. 향후 공식 출판 시 내용이 변경될 수 있습니다.
