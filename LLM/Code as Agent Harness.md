
# Code as Agent Harness

> **논문 정보**
> - **제목**: Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems
> - **저자**: Xuying Ning, Katherine Tieu, Dongqi Fu 외 39인 (42명 공동 저술)
> - **소속**: UIUC, Meta, Stanford 등
> - **arXiv**: [2605.18747](https://arxiv.org/abs/2605.18747)
> - **출판일**: 2026년 5월 18일 (arXiv v1)
> - **유형**: 서베이(Survey) 논문 (100페이지 이상)

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

최근 LLM은 경쟁 프로그래밍부터 저장소 수준의 소프트웨어 엔지니어링에 이르기까지 코드 이해와 생성에서 강력한 능력을 보여왔다. 에이전트 시스템에서 코드는 더 이상 단순한 출력 결과물이 아니며, 에이전트 추론, 행동, 환경 모델링, 실행 기반 검증을 위한 운영 기반(operational substrate)으로 점점 더 많이 활용되고 있다.

이 논문의 핵심 주장은 다음과 같이 한 문장으로 요약됩니다:

> **코드는 에이전트 인프라의 기반을 중심에 두는 통합된 관점으로서 "에이전트 하네스(agent harness)"의 렌즈를 통해 이 변화를 바라볼 것을 제안한다.**

### 🏆 주요 기여

이 논문은 UIUC, Meta, Stanford 등 연구자들이 참여한 100페이지 이상의 보고서로, 현대 에이전트 시스템에서 코드는 더 이상 단순한 출력물이 아니라는 핵심 주장을 담고 있다.

**코드의 4가지 고유한 속성 제시:**

코드의 내재적 속성으로 **실행 가능성(executability)**(모델 출력이 검증 가능한 연산으로 변환), **검사 가능성(inspectability)**(중간 연산이 구조화된 추적으로 노출), **상태 보존성(statefulness)**(태스크 진행 상태가 지속적으로 표현됨)을 강조한다.

여기에 더해, 미래 에이전트 시스템에 필요한 4가지 속성으로 **실행 가능성(executability), 검사 가능성(inspectability), 상태 보존성(statefulness), 제어 가능성(controllability)**을 제시한다.

**에이전트 하네스(Agent Harness)의 정의:**

에이전트 하네스란 LLM을 도구, API, 샌드박스, 메모리, 검증기, 권한 경계, 실행 루프, 피드백 채널로 둘러싸는 소프트웨어 레이어를 의미하며, 이를 통해 상태 비보존 모델을 장기 실행 태스크를 수행할 수 있는 기능적 에이전트로 전환한다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 구조, 성능, 한계

### 🔴 2.1 해결하고자 하는 문제

LLM의 코드 생성 및 이해 능력이 AI 에이전트 설계를 근본적으로 재편하고 있으며, 코드는 이제 단순한 출력을 넘어 에이전트 추론, 행동, 환경 모델링, 실행 기반 검증을 가능하게 하는 운영 기반으로 기능한다.

기존 문제점을 구체화하면:

1. **단일 생성 단계의 한계**: 하네스 메커니즘은 코드 기반 에이전트를 단일 생성 단계 이상으로 신뢰할 수 있게 만드는 핵심 시스템 레이어를 형성하며, 코드가 에이전트 루프에 진입하면 소프트웨어 생성은 단순히 정확한 프로그램을 생성하는 문제만이 아니다.

2. **시스템-제공 하네스와 에이전트-생성 코드 결합의 미개척 분야**: 시스템-제공 하네스 인프라는 도구, API, 샌드박스, 메모리 시스템 등으로 구성되지만, 에이전트-주도 코드 아티팩트(agent-initiated code artifacts)는 에이전트가 태스크 실행 루프 내에서 생성, 실행, 관찰, 수정, 지속 및 공유하는 대화형 코드 객체로서 상대적으로 덜 탐구된 영역이다.

---

### 🟢 2.2 제안하는 방법 및 프레임워크 구조

논문은 **3개의 연결된 레이어(Three Connected Layers)**로 구성된 통합 프레임워크를 제안합니다:

이 구성은 코드가 에이전트 루프 내에서 운영 매체가 되는 방식을 따른다: 먼저 추론, 행동, 환경 표현을 위한 하네스 인터페이스로 진입하고, 이후 시간이 지남에 따라 계획, 메모리, 도구 사용, 실행, 수정을 관리하는 하네스 메커니즘을 지원하며, 최종적으로 여러 에이전트가 저장소, 테스트, 추적, 워크플로우, 실행 상태에 걸쳐 조율하는 공유 아티팩트가 된다.

#### **Layer 1: 하네스 인터페이스 (Harness Interface)**

하네스 인터페이스는 모델과 태스크 환경 사이의 기본 인터페이스를 형성하는 방법을 연구하며, 이 레이어에서 코드는 모델 출력을 실행 가능하고 검사 가능한 구조로 변환하는 매체이다.

코드의 역할을 다음 세 가지로 분류:

| 역할 | 설명 |
|------|------|
| **Code for Reasoning** | 코드는 추론을 표현할 수 있으며, 모델은 중간 단계를 프로그램으로 표현하고, 실행한 후 출력을 검사하고 스스로 수정할 수 있다. |
| **Code for Acting** | 코드는 행동을 표현할 수 있으며, "이것을 클릭하라"는 자유 형식 언어 대신 에이전트는 구조화된 함수, 스크립트, 브라우저 명령, API, 환경 제어를 호출할 수 있다. |
| **Code for Environment Modeling** | 코드는 세계를 모델링할 수 있으며, 시뮬레이션, 파서, 의존성 그래프, 테스트 하네스, 상태 기계가 에이전트의 환경 동작 표현이 될 수 있다. |

#### **Layer 2: 하네스 메커니즘 (Harness Mechanisms)**

하네스 메커니즘은 장기 실행(long-horizon execution)을 위한 계획(planning), 메모리(memory), 도구 사용(tool use), 그리고 하네스를 신뢰할 수 있고 적응적으로 만드는 피드백 기반 제어 및 최적화를 포함한다.

수식적으로 에이전트 실행 루프를 다음과 같이 표현할 수 있습니다:

$$\mathcal{A} = \mathcal{M} + \mathcal{H}$$

여기서:
- $\mathcal{A}$: 에이전트 (Agent)
- $\mathcal{M}$: 기반 LLM 모델 (Model)
- $\mathcal{H}$: 하네스 (Harness = Tools + Memory + Planning + Feedback + Verifier)

피드백 기반 제어 루프는 다음과 같이 표현됩니다:

$$s_{t+1} = f_{\text{exec}}(a_t, s_t), \quad a_t = \pi_\mathcal{M}(s_t, \mathcal{H})$$

$$r_t = \text{Verifier}(s_{t+1}), \quad \text{Planner} \leftarrow r_t$$

여기서:
- $s_t$: 시간 $t$에서의 환경 상태 (Environment State)
- $a_t$: 에이전트의 코드 기반 행동 (Code-based Action)
- $\pi_\mathcal{M}$: LLM 정책 함수 (Policy)
- $r_t$: 실행 피드백 신호 (Execution Feedback)

**메모리 관리** 측면에서는 특히 일반화에 중요한 경험적 메모리(Experiential Memory)가 강조됩니다:

코드 에이전트가 단일 태스크 완료에서 지속적 수정 및 크로스 프로젝트 일반화로 이동함에 따라, 경험적/에피소드적 메모리에 대한 관심이 높아지고 있다. 작업 메모리(working memory)나 시맨틱 메모리와 달리, 경험적 메모리는 수정 궤적, 실패 사례, 디버깅 기록, 고차원 전략 패턴과 같이 태스크에 걸쳐 축적된 재사용 가능한 경험을 포착한다.

$$\mathcal{M}_{\text{exp}} = \{(q_i, \tau_i, r_i)\}_{i=1}^{N}$$

여기서:
- $q_i$: $i$번째 태스크 쿼리 (Task Query)
- $\tau_i$: 실행 궤적 (Execution Trajectory)
- $r_i$: 결과 및 수정 기록 (Repair Record)

#### **Layer 3: 하네스 스케일링 - 멀티 에이전트 (Scaling the Harness)**

하네스를 단일 에이전트에서 협업 에코시스템으로 확장할 때, 하네스는 개별 추론 및 실행뿐 아니라 역할 조율, 중간 아티팩트 공유, 공통 상태 유지, 집합적 진행 검증을 지원해야 한다.

멀티 에이전트 코드 중심 시스템은 매니저, 플래너, 코더, 리뷰어, 테스터와 같은 에이전트 역할과 프로그래밍, 수정, 토론, 레드팀, 적대적 상호작용과 같은 협업 모드, 그리고 중앙 집중형에서 분산형까지 다양한 워크플로우 토폴로지를 통해 검토된다.

---

### 🟡 2.3 성능 향상 사례 (Survey 내 대표 사례)

이 논문은 서베이 논문이므로 자체 실험 결과 대신, 다음 사례를 대표 예시로 제시합니다:

실제 사례로, 2026년 3월 LangChain 엔지니어링 팀은 기반 모델을 전혀 변경하지 않고 하네스 최적화만으로 코딩 에이전트를 Terminal Bench 2.0에서 30위에서 5위로 끌어올렸다.

$$\Delta \text{Rank} = 30 \to 5 \quad (\text{Model unchanged, Harness optimized only})$$

이는 하네스 엔지니어링의 파급력을 직접적으로 보여주는 사례입니다.

Cursor의 대규모 자율 코딩 실험에서도 플래너-워커 조율(planner-worker coordination)이 단일 에이전트 집중 태스크에서 공유 프로젝트의 병렬 에이전트 작업으로 확장하는 방법으로 부각되었다.

---

### 🔴 2.4 한계 (Open Challenges)

하네스 엔지니어링의 미해결 도전 과제로는 최종 태스크 성공 이상의 평가, 불완전한 피드백 하의 검증, 회귀 없는 하네스 개선, 멀티 에이전트 간 일관된 공유 상태 유지, 안전 위험 행동에 대한 인간 감독, 멀티모달 환경으로의 확장 등이 포함된다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화(Generalization)와 관련된 가장 핵심적인 내용은 **경험적 메모리(Experiential Memory)**와 **크로스 프로젝트 일반화(Cross-Project Generalization)** 개념입니다.

### 📈 3.1 일반화를 위한 핵심 메커니즘

**① 경험적 메모리 기반 일반화**

코드 에이전트가 단일 태스크 완료에서 지속적 수정 및 크로스 프로젝트 일반화로 이동함에 따라 경험적/에피소드적 메모리에 대한 관심이 증가하고 있으며, 이 메모리는 수정 궤적, 실패 사례, 디버깅 기록, 고차원 전략 패턴과 같이 태스크 전반에 걸쳐 축적된 재사용 가능한 경험을 포착한다.

일반화 가능성을 수식으로 표현하면:

$$\text{Gen}(\mathcal{A}) = \frac{1}{|\mathcal{T}_{\text{new}}|} \sum_{t \in \mathcal{T}_{\text{new}}} \mathbf{1}\left[\text{Solve}(t \mid \mathcal{M}_{\text{exp}}) = \text{correct}\right]$$

여기서 $\mathcal{T}\_{\text{new}}$는 학습 중 보지 못한 새로운 태스크 집합이며, $\mathcal{M}_{\text{exp}}$는 이전 태스크에서 축적된 경험적 메모리입니다.

**② 코드의 구조적 일반성**

자연어는 모호하지만, 코드는 명확하다. 코드는 실행되거나 실행되지 않거나 둘 중 하나이다.

이 명확성은 에이전트가 다른 도메인에서도 동일한 검증 논리를 재사용할 수 있게 해주며, 일반화의 기반이 됩니다:

$$\text{Verifier}(s) : \text{Program} \to \{0, 1\} \quad \forall s \in \mathcal{S}_{\text{domain}}$$

**③ 실행 가능성을 통한 피드백 기반 자기 수정**

코드를 에이전트 하네스로 채택하는 것은 더 강건한 AI 시스템을 위한 로드맵을 제공하며, 계획, 메모리, 도구 사용과 같은 메커니즘에 집중하고 피드백 기반 제어를 통해 신뢰성을 높임으로써 에이전트는 장기 실행(long-horizon execution)을 달성할 수 있다.

**④ 멀티 에이전트를 통한 일반화 확장**

이를 멀티 에이전트 환경으로 확장하면, 공유 코드 아티팩트가 조율과 검증을 촉진하여 이러한 이점이 더욱 증폭된다. 이 접근법은 DevOps에서 과학적 발견에 이르기까지 복잡한 애플리케이션에 필수적인 실행 가능하고 검증 가능하며 일관된 상태를 유지하는 AI 에이전트를 제공할 것을 약속한다.

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 🌟 4.1 미래 연구에 미치는 영향

**① 에이전트 설계 패러다임의 전환**

에이전트는 모델 그 자체가 아니다. 에이전트는 하네스 안에 있는 모델이다. 이 관점은 기존의 "더 큰 모델 = 더 좋은 에이전트"라는 스케일링 법칙 중심의 연구 방향을 벗어나, 하네스 엔지니어링 자체를 독립적인 연구 분야로 격상시킵니다.

**② 소프트웨어 엔지니어링 관점의 통합**

에이전트 엔지니어링이 프롬프트 엔지니어링과는 다르게 시스템 엔지니어링처럼 보이기 시작한다.

**③ 응용 도메인의 통일적 이해**

이 서베이는 분야에 더 명확한 지도를 제공한다: 인터페이스, 메커니즘, 스케일링. 코딩 에이전트, GUI 에이전트, 임베디드 에이전트, 과학 에이전트, DevOps 에이전트, 엔터프라이즈 워크플로우를 하나의 공유된 아이디어 아래 연결한다.

**④ 평가 방법론 혁신**

미해결 도전 과제로 최종 태스크 성공 이상의 평가, 불완전한 피드백 하의 검증, 회귀 없는 하네스 개선, 멀티 에이전트 간 일관된 공유 상태, 안전 위험 행동의 인간 감독, 멀티모달 환경으로의 확장이 포함된다.

이는 다음과 같은 새로운 평가 지표 연구를 촉진할 것입니다:

$$\text{Score}_{\text{agent}} = \alpha \cdot \text{TaskSuccess} + \beta \cdot \text{VerifiabilityRate} + \gamma \cdot \text{StateConsistency} + \delta \cdot \text{SafetyScore}$$

### 🔍 4.2 앞으로 연구 시 고려할 점

#### (1) 하네스-모델 공동 최적화
현재 연구의 대부분은 모델 또는 하네스를 독립적으로 최적화하지만, 진정한 성능 향상을 위해서는 공동 최적화(Joint Optimization)가 필요합니다:

```math
\theta^*, \phi^* = \arg\max_{\theta, \phi} \mathbb{E}_{t \sim \mathcal{T}}\left[\text{Reward}(\mathcal{M}_\theta, \mathcal{H}_\phi, t)\right]

```

#### (2) 안전성과 인간 감독
사전 구축된 하네스가 AI 에이전트에 일반 목적 실행 능력을 제공하지만, 엔지니어링 팀은 조직별 컴플라이언스, 안전성, 책임성을 보장하기 위해 맞춤형 스캐폴딩을 구축해야 한다.

#### (3) 멀티모달 환경으로의 확장
코드 기반 하네스를 이미지, 오디오, 비디오 등 멀티모달 입출력으로 확장하는 연구가 필요합니다.

#### (4) 회귀 없는 하네스 개선 (Regression-Free Harness Improvement)
반복적 디버깅은 하네스 루프를 완성시킨다: 개발 환경이 피드백(컴파일러 진단, 런타임 오류, 테스트, 비판)을 노출하면 에이전트는 이 신호를 진단, 수정, 점진적으로 더 나은 디버깅 행동으로 변환한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | Code as Harness와의 관계 |
|------|------|-----------|--------------------------|
| **Codex (OpenAI)** | 2021 | LLM 코드 생성 | 코드를 출력(output)으로만 봄 → 본 논문의 출발점 |
| **ReAct** | 2023 | 추론+행동 교차 반복 | 하네스 인터페이스 레이어의 원형 |
| **AutoGPT / BabyAGI** | 2023 | 자율 루프 에이전트 | 하네스 메커니즘(계획, 메모리)의 초기 구현 |
| **SWE-bench** | 2024 | 저장소 수준 코드 수정 | 크로스-프로젝트 일반화 벤치마크 |
| **Claude Code** | 2025 | 파일 I/O, 터미널, 권한 제어 | Claude Code는 파일 읽기/쓰기 도구, 터미널 명령 실행 능력, 멀티 단계 실행 루프, 위험 행동 전 인간 승인을 요청하는 권한 제어를 기본 하네스로 탑재한다. |
| **Agentic Harness Engineering (AHE)** | 2026 | 관찰 기반 하네스 자동 진화 | 시드 구성인 NexAU0는 모델에 bash 도구만 노출하는 단순한 코드 에이전트로, AHE 외부 루프의 모든 반복은 이 공통 시작점 대비 이득을 측정한다. |
| **Code as Agent Harness (본 논문)** | 2026 | 통합 서베이 프레임워크 | 위 모든 연구를 3-레이어로 통합한 통일된 로드맵 제공 |

코드를 에이전트 AI의 하네스로 중심에 두는 이 서베이는 실행 가능하고 검증 가능하며 상태 보존적인 AI 에이전트 시스템을 향한 통합된 로드맵을 제공한다.

---

## ⚠️ 중요 고지

본 논문은 특정 신규 모델이나 알고리즘을 제안하는 원구논문(research paper)이 아닌 **서베이(survey) 논문**입니다. 따라서:
- **독자적인 실험 수치나 벤치마크 결과표가 없습니다**
- 제시된 수식들은 논문에서 다루는 개념을 제가 정형화한 것으로, 논문에 명시된 수식은 아닙니다
- 수식 관련 내용은 확인이 불가능한 부분이 있으므로, 정확한 수식은 논문 원문 PDF를 직접 확인하시기를 권장합니다

---

## 📚 참고자료 (출처)

1. **arXiv 원문**: Xuying Ning et al., *Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems*, arXiv:2605.18747, May 2026. https://arxiv.org/abs/2605.18747
2. **HuggingFace Papers**: https://huggingface.co/papers/2605.18747
3. **Cool Papers (요약)**: https://papers.cool/arxiv/2605.18747
4. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/code-as-agent-harness
5. **ResearchGate PDF**: https://www.researchgate.net/publication/404992220_Code_as_Agent_Harness
6. **StartupHub.ai 분석**: https://www.startuphub.ai/ai-news/ai-research/2026/code-as-the-agent-harness
7. **ABV Applied AI Reviews (Medium)**: https://abvcreative.medium.com/code-as-agent-harness-the-boring-layer-that-may-decide-whether-agents-actually-work-a63d11053822
8. **GitHub Awesome Papers**: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers
9. **Faros.ai - Harness Engineering 2026**: https://www.faros.ai/blog/harness-engineering
10. **Agentic Harness Engineering (AHE) 관련 논문**: arXiv:2604.25850, https://arxiv.org/html/2604.25850v1
