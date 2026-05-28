
# OpenHands: An Open Platform for AI Software Developers as Generalist Agents

> **논문 기본 정보**
> - **제목**: OpenHands: An Open Platform for AI Software Developers as Generalist Agents
> - **저자**: Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu 외 20인 (총 24인)
> - **arXiv**: [2407.16741](https://arxiv.org/abs/2407.16741) (v1: 2024.07.23, v3: 2025.04.18)
> - **게재**: ICLR 2025 (International Conference on Learning Representations)
> - **구 명칭**: OpenDevin

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

소프트웨어는 인간이 보유한 가장 강력한 도구 중 하나이며, 동시에 LLM의 발전으로 주변 환경에 상호작용하고 변화를 일으키는 AI 에이전트의 급속한 발전이 이루어지고 있다.

OpenHands의 핵심 주장은 다음과 같다:

> **"AI 에이전트가 인간 개발자처럼 소프트웨어를 통해 세상과 상호작용할 수 있는 범용(Generalist) 플랫폼이 필요하다"**

OpenHands는 인간 개발자와 유사한 방식으로 — 코드 작성, 커맨드 라인 상호작용, 웹 브라우징 — 세상과 상호작용하는 강력하고 유연한 AI 에이전트 개발을 위한 플랫폼이다.

### 1.2 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|---|---|
| 오픈 플랫폼 제공 | 에이전트 구현·평가·배포를 위한 통합 프레임워크 |
| CodeAct 기반 범용 에이전트 | 코드 실행 기반의 강력한 Generalist 에이전트 |
| 샌드박스 런타임 | Docker 기반 안전한 코드 실행 환경 |
| 멀티에이전트 협력 | 에이전트 간 위임·협력 메커니즘 |
| 15개 벤치마크 평가 프레임워크 | 통합 평가 체계 |
| MIT 오픈소스 생태계 | 커뮤니티 기반 확장 가능성 |

이 플랫폼은 새로운 에이전트 구현, 코드 실행을 위한 샌드박스 환경과의 안전한 상호작용, 다중 에이전트 간 조정, 평가 벤치마크 통합을 지원한다.

MIT 라이선스로 공개된 OpenHands는 학계와 산업계에 걸쳐 188명 이상의 기여자로부터 2,100건 이상의 기여를 받은 커뮤니티 프로젝트이다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

### 2.1 해결하고자 하는 문제

에이전트가 복잡한 소프트웨어 시스템에서 효과적으로 코드를 생성·수정하고, 실시간으로 정보를 수집하여 문제를 디버깅하며, 사용자 시스템에 부작용 없이 안전하게 개발을 수행하도록 하는 것이 핵심 과제이다.

구체적으로는 세 가지 문제를 해결한다:

1. **인터페이스 문제**: 에이전트가 환경과 상호작용하는 표준화된 방법 부재
2. **안전성 문제**: 임의 코드 실행 시 호스트 시스템 보호 필요
3. **평가 문제**: 다양한 태스크에 걸친 통합적·재현 가능한 평가 체계 부재

---

### 2.2 제안하는 방법 및 수식

#### 2.2.1 Event Stream 기반 Perception-Action Loop

OpenHands의 핵심 설계 원리는 **이벤트 스트림(Event Stream)** 추상화이다.

OpenHands는 행동(Action)과 관찰(Observation)을 캡처하는 이벤트 스트림 추상화를 통해 에이전트-환경 인터페이스를 정의하며, 인간 소프트웨어 개발자가 수행하는 것과 유사한 지각-행동 루프를 형성한다. 각 에이전트는 환경 이벤트 이력을 읽고 현재 세션에서 실행될 다음 원자적 행동을 생성한다.

이 루프는 다음의 수식으로 형식화할 수 있다:

$$a_t = \pi_\theta(s_t), \quad s_t = \{e_0, e_1, \ldots, e_{t-1}\}$$

여기서:
- $a_t$: 시각 $t$에서의 에이전트 행동 (Action)
- $\pi_\theta$: LLM으로 파라미터화된 에이전트 정책
- $s_t$: 현재까지의 이벤트 이력 (Event Log)
- $e_i$: $i$번째 이벤트 (Action 또는 Observation)

대화(Conversation)는 Action과 Observation의 타입이 정해진 Pydantic 이벤트 스트림으로, 불변(immutable)하고 재실행(replayable) 가능하다. 이것이 OpenHands의 가장 중요한 설계 결정으로, 자율 에이전트는 이벤트 이력에서 다음 이벤트로의 순함수(pure function)로 모델링되며 루프 안에서 실행된다. 이를 통해 일시정지, 재개, 포크, 결정론적 재실행, 완전한 감사 추적이 자연스럽게 지원된다.

#### 2.2.2 CodeAct 아키텍처

CodeActAgent는 원 논문의 핵심 관찰에 기반한다: LLM에게 각자의 JSON 스키마를 가진 20개의 맞춤형 도구를 제공하는 대신, bash, Python, 브라우저 DSL을 제공하고 모든 것을 코드로 표현하게 한다. 경험적으로 이것이 훨씬 더 잘 일반화되며 파싱 오류를 크게 줄인다.

CodeAct의 행동 공간은 다음과 같이 정의된다:

$$\mathcal{A} = \mathcal{A}_{\text{code}} \cup \mathcal{A}_{\text{bash}} \cup \mathcal{A}_{\text{browse}} \cup \mathcal{A}_{\text{delegate}}$$

- $\mathcal{A}_{\text{code}}$: Python/코드 실행 행동
- $\mathcal{A}_{\text{bash}}$: 쉘 커맨드 실행 행동
- $\mathcal{A}_{\text{browse}}$: 웹 브라우저 상호작용 행동
- $\mathcal{A}_{\text{delegate}}$: 서브 에이전트에 위임 행동

#### 2.2.3 에이전트 루프 (Agent Loop)

$$\text{for } t = 0, 1, 2, \ldots, T_{\max}:$$

$$\quad \text{prompt}_t = \text{Condense}(\{e_0, \ldots, e_{t-1}\})$$

$$\quad a_t = \text{LLM}(\text{prompt}_t)$$

$$\quad o_t = \text{Runtime.execute}(a_t)$$

$$\quad \mathcal{E} \leftarrow \mathcal{E} \cup \{a_t, o_t\}$$

여기서 $\text{Condense}(\cdot)$는 컨텍스트 길이 제한 내에서 이벤트 이력을 요약하는 함수이며, $T_{\max}$는 최대 반복 횟수이다.

---

### 2.3 모델 구조

#### 전체 아키텍처

```
┌─────────────────────────────────────────────┐
│              OpenHands Platform              │
│                                             │
│  ┌──────────┐    ┌────────────────────────┐  │
│  │  Agent   │───▶│     Event Stream        │  │
│  │  (LLM)   │    │  (Actions/Observations) │  │
│  └──────────┘    └──────────┬─────────────┘  │
│        ▲                    │                 │
│        │                    ▼                 │
│  ┌─────┴──────┐   ┌─────────────────────┐    │
│  │Observation │   │   Sandboxed Runtime  │    │
│  │  Return    │   │   (Docker Container) │    │
│  └────────────┘   └─────────────────────┘    │
└─────────────────────────────────────────────┘
```

OpenHands의 핵심 아키텍처 구성 요소로는 Docker 기반 샌드박싱(각 세션이 SSH로 접근되는 격리 컨테이너), Jupyter 커널 환경(상태 유지 코드 상호작용), Browser Agent API(BrowserGym 인터페이스), 멀티에이전트 위임(계층적 에이전트 구조), 그리고 코드 실행과 추론을 결합한 기본 범용 에이전트인 CodeAct 아키텍처가 있다.

OpenHands는 10개 이상의 구현된 에이전트를 포함하는 에이전트 허브를 보유하며, CodeAct 아키텍처 기반의 강력한 범용 에이전트와 함께 웹 브라우징 및 코드 편집 전문 에이전트가 포함된다.

#### 샌드박스 런타임

OpenHands Docker Runtime은 AI 에이전트 행동의 안전하고 유연한 실행을 가능하게 하는 핵심 구성 요소이다. Docker를 사용해 임의 코드를 호스트 시스템 위험 없이 실행하는 샌드박스 환경을 생성한다. 보안(신뢰할 수 없는 코드 실행 위험 방지), 일관성(환경 차이 제거), 리소스 제어(프로세스 폭주 방지)를 위해 이 환경이 필요하다.

#### AgentSkills 라이브러리

AgentSkills 라이브러리는 SWE-Agent와 Aider로부터 적용된 파일 편집 유틸리티를 포함하며, 비전-언어 모델(예: GPT-4V)을 사용해 이미지에서 정보를 추출하는 `parse_image`와 PDF 텍스트 읽기를 위한 `parse_pdf` 등 멀티모달 문서 지원 도구도 포함한다.

#### 멀티에이전트 시스템

OpenHands는 멀티에이전트 위임을 지원하여, 전문화된 에이전트가 서브태스크에서 협력할 수 있으며, 예를 들어 범용 CodeActAgent가 웹 브라우징을 BrowsingAgent에 위임하는 구조가 예시이다.

---

### 2.4 성능 향상

OpenHands 내의 포괄적인 평가 프레임워크는 소프트웨어 엔지니어링과 웹 브라우징 태스크를 포함한 15개의 어려운 벤치마크에서 에이전트의 체계적 평가를 지원한다.

#### SWE-Bench 성능

OpenHands는 Claude Sonnet 4.5로 SWE-Bench Verified에서 약 77%의 점수를 달성한다.

OpenHands는 자체 하네스에서 Claude 3.5 Sonnet Thinking을 사용해 SWE-bench Verified 77.6%를 보고하며, 표준화된 mini-SWE-agent 하네스에서는 일반적으로 72~76% 수준이다.

#### 성능 비교표 (SWE-Bench Verified 기준)

| 시스템 | 백본 LLM | SWE-Bench Verified 점수 |
|---|---|---|
| OpenHands + CodeAct v2.1 | Claude-3-5-Sonnet | ~41% (2024.11) |
| OpenHands (자체 평가) | Claude Sonnet 4.5 | ~77% (2025) |
| SWE-Agent | Claude 3.5 Sonnet | ~43% (2024) |
| AutoCodeRover-v2.0 | Claude-3.5-Sonnet | ~50% (2024) |

---

### 2.5 한계

단일 통합 행동 공간(unified action space)의 단점은 LLM이 강력한 코드 생성기여야 한다는 점이다. 더 약한 모델에서는 더 좁고 더 가이드된 도구가 필요할 수 있다.

Docker 의존성, 즉 완전한 샌드박스를 위해 Docker 데몬이 실행 중이어야 하므로 모든 시스템에서 사용 가능하지 않을 수 있다는 점도 한계이다.

V0는 모든 도구 호출이 안전성과 재현성을 위해 샌드박스 Docker 컨테이너 내에서 실행되어야 한다는 가정에 기반했다. 그러나 이로 인해 각 대화가 두 독립적 프로세스(에이전트와 샌드박스)로 분리되어 잠재적으로 상태가 불일치하는 여러 층의 마찰이 발생했다. 샌드박스가 충돌하는 동안 에이전트가 계속 실행되거나 그 반대의 경우가 발생하여 세션이 손상되는 문제가 있었다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 CodeAct를 통한 일반화

CodeActAgent의 핵심 통찰은, LLM에게 각자의 JSON 스키마를 가진 도구를 개별 제공하는 대신 bash, Python, 브라우저 DSL을 주고 모든 것을 코드로 표현하게 하는 것이다. 경험적으로 이것이 훨씬 더 잘 일반화되며 파싱 오류를 크게 줄인다.

수학적으로, 기존 방식의 도구별 전문화 접근은:

$$\pi_\theta(a | s) = \sum_{k=1}^{K} \mathbf{1}[a \in \mathcal{A}_k] \cdot p_k(a|s)$$

와 같이 $K$개의 도구 공간으로 분리되는 반면, CodeAct는:

$$\pi_\theta(a | s) = p_{\text{code}}(a | s), \quad a \in \mathcal{A}_{\text{unified}}$$

처럼 통합 코드 공간에서 작동하여 **분포 외(Out-of-Distribution) 태스크에 대한 일반화** 능력이 향상된다.

### 3.2 Critic 모델과 추론 시간 스케일링을 통한 일반화

실제 일반화는 보장하기 어렵지만, 충분한 데이터가 있다면 학습된 Critic 모델이 SWE-Bench를 넘어서 다양한 소프트웨어 엔지니어링 시나리오로 일반화될 수 있다고 믿는다. 이는 일상적인 코딩 태스크에서의 실제 문제를 해결하는 데 유용한 도구가 될 것이다.

추론 시간 스케일링(Inference-Time Scaling)을 통한 일반화 가능성 공식:

$$a^* = \arg\max_{a \in \{a_1, \ldots, a_N\}} \text{Critic}(a | s)$$

여기서 $N$은 병렬로 샘플링된 솔루션 수이며, $\text{Critic}(\cdot)$은 학습된 평가 모델이다.

현재 구현은 여러 궤적 중 최선의 완성된 솔루션을 선택하는 데 초점을 맞추고 있으나, 각 궤적 전반에 걸쳐 예측된 중간 보상이 에이전트의 능력을 향상시키는 흥미로운 가능성을 열어준다. 여러 완성된 솔루션 생성이 실용적이지 않은 시나리오에서도 더 효율적인 지원을 가능하게 할 수 있도록 이러한 신호를 더 깊이 통합하는 작업을 진행 중이다.

### 3.3 실제 환경 일반화 한계

최고 성능 에이전트인 OpenHands + Claude 3.7 Sonnet은 SWE-bench-Live에서 19.25%의 해결률만을 달성하는 반면, 동일한 설정이 SWE-bench Verified에서는 43.20%를 달성하여 두 배 이상의 차이를 보인다. 이 격차는 벤치마크 친숙도뿐만 아니라 SWE-bench-Live의 더 높은 다양성에서 비롯된 것으로 분석된다. 이러한 결과는 정적이고 수동으로 큐레이션된 벤치마크의 한계를 부각시키며, 견고하고 일반화 가능한 코드 에이전트 시스템 발전을 위해 동적이고 자동으로 업데이트되는 테스트베드의 중요성을 강조한다.

이를 일반화 갭(Generalization Gap)으로 정의하면:

$$\Delta_{\text{gen}} = \text{Score}_{\text{static}} - \text{Score}_{\text{live}} = 43.20\% - 19.25\% = 23.95\%$$

이는 현재 AI 에이전트의 일반화 성능이 정적 벤치마크 점수에 비해 실제 환경에서 크게 저하됨을 의미한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 소프트웨어 엔지니어링 에이전트 계보

| 연구 | 연도 | 핵심 기여 | OpenHands와의 차이 |
|---|---|---|---|
| **SWE-bench** (Jimenez et al.) | 2023 | 실제 GitHub 이슈 기반 평가 | 평가 벤치마크 (플랫폼 아님) |
| **SWE-Agent** (Yang et al.) | 2024 | ACI(Agent-Computer Interface) | 단일 에이전트 특화, 오픈소스 |
| **Devin** (Cognition AI) | 2024 | 최초의 상용 AI 소프트웨어 엔지니어 | 폐쇄 소스, 단일 제품 |
| **OpenHands (OpenDevin)** | 2024 | 개방형 범용 플랫폼 | 오픈소스, 멀티에이전트, 15개 벤치마크 |
| **Agentless** | 2024 | 비에이전트 방식 패치 생성 | 에이전트 루프 없이 localize→repair |
| **SWE-bench-Live** | 2025 | 동적·실시간 벤치마크 | 평가 환경 개선 |
| **SWE-EVO** | 2025 | 장기적 소프트웨어 진화 시나리오 | 다중 커밋·장기 계획 평가 |

최근 모델들은 SWE-Bench-Verified에서 약 75%를 달성하고 있어 포화(saturation) 징후를 보이고 있으며, SWE-Bench는 이산적 이슈 해결에 집중하여 소프트웨어 엔지니어링의 핵심 과제인 기존 시스템의 지속적 진화를 포착하지 못한다는 한계가 있다.

### 4.2 에이전트 프레임워크 비교

에이전트 프레임워크는 일반적으로 1) 에이전트가 세상과 상호작용하는 인터페이스(JSON 기반 함수 호출 또는 코드 실행 등), 2) 에이전트가 운영되는 환경, 3) 인간-에이전트 또는 에이전트-에이전트 통신을 위한 상호작용 메커니즘을 포함한다.

OpenHands는 이 세 요소를 모두 통합한 종합 플랫폼으로, SWE-Agent, LangChain, AutoGPT 등 기존 프레임워크와 차별화된다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### (1) 에이전트 연구 표준화 기여
OpenHands는 단순한 개념적 프레임워크가 아니라 에이전트, 환경, 평가의 포괄적이고 즉시 사용 가능한 구현을 포함한다. 이는 이후 연구자들이 동일한 기반 위에서 반복 가능한 실험을 진행할 수 있는 기반을 마련한다.

#### (2) 멀티에이전트 연구 방향 제시
OpenHands의 위임(delegation) 구조는 다음과 같은 연구 방향을 제시한다:

$$\pi_{\text{meta}}(s) \rightarrow \{(\pi_{\text{sub}_1}, \tau_1), \ldots, (\pi_{\text{sub}_K}, \tau_K)\}$$

범용 에이전트가 여러 전문 에이전트에 서브태스크 $\tau_k$를 배분하는 계층적 구조 연구가 활성화될 것이다.

#### (3) 실제 소프트웨어 개발 자동화
OpenHands 에이전트가 버그 리포트 및 기능 요청과 같은 실제 소프트웨어 엔지니어링 이슈 해결에 얼마나 효과적인지에 대한 연구 질문을 제기하며, 이는 실용적 자동화 연구의 핵심 과제가 된다.

#### (4) 아키텍처 진화 연구
V0의 모놀리식 샌드박스 중심 설계가 구성 요소를 강하게 결합하고 중복 구현을 요구했다면, V1은 명확한 경계, 선택적 샌드박싱, 재사용 가능한 패키지를 갖춘 모듈형 SDK로 재구성되었다. 이 아키텍처 진화는 프로덕션급 AI 에이전트 설계의 모범 사례로 기능할 것이다.

### 5.2 앞으로 연구 시 고려할 점

#### ① 일반화 갭(Generalization Gap) 해소
정적이고 수동으로 큐레이션된 벤치마크의 한계를 극복하고 견고하고 일반화 가능한 코드 에이전트 시스템 발전을 위해 동적이고 자동으로 업데이트되는 테스트베드가 필요하다.

연구 방향:

```math
\mathcal{L}_{\text{gen}} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{\text{live}}} [-\log \pi_\theta(a^* | s)]
```

정적 데이터셋이 아닌 실시간 동적 데이터($\mathcal{D}_{\text{live}}$)로 학습하는 지속적 학습 전략 필요.

#### ② 장기 과제(Long-Horizon Task) 처리
실제 환경에서는 노력의 최대 80%가 레거시 코드 유지 및 진화에 소요되며, 모듈, 버전, 명세에 걸친 조율된 변경이 필요하다.

이를 위해 장기 메모리와 계획 능력이 통합된 에이전트 설계가 요구된다:

$$\pi_\theta(a_t | s_t, \mathcal{M}_t), \quad \mathcal{M}_t = \text{Memory}(e_0, \ldots, e_{t-1})$$

#### ③ 비용-성능 트레이드오프
최대 반복 횟수(`MAX_ITERATIONS`, 기본값 ~100), LLM 재시도 횟수(`LLM_NUM_RETRIES`, 기본값 8), 그리고 대화를 중단하는 하드 누적 비용 한도를 반드시 설정해야 한다. 이 세 가지 없이 헤드리스 에이전트를 배포해서는 안 된다.

비용 $C$와 성능 $P$의 트레이드오프:
$$\text{Efficiency} = \frac{P(\theta)}{C(\theta)} = \frac{\text{Resolve Rate}}{\text{Total LLM Calls} \times \text{Cost per Call}}$$

#### ④ 약한 LLM 백본에서의 성능 저하 대응
CodeAct의 단일 통합 행동 공간은 LLM이 강력한 코드 생성기여야 한다는 점에서 약한 모델에는 더 좁고 가이드된 도구가 필요할 수 있다. Claude Sonnet 4.5 / GPT-5 수준에서는 "쉘 제공"이 가장 강력한 기준이 된다.

#### ⑤ 보안 및 신뢰 문제
결정론적 테스트 프레임워크는 재현 가능하고 저렴한 에이전트 실행을 위해 LLM 완성을 모킹하며, 이러한 격리는 학문적 재현성과 산업적 CI 시스템 배포 모두의 선제 조건이 된다.

---

## 📚 참고 자료 및 출처

| 번호 | 자료 |
|---|---|
| 1 | **논문 원문**: Wang et al., "OpenHands: An Open Platform for AI Software Developers as Generalist Agents," *ICLR 2025*. arXiv:2407.16741. https://arxiv.org/abs/2407.16741 |
| 2 | **ICLR 2025 공식 게재**: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html |
| 3 | **HTML 전문 (v3)**: https://arxiv.org/html/2407.16741v3 |
| 4 | **OpenHands GitHub**: https://github.com/All-Hands-AI/OpenHands |
| 5 | **Semantic Scholar**: https://www.semanticscholar.org/paper/OpenHands:-An-Open-Platform-for-AI-Software-as-Wang-Li/1d07e5b6f978cf69c0186f3d5f434fa92d471e46 |
| 6 | **OpenHands 공식 문서 (Runtime Architecture)**: https://docs.openhands.dev/openhands/usage/architecture/runtime |
| 7 | **OpenHands Blog (SOTA SWE-Bench)**: https://www.openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model |
| 8 | **OpenHands SDK 논문**: "The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents," arXiv:2511.03690 |
| 9 | **SWE-bench-Live 논문**: "SWE-bench Goes Live!", arXiv:2505.23419 |
| 10 | **SWE-EVO 논문**: "SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios," arXiv:2512.18470 |
| 11 | **SWE-bench 공식 리더보드**: https://www.swebench.com |
| 12 | **OpenHands vs SWE-Agent 비교**: https://localaimaster.com/blog/openhands-vs-swe-agent |
| 13 | **DEV Community - OpenHands Deep Dive**: https://dev.to/truongpx396/openhands-deep-dive-build-your-own-guide-1al0 |
| 14 | **Emergent Mind - OpenHands Agent Framework**: https://www.emergentmind.com/topics/openhands-agent-framework |
