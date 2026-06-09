# MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

MOSS의 핵심 주장은 다음과 같습니다:

> **기존의 자기 진화(Self-Evolving) 에이전트 시스템들은 모두 텍스트 변경 가능한 아티팩트(스킬 파일, 프롬프트, 메모리 스키마, 워크플로우 그래프)에만 진화 범위를 제한**하고 있으며, 에이전트 하네스(harness) — 라우팅, 상태 관리, 디스패치, 훅 — 는 결코 수정하지 않는다. 이 물리적 한계를 극복하기 위해, **소스 레벨 적응(source-level adaptation)이 근본적으로 더 일반적인 진화 매체**임을 주장한다.

소스 레벨 적응이 우월한 네 가지 이유:

1. **튜링 완전성(Turing-completeness)**: 소스 코드 설계 공간은 모든 텍스트 변경 가능한 에이전트 설계 공간을 엄격한 부분집합으로 포함하는 보편적 탐색 공간
2. **엄격한 상위 집합(Strict superset)**: 프롬프트 편집으로 달성할 수 있는 모든 것을 코드 편집으로도 달성 가능하나 역은 성립하지 않음
3. **결정론적 효과(Deterministic effect)**: 라우팅 로직과 훅 순서는 코드로 실행되므로 기반 모델의 준수 여부에 의존하지 않음
4. **장기 컨텍스트 드리프트에 대한 저항성(Resistance to long-context drift)**: 소스 레벨 수정은 텍스트가 아닌 행동으로 인코딩되므로 시스템이 노화해도 저하되지 않음

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **소스 레벨 하네스 진화** | 기존 시스템이 도달하지 못한 에이전트 하네스 레이어까지 진화 범위 확장 |
| **프로덕션 기반 방향성 진화** | 합성 벤치마크가 아닌 실제 프로덕션 실패 증거로 앵커된 진화 |
| **결정론적 7단계 파이프라인** | Locate → Plan → Plan-Review → Implement → Code-Review → Task-Evaluate → Verdict |
| **런타임 검증** | 에페머럴(ephemeral) 트라이얼 워커를 통한 프로덕션 동등 환경에서의 검증 |
| **무중단 컨테이너 스왑** | 사용자 상태 보존과 헬스 프로브 게이트 롤백이 포함된 인플레이스 스왑 |
| **플러그형 코딩 에이전트 CLI** | Claude Code, OpenAI Codex, DeepSeek-TUI, OpenCode 지원 |
| **성능 향상** | OpenClaw에서 4-태스크 평균 grader 점수를 0.25 → 0.61로 단일 사이클에서 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 정의

배포 후 에이전트 시스템의 정적성(static nature) 문제:

$$\text{실패 모드가 반복} \Rightarrow \text{다음 인간 주도 업데이트까지 지속}$$

기존 자기 진화 시스템의 범위 한계:

$$\text{진화 범위}_{\text{기존}} \subseteq \{\text{스킬}, \text{프롬프트}, \text{메모리}, \text{워크플로우}\}$$

$$\text{진화 범위}_{\text{MOSS}} = \{\text{스킬}, \text{프롬프트}, \text{메모리}, \text{워크플로우}, \textbf{하네스}\}$$

하네스에서 발생하는 실패 유형:
- 잘못 라우팅된 메시지
- 훅 실행 순서 오류
- 세션 상태 손상
- 동시 스킬 간 원자성(atomicity) 버그

이러한 실패는 **텍스트 레이어로는 물리적으로 도달 불가능(physically unreachable)**:

$$\nexists\, \Delta_{\text{text}}: \text{fix}(\text{harness bug}) = \Delta_{\text{text}}(\text{prompt} \cup \text{skill} \cup \text{memory})$$

### 2.2 제안하는 방법

#### 방향성 진화 (Directed Evolution)

진화는 합성 벤치마크 대신 프로덕션 실패 증거 배치로 앵커됩니다:

$$\mathcal{B} = \{c_i \mid \text{score}(c_i) \in \{\text{weak}, \text{missing}\}\}$$

여기서 $c_i$는 대화 청크(chunk)이며, 배치 $\mathcal{B}$는 두 경로로 수집됩니다:

$$\mathcal{B} = \mathcal{B}_{\text{auto-scan}} \cup \mathcal{B}_{\text{user-flag}}$$

- $\mathcal{B}_{\text{auto-scan}}$: 주기적 크론 잡이 세션 JSONL을 스캔하여 수집
- $\mathcal{B}_{\text{user-flag}}$: 사용자가 불만족 표현 시 `moss evo flag` 호출로 수집

배치 밀봉 조건:

$$|\mathcal{B}| \geq \theta_{\text{batch}} \quad (\text{기본값: } \theta_{\text{batch}} = 8)$$

#### 진화 루프 (Evolution Loop)

4개 중첩 레이어 구조:

$$\underbrace{\text{Layer 0: Pre-loop Baseline}}_{\text{키포인트 행렬 고정}} \rightarrow \underbrace{\text{Layer 1: Iteration Loop}}_{\text{iter } 1 \rightarrow \cdots \rightarrow \text{iter } N}$$

$$\text{Layer 2: Stage Pipeline} = \bigcirc_1 \rightarrow \bigcirc_2 \rightarrow \bigcirc_3 \rightarrow \bigcirc_4 \rightarrow \bigcirc_5 \rightarrow \bigcirc_6 \rightarrow \bigcirc_7$$

$$\text{Layer 3: Stage-Internal Retry Rounds} = \text{round}_0 \rightarrow \text{round}_1 \rightarrow \cdots$$

루프 종료 조건 (4가지 verdict):

$$v \in \{v_{\text{CONVERGED}},\; v_{\text{NEED MORE WORK}},\; v_{\text{FUNDAMENTAL LIMIT MODEL}},\; v_{\text{FUNDAMENTAL LIMIT ARCH}}\}$$

플래토 가드(Plateau Guard) 조건:

$$\forall\, i \in \{k-w, \ldots, k\}: \Delta \text{KP}_i = 0 \Rightarrow v_k := v_{\text{CONVERGED}}$$

여기서 $w$는 depth-dependent 플래토 윈도우이고, $\text{KP}_i$는 반복 $i$에서의 키포인트 점수 행렬입니다.

#### 7단계 파이프라인 (Stage Pipeline)

각 단계를 수식으로 나타내면:

**Stage 1 (Locate)**: 실패 증거에서 진단 생성
$$d = \text{Locate}(\mathcal{T}_{\text{baseline}}, \mathcal{B}) \quad \text{(수정 제안 없이 진단만)}$$

**Stage 2 (Plan)**: 수정 계획 생성
$$p = \text{Plan}(d, \mathcal{B}) = \{f_1, f_2, \ldots, f_m\} \quad f_i: (\text{파일, 변경 사항})$$

**Stage 3 (Plan-Review)**: 계획 검토 (승인/거부)
$$\text{Plan-Review}(p) \in \{v_{\text{approve}}, v_{\text{reject-arch}}, v_{\text{reject-narrow}}\}$$

계획-검토 루프:
$$p^* = \arg\min_{r \leq R_{\text{plan}}} \text{rounds until } \text{Plan-Review}(p^r) = v_{\text{approve}}$$

**Stage 4 (Implement)**: 코드 구현 (단일 git commit)
$$\Delta_{\text{code}} = \text{Implement}(p^*)$$

**Stage 5 (Code-Review)**: 코드 검토
$$\text{Code-Review}(\Delta_{\text{code}}, p^*) \in \{v_{\text{approve}}, v_{\text{reject}}\}$$

코드-검토 루프 (거부 시 working tree hard-reset):
$$\Delta^* = \arg\min_{r \leq R_{\text{code}}} \text{rounds until } \text{Code-Review}(\Delta^r, p^*) = v_{\text{approve}}$$

**Stage 6 (Task-Evaluate)**: 4수준 정성적 척도로 키포인트 점수 산출
$$\text{KP}_{i,j} \in \{\text{strong}, \text{adequate}, \text{weak}, \text{missing}\}$$

$$M_{\text{iter}} = [\text{KP}_{i,j}]_{\text{task}_i, \text{keypoint}_j}$$

**Stage 7 (Verdict)**: 전체 반복 간 키포인트 행렬 비교로 verdict 결정
$$v = \text{Verdict}(M_{\text{iter}}, M_{\text{baseline}}, M_{\text{iter-1}}, \ldots)$$

#### 런타임 검증 (Runtime Verification)

에페머럴 트라이얼 워커 기반 검증:

$$\text{Score}_{\text{trial}} = \frac{1}{N \cdot |\mathcal{B}|} \sum_{n=1}^{N} \sum_{b \in \mathcal{B}} \text{Eval}(\text{Worker}_n(\text{CandidateImage}, b))$$

여기서 $N$은 병렬 트라이얼 워커 수입니다.

#### 인플레이스 컨테이너 스왑 (In-Place Container Swap)

헬스 프로브 조건 (90초 윈도우, 5초 간격, 4개 헬스 체크):

$$\text{commit swap} \Leftrightarrow \exists\, t_1, t_2, t_3 \text{ consecutive}: \bigwedge_{k=1}^{4} \text{HealthCheck}_k(t_i) = \text{pass}, \quad i \in \{1,2,3\}$$

롤백 조건:
$$\neg \text{commit swap} \Rightarrow \text{rollback to } \text{Image}_{\text{last-known-good}}$$

### 2.3 모델 구조

MOSS의 시스템 아키텍처는 5개 컴포넌트로 구성됩니다:

```
┌─────────────────────────────────────────────────┐
│                  Host Machine                    │
│  ┌──────────────────────┐  ┌─────────────────┐  │
│  │  moss-gateway         │  │   host-daemon   │  │
│  │  container            │←─│  asyncio RPC    │  │
│  │  (agent + evolution   │→─│  swap supervisor│  │
│  │   service + CLI)      │  │  auto-scan engine│ │
│  └──────────────────────┘  └────────┬────────┘  │
│         ↑ HTTP (api+hooks)          │ spawn      │
│         ↓ Unix socket (RPC)    ┌────┴──────────┐ │
│                                │ Coding-Agent  │ │
│                                │ CLI (per stage)│ │
│                                └───────────────┘ │
│                                ┌───────────────┐ │
│                                │ Trial Workers │ │
│                                │ ×N (ephemeral)│ │
│                                └───────────────┘ │
└─────────────────────────────────────────────────┘
```

**컴포넌트 역할 분담**:

| 컴포넌트 | 책임 |
|---------|------|
| moss-gateway container | 사용자 대면 에이전트, 인컨테이너 진화 서비스, moss CLI |
| host-daemon | 컨테이너 수명 주기 관리, 스왑 감독, 자동 스캔 엔진 |
| Coding-Agent CLI | 실제 코드 편집 수행 (4개 플러그인: Claude Code, OpenAI Codex, DeepSeek-TUI, OpenCode) |
| Trial Workers | 후보 이미지에 대한 배치 재실행 검증 |
| Control Surface CLI | `moss evo` 9개 하위 명령으로 진화 수명 주기 제어 |

**제어 흐름 (3-piece interface)**:
$$\text{CLI} \xrightarrow{\text{in}} \text{Agent} \xleftarrow{\text{webhook}} \text{MOSS}$$
$$\text{CapabilityDoc} \xrightarrow{\text{reference}} \text{Agent}$$

### 2.4 성능 향상

#### 정량적 결과 (Table 2)

| Task | Baseline | Iter 1 | $\Delta$ |
|------|----------|--------|----------|
| T141zh_sla_compliance_audit | 0.3273 | 0.5330 | +0.2057 |
| T142_sla_compliance_audit | 0.2527 | 0.5453 | +0.2926 |
| T137zh_restock_chain_check | 0.2213 | 0.4567 | +0.2354 |
| T138_restock_chain_check | 0.2090 | 0.9049 | **+0.6959** |
| **mean** | **0.2526** | **0.6100** | **+0.3574** |

$$\bar{s}_{\text{baseline}} = 0.2526, \quad \bar{s}_{\text{iter1}} = 0.6100, \quad \Delta\bar{s} = +0.3574 \; (+141.5\%)$$

#### 구체적 개선 내용

**하네스 수정 내용** (3개 파일, 177 insertions, 1 deletion):
1. 도구 결과 미디에이터의 새 어노테이션 브랜치 및 헬퍼 함수
2. before-tool-call 훅 체인에 사전 호출 거부 게이트 추가
3. 새 미디에이터 테스트 파일

$$\text{ProximalCause} = \text{HarnessChange} \neq \text{PromptChange} \neq \text{ModelChange}$$

### 2.5 한계점

논문에서 명시적 또는 암묵적으로 인정된 한계:

1. **단일 사이클 증거**: 케이스 스터디가 단 4개의 태스크, 단일 반복으로만 구성 — 통계적 유의성 미확보
2. **동일 배치 재사용**: 진화에 사용한 배치와 테스트 배치가 동일 — 과적합 위험

$$\text{Test set} = \text{Training batch} \Rightarrow \text{낙관적 편향 가능}$$

3. **단일 에이전트 시스템**: 멀티 에이전트 시스템에서의 검증 부재
4. **모델 의존성**: DeepSeek V3.2에서만 검증; 범용성 미확인
5. **회귀 테스트 부재**: 패치 후 다른 태스크에 대한 성능 저하 여부 미검증
6. **트라이얼 워커 격리**: 라이브 컨테이너의 사용자 상태나 실제 트래픽 없이 격리된 환경에서만 검증
7. **확장성 미검증**: 매우 대규모 코드베이스에서의 성능 불명확
8. **보안 고려사항**: 자기 수정 코드의 악의적 활용 가능성에 대한 논의 부재

---

## 3. 일반화 성능 향상 가능성

### 3.1 일반화를 지원하는 설계 요소

#### 기판 독립적 아키텍처 (Substrate-Agnostic Design)

MOSS의 기판 계약은 5개의 원시 기능만을 요구합니다:

$$\text{substrate} \supseteq \{\text{shell exec}, \text{fs read}, \text{periodic sched}, \text{webhook delivery}, \text{sysprompt inject}\}$$

$$\Rightarrow \text{MOSS 호스팅 가능 (코드 변경 없이)}$$

이는 OpenClaw, Hermes Agent 등 다양한 프로덕션 에이전트로의 이식 가능성을 시사합니다.

#### 플러그형 코딩 에이전트 (Pluggable Coding Agent)

4-메서드 러너 인터페이스를 통해 코딩 에이전트 공급자와 MOSS 진화 기계를 완전히 분리:

$$\text{MOSS}_{\text{evolution logic}} \perp \text{CodingAgent}_{\text{provider}}$$

새 제공자 추가 = 파일 1개 + 레지스트리 1줄

#### 튜링 완전 진화 공간

$$\mathcal{S}_{\text{source-level}} \supset \mathcal{S}_{\text{workflow}} \supset \mathcal{S}_{\text{memory}} \supset \mathcal{S}_{\text{prompt}} \supset \mathcal{S}_{\text{skill}}$$

이 포함 관계는 이론적으로 MOSS가 어떤 텍스트 변경 가능 진화 방법으로도 달성 가능한 개선을 수행할 수 있음을 보장합니다.

#### 결정론적 효과 vs. 기반 모델 준수

$$\text{효과}_{\text{text-mutable}} \propto f(\text{base model capability})$$

$$\text{효과}_{\text{source-level}} = \text{deterministic}, \quad \forall \text{base model}$$

이는 기반 모델이 교체되거나 성능이 변화해도 하네스 수정의 효과가 일정함을 의미합니다.

### 3.2 일반화의 한계

#### 작업 특정성 문제

$$\mathcal{B} \text{ (특정 배치)}에 \text{ 최적화} \Rightarrow \text{인접 태스크 회귀 위험}$$

논문의 실험 설계가 동일 배치로 훈련/테스트를 수행했기 때문에, 실제 일반화 성능은 과대평가되었을 가능성이 있습니다.

#### 코드베이스 크기와 수정 성공률

대형 코드베이스일수록 교차 파일 불변성과 동시성 상호작용이 복잡해집니다:

$$\text{수정 성공률} \downarrow \text{ as } |\text{codebase}| \uparrow$$

#### 언어 및 프레임워크 의존성

소스 레벨 수정은 코드베이스의 언어, 프레임워크, 아키텍처 패턴에 따라 코딩 에이전트의 역량이 달라집니다.

### 3.3 일반화 성능 향상을 위한 방향

논문에서 직접적으로 제시하진 않았으나, 다음과 같은 방향이 추론 가능합니다:

**다중 배치 앙상블 진화**:

```math
\mathcal{B}^* = \bigcup_{t=1}^{T} \mathcal{B}_t, \quad v^* = \text{Verdict}\left(\bigcap_{b \in \mathcal{B}^*} M_b\right)
```

다양한 실패 시나리오를 커버하는 배치를 사용하면 더 일반적인 수정으로 이어질 가능성이 높습니다.

**회귀 방지 키포인트 세트**:

기존 성공 케이스에 대한 회귀 키포인트를 Task-Evaluate에 포함:

$$M_{\text{extended}} = M_{\text{failure batch}} \cup M_{\text{regression guard}}$$

**다중 기판 크로스 검증**:

동일한 하네스 수정이 다른 기판에도 적용 가능한지 검증하는 메커니즘.

---

## 4. 미래 연구에 미치는 영향과 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### 패러다임 전환 (Paradigm Shift)

MOSS는 에이전트 시스템 진화 연구에 있어 중요한 패러다임 전환을 제시합니다:

$$\text{기존}: \quad \text{진화} \subseteq \mathcal{L}_{\text{text}}$$

$$\text{MOSS}: \quad \text{진화} \subseteq \mathcal{L}_{\text{code}} \supset \mathcal{L}_{\text{text}}$$

이는 향후 자기 진화 에이전트 연구의 기준선을 소스 레벨까지 끌어올립니다.

#### 프로덕션 환경의 AI 시스템 자율성 연구 활성화

MOSS는 실제 배포 환경에서의 에이전트 자율성 연구를 위한 실용적 토대를 제공합니다:
- **연속적 자율 개선 루프** 설계 패턴 확립
- **실패 증거 기반 진화** 방법론 표준화 가능성
- **인간-에이전트 협력적 진화** (user-consent gate) 프레임워크

#### 안전성 및 거버넌스 연구의 필요성

소스 레벨 자기 수정은 새로운 안전 문제를 제기합니다:
- 의도치 않은 기능 도입 가능성
- 악의적 활용 시나리오 (adversarial self-modification)
- 감사 가능성(auditability)과 설명 가능성

### 4.2 앞으로 연구 시 고려할 점

#### (1) 일반화 검증 강화

현재 논문의 실험은 동일 배치로 훈련/테스트를 수행하여 일반화 성능을 과대평가할 위험이 있습니다:

$$\text{권고}: \quad \mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{test}} = \emptyset$$

분리된 홀드아웃 배치, 다양한 태스크 도메인, 다국어 설정에서의 검증이 필요합니다.

#### (2) 다중 반복 실험

단일 사이클 결과만으로는 신뢰성이 부족합니다. 다중 반복 실험과 통계적 유의성 검증이 필요합니다:

$$\bar{s} \pm \sigma, \quad p < 0.05, \quad n_{\text{trials}} \geq 30$$

#### (3) 회귀 추적 시스템

패치 후 다른 태스크에 대한 성능 저하를 체계적으로 추적하는 메커니즘이 필요합니다:

$$\text{NetGain} = \Delta s_{\text{target}} - \sum_{j \notin \mathcal{B}} \max(0, -\Delta s_j)$$

#### (4) 보안 및 안전성 프레임워크

소스 레벨 자기 수정에 대한 안전 경계를 명확히 정의해야 합니다:
- **코드 수정 범위 제한**: 특정 모듈이나 파일에만 접근 허용
- **수정 diff 감사**: 모든 변경사항의 자동 보안 스캔
- **롤백 SLA**: 실패 감지 후 롤백까지의 최대 허용 시간

#### (5) 다중 에이전트 환경 확장

현재 MOSS는 단일 인스턴스 시스템으로 설계되었으나, 실제 엔터프라이즈 환경에서는 다중 에이전트 협력이 필요합니다:

$$\text{Multi-MOSS}: \quad \mathcal{E}_{\text{shared}} = \bigcup_{i=1}^{K} \mathcal{E}_i, \quad \text{consensus}(v_1, \ldots, v_K)$$

#### (6) 메타 학습과의 통합

MOSS의 진화 파이프라인 자체를 학습 가능하게 만들면 더욱 효율적인 진화가 가능합니다:

$$\theta^*_{\text{pipeline}} = \arg\max_\theta \mathbb{E}_{\mathcal{B}}\left[\text{Score}(\text{MOSS}_\theta(\mathcal{B}))\right]$$

#### (7) 비용-효용 분석 필요

소스 레벨 진화는 컴퓨팅 비용이 상당합니다. 진화 트리거 조건과 비용 간의 트레이드오프를 최적화해야 합니다:

$$\text{EvolutionDecision} = \begin{cases} \text{trigger} & \text{if } \mathbb{E}[\Delta s] \cdot v_{\text{improvement}} > C_{\text{evolution}} \\ \text{skip} & \text{otherwise} \end{cases}$$

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 진화 범위 | 환경 | 피드백 신호 | MOSS와의 차이 |
|------|------|-----------|------|-------------|--------------|
| **EvoAgentX** (Wang et al.) | 2025 | 스킬, 워크플로우, 프롬프트 | 연구 환경 | 벤치마크 점수 | 하네스 미수정, 프로덕션 미검증 |
| **SICA** (Robeyns et al.) | 2025 | 소스 코드 | 최소 스캐폴드 | SWE-Bench | 프로덕션 기판 미적용, 실패 증거 미사용 |
| **Darwin Gödel Machine** (Zhang et al.) | 2025 | 소스 코드 | 최소 스캐폴드 | 벤치마크 + 오픈엔드 탐색 | 탐색적 패러다임, 프로덕션 미적용 |
| **Hermes Agent** (Nous Research) | 2024/2026 | 스킬, 메모리 | 프로덕션 | DSPy+GEPA 최적화 | 하네스 미수정 |
| **SkillClaw** (Ma et al.) | 2026 | 스킬 | 프로덕션 | 스킬 성능 | 텍스트 레이어만, 하네스 미수정 |
| **GenericAgent** (Liang et al.) | 2026 | 스킬, 메모리 | 프로덕션 | 컨텍스트 밀도 | 텍스트 레이어만 |
| **HyperAgents** (Zhang et al.) | 2026 | 소스 코드 + 메타 절차 | 최소 스캐폴드 | 벤치마크 | 프로덕션 미적용 |
| **Meta-Harness** (Lee et al.) | 2026 | 하네스 최적화 | 연구 환경 | 실행 트레이스 + 벤치마크 | 프로덕션 컨테이너 스왑 미지원 |
| **Capability Evolver** (Wang et al.) | 2026 | 행동 유전자, 메모리 캡슐 | 프로덕션 | 경험 기반 | 텍스트 레이어만 |
| **MOSS** (본 논문) | 2026 | **스킬+프롬프트+메모리+하네스** | **프로덕션** | **프로덕션 실패 증거** | 유일하게 하네스 포함 |

### 핵심 차별화 요약

$$\text{MOSS} = \underbrace{\text{소스 레벨 진화}}_{\text{SICA, DGM 계열}} + \underbrace{\text{프로덕션 기판}}_{\text{Hermes, SkillClaw 계열}} + \underbrace{\text{실패 증거 앵커링}}_{\text{Meta-Harness 영향}}$$

ReAct (Yao et al., 2022), Toolformer (Schick et al., 2023), ToolLLM (Qin et al., 2024), MetaGPT (Hong et al., 2024), AutoGen (Wu et al., 2024), DSPy (Khattab et al., 2023) 등 기반 연구들은 MOSS의 에이전트 인프라 토대를 형성합니다.

---

## 참고 자료

**주요 논문 (본 분석의 직접 근거)**

- Cai, Q., Zhang, Y., Jia, X., Zheng, H., Xue, W., Song, J., Tian, X., & Guo, Y. (2026). **MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems**. arXiv:2605.22794v2

**논문 내 인용 문헌 (분석에 활용)**

- Robeyns, M., Szummer, M., & Aitchison, L. (2025). A self-improving coding agent. arXiv:2504.15228
- Zhang, J., Hu, S., Lu, C., Lange, R., & Clune, J. (2025). Darwin Godel Machine: Open-ended evolution of self-improving agents. arXiv:2505.22954
- Zhang, J., Zhao, B., Yang, W., Foerster, J., Clune, J., Jiang, M., Devlin, S., & Shavrina, T. (2026). HyperAgents. arXiv:2603.19461
- Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. (2026). Meta-Harness: End-to-end optimization of model harnesses. arXiv:2603.28052
- Wang, Y., Liu, S., Fang, J., & Meng, Z. (2025). EvoAgentX: An automated framework for evolving agentic workflows. EMNLP 2025 System Demonstrations, pp. 643–655
- Nous Research. (2024). Hermes Agent. GitHub: https://github.com/NousResearch/hermes-agent
- Nous Research. (2026). Hermes Agent Self-Evolution. GitHub: https://github.com/NousResearch/hermes-agent-self-evolution
- Ma, Z., Yang, S., Ji, Y., Wang, X., Wang, Y., Hu, Y., Huang, T., & Chu, X. (2026). SkillClaw: Let skills evolve collectively with agentic evolver. arXiv:2604.08377
- Liang, J., Han, J., Li, W., et al. (2026). GenericAgent: A token-efficient self-evolving LLM agent via contextual information density maximization. arXiv:2604.17091
- Wang, J., Ren, Y., & Zhang, H. (2026). From procedural skills to strategy genes: Towards experience-driven test-time evolution. arXiv:2604.15097
- Hu, S., Lu, C., & Clune, J. (2025). Automated design of agentic systems. ICLR 2025, pp. 21344–21377
- Khattab, O., et al. (2023). DSPy: Compiling declarative language model calls into self-improving pipelines. arXiv:2310.03714
- Agrawal, L.A., et al. (2025). GEPA: Reflective prompt evolution can outperform reinforcement learning. arXiv:2507.19457
- Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing reasoning and acting in language models. arXiv:2210.03629
- Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS 36, pp. 68539–68551
- Qin, Y., et al. (2024). ToolLLM: Facilitating large language models to master 16000+ real-world APIs. ICLR 2024, pp. 9695–9717
- Hong, S., et al. (2024). MetaGPT: Meta programming for a multi-agent collaborative framework. ICLR 2024, pp. 23247–23275
- Wu, Q., et al. (2024). AutoGen: Enabling next-gen LLM applications via multi-agent conversations. COLM 2024
- Li, G., et al. (2023). CAMEL: Communicative agents for "mind" exploration of large language model society. NeurIPS 36, pp. 51991–52008
- Qian, C., et al. (2024). ChatDev: Communicative agents for software development. ACL 2024, pp. 15174–15186
- Wang, L., et al. (2024). A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345
- Guo, T., et al. (2024). Large language model based multi-agents: A survey of progress and challenges. arXiv:2402.01680
- Liu, A., et al. (2025). DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv:2512.02556
- Achiam, J., et al. (2023). GPT-4 technical report. arXiv:2303.08774
- Ye, B., Li, R., Yang, Q., et al. (2026). Claw-eval: Toward trustworthy evaluation of autonomous agents. arXiv:2604.06132
- OpenClaw. (2024). OpenClaw — Personal AI Assistant. GitHub: https://github.com/openclaw/openclaw

> **⚠️ 정확도 고지**: 본 분석은 제공된 PDF 원문(arXiv:2605.22794v2)에 직접 근거하여 작성되었습니다. 논문이 2026년 5월 23일 제출된 preprint임을 감안하여, 일부 인용 문헌(2026년 날짜 포함)의 독립적 검증이 어려울 수 있습니다. 수식은 논문의 서술적 내용을 수학적으로 형식화한 것이며, 일부는 저자의 의도를 바탕으로 재구성되었음을 밝힙니다.
