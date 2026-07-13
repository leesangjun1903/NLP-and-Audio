# Governance Gaps in Agent Interoperability Protocols: What MCP, A2A, and ACP Cannot Express

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **"에이전트 상호운용성 프로토콜(MCP, A2A, ACP, ANP, ERC-8004)은 에이전트 간 조율(coordination)은 가능하게 하지만, 에이전트 커뮤니티의 집합적 거버넌스(governance)를 위한 프로토콜 수준의 기본 요소(primitives)를 전혀 인코딩하지 않는다."**

즉, 현재의 에이전트 상호운용성 표준은 **작업 조율(task coordination) 계층**은 갖추었지만, **거버넌스 계층(governance layer)** 은 구조적으로 누락된 상태입니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **1. 거버넌스 요구사항 분류 체계(Taxonomy)** | 조직이론·MAS 연구·기업 거버넌스 표준에서 도출한 6차원 분류 체계(G1~G6) 제시 |
| **2. 체계적 갭 매트릭스(Gap Matrix)** | 5개 프로토콜 × 6개 차원을 Supported / Partial / Absent로 분류 |
| **3. 확장성·시급성 평가** | 기존 확장 메커니즘으로 해결 가능한 갭 vs. 새로운 아키텍처 계층이 필요한 구조적 갭을 구분 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

기업 환경에서 이종(heterogeneous) 에이전트 군집이 집합적 결정을 내려야 하는 시나리오(예: 은행의 자율 코딩 에이전트가 프로덕션 시스템 수정을 승인해야 하는 상황)에서:

$$\text{핵심 질문: "어떤 에이전트가 이 작업을 수행할 수 있는가?"} \neq \text{"에이전트들이 집합적으로 무엇을 결정해야 하는가?"}$$

현재 프로토콜들은 전자(조율)만 답할 수 있고, 후자(거버넌스)는 전혀 표현하지 못한다는 것이 핵심 문제입니다.

### 2-2. 제안하는 방법: 6차원 거버넌스 요구사항 분류 체계

논문은 새로운 ML 모델이 아닌 **분석적 분류 체계(taxonomy)**와 **갭 매트릭스**를 제안합니다.

#### 거버넌스 6차원 (G1~G6)

$$\mathcal{G} = \{G_1, G_2, G_3, G_4, G_5, G_6\}$$

각 차원의 정의:

| 차원 | 이름 | 정의 |
|------|------|------|
| $G_1$ | Membership (멤버십) | 커뮤니티 참여자의 입장·초대·제거·역할 할당을 프로토콜이 인코딩 |
| $G_2$ | Deliberation (심의) | 순서·도전·응답 의미론을 포함한 구조화된 논증 교환을 인코딩 |
| $G_3$ | Voting (투표) | 정족수·라운드·입장 결의를 포함한 선호 집계를 인코딩 |
| $G_4$ | Dissent Preservation (반대 보존) | 결정 산출물에서 소수 의견이 묵살되지 않도록 보장 |
| $G_5$ | Human Escalation (인간 에스컬레이션) | 결정을 인간 권한으로 라우팅하는 조건과 메커니즘을 정의 |
| $G_6$ | Audit/Replay (감사/재현) | 의사결정 과정의 결정론적 재구성을 가능하게 하는 변조 방지 이벤트 로그 생성 |

#### 갭 분류 기준

각 (프로토콜, 차원) 쌍에 대해:

$$\text{Classification}(P_i, G_j) \in \{\text{Supported}, \text{Partial}, \text{Absent}\}$$

점수화:

$$\text{Score}(P_i, G_j) = \begin{cases} 2 & \text{if Supported} \\ 1 & \text{if Partial} \\ 0 & \text{if Absent} \end{cases}$$

$$\text{Total Coverage}(P_i) = \sum_{j=1}^{6} \text{Score}(P_i, G_j), \quad \text{Max} = 12$$

#### 분류 체계의 이론적 근거

| 이론 출처 | 기여 |
|-----------|------|
| Habermas의 소통적 합리성 | 구조화된 논증, 합의 형성 → $G_2$ |
| Robert's Rules of Order (의사 규칙) | 정족수, 토론, 투표, 기록된 반대 → $G_1, G_3, G_4$ |
| Ostrom의 제도 분석 프레임워크 | 경계 규칙(멤버십), 선택 규칙(의사결정) → $G_1, G_3$ |
| SR 11-7, ISO/IEC 42001, EU AI Act | 감사 가능성, 인간 감독 → $G_5, G_6$ |

### 2-3. 모델 구조: 에이전트 상호운용성 프로토콜 스택

논문이 제안하는 계층 구조:

```
┌─────────────────────────────────────────────────┐
│   Layer 4: Agent-to-Community (GOVERNANCE)      │ ← 누락된 계층
│   G1 G2 G3 G4 G5 G6 (모두 Absent)              │ ← 이 논문의 기여
├─────────────────────────────────────────────────┤
│   Layer 3: Trust & Reputation                   │
│   ERC-8004 (Identity + Reputation + Validation) │
├─────────────────────────────────────────────────┤
│   Layer 2: Agent-to-Agent                       │
│   A2A v1.0.1 | ACP | ANP                        │
├─────────────────────────────────────────────────┤
│   Layer 1: Agent-to-Tool                        │
│   MCP v1.1 (Tools + Resources + Prompts)        │
└─────────────────────────────────────────────────┘
```

### 2-4. 갭 매트릭스 결과

| 프로토콜 | $G_1$ | $G_2$ | $G_3$ | $G_4$ | $G_5$ | $G_6$ | 커버리지 |
|---------|-------|-------|-------|-------|-------|-------|---------|
| MCP v1.1 | Absent | Absent | Absent | Absent | Absent | **Partial** | 1/12 |
| A2A v1.0.1 | **Partial** | Absent | Absent | Absent | Absent | Absent | 1/12 |
| ACP | **Partial** | **Partial** | Absent | Absent | Absent | Absent | 2/12 |
| ANP | Absent | Absent | Absent | Absent | Absent | Absent | 0/12 |
| ERC-8004 | **Partial** | Absent | Absent | Absent | Absent | **Partial** | 2/12 |
| **합계** | Partial | Partial | **Absent** | **Absent** | **Absent** | Partial | — |

$$\text{최대 가능 점수} = 6 \times 2 = 12, \quad \text{실제 최고점} = \frac{2}{12} \approx 16.7\%$$

$G_3$ (투표), $G_4$ (반대 보존), $G_5$ (인간 에스컬레이션)은 **5개 프로토콜 전체에서 완전히 Absent**.

### 2-5. 확장 가능한 갭 vs. 구조적 갭

| 프로토콜 | 갭 유형 | 이유 |
|---------|---------|------|
| A2A | **확장 가능** | 확장 메커니즘이 "새로운 데이터, RPC 메서드, 상태 기계" 지원 |
| MCP | **구조적으로 어색** | 클라이언트-서버 아키텍처는 커뮤니티 거버넌스에 부적합 |
| ERC-8004 | **범위 제한** | 온체인 아키텍처의 지연·비용 문제로 실시간 심의 불가 |

### 2-6. 한계점

1. **시간적 한계**: 2026년 6월 기준 명세서 평가 (프로토콜 빠르게 진화)
2. **분류 주관성**: "Partial" 분류에 판단이 개입
3. **문화적 편향**: 서구 조직 이론(Habermas, Robert's Rules) 중심
4. **명세서 vs. 구현**: 프로토콜이 인코딩하는 것만 평가, 그 위에 구축 가능한 것은 평가 제외

---

## 3. 모델의 일반화 성능 향상 가능성

> ⚠️ **중요한 전제**: 이 논문은 ML 모델을 제안하는 실증 연구가 아니라 **프로토콜 분석 및 분류 체계 연구**입니다. 따라서 전통적 의미의 "모델 일반화 성능"은 직접 해당되지 않습니다. 대신 **분석 프레임워크(taxonomy)의 일반화 가능성**과 **거버넌스 갭이 멀티에이전트 시스템의 일반화를 저해하는 구조적 문제**로 해석하여 분석합니다.

### 3-1. 거버넌스 분류 체계(Taxonomy)의 일반화 가능성

논문이 제안하는 6차원 분류 체계 $\mathcal{G} = \{G_1, ..., G_6\}$는 다음 조건에서 일반화됩니다:

$$\text{Coverage}(\mathcal{G}, P_{\text{new}}) = \frac{1}{6}\sum_{j=1}^{6} \mathbf{1}[\text{Score}(P_{\text{new}}, G_j) > 0]$$

**일반화가 가능한 이유:**
- G1~G6는 특정 기술 스택이 아닌 **조직이론·의회 규칙·제도 경제학**에서 도출
- 새로운 에이전트 프로토콜(예: ANP의 후속, 미래의 IEEE 표준)에도 동일 체계 적용 가능
- 엔터프라이즈 AI 거버넌스 표준(SR 11-7, EU AI Act)과 정렬되어 규제 요건 변화에도 견고함

**일반화의 한계:**
- 비서구권 거버넌스 모델(예: 유교적 합의 방식, 분권화된 전통적 의사결정 구조)은 반영되지 않음
- 디지털 조직이 아닌 물리적 사이버-물리 시스템(CPS) 에이전트에 대한 적용 여부 불명확

### 3-2. 거버넌스 프리미티브 부재가 MAS 일반화에 미치는 영향

논문이 인용하는 Bracale Syrnikov et al. [13]의 연구는 거버넌스 메커니즘의 효과를 정량적으로 보여줍니다:

$$\text{LLM 담합(Collusion) 발생률}: 50\% \xrightarrow{\text{거버넌스 그래프 적용}} 5.6\%$$

이는 프로토콜 수준의 거버넌스 프리미티브가 없을 경우:

$$P(\text{비의도적 집합 행동 오류}) \propto \frac{1}{|\text{거버넌스 제약}|}$$

즉, 거버넌스 제약이 없을수록 에이전트 커뮤니티의 집합적 행동이 **비결정론적·비일반화적**이 됩니다.

### 3-3. 프로토콜 표준화가 일반화 성능에 기여하는 구조

현재 상황의 문제점:

$$\text{각 애플리케이션이 거버넌스를 재구현} \Rightarrow \text{비호환적 사일로(silos) 형성}$$

논문이 제안하는 해결 방향:

$$\text{프로토콜 네이티브 거버넌스} \Rightarrow \text{상호운용 가능한 거버넌스 도구} + \text{표준 감사 포맷} + \text{조합 가능한 거버넌스 규칙}$$

이를 형식화하면, 프로토콜 $P$에 거버넌스 계층 $\mathcal{L}_G$를 추가할 때:

$$\text{Interoperability}(P + \mathcal{L}_G) > \sum_{i} \text{Interoperability}(\text{App}_i^{\text{custom governance}})$$

즉, 거버넌스의 **프로토콜 네이티브화**가 멀티에이전트 시스템 전체의 일반화 성능(다양한 도메인·조직·규제 환경에서의 재사용성)을 높입니다.

### 3-4. 향후 거버넌스 프리미티브가 MAS 일반화를 향상시킬 수 있는 방향

| 거버넌스 차원 | 일반화 기여 메커니즘 |
|-------------|-------------------|
| $G_1$ (멤버십) | 도메인 특화 에이전트 풀 구성 → 전문화된 커뮤니티 형성 |
| $G_2$ (심의) | 증거 기반 주장 교환 → 소수 에이전트의 도메인 지식 활용 가능 |
| $G_3$ (투표) | 집합적 선호 집계 → 단일 에이전트 편향 완화 |
| $G_4$ (반대 보존) | 소수 의견 보존 → 데이터 분포 쏠림에 대한 견제 |
| $G_5$ (인간 에스컬레이션) | 불확실한 상황에서 인간 감독 → 분포 이탈(out-of-distribution) 케이스 처리 |
| $G_6$ (감사/재현) | 결정 과정 추적 가능 → 실패 원인 분석 및 개선 루프 형성 |

특히 $G_4$ (반대 보존)와 $G_5$ (인간 에스컬레이션)는 **분포 외 입력(OOD input)** 에 대한 에이전트 시스템의 강건성과 직결됩니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4-1. 향후 연구에 미치는 영향

#### (1) 거버넌스 프로토콜 설계 연구의 촉발

이 논문은 "거버넌스 계층(Layer 4)"이라는 새로운 연구 영역을 명시적으로 정의합니다. 구체적으로:

- **A2A 확장 기반 거버넌스 프리미티브 설계**: A2A의 확장 메커니즘을 활용한 G1~G6 인코딩
- **MCP 거버넌스 세션 프로토콜**: 클라이언트-서버 아키텍처를 넘어선 커뮤니티 참여 패턴 연구
- **ERC-8004 기반 온체인 거버넌스 최적화**: 지연·비용 문제를 해결하는 레이어2 솔루션

$$\text{미래 프로토콜 설계 요구사항: } P_{\text{future}} \supseteq \mathcal{G} = \{G_1, G_2, G_3, G_4, G_5, G_6\}$$

#### (2) 멀티에이전트 시스템 연구 방향 전환

| 기존 연구 방향 | 이 논문 이후 필요한 방향 |
|-------------|----------------------|
| 에이전트 간 태스크 위임 최적화 | 에이전트 커뮤니티의 집합적 의사결정 메커니즘 |
| 단일 에이전트 성능 향상 | 거버넌스 제약 하에서의 집합 성능 |
| 비공식 합의 방법론 | 프로토콜 네이티브 투표·반대 보존 메커니즘 |
| 임시 감사(ad hoc audit) | 결정론적 재현 가능한 감사 로그 표준 |

#### (3) 기업 AI 거버넌스 실무에의 영향

- **AWS Bedrock AgentCore** 등 프로덕션 에이전트 인프라의 거버넌스 갭 가시화
- 규제 기관(EU AI Act, SR 11-7)이 프로토콜 수준 거버넌스 요구사항을 명시할 근거 제공
- 기업 AI 감사 표준화 논의의 기술적 기반 마련

#### (4) 관련 분야와의 교차 연구 촉진

- **형식 검증(Formal Verification)**: 거버넌스 프리미티브의 정확성 증명
- **게임이론**: 투표 메커니즘 설계(Mechanism Design) 관점의 G3 구현
- **법학·규제과학**: G5(인간 에스컬레이션) 트리거 조건의 법적 해석

### 4-2. 향후 연구 시 고려할 점

#### (1) 표준화 타이밍 문제

논문이 강조하는 **시급성**:

$$\text{표준 제안 기회 창 (Window of Opportunity)} \approx 6 \sim 12 \text{ 개월}$$

A2A 공개 후 6개월이 지났지만 거버넌스 확장이 제안되지 않은 상황에서, **드 팩토 표준(de facto standards)**이 임시 구현으로 먼저 형성될 위험이 있습니다. 연구자들은:

- 프로토콜 워킹그룹(Linux Foundation A2A WG 등)에 직접 기여
- 프로토타입 구현을 통한 실현 가능성 검증 병행

#### (2) 분류 기준의 문화적 확장

$$\mathcal{G}_{\text{Western}} = \{G_1, ..., G_6\} \rightarrow \mathcal{G}_{\text{Universal}} = \mathcal{G}_{\text{Western}} \cup \mathcal{G}_{\text{Non-Western}}$$

비서구적 거버넌스 전통(예: 사회적 합의 방식, 위계적 의사결정)을 반영한 확장 분류 체계 연구가 필요합니다.

#### (3) 성능 vs. 거버넌스 트레이드오프 정량화

프로토콜에 거버넌스 계층을 추가할 때:

$$\text{Latency Overhead}(\mathcal{L}_G) = f(|G_2|, |G_3|, \text{Byzantine Fault Tolerance})$$

특히 실시간 에이전트 시스템에서 $G_2$ (심의)와 $G_3$ (투표)의 라운드 수가 지연 시간에 미치는 영향을 실증적으로 측정해야 합니다.

#### (4) 보안 및 적대적 환경 고려

$$P(\text{거버넌스 조작}) = h(\text{투표 조작 가능성}, \text{멤버십 스푸핑}, \text{로그 변조})$$

거버넌스 프리미티브 자체가 공격 벡터가 될 수 있으므로:
- $G_3$ (투표): 시빌 공격(Sybil Attack) 방어
- $G_6$ (감사): 로그 변조 방지 (HMAC 체인, ZK 증명 활용)
- $G_4$ (반대 보존): 소수 의견 위조 방지

#### (5) 형식 명세(Formal Specification) 필요성

현재 논문은 자연어 기반 분류이므로, 후속 연구에서는:

$$\forall P \in \mathcal{P}, \forall G_j \in \mathcal{G}: \text{Score}(P, G_j) \text{를 형식적으로 검증 가능한 명세로 표현}$$

TLA+, Alloy, 또는 CSP와 같은 형식 명세 언어를 활용한 거버넌스 프리미티브 검증이 요구됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 내용 | 본 논문과의 관계 |
|------|---------|----------------|
| **Bracale Syrnikov et al. (2026)** *"Institutional AI: Governing LLM Collusion via Public Governance Graphs"* arXiv:2601.11369 | 거버넌스 그래프로 LLM 담합 50% → 5.6% 감소 | 거버넌스의 **필요성** 검증; 본 논문은 **어디에 없는지** 식별 |
| **Ruan (2026)** *"Governance by Design: A Parsonian Institutional Architecture for Internet-wide Agent Societies"* arXiv:2604.11337 | Parsons AGIL 프레임워크로 에이전트 사회 16셀 제도 아키텍처 도출 | **사회학적** 분석; 본 논문은 **프로토콜 수준** 구체적 갭 매트릭스 제공 |
| **Wang et al. (2026)** *"From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation"* arXiv:2604.07667 | 사후(post-hoc) 심의 결정을 위한 순응적 사회적 선택 이론 | $G_2$, $G_3$ 구현 메커니즘 제안; 본 논문은 이런 메커니즘이 프로토콜에 없음을 규명 |
| **De Curto et al. (2026)** *"LLM Constitutional Multi-Agent Governance"* arXiv:2603.13189 | 제약 필터링과 벌칙 효용 최적화를 이용한 헌법적 MAS 거버넌스 (CMAG) 제안 | 거버넌스 **메커니즘** 설계; 본 논문은 이를 수용할 **프로토콜 계층**의 부재를 지적 |
| **Hu & Rong (2025)** *"A Protocol Trust Taxonomy for Agent Interoperability"* arXiv:2511.03434 | A2A, ANP, ERC-8004에 대한 6가지 신뢰 모델 분류 | 신뢰·평판에 초점; 본 논문은 신뢰 이상의 **거버넌스 차원** 전체를 분석 |
| **Ehtesham et al. (2025)** *"A Survey of Agent Communication Protocols"* arXiv:2505.02279 | 에이전트 통신 프로토콜 서술적 비교 | 거버넌스 특화 분석 없음; 본 논문이 이 공백을 채움 |
| **Gupta et al. (2025)** *"The Role of Social Learning and Collective Norm Formation in LLM Multi-Agent Systems"* AAMAS 2025 | Ostrom CPR 원칙을 LLM 에이전트에 적용, 사회적 학습으로 규범 형성 | $G_1$(멤버십), $G_2$(심의) 관련 메커니즘; 프로토콜 인코딩 부재 문제는 미해결 |
| **AgentRFC Authors (2026)** *"AgentRFC: A Reference Framework for Agent Interoperability"* arXiv:2603.23801 | 에이전트 상호운용성을 위한 6계층 참조 스택 제안 | **계층 식별**에 초점; 본 논문은 각 계층 내 **갭**을 분석 |

### 비교 분석 종합

$$\text{기존 연구} = \begin{cases} \text{거버넌스 메커니즘 설계} \\ \text{프로토콜 서술적 비교} \\ \text{사회학적 아키텍처 이론} \end{cases}$$

$$\text{본 논문의 차별점} = \text{명세 수준(specification-level)에서 프로토콜 × 거버넌스 갭을 정량적으로 매핑}$$

본 논문은 기존 연구들이 **"어떻게 거버넌스를 구현할 것인가"** 에 집중하는 동안, **"현재 표준에 거버넌스 인프라가 존재하는가"** 라는 선행 질문에 답함으로써 독창적 기여를 합니다.

---

## 참고자료 (논문 내 인용 문헌 기준)

1. **Kang, R. & Diponegoro, Y. (2026).** "Governance Gaps in Agent Interoperability Protocols: What MCP, A2A, and ACP Cannot Express." arXiv:2606.31498v1 [cs.MA]. *(분석 대상 논문)*
2. Anthropic. "Model Context Protocol Specification." (2024, schema 2025-11-25). https://modelcontextprotocol.io/specification
3. Google & Linux Foundation. "Agent2Agent Protocol." v1.0.1 (May 2026). https://a2a-protocol.org/
4. IBM Research. "Agent Communication Protocol." (2025). https://github.com/ibm/agent-communication-protocol
5. ANP Community. "Agent Network Protocol." (2025). https://github.com/agent-network-protocol/
6. Ethereum Community. "ERC-8004: Trustless Agents." (Draft, 2025-08-13). https://eips.ethereum.org/EIPS/eip-8004
7. Amazon Web Services. "Amazon Bedrock AgentCore." (2026). https://aws.amazon.com/bedrock/agentcore/
8. Linux Foundation. "A2A Protocol Extensions." (2026). https://a2a-protocol.org/latest/topics/extensions/
9. Habermas, J. *Between Facts and Norms.* MIT Press, 1996.
10. Robert, H.M. et al. *Robert's Rules of Order Newly Revised*, 12th ed. PublicAffairs, 2020.
11. Ostrom, E. *Governing the Commons.* Cambridge University Press, 1990.
12. Sierra, C. et al. "A Framework for Argumentation-Based Negotiation." *Argumentation in Multi-Agent Systems.* Springer, 2004.
13. Ruan, Y. "Governance by Design: A Parsonian Institutional Architecture for Internet-wide Agent Societies." arXiv:2604.11337 (2026).
14. Bracale Syrnikov, A. et al. "Institutional AI: Governing LLM Collusion via Public Governance Graphs." arXiv:2601.11369 (2026).
15. OCC. "Supervisory Guidance on Model Risk Management (SR 11-7)." OCC Bulletin 2011-12.
16. ISO/IEC 42001:2023 Artificial Intelligence Management System.
17. European Parliament and Council. "Regulation (EU) 2024/1689 (Artificial Intelligence Act)." 2024.
18. Hu, Y. & Rong, W. "A Protocol Trust Taxonomy for Agent Interoperability." arXiv:2511.03434 (2025).
19. Ehtesham, A. et al. "A Survey of Agent Communication Protocols." arXiv:2505.02279 (2025).
20. AgentRFC Authors. "AgentRFC: A Reference Framework for Agent Interoperability." arXiv:2603.23801 (2026).
21. De Curto, J. et al. "LLM Constitutional Multi-Agent Governance." arXiv:2603.13189 (2026).
22. Wang, X. et al. "From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation." arXiv:2604.07667 (2026).
23. Gupta, S. et al. "The Role of Social Learning and Collective Norm Formation in LLM Multi-Agent Systems." *AAMAS 2025* (2025).
