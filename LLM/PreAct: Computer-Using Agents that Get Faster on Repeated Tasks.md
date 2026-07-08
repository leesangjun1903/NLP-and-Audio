# PreAct: Computer-Using Agents that Get Faster on Repeated Tasks 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**PreAct**는 컴퓨터 사용 에이전트(Computer-Using Agent, CUA)가 **반복 작업을 수행할 때 매번 처음부터 재추론(re-derive)하는 비효율성**을 해결하고자 한다. 논문의 핵심 주장은 다음과 같다:

> *"에이전트는 이미 성공한 작업을 다시 파생할 필요가 없다. 성공한 실행을 상태 기계(state machine) 프로그램으로 컴파일하고, 이후 실행 시 직접 재생(replay)함으로써 언어 모델 호출 없이 8.5–13× 빠르게 실행할 수 있다."*

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **상태 기계 기반 메모리 표현** | 성공한 에이전트 실행을 검증 가능한 상태 기계 프로그램으로 컴파일 |
| **Verify-Before-Store Gate** | 컴파일된 프로그램을 저장 전 독립적으로 재검증하여 "실행은 되나 목표 미달성" 프로그램 차단 |
| **8.5–13× 속도 향상** | 웜(warm) 재생 시 언어 모델 호출 없이 결정론적 실행 |
| **Verified Compile-Extend-Replace Loop** | append-only가 아닌 mutable corpus로 자기 확장/정제 |
| **부정적 발견의 엄밀한 보고** | 프롬프트 내용, 런타임 가드레일, 셀렉터 선택 등이 결과에 영향 없음을 명시 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

현재 CUA 시스템은 **ReAct 패러다임**을 따라 매 단계 observe–reason–act 루프를 실행한다. 이는 다음 비효율을 낳는다:

$$\text{Cost}_{\text{CUA}}(n) = n \cdot C_{\text{solve}}$$

여기서 $n$은 반복 횟수, $C_{\text{solve}}$는 한 번 풀 때의 LLM 비용이다. 즉 **동일 작업을 $n$번 반복해도 비용은 선형 증가**한다.

PreAct는 이를 다음으로 교체한다:

$$\text{Cost}_{\text{PreAct}}(n) = C_{\text{solve}} + C_{\text{compile+verify}} + (n-1) \cdot \epsilon_{\text{replay}}$$

여기서 $\epsilon_{\text{replay}} \approx 0$ (LLM 호출 없음)이고, $C_{\text{compile+verify}} \approx 1.62 \sim 2.17 \cdot C_{\text{solve}}$ (Android: +162%, OSWorld: +217%)이다. 즉 **$n \geq 3$부터 PreAct가 우위**에 서게 된다.

---

### 2.2 제안하는 방법 및 수식

#### 프로그램의 형식적 정의

프로그램은 4-튜플로 정의된다:

$$P = (S, T, M, V)$$

- $S$: 상태(state) 집합 — 각 상태 $s_i \in S$는 (id, description, verification predicate) 포함
- $T$: 전이(transition) 집합 — 각 전이 $(s_i, s_j, a) \in T$는 액션 $a$를 수행해 $s_i \to s_j$로 이동
- $M$: 메타데이터 — task description, app context, parameter schema, **dedup signature** 포함
- $V$: 데이터 추출 predicate 집합 (QA 작업용, e.g., `inspect_text`)

#### Algorithm 1: PreAct Verified Compile-Extend-Replace Loop

```
Require: Task goal T, environment env, corpus C, threshold τ (default 0.5)

1: P ← SELECTOR(T, C)                    ▷ LLM-agentic; may return ⊥
2: r ← NONE
3: if P ≠ ⊥ then
4:   r ← REPLAY(P, env)                  ▷ 그래프 순회; 각 상태 predicate 검증
5:   if r.success ∧ r.cov > τ then
6:     return r                           ▷ warm path: 신뢰 가능한 replay
7:   end if
8: end if
9: r ← CUA(T, env, r)                    ▷ hybrid: r≠NONE이면 partial-replay 상태에서 계속
10: if ¬r.success then return r
11: end if
12: P' ← COMPILE(r.trace)                ▷ LLM 기반 trace → state-machine
13: env' ← RESET(env, T)                 ▷ 독립 재평가 환경
14: r' ← REPLAY(P', env')
15: score' ← EVALUATE(env', T)
16: if r'.success ∧ score' ≥ 1.0 then    ▷ Double gate
17:   C ← UPSERT(C, P')                  ▷ dedup signature로 삽입; 충돌 시 교체
18: end if
19: return r
```

#### Verify-Before-Store Gate의 조건

$$\text{store}(P') \iff r'.\text{success} \;\land\; \text{EVALUATE}(\text{env}', T) \geq 1.0$$

이 이중 조건이 없으면 **cov=100%/score=0** 프로그램이 corpus에 축적된다. 예: 연락처 폼을 모두 채우고 Save를 눌렀으나, 필드 값이 실제로 저장되지 않아 연락처가 존재하지 않는 경우.

#### Corpus 단조 증가 조건

$$\mathbb{E}[\text{Quality}(C_{t+1})] \geq \mathbb{E}[\text{Quality}(C_t)] \quad \forall t$$

각 UPSERT는 독립적으로 재검증을 통과한 프로그램만 추가하므로, **기댓값 기준 corpus 품질은 감소하지 않는다** (단, 이후 파라미터 바인딩에서 실패 가능성은 여전히 존재).

---

### 2.3 모델 구조

PreAct는 고정된 하네스(harness) 코드 위에서 동작하며, corpus만이 성장하는 구조다.

```
User Goal T
    │
    ▼
┌──────────────────┐
│  Program Selector │ ←── (LLM-agentic 또는 embedding retriever)
└──────────────────┘
    │ P (retrieved) / ⊥ (no candidate)
    ▼
┌──────────────────────┐        replay fail
│  State-machine Replayer │─────────────────►┐
└──────────────────────┘                    │
    │ r.success ∧ r.cov > τ                 │
    ▼                                       ▼
  return (warm)                    ┌──────────────┐
                                   │  CUA fallback │
                                   └──────────────┘
                                        │ trace
                                        ▼
                                   ┌─────────────┐
                                   │   Compiler   │
                                   └─────────────┘
                                        │ P'
                                        ▼
                             ┌─────────────────────────┐
                             │  Verify-Before-Store Gate│
                             │  (reset env + replay P') │
                             └─────────────────────────┘
                                        │ pass / reject
                                        ▼
                              ┌──────────────────┐
                              │  Program Corpus C │ (UPSERT)
                              └──────────────────┘
```

**각 컴포넌트:**

| 컴포넌트 | 역할 |
|---|---|
| **Program Selector** | 목표 $T$와 corpus를 비교해 적합한 프로그램 검색 (LLM 또는 embedding) |
| **State-machine Replayer** | 프로그램 그래프를 순회하며 각 상태 predicate를 라이브 화면과 대조 후 액션 실행 |
| **CUA Fallback** | 재생 실패 시 전체 에이전트(T3A 또는 Claude Computer-Use API)로 제어 이전 |
| **Compiler** | 성공 trace를 LLM이 상태 기계 JSON으로 변환 |
| **Verify-Before-Store Gate** | 컴파일된 프로그램을 깨끗한 환경에서 재실행 + 독립 평가자 채점 |
| **Program Corpus** | dedup signature 기반 UPSERT로 관리되는 mutable 프로그램 라이브러리 |

---

### 2.4 성능 향상

#### 속도 향상

| 플랫폼 | 속도 향상 |
|---|---|
| WebArena | 8.5–13× (wall-clock) |
| Android | 수 초 내 완료 (vs 수십 초) |

#### 성공률 향상 (cold → warm, gate ON)

| 플랫폼 | $\Delta$ gate ON | $\Delta$ gate OFF | Gate 기여 |
|---|---|---|---|
| AndroidWorld (n=5, Gemini) | $+1.2 \pm 0.45$ | $-1.4 \pm 0.89$ | **2.6 tasks** |
| OSWorld (n=5, Claude) | $+0.2 \pm 0.45$ | $-2.4 \pm 0.55$ | **2.6 tasks** |
| WebArena (n=4+4, Claude) | $-4.0 \pm 2.94$ | $-5.75 \pm 1.71$ | **1.75 tasks** |

Gate가 없으면 세 플랫폼 모두에서 반복 실행이 **성능을 오히려 저하**시킨다.

#### 비용

```math
C_{\text{compile+verify}} \approx \begin{cases} +162\% \cdot C_{\text{solve}} & \text{(Android)} \\ +217\% \cdot C_{\text{solve}} & \text{(OSWorld)} \end{cases}
```

이 비용은 1회성이며, 이후 재생 시 $\epsilon \approx 0$이므로 **3번째 실행부터 완전 상쇄**된다.

---

### 2.5 한계 (논문이 명시한 6가지)

| 한계 | 내용 |
|---|---|
| **(L1) 아키텍처 기준선** | flat-script 대비 +0.67 tasks, $p=0.125$ — 유의미하나 통계적 유의성 부족 |
| **(L2) 벤치마크 커버리지** | 6~15개 작업의 소규모 서브셋; 절대 pp 크기는 분모에 의존 |
| **(L3) 컴파일 비용** | 검증 재실행이 원본 실행의 1.6~2.2배 추가 비용 발생 |
| **(L4) 컴파일 충실도** | LLM 컴파일러가 손실 상태 기계 생성 (Android: ~28% navigate_back 누락) |
| **(L5) 셀렉터 비결정성** | 의역 강건성 75.6%; 보수적 설계로 오검색은 0% |
| **(L6) 리셋 가능 환경 가정** | 비가역적 부작용(이메일 발송, 결제) 있는 작업에는 적용 불가 |

---

## 3. 일반화 성능 향상 가능성 (핵심 분석)

### 3.1 현재의 일반화 실패: OOD 실험 결과

논문 §5.6에서 **Out-of-Distribution(OOD) 일반화 실험**을 직접 수행했다:

- **설정:** OSWorld test_tiny (6개 작업)으로 구축된 corpus를 test_small의 30개 OOD 작업에 적용
- **결과:**

$$\text{Warm-OOD SR} = 55.6\% \pm 12\% \quad \text{vs} \quad \text{Cold-OOD SR} \approx 67\%$$

$$\Delta_{\text{OOD}} \approx -11 \text{ pp}$$

**corpus는 OOD에 전이되지 않으며, 오히려 소폭 해롭다.** 셀렉터가 "Chrome history-clean" 프로그램을 "Thunderbird email" 작업에 잘못 검색하여, 소스 도메인의 XPath predicate가 OOD 페이지에서 실패하고 재생 예산을 소모한다.

### 3.2 일반화를 저해하는 구조적 요인

```
[일반화 실패의 3가지 구조적 원인]

1. XPath/UI 셀렉터의 취약성 (Brittleness)
   └─ 각 상태의 검증 predicate가 플랫폼 특이적 XPath에 의존
      └─ 새로운 앱 버전, 다른 도메인의 UI에서 즉시 실패

2. 파라미터화의 한계 (Shallow Parametrization)
   └─ 예: first_name, last_name, phone_number는 추출되나
      작업 구조 자체가 바뀌면(필드 순서 변경, 추가 단계) 프로그램 무효화

3. 셀렉터의 표면적 유사성 오류 (Surface-Level Matching)
   └─ 의미 임베딩이 어휘 유사성("open", "message")에 속아
      다른 작업 가족(task family)의 프로그램을 검색
```

### 3.3 일반화 성능 향상을 위한 논문의 시사점 및 미래 방향

#### (A) 파라미터화 범위 확장
현재: 구체적 값(Emilia Gonzalez → first_name 파라미터)만 추출  
개선 방향: **작업 구조 자체를 파라미터화** (예: 필드 순서, 조건부 분기)

$$P_{\text{generic}} = f(P_{\text{specific}}, \theta_{\text{structure}})$$

#### (B) 상태 predicate의 추상화
현재: `resource_id=com.google.android.contacts:id/floating_action_button` (구체적 XPath)  
개선 방향: **시맨틱 predicate** — "연락처 앱이 열려 있고 + 버튼이 보인다"는 개념 수준 표현  
이는 Vision-Language Model 기반 predicate 검증으로 연결 가능

#### (C) 셀렉터 개선
논문이 발견한 중요한 사실:
- **embedding retriever (MiniLM-L6, bge-large)**가 LLM 셀렉터보다 작업 가족 검색에서 100% 정확도 달성
- 그러나 OOD에서의 false positive를 억제하려면 threshold $\tau$ 튜닝이 필수

$$\text{sim}(q, p) > \tau \implies \text{retrieve}(p), \quad \text{optimal } \tau \in [0.5, 0.65] \text{ (MiniLM)}$$

#### (D) Corpus 내재화 (Internalizing the Corpus)
논문 §5.8의 미래 연구 방향:

$$\theta^* = \arg\max_\theta \mathbb{E}_{P \in C_{\text{verified}}} [\log p_\theta(\text{actions} | \text{state}, P)]$$

검증된 corpus를 **모델 가중치로 내재화**하면 inspectability와 generality 사이의 tradeoff를 조절할 수 있다. 검증 corpus가 자연스러운 지도학습 데이터가 된다.

#### (E) OOD를 위한 실질적 권고사항 (논문 §Appendix D)
> *"corpora built on small in-distribution task subsets should not be deployed cross-domain without retraining the selector or extending the corpus to cover the new domain."*

즉, 일반화를 위해서는:
1. **도메인별 corpus 분리 관리**
2. **새 도메인용 corpus 확장 후 배포**
3. **셀렉터 재학습 또는 threshold 재조정**

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구와의 비교표 (논문 Table 1 확장)

| 시스템 | 연도 | 메모리 표현 | LLM-free 실행 | 자기 확장 | 검증 게이트 |
|---|---|---|---|---|---|
| **Voyager** (Wang et al., TMLR 2024) | 2023 | Code skills | ❌ (LLM 실행) | ✅ (자기 검증) | 부분적 |
| **ExpeL** (Zhao et al., AAAI 2024) | 2023 | 경험 규칙 | ❌ | ✅ (append) | ❌ |
| **ReAct** (Yao et al., ICLR 2023) | 2022 | 없음 | ❌ | ❌ | ❌ |
| **Muscle-Mem** (Pig.dev, 2025) | 2025 | 선형 액션 캐시 | ✅ (blind) | ❌ (append) | ❌ |
| **AgentRR** (Feng et al., 2025) | 2025 | 다중 경험 레벨 | ❌ (간접) | ❌ (append) | ❌ |
| **Workflow-Use** (Browser-Use, 2025) | 2025 | 선형 스크립트 | ✅ (blind) | ❌ (append) | ❌ |
| **ActionEngine** (Zhong et al., 2026) | 2026 | State machine (crawl) | ✅ (flat Python) | ❌ (재크롤) | ❌ |
| **TroVE** (Wang et al., 2024) | 2024 | 검증 가능 toolbox | ❌ (LLM 실행) | ✅ | 부분적 |
| **SAGE** (Liang et al., 2024) | 2024 | 파라미터화 스킬 | ❌ | ✅ | ❌ |
| **PreAct** (Li, 2026) | **2026** | **State machine** | ✅ **(직접 실행)** | ✅ **(UPSERT)** | ✅ **(이중)** |

### 4.2 핵심 차별점 분석

**PreAct vs. Muscle-Mem:**
- Muscle-Mem: 선형 캐시, 맹목적 재생, cache miss 시 CUA 폴백
- PreAct: 상태 기계, 단계별 predicate 검증, 이중 게이트
- WebArena에서 동등 성능 달성 (gate + fallback 조합 시 $p \approx 0.84$, 통계적 차이 없음)

**PreAct vs. ActionEngine:**
- ActionEngine: 크롤링으로 상태 기계 생성 → flat Python으로 변환 후 기계 폐기
- PreAct: 실제 성공 실행에서 컴파일 → **상태 기계를 직접 실행** → predicate 기반 fallback 가능

**PreAct vs. Voyager:**
- Voyager: 코드 스킬이 여전히 LLM-bound 루프 내에서 실행 (런타임 LLM 필요)
- PreAct: 검증된 상태 기계를 **LLM 없이 직접 실행**

### 4.3 포지셔닝 다이어그램 (논문 Figure 3 기반)

```
        ↑
  LLM   │  RPA scripts   •            ★ PreAct
없이    │                Muscle-Mem   (직접 실행 + 자기 확장)
실행   │  Workflow-Use  •
       │  ActionEngine  •
       │               AgentRR    Voyager •
       │         memory systems  TroVE •
       │         (Mem0, A-MEM)   ExpeL •
       │
       └──────────────────────────────────→
              자기 확장/코퍼스 정제 →
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 미래 연구에 미치는 영향

#### (A) 효율성의 재정의
PreAct는 **"반복 작업에서의 효율성"을 1급 평가 지표(first-class metric)**로 격상시킨다. 기존 벤치마크가 성공률만 측정하는 것과 달리, 앞으로의 CUA 연구는 다음을 함께 보고해야 할 것이다:

$$\text{평가 지표} = \{\text{SR}_\text{cold}, \text{SR}_\text{warm}, \Delta_\text{SR}, \text{speedup}_\text{replay}, C_\text{compile}\}$$

#### (B) "검증 없는 재사용은 퇴보다"라는 원칙 확립
논문의 가장 강력한 메시지:

> *"A memory that grows without verification does not make an agent better over time; it makes the agent repeat the same mistake on every reuse."*

이는 스킬 라이브러리, RAG 기반 에이전트 메모리 등 **모든 재사용 기반 시스템에 적용되는 설계 원칙**이다. 향후 연구에서 메모리 시스템의 신뢰성 보장은 핵심 연구 과제가 될 것이다.

#### (C) 코드-as-메모리 패러다임 강화
PreAct는 "recall과 action이 동일한 단계"임을 구현함으로써, **실행 가능한 코드가 에이전트 메모리의 최적 표현**임을 실험적으로 지지한다. 이는 CodeAct, TroVE, LATM 등의 방향과 일치하나, **검증된 상태 기계**라는 더 구체적인 형태로 제안된다.

#### (D) 외재화된 세계 모델로서의 의의
각 저장된 프로그램이 특정 작업의 동역학을 검증 가능한 형태로 외재화한다:

$$M_{\text{task}} = \{(s_i, \text{predicate}_i, a_{i \to j}, s_j)\}_{i,j}$$

이는 가중치에 내재된 세계 모델과 달리 **투명하고, 교체 가능하며, 재훈련 없이 수정 가능**하다. 그러나 generality가 낮다는 tradeoff가 존재한다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (A) 대규모 Corpus에서의 확장성 문제

논문의 corpus는 최대 58개 Android 프로그램에 불과하다. $10^3$ – $10^4$ 규모에서 다음 문제가 발생할 수 있다:

$$\text{verify-replay cost} \propto \text{compile attempts (not stored programs)}$$

**고려할 점:**
- 셀렉터의 판별 정확도: 후보 집합이 커질수록 false positive 위험 증가
- dedup signature 충돌: 유사 작업 가족 간 signature 설계 재검토 필요
- 검증 재실행 비용의 비례적 증가에 대한 분산/캐싱 전략

#### (B) 비가역적 부작용 처리 (L6)

현재 verify-gate는 **환경이 리셋 가능하다고 가정**한다. 실제 환경에서:

$$\text{email sent} + \text{payment charged} \neq \text{idempotent}$$

**연구 방향:**
- **Read-only 검증 경로**: 최종 상태를 읽기 전용으로 확인
- **샌드박스 드라이런**: 격리 환경에서 프로그램 검증
- **트랜잭션 롤백**: 검증 후 상태를 원복하는 메커니즘

#### (C) 컴파일러 충실도 향상

현재 LLM 컴파일러의 주요 결함:
- Android: ~28% 프로그램에서 `navigate_back` 누락
- WebArena: 100% 프로그램이 동적 상태에 `inspect_screenshot` 의존

**연구 방향:**
- 컴파일 시 도메인별 체크리스트 생성 (Android nav-heavy 작업 → navigate_back 강제 포함)
- 동적 vs 정적 상태 분류기로 WebArena 스타일 작업에서의 컴파일 전략 분기

#### (D) OOD 일반화를 위한 추상화 레벨 조정

현재 predicate: `resource_id=com.google.android.contacts:id/floating_action_button`  
→ 특정 앱 버전, 특정 플랫폼에 종속

**연구 방향:**
- **다중 추상화 레벨의 predicate**: XPath(구체) + 시맨틱(추상)을 계층적으로 저장
- **VLM 기반 predicate**: 스크린샷에서 의미 기반 조건 검증 (e.g., "연락처 생성 버튼이 화면에 보임")
- 도메인 어댑터를 corpus에 결합하여 다른 앱 버전에 자동 적응

#### (E) 장기 작업(Long-Horizon Tasks) 통합

현재 벤치마크는 10–30 액션 규모. 수백 액션이 필요한 복합 워크플로우에서:

$$\text{replay failure} \propto \text{task horizon}$$

**고려할 점:**
- 작업 분해(goal decomposition) 플래너와 PreAct를 통합
- 부분 재생 성공(partial replay)의 신용 할당(credit assignment) 문제
- 분기 상태(branching states)의 동적 추가: 앱 버전별 다이얼로그 처리

#### (F) 강화학습과의 통합

논문은 PreAct가 보상 신호 없이 동작하나, 미래 연구 방향으로:

$$\theta^* = \arg\max_\theta \mathbb{E}_{P \in C_{\text{verified}}} [R(\text{REPLAY}(P, \text{env}))]$$

검증된 corpus를 **오프라인 강화학습의 시연 데이터(demonstration)**로 활용하면, 외재화된 세계 모델을 가중치로 내재화하는 경로를 열 수 있다. 특히 DigiRL(Bai et al., NeurIPS 2024) 등 디바이스 제어 RL 연구와의 시너지가 기대된다.

---

## 참고 자료 (출처)

**논문 원문:**
- Bojie Li, "PreAct: Computer-Using Agents that Get Faster on Repeated Tasks," arXiv:2606.17929v1 [cs.AI], 16 Jun 2026.
  - Code: https://github.com/19PINE-AI/PreAct
  - Website: https://01.me/research/PreAct/

**논문 내 인용 문헌 (본 분석에서 직접 참조한 것):**
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023.
- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," TMLR 2024. arXiv:2305.16291.
- Rawles et al., "AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents," ICLR 2025.
- Xie et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments," NeurIPS 2024.
- Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents," ICLR 2024.
- Pig.dev, "Muscle-Mem: A cache for AI agents to learn and replay complex behaviors," 2025. https://github.com/pig-dot-dev/muscle-mem
- Browser-Use, "Workflow-Use: Create and run workflows (RPA 2.0)," 2025. https://github.com/browser-use/workflow-use
- Zhong et al., "ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory," arXiv:2602.20502, 2026.
- Zhao et al., "ExpeL: LLM Agents are Experiential Learners," AAAI 2024.
- Bai et al., "DigiRL: Training In-the-Wild Device-Control Agents with Autonomous Reinforcement Learning," NeurIPS 2024.
- Feng et al., "Get Experience from Practice: LLM Agents with Record & Replay," arXiv:2505.17716, 2025.
- Wang et al., "TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks," arXiv:2401.12869, 2024.
- Qin et al., "UI-TARS: Pioneering Automated GUI Interaction with Native Agents," arXiv:2501.12326, 2025.
