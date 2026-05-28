# ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

ARIS의 근본적인 주장은 다음 하나의 공리(axiom)로 압축됩니다:

> *"Any long-term task performed by a single agent is unreliable."*
> *단일 에이전트가 수행하는 장기 태스크는 신뢰할 수 없다.*

이 전제 위에서, 논문은 자율 ML 연구 워크플로우의 핵심 실패 모드가 **가시적 오류(visible breakdown)** 가 아니라 **그럴듯한 근거 없는 성공(plausible unsupported success)** 임을 강조합니다. 즉, 에이전트가 생성한 주장(claim)이 실험 증거를 초과하거나, 결과가 실제이지만 잘못 보고되는 상황이 더 위험하다는 것입니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **교차 패밀리 적대적 협력** | 실행자(executor)와 검토자(reviewer)를 서로 다른 모델 패밀리에서 선택 |
| **3계층 아키텍처** | 실행(Execution) / 조율(Orchestration) / 보증(Assurance) 계층 분리 |
| **증거-주장 감사 캐스케이드** | 3단계 실험 무결성 검증 파이프라인 |
| **65개 이상의 재사용 가능 스킬** | Markdown으로 정의된 모듈화 스킬 라이브러리 |
| **영속적 연구 위키(Research Wiki)** | 세션 간 지식을 누적하는 구조화 메모리 |
| **메타 최적화 루프** | 하네스(harness) 자체를 개선하는 외부 루프 |

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

### 2.1 해결하고자 하는 문제

ARIS가 이전 시스템(AI Scientist, Agent Laboratory 등)의 세 가지 반복적 한계를 지적합니다:

**(1) 동일 모델 자기 검토 문제 (Same-model self-refinement)**

Self-Refine (Madaan et al., 2023), Reflexion (Shinn et al., 2024) 계열 시스템은 생성자(generator)와 검증자(validator)가 동일한 귀납적 편향(inductive bias)을 공유합니다. 이 경우 상관된 오류(correlated errors)가 걸러지지 않습니다.

**(2) 단계 간 강한 결합(Tightly coupled pipeline)**

워크플로우가 하나의 불투명한 에이전트 경로(opaque agent trajectory) 내부에 숨겨져 있어, 개별 단계의 교체나 중간 상태에서의 재시작이 어렵습니다.

**(3) 실험 무결성 검증 부재**

시스템 수준의 실험 결과 검증 및 논문 품질 감사 메커니즘이 명시적으로 존재하지 않습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 적대적 밴딧 유추(Adversarial Bandit Analogy)

논문은 단일 모델 자기 검토를 **확률적 밴딧(stochastic bandit)**, 교차 모델 검토를 **적대적 밴딧(adversarial bandit)** 에 비유합니다.

단일 모델 자기 검토에서의 리뷰 점수를 $R_t$라 하면:

$$R_t = f_\theta(a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

여기서 생성자와 검증자가 동일한 파라미터 $\theta$를 공유하므로, 오류 $\epsilon_t$는 독립적이지 않고 서로 상관됩니다 ($\text{Cov}(\epsilon_t^{\text{gen}}, \epsilon_t^{\text{val}}) \neq 0$).

반면 ARIS의 교차 패밀리 설정에서:

$$R_t = f_{\theta_{\text{exec}}}(a_t) + g_{\theta_{\text{rev}}}(a_t) + \epsilon_t$$

검토자 $g_{\theta_{\text{rev}}}$는 실행자 $f_{\theta_{\text{exec}}}$와 독립적인 파라미터를 가지므로:

$$\text{Cov}(\epsilon_t^{\text{exec}}, \epsilon_t^{\text{rev}}) \approx 0$$

이 적대적 검토자는 실행자가 예상하지 못한 취약점을 능동적으로 탐색하므로, 적대적 밴딧은 확률적 밴딧보다 근본적으로 "속이기 어렵습니다(harder to game)."

#### 2.2.2 비판-행동 루프(Critique-to-Action Loop)

Auto Review Loop(Workflow 2)의 수렴 조건을 공식화하면:

$$\text{Accept artifact} \iff \left( s_t \geq \tau \right) \land \left( \text{all critical items resolved} \right)$$

또는:

$$\text{Terminate} \iff t = T_{\max}$$

여기서:
- $s_t$: 라운드 $t$에서의 검토자 점수 (기본값 $\tau = 6/10$)
- $T_{\max}$: 최대 라운드 수 (기본값 $T_{\max} = 4$)

각 라운드의 흐름:

$$a_t \xrightarrow{\text{reviewer}} (s_t, \mathcal{C}_t) \xrightarrow{\text{executor}} a_{t+1}$$

여기서 $\mathcal{C}_t$는 구조화된 액션 아이템(actionable critique) 집합입니다.

#### 2.2.3 증거-주장 감사 캐스케이드(Evidence-to-Claim Audit Cascade)

3단계 감사 파이프라인을 수학적으로 표현하면:

**Stage 1 (Experiment-Integrity Audit):**

$$\text{IntegrityStatus}(e) = \begin{cases} \texttt{pass} & \text{if no failure mode detected} \\ \texttt{warn} & \text{if minor issue detected} \\ \texttt{fail} & \text{if integrity failure mode present} \end{cases}$$

5가지 무결성 실패 모드 $\mathcal{F} = \{f_1, f_2, f_3, f_4, f_5\}$:
- $f_1$: 모델 유래 참조 레이블(model-derived reference labels)
- $f_2$: 자기 정규화 점수(self-normalized scores)
- $f_3$: 유령 결과(phantom results)
- $f_4$: 사용되지 않는 메트릭 인플레이션(dead-code inflation)
- $f_5$: 범위 인플레이션(scope inflation)

**Stage 2 (Result-to-Claim Mapping):**

각 주장 $c_i$에 대해:

$$\text{Verdict}(c_i) = \begin{cases} \texttt{supported} & \text{if } \text{IntegrityStatus}(e_i) \neq \texttt{fail} \land \mathcal{E}(c_i) \text{ consistent} \\ \texttt{partially supported} & \text{if partial evidence} \\ \texttt{invalidated} & \text{if } \text{IntegrityStatus}(e_i) = \texttt{fail} \lor \mathcal{E}(c_i) \text{ contradicts } c_i \end{cases}$$

**Stage 3 (Paper-Claim Audit):**

신선한 제로 컨텍스트 검토자(fresh zero-context reviewer)가 각 정량적 주장 $\hat{c}_i$를 클레임 원장(claim ledger) $\mathcal{L}$과 대조:

$$\text{AuditStatus}(\hat{c}_i) \in \{\texttt{exact match}, \texttt{rounding ok}, \texttt{number mismatch}, \texttt{config mismatch}, \texttt{missing evidence}\}$$

#### 2.2.4 노력 수준(Effort Levels) 스케일링

$$\text{Coverage}(E) = \alpha_E \cdot \text{Coverage}(\text{balanced})$$

여기서 스케일 인수 $\alpha_E$:

| Effort Level | $\alpha_E$ |
|---|---|
| `lite` | $\approx 0.4\times$ |
| `balanced` | $1\times$ (기본값) |
| `max` | $\approx 2.5\times$ |
| `beast` | $\approx 5\text{–}8\times$ |

단, 검토자의 추론 예산(reasoning budget)은 effort level에 관계없이 항상 `xhigh`로 고정됩니다.

---

### 2.3 모델 구조 (3계층 아키텍처)

```
┌─────────────────────────────────────────────────────┐
│              META-OPTIMIZATION (Outer Loop)          │
│    events.jsonl → /meta-optimize → reviewer-gated   │
├─────────────────────────────────────────────────────┤
│              ASSURANCE LAYER                         │
│  A1: experiment-audit → A2: result-to-claim →       │
│  A3: paper-claim-audit + ManuscriptQA               │
├─────────────────────────────────────────────────────┤
│              ORCHESTRATION LAYER                     │
│  W1 → W1.5 → W2 → W3 → W4                          │
│  (Idea Discovery → Experiment → Review → Write →    │
│   Rebuttal)                                         │
├─────────────────────────────────────────────────────┤
│              EXECUTION LAYER                         │
│  65+ SKILL.md files + Research Wiki + FigureSpec    │
├─────────────────────────────────────────────────────┤
│  Executor (Claude/Codex) ⟷ Reviewer (GPT/Gemini)   │
└─────────────────────────────────────────────────────┘
```

#### 핵심 구성 요소

**실행 계층(Execution Layer):**
- 65개 이상의 Markdown 정의 스킬 (`SKILL.md`)
- 연구 위키: 4가지 엔티티 타입 (papers, ideas, experiments, claims)
- 8가지 타입 관계 그래프 (`extends`, `contradicts`, `supports`, `invalidates` 등)
- FigureSpec JSON → SVG 결정론적 렌더러

**조율 계층(Orchestration Layer):**
- 5개 엔드-투-엔드 워크플로우 (W1, W1.5, W2, W3, W4)
- 4단계 노력 사전 설정(effort presets)
- 6개 MCP 브리지 (Codex, Oracle, Claude, Gemini, MiniMax, llm-chat)

**보증 계층(Assurance Layer):**
- 3단계 증거-주장 감사 캐스케이드
- 5패스 과학적 편집 파이프라인
- 수학적 증명 검증기 (20개 카테고리 분류)
- 시각적 PDF 검토 (LaTeX 소스 + 컴파일된 PDF 동시 제공)
- 인용 감사 (존재성 + 메타데이터 정확성 + 맥락 적절성)

**교차 모델 적대적 협업(Cross-Model Adversarial Collaboration):**

```
Executor (Claude/Codex/Cursor)
    ↓ generates artifact
Reviewer (GPT-5.4/Gemini) [다른 모델 패밀리]
    ↓ scores + structured critique
Executor: revise per action items
    ↓
Convergence Check: s_t ≥ τ or t = T_max?
    ├── Yes → Accept
    └── No → Next round
         └── Experiment fails? → Auto-debug (3 retries)
              └── Still fails? → /codex:rescue (3rd model)
```

**연구 위키의 나선형 학습(Spiral Learning):**

위키 **없이**:
$$\text{Session}_1: A \to \text{fail}, \quad \text{Session}_2: A \to \text{fail (forgot)}, \quad \ldots$$

위키 **있음**:
$$\text{Session}_1: A \to \text{fail} \to \mathcal{W}=\{A^\times\}$$
$$\text{Session}_2: \text{read } \mathcal{W} \to \text{skip } A \to B \to \checkmark \to \mathcal{W}=\{A^\times, B^\checkmark\}$$
$$\text{Session}_3: \text{build on } B \to C, D \to \mathcal{W}=\{A^\times, B^\checkmark, C^\times, D^\checkmark\}$$

---

### 2.4 성능 향상

논문은 단일 야간 실행(overnight run) 관찰 증거만을 보고합니다:

| 지표 | 시작 | 종료 | 변화 |
|---|---|---|---|
| 내부 검토자 점수 | 5.0/10 | 7.5/10 | $+50\%$ |
| GPU 실험 수 | 0 | 20+ | — |
| 소요 시간 | — | ~8시간 | — |
| 검토-수정 라운드 | — | 4회 | — |

> ⚠️ **중요한 주의사항**: 논문 자체가 이 결과가 **단일 경로(single trajectory)에서의 관찰적 증거**임을 명시합니다. 교차 패밀리 검토의 우월성을 인과적으로 증명하는 통제된 실험은 아직 수행되지 않았습니다.

---

### 2.5 한계

| 한계 | 설명 |
|---|---|
| **통제된 평가 부재** | 관찰적 증거에만 의존; 인과 추론 불가 |
| **정확성 미보장** | LLM 출력의 허위 정보, 방법론적 결함 제거 불가 |
| **감사 한계** | 3단계 감사는 권고적(advisory) 수준; 형식 검증 시스템 아님 |
| **검토자 편향 증폭** | 루프가 검토자 모델의 선호도에 과적합될 위험 |
| **보안 우려** | 저장소 수준 검토 시 소스 코드가 외부 LLM API에 전송됨 |
| **로컬 전용 검토 미구현** | 기밀 환경에서의 활용 제한 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 교차 패밀리 설계가 일반화에 기여하는 메커니즘

ARIS의 교차 패밀리(cross-family) 접근 방식은 다음 세 가지 경로를 통해 일반화 성능에 기여합니다:

**① 귀납적 편향의 다양화(Diversification of Inductive Biases)**

동일 모델 패밀리 내 검토는 공유된 사전 지식(prior knowledge)과 훈련 데이터로 인해 비상관 오류(uncorrelated errors)를 발견하기 어렵습니다. Claude 계열 실행자 + GPT 계열 검토자 조합은:

$$\text{Bias}(\theta_{\text{exec}}) \not\approx \text{Bias}(\theta_{\text{rev}})$$

이로 인해 단일 모델로는 발견하기 어려운 체계적 오류(systematic errors)가 노출될 가능성이 높아집니다. 이는 Du et al. (2024)의 다중 에이전트 토론(multi-agent debate)과 Liang et al. (2024a)의 이질적 LLM 참여 연구에서 확인된 효과입니다.

**② 영속적 연구 위키를 통한 세션 간 일반화**

단일 세션 학습에서 다중 세션 누적 학습으로 전환합니다:

$$\mathcal{W}_{t+1} = \mathcal{W}_t \cup \{(\text{idea}_t, \text{outcome}_t, \text{claim status}_t)\}$$

실패한 아이디어가 "금지 목록(banlist)"으로 유지되어, 동일한 방향을 재탐색하는 비효율을 방지합니다. 이는 아이디어 공간에서의 일반화된 탐색 효율을 높입니다.

**③ 메타 최적화를 통한 하네스 수준의 일반화**

패시브 이벤트 로깅 → 패턴 분석 → 검토자 게이트 적용의 외부 루프:

$$\mathcal{H}_{t+1} = \mathcal{H}_t + \Delta\mathcal{H}_t, \quad \text{if reviewer}(\Delta\mathcal{H}_t) \geq 7/10$$

여기서 $\mathcal{H}$는 하네스(스킬 프롬프트, 기본 파라미터, 수렴 규칙)입니다. 이 메커니즘은 특정 작업에 과적합된 하네스를 점차 일반화하는 방향으로 개선시킬 잠재력을 가집니다.

### 3.2 일반화 성능과 관련한 한계

**검토자 편향 과적합 위험:**

루프가 반복되면, 실행자는 실제 과학적 품질 향상이 아닌 특정 검토자 모델의 선호도에 맞는 결과물을 생성하는 방향으로 편향될 수 있습니다:

$$a^* = \arg\max_{a} \mathbb{E}_{\theta_{\text{rev}}}[s(a)] \neq \arg\max_{a} \text{TrueQuality}(a)$$

**재귀적 훈련 데이터 품질 저하 위험:**

논문은 Shumailov et al. (2024)의 연구를 인용하며, 모델 생성 데이터로 재귀적 훈련 시 품질이 반복에 걸쳐 저하될 수 있음을 경고합니다. 교차 패밀리 검토자 분리가 이 문제를 완화하는 후보 메커니즘으로 제안되지만, 장기 자기 개선에서의 효과는 아직 미검증 상태입니다.

**통제된 일반화 평가 프로토콜의 부재:**

현재 보고된 성과는 단일 논문, 단일 야간 실행에 기반합니다. Appendix E에 제시된 미래 작업 프로토콜은 12개 이상의 논문 초안, 5가지 조건(A~E)에서의 통제된 비교를 계획하지만, 아직 실행되지 않았습니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

**① 하네스 엔지니어링의 새로운 패러다임**

ARIS는 자율 연구 시스템의 성능이 단순히 모델 가중치가 아니라, 모델을 둘러싼 **시스템 로직(harness)**에 크게 의존한다는 점을 체계적으로 보여줍니다. 이는 Lee et al. (2026)의 Meta-Harness 연구와 함께, 하네스 엔지니어링을 독립적인 연구 분야로 확립시키는 데 기여합니다.

**② 다중 에이전트 연구의 실용적 설계 원칙 제시**

Du et al. (2024), Liang et al. (2024a)의 이론적 연구에서 실제 연구 워크플로우로 교차 패밀리 검토 원칙을 적용한 첫 번째 본격적인 구현 사례입니다. 향후 다중 에이전트 연구에서:
- 최적 위원회 크기(committee size) 탐구
- 검토자 다양성(reviewer diversity)의 정량적 측정
- 도메인별 최적 모델 패밀리 조합 탐색

에 대한 연구 방향을 제시합니다.

**③ AI 연구 무결성(Research Integrity) 도구의 발전**

3단계 증거-주장 감사 캐스케이드는 자율 연구 시스템의 "허위 성공(plausible unsupported success)" 문제를 해결하는 첫 번째 체계적 시도입니다. Luo et al. (2025)가 지적한 자율 연구 시스템의 무결성 실패 모드(부적절한 벤치마크 선택, 데이터 누출, 메트릭 오용 등)에 대한 방어 체계로서, 향후 자율 연구 시스템의 표준 컴포넌트로 자리잡을 가능성이 있습니다.

**④ LLM 자기 개선(Self-Improvement) 연구에의 적용**

논문은 Bai et al. (2022), Yuan et al. (2024)의 자기 개선 접근법에 교차 모델 책임성 프리미티브(cross-model accountability primitives)를 삽입하는 방향을 제안합니다:

$$\text{TrainingSignal}_{t+1} = \text{CrossFamilyAudit}(\text{ModelOutput}_t)$$

이는 재귀적 모델 붕괴(model collapse, Shumailov et al., 2024)를 방지하는 실용적 메커니즘으로 탐구될 수 있습니다.

---

### 4.2 앞으로 연구 시 고려해야 할 점

**① 통제된 실험 설계의 필요성**

현재 ARIS의 주장은 관찰적 증거에만 기반합니다. Appendix E에 제시된 벤치마크 프로토콜을 확장하여:

$$\text{Condition A}: \text{Single-model self-critique}$$
$$\text{Condition B}: \text{Same-model two-agent}$$
$$\text{Condition C}: \text{Cross-model (ARIS default)}$$
$$\text{Condition D}: \text{Cross-model reversed}$$
$$\text{Condition E}: \text{Same-model for second model}$$

컴퓨팅 매칭(compute-matched) 조건에서의 엄밀한 비교 실험이 필요합니다.

**② 검토자 편향 측정 및 완화**

검토자가 일관되게 특정 방법론을 선호할 경우, 실행자가 실제 품질 향상이 아닌 검토자 선호도에 최적화될 수 있습니다. 이를 측정하기 위해:

$$\text{ReviewerBias}(\theta_{\text{rev}}) = D_{KL}(P(\text{score}|\theta_{\text{rev}}) \| P(\text{TrueQuality}))$$

를 추정하는 메커니즘이 필요합니다. 여러 검토자 모델의 앙상블, 또는 주기적 검토자 교체 정책이 대안으로 고려될 수 있습니다.

**③ 로컬 검토자 모델의 개발**

현재 저장소 수준 검토는 소스 코드를 외부 LLM API로 전송합니다. 기밀성이 중요한 산업 환경에서의 활용을 위해, 로컬 전용(local-only) 검토자 경로 개발이 필요합니다. 이는 고성능 로컬 LLM(예: Llama, Mistral 계열)의 검토 역할 수행 능력 평가를 요구합니다.

**④ 최적 위원회 크기 연구**

ARIS는 2명(실행자 + 검토자)이 자기 검토 맹점을 제거하기 위한 최소 구성이라고 주장하며, 2인 게임이 $n$인 게임보다 Nash 균형으로 수렴이 효율적이라는 논거를 제시합니다. 그러나:

$$\text{OptimalCommittee} = \arg\max_n \left[\text{ReviewQuality}(n) - \text{APICost}(n)\right]$$

에 대한 경험적 연구가 필요합니다. $n > 2$의 검토자 위원회가 특정 도메인에서 더 나은 성능을 보일 수 있습니다.

**⑤ 연구 맛(Research Taste)의 정량화**

논문은 아이디어 품질에 연구 맛(research taste)이 중요하다고 언급하며, Tong et al. (2026)의 "AI can learn scientific taste" 연구를 권고합니다. 향후 연구에서는 아이디어의 신선함, 중요성, 실현 가능성을 정량화하는 메트릭 개발이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 시스템 | 연도 | 교차 패밀리 정책 | 적대적 검토 | 보증 스택 | E2E 워크플로우 | 주요 특징 |
|---|---|---|---|---|---|---|
| **Self-Refine** (Madaan et al.) | 2023 | 없음 | ✗ | ✗ | ✗ | 단일 모델 자기 피드백 |
| **Reflexion** (Shinn et al.) | 2024 | 없음 | ✗ | ✗ | ✗ | 언어 에이전트 강화 학습 |
| **Multi-Agent Debate** (Du et al.) | 2024 | 없음 | 부분 | ✗ | ✗ | 다중 에이전트 추론 개선 |
| **MetaGPT** (Hong et al.) | 2023 | 없음 | 부분 | ✗ | ✗ | 소프트웨어 개발 초점 |
| **AutoGen** (Wu et al.) | 2023 | 없음 | ✗ | ✗ | ✗ | 범용 다중 에이전트 |
| **AI Scientist** (Lu et al.) | 2024 | 없음 | 부분 | 부분 | ✓ | 아이디어→논문 자동화 |
| **AI Scientist-v2** (Yamada et al.) | 2025 | 없음 | 부분 | 부분 | ✓ | 에이전틱 트리 탐색 |
| **Agent Laboratory** (Schmidgall et al.) | 2025 | 없음 | ✗ | ✗ | ✓ | 인간-in-the-loop |
| **data-to-paper** (Ifargan et al.) | 2025 | 없음 | 부분 | 부분 | ✓‡ | 데이터→논문, 역추적 |
| **OpenHands** (Wang et al.) | 2025 | 없음 | ✗ | ✗ | ✗ | 범용 AI 소프트웨어 개발자 |
| **AutoResearchClaw** (Liu et al.) | 2026 | 미공개 | 미공개 | 미공개 | ✓ | 아이디어→논문 완전 자동화 |
| **EvoScientist** (Lyu et al.) | 2026 | 미공개 | 미공개 | 미공개 | ✓ | 다중 에이전트 진화적 탐색 |
| **ARIS** (Yang et al.) | 2026 | **기본(default)** | **✓** | **✓** | **✓** | 교차 패밀리, 모듈화, 보증 스택 |

### 핵심 차별점

**AI Scientist vs. ARIS:**
- AI Scientist는 단일 모델 패밀리로 실행 및 검토를 수행하여 상관 오류에 취약합니다.
- ARIS는 교차 패밀리 검토를 기본값으로 설정하고, 3단계 증거-주장 감사를 추가합니다.

**Agent Laboratory vs. ARIS:**
- Agent Laboratory는 인간-in-the-loop 체크포인트를 추가하지만, 시스템 수준의 적대적 검토 메커니즘이 없습니다.
- ARIS는 자동화와 무결성 보장을 동시에 추구하며, 인간 참여를 선택적으로 구성 가능합니다.

**Multi-Agent Debate vs. ARIS:**
- Du et al. (2024)은 이질적 LLM 토론의 이론적/실험적 가치를 보여주지만, 완전한 연구 워크플로우를 지원하지 않습니다.
- ARIS는 이 원칙을 실제 ML 연구 워크플로우에 통합한 첫 번째 오픈소스 구현입니다.

---

## 참고 자료 (출처)

**주요 참고 논문 (논문 내 인용 기준):**

1. **Yang, R., Li, Y., & Li, S. (2026)**. ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration. *arXiv preprint arXiv:2605.03042v1*. https://arxiv.org/abs/2605.03042

2. **Lu, C. et al. (2024)**. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.

3. **Yamada, Y. et al. (2025)**. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. *arXiv preprint arXiv:2504.08066*.

4. **Schmidgall, S. et al. (2025)**. Agent Laboratory: Using LLM Agents as Research Assistants. *Findings of ACL: EMNLP 2025*, pp. 5977–6043.

5. **Du, Y. et al. (2024)**. Improving Factuality and Reasoning in Language Models through Multiagent Debate. *ICML 2024*.

6. **Liang, T. et al. (2024a)**. Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. *EMNLP 2024*, pp. 17889–17904.

7. **Madaan, A. et al. (2023)**. Self-Refine: Iterative Refinement with Self-Feedback. *arXiv:2303.17651*.

8. **Shinn, N. et al. (2024)**. Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv:2303.11366*.

9. **Shumailov, I. et al. (2024)**. AI Models Collapse When Trained on Recursively Generated Data. *Nature*, 631(8022), 755–759.

10. **Luo, Z., Kasirzadeh, A., & Shah, N.B. (2025)**. The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems. *arXiv:2509.08713*.

11. **Lee, Y. et al. (2026)**. Meta-Harness: End-to-End Optimization of Model Harnesses. *arXiv:2603.28052*.

12. **Ifargan, T. et al. (2025)**. Autonomous LLM-Driven Research—From Data to Human-Verifiable Research Papers. *NEJM AI*, 2(1):AIoa2400555.

13. **Wu, Q. et al. (2023)**. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.

14. **Hong, S. et al. (2023)**. MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. *ICLR 2023*.

15. **Wang, X. et al. (2025)**. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. *ICLR 2025*.

16. **Zheng, L. et al. (2023)**. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 36*, 46595–46623.

17. **Liu, R. & Shah, N.B. (2023)**. ReviewerGPT? An Exploratory Study on Using Large Language Models for Paper Reviewing. *arXiv:2306.00622*.

18. **Bai, Y. et al. (2022)**. Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

19. **Yuan, W. et al. (2024)**. Self-Rewarding Language Models. *arXiv:2401.10020*.

20. **Tong, J. et al. (2026)**. AI Can Learn Scientific Taste. *arXiv:2603.14473*.

21. **Liang, W. et al. (2024b)**. Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis. *NEJM AI*, 1(8):AIoa2400196.

22. **GitHub Repository**: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
