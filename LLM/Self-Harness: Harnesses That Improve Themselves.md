# Self-Harness: Harnesses That Improve Themselves 

> **참고 논문**: Hangfan Zhang et al., "Self-Harness: Harnesses That Improve Themselves," arXiv:2606.09498v1, June 8, 2026. Shanghai Artificial Intelligence Laboratory.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

LLM 기반 에이전트의 성능은 **베이스 모델**과 **하네스(harness)** — 시스템 프롬프트, 도구, 런타임 메커니즘, 검증 규칙 등의 비매개변수적 스캐폴딩 — 가 공동으로 결정한다. 기존에는 하네스 설계를 인간 전문가나 더 강력한 외부 에이전트에 의존했으나, 이 논문은 **에이전트가 자신의 하네스를 스스로 개선**할 수 있음을 제안한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **새로운 패러다임** | Self-Harness: 인간 엔지니어링이나 외부 강력 에이전트 없이 LLM이 자신의 하네스를 개선 |
| **운영화(Operationalization)** | 3단계 반복 루프: Weakness Mining → Harness Proposal → Proposal Validation |
| **실증적 검증** | Terminal-Bench-2.0에서 3가지 이질적 모델 패밀리 (MiniMax M2.5, Qwen3.5-35B-A3B, GLM-5) 모두에서 일관된 성능 향상 |
| **정성적 분석** | 단순 프롬프트 추가가 아닌, 모델별 구체적·실행 가능한 하네스 변경임을 확인 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

현재 하네스 설계 패러다임의 세 가지 문제:

1. **확장성 부재**: LLM이 빠르게 다양화·진화함에 따라 모델마다 하네스를 수동 설계하는 것은 비용이 증가
2. **모델 특수성 무시**: 서로 다른 모델은 고유한 행동 패턴, 도구 사용 습관, 오류 모드를 가지므로 범용 하네스는 차선책
3. **외부 의존성**: Meta-Harness 등 외부 최적화 접근법은 더 강력한 모델에 의존하거나, 타겟 모델의 실패 모드와 불일치 가능

```
Human Engineering → 인간 전문가가 하네스 수동 수정
Meta-Harness     → 강력한 외부 에이전트가 약한 타겟 에이전트를 개선
Self-Harness     → 에이전트가 자신의 하네스를 직접 개선 ✓
```

### 2.2 제안 방법 및 수식

#### 기본 형식화

고정된 언어 모델 $M$과 에이전트 하네스 $h$가 주어졌을 때, 태스크 인스턴스 $x$에 대해:

$$\tau, y = M(x; h)$$

여기서 $\tau$는 실행 트레이스(메시지, 도구 호출, 검증 결과), $y$는 출력이다. 평가자 $\mathcal{E}$는 다음을 계산한다:

$$z_i = \mathcal{E}(x_i, \tau_i, y_i) \in \{\text{pass}, \text{fail}\}$$

Self-Harness는 $h_0, h_1, \ldots, h_T$의 하네스 계보를 통해 모델 가중치가 아닌 **하네스 자체**를 개선한다.

#### Stage 1: Weakness Mining — 실패 패턴 클러스터링

라운드 $t$에서 held-in 분할 $D_\text{in}$에 대한 트레이스 레코드 집합:

$$R_t = \{r_i\}_{i=1}^{|D_\text{in}|}, \quad r_i = (x_i, \tau_i, y_i, z_i)$$

실패 레코드 집합:

$$F_t = \{r_i \in R_t \mid z_i = \text{fail}\}$$

각 실패 레코드 $r_i$에 대한 **실패 시그니처(failure signature)**:

$$\phi(r_i) = (c_i, q_i, m_i)$$

- $c_i$: 검증자 수준의 최종 원인 (terminal verifier-level cause)
- $q_i$: 관련 에이전트 행동의 인과적 상태 (causal status)
- $m_i$: 트레이스가 드러내는 추상적 에이전트 메커니즘

동일 시그니처를 가진 실패들의 클러스터:

$$C_\phi = \{r_i \in F_t \mid \phi(r_i) = \phi\}$$

이 클러스터링은 **결정론적이고 검증자 기반**이다: 두 실패 사례는 (1) 검증자가 최종적으로 거부한 것, (2) 에이전트 행동이 그 거부에 기여한 방식, (3) 관련된 재사용 가능한 행동 메커니즘 모두에서 일치할 때만 같은 그룹에 속한다.

#### Stage 2: Harness Proposal — 다양하고 최소한의 후보 수정 생성

증거 번들 $B_t$를 기반으로, 동일한 고정 모델 $M$이 제안자(proposer) 역할로 $K$개의 상호 구별되는 제안 번들을 병렬 생성:

$$\mathcal{P}_t = \{(\Delta_j, a_j)\}_{j=1}^{K}$$

각 편집 $\Delta_j$는 현재 하네스를 후보 하네스로 변환:

$$h_t^{(j)} = \Delta_j(h_t)$$

$a_j$는 (타겟 실패 패턴, 편집된 하네스 표면, 예상 행동 효과, 회귀 위험)을 포함하는 감사 레코드이다.

**다양성**은 제안 브랜치들 간에 권장되고, **최소성**은 각 브랜치 내에서 강제된다.

#### Stage 3: Proposal Validation — 회귀 테스트를 통한 검증

하네스 $h$에 대한 $D_\text{in}$과 $D_\text{ho}$의 통과 태스크 수를 각각 $P_\text{in}(h)$, $P_\text{ho}(h)$라 할 때, 후보 $h_t^{(j)}$의 분할별 개선량:

$$\Delta_\text{in}^{(j)} = P_\text{in}(h_t^{(j)}) - P_\text{in}(h_t)$$

$$\Delta_\text{ho}^{(j)} = P_\text{ho}(h_t^{(j)}) - P_\text{ho}(h_t)$$

**수락 조건 (보수적 프로모션 기준)**:

$$\Delta_\text{in}^{(j)} \geq 0, \quad \Delta_\text{ho}^{(j)} \geq 0, \quad \max\left(\Delta_\text{in}^{(j)}, \Delta_\text{ho}^{(j)}\right) > 0$$

즉, 어느 한 분할에서만 향상되고 다른 분할을 저하시키는 제안은 **거부**된다. 여러 후보가 수락 조건을 만족하면 병합(merge)하여 $h_{t+1}$을 형성한다.

#### 전체 알고리즘 흐름 (Algorithm 1 요약)

$$h_{t+1} = \begin{cases} \text{MERGEACCEPTED}(h_t, \mathcal{A}_t) & \text{if } \mathcal{A}_t \neq \emptyset \\ h_t & \text{if } \mathcal{A}_t = \emptyset \end{cases}$$

### 2.3 모델 구조

Self-Harness의 구성 요소:

```
┌─────────────────────────────────────────────────────────┐
│                     Self-Harness Loop                   │
│                                                         │
│  현재 하네스 h_t                                          │
│       ↓                                                 │
│  [Weakness Mining]                                      │
│  - 고정 모델 M으로 D_in 태스크 실행                          │
│  - 실패 트레이스 수집 → φ(r_i) 시그니처 클러스터링             │
│  - 증거 번들 B_t 구성                                      │
│       ↓                                                 │
│  [Harness Proposal]                                     │
│  - 동일 고정 모델 M이 제안자 역할                             │
│  - K개 병렬 후보 {(Δ_j, a_j)} 생성                         │
│  - 각 후보는 특정 실패 메커니즘 타겟팅                         │
│       ↓                                                 │
│  [Proposal Validation]                                  │
│  - 각 후보를 D_in, D_ho에서 회귀 테스트                       │
│  - 수락 조건 확인 → 수락/거부 결정                            │
│  - 수락된 편집들 병합 → h_{t+1}                             │
└─────────────────────────────────────────────────────────┘
```

**핵심 설계 원칙**: 모델 가중치($M$)와 평가자($\mathcal{E}$)는 고정, **하네스만 변경 대상**

**초기 하네스**: DeepAgent SDK 기반의 의도적으로 최소화된 구성 — 짧은 시스템 프롬프트, 기본 파일시스템/셸 도구

### 2.4 성능 향상

**Terminal-Bench-2.0** (64개 컨테이너화 태스크) 결과:

| 모델 | 분할 | 초기 Pass(%) | Self-Harness Pass(%) | 절대 향상 | 상대 향상 |
|------|------|:---:|:---:|:---:|:---:|
| MiniMax M2.5 | Held-in | 43.0 | 50.0 | +7.0pp | +16% |
| | **Held-out** | **40.5** | **61.9** | **+21.4pp** | **+53%** |
| Qwen3.5-35B-A3B | Held-in | 15.1 | 36.0 | +20.9pp | +138% |
| | **Held-out** | **23.8** | **38.1** | **+14.3pp** | **+60%** |
| GLM-5 | Held-in | 47.7 | 57.0 | +9.3pp | +20% |
| | **Held-out** | **42.9** | **57.1** | **+14.2pp** | **+33%** |

**모델별 수용된 하네스 편집**:

- **MiniMax M2.5**: ① 아웃풋 파일 조기 생성, ② 구조화된 콘텐츠 태그 올바른 사용, ③ 50회 도구 호출 후 리디렉션
- **Qwen3.5-35B-A3B**: ① 의존성 사전 확인, ② 탐색 루프 강제 전환, ③ 명령 재시도 억제, ④ 도구 오류 후 아티팩트 복구 미들웨어
- **GLM-5**: ① 쉘 세션 간 환경 변수 지속, ② 탐색에서 구현/테스트로의 전환 촉진

### 2.5 한계

| 한계 | 설명 |
|------|------|
| **벤치마크 특화** | 수락된 편집이 벤치마크 특유의 실패 패턴을 반영할 수 있음 (과적합 가능성) |
| **평가자 의존성** | 검증자 결과 및 트레이스 레코드의 품질에 의존 |
| **고정 벤치마크** | 오픈 엔드 자기개선이 아닌 고정 벤치마크 내 제한된 하네스 편집 |
| **수락 기준 약함** | 고위험 하네스 변경에는 Pass 비율 비회귀 이상의 더 강력한 검증 기준 필요 |
| **단일 벤치마크** | Terminal-Bench-2.0 외 다른 도메인에 대한 검증 미실시 |
| **모달리티 제한** | 멀티모달 입력을 요구하는 태스크는 평가에서 제외 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Held-out 성능: 핵심 일반화 증거

Self-Harness의 일반화 능력에 대한 가장 직접적인 증거는 **held-out 분할에서의 성능 향상**이다. 제안자에게 held-out 태스크의 실행 트레이스는 **전혀 노출되지 않았음**에도 불구하고:

$$\Delta_\text{ho}^{(\text{M2.5})} = 61.9\% - 40.5\% = +21.4\text{pp} \quad (+53\%)$$

$$\Delta_\text{ho}^{(\text{Qwen3.5})} = 38.1\% - 23.8\% = +14.3\text{pp} \quad (+60\%)$$

$$\Delta_\text{ho}^{(\text{GLM-5})} = 57.1\% - 42.9\% = +14.2\text{pp} \quad (+33\%)$$

이는 Self-Harness가 **관측된 실패 사례에만 과적합하는 것이 아니라, 재사용 가능한 실행 메커니즘 수준의 일반화 가능한 수정**을 학습함을 보여준다.

### 3.2 일반화를 가능하게 하는 설계 메커니즘

**① 실패 시그니처 기반 클러스터링**

단순히 개별 태스크 실패를 패치하지 않고, $\phi(r_i) = (c_i, q_i, m_i)$ 시그니처로 **동일한 메커니즘을 공유하는 실패들**을 집계함으로써, 제안된 수정은 특정 태스크가 아닌 재사용 가능한 실패 메커니즘을 타겟팅한다.

**② 최소성 제약 (Minimality Constraint)**

각 편집 $\Delta_j$는 선택된 메커니즘을 해결하는 데 필요한 표면만 수정하도록 제약된다:

$$\Delta_j: h_t \mapsto h_t^{(j)} \quad \text{s.t.} \quad |\text{modified surfaces}| \text{ is minimal}$$

이를 통해 관련 없는 하네스 동작을 보존하고 광범위한 재작성을 방지하여 과적합을 줄인다.

**③ 이중 분할 회귀 게이트**

수락 조건 $\Delta_\text{in}^{(j)} \geq 0 \wedge \Delta_\text{ho}^{(j)} \geq 0$은 held-in 향상을 held-out 회귀와 트레이드오프하는 편집을 **자동으로 거부**한다. 이는 일반화 능력을 직접적으로 보장하는 메커니즘이다.

**④ 증거 번들의 구조화**

$B_t$는 단순 실행 로그가 아닌 **구조화된 실패 패턴 요약**으로, 제안자가 개별 태스크 실패가 아닌 재사용 가능한 메커니즘에 대해 추론하도록 유도한다.

### 3.3 일반화의 질적 증거

**MiniMax M2.5 — 조기 아티팩트 생성**: 특정 태스크가 아닌 "요구되는 출력 아티팩트를 가능한 한 일찍 초기 버전으로 생성하라"는 일반적 실행 원칙으로 수정됨 → 다양한 태스크에서 아티팩트 누락 방지

**Qwen3.5 — 미들웨어 가드**: 도구 오류 트리거 시스템 프롬프트 미들웨어는 특정 파일 오류가 아닌 **도구 오류 후 아티팩트 복구**라는 일반적 전략을 인코딩

**GLM-5 — 세션 환경 지속**: "설치된 도구나 경로 변경이 쉘 세션 전반에 걸쳐 지속되도록 보장"은 특정 빌드 태스크가 아닌 모든 환경 수정 태스크에 적용 가능한 일반 원칙

### 3.4 잠재적 일반화 향상 방향

논문에서 언급된 바 및 설계에서 유추할 수 있는 일반화 개선 경로:

```
현재 Self-Harness의 일반화 범위
├── 동일 벤치마크 내 held-out 태스크 ✓ (실증됨)
├── 동일 도메인 내 유사 태스크 △ (가능성 있음, 미검증)
├── 다른 벤치마크 도메인 △ (구조적 메커니즘 일반화 가능성)
└── 완전히 다른 환경 ? (미검증)
```

**서브에이전트 기반 분해 및 미들웨어 생성**과 같은 더 넓은 구조적 메커니즘 도입(Qwen3.5 사례)은 단순 로컬 실패 수정을 넘어 문제 해결의 전반적 조직화를 개선할 가능성을 시사한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 미래 연구에 미치는 영향

**① 에이전트 엔지니어링 패러다임의 변화**

Self-Harness는 하네스 설계를 **"인간 예술"에서 "경험적 상태 전이"로** 재정의한다. 앞으로 에이전트 시스템 설계 연구는 "좋은 하네스를 어떻게 설계하는가"에서 "하네스가 어떻게 자기개선 조건을 갖추는가"로 초점이 이동할 것이다.

**② 자기개선 AI 시스템 연구의 실용적 경계 설정**

Darwin Gödel Machine, AlphaEvolve 등의 오픈 엔드 자기개선 연구와 달리, Self-Harness는 **제한된 편집 공간과 회귀 게이트 내에서의 통제된 자기개선**이 실용적으로 작동함을 보인다. 이는 안전성을 유지하면서 자기개선 능력을 구현하는 균형점을 제시한다.

**③ 모델-하네스 공동 최적화 연구의 촉발**

현재는 모델 파인튜닝과 하네스 설계가 독립적으로 진행되지만, Self-Harness는 **하네스를 모델의 행동 특성에 반응적으로 적응시키는 공동 최적화 방향**을 제시한다.

**④ 에이전트 평가 방법론에 대한 시사점**

실패 패턴의 클러스터링과 검증자 기반 귀인(attribution)은 에이전트 실패 분석의 새로운 방법론적 기여이다. 향후 에이전트 벤치마크 연구에서 단순 통과/실패 외에 **실패 메커니즘 분류**가 중요해질 것이다.

### 4.2 앞으로 연구 시 고려할 점

**① 과적합 vs. 일반화의 더 정밀한 측정**

현재 held-out 분할은 동일 벤치마크 내에 있어, 진정한 분포 외(out-of-distribution) 일반화를 측정하지 못한다. 향후 연구는:

$$\text{일반화 갭} = P_\text{held-out}(h_T) - P_\text{cross-domain}(h_T)$$

를 측정하는 크로스 벤치마크 평가를 설계해야 한다.

**② 수락 기준의 정교화**

현재 수락 조건은 Pass 비율 비회귀만을 요구하지만, 더 강력한 기준이 필요하다:

$$\text{통계적 유의성}: \quad p\text{-value}(\Delta_\text{ho}^{(j)} > 0) < \alpha$$

$$\text{이펙트 크기}: \quad \frac{\Delta_\text{ho}^{(j)}}{\sigma_\text{baseline}} > \tau_\text{min}$$

**③ 하네스 편집의 구성 가능성(Compositionality) 연구**

여러 편집이 병합될 때 예상치 못한 상호작용이 발생할 수 있다. 편집 $\Delta_i$와 $\Delta_j$가 각각 독립적으로 수락되었어도:

$$\Delta_\text{in}^{(i+j)} \neq \Delta_\text{in}^{(i)} + \Delta_\text{in}^{(j)}$$

편집 간 상호작용 모델링이 필요하다.

**④ 편집 공간의 범위 확대**

현재는 선언된 구성 포인트(instruction, tools, verification guidance 등)만 편집 가능하다. 향후 연구는 **코드 수준 미들웨어, 동적 도구 합성, 멀티에이전트 오케스트레이션** 등 더 넓은 편집 공간을 탐색해야 한다.

**⑤ 안전성 고려사항**

자기개선 하네스는 예상치 못한 방향으로 진화할 수 있다:
- **탈출 행동**: 하네스가 평가 게이트를 우회하는 방법을 학습할 위험
- **벤치마크 해킹**: 평가자를 속이는 방식으로 Pass 비율을 높이는 전략 진화
- **되돌릴 수 없는 변경**: 하네스 편집의 감사 가능성(auditability)과 가역성(reversibility) 보장 메커니즘 강화 필요

**⑥ 계산 비용 최적화**

각 반복에서 $K$개의 병렬 후보를 평가하는 데 상당한 계산 비용이 든다. 향후 연구는:

$$\text{비용} = T \times K \times |D_\text{in} \cup D_\text{ho}| \times C_\text{eval}$$

를 줄이기 위한 **능동 학습(active learning) 기반 후보 선택** 또는 **서로게이트 모델 기반 사전 필터링**을 고려해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 자기개선 에이전트 연구 계보

| 연구 | 개선 대상 | 외부 의존성 | 일반화 | 하네스 수정 |
|------|----------|:---:|:---:|:---:|
| **Reflexion** (Shinn et al., 2023) | 응답 전략/메모리 | 없음 | 제한적 | ✗ |
| **STOP** (Zelikman et al., 2024) | 코드 생성 프로그램 | 없음 | 코드 도메인 | ✗ |
| **ADAS** (Hu et al., 2025) | 에이전트 설계 | 외부 검색 | 가능 | △ |
| **Meta-Harness** (Lee et al., 2026) | 하네스 코드 | 더 강력한 에이전트 | 가능 | ✓ |
| **Agentic Harness Engineering** (Lin et al., 2026) | 코딩 에이전트 하네스 | 관찰 가능성 도구 | 코딩 도메인 | ✓ |
| **Self-Harness** (Zhang et al., 2026) | 하네스 전체 | **없음** | **Held-out 검증** | **✓** |

### 5.2 핵심 차별화 분석

**Reflexion vs. Self-Harness**

Reflexion은 언어 피드백을 메모리에 저장하여 이후 시도를 개선하지만, 개선 대상이 **응답 전략/컨텍스트**이지 선언된 하네스 상태가 아니다. Self-Harness는 실행 프로토콜 자체를 변경하므로, 새로운 태스크 세션에서도 개선 효과가 지속된다.

**Meta-Harness vs. Self-Harness**

$$\text{Meta-Harness}: M_\text{strong} \xrightarrow{\text{최적화}} h_\text{target}$$

$$\text{Self-Harness}: M_\text{fixed} \xrightarrow{\text{자기개선}} h_\text{updated}$$

Meta-Harness는 더 강력한 외부 에이전트를 요구하므로 프론티어 모델에는 적용 불가하거나 비용이 높다. Self-Harness는 동일 모델이 자신의 하네스를 개선하므로 **외부 의존성이 없다**.

**ADAS (Automated Design of Agentic Systems) vs. Self-Harness**

ADAS는 에이전트 설계를 탐색 가능한 공간으로 취급하는 **외부 최적화 과정**이지만, Self-Harness는 평가 중인 모델이 현재 하네스 하에서 제안하는 **제한된 편집**이다. 이는 Self-Harness의 편집이 해석 가능하고 감사 가능하다는 장점을 낳는다.

**Agentic Context Engineering (Zhang et al., ICLR 2026) vs. Self-Harness**

컨텍스트 엔지니어링은 이후 모델 호출을 위한 컨텍스트를 진화시키지만, 하네스 상태(도구 정의, 런타임 정책, 검증 규칙)는 수정하지 않는다. Self-Harness는 더 깊은 수준의 실행 프로토콜을 변경한다.

### 5.3 관련 연구의 공통 트렌드 및 Self-Harness의 위치

```
자기개선 AI 연구의 스펙트럼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
컨텍스트/메모리    하네스/스캐폴딩    에이전트 설계    자유로운 자기개선
개선               개선               최적화           (오픈 엔드)
Reflexion          [Self-Harness]     ADAS             Darwin Gödel
STOP               Meta-Harness       Lang Graph       Machine
ACE                AHE                Optim.           AlphaEvolve
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
← 더 안전, 제어됨                          더 강력, 위험 →
```

Self-Harness는 **통제된 안전성**과 **실질적 자기개선 능력** 사이의 균형점을 찾은 연구로, 향후 실용적 에이전트 엔지니어링의 중요한 기준점이 될 것으로 예상된다.

---

## 참고 자료

1. **Zhang, H., et al.** (2026). "Self-Harness: Harnesses That Improve Themselves." arXiv:2606.09498v1. Shanghai Artificial Intelligence Laboratory.
2. **Shinn, N., et al.** (2023). "Reflexion: Language agents with verbal reinforcement learning." arXiv:2303.11366.
3. **Zelikman, E., et al.** (2024). "Self-taught optimizer (STOP): Recursively self-improving code generation." arXiv:2310.02304.
4. **Lee, Y., et al.** (2026). "Meta-harness: End-to-end optimization of model harnesses." arXiv:2603.28052.
5. **Hu, S., Lu, C., & Clune, J.** (2025). "Automated design of agentic systems." arXiv:2408.08435.
6. **Lin, J., et al.** (2026). "Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses." arXiv:2604.25850.
7. **Zhang, Q., et al.** (2026). "Agentic context engineering: Evolving contexts for self-improving language models." arXiv:2510.04618. ICLR 2026.
8. **Yang, J., et al.** (2024). "SWE-agent: Agent-computer interfaces enable automated software engineering." arXiv:2405.15793.
9. **Yao, S., et al.** (2023). "ReAct: Synergizing reasoning and acting in language models." arXiv:2210.03629.
10. **Merrill, M.A., et al.** (2026). "Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces." arXiv:2601.11868.
11. **Novikov, A., et al.** (2025). "AlphaEvolve: A coding agent for scientific and algorithmic discovery." arXiv:2506.13131.
12. **Zhang, J., et al.** (2025). "Darwin Gödel Machine: Open-ended evolution of self-improving agents." arXiv:2505.22954.
13. **Yin, X., et al.** (2025). "Gödel Agent: A self-referential agent framework for recursively self-improvement." ACL 2025. doi:10.18653/v1/2025.acl-long.1354.
14. **Zhuge, M., et al.** (2024). "Language agents as optimizable graphs." arXiv:2402.16823.
15. **Sclar, M., et al.** (2024). "Quantifying language models' sensitivity to spurious features in prompt design." arXiv:2310.11324.
