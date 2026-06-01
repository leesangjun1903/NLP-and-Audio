# Scaling Laws for Agent Harnesses via Effective Feedback Compute

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다: **에이전트 하네스(Agent Harness)의 성능은 원시 계산량(raw compute)이 아니라, 유효 피드백 컴퓨트(Effective Feedback Compute, EFC)에 의해 결정된다.**

기존 연구들은 테스트 타임 스케일링을 토큰 수, 툴 호출 횟수, 실행 시간, 비용 등 원시 지출(raw expenditure)로 측정했습니다. 그러나 동일한 원시 예산을 가진 두 trajectory가 성능에서 크게 차이날 수 있으며, 그 이유는 피드백의 **질(quality)**에 있습니다.

### 주요 기여 (세 가지)

**(i) EFC의 공식화:** 유효 피드백을 측정하는 trace-level 스케일링 좌표를 형식화하고, oracle 접근이 불가능한 환경을 위한 Estimated-EFC 및 NRS-EFC를 제안합니다.

**(ii) EFC의 우월성 실증:** EFC 및 태스크 수요 정규화 EFC가 원시 컴퓨트 기준선 및 강력한 다변량 SAS 기준선을 일관적으로 능가함을 보입니다.

**(iii) 하네스 스케일링 분해:** 하네스 효율성($\eta$)과 태스크 수요($D_{\text{task}}$)라는 두 메커니즘으로 분해하여, 성공적인 하네스는 원시 예산을 유효 피드백으로 효율적으로 변환하면서 동시에 태스크 요구 수준을 충족해야 함을 보입니다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

언어 모델 기반 에이전트 시스템에서 **하네스 설계가 테스트 타임 스케일링의 핵심**이 되었습니다. 그러나 기존 분석은 다음과 같은 근본적 문제를 가집니다:

- 원시 토큰 수, 툴 호출 횟수 등이 **유용한 피드백과 무관한 중복/불안정 상호작용을 구별하지 못함**
- 동일한 원시 예산으로도 성능이 크게 달라질 수 있음
- 태스크 간 피드백 요구량이 다름에도 이를 정규화하는 좌표가 없음

**중심 질문:** 폐쇄 루프 에이전트 하네스 성능을 위한 스케일링 좌표로 어떤 양(quantity)을 사용해야 하는가?

---

### 2.2 문제 정의 및 수식

#### 에이전트 하네스의 폐쇄 루프 계산

태스크 인스턴스 $x \sim \mathcal{T}$에 대해, 하네스 $h \in \mathcal{H}$와 기반 모델 $m$이 생성하는 trajectory:

$$\tau = \{(s_t, a_t, o_t, u_t)\}_{t=1}^T \tag{1}$$

- $s_t$: step $t$ 이전 에이전트 상태
- $a_t$: 모델 액션 또는 툴 호출
- $o_t$: 결과 관측값
- $u_t$: 에이전트 상태, 메모리, 플랜, 후보 솔루션에 대한 하네스 업데이트

성공률과 실패율:

$$S(x, h, m, b) = \mathbb{E}[g_x(\hat{y})], \quad E(x, h, m, b) = 1 - S(x, h, m, b) \tag{2}$$

---

#### EFC의 이벤트 수준 정의

각 피드백 이벤트 $e_t$는 네 가지 제한된 인수를 받습니다:

$$I_t, V_t, R_t, M_t \in [0, 1] \tag{4}$$

| 인수 | 의미 |
|------|------|
| $I_t$ (Informativeness) | 태스크 관련 정보를 새롭게 드러내는가 |
| $V_t$ (Validity) | 결정론적 체커, 실행 결과, 유닛 테스트 등 신뢰할 수 있는 증거에 기반하는가 |
| $R_t$ (Non-redundant Relevance) | 현재 서브골을 다루고, 기존 trajectory에 없는 정보를 추가하는가 |
| $M_t$ (Memory Update) | 이후 액션에 영향을 줄 수 있도록 플랜/상태/메모리를 변경하는가 |

이벤트 기여도 ($\kappa = 10$):

$$\text{EFC}_t = \kappa I_t V_t R_t M_t \tag{5}$$

런 수준 EFC:

$$\text{EFC}(\tau) = \sum_{t=1}^{T_{\text{fb}}} \text{EFC}_t = \kappa \sum_{t=1}^{T_{\text{fb}}} I_t V_t R_t M_t \tag{6}$$

**곱 형태의 의의:** 하나의 인수라도 낮으면 전체 기여가 급감하는 병목(bottleneck) 효과를 가집니다. 즉, 유효하지 않거나, 중복되거나, 보존되지 않는 피드백은 거의 기여하지 않습니다.

---

#### Oracle-EFC vs. Estimated-EFC

**Oracle-EFC:** 합성 제어 태스크에서 숨겨진 상태와 ground-truth progress로부터 직접 계산

**Estimated-EFC:** 실제 태스크에서는 숨겨진 상태에 접근 불가하므로, trace-observable 특성 벡터로 추정:

$$\phi(e_t) = [c_t, h_t, z_t, p_t, m_t, a_t, q_t, \Delta_t, \rho_t] \tag{7}$$

이벤트 수준 추정기:

$$\widehat{\text{EFC}}_t = \max\left(0,\ \exp\left(\theta_0 + \theta^\top \phi(e_t)\right) - 1\right) \tag{8}$$

런 수준 추정:

$$\widehat{\text{EFC}}(\tau) = \sum_{t=1}^{T_{\text{fb}}} \widehat{\text{EFC}}_t \tag{9}$$

**실제 실행 trace를 위한 상태 인식 변형 (Stable-EFC):**

$$\widehat{\text{EFC}}_t^{\text{stable}} = \widehat{\text{EFC}}_t \cdot Q_t \cdot G_t \cdot \Lambda_t \tag{10}$$

**비중복 안정 변형 (NRS-EFC):** 반복 실패를 더 강하게 할인:

$$\widehat{\text{EFC}}_t^{\text{nr}} = \frac{\widehat{\text{EFC}}_t \cdot Q_t \cdot G_t^{\text{nr}} \cdot \Lambda_t^{\text{nr}}}{1 + 0.35 A_t} \tag{11}$$

여기서 $A_t$는 시도 인덱스입니다.

---

#### 태스크 수요 정규화

서로 다른 피드백 요구량을 가진 태스크들을 비교하기 위한 태스크 수요 정규화:

$$D_{\text{task}} = L \cdot H_{\text{tool}} \cdot S_{\text{state}} \cdot (1 + N_{\text{obs}}) \cdot (1 - V_{\text{oracle}}) \tag{12}$$

| 항목 | 의미 |
|------|------|
| $L$ | 최소 추론/액션 단계 수 |
| $H_{\text{tool}}$ | 툴 선택 엔트로피 |
| $S_{\text{state}}$ | 상태 추적 요구량 |
| $N_{\text{obs}}$ | 관측 노이즈/모호성 |
| $V_{\text{oracle}}$ | 검증 신호 가시성 (높을수록 수요 감소) |

정규화 변수:

$$X = \frac{\text{EFC}}{D_{\text{task}}}, \quad \hat{X} = \frac{\widehat{\text{EFC}}}{D_{\text{task}}} \tag{13}$$

하네스 효율성:

$$\eta = \frac{\text{EFC}}{C_{\text{raw}}}, \quad \hat{\eta} = \frac{\widehat{\text{EFC}}}{C_{\text{raw}}} \tag{14}$$

---

#### 스케일링 모델

모든 스케일링 분석에 사용하는 공통 거듭제곱 법칙 실패 모델:

$$E(z) = E_\infty + A z^{-\alpha} \tag{15}$$

- $E(z)$: 예측 실패율
- $E_\infty$: 환원 불가 오류
- $A$: 스케일 파라미터
- $\alpha$: 스케일링 지수

평가 지표 $R^2$:

$$R^2 = 1 - \frac{\sum_i (\bar{E}_i - \hat{E}_i)^2}{\sum_i (\bar{E}_i - \bar{E})^2} \tag{17}$$

---

### 2.3 모델 구조

논문은 7가지 하네스 패밀리(H0–H6)를 설계하여 실험합니다:

| 하네스 | 이름 | 특징 |
|--------|------|------|
| H0 | Direct Answer | 단일 패스, 피드백 없음 |
| H1 | Checklist Verify | 경량 검증 추가 |
| H2 | Routed Tools | 툴 라우팅 도입 |
| H3 | Stateful Memory | 상태 유지 메모리 강조 |
| H4 | High Budget Noisy | 고예산이지만 낮은 품질 (음성 대조군) |
| H5 | Closed Loop | 라우팅+검증+메모리 통합 |
| H6 | Deep Closed Loop | H5 확장, 더 깊은 상호작용 |

**3개의 태스크 레이어**로 평가:
1. **합성 제어 태스크** (Needle Lookup, State Tracking, Rule Filter)
2. **반현실적 실행 가능 태스크** (HumanEval-style 코드)
3. **실제 벤치마크** (HumanEval, TerminalBench 2.0, SWE-bench Verified)

**평가 모델:** DeepSeek-V4-Flash, gpt-5.4-nano, Claude-Haiku-4.5

---

### 2.4 성능 향상

| 스케일링 좌표 | $R^2$ (제어 실험) | $R^2$ (실제 trace) | $R^2$ (전향적 검증) |
|--------------|-------------------|-------------------|---------------------|
| Raw Tokens | 0.33 | -0.08 | -0.11 |
| Tool Calls | 0.42 | -0.02 | -0.04 |
| Raw Cost | 0.38 | -0.07 | -0.09 |
| SAS | 0.88 | 0.43 | 0.26 |
| Oracle-EFC | 0.94 | — | — |
| Estimated-EFC | 0.94 | — | — |
| NRS-EFC | — | 0.89 | 0.77 |
| Oracle-EFC/ $D_{\text{task}}$ | **0.99** | — | — |
| NRS-EFC/ $D_{\text{task}}$ | — | **0.92** | **0.85** |

**매칭 예산 개입 실험 결과:**
- 동일 원시 예산 하에서 피드백 품질만 개선 → 성공률 $0.27 \to 0.90$ ($p = 1.0 \times 10^{-300}$)
- 하네스 효율성이 성공률을 거의 완전히 설명 ($R^2 = 0.97$), 반면 원시 비용은 거의 설명 못함 ($R^2 = 0.01$)

---

### 2.5 한계

1. **EFC 인수($I, V, R, M$)의 수동 설계 의존성:** 일부 인수는 hand-crafted 공식에 의존하며, 열린 환경에서 자동화가 어렵습니다.
2. **Oracle-EFC의 제한적 적용 범위:** Oracle-EFC는 합성 제어 태스크에서만 측정 가능하며, 실제 환경에서는 Estimated-EFC로 대체해야 합니다.
3. **$D_{\text{task}}$ 보정 필요성:** 이질적인 태스크 혼합 환경에서는 수동 설계된 $D_{\text{task}}$가 효과가 떨어지며, 보정(calibration)이 필수적입니다.
4. **Terminal 태스크의 낮은 효율성:** 터미널 상호작용 태스크에서 모든 하네스가 낮은 $\eta \approx 0.1$을 보이며, 해당 환경에서 EFC를 높이는 방법이 명확하지 않습니다.
5. **모델 특화 검증 부재:** DeepSeek, GPT, Claude 계열에 대해서만 실험하여 더 넓은 모델군으로의 일반화 여부가 불확실합니다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 EFC가 일반화 성능을 향상시키는 메커니즘

EFC 프레임워크는 모델의 일반화 성능 향상과 직접적으로 연결되는 여러 메커니즘을 제시합니다.

#### (a) 태스크 수요 정규화를 통한 도메인 일반화

$\text{EFC}/D_{\text{task}}$ 좌표는 서로 다른 태스크 패밀리 간의 스케일 불일치를 제거합니다:

- Oracle-EFC만 사용 시 ($R^2 = 0.90$): 태스크 패밀리 간 잔여 오프셋 존재
- Oracle-EFC/ $D_{\text{task}}$ 사용 시 ($R^2 = 0.96$): 패밀리 간 불일치 제거

이는 동일한 EFC 양이라도 태스크 요구량 대비 충분성이 성능을 결정한다는 것을 의미하며, **태스크 독립적인 일반화 좌표**를 제공합니다.

#### (b) 비라벨 trace로부터의 추정 가능성

Estimated-EFC는 최종 성공 레이블을 사용하지 않고, 순수히 trace-time 신호로부터 Oracle-EFC의 대부분을 복원합니다:

$$\widehat{\text{EFC}}_t = \max\left(0, \exp(\theta_0 + \theta^\top \phi(e_t)) - 1\right)$$

이는 **레이블이 없는 새로운 태스크**에도 EFC 좌표를 적용할 수 있음을 의미합니다. Held-out 실험에서 Estimated-EFC/ $D_{\text{task}}$는 $R^2 = 0.93$으로, Oracle에 근접하는 성능을 보였습니다.

#### (c) 보정된 태스크 수요의 이전 가능성 (Transfer)

이질적인 태스크 혼합에서 보정된 $D_{\text{task}}$ 지수를 학습하면 미관측 태스크에도 일반화됩니다:

- Raw compute: $R^2 = -0.42$ (전이 실패)
- NRS-EFC: $R^2 = 0.70$
- Fitted NRS-EFC/ $D_{\text{task}}$: $R^2 = 0.83$ (최선)

**보정 가중치 예시:**
- 툴 엔트로피($H_{\text{tool}}$)를 상향 가중
- 추론 깊이($L$)와 상태 압력($S_{\text{state}}$)을 하향 가중

이는 EFC 프레임워크가 **태스크 분포 이동(distribution shift)에 강건한** 일반화 특성을 가짐을 시사합니다.

#### (d) 모듈별 기여도와 일반화의 관계

하네스 모듈 제거 실험에서 $\eta$가 성공률을 거의 완전히 설명함($R^2 = 0.97$):

$$\text{성공률} \approx f(\eta), \quad \eta = \frac{\text{EFC}}{C_{\text{raw}}}$$

이는 **라우터 품질, 검증기 강도, 메모리 신뢰도**를 높이면 동일한 raw 예산으로 더 높은 일반화 성능을 달성할 수 있음을 의미합니다.

#### (e) 하네스 효율성의 태스크 종속성

실제 trace 분석에서 $\eta$가 슬라이스별로 다르게 나타남:

| 태스크 슬라이스 | 최고 성능 하네스 | $\eta$ |
|----------------|----------------|--------|
| HumanEval | H5, H6 | $\approx 1.9$ |
| Terminal | 모두 낮음 | $\approx 0.1$ |
| SWE micro | H0, H3 | 중간 |

이는 일반화 향상을 위해 **하네스를 태스크 구조에 적응시켜야 함**을 시사합니다. EFC 프레임워크는 이러한 적응의 방향성을 정량적으로 제시합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (a) 에이전트 시스템 설계 패러다임 전환
이 논문은 **"더 많은 컴퓨트 = 더 나은 성능"이라는 단순 가정을 명시적으로 반박**합니다. 앞으로의 에이전트 시스템 연구는 원시 예산 최적화보다 **피드백 품질 최적화**에 초점을 맞춰야 합니다.

#### (b) 적응형 예산 할당 연구
EFC를 목적 함수로 사용하는 **EFC-guided 적응형 예산 할당(adaptive budget allocation)** 연구로 이어질 수 있습니다. 즉, 피드백이 유효하고 비중복적인 경우에만 추가 예산을 할당하는 방식입니다.

#### (c) 하네스 자동 설계(AutoHarness) 연구
EFC 효율성 $\eta$를 최대화하는 방향으로 하네스 구성 요소(라우터, 검증기, 메모리)를 자동 탐색하는 **Neural Architecture Search** 유사 접근법이 가능해집니다.

#### (d) 멀티에이전트 시스템으로의 확장
SAS (Kim et al., 2026b)와의 비교에서 EFC가 우월함을 보였으나, **멀티에이전트 협력 시스템**에서 에이전트 간 피드백 교환을 EFC로 모델링하는 연구가 필요합니다.

#### (e) 강화학습과의 통합
EFC 개념은 RL에서의 **보상 신호 품질(reward signal quality)**과 유사합니다. EFC를 intrinsic reward로 활용하거나, trajectory-level reward shaping에 적용하는 연구로 확장될 수 있습니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (a) EFC 추정의 자동화
현재 Estimated-EFC는 수동으로 설계된 특성 벡터 $\phi(e_t)$에 의존합니다. 더 복잡한 환경에서는:
- **LLM 기반 자동 EFC 추정기** 개발 필요
- $I_t, V_t, R_t, M_t$ 인수의 자동 측정 방법 연구

#### (b) $D_{\text{task}}$ 보정의 어려움
이질적 태스크 혼합에서는 fitted $D_{\text{task}}$가 필요하나, 보정 데이터의 품질과 양이 중요합니다:
- **메타 학습(meta-learning) 방식**으로 소수의 보정 데이터만으로 $D_{\text{task}}$ 지수를 추정하는 방법 필요
- 보정 태스크와 평가 태스크 간의 분포 차이 관리

#### (c) 열린 환경(Open-ended Environment)으로의 확장
현재 EFC는 자동 평가기(automatic evaluator)가 있는 태스크에 최적화됩니다:
- **주관적 평가가 필요한 태스크**(창의적 글쓰기, 장기 대화 등)에서의 EFC 적용 방법 연구
- 인간 피드백(RLHF)을 EFC 프레임워크에 통합하는 방법

#### (d) 효율성-품질 트레이드오프 관리
높은 EFC를 달성하기 위해 강력한 검증기와 메모리 시스템이 필요하지만, 이는 추가적인 계산 비용을 수반합니다:
- $\eta = \text{EFC}/C_{\text{raw}}$를 최대화하면서 절대 EFC도 충분히 확보하는 **파레토 최적 설계** 탐색 필요

#### (e) 인과성 vs. 상관성 문제
EFC가 강한 예측력($R^2 = 0.99$)을 보이지만, **인과 관계**를 더 엄밀하게 검증해야 합니다:
- 매칭 예산 개입 실험이 이를 일부 지지하지만, 더 다양한 환경에서의 반사실적(counterfactual) 실험 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 스케일링 법칙 연구

| 연구 | 핵심 스케일링 좌표 | EFC 논문과의 관계 |
|------|-------------------|--------------------|
| Kaplan et al. (2020) *"Scaling Laws for Neural Language Models"* | 모델 파라미터, 데이터, 훈련 컴퓨트 | 사전학습 스케일링의 기반. EFC는 이를 테스트 타임으로 확장 |
| Hoffmann et al. (2022) *"Training Compute-Optimal Large Language Models"* (Chinchilla) | 최적 모델-데이터 균형 | 훈련 효율성 개념을 EFC는 추론 효율성으로 전환 |
| Snell et al. (2025) *"Scaling LLM Test-Time Compute Optimally"* | 테스트 타임 토큰/샘플 수 | EFC는 이 접근법이 "얼마나 쓰나"가 아닌 "얼마나 유효하게 쓰나"를 주목해야 함을 비판 |
| Brown et al. (2025) *"Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"* | 반복 샘플링 횟수 | 반복 자체보다 유효 피드백의 누적이 더 중요함을 EFC가 보임 |

### 5.2 에이전트 하네스 및 피드백 연구

| 연구 | 핵심 기여 | EFC 논문과의 비교 |
|------|-----------|------------------|
| Yao et al. (2023b) *"ReAct: Synergizing Reasoning and Acting"* | 추론-행동 통합 루프 | EFC는 ReAct 스타일 루프에서 유효 피드백을 정량화하는 도구 제공 |
| Shinn et al. (2023) *"Reflexion"* | 언어 강화 학습, 자기 반성 | 자기 반성의 효과를 EFC의 $I_t, M_t$ 인수로 설명 가능 |
| Madaan et al. (2023) *"Self-Refine: Iterative Refinement with Self-Feedback"* | 자기 피드백 반복 개선 | EFC는 자기 피드백의 유효성($V_t$)과 비중복성($R_t$)이 핵심임을 지적 |
| Lightman et al. (2024) *"Let's Verify Step by Step"* | 과정 보상 모델(PRM) | 단계별 검증의 유효성이 EFC의 $V_t$ 인수와 직접 연결 |
| Yao et al. (2023a) *"Tree of Thoughts"* | 트리 탐색 기반 추론 | 탐색 vs. 착취 트레이드오프를 EFC 효율성으로 분석 가능 |
| Zhou et al. (2024) *"Language Agent Tree Search (LATS)"* | 통합된 추론-행동-계획 | 트리 탐색의 각 노드에서 EFC 기여를 평가하는 연구로 확장 가능 |

### 5.3 시스템 수준 스케일링 연구

| 연구 | 핵심 기여 | EFC 논문과의 비교 |
|------|-----------|------------------|
| Kim et al. (2026b) *"Towards a Science of Scaling Agent Systems" (SAS)* | 다변량 시스템 수준 스케일링 | EFC 논문의 직접 비교 대상. SAS($R^2 = 0.88$) < EFC($R^2 = 0.94$) |
| Kim et al. (2026a) *"Scaling Test-Time Compute for Agentic Coding"* | 코딩 에이전트 테스트 타임 스케일링 | EFC가 코딩 태스크에서도 우월함을 HumanEval 실험으로 확인 |
| Li et al. (2026b) *"Benchmark Test-Time Scaling of General LLM Agents"* | 범용 에이전트 벤치마크 스케일링 | EFC와 보완적: 벤치마크 성능 측정 vs. 스케일링 좌표 정의 |
| Zhu et al. (2025) *"Scaling Test-Time Compute for LLM Agents"* | 테스트 타임 컴퓨트 스케일링 | 원시 컴퓨트 기반 접근의 한계를 EFC가 정량적으로 보임 |

### 5.4 비교 분석 요약

```
스케일링 연구의 진화:
사전학습 스케일링 → 테스트타임 스케일링 → 에이전트 하네스 스케일링
(Kaplan 2020)      (Snell 2025)           (Zhang et al. 2026, 본 논문)
     ↓                   ↓                        ↓
파라미터/데이터/컴퓨트   샘플수/토큰수          EFC/D_task
```

**EFC 논문의 차별점:**
- 기존 연구들이 **"얼마나 계산하는가"**에 집중할 때, EFC는 **"얼마나 유효하게 계산하는가"**를 측정
- trace-level 단일 스칼라로 하네스 성능을 예측하는 최초의 체계적 접근
- 전향적 검증(prospective validation)을 통해 사후 적합이 아님을 보증

---

## 참고자료

1. **Zhang et al. (2026)** - *"Scaling Laws for Agent Harnesses via Effective Feedback Compute"* (arXiv:2605.29682v1) — 본 논문
2. **Kaplan et al. (2020)** - *"Scaling Laws for Neural Language Models"* (arXiv:2001.08361)
3. **Hoffmann et al. (2022)** - *"Training Compute-Optimal Large Language Models"* (NeurIPS 2022)
4. **Brown et al. (2025)** - *"Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"* (OpenReview)
5. **Snell et al. (2025)** - *"Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Parameters for Reasoning"* (ICLR 2025)
6. **Kim et al. (2026a)** - *"Scaling Test-Time Compute for Agentic Coding"* (arXiv:2604.16529)
7. **Kim et al. (2026b)** - *"Towards a Science of Scaling Agent Systems"* (arXiv:2512.08296)
8. **Yao et al. (2023a)** - *"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"* (NeurIPS 2023)
9. **Yao et al. (2023b)** - *"ReAct: Synergizing Reasoning and Acting in Language Models"* (ICLR 2023)
10. **Shinn et al. (2023)** - *"Reflexion: Language Agents with Verbal Reinforcement Learning"* (NeurIPS 2023)
11. **Madaan et al. (2023)** - *"Self-Refine: Iterative Refinement with Self-Feedback"* (NeurIPS 2023)
12. **Lightman et al. (2024)** - *"Let's Verify Step by Step"* (ICLR 2024)
13. **Zhou et al. (2024)** - *"Language Agent Tree Search Unifies Reasoning Acting and Planning"* (ICLR 2024)
14. **Chen et al. (2021)** - *"Evaluating Large Language Models Trained on Code"* (arXiv:2107.03374) — HumanEval
15. **Jimenez et al. (2024)** - *"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"* (ICLR 2024)
16. **Merrill et al. (2026)** - *"Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces"* (ICLR 2026)
17. **Zhu et al. (2025)** - *"Scaling Test-Time Compute for LLM Agents"* (arXiv:2506.12928)
18. **Li et al. (2026b)** - *"Benchmark Test-Time Scaling of General LLM Agents"* (arXiv:2602.18998)
