# Discovering Multiagent Learning Algorithms with Large Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **다중 에이전트 강화학습(MARL)** 알고리즘의 설계를 인간의 직관과 시행착오에 의존하는 수동적 과정에서, **대규모 언어 모델(LLM) 기반 진화적 코드 탐색**을 통한 자동 발견(automated discovery)으로 전환할 것을 제안한다. 핵심 주장은 다음과 같다:

1. **LLM 기반 진화(AlphaEvolve)**를 활용하면 단순 하이퍼파라미터 튜닝을 넘어, 알고리즘의 소스 코드 자체를 진화시켜 인간이 직관적으로 설계하기 어려운 새로운 메커니즘을 발견할 수 있다.
2. 이 프레임워크를 불완전 정보 게임 풀이의 두 주요 패러다임—**CFR(Counterfactual Regret Minimization)**과 **PSRO(Policy Space Response Oracles)**—에 적용하여, 각각 **VAD-CFR**과 **SHOR-PSRO**라는 새로운 알고리즘 변종을 발견하였다.
3. 발견된 알고리즘들은 훈련에 사용하지 않은 테스트 게임에서도 기존 SOTA 대비 우수하거나 동등한 성능을 보이며, **일반화 능력(generalization)**을 입증하였다.

**주요 기여:**
- LLM 기반 심볼릭 코드 진화를 통한 다중 에이전트 학습 알고리즘 자동 설계 프레임워크 제시
- VAD-CFR: 변동성 적응형 할인, 비대칭 즉시 부스팅, 하드 웜스타트 등 비직관적 메커니즘의 발견
- SHOR-PSRO: Optimistic Regret Matching과 Smoothed Best Pure Strategy의 하이브리드 블렌딩 및 동적 어닐링 메커니즘 발견
- 11개 게임에 걸친 광범위한 실험적 검증

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

MARL에서 CFR과 PSRO는 이론적으로 견고하지만, 실제 성능을 좌우하는 핵심 설계 선택(예: 후회값 할인 방식, 메타 전략 솔버 구성)은 **방대한 조합적 설계 공간** 내에서 인간의 직관에 의존하여 수동으로 결정되어 왔다. 이 논문은 이러한 수동 설계 과정 자체를 자동화하는 것을 목표로 한다.

### 2.2 게임 이론적 기초 및 성능 지표

$N$-플레이어 확장형 게임(Extensive-Form Game)에서 **내쉬 균형(Nash Equilibrium)** $\sigma^*$은 다음을 만족하는 전략 프로파일이다:

```math
u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i', \sigma_{-i}^*) \quad \forall \sigma_i', \forall i \in \mathcal{N}
```

성능 지표로 사용되는 **착취가능성(Exploitability)**은 다음과 같이 정의된다:

$$\text{Exploitability}(\sigma) = \frac{1}{N} \sum_{i \in \mathcal{N}} \left( \max_{\sigma_i'} u_i(\sigma_i', \sigma_{-i}) - u_i(\sigma) \right)$$

### 2.3 제안하는 방법: AlphaEvolve 프레임워크

#### 2.3.1 전체 구조

AlphaEvolve는 LLM의 코드 생성 능력과 진화 알고리즘의 선택 압력을 결합한다:

- **개체군 초기화**: 기준 알고리즘(예: CFR+, Uniform PSRO)의 코드로 초기화
- **LLM 기반 돌연변이**: 적합도 기반으로 부모 알고리즘 $A$를 선택 후, LLM(Gemini 2.5 Pro)에 코드를 입력하여 의미론적으로 유의미한 변형 $A'$를 생성
- **자동 평가**: 프록시 게임 세트에서 $A'$를 실행하여 적합도(음의 착취가능성) 계산
- **진화적 선택**: 유효한 후보를 개체군에 추가

#### 2.3.2 최적화 목표

훈련 게임 집합 $\mathcal{G}$에 대해 다음을 최적화한다:

$$-\frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \text{Exploitability}(A(g)_K)$$

여기서 $A(g)_K$는 알고리즘 $A$가 게임 $g$에서 $K$ 이터레이션 후 생성한 전략이다.

### 2.4 CFR 기초 수식 및 VAD-CFR

#### 표준 CFR 수식

정보집합 $I$에서 행동 $a$의 **반사실적 가치(Counterfactual Value)**:

$$v_i(\sigma, I, a) = \sum_{h \in I} \pi_{-i}^{\sigma}(h) \sum_{z \in \mathcal{Z}, h \sqsubset z} \pi^{\sigma}(z \mid h, a) u_i(z)$$

이터레이션 $t$에서의 **즉시 반사실적 후회(Instantaneous Counterfactual Regret)**:

$$r^t(I, a) = v_i(\sigma^t, I, a) - \sum_{a' \in \mathcal{A}(I)} \sigma^t(I, a') v_i(\sigma^t, I, a')$$

표준 CFR의 **누적 후회**:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

**Regret Matching**에 의한 정책 도출:

$$\sigma^{t+1}(I, a) = \frac{\max(R^t(I, a), 0)}{\sum_{a'} \max(R^t(I, a'), 0)}$$

#### VAD-CFR의 핵심 메커니즘

**① 변동성 적응형 할인 (Volatility-Adaptive Discounting)**

즉시 후회 크기의 EWMA를 통해 변동성을 추적한다:

$$\text{EWMA}_t = 0.1 \cdot \|r^t\|_\infty + 0.9 \cdot \text{EWMA}_{t-1}$$

$$\text{volatility} = \min\left(1.0,\; \frac{\text{EWMA}_t}{2.0}\right)$$

적응적 할인 지수를 계산한다:

$$\alpha_{\text{eff}} = \max(0.1,\; 1.5 - 0.5 \cdot \text{volatility})$$

$$\beta_{\text{eff}} = -0.1 - 0.5 \cdot \text{volatility}, \quad \beta_{\text{eff}} = \min(\alpha_{\text{eff}}, \beta_{\text{eff}})$$

양/음 누적 후회에 대한 할인 계수:

$$d^+ = \frac{(t+1)^{\alpha_{\text{eff}}}}{(t+1)^{\alpha_{\text{eff}}} + 1}, \quad d^- = \frac{(t+1)^{\beta_{\text{eff}}}}{(t+1)^{\beta_{\text{eff}}} + 1}$$

**② 비대칭 즉시 부스팅 (Asymmetric Instantaneous Boosting)**

$$\tilde{r}^t(I, a) = \begin{cases} 1.1 \cdot r^t(I, a) & \text{if } r^t(I, a) > 0 \\ r^t(I, a) & \text{otherwise} \end{cases}$$

**③ 후회 누적 업데이트**

$$R^{t+1}(I, a) = \max\left(-20,\; d \cdot R^t(I, a) + \tilde{r}^t(I, a)\right)$$

여기서 $d = d^+$ if $R^t(I, a) \geq 0$, $d = d^-$ otherwise.

**④ 정책 도출 - 미래 투영(Future Projection)**

감쇠하는 낙관주의(optimism) 스케줄:

$$\text{optimism} = \frac{1}{1 + t/100} \cdot \max(0, 1 - 0.5 \cdot \text{volatility})$$

투영된 후회값에 비선형 스케일링 적용:

$$\sigma^{t+1}(I, a) \propto \left[\max\left(0,\; d \cdot R^t(I, a) + \tilde{r}^t(I, a) + \text{optimism}\right)\right]^{1.5}$$

**⑤ 하드 웜스타트 및 후회 크기 가중 정책 누적**

- 이터레이션 500 이전에는 정책 누적을 하지 않음
- 이후 가중치: $w = (t+1)^\gamma \cdot w_{\text{mag}} \cdot w_{\text{stable}}$
  - $\gamma = \min(4.0,\; 2.0 + 1.5 \cdot \text{volatility})$
  - $w_{\text{mag}} = (1 + \|r^t\|_\infty / 2)^{0.5}$
  - $w_{\text{stable}} = 1 / (1 + \|r^t\|_\infty^{1.5})$

### 2.5 SHOR-PSRO

표준 PSRO에서 오라클은 상대의 메타 전략에 대한 최적 반응을 계산한다:

$$\sigma_i^{k+1} \in \arg\max_{\sigma_i} \mathbb{E}_{\sigma_{-i} \sim \phi_{-i}}[u_i(\sigma_i, \sigma_{-i})] $$

#### SHOR-PSRO의 하이브리드 블렌딩 메커니즘

매 내부 솔버 이터레이션에서:

$$\sigma_{\text{hybrid}} = (1 - \lambda) \cdot \sigma_{ORM} + \lambda \cdot \sigma_{Softmax} $$

여기서 $\sigma_{ORM}$은 Optimistic Regret Matching의 출력이고, $\sigma_{Softmax}$는 순수 전략에 대한 Boltzmann 분포이다.

#### 동적 어닐링 스케줄

PSRO 이터레이션 진행도 $\text{prog} = \min(1.0, k/75)$에 따라:

$$\lambda: 0.30 \to 0.05, \quad \text{diversity}: 0.05 \to 0.001, \quad \text{temp}: 0.50 \to 0.01$$

#### 훈련-평가 비대칭

| 구분 | 블렌딩 계수 $\lambda$ | 반환 전략 | 다양성 보너스 |
|---|---|---|---|
| 훈련 솔버 | 동적 어닐링 ($0.3 \to 0.05$) | 평균 전략 | 감쇠 ($0.05 \to 0.001$) |
| 평가 솔버 | 고정 ($0.01$) | 최종 이터레이트 | 없음 ($0.0$) |

### 2.6 성능 향상

#### VAD-CFR 성능
- **훈련 게임**: 4개 훈련 게임 모두에서 기존 SOTA(DCFR, PCFR+, DPCFR+ 등) 대비 우수한 수렴 속도
- **테스트 게임**: 3-player Leduc Poker에서 착취가능성 $10^{-3}$ 이하 도달, 6-sided Liar's Dice에서 DCFR과 동등
- **종합**: **11개 게임 중 10개**에서 SOTA와 동등 이상 성능

#### SHOR-PSRO 성능
- 3-player Kuhn Poker에서 착취가능성 $< 10^{-3}$을 PRD, RM보다 빠르게 달성
- 6-sided Liar's Dice에서 정적 솔버 대비 명확한 우위
- **11개 게임 중 8개**에서 SOTA와 동등 이상 성능

### 2.7 한계

1. **이론적 수렴 보장의 부재**: VAD-CFR과 SHOR-PSRO는 순수 실험적으로 발견된 알고리즘으로, CFR의 $O(1/\sqrt{T})$ 후회 상한과 같은 형식적 수렴 보장이 제공되지 않는다.
2. **프록시 게임 의존성**: 훈련 게임 세트의 선택이 발견되는 알고리즘에 영향을 미침 (예: AOD-CFR은 다른 훈련 게임 세트에서 발견됨)
3. **매직 넘버 문제**: VAD-CFR의 웜스타트 임계값 500은 1000 이터레이션 평가 지평에 특화된 것일 수 있으며, LLM이 이 지평을 알지 못했음에도 생성했지만, 다른 지평에서의 적합성은 검증되지 않았다.
4. **확장성 미검증**: 대규모 딥 RL 에이전트나 함수 근사 기반 환경에서의 적용이 검증되지 않았다.
5. **소규모/중규모 게임 한정**: 모든 실험이 정확한 착취가능성 계산이 가능한 소·중규모 게임에서 수행되었다.
6. **4-player Kuhn Poker에서의 예외**: VAD-CFR이 11개 중 유일하게 이 게임에서 SOTA를 달성하지 못했다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 설계 원칙

이 논문은 일반화를 체계적으로 다루기 위해 **훈련-테스트 분리 프로토콜**을 채택한다:

- **훈련 세트** (4개 게임): 3-player Kuhn Poker, 2-player Leduc Poker, 4-card Goofspiel, 5-sided Liars Dice
- **테스트 세트** (4개 + 추가 7개 = 총 11개): 더 크고 복잡한 게임 변종

이러한 분리는 발견된 알고리즘이 훈련 게임에 과적합(overfitting)하지 않았음을 검증한다.

### 3.2 일반화 성능의 핵심 원천

#### (1) 적응적 메커니즘의 역할

VAD-CFR의 변동성 적응형 할인은 게임 토폴로지에 무관하게 학습 동역학의 "상태"에 반응한다. 즉, 후회 업데이트의 변동성이라는 **게임 불변적(game-invariant) 시그널**에 기반하므로, 특정 게임 구조에 결합되지 않고 다양한 환경에서 작동한다.

#### (2) 하드 웜스타트의 일반화 효과

초기 500 이터레이션의 정책 누적 억제는 **초기 노이즈 필터링**으로 작용한다. 논문은 이를 "early-stage noise from polluting the solution quality"를 방지하는 것으로 설명하며, 이는 게임 구조와 무관하게 초기 학습이 불안정하다는 보편적 현상에 대응한다.

#### (3) 다중 목표 최적화

AlphaEvolve가 $|\mathcal{G}|+1$개의 적합도 점수(각 게임별 + 평균)를 동시에 최적화하므로, 특정 게임에만 특화된 솔루션을 억제한다.

### 3.3 일반화 성능의 실증적 증거

| 게임 | VAD-CFR 성능 | SHOR-PSRO 성능 |
|---|---|---|
| 훈련 게임 (4개) | 전체 SOTA 이상 | 전체 SOTA 이상 |
| 테스트 게임 (7개) | 10/11 SOTA 이상 | 8/11 SOTA 이상 |

특히 3-player Leduc Poker(테스트)에서 VAD-CFR은 대부분의 기준선이 정체하는 수준 이하인 $10^{-3}$ 착취가능성까지 도달하였다.

### 3.4 일반화 성능 향상을 위한 향후 방향

1. **훈련 게임 세트 확대 및 다양화**: 더 이질적인 게임 구조를 포함하면 발견 알고리즘의 범용성을 높일 수 있다.
2. **적응적 하이퍼파라미터의 메타 학습**: 웜스타트 임계값(500)이나 어닐링 지평(75)과 같은 상수도 게임 크기에 비례하여 자동 조정되도록 진화시킬 수 있다.
3. **함수 근사 환경으로의 확장**: 딥 CFR이나 대규모 PSRO에서의 검증이 필요하다.
4. **이론적 뒷받침 구축**: 발견된 메커니즘의 수렴 보장을 사후적으로 분석하면, 어떤 조건에서 일반화가 보장되는지 규명할 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 패러다임의 전환

이 논문은 **"알고리즘 설계를 코드 공간에서의 탐색 문제로 환원"**하는 새로운 연구 패러다임을 다중 에이전트 학습 분야에 도입한다. 이는 다음과 같은 파급 효과를 갖는다:

- **인간 직관의 한계 극복**: VAD-CFR의 비선형 정책 스케일링($R^{1.5}$)이나 SHOR-PSRO의 훈련-평가 비대칭 같은 메커니즘은 인간 연구자가 직관적으로 시도하기 어려운 설계이다.
- **알고리즘 발견의 민주화**: LLM 기반 접근은 도메인 전문성이 부족한 연구자도 고성능 알고리즘을 발견할 수 있게 한다.
- **해석 가능성 유지**: 신경망 기반 메타러닝과 달리, 심볼릭 코드로 표현되어 사후 분석 및 이론적 이해가 가능하다.

### 4.2 향후 연구 시 고려할 점

1. **이론-실험 간극 메우기**: 발견된 알고리즘의 후회 상한(regret bound)을 사후적으로 증명하는 연구가 필요하다. 현재 VAD-CFR은 비표준적 비선형 변환($R^{1.5}$)과 음수 후회 캡($-20$)을 사용하므로, 기존 CFR 수렴 증명 프레임워크가 직접 적용되지 않는다.

2. **확장성 검증**: 대규모 게임(예: No-Limit Texas Hold'em, StarCraft II)에서의 검증이 필수적이다. 논문의 저자들도 "future work will explore the application of this evolutionary framework to fully deep reinforcement learning agents"를 언급한다.

3. **재현성 및 안정성**: 진화적 탐색의 확률적 특성상, 동일한 프레임워크를 재실행했을 때 유사한 품질의 알고리즘이 발견되는지 검증이 필요하다. AOD-CFR과 VAD-CFR이 서로 다른 훈련 세트에서 발견되었다는 사실은 탐색 결과의 민감성을 시사한다.

4. **검색 비용**: AlphaEvolve 실행에 필요한 계산 비용(LLM 호출 횟수, GPU 시간 등)이 보고되지 않았으며, 이는 실용성 평가에 중요하다.

5. **일반합 게임 및 협력적 메커니즘**: 논문이 주로 영합(zero-sum) 또는 경쟁적 게임에 초점을 맞추고 있으므로, 일반합 게임에서의 협력적 메커니즘 발견으로의 확장이 중요한 후속 방향이다.

6. **프롬프트 엔지니어링의 영향**: LLM에 제공되는 프롬프트(Listing 8, 9)가 탐색 공간을 암묵적으로 제한하므로, 프롬프트 설계가 발견 결과에 미치는 영향에 대한 체계적 분석이 필요하다.

### 4.3 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 접근법 | 탐색 공간 | 해석가능성 | MARL 적용 | 본 논문과의 비교 |
|---|---|---|---|---|---|
| **AutoML-Zero** (Real et al., 2020) | 유전 프로그래밍 | 기본 수학 연산 | 낮음 | ✗ | 더 원시적인 탐색 공간; LLM의 의미론적 이해 없음 |
| **Co-Reyes et al. (2021)** | 진화적 RL 알고리즘 발견 | 프로그램 그래프 | 중간 | 단일 에이전트 | 단일 에이전트 RL에 한정; 다중 에이전트 게임 이론 미적용 |
| **Farina et al. (2021, PCFR+)** | 수동 설계 | 고정 (Predictive RM) | 높음 | ✓ | VAD-CFR이 다수 게임에서 PCFR+ 능가 |
| **AutoCFR** (Xu et al., 2022) | 자동화된 CFR 설계 | 제한된 파라메트릭 | 중간 | ✓ | 비교적 제약된 탐색 공간; 코드 수준 진화 없음 |
| **Feng et al. (2021, NeuPL)** | 신경 자동 커리큘럼 | 신경망 파라미터 | 낮음 | ✓ (2-player) | 신경망 기반이라 해석 어려움; 심볼릭 발견과 대조적 |
| **Dynamic DCFR** (Xu et al., 2024a) | 동적 할인 | 학습된 할인 스케줄 | 중간 | ✓ | VAD-CFR과 유사한 동적 할인이나, 변동성 기반이 아닌 학습 기반 |
| **DPCFR+** (Xu et al., 2024b) | 가중 후회 + 낙관적 OMD | 수동 설계 | 높음 | ✓ | VAD-CFR이 다수 게임에서 DPCFR+ 능가 |
| **Sychrovský et al. (2024)** | CFR 프레임워크 내 메타 학습 | No-regret 프레임워크 | 중간 | ✓ | 이론적으로 견고하나 탐색 공간이 제한적 |
| **HS-PCFR+** (Zhang et al., 2026) | 하이퍼파라미터 스케줄 | 스케줄 공간 | 높음 | ✓ | VAD-CFR이 대부분 게임에서 동등 이상; 스케줄 vs. 코드 진화의 차이 |
| **Oh et al. (2025, Nature)** | 진화적 RL 발견 | 프로그램 코드 | 중간 | 단일 에이전트 | 단일 에이전트에서의 성공을 다중 에이전트로 확장한 것이 본 논문 |
| **AlphaEvolve** (Novikov et al., 2025) | LLM 기반 코드 진화 | 임의 코드 | 높음 | ✗ (수학/조합) | 본 논문이 AlphaEvolve를 MARL에 최초 적용 |

**핵심 차별점 요약**: 기존 자동화 연구(AutoCFR, Dynamic DCFR)는 주로 파라메트릭 탐색 공간에서 하이퍼파라미터나 가중 함수를 최적화한 반면, 본 논문은 **소스 코드 수준의 심볼릭 진화**를 통해 완전히 새로운 메커니즘(예: 비선형 정책 스케일링, 하드 웜스타트, 훈련-평가 비대칭)을 발견한다. 이는 탐색 공간의 표현력(expressiveness)에서 근본적 차이를 만든다.

---

## 결론

이 논문은 LLM 기반 진화적 코드 탐색이 다중 에이전트 학습 알고리즘 설계의 새로운 패러다임이 될 수 있음을 설득력 있게 보여준다. VAD-CFR과 SHOR-PSRO의 발견은 인간 직관의 한계를 넘어서는 알고리즘적 혁신이 가능함을 입증하며, 특히 훈련-테스트 분리를 통한 일반화 검증은 방법론적 엄밀성을 높인다. 다만, 이론적 수렴 보장의 부재, 대규모 환경에서의 미검증, 탐색 비용의 불투명성 등은 향후 연구에서 반드시 해결해야 할 과제이다.
