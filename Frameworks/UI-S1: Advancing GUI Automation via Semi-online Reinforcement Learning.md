
# UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning

## 1. 핵심 주장 및 기여

**UI-S1 논문의 핵심 주장**[1]

UI-S1은 **반온라인 강화학습(Semi-online Reinforcement Learning)**이라는 혁신적인 패러다임을 제시하여 GUI 자동화 에이전트의 훈련 효율성과 다중 턴 성능 간의 근본적 딜레마를 해결한다. 논문의 주요 주장은 다음과 같다:

(1) **오프라인-온라인 RL의 본질적 한계**: 오프라인 RL은 훈련 안정성이 우수하나 단계별 감독만 가능하여 다중 턴 작업 실행에 실패하고, 온라인 RL은 환경 상호작용을 통해 다중 턴 신호를 획득하지만 희소 보상과 높은 배포 비용으로 인해 실무적 장벽이 크다.[1]

(2) **모델 출력 불일치 문제**: 오프라인 훈련(전문가 시연에 의존)과 온라인 배포(모델의 자체 생성 출력 의존) 간 분포 불일치가 다중 턴 성능 격차의 핵심 원인이다.[1]

(3) **반온라인 RL의 우월성**: 패치 모듈과 이원 레벨 어드밴티지를 통해 오프라인 데이터만 활용하면서도 온라인 RL의 다중 턴 학습 신호를 획득할 수 있다.

**주요 기여**[1]

- **훈련 패러다임**: 정적 궤적에서 온라인 롤아웃 역학을 시뮬레이션하는 반온라인 RL 패러다임과 액션 불일치 복구용 패치 모듈 제시
- **성능 최적화**: 할인 미래 보상과 이원 레벨 어드밴티지를 정책 최적화에 통합하여 단계 수준 정확도와 궤적 수준 작업 완료 간 균형 달성
- **평가 메트릭**: 온라인 성능과 강한 상관성(R²=0.934)을 보이는 반온라인 성능(SOP) 메트릭 제안
- **경험적 성과**: UI-S1-7B가 7B 모델 중 SOTA 달성, AndroidWorld에서 +12.0%, AITW-Gen에서 +23.8% 성능 향상

***

## 2. 문제 설정 및 해결 방법

### 2.1 해결하고자 하는 문제

**문제 형식화**[1]

GUI 자동화는 다중 턴 순차적 의사결정 문제로 형식화된다. 고수준 명령어 $I$에 대해 에이전트는 일련의 액션을 통해 그래픽 인터페이스와 상호작용하여 목표를 달성해야 한다.

시간 $t$에서 에이전트는 현재 상태 $S_t \in \mathcal{S}$ (보통 인터페이스 스크린샷)를 관찰하고 과거 상호작용의 이력을 유지한다:

```math
H_t = \{(S_1, a_1, \mathcal{T}_1), (S_2, a_2, \mathcal{T}_2), \ldots, (S_{t-1}, a_{t-1}, \mathcal{T}_{t-1})\}
```

[1]

에이전트는 다음 액션 및 추론 과정을 생성한다:

$$a_t, \mathcal{T}_t \sim \pi(\cdot | I, S_t, H_t)$$

[1]

여기서 $\pi$는 정책 모델을 나타낸다.

**핵심 문제점**[1]

(1) **훈련-배포 불일치**: 오프라인 RL은 정적 궤적에서 전문가 시연에 조건화된 상태에서 훈련되지만( $H_t^{\text{static}} = \{(S_1^\*, a_1^\*), \ldots, (S_{t-1}^\*, a_{t-1}^*)\}$ ), 실제 배포는 모델의 자체 생성 출력에 조건화됨( 

```math
H_{t}^{\text{online}}=\{(S_{1},a_{\pi }\_1,\mathcal{T}_{\pi }\_1),\dots ,(S_{t-1},a_{\pi }\_t-1,\mathcal{T}_{\pi }\_t-1)\}
```

)[1]

(2) **지역 최적화 편향**: 오프라인 RL은 단계별 보상만 최적화하여 미래 목표를 무시하고 다중 턴 계획 실패

(3) **희소 보상**: 온라인 RL은 작업 완료 시에만 보상을 받아 복잡한 작업에서 비효율적인 훈련 발생

### 2.2 제안하는 방법: 반온라인 강화학습

**반온라인 롤아웃**[1]

전문가 궤적 $\tau^\* = \{(S_1^\*, a_1^\*), \ldots, (S_T^\*, a_T^*)\}$가 주어졌을 때, 정책 모델에서 $N$개의 롤아웃을 샘플링한다. $i$번째 후보 궤적은:

$$\tau^i = \{(S_1^i, a_1^i), (S_2^i, a_2^i), \ldots, (S_T^i, a_T^i)\}, \quad i = 1, \ldots, N$$

[1]

에이전트는 자신이 생성한 이력을 유지한다:

$$H_t^i = \{(S_1^i, a_1^i, \mathcal{T}_1^i), \ldots, (S_t^{i-1}, a_{t-1}^i, \mathcal{T}_{t-1}^i)\}$$

[1]

각 단계에서 정책은 이 자기 생성 이력을 기반으로 액션을 생성한다. 전문가 궤적을 사용하여 환경 역학을 근사한다:

$$S_t^{i+1} = \begin{cases} S_{t+1}^* & \text{if } \text{Matches}(a_t^i, a_t^*) \\ \text{None} & \text{otherwise} \end{cases}$$

[1]

액션이 전문가 시연과 일치하면 전문가 궤적에서 다음 상태를 획득하고, 일치하지 않으면 롤아웃이 종료된다.

**패치 모듈을 통한 궤적 복구**[1]

조기 종료를 방지하고 데이터 활용성을 개선하기 위해 패치 모듈 $\mathcal{P}$를 도입한다. 단계 $t$에서 불일치가 발생하면, 모듈은 부정확한 액션을 전문가 액션 $a_t^*$로 대체하고 합성 추론 $\mathcal{T}_t^{\text{patch}}$를 생성한다.

**세 가지 패치 전략**[1]

| 패치 방법 | 함수 정의 |
|:--|:--|
| Thought-Free Patch | $F(a_t, \mathcal{T}_t) = (a_t^*, \emptyset)$ |
| Off-Policy Thought Patch | $F(a_t, \mathcal{T}\_t) = (a_t^\*, \mathcal{M}_0(I, a_t^*, S_t))$ |
| On-Policy Thought Patch | $F(a_t, \mathcal{T}\_t) = (a_t^\*, \mathcal{M}(I, a_t^*, H_t, S_t))$ |

**보상 계산**[1]

각 단계의 복합 보상은 다음과 같이 계산된다:

$$r_t = 0.1 \cdot r_{\text{format}} + 0.4 \cdot \mathbb{I}[r_{\text{format}}=1] \cdot r_{\text{type}} + 0.5 \cdot \mathbb{I}[r_{\text{format}} \cdot r_{\text{type}}=1] \cdot r_{\text{acc}}$$[1]

여기서 $r_{\text{format}}$, $r_{\text{type}}$, $r_{\text{acc}}$는 각각 응답 형식화, 액션 유형 정확성, 정확한 일치 정도를 평가한다.

**할인 미래 보상**[1]

장시간 의존성을 포착하기 위해 할인 미래 반환을 계산한다:

```math
R_t^i = \sum_{k=t}^{t_{\text{end}}} \gamma^{k-t} r_k^i, \quad t_{\text{end}} := \min\left(\max\left\{k \geq t \left| \forall j \in [t, k], \text{Matches}(a_j^i, a_j^*) \right. \right\} + 1, T\right)
```

[1]

여기서 $\gamma \in (0, 1)$은 현재 결정이 미래 결과에 미치는 영향을 가중화하고, $t_{\text{end}}$는 예측 액션이 여전히 전문가와 일치하는 자연 궤적 세그먼트의 마지막 단계이다.

**이원 레벨 어드밴티지**[1]

**단계 수준 어드밴티지**는 같은 타임스텝에서 궤적 간 반환을 비교하여 지역 최적화 신호를 포착한다:

$$A^S(a_t^i) = \frac{R_t^i - \mu_t}{\sigma_t}$$[1]

여기서 $\mu_t$와 $\sigma_t$는 타임스텝 $t$에서 모든 롤아웃에 걸쳐 계산된다.

**에피소드 수준 어드밴티지**는 전체 작업 완료 신호를 포착한다:

$$A^E(\tau^i) = \frac{R(\tau^i) - \mu_{\tau}}{\sigma_{\tau}}$$[1]

두 어드밴티지를 통합된 어드밴티지로 결합한다:

$$A(a_t^i) = A^E(\tau^i) + \omega \cdot A^S(a_t^i)$$[1]

**정책 최적화 목적 함수**[1]

반온라인 RL은 다음 목적 함수를 통해 정책을 최적화한다:

$$\mathcal{J}(\theta) = \mathbb{E}_{\substack{\{\tau^i\}_N \sim_{\mathcal{P}} \pi_{\theta_{\text{old}}}(\cdot|I) \\ \{o_{i,t}\}^T_{t=1} \sim \tau^i}} \frac{1}{K} \sum^N_{i=1} \sum^T_{t=1} \sum^{|o_{i,t}|}_{k=1} \min\left(\rho(\theta)A(a_t^i), \text{clip}(\rho(\theta), 1 \pm \epsilon)A(a_t^i)\right) - \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

[1]

여기서:
- 첨자 $\mathcal{P}$는 궤적이 패치 모듈로 강화된 롤아웃을 통해 생성됨을 나타낸다
- $K$는 전체 토큰 수이다
- $\rho(\theta) = \frac{\pi_\theta(o_{i,t,k}|I,o_{i,t,<k})}{\pi_{\theta_{\text{old}}}(o_{i,t,k}|I,o_{i,t,<k})}$는 중요도 샘플링 비율이다
- $\beta$는 KL 페널티 강도를 제어한다

효과적인 학습과 충분한 탐색을 보장하기 위해 최소 어드밴티지 분산을 강제한다: $\sigma(\{A(a_t^i)\}) > \eta$. 이 다양성 임계값이 충족될 때까지 동적 샘플링을 수행한다.

***

## 3. 모델 구조 및 성능 분석

### 3.1 모델 아키텍처

UI-S1의 전체 훈련 파이프라인은 **두 단계 구조**를 따른다:[1]

**1단계: 감독 미세 조정(SFT)**
- AndroidControl-Train 및 Amex 데이터셋에서 기본 모델(Qwen2.5VL-7B)을 훈련
- 전문가 시연에 대한 모방 학습을 통해 행동 사전 설정

**2단계: 반온라인 RL**
- Thought-Free Patch와 $\epsilon=1$을 최적 구성으로 사용
- 이원 레벨 어드밴티지 가중치: $\omega = 1.0$
- 할인 계수: $\gamma = 0.5$
- 최소 어드밴티지 분산 임계값: $\eta = 0.3$
- 배치 크기: 32, 총 5 에포크 훈련

### 3.2 성능 향상 결과

**다중 턴 벤치마크 성과**[1]

| 모델 | SOP-PG | SOP-TSR | AITW-Gen | AndroidWorld | MiniWob++ |
|:--|:--|:--|:--|:--|:--|
| 기본 모델 | 16.8 | 9.1 | 50.5 | 14.9 | 28.8 |
| SFT 적용 | 17.0 | 9.3 | 58.9 | 21.7 | 28.5 |
| Offline RL | 18.3 | 10.5 | 54.6 | 15.7 | 19.8 |
| 반온라인 RL만 | 30.6 | 16.0 | 70.2 | 30.4 | 36.3 |
| **UI-S1-7B (최종)** | **32.4** | **16.3** | **74.3** | **34.0** | **40.2** |
| **개선도** | **+15.6** | **+7.2** | **+23.8** | **+19.1** | **+11.4** |

기본 모델 대비 UI-S1-7B는 AndroidWorld에서 +19.1%, AITW-Gen에서 +23.8%의 현저한 향상을 달성했다.[1]

**단일 턴 벤치마크 성과**[1]

| 모델 | SS-Pro (GR) | AC-High (SR) | GUI Odyssey (SR) |
|:--|:--|:--|:--|
| 기본 모델 | 72.5 | 67.4 | 52.4 |
| UI-S1-7B | 73.4 | 76.3 | 59.5 |
| 개선도 | +0.9 | +8.9 | +7.1 |

반온라인 RL은 단일 턴 성능을 희생하지 않으면서 다중 턴 능력을 크게 향상시켰다.

**모델 크기별 일반화 성능**[1]

| 모델 | SOP-Score | AndroidWorld |
|:--|:--|:--|
| Qwen2.5VL-3B | 2.4 | 3.7 |
| UI-S1-3B | 10.6 (+342%) | 11.9 (+222%) |
| Qwen2.5VL-7B | 13.0 | 14.0 |
| UI-S1-7B | 23.3 (+79%) | 26.9 (+92%) |
| Qwen2.5VL-32B | 14.0 | 21.2 |
| UI-S1-32B | 27.4 (+96%) | 33.2 (+57%) |

모든 모델 크기에서 반온라인 RL은 일관된 개선을 보였으며, 모델 크기가 증가할수록 상대 개선도는 감소하는 경향을 나타낸다.

**패치 전략 비교**[1]

1000개 학습 샘플 기준:
- Thought-Free Patch: SOP-Score 25.7 (낮은 계산 비용)
- Off-Policy Thought Patch: SOP-Score 22.6 (보조 모델과 정책 모델 간 분포 불일치로 인한 성능 저하)
- On-Policy Thought Patch: SOP-Score 26.1 (최고 성능, 높은 계산 비용)

최종 구성: Thought-Free Patch with $\epsilon=1$ (효율성과 성능 간 최적 균형)

### 3.3 반온라인 성능(SOP) 메트릭의 유효성

**온라인 메트릭과의 상관성**[1]

- SOP ↔ AndroidWorld: $R^2 = 0.934$ (매우 강한 상관성)
- 기존 오프라인 메트릭(AndroidControl-High): $R^2 = 0.470$ (약한 상관성)
- 기존 오프라인 메트릭(GUI Odyssey): $R^2 = 0.398$ (약한 상관성)

**SOP 메트릭 정의**[1]

$$\text{PG} = \frac{1}{N} \sum_{i=1}^{N} \frac{s_i}{t_i}, \quad \text{TSR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[s_i = t_i], \quad \text{Score} = \frac{\text{PG} + \text{TSR}}{2}$$

여기서 $s_i$는 성공한 단계 수, $t_i$는 전문가 궤적의 총 단계 수이다.

SOP는 모델 생성 이력을 유지하며 액션 불일치 시에만 종료하므로, 실제 배포 조건을 정확히 반영한다.

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 일반화 능력 분석

**단일 턴 → 다중 턴 전이**[1]

UI-S1은 단일 턴 성능을 유지하면서 다중 턴 능력을 현저히 향상시켰다. 이는 반온라인 RL이 모델의 근본적인 이해 능력을 손상시키지 않으면서도 순차적 추론 능력을 선택적으로 강화했음을 시사한다.

**도메인 외 일반화**[1]

표 2의 결과는 다양한 벤치마크에서 일관된 개선을 보여준다:
- AndroidWorld (동적, 실제 앱 기반): +19.1%
- AITW-Gen (웹 기반, 다양한 작업): +23.8%
- MiniWob++ (미니 웹 작업): +11.4%

이러한 다중 도메인 개선은 반온라인 RL의 학습 신호가 특정 도메인에 국한되지 않고 일반화되는 능력을 보유했음을 시사한다.

### 4.2 데이터 확장성

**확장 법칙 분석**[1]

데이터 크기에 따른 성능 곡선이 지수 함수 $y = A + B \cdot e^{C + kx}$를 따르며, 패치 임계값 $\epsilon$에 따른 확장 계수:
- $\epsilon = 0$: $k = 1.13$
- $\epsilon = 1$: $k = 0.91$
- $\epsilon = 2$: $k = 0.79$
- $\epsilon = \infty$: $k = 0.73$

더 큰 $\epsilon$ 값이 절대 성능뿐 아니라 **데이터 효율성도 개선**하여, 각 훈련 샘플에서 더 효과적으로 학습함을 나타낸다.

### 4.3 할인 계수의 영향

**미래 보상의 중요성**[1]

그림 6에서 보면 여러 $\gamma$ 값에 대한 훈련 동역학 비교:
- $\gamma = 0$ (미래 보상 없음): 최악의 성과
- $\gamma = 0.5$: 최적 성과
- $\gamma = 0.9$: 중간 성과

반온라인 RL은 훈련 단계를 거치면서 작업 성공률이 상승하는 반면, 전통적 오프라인 RL은 반대의 동향을 보인다. 이는 역사적 맥락 연속성이 다중 턴 패러다임 학습을 가능하게 함을 증명한다.

### 4.4 훈련 패러다임 복합 효과

**SFT + 반온라인 RL**[1]

- SFT만: AndroidWorld 21.7%
- 반온라인 RL만: AndroidWorld 30.4%
- **SFT + 반온라인 RL**: AndroidWorld 34.0%

결합 접근법은 단순 합보다 큰 시너지 효과를 보여주며, SFT가 행동 사전 설정을 제공하고 반온라인 RL이 이를 기반으로 다중 턴 추론 능력을 세밀하게 조정한다.

***

## 5. 한계 및 고려사항

### 5.1 기술적 한계

**패치 모듈의 분포 불일치**[1]

Off-Policy Thought Patch에서 보조 모델(예: DeepSeek-R1)의 생각 과정이 정책 모델의 추론 스타일과 일치하지 않아 성능 저하 발생. 이는 모델 간 추론 양식의 불일치가 여전히 해결되지 않은 문제임을 시사한다.

**조기 종료 문제**[1]

패치 임계값 $\epsilon$이 낮을수록 초기 불일치 시 롤아웃이 조기 종료되어 후속 궤적 단계의 학습 신호에 접근할 수 없다. 이는 장기간 의존성 학습을 제한할 수 있다.

**보상 함수의 단순화**[1]

현재 보상 함수는 형식화, 유형, 정확성 세 가지 명시적 신호에만 의존하며, 이는 복잡한 실제 작업의 미묘한 성공 조건을 완전히 포착하지 못할 수 있다.

### 5.2 평가 메트릭의 한계

**SOP 메트릭의 근사성**[1]

SOP는 실제 온라인 환경을 완벽히 재현하지 못한다. 전문가 궤적에 의존하여 상태를 제공하므로, 모델의 액션이 크게 벗어나는 경우 실제 환경의 새로운 상태와 SOP의 상태 차이가 누적될 수 있다.

**단일 턴 성능 개선의 한계**[1]

표 3에서 보듯이 일부 단일 턴 메트릭에서 상대적으로 작은 개선(SS-Pro +0.9%)을 보였다. 이는 반온라인 RL이 다중 턴 최적화에 편향되어 있음을 시사한다.

### 5.3 실무적 한계

**전문가 궤적 의존성**[1]

반온라인 RL은 여전히 고품질의 전문가 시연 데이터가 필요하다. 이는 새로운 도메인이나 복잡한 작업에 적용할 때 데이터 수집 비용을 제한한다.

**계산 오버헤드**[1]

패치 모듈과 이중 어드밴티지 계산이 훈련 비용을 증가시킨다. Thought-Free Patch 기준 GPU 시간 분석에서도 온라인 RL과 비교하면 계산 효율성에서 여전히 개선 여지가 있다.

**일반 컴퓨터 시스템으로의 확장**[1]

대부분의 평가가 모바일 환경(Android)에 집중되어 있으며, 데스크톱 웹 환경으로의 완전한 전이 효과는 제한적으로 검증되었다. 표 2에서 AndroidWorld와 MiniWob++의 개선도가 다양함이 이를 반영한다.

***

## 6. 논문의 영향과 향후 연구 방향

### 6.1 학문적 영향

**반온라인 강화학습 패러다임의 일반화 가능성**[2]

UI-S1의 반온라인 RL 접근법은 GUI 자동화를 넘어 다중 턴 상호작용이 필요한 모든 RL 도메인에 적용될 수 있다. 로봇 조작, 웹 에이전트, 문제 해결 작업 등 복잡한 순차적 의사결정이 필요한 분야에서 동일한 원리를 활용할 수 있다.

**오프라인-온라인 RL의 이론적 진전**[3]

기존 오프라인-온라인 RL 연구는 분포 이동(distribution shift)을 관리하는 데 초점을 맞췄으나, UI-S1은 **훈련-배포 동역학 불일치**라는 새로운 관점을 제시한다. 이는 향후 오프라인 RL 이론에 새로운 연구 방향을 제공한다.

**다중 턴 추론 평가의 개선**[4]

SOP 메트릭은 기존 벤치마크의 약점(R²=0.470)을 해결하고 온라인 성능과 강한 상관성(R²=0.934)을 보여준다. 이는 향후 LLM 에이전트 평가 표준으로 채택될 가능성이 높다.

### 6.2 실무적 시사점

**생산 수준의 GUI 에이전트 개발**[5][6]

UI-S1-7B의 성능 향상(+23.8% on AITW-Gen)은 상업적 배포 가능성을 시사한다. 특히 MobileGUI-7B(온라인 RL)와의 경쟁력 있는 성과(AndroidWorld에서 UI-S1: 34.0% vs MobileGUI: 30.0%)는 오프라인 훈련의 실무적 이점(낮은 배포 비용, 높은 재현성)을 보여준다.

**데이터 효율성의 중요성**[7][8]

반온라인 RL의 데이터 확장 법칙 분석($\epsilon$ 증가에 따른 $k$ 감소)은 제한된 데이터 환경에서도 효과적인 훈련이 가능함을 시사한다. 이는 새로운 도메인 적응 시나리오에서 중요하다.

### 6.3 향후 연구 시 고려할 점

**패치 모듈의 이론적 정당화**[1]

현재 패치 모듈은 경험적으로 효과적이지만 이론적 근거가 부족하다. 향후 연구는 다음을 탐구해야 한다:

1. **패치 된 궤적의 분포 특성**: 패치 된 궤적이 실제 정책 궤적 분포에 얼마나 가까운가?
2. **패치 횟수와 성능의 관계**: 최적 패치 횟수 $\epsilon^*$를 이론적으로 유도할 수 있는가?
3. **거짓 신호의 영향**: 잘못된 패치가 정책 학습에 미치는 장기적 영향 분석

**보상 함수의 고도화**[1]

현재 선형 결합 보상 함수는 단순하다. 향후 연구 방향:

1. **계층적 보상 구조**: 작업의 부분 목표 달성도를 더 세밀하게 측정하는 다층 보상
2. **학습 가능한 보상**: 메타 학습을 통해 도메인별 최적 보상 함수를 자동으로 발견
3. **인간 피드백 통합**: RLHF 스타일의 비용 함수 학습

**도메인 확장 전략**[9][10][11]

1. **크로스 플랫폼 일반화**: 모바일(Android)에서 웹, 데스크톱으로의 전이 효율성 개선
2. **다국어 지원**: 중국어, 일본어 등 다양한 언어의 GUI 작업에 대한 일반화
3. **복잡한 멀티 앱 작업**: 현재는 단일 또는 소수 앱 간 전환, 향후 복잡한 다중 앱 워크플로우 지원

**이론적 분석의 확대**[12][2]

1. **다중 턴 성공률의 하한 증명**: 단일 턴 GRPO 개선이 다중 턴 성공률에 미치는 이론적 보장
2. **일반화 경계**: 학습 데이터 크기와 새 도메인 성능 간의 표본 복잡도 분석
3. **수렴 보장**: 반온라인 RL의 수렴 속도 및 점근적 최적성 증명

**평가 메트릭의 고도화**

1. **중간 상태 검증**: 최종 상태뿐 아니라 모든 중간 단계의 정확성 평가[13]
2. **반사(Reflection) 능력 측정**: 에러 발생 시 복구 능력 평가
3. **실시간 신뢰도 측정**: 작업 진행 중 성공 가능성의 동적 예측

**컴퓨터 사용 에이전트의 통합**[14][15]

1. **일반 컴퓨터 제어**: GUI 이해를 넘어 파일 시스템, 터미널 명령 실행
2. **보안 및 개인정보 보호**: 민감한 정보 식별 및 보호 메커니즘
3. **사용자 맞춤화**: 개별 사용자의 선호도 학습 및 반영

***

## 7. 관련 최신 연구 현황 (2020년 이후)

### 7.1 GUI 에이전트 강화학습의 진화

**오프라인 RL 기반 GUI 에이전트 (2023-2024)**[9][7]

- **UI-R1** (2025): 규칙 기반 RL로 MLLM의 추론 능력을 향상시키는 첫 프레임워크. 136개 고품질 작업 데이터셋으로 3B 모델이 7B 모델 성능 달성
- **GUI-R1** (2025): GRPO 기반 GUI 에이전트의 일반화 R1 스타일 모델
- **UI-TARS** (2025): 원시 에이전트로 자동화 GUI 상호작용 개척

**온라인 RL 기반 GUI 에이전트 (2024-2025)**[6][16][5]

- **MobileGUI-RL**: 자체 탐색, 작업 필터링, 궤적 수준 강화학습을 통한 확장 가능한 파이프라인. 사전 정의된 검증 논리 불필요
- **MobileRL**: 난이도 적응 커리큘럼과 최단 경로 보상 조정으로 온라인 RL 훈련 안정화

### 7.2 다중 턴 추론과 신용 할당 (2024-2025)**

**신용 할당 개선**[2][12]

- **Turn-Level Credit Assignment**: 궤적 수준 어드밴티지 추정에서 벗어나 각 턴의 기여도를 세밀하게 측정. 다중 턴 도구 사용 작업에서 100% 성공률 달성
- **Single-Turn to Multi-Turn Transfer**: 단일 턴 GRPO 개선이 다중 턴 작업 성공 확률을 높이는 이론적 증명 제시

### 7.3 시각 기반 GUI 이해 (2024-2025)**

**GUI 그라운딩의 발전**[10][17][18][19]

- **UGround**: GUI 요소의 보편적 시각 그라운딩 실현
- **SeeClick**: GUI 그라운딩을 위한 고급 시각 GUI 에이전트
- **OS-Atlas**: 다중 플랫폼(Windows, Linux, macOS, Android)에 걸친 기초 행동 모델
- **V-Zen**: 이중 해상도 이미지 인코더를 갖춘 혁신적 MLLM

### 7.4 자가 진화 및 탐색 기반 학습 (2024-2025)**

**강화학습을 통한 자가 개선**[11][19][10][9]

- **GUI-Shift**: 역동역 작업으로 자체 감독 학습, 11.2% 이상의 GUI 자동화 정확성 증진
- **GUI-Bee**: Q-값 기반 맥락 내 강화학습으로 새로운 환경으로의 그라운딩 정렬
- **UI-AGILE**: 연속 보상 함수와 "Simple Thinking" 보상으로 단기 속도와 그라운딩 정확성 균형

### 7.5 벤치마크 및 평가 방법론 (2024-2025)**

**동적 벤치마크의 등장**[20][13]

- **AndroidWorld**: 116개 작업에 걸쳐 20개 실제 Android 앱에서 매개변수화되고 표현된 동적 작업 생성 (2024)
- **AITW (Android in-the-Wild)**: 실제 행동에 기반한 작업 평가로 단계별 중간 정확성 측정
- **ProBench**: 중간 상태 검증으로 평가 정밀도 향상 (2025)

**다중 턴 추론 평가**[21][22]

- **MTR-Bench**: 4개 클래스, 40개 작업, 3600개 인스턴스를 포함한 LLM 다중 턴 추론 종합 벤치마크
- **TurnBench**: 규칙 기반 피드백으로 중간 추론 단계 평가

### 7.6 오프라인-온라인 RL의 이론적 진전 (2023-2025)**

**분포 이동 해결**[23][24][25][26]

- **O2AC (Offline-Online Actor-Critic)**: 행동 복제 제약으로 오프라인 단계의 분포 이동 해결, 온라인 단계에서 제약 영향 점진적 감소
- **OEMA (Optimistic Exploration and Meta Adaptation)**: 낙관주의 원칙과 메타 학습 기반 적응으로 샘플 효율성 개선
- **EDIS (Energy-guided Diffusion Sampling)**: 확산 모델을 활용한 오프라인 지식 추출과 온라인 데이터 생성

**오프라인 RL의 분포 이동 분석**[27][3]

- **RO2O (Robust Offline-to-Online)**: 불확실성과 평탄성을 통한 견고한 오프라인-온라인 적응
- **MOOD-CRL**: 인과 추론으로 분포 외 적응

### 7.7 컴퓨터 사용 에이전트의 확장**[15][14]

**일반화된 컴퓨터 제어 패러다임**

- **OS Agents (Operating System Agents)**: GUI 이해를 넘어 전체 운영 체제와의 상호작용 (2025)
- **Agent S2**: 전문가-일반가 혼합 구조로 구성 요소 책임을 분할하고 개선된 그라운딩 기법 도입
- **AutoGLM**: 자기 진화 온라인 커리큘럼 강화학습

***

## 8. 결론

UI-S1은 **반온라인 강화학습**이라는 혁신적 패러다임을 통해 GUI 자동화 에이전트의 장기적 다중 턴 추론 능력을 획기적으로 향상시켰다. 오프라인 RL의 훈련 효율성과 온라인 RL의 시간적 최적화 목표를 결합함으로써, 단일 논문에서 다중 턴 성능에서 +23.8%, 온라인 평가 메트릭 상관성을 R²=0.934까지 달성하였다.

**핵심 성과:**
- 패치 모듈을 통한 궤적 복구로 제한된 오프라인 데이터에서 최대한의 학습 신호 추출
- 이원 레벨 어드밴티지로 단계 정확성과 에피소드 완료 간 균형 달성
- 모든 모델 크기(3B, 7B, 32B)에서 일관된 일반화 성능 입증

**향후 연구의 초점:**
1. 패치 모듈의 이론적 정당화 및 최적 패치 횟수의 수학적 도출
2. 학습 가능한 보상 함수 설계로 도메인별 최적화 자동화
3. 크로스 플랫폼 및 다국어 환경으로의 확장
4. 사람-기계 협력 시나리오에서의 적응 학습

UI-S1은 단순히 GUI 자동화를 넘어 **복잡한 다중 턴 상호작용이 필요한 모든 RL 도메인의 패러다임 변화**를 주도할 수 있는 잠재력을 보여준다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/14c12ff5-2911-456d-be52-24e416c0887c/2509.11543v2.pdf)
[2](https://arxiv.org/html/2509.20616v1)
[3](https://cameronrwolfe.substack.com/p/online-rl)
[4](https://arxiv.org/html/2503.22458v1)
[5](https://arxiv.org/abs/2509.18119)
[6](https://arxiv.org/abs/2507.22025)
[7](https://www.semanticscholar.org/paper/57f0694199b43ccbba9c0d4224bebd450f7c3af8)
[8](https://www.emergentmind.com/topics/agentgym-rl-framework)
[9](https://arxiv.org/abs/2505.12370)
[10](https://arxiv.org/abs/2505.12493)
[11](https://www.semanticscholar.org/paper/3433b31214d5771bb35e790637cf90be50c8e113)
[12](https://openreview.net/forum?id=h83vIG5Hre)
[13](https://arxiv.org/html/2511.09157v1)
[14](https://arxiv.org/abs/2508.04482)
[15](https://arxiv.org/html/2504.00906v1)
[16](https://arxiv.org/html/2507.05720v1)
[17](http://arxiv.org/pdf/2410.23218v1.pdf)
[18](http://arxiv.org/pdf/2405.15341.pdf)
[19](https://arxiv.org/html/2501.13896)
[20](https://openreview.net/forum?id=il5yUQsrjC)
[21](https://arxiv.org/abs/2505.17123)
[22](https://aclanthology.org/2025.findings-emnlp.1084.pdf)
[23](https://www.semanticscholar.org/paper/88ca2f21aa90b1c6ca9204c08d76a1228bc9c225)
[24](https://ieeexplore.ieee.org/document/9964421/)
[25](https://ieeexplore.ieee.org/document/10210487/)
[26](https://arxiv.org/abs/2407.12448)
[27](https://arxiv.org/abs/2405.03892)
[28](https://dl.acm.org/doi/10.1145/3705328.3747995)
[29](https://arxiv.org/abs/2504.20464)
[30](https://www.semanticscholar.org/paper/83bd2fb29b2d5e78ca2a1a13b589b94927e3e485)
[31](https://arxiv.org/abs/2406.11896)
[32](http://arxiv.org/pdf/2405.14751.pdf)
[33](https://arxiv.org/html/2410.19528v3)
[34](https://arxiv.org/html/2411.00820v1)
[35](https://arxiv.org/html/2503.21620v1)
[36](https://arxiv.org/pdf/2306.03530.pdf)
[37](https://arxiv.org/pdf/2502.18906.pdf)
[38](https://arxiv.org/pdf/2501.19245.pdf)
[39](https://arxiv.org/html/2503.08464v2)
[40](https://aclanthology.org/2025.emnlp-demos.12.pdf)
[41](https://aclanthology.org/2025.findings-acl.1158.pdf)
[42](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)
[43](https://www.ericsson.com/en/blog/2023/11/reinforcement-learning)
[44](https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List)
[45](https://arxiv.org/html/2507.22025v1)
[46](https://aacrjournals.org/mct/article/24/10_Supplement/C070/766573/Abstract-C070-Evolving-Paradigms-and-shifting)
[47](https://soziopolit.sgu.ru/ru/articles/tri-goda-svo-dinamika-izmeneniya-otnosheniya-molodezhi-v-kontekste-vliyaniya-media)
[48](https://ijsra.net/node/5102)
[49](https://www.tandfonline.com/doi/full/10.1080/17460441.2025.2552144)
[50](https://academic.oup.com/ndt/article/doi/10.1093/ndt/gfaf116.1729/8296506)
[51](https://academic.oup.com/eurpub/article/doi/10.1093/eurpub/ckaf161.500/8302477)
[52](https://invergejournals.com/index.php/ijss/article/view/132)
[53](https://medworksmedia.com/pb-article/the-black-book-of-psychotropic-dosing-and-monitoring/)
[54](https://journals.physiology.org/doi/10.1152/physiol.2025.40.S1.1021)
[55](http://arxiv.org/pdf/2410.05243.pdf)
[56](https://arxiv.org/abs/2503.00401)
[57](https://arxiv.org/html/2412.01268v1)
[58](https://arxiv.org/pdf/2310.04716.pdf)
[59](https://aclanthology.org/2025.emnlp-main.1688.pdf)
[60](https://arxiv.org/html/2509.11543v1)
[61](https://verl.readthedocs.io/en/latest/algo/grpo.html)
[62](https://openreview.net/pdf/d34f5212cbd1376fcebf133a0f2684c5edf0d909.pdf)
[63](https://www.vldb.org/pvldb/vol17/p414-chen.pdf)
[64](https://cameronrwolfe.substack.com/p/grpo)
[65](https://arxiv.org/html/2505.12370v1)
[66](https://openreview.net/pdf?id=SJPq1xBPHX)
[67](https://abderrahmanskiredj.github.io/the-illustrated-grpo/)
[68](https://ojs.aaai.org/index.php/AAAI/article/view/29083)
[69](https://arxiv.org/abs/2410.14957)
[70](https://arxiv.org/abs/2411.17764)
[71](https://www.semanticscholar.org/paper/0e1a3c83fa7184211ee331a5fece022424bbe65d)
[72](https://arxiv.org/abs/2405.13193)
[73](http://arxiv.org/pdf/2407.12448.pdf)
[74](https://arxiv.org/html/2309.16973)
[75](http://arxiv.org/pdf/2212.08131.pdf)
[76](https://arxiv.org/html/2410.18626)
[77](https://arxiv.org/pdf/2310.04579.pdf)
[78](https://arxiv.org/pdf/2306.07541.pdf)
[79](https://arxiv.org/html/2405.14374v1)
[80](http://arxiv.org/pdf/2406.09486.pdf)
[81](https://www.ijcai.org/proceedings/2024/0477.pdf)
[82](https://arxiv.org/abs/2510.21339)
[83](https://openreview.net/forum?id=wbwTF909Ve)
[84](https://proceedings.mlr.press/v244/yu24a.html)
[85](https://www.emergentmind.com/topics/android-agent-arena-a3)
[86](https://www.sciencedirect.com/science/article/abs/pii/S0893608025010007)
[87](https://www.themoonlight.io/en/review/androidworld-a-dynamic-benchmarking-environment-for-autonomous-agents)
