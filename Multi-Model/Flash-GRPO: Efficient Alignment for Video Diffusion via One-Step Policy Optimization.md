
# Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization 

> **논문 정보**
> - **제목:** Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization
> - **저자:** Xiaoxuan He, Siming Fu, Zeyue Xue, Weijie Wang, Ruizhe He, Yuming Li, Dacheng Yin, Shuai Dong, Haoyang Huang, Hongfa Wang, Nan Duan, Bohan Zhuang (저장대학교, Joy Future Academy 등)
> - **arXiv:** [2605.15980](https://arxiv.org/abs/2605.15980) (제출: 2026.05.15, v2: 2026.06.03)
> - **학술대회:** **ICML 2026** (accepted)
> - **코드:** https://github.com/Shredded-Pork/Flash-GRPO

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Group Relative Policy Optimization(GRPO)은 비디오 디퓨전 모델을 인간의 선호와 정렬하는 데 필수적인 기법으로 부상했지만, 14B 파라미터 모델 훈련 시 실험당 수백 GPU-day가 요구되는 심각한 계산 병목을 겪는다.

기존 효율화 방법들은 슬라이딩 윈도우를 통한 타임스텝 서브샘플링으로 비용을 줄이지만, 최적화를 근본적으로 훼손하며 심각한 불안정성과 함께 풀 궤적(full trajectory) 성능에 미치지 못하는 문제가 있다.

이에 대해 Flash-GRPO는 정확한 어드밴티지 추정을 위한 **Iso-temporal Grouping**과 균형 잡힌 최적화를 위한 **Temporal Gradient Rectification**을 결합한 원칙 기반의 단일 스텝(single-step) 훈련 프레임워크로, 최소한의 계산 비용으로 풀 궤적 수준의 성능을 달성한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **방법론** | Iso-temporal Grouping + Temporal Gradient Rectification |
| **효율성** | 6× 훈련 비용 가속화 달성 |
| **확장성** | 1.3B ~ 14B 파라미터 모델에서 검증 |
| **성능** | 풀 궤적 방식 대비 동등 혹은 우수한 정렬 품질 |
| **안정성** | 단일 스텝 훈련에서 안정적, 단조로운 수렴 |
| **공개성** | ICML 2026 게재, 코드 전면 공개 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

이 연구는 비디오 디퓨전 모델을 인간 선호와 정렬하는 과정에서 발생하는 높은 훈련 비용과 불안정성을 해결하고자 하며, 기존 방법들이 타임스텝 샘플링의 비최적 최적화로 인해 이 문제를 더욱 악화시키는 현실에 주목한다.

연구진은 단일 스텝 비디오 GRPO에서의 최적화 불안정성의 두 가지 근본 원인—정책 성능을 노이즈 난이도와 뒤섞는 **타임스텝 혼합 어드밴티지 추정**과 디퓨전 궤적에 걸쳐 크기 불균형을 초래하는 **시간 의존적 그래디언트 스케일링**—을 규명하고, 이에 대한 이론적 유도와 실험적 검증을 제공한다.

#### 문제 1: 타임스텝 혼합 분산 (Timestep-Confounded Variance)

비디오 디퓨전 모델의 역방향 과정(denoising)에서, 각 타임스텝 $t$의 SDE는 다음과 같이 표현된다:

$$
d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)dt + g(t)d\mathbf{w}_t
$$

표준 GRPO에서는 서로 다른 롤아웃(rollout) 간에 타임스텝 $t$가 무작위로 다르게 샘플링되어, 어드밴티지(advantage) 추정치가 정책 품질이 아닌 **타임스텝의 난이도 차이**에 의해 오염된다. 이를 수식으로 표현하면:

$$
\hat{A}_i = r(\mathbf{x}^{(i)}) - \bar{r}, \quad \mathbf{x}^{(i)} \sim \pi_\theta(\cdot | t^{(i)})
$$

여기서 $t^{(i)}$가 롤아웃마다 다를 경우, $\hat{A}_i$는 정책의 실제 성능이 아닌 타임스텝 $t^{(i)}$의 노이즈 수준에 의해 결정된다.

#### 문제 2: 그래디언트 스케일 이질성 (Gradient Scale Heterogeneity)

정책 그래디언트는 SDE 이산화(discretization)로부터 비롯된 시간 의존적 스케일링 인자를 내재적으로 포함하며, 이는 디퓨전 궤적 전체에 걸쳐 몇 차수(order of magnitude)씩 변화한다. 이로 인해 실제 중요도와 무관하게 초기 타임스텝이 파라미터 업데이트를 지배하는 심각한 최적화 불균형이 발생한다.

SDE 이산화 후 정책 그래디언트 목적함수는 다음과 같이 표현된다:

$$
\mathcal{L}_\text{GRPO}(\theta) = \mathbb{E}_{t, \mathbf{x}_0} \left[ \lambda(t) \cdot \hat{A} \cdot \log \pi_\theta(\mathbf{x}_t | \mathbf{x}_{t+1}) \right]
$$

여기서 $\lambda(t)$는 타임스텝 $t$에 따라 **수 차수**씩 달라지는 시간 의존 스케일 인자이다.

---

### 2-2. 제안하는 방법 (Flash-GRPO)

#### 방법 1: Iso-temporal Grouping (동일 타임스텝 그룹화)

각 프롬프트는 단일 샘플링된 타임스텝에서 ODE→SDE 전환을 수행하여 탐색 및 그래디언트 계산을 수행하고, 나머지 타임스텝은 정확한 보상 신호를 위해 결정론적 ODE를 사용한다. 각 그룹 내의 롤아웃은 이 전환 타임스텝을 공유하지만 초기 노이즈에서 차이를 두어, 정책 유발 분산과 타임스텝 유발 분산을 인수분해(factorize)한다.

수식으로 나타내면, 그룹 $\mathcal{G}_t$ 내의 어드밴티지 추정은:

$$
\hat{A}_i = r(\mathbf{x}_0^{(i)}) - \frac{1}{|\mathcal{G}_t|}\sum_{j \in \mathcal{G}_t} r(\mathbf{x}_0^{(j)}), \quad \forall i \in \mathcal{G}_t
$$

동일 타임스텝 $t$를 공유하는 그룹 내에서 베이스라인을 계산하므로, 타임스텝 난이도 편향이 제거된다.

$$
\text{Var}[\hat{A}_i | \mathcal{G}_t] = \text{Var}_{\text{policy}}[\hat{A}_i] \quad (\text{타임스텝 분산 제거})
$$

#### 방법 2: Temporal Gradient Rectification (시간적 그래디언트 교정)

Temporal Gradient Rectification은 $\lambda(t)$를 명시적으로 1로 정규화하여, 모든 타임스텝의 기여를 균일하게 만들고 이산화로 인한 편향을 최적화 역학에서 제거한다.

교정된 목적함수:

$$
\mathcal{L}_\text{Flash-GRPO}(\theta) = \mathbb{E}_{t, \mathbf{x}_0} \left[ \frac{1}{\lambda(t)} \cdot \lambda(t) \cdot \hat{A} \cdot \log \pi_\theta(\mathbf{x}_t | \mathbf{x}_{t+1}) \right] = \mathbb{E}_{t, \mathbf{x}_0} \left[ \hat{A} \cdot \log \pi_\theta(\mathbf{x}_t | \mathbf{x}_{t+1}) \right]
$$

즉, 각 타임스텝의 그래디언트 기여를 $1/\lambda(t)$로 스케일링하여 균일한 최적화 기여를 보장한다.

---

### 2-3. 모델 구조

연구는 Wan2.1 패밀리를 기반 모델로 사용하며, 1.3B 및 대규모 14B 변형 모두에서 방법의 유효성을 검증한다.

훈련에는 DanceGRPO의 프롬프트 데이터셋을 활용하며, 별도의 300개 프롬프트 분할을 평가용으로 사용한다.

평가 지표로는 **HPSv3** (Human Preference Score v3), **VBench** (비디오 품질 종합 벤치마크)를 사용하며, 미적 품질(aesthetic quality), 이미지 품질(image quality), 주체 일관성(subject consistency), 객체 클래스(object class)를 평가하여 RL 파인튜닝이 백본 모델의 생성 능력을 유지하는지 확인한다.

---

### 2-4. 성능 향상

효율성 비교에서 Flash-GRPO는 훈련 비용에서 **6배 가속**을 달성하면서도 더 높은 평가 성능을 달성한다.

KL 정규화 적용 시, Flash-GRPO는 HPSv3에서 약 **5.35** 점수를 달성하며 Flow-GRPO-Fast1보다 빠른 수렴과 더 높은 성능 상한을 보인다.

Flash-GRPO는 향상된 시간적 역동성(열차 시퀀스), 개선된 시각적 품질(아이언맨), 더 나은 프롬프트 충실도(고양이와 그릇)를 가진 비디오를 생성한다.

코드와 함께 8 GPU 버전도 공개되어, 약 40시간 만에 동일한 성능을 달성할 수 있다.

| 방법 | 훈련 비용 | HPSv3 점수 | 안정성 |
|---|---|---|---|
| Flow-GRPO (Full Trajectory) | 높음 (기준) | 기준 | 보통 |
| Flow-GRPO-Fast1 (슬라이딩 윈도우) | 낮음 | 낮음 | 불안정 |
| **Flash-GRPO** | **6× 감소** | **최고 (SoTA)** | **안정적** |

---

### 2-5. 한계점

논문에서 명시적으로 언급하거나 검색 결과에서 추론할 수 있는 한계:

1. **단일 보상 모델 의존성:** 단일 보상 함수에 주로 의존하는 기존 접근법들은 특정 지표에 과적합하여 보상 해킹(reward hacking)이나 다중 목적 간 불균형 최적화 문제가 발생할 수 있다.

2. **평가 분포의 제한:** 훈련 및 평가에 DanceGRPO의 프롬프트 데이터셋을 사용하는 제약이 있어, 완전히 다른 도메인에서의 검증은 미흡하다.

3. **단일 스텝 가정의 근사:** 단일 스텝에서의 ODE→SDE 전환은 전체 디퓨전 궤적을 근사하므로, 특정 복잡한 동작이나 장면에서 한계가 있을 수 있다.

4. **모델 아키텍처 의존성:** Wan2.1 기반으로 검증되어 다른 아키텍처(예: HunyuanVideo 외)로의 직접 이전(transfer)에 대한 검증이 제한적이다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 스케일 일반화

Flash-GRPO는 1.3B부터 14B 파라미터에 이르는 모델을 대상으로 평가되어, 훈련 효율성과 안정성을 크게 향상시키며 제한된 계산 예산에서 풀 궤적 훈련 대비 우수한 정렬 성능을 달성하고 최신 결과를 확립한다.

이는 모델 규모에 대한 **강한 일반화 능력**을 시사한다.

### 3-2. 프롬프트 일반화

Flash-GRPO는 훈련 및 평가 모두에서 더 빠른 수렴과 더 높은 성능 상한을 달성하는 반면, Flow-GRPO-Fast1은 **제한된 일반화(limited generalization)** 속에서 조기에 수렴이 정체된다.

이 비교는 Flash-GRPO가 보지 않은 평가 프롬프트에서도 안정적인 성능을 보임을 의미한다.

### 3-3. 태스크 일반화 가능성

Iso-temporal Grouping과 Temporal Gradient Rectification은 비디오 디퓨전에 특화된 기법이지만, 그 원리는 일반적인 플로우 매칭(flow matching) 기반 모델에도 적용 가능하다. DanceGRPO와 Flow-GRPO가 텍스트-투-이미지, 텍스트-투-비디오, 이미지-투-비디오 등 다양한 태스크에서 GRPO의 적용 가능성을 보였듯, Flash-GRPO의 두 핵심 메커니즘 역시 이미지 생성 도메인 등으로의 확장 가능성을 내포한다.

### 3-4. 다양한 보상 모델로의 일반화

현재 HPSV3 및 Wan2.1-1.3B 기반 보상 모델을 사용하지만, 프레임워크 자체는 다양한 보상 모델(미적 품질, 동작 품질, 텍스트 정렬 등)과 결합 가능한 구조를 가진다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 특징 | 한계 |
|---|---|---|---|
| **DDPO** (Black et al., 2023) | 디노이징을 MDP로 모델링 | 온라인 정책 최적화 | 안정성 문제 |
| **Diffusion-DPO** (Wallace et al., 2024) | DPO를 디퓨전에 적용 | 명시적 보상 모델 불필요 | 오프라인 방식 한계 |
| **DanceGRPO** (Xue et al., 2025) | GRPO → 시각 생성 확장 | 통합 프레임워크, ODE-SDE 전환 | 계산 비용 높음 |
| **Flow-GRPO** (Liu et al., 2025) | 플로우 매칭에 온라인 RL | 플로우 모델 특화 | 계산 비용 높음 |
| **BranchGRPO** (Li et al., 2025) | 구조적 브랜칭 기반 GRPO | 안정성·효율성 개선 | 구조 복잡성 |
| **MixGRPO** (Li et al., 2025) | Mixed ODE-SDE 전략 | 계산 오버헤드 감소 | 부분적 개선 |
| **TempFlow-GRPO** (He et al., 2025) | 시간적 가중치 부여 | 타임스텝별 크레딧 할당 | 완전한 효율화 미흡 |
| **Flash-GRPO** (He et al., **2026**) | 단일 스텝 + Iso-Grouping + Gradient Rectification | **6× 가속 + SoTA 성능** | 단일 보상 의존, 평가 범위 제한 |

TempFlow-GRPO는 균일한 크레딧 할당을 사용한 희소 터미널 보상의 한계를 부각시키며, 디노이징 스텝에 걸친 시간적 가중치 부여를 제안하였다.

DanceGRPO는 공유 노이즈 전략으로 정책 유도 개선을 분리하고, MixGRPO는 계산 오버헤드를 줄이는 혼합 ODE-SDE 전략을 채택했다. Chunk-GRPO와 E-GRPO는 결과 보상의 희소성을 완화하기 위한 스텝 집계를 제안하고, TempFlow-GRPO와 TreeGRPO는 더 세밀한 크레딧 할당을 위한 분기 구조를 활용한다.

Flash-GRPO는 이 흐름에서 **이론적으로 두 가지 근본 원인을 명확히 규명**하고, 각각에 대한 원칙 기반 해결책을 제시한 것이 차별점이다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 연구에 미치는 영향

1. **비디오 RL 정렬의 접근성 확대**
Flash-GRPO는 14B 파라미터 비디오 디퓨전 모델 정렬의 GPU 비용을 수백 훈련일에서 그 일부로 줄여, 고품질 비디오 RL 정렬을 실용적으로 만든다.

2. **이론적 기반 강화**
두 메커니즘은 단일 스텝 훈련에서 실질적으로 감소된 계산 비용으로 훈련 안정성과 풀 궤적 방법과 비교 가능한 성능을 달성하며, 효율성-품질 트레이드오프를 제거했다는 것을 실험적으로 입증한다.

3. **ICML 2026 채택의 의의**
이 방법은 ICML 2026에서 채택되어, 최상위 머신러닝 학술대회에서 동료 심사를 통과한 최초의 비디오 디퓨전 정렬 프레임워크 중 하나로 자리매김한다.

### 5-2. 향후 연구 시 고려할 점

1. **다중 보상 결합(Multi-Reward):**
단일 보상 함수에 의존하는 기존 접근은 특정 지표 과적합, 보상 해킹, 다중 목적 간 불균형 최적화를 초래할 수 있으므로, Flash-GRPO와 다중 보상 함수(미적 품질 + 동작 품질 + 텍스트 정렬)를 결합하는 연구가 필요하다.

2. **이미지 생성으로의 확장:**
Flash-GRPO의 핵심 원리(Iso-temporal Grouping, Temporal Gradient Rectification)는 이미지 생성 디퓨전 모델에도 적용 가능하며, 이에 대한 체계적 검증이 필요하다.

3. **프로세스 보상 모델(PRM) 통합:**
TempFlow-GRPO가 지적했듯, 디노이징 스텝에 걸친 시간적 가중치 부여(temporally-aware weighting)를 Flash-GRPO와 결합하면, 더욱 정교한 크레딧 할당이 가능할 것이다.

4. **다양한 아키텍처 검증:**
현재 Wan2.1 기반 검증에 집중되어 있으므로, HunyuanVideo, FLUX 등 다양한 기반 모델로의 이전 가능성(transferability)을 실험적으로 검증해야 한다.

5. **이론적 수렴 보장:**
단일 스텝 근사가 전체 궤적 최적화와 얼마나 긴밀하게 연결되는지에 대한 이론적 수렴 증명 연구가 필요하다.

6. **보상 해킹 방지:**
DanceGRPO와 PREF-GRPO 등이 보상 해킹을 완화하기 위해 ODE-SDE 전환이나 쌍별 선호도(pairwise preferences)를 탐색하듯, Flash-GRPO에서도 장기 훈련 시 보상 해킹 현상에 대한 체계적 분석이 요구된다.

---

## 참고 자료 출처

| # | 출처 | 유형 |
|---|---|---|
| 1 | [arXiv:2605.15980](https://arxiv.org/abs/2605.15980) — Flash-GRPO 원본 논문 | **핵심 논문** |
| 2 | [arxiv.org/html/2605.15980](https://arxiv.org/html/2605.15980) — Flash-GRPO HTML 전문 | 논문 본문 |
| 3 | [ICML 2026 Poster](https://icml.cc/virtual/2026/poster/63629) — 공식 학회 발표 페이지 | 학회 발표 |
| 4 | [Flash-GRPO GitHub](https://github.com/Shredded-Pork/Flash-GRPO) — 공식 코드 | 코드 |
| 5 | [Flash-GRPO 프로젝트 페이지](https://shredded-pork.github.io/Flash-GRPO.github.io/) | 데모 |
| 6 | [ResearchGate](https://www.researchgate.net/publication/404951028) — PDF 전문 | 논문 PDF |
| 7 | [studio.aifilms.ai — Flash-GRPO 블로그](https://studio.aifilms.ai/blog/flash-grpo-video-diffusion-alignment-icml-2026) | 해설 |
| 8 | arXiv:2505.07818 — **DanceGRPO** (Xue et al., 2025) | 비교 논문 |
| 9 | arXiv:2505.05470 — **Flow-GRPO** (Liu et al., 2025) | 비교 논문 |
| 10 | arXiv:2509.06040 — **BranchGRPO** (Li et al., 2025) | 비교 논문 |
| 11 | arXiv:2508.04324 — **TempFlow-GRPO** (He et al., 2025) | 비교 논문 |
| 12 | MDPI Sensors 2026 — **Flow-Multi** (다중 보상 프레임워크) | 관련 논문 |
| 13 | OpenReview ICLR 2026 — BranchGRPO 정식 논문 | 비교 논문 |
