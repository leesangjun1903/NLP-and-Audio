# Single-Rollout Asynchronous Optimization (SAO) for Agentic Reinforcement Learning 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 LLM의 에이전틱(Agentic) 강화학습을 위한 **Single-rollout Asynchronous Optimization (SAO)** 를 제안합니다. 기존 비동기 RL 시스템들이 처리량(throughput)에만 집중하고 학습 안정성과 효과성을 간과한 반면, SAO는 두 가지 핵심 문제인 **훈련 불안정성**과 **오프-폴리시(off-policy) 드리프트**를 동시에 해결합니다.

### 주요 기여 (4가지)

| 기여 항목 | 설명 |
|---|---|
| **DIS (Direct Double-Sided Importance Sampling)** | 토큰 수준 이중 클리핑으로 정책 지연 안정화 |
| **Single-Rollout Sampling** | 그룹 샘플링 대신 프롬프트당 1개의 롤아웃 사용, off-policy 감소 |
| **개선된 가치 모델 학습** | Frozen-Attention 전략 + 더 빠른 critic 업데이트 |
| **Skip-Observation Token-level GAE** | 에이전틱 다중-턴 궤적에 특화된 어드밴티지 추정 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 동기식 RL의 비효율성

기존 동기식 RL 파이프라인(PPO, GRPO 등)은 전체 배치의 롤아웃이 완료될 때까지 학습을 대기합니다. 에이전틱·코딩 작업에서는 롤아웃 길이가 극단적으로 가변적이어서 GPU 자원이 **"가장 느린 롤아웃"을 기다리며 낭비**됩니다.

#### 문제 2: 비동기 RL에서의 Policy Lag (오프-폴리시)

비동기 RL에서는 롤아웃 생성 중에 모델이 여러 차례 업데이트될 수 있습니다. 이로 인해:
- 정확한 행동 확률 $\pi_{\theta_{old}}$ 추적이 불가능
- 과거 체크포인트 $\{\pi_{\theta^{(1)}\_{old}}, \ldots, \pi_{\theta^{(N)}_{old}}\}$ 관리가 메모리상 비현실적

#### 문제 3: GRPO의 구조적 비적합성

GRPO는 그룹 내 상대적 보상으로 어드밴티지를 추정하므로:
- 그룹 완성을 기다려야 하는 **암묵적 동기화 장벽** 발생
- 단일 피드백만 제공되는 **실제 온라인/에이전틱 환경과 구조적으로 불일치**

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 기본 RL 최적화 목표

모델 $\pi_\theta$가 쿼리 $q$에 대해 응답 $y = [y_1, \ldots, y_{|y|}]$를 생성할 때의 통합 목표:

$$\mathcal{L}_{PPO} = \mathbb{E}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

여기서 $r_t(\theta) = \frac{\pi_\theta(y_t|q, y_{<t})}{\pi_{\theta_{old}}(y_t|q, y_{<t})}$

#### (B) PPO의 GAE (Generalized Advantage Estimation)

$$\hat{A}_t^{GAE} = \sum_{l=0}^{|y|-t-1}(\gamma\lambda)^l \delta_{t+l}$$

여기서 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

비평가(Critic) $V_\phi$는 다음 손실로 훈련:

$$\mathcal{L}_\phi^{VF} = \mathbb{E}\left[(V_\phi(q, y_{<t}) - R)^2\right]$$

#### (C) SAO의 핵심: DIS (Direct Double-Sided Importance Sampling)

기존 $\pi_{\theta_{old}}$ 대신 롤아웃 정책 $\pi_{rollout}$을 직접 사용:

$$r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{rollout}(a_t|s_t)\right) \tag{2}$$

**이중 클리핑 보정 함수** (토큰이 신뢰 구간 외부이면 그래디언트를 완전 차단):

$$f(x; \epsilon_\ell, \epsilon_h) = \begin{cases} x, & \text{if } 1 - \epsilon_\ell < x < 1 + \epsilon_h \\ 0, & \text{otherwise} \end{cases} \tag{3}$$

**SAO의 최종 최적화 목표:**

$$L(\theta) = \hat{\mathbb{E}}_t\left[f(r_t(\theta), \epsilon_l, \epsilon_h)\hat{A}_t \log \pi_\theta(a_t|s_t)\right] \tag{1}$$

> 실험 하이퍼파라미터: 추론 태스크 $\epsilon_{low}=0.3, \epsilon_{high}=5.0$, 코딩 태스크 $\epsilon_{low}=0.8, \epsilon_{high}=3.0$

#### (D) Skip-Observation Token-level GAE (에이전틱 태스크 특화)

에이전틱 궤적: $T = [a_0, o_0, a_1, o_1, \ldots]$ 에서 관찰 토큰 $o_i$는 모델이 생성하지 않으므로, 이를 우회하는 어드밴티지 추정:

$$\hat{A}(a_{i,N}) = \delta + \gamma\lambda\hat{A}(a_{i+1,0}) \tag{4}$$

TD 잔차 $\delta$는 관찰 구간을 건너뛰어 계산:

$$\delta = r_t + \gamma V(a_{i+1,0}) - V(a_{i,N}) \tag{5}$$

여기서 $a_{i,N}$은 액션 $i$의 마지막 토큰, $a_{i+1,0}$은 다음 액션의 첫 번째 토큰

#### (E) Length-Adaptive GAE

$$\lambda_{policy} = 1 - \frac{1}{\alpha \cdot l}, \quad \alpha = 1.5$$

---

### 2.3 모델 구조

SAO는 **Actor-Critic 구조**를 채택하며, 다음 설계 원칙을 따릅니다:

```
┌─────────────────────────────────────────────────────┐
│              SAO 시스템 아키텍처                      │
├──────────────┬──────────────────────────────────────┤
│  Rollout     │  - πrollout이 비동기적으로 롤아웃 생성  │
│  Engine      │  - 완료 즉시 학습 큐에 추가            │
│              │  - token-level log-prob 저장           │
├──────────────┼──────────────────────────────────────┤
│  Actor       │  - 정책 πθ (LLM 기반)                 │
│  (Policy)    │  - 학습률: 1×10⁻⁶                     │
│              │  - 배치당 1회 업데이트                 │
├──────────────┼──────────────────────────────────────┤
│  Critic      │  - 가치 모델 Vφ                        │
│  (Value)     │  - Frozen-Attention 전략              │
│  Model       │  - MoE projection만 업데이트           │
│              │  - 배치당 K=2회 업데이트 (TTUR)        │
│              │  - 학습률: 5×10⁻⁶                     │
└──────────────┴──────────────────────────────────────┘
```

**가치 모델의 핵심 설계 원칙:**
1. **Frozen-Attention**: Attention 레이어 파라미터 동결 → MoE projection만 최적화
   - 근거: Full-Attention 레이어에서 그래디언트 폭발이 집중적으로 발생함을 실험으로 확인
2. **Faster Value Update (TTUR)**: 정책 1회 업데이트당 비평가 $K=2$회 업데이트
3. **Scaled Value Pretraining**: 가치 모델 사전학습 데이터 규모 증가로 cold start 문제 해결

---

### 2.4 성능 향상

#### 수학 추론 벤치마크 (Table 1)

| 모델 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
|---|---|---|---|---|
| SFT Baseline | 80.4% | 53.3% | 75.2% | 53.3% |
| GRPO | 84.2% | 54.8% | 76.0% | 55.8% |
| **SAO (ours)** | **97.3%** | **74.8%** | **88.3%** | **74.0%** |

- GRPO 대비 AIME2025에서 **+13.1%p** 향상
- BeyondAIME에서 **+20.0%p** 향상

#### 코딩 벤치마크 (Table 2) - SWE-Bench Verified

| 모델 | Accuracy |
|---|---|
| Qwen3-30B-A3B | 23.0% |
| + GRPO (w/ DIS) | 27.0% |
| **+ SAO (ours)** | **29.8%** |

#### 훈련 안정성

- **Vanilla GRPO**: 약 160 스텝에서 **훈련 붕괴(collapse)**
- **GRPO + DIS**: 안정적 훈련 가능하나 약 400 스텝 이후 SAO에 비해 성능 낮음
- **SAO**: **1,000 스텝 이상 안정적 훈련** 가능

#### Ablation Study 결과 (Table 4)

| 변형 | AIME2025 | BeyondAIME |
|---|---|---|
| **SAO (full)** | **97.3** | **74.8** |
| w/o Faster value (K=1) | 95.0 | 69.8 |
| w/o Frozen attention | 90.6 | 74.5 |
| Vanilla VAPO (w/o DIS) | 91.3 | 69.0 |
| Running mean baseline | 79.8 | 55.3 |

---

### 2.5 한계점

1. **모델 규모 의존성**: Qwen3-30B-A3B 기반 실험에 집중 → 소규모 모델 또는 비에이전틱 RLHF 환경으로의 전이 가능성 불명확
2. **인프라 요구사항**: 토큰 수준 행동 확률을 비동기 생성 중에도 안정적으로 보존하는 인프라 필요
3. **온라인 학습 실험의 제한**: 실제 사용자 환경이 아닌 시뮬레이션된 선호 변화만 검증 → 실제 배포 시 강화된 모니터링·프라이버시 검토 필요
4. **가치 모델 의존**: 강한 사전학습된 가치 모델이 없으면 cold start 문제 발생 가능
5. **밀도 높은 보상 환경**: 짧은 롤아웃과 밀도 높은 보상 환경에서의 성능 미검증

---

## 3. 일반화 성능 향상 가능성

SAO가 일반화에 기여하는 메커니즘을 다각도로 분석합니다.

### 3.1 Single-Rollout의 일반화 향상 원리

그룹 기반 방법(GRPO)의 어드밴티지 추정:

$$\hat{A}_i^{GRPO} = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

이 추정은 **같은 프롬프트 내 다른 샘플들과의 상대적 비교**에 의존하므로, 그룹 구성에 따라 어드밴티지 추정이 달라지는 **편향(bias) 문제**가 있습니다. 반면 SAO의 가치 모델 기반 어드밴티지는 **절대적 상태 가치**로 추정되어 더 안정적인 학습 신호를 제공합니다.

### 3.2 Off-Policy 감소와 일반화의 관계

비동기 환경에서 그룹 샘플링이 유발하는 off-policy 문제:

$$\text{Policy Lag} \propto \text{Group Wait Time} = \max_{j \in G}(\text{generation time}_j)$$

Single-rollout은 이 대기 시간을 제거하여:

$$\text{Policy Lag}_{SAO} \approx \text{individual generation time} \ll \text{Policy Lag}_{GRPO}$$

Policy lag가 줄어들수록 현재 정책에 더 적합한 데이터로 학습되어 **과거 정책에 과적합(overfit)되는 위험이 감소**합니다.

### 3.3 온라인 학습(Non-stationary) 환경에서의 일반화

논문은 **시뮬레이션된 온라인 학습** 실험(글쓰기 스타일 선호도 변화: Cute → Chuunibyou → Classical)을 통해 일반화를 검증합니다.

보상 함수:

$$r = r_{quality} \times r_{style}, \quad r_{quality}, r_{style} \in \{0, 1\}$$

**러닝 미 베이스라인 어드밴티지** 추정:

$$\hat{A} = r - \mathbb{E}[r_{window}]$$

실험 결과:
- Running Mean 베이스라인: 선호도 변화 후 **적응 지연(lag)** 심각
- SAO: 가치 기반 비평가가 보상 분포 변화를 **즉각적으로 추적**, 빠른 정책 재정렬

이는 SAO가 **비정상(non-stationary) 환경에서의 강한 일반화 능력**을 지님을 보여주며, 실제 사용자 피드백이 지속적으로 변하는 실제 배포 시나리오에서의 적용 가능성을 시사합니다.

### 3.4 Frozen-Attention 전략과 일반화

Frozen-Attention 전략은 사전학습된 어텐션 가중치의 **의미론적 표현 능력을 보존**하면서 MoE 레이어만 태스크 특화 학습합니다. 이는 **catastrophic forgetting 방지** 측면에서 일반화에 기여:

$$\theta_{V_\phi} = \{\theta_{attn}^{frozen}, \theta_{MoE}^{trainable}\}$$

이 설계는 전이 학습(transfer learning)에서의 **feature freezing** 전략과 유사하며, 가치 모델이 다양한 태스크 분포에서 강건한 추정을 유지하게 합니다.

### 3.5 Token-level vs. Step-level의 일반화 영향 (Appendix A.1)

| 방법 | AIME2025 | BeyondAIME |
|---|---|---|
| Step-level (Average) | 85.8 | 60.5 |
| Step-level (Last-Token) | 87.3 | 62.8 |
| **Token-level (SAO)** | **89.8** | **66.8** |

토큰 수준 학습이 **더 세밀한 감독 신호**를 제공하여 복잡한 추론 궤적의 논리적 전환을 포착, 일반화에 유리합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (A) 비동기 RL의 새로운 패러다임 제시

SAO는 비동기 RL에서 **효율성과 안정성의 동시 달성이 가능함**을 실증적으로 보여줍니다. 이는:
- 기존 연구들이 효율성 또는 안정성 중 하나만 집중했던 한계를 극복
- 비동기 RL을 실제 대규모 LLM 훈련에 적용하는 **사실상의 기준(de facto baseline)** 역할 가능

#### (B) 에이전틱 AI 훈련 인프라 설계 방향

SAO의 성공적인 GLM-5.2 (750B-A40B) 적용은:
- 수백 억~수천 억 파라미터 모델에 단일 롤아웃 기반 RL이 **실용적임을 산업적으로 검증**
- 향후 에이전틱 AI 훈련 파이프라인 설계 시 **비동기 단일 롤아웃 아키텍처**가 유력한 선택지

#### (C) 가치 모델 연구의 재부상

GRPO 등 비평가-없는(critic-free) 방법이 주류였으나, SAO는:
- **잘 훈련된 가치 모델의 중요성을 재조명**
- Frozen-Attention, TTUR 등 LLM 특화 가치 모델 학습 기법의 연구 활성화 촉진

#### (D) 온라인/연속 학습(Continual Learning) 연구 가능성

단일 롤아웃 기반 SAO가 비정상 환경에서 빠른 적응을 보인 것은:
- **지속적 온라인 학습(continual online RL)** 연구의 중요한 선례
- 실시간 사용자 피드백 기반의 개인화 LLM 훈련으로의 확장 가능성

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표연도 | 방법 | 장점 | 한계 |
|---|---|---|---|---|
| **PPO** (Schulman et al., 2017) | 2017 | Actor-Critic + 클리핑 | 안정적 | 동기식, 가치 모델 필요 |
| **GRPO** (Shao et al., 2024; DeepSeek-AI, 2024) | 2024 | 그룹 상대 보상 정규화 | 비평가 불필요, 구현 단순 | 그룹 완성 대기, 비동기 부적합 |
| **RLOO** (Ahmadian et al., 2024) | 2024 | REINFORCE + leave-one-out baseline | 간단한 분산 감소 | 여전히 그룹 샘플 필요 |
| **Async RLHF** (Noukhovitch et al., 2024) | 2024 | 온라인 off-policy RLHF 특성 분석 | 효율성 분석 | 알고리즘 개선 미흡 |
| **VAPO** (Yue et al., 2025) | 2025 | Length-adaptive GAE + clip-higher | 분산 감소 | 비동기 환경에서 불안정 |
| **AReaL** (Fu et al., 2025) | 2025 | 완전 분리된 롤아웃-학습, staleness-aware | 처리량 최적화 | 알고리즘 안정성 부차적 |
| **ROLL Flash** (Lu et al., 2025) | 2025 | 세밀한 병렬성, 롤아웃-학습 분리 | 시스템 효율성 | 알고리즘 설계 미집중 |
| **MobileRL** (Xu et al., 2025) | 2025 | difficulty-adaptive GRPO for GUI | 멀티턴 GUI 에이전트 특화 | 도메인 특화, 그룹 기반 |
| **SPO** (Xu & Ding, 2025) | 2025 | Single-stream policy optimization | 단일 롤아웃 | 데이터 난이도 사전 정보 필요 |
| **DCPO** (Yang et al., 2025) | 2025 | Dynamic clipping policy optimization | 적응적 클리핑 | 동기식 중심 |
| **GSPO** (Zheng et al., 2025) | 2025 | Group sequence-level importance weighting | 시퀀스 레벨 안정화 | 동기식 중심 |
| **SAO (본 논문)** | 2026 | Single-rollout + DIS + Frozen-Attention + Skip-Obs GAE | 비동기 안정성 + 효과성 동시 달성 | 대규모 모델 한정 검증 |

**SAO의 차별점 요약:**

$$\underbrace{AReaL/ROLL\ Flash}_{\text{효율성 우선, 안정성 부차}} \xrightarrow{\text{SAO}} \underbrace{\text{효율성 + 안정성 + 효과성}}_{\text{3가지 동시 달성}}$$

---

### 4.3 향후 연구 시 고려할 점

#### ① 소규모 모델로의 전이 가능성 검증

SAO는 Qwen3-30B-A3B에서만 검증됨. 향후 연구 방향:
- 7B, 13B 등 소규모 MoE/Dense 모델에서의 Frozen-Attention 전략 유효성 검증
- 가치 모델 사전학습 규모와 성능의 **스케일링 법칙(scaling law)** 분석

#### ② 가치 모델 사전학습 최적화

논문은 가치 모델 사전학습 규모 확대의 중요성을 언급하나 구체적 방법론 미제시:
- 어떤 데이터 분포로 가치 모델을 사전학습해야 최적인지 연구 필요
- Self-play 또는 synthetic rollout을 활용한 **가치 모델 자동 사전학습** 방법 탐색

#### ③ 클리핑 하이퍼파라미터의 자동 조정

SAO는 $\epsilon_{low}, \epsilon_{high}$를 수동 설정:
- 태스크 또는 정책 발산 정도에 따른 **적응적 클리핑 경계 조정** 알고리즘 연구 필요
- DCPO (Yang et al., 2025)의 동적 클리핑과 결합한 하이브리드 접근법

#### ④ 보상 신호의 품질 및 다양성

현재 SAO는 이진(binary) 또는 단순 보상 신호 사용:
- **밀도 높은 보상(dense reward)** 환경 (예: 프로세스 보상 모델)에서의 성능 검증 필요
- 다중 목적 보상 함수와의 결합 방법 연구

#### ⑤ 실제 온라인 배포 환경에서의 안전성

논문은 시뮬레이션된 선호 변화만 실험:
- 실제 사용자 피드백 스트림에서의 **분포 변화 감지 및 적응 메커니즘** 연구
- 유해한 목표로의 최적화를 방지하는 **안전 제약(safety constraint)** 통합

#### ⑥ 비동기 RL의 이론적 수렴 보장

현재 SAO는 경험적 검증에 집중:
- Policy lag가 있는 비동기 환경에서 SAO의 **수렴성(convergence) 이론적 분석**
- Off-policy 오차의 상한(upper bound) 도출

---

## 참고 자료

**논문 본문 (주요 참고):**
- Hou, Z., Li, Y., Tang, J., & Dong, Y. (2026). *Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*. arXiv:2607.07508v1

**논문 내 인용 참고문헌:**
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300
- DeepSeek-AI. (2024a). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- DeepSeek-AI. (2024b). *DeepSeek-V3 Technical Report*
- Yue, Y. et al. (2025). *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks*. arXiv:2504.05118
- Fu, W. et al. (2025). *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning*. arXiv:2505.24298
- Noukhovitch, M. et al. (2024). *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models*. arXiv:2410.18252
- Lu, H. et al. (2025). *ROLL Flash: Accelerating RLVR and Agentic Training with Asynchrony*. arXiv:2510.11345
- Xu, Y. et al. (2025). *MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents*. arXiv:2509.18119
- Xu, Z. & Ding, Z. (2025). *Single-Stream Policy Optimization*. arXiv:2509.13232
- Yang, S. et al. (2025b). *DCPO: Dynamic Clipping Policy Optimization*. arXiv:2509.02333
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization*. arXiv:2507.18071
- Ahmadian, A. et al. (2024). *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*. arXiv:2402.14740
- Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML 2016
- Ouyang, L. et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022
- Jimenez, C. E. et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*. arXiv:2310.06770
- Yang, A. et al. (2025a). *Qwen3 Technical Report*. arXiv:2505.09388
- Team GLM et al. (2025). *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*. arXiv:2508.06471
