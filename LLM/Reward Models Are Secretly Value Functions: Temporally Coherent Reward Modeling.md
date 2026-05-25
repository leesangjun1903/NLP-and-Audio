# Reward Models Are Secretly Value Functions: Temporally Coherent Reward Modeling

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

표준 RLHF 보상 모델(Reward Model, RM)은 응답의 **최종 토큰(EOS)에서만** 점수를 계산하도록 학습된다. 이는 중간 토큰에서의 풍부한 신호를 완전히 버리는 설계이며, 결과적으로 중간 토큰 출력은 무작위 노이즈에 가깝다. 본 논문의 핵심 주장은 다음과 같다:

> **잘 훈련된 보상 모델의 임의 토큰 $k$에서의 출력은, 지금까지 생성된 부분 응답이 주어졌을 때 최종 보상의 조건부 기댓값(conditional expectation)을 나타내야 한다.**

$$r(x, y_{0..k}) = \mathbb{E}[r(x, y) \mid x, y_{0..k}]$$

이 원리는 보상 모델 출력이 응답이 진행됨에 따라 부드럽고 해석 가능한 궤적을 형성해야 한다는 **시간적 일관성(Temporal Coherence)** 개념으로 이어진다.

### 주요 기여 3가지

| 기여 | 내용 |
|---|---|
| **해석 가능성** | 중간 토큰 정확도를 50%(무작위)에서 88.9%로 향상 |
| **PRM 성능** | 스텝 레이블 없이 ProcessBench에서 평균 F1 44.9% 달성 |
| **PPO 효율화** | 보상+가치 모델 통합으로 GPU 메모리 27%, 학습 시간 19% 절감 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

표준 Bradley-Terry 손실 함수:

$$\mathcal{L}_{BT}(x, y^w, y^l) = -\log(\sigma(r(x, y^w) - r(x, y^l)))$$

이 손실은 오직 최종 토큰($y_K = \text{<EOS>}$)의 출력 $r(x, y)$에만 적용된다. 결과적으로:

- **중간 토큰 출력은 노이즈**: 중간 위치에서의 쌍 비교 정확도가 약 50%(무작위 수준)
- **신호 낭비**: 시퀀스 전체에 걸쳐 분포된 품질 정보를 완전히 무시
- **부분 응답 평가 불가**: 생성 중간에 응답 품질을 추적할 수 없음
- **PPO에서 가치 모델과 보상 모델의 이중 부담**: 별도의 가치 모델(Value Model)이 필요하여 메모리·연산량 증가

### 2.2 제안하는 방법 (수식 포함)

TCRM은 표준 Bradley-Terry 손실에 **두 가지 정규화 항**을 추가한다.

#### (A) 룩어헤드 일관성 손실 (Lookahead Consistency Loss)

중간 토큰 출력이 최종 토큰 출력에 수렴하도록 강제한다.

$$\mathcal{L}_{LA}(x, y) = \sum_{k=0}^{K-1} \left(r(x, y_{0..k}) - \text{SG}[r(x, y)]\right)^2 $$

- $\text{SG}[\cdot]$: Stop-Gradient 연산 (DQN의 Bellman 타깃 분리와 동일한 역할)
- **Lemma 1**에 의해, 이 손실의 최소화 해(minimizer)는 조건부 기댓값 $\mathbb{E}[r(x,y) \mid x, y_{0..k}]$임이 증명됨
- RL과의 대응: **Monte Carlo (MC) 가치 학습** 목표

$$\mathcal{L}_{VM\text{-}MC} = \left(V(x, y_{0..k}) - r(x, y)\right)^2 $$

#### (B) 평활도 손실 (Smoothness Loss)

인접한 토큰 간 출력 차이를 최소화하여 궤적을 부드럽게 만든다.

$$\mathcal{L}_{sm}(x, y) = \sum_{k=1}^{K} \left(r(x, y_{0..k-1}) - \text{SG}[r(x, y_{0..k})]\right)^2 $$

- **Lemma 2**에 의해, 재귀적 최소화는 Doob 마팅게일(martingale)을 형성:
  - $X_t^* = \mathbb{E}[X_{t+1}^* \mid \mathcal{F}_t]$
  - 따라서 $X_t^* = \mathbb{E}[X_T \mid \mathcal{F}_t]$ (타워 성질에 의해)
- RL과의 대응: **Temporal Difference (TD) 가치 학습** 목표

$$\mathcal{L}_{VM\text{-}TD} = \left(V(x, y_{0..k}) - \text{SG}[V(x, y_{0..k+1})]\right)^2 $$

#### (C) 전체 손실 함수

$$\mathcal{L}_{overall}(x, y^w, y^l) = \mathcal{L}_{BT}(x, y^w, y^l) + a_{sm}(\mathcal{L}_{sm}(x, y^w) + \mathcal{L}_{sm}(x, y^l)) + a_{LA}(\mathcal{L}_{LA}(x, y^w) + \mathcal{L}_{LA}(x, y^l)) $$

- 실험에서 $a_{sm} = 0.1$, $a_{LA} = 0.01$ 사용
- **아키텍처, 학습 데이터, 추론 과정의 변경 없이** 단순히 손실 함수에 항을 추가하는 방식

#### (D) 토큰 수준 기여도 분해

$$r(x, y) = r(x, y_0) + \sum_{k=1}^{K} d(y_k \mid x, y_{0..k-1})$$

여기서 $d(y_k \mid x, y_{0..k-1}) = r(x, y_{0..k}) - r(x, y_{0..k-1})$은 각 토큰의 증분 기여도.

### 2.3 모델 구조

- **기반 아키텍처**: 표준 decoder-only transformer (Qwen3 0.6B~32B, Llama 3.1 8B)
- **인과 마스킹(Causal Masking)**: 위치 $k$의 출력이 $y_0, ..., y_k$에만 의존하므로 조건부 기댓값 표현에 자연스럽게 적합
- **추가 헤드 없음**: 기존 단일 스칼라 출력 헤드를 그대로 활용
- **학습 데이터**: Skywork-Reward-Preference-80K-v0.2 (90/10 분할)
- **비교 모델**: Baseline (BT only), ImplicitPRM (DPO), TC-λ

```
[Prompt Tokens] → [Response Tokens y_0, y_1, ..., y_K]
                         ↑           ↑              ↑
                    r(x,y_{0..0}) r(x,y_{0..1}) r(x,y) [EOS]
                    ← 조건부 기댓값으로 정렬 (TCRM) →
```

### 2.4 성능 향상

**Table 1 요약 (주요 결과)**

| 모델 | 방법 | Final 정확도 | Middle 정확도 | Final Delta (MSE) |
|---|---|---|---|---|
| Qwen3 32B | Baseline | 93.5% | 60.1% | 2.10 |
| Qwen3 32B | ImplicitPRM | 82.6% | 78.3% | 4.78 |
| Qwen3 32B | **TCRM** | **93.6%** | **88.9%** | **0.34** |
| Llama 3.1 8B | Baseline | 92.6% | 51.1% | 13.62 |
| Llama 3.1 8B | **TCRM** | **92.5%** | **84.7%** | **1.22** |

**ProcessBench F1 성능 (Table 2)**

| 모델 | 평균 F1 |
|---|---|
| Qwen2.5-Math-PRM-72B (스텝 레이블 사용) | 78.3% |
| ImplicitPRM-DPO (결과 레이블만 사용) | 43.2% |
| **TCRM (결과 레이블만 사용)** | **44.9%** |

**PPO 효율성 (Table 16)**

| 설정 | Peak GPU 메모리 | 학습 스텝 시간 |
|---|---|---|
| Regular PPO | 75.1 GB | 66.4초 |
| TCRM Frozen VM | **54.8 GB (-27%)** | **54.1초 (-19%)** |
| TCRM LoRA | 54.9 GB (-27%) | 67.7초 (+2%) |

### 2.5 한계점

1. **조건부 기댓값의 근사성**: 중간 출력이 엄밀한 조건부 기댓값이 아님. 잔차(residual)와 중간 점수 간에 비제로(non-zero) 상관관계가 남아 있음 (Appendix E 참조)
2. **분포 이동(Distribution Shift)**: 조건부 기댓값이 학습 데이터의 연속 토큰 분포를 가정하므로, PPO 학습 중 정책이 변하면 정확도가 감소할 수 있음 (실험적으로는 영향이 제한적)
3. **수학 도메인 특화 필요**: ProcessBench에서 높은 성능을 위해 수학 전용 데이터로 재학습이 필요
4. **ImplicitPRM과의 소규모 비교**: 작은 모델(0.6B, 1.7B)에서는 ImplicitPRM이 중간 토큰 정확도에서 더 높은 성능을 보이기도 함
5. **엔트로피 인식 평활화 미구현**: 중요 토큰과 필러 토큰을 구분하는 적응적 평활화가 향후 과제로 남음

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 크로스 데이터셋 일반화 (Appendix D)

세 개의 데이터셋(Helpful-Harmless, Skywork-Reward-80K, UltraFeedback)을 교차 평가한 결과:

**Final 토큰 정확도 일반화 비교:**

$$\text{TCRM}_{HH \to Skywork} = 73.4\% \quad \text{vs} \quad \text{Baseline}_{HH \to Skywork} = 74.0\%$$

- TCRM의 최종 토큰 정확도 일반화는 베이스라인과 동등하거나 더 우수함 (특히 UltraFeedback)

**Middle 토큰 정확도 일반화 비교:**

$$\text{TCRM}_{HH \to Skywork} = 66.5\% \quad \text{vs} \quad \text{Baseline}_{HH \to Skywork} \approx 53.8\%$$

> **핵심 발견**: 중간 토큰 정확도의 교차 데이터셋 감소율이 최종 토큰 정확도의 감소율과 **유사한 수준**이다. 즉, TCRM의 중간 토큰 예측 품질이 분포 변화에 대해 최종 토큰 예측만큼 잘 일반화된다.

**MSE 일반화 비교 (Table 11):**

| | Baseline (평균) | TCRM (평균) |
|---|---|---|
| 동일 데이터셋 | ~8.0 | ~0.7 |
| 교차 데이터셋 | ~7.5 | ~0.8 |

TCRM의 MSE가 절대적으로 훨씬 낮으며, 교차 데이터셋에서도 유사한 수준을 유지한다.

### 3.2 스케일링 일반화

RewardBench 2 결과(Table 14)에서 TCRM은 대형 모델에서 더욱 두드러진 성능 향상을 보인다:

- **Qwen3-32B**: TCRM 74.4 vs Baseline 73.4 vs ImplicitPRM 52.0
- **Llama3.1-8B**: TCRM 68.1 vs Baseline 67.3 vs ImplicitPRM 52.5

반면 ImplicitPRM과 TC-λ는 모델 규모가 커질수록 성능이 오히려 하락한다. **TCRM은 모델 크기에 따라 더 효과적으로 스케일링된다.**

### 3.3 도메인 일반화 (수학 추론)

TCRM은 일반 선호도 데이터(Skywork)로 학습 후 수학 데이터로 미세조정 없이는 ProcessBench에서 제한적 성능(F1 23.7%)을 보인다. 그러나 수학 전용 데이터로 학습 시 F1 44.9%로 동 카테고리 최고 성능을 달성한다. 이는 **도메인 특화 미세조정이 일반화 성능 향상의 핵심 변수**임을 시사한다.

### 3.4 PPO 분포 이동에 대한 일반화

> "TCRM estimates the conditional expectation of reward under the assumption that future tokens follow the training distribution. During PPO, generation distributions shift as the policy changes; empirically, however, even a frozen TCRM maintains quality comparable to the more resource-intensive baseline."

동결된(frozen) TCRM이 PPO 학습 중 분포가 변해도 경쟁력 있는 가치 모델로 기능한다. LLM-as-a-Judge 비교에서 TCRM 33.7% 대 Baseline 34.3% ($p = 0.85$)로 통계적으로 동등하다. 이는 **TCRM이 학습된 분포를 벗어난 상황에서도 상당한 일반화 능력**을 보임을 의미한다.

### 3.5 일반화 성능 향상 메커니즘 정리

| 메커니즘 | 설명 |
|---|---|
| **조건부 기댓값 유도** | 이론적으로 보장된 최적해가 데이터 전반에 걸쳐 유의미한 구조를 학습하게 함 |
| **더 많은 학습 신호** | 각 응답의 $K$개 토큰 위치 모두에서 기울기가 흐르므로 표현 학습이 풍부해짐 |
| **평활화 정규화** | 과적합을 억제하고 중간 예측의 분산을 줄임 |
| **가치 함수 구조** | 마팅게일 성질로 인해 시간적으로 일관된 표현이 분포 이동에 더 강건함 |

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 RLHF 보상 모델링 계보

| 논문 | 연도 | 접근 방식 | TCRM과의 비교 |
|---|---|---|---|
| **Ouyang et al. (InstructGPT)** | 2022 | 표준 BT 손실, EOS 토큰 | TCRM의 베이스라인; 중간 토큰 무시 |
| **Lightman et al. (Let's Verify Step by Step)** | 2023 | 인간 스텝 레이블 기반 PRM | 고성능이지만 레이블 비용 극히 높음; TCRM은 결과 레이블만 사용 |
| **Wu et al. (Fine-Grained Human Feedback)** | 2023 | 인간 토큰 수준 레이블 | 밀집 피드백의 유용성 입증; 레이블 비용 문제 동일 |
| **Rafailov et al. (From r to Q*)** | 2024 | DPO 기반, LM이 Q-함수 | TCRM과 유사한 RL 연결; TCRM은 표준 RM 아키텍처 유지 |
| **Yuan et al. (ImplicitPRM)** | 2024 | DPO 스타일 목표로 PRM | 소규모에서 경쟁적이나 대규모에서 성능 하락 |
| **Chan et al. (Dense Reward for Free)** | 2024 | 어텐션 가중치 기반 토큰 기여도 | 추론 시간 기법; 학습 변경 없음. TCRM은 훈련 시 명시적 유도 |
| **Maystre et al. (TC-λ)** | 2025 | 시간 일관성 평활화 (분류 문제) | TCRM과 유사하나 텍스트 분류에만 적용; LM 도메인 확장이 TCRM의 기여 |
| **Zhang et al. (PRM Lessons)** | 2025 | LLM 자동 생성 스텝 레이블 | 고품질이나 비용 높은 외부 모델 의존; TCRM은 결과 레이블만으로 가능 |
| **Yin et al.** | 2025 | 세그먼트 수준 보상 | 토큰→청크 집계; TCRM의 토큰 보상도 집계 가능 |

### 4.2 가치 함수와 보상 모델 통합 관점

**Rafailov et al. (2024) "From r to Q*"** 는 언어 모델이 암묵적으로 Q-함수로 해석될 수 있음을 이론적으로 보였다. TCRM은 이를 한 단계 더 나아가 **명시적 정규화 손실을 통해 보상 모델이 실질적인 가치 함수처럼 행동하도록 유도**하며, PPO에서 실제로 가치 모델을 대체하는 실용적 결과를 제시한다.

### 4.3 차별화 포인트 요약

```
표준 RM          → EOS만 학습, 중간 신호 낭비
ImplicitPRM      → DPO 스타일, 대규모에서 성능 하락
TC-λ             → 평활화만, 수학적 보장 부족, 분류 문제에만 검증
Chan et al.      → 추론 시간 기법, 학습 개선 없음
TCRM (본 논문)   → 이론적 보장(Lemma 1,2), 결과 레이블만 필요,
                   아키텍처 변경 없음, 대규모 스케일링 우수
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 보상 모델 설계 패러다임 전환
TCRM은 보상 모델이 단순한 "최종 점수 분류기"가 아니라 **상태 가치 함수(state value function)**로 설계되어야 한다는 새로운 패러다임을 제시한다. 이는 향후 보상 모델 연구의 기본 설계 원칙으로 자리잡을 가능성이 높다.

#### (2) RLHF 파이프라인 단순화
별도의 보상 모델과 가치 모델을 유지하는 표준 PPO 파이프라인 대신, 단일 TCRM으로 두 역할을 수행하는 **단순화된 RLHF 파이프라인**이 가능해진다. 이는 특히 대규모 모델 학습에서 자원 제약을 완화하는 실용적 가치가 크다.

#### (3) 프로세스 보상 모델(PRM)의 민주화
고가의 스텝 레이블 없이 결과 레이블만으로 PRM 수준의 평가가 가능함을 보여줬다. 이는 **수학, 코딩, 과학 추론 등 다양한 도메인**에서 세밀한 보상 신호를 저비용으로 구축하는 연구를 촉진할 것이다.

#### (4) 온라인 RL과의 결합
토큰 수준 보상을 실시간으로 제공할 수 있으므로, **밀집 토큰 보상(dense token-level reward)**을 활용하는 온라인 RL 알고리즘(예: GRPO, RLOO의 변형)과의 통합 연구가 기대된다.

#### (5) LLM 안전성 및 해석 가능성 연구
$d(y_k \mid x, y_{0..k-1}) = r(x, y_{0..k}) - r(x, y_{0..k-1})$ 를 통한 토큰 수준 기여도 분해는 **안전 연구에서의 유해 콘텐츠 탐지**, 응답 디버깅, 모델 행동 분석에 직접 활용 가능하다.

### 5.2 앞으로 연구 시 고려할 점

#### (A) 이론적 한계 극복
- **조건부 기댓값의 완전한 달성**: 현재 구현에서는 잔차 상관관계가 남아 있어 엄밀한 조건부 기댓값이 아님. 이를 개선하기 위한 **보정(calibration) 기법** 연구가 필요하다.
- **정지 기울기의 최적화 방향**: Stop-Gradient 적용이 수렴 안정성을 높이지만, 이론적으로 최적이 아닐 수 있다. 대안적 타깃 네트워크 설계(예: EMA 기반) 연구가 필요하다.

#### (B) 엔트로피 인식 적응적 평활화
현재 평활화 손실은 모든 인접 토큰에 균등 적용된다. **정보량이 높은 토큰(핵심 단어, 오류 위치)과 필러 토큰을 구분하는 적응적 평활화**가 성능을 더욱 향상시킬 수 있다:

$$\mathcal{L}_{sm}^{adaptive}(x, y) = \sum_{k=1}^{K} w_k \cdot \left(r(x, y_{0..k-1}) - \text{SG}[r(x, y_{0..k})]\right)^2$$

여기서 $w_k$는 토큰의 정보 엔트로피 또는 어텐션 가중치 기반 가중치.

#### (C) 분포 이동 강건성 향상
PPO 학습 중 정책이 변하면 TCRM의 가정(학습 데이터 분포로의 연속)이 깨진다. **온라인 업데이트(online TCRM)** 또는 **분포 강건 학습(distributionally robust training)** 방법론과의 결합이 중요한 연구 과제이다.

#### (D) 정규화 계수의 자동 조정
현재 $a_{sm} = 0.1$, $a_{LA} = 0.01$은 수동 설정값이다. **메타 학습(meta-learning)이나 적응형 가중치 스케줄링**을 통해 최종 토큰 정확도를 유지하면서 중간 토큰 정확도를 극대화하는 자동화된 방법이 필요하다.

#### (E) 다중 보상 신호와의 통합
실제 RLHF 환경에서는 안전성, 유용성, 무해성 등 여러 차원의 보상이 존재한다. **다목적 TCRM(multi-objective TCRM)** 설계 및 각 목표에 대한 시간적 일관성 보장 연구가 필요하다.

#### (F) 더 긴 시퀀스에서의 검증
현재 실험은 최대 1024 토큰으로 제한된다. **장문 응답(Long-form generation)**, 특히 수천 토큰의 추론 체인에서 토큰 수준 신호의 노이즈 특성과 TCRM의 유효성을 검증해야 한다.

#### (G) GRPO, DPO 등 다른 RL 알고리즘과의 통합
본 논문은 PPO에서의 효과만 검증했다. **GRPO(Group Relative Policy Optimization)**, **REINFORCE++**, **RLOO** 등 최근 부상하는 알고리즘에서 TCRM의 가치 모델 역할이 동일하게 유효한지 검증이 필요하다.

---

## 참고자료 및 출처

**주요 참고자료:**

1. **Nikulkov, A. (2026)**. "Reward Models Are Secretly Value Functions: Temporally Coherent Reward Modeling." *arXiv:2604.22981v1* [cs.LG], April 24, 2026. (본 논문)

2. **Ouyang, L. et al. (2022)**. "Training language models to follow instructions with human feedback." *Advances in Neural Information Processing Systems*, 35:27730–27744.

3. **Lightman, H. et al. (2023)**. "Let's verify step by step." *The Twelfth International Conference on Learning Representations.*

4. **Rafailov, R., Hejna, J., Park, R., and Finn, C. (2024)**. "From r to Q*: Your language model is secretly a Q-function." *arXiv:2404.12358.*

5. **Yuan, L. et al. (2024)**. "Free process rewards without process labels." *arXiv:2412.01981.*

6. **Maystre, L. et al. (2025)**. "Incremental sequence classification with temporal consistency." *arXiv:2505.16548.*

7. **Chan, A. J. et al. (2024)**. "Dense reward for free in reinforcement learning from human feedback." *arXiv:2402.00782.*

8. **Liu, C. Y. et al. (2024b)**. "Skywork-reward: Bag of tricks for reward modeling in LLMs." *arXiv:2410.18451.*

9. **Zheng, C. et al. (2024)**. "Processbench: Identifying process errors in mathematical reasoning." *arXiv:2412.06559.*

10. **Malik, S. et al. (2025)**. "Rewardbench 2: Advancing reward model evaluation." *arXiv:2506.01937.*

11. **Schulman, J. et al. (2017)**. "Proximal policy optimization algorithms." *arXiv:1707.06347.*

12. **Wu, Z. et al. (2023)**. "Fine-grained human feedback gives better rewards for language model training." *Advances in Neural Information Processing Systems*, 36:59008–59033.

13. **Yin, Y. et al. (2025)**. "Segmenting text and learning their rewards for improved RLHF in language model." *arXiv:2501.02790.*

14. **Zhang, Z. et al. (2025)**. "The lessons of developing process reward models in mathematical reasoning." *arXiv:2501.07301.*
