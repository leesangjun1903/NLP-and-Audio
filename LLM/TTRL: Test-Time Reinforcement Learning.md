# TTRL: Test-Time Reinforcement Learning

## 1. 핵심 주장 및 주요 기여 요약

**TTRL(Test-Time Reinforcement Learning)** 은 정답 레이블이 없는 테스트 데이터만으로 LLM을 강화학습(RL)으로 훈련시키는 새로운 패러다임입니다. 핵심 아이디어는 **다수결 투표(majority voting)** 로 의사 레이블(pseudo-label)을 추정하여 규칙 기반 보상(rule-based reward)을 계산하고, 이를 GRPO 같은 RL 알고리즘의 학습 신호로 사용하는 것입니다.

**주요 기여는 다음 세 가지로 요약됩니다.**
- 다수결 투표가 비지도 환경에서 효과적인 보상 추정기로 작동함을 실증.
- TTRL이 자신의 학습 신호 상한(maj@n)과 자기 학습(self-training) 상한을 **초과**하며, 정답 레이블을 사용한 직접 RL(RL Leakage)의 성능에 근접.
- Qwen2.5-Math-7B가 AIME 2024에서 pass@1 16.7 → 43.3으로 **약 159.3% 향상**, 평균 76% 성능 개선 (출처: 본 논문 Table 1, Table 4).

---

## 2. 문제 정의·방법론·구조·성능·한계

### 2.1 해결하고자 하는 문제

기존 RLHF나 RLVR(Reinforcement Learning with Verifiable Rewards) 기법은 모두 **레이블된 데이터**나 **검증 가능한 정답**에 의존합니다. 그러나 ARC-AGI-2 같은 새로운 벤치마크에서는 OpenAI o3조차 4%만 풀 수 있고, 라벨링 비용이 폭증합니다. 따라서 논문은 다음 질문을 제기합니다.

> **"테스트 시점에서 정답 레이블 없이 어떻게 RL 보상을 얻을 수 있는가?"**

### 2.2 제안하는 방법 (수식 포함)

#### (1) 정책 최적화 목적 함수

입력 프롬프트 $x$가 주어지면, 모델은 정책 $\pi_\theta(y \mid x)$로부터 $N$개의 후보 출력 $\{y_1, y_2, \ldots, y_N\}$을 샘플링합니다. 그 후 합의(consensus) 출력 $y^*$를 다수결로 도출하여 보상을 계산합니다. RL 목적 함수는 다음과 같습니다.

$$
\max_{\theta} \; \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ r(y, y^*) \big]
$$

파라미터 업데이트는 정책 경사 상승법(policy gradient ascent)으로 수행됩니다.

$$
\theta \;\leftarrow\; \theta + \eta \, \nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ r(y, y^*) \big]
$$

#### (2) 다수결 보상 함수

예측 집합 $\mathcal{P} = \{\hat{y}\_i\}_{i=1}^{N}$에서 가장 빈도가 높은 답을 의사 레이블 $y$로 정합니다.

$$
y \;=\; \arg\max_{\hat{y}_i} \; s(\hat{y}_i, x), \quad s(\hat{y}_i, x) = \sum_{j=1}^{N} \mathbb{1}[\hat{y}_j = \hat{y}_i]
$$

이후 규칙 기반 보상은 다음과 같이 정의됩니다.

$$
R(\hat{y}_i, y) \;=\; \begin{cases} 1, & \text{if } \hat{y}_i = y \\ 0, & \text{otherwise} \end{cases}
$$

### 2.3 모델 구조 및 학습 파이프라인

TTRL은 **TTS(Test-Time Scaling) + TTT(Test-Time Training)** 의 결합 구조입니다.

1. **Label Estimation**: 테스트 질의 $q$에 대해 LLM이 $M$개 응답 $\{\hat{y}_1, \ldots, \hat{y}_M\}$ 생성.
2. **Majority Voting**: 합의 답 $y$ 도출.
3. **Reward Calculation**: 각 샘플 $\hat{y}_i$에 대해 $R(\hat{y}_i, y)$ 계산.
4. **Policy Optimization**: GRPO/PPO/PRIME 등을 사용해 $\theta$ 업데이트.

구현 세부 사항(논문 Section 3.1 기준): 각 프롬프트당 64개 응답을 다수결 투표용으로 샘플링한 뒤 32개를 학습용으로 다운샘플링, AdamW + cosine LR ($5 \times 10^{-7}$), 8 × NVIDIA A100 80GB 사용.

### 2.4 성능 향상 결과

| 모델 | 벤치마크 | 향상 |
|------|---------|------|
| Qwen2.5-Math-7B | AIME 2024 | 16.7 → 43.3 (+159.3%) |
| Qwen2.5-Math-7B | AMC | 38.6 → 67.5 (+74.9%) |
| Qwen2.5-Math-7B | MATH-500 | 50.6 → 84.2 (+66.4%) |
| Qwen2.5-Math-1.5B | MATH-500 | 32.7 → 73.0 (+123.2%) |
| Qwen3-8B | AIME 2024 | 72.5 → 82.5 |

(출처: 본 논문 Table 1, Table 4, Figure 3)

특히 흥미로운 발견은 **TTRL의 avg@64가 backbone의 maj@64를 일관되게 초과**한다는 점입니다(Figure 7). 즉, 자신의 학습 신호 상한선(self-training upper bound)을 넘어서는 "자기-부트스트래핑(self-bootstrapping)" 현상이 관찰됩니다.

### 2.5 왜 작동하는가? — "Lucky Hit" 메커니즘

논문의 핵심 통찰은 **레이블 정확도가 낮아도 보상 정확도는 높을 수 있다**는 것입니다. AIME 2024에서 라벨 정확도는 약 37%지만 보상 정확도는 92%에 달합니다. 이유는 다음과 같습니다.

- 모델 출력이 **분산되어(scattered) 있고** 대부분 틀렸기 때문에, 추정 라벨이 틀려도 그것이 "잘못된 예측"과 다르기만 하면 음의 보상을 정확히 부여.
- RL은 SFT보다 보상 노이즈에 강건함(Razin et al., 2025; Chu et al., 2025).

### 2.6 한계 (논문 §4.3 및 §7)

1. **하이퍼파라미터 민감도**: temperature(0.6 vs 1.0), batch size, episode 수에 매우 민감하며 잘못 설정 시 학습 실패.
2. **사전 지식 의존성**: 백본 모델의 사전 지식이 부족하면 어려운 문제(MATH-500 Level 5에서 향상폭이 75.3%로 Level 1의 175.3%보다 적음)에서 학습이 정체됨.
3. **이론적 분석 부재**: 수렴성, 두 상한선(maj@n, RL Leakage)에 대한 형식적 증명 없음.
4. **데이터 큐레이션 부재**: 커리큘럼 학습 같은 기법이 통합되지 않아 난이도 적응 능력이 제한됨.

(출처: arXiv:2504.16084v3)

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

이 부분은 본 논문이 가장 강조하는 측면이며, 다음의 4가지 증거가 제시됩니다.

### 3.1 분포 외(out-of-distribution) 일반화

논문 Figure 4에 따르면, **AIME 2024에서만 TTRL을 수행한 후** AMC와 MATH-500을 평가했을 때:
- AMC: 39.8 → 60.1
- MATH-500: 52.6 → 75.4

즉, 단일 도메인 적응이 다른 벤치마크로 전이됩니다. 이는 TTRL이 **데이터셋 특정 과적합이 아니라 일반화 가능한 추론 능력**을 획득함을 시사합니다.

### 3.2 모델 크기에 따른 자연스러운 스케일링

1.5B → 7B → 32B로 갈수록 향상 폭이 증가:

$$
\Delta_{\text{Qwen2.5-Math}} = \begin{cases} +17.5 & (1.5\text{B}) \\ +23.8 & (7\text{B}) \\ +18.7 & (32\text{B, vanilla}) \end{cases}
$$

큰 모델일수록 다수결이 더 정확한 의사 레이블을 생성하므로, 자기-개선(self-evolution)의 품질이 높아집니다.

### 3.3 maj@n 상한을 초과하는 자기-부트스트래핑

전통적 self-training은 초기 모델의 maj@n에 의해 상한이 결정되지만, TTRL은 이를 **20포인트 이상 초과**합니다(Figure 6). 이는 다음 두 가지 이유 때문입니다.

- 보상은 라벨보다 **밀집된 신호(dense signal)** 를 제공.
- 온라인 학습이므로 모델이 개선되면서 라벨 품질도 동적으로 향상.

### 3.4 다양한 RL 알고리즘과의 호환성

GRPO, PPO, PRIME 모두 유사한 학습 곡선을 보임(Figure 5)으로써 TTRL이 알고리즘에 종속되지 않은 **일반적 프레임워크**임을 입증.

### 3.5 다른 모달리티/도메인으로의 확장 가능성

후속 연구 AQA-TTRL은 오디오 질의응답 도메인으로 TTRL을 확장하여 다수결 투표 기반 자기 적응이 비텍스트 도메인에서도 작동함을 확인했습니다(출처: arXiv:2510.05478, "AQA-TTRL: Self-Adaptation in Audio Question Answering with Test-Time Reinforcement Learning").

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향: 패러다임 전환의 신호탄

Silver & Sutton(2025)이 제창한 **"경험의 시대(era of experience)"** 비전과 부합하며, 다음의 패러다임 전환을 가속화합니다.

1. **RLIF (Reinforcement Learning from Internal Feedback)** 의 확립: 외부 보상 모델 없이 모델 내부 합의로 학습.
2. **연속적 자기-진화(continuous self-evolution)**: 배포 후에도 정적이지 않은 LLM 시스템.
3. **데이터 라벨링 비용 절감**: 어려운 도메인일수록 더 큰 효용.

### 4.2 향후 연구 시 고려해야 할 점

#### (1) False-Popular Mode Collapse (가짜 다수 모드 붕괴)

$T^3RL$ 논문(arXiv:2603.02203)이 지적했듯, 모델 추론에 편향이 있으면 **합의가 정답과 상관관계를 잃을 수 있습니다**. 이를 위해 코드 인터프리터 같은 **외부 검증 도구**의 통합이 필요합니다.

#### (2) 다수결 정보 손실

DARE(Distribution-Aware Reward Estimation, arXiv:2601.21804) 연구는 다수결이 **non-majority correct actions의 정보를 폐기**한다는 한계를 지적하며, 전체 롤아웃 분포 기반 보상을 제안. AIME 2024에서 25.3% 추가 향상을 보고함.

#### (3) 다양성 붕괴(Diversity Collapse)

TEMPO(arXiv:2604.19295)는 TTRL이 pass@K를 손상시킬 수 있음을 지적하며, critic-policy 교대 설계로 이를 완화. OLMO3-7B의 AIME 2024 정확도를 33.0%→51.1%로 끌어올림.

#### (4) 사전 지식 의존성 완화

본 논문 §4.3에서 인정하듯, 백본 모델의 사전 지식이 약하면 TTRL이 실패합니다. 향후 연구는 **커리큘럼 학습**, **데이터 필터링**, 또는 **소량의 인간 피드백 부트스트래핑**을 통합해야 할 것입니다.

#### (5) 수렴성에 대한 이론적 분석

논문 §7에서 직접 언급하듯, TTRL이 두 상한(maj@n, RL Leakage)으로 수렴하는 조건의 **형식적 분석**이 시급합니다.

#### (6) 검증 가능성이 낮은 도메인으로의 확장

수학·코드처럼 답이 명확한 영역 외에, 개방형 추론·과학적 발견·에이전트 작업 등으로 확장 시 **다수결의 의미가 모호**해지므로 새로운 합의 메커니즘이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | TTRL과의 차이점 |
|------|------|-------------|---------------|
| **TTT (Sun et al.)** | 2020 | 분포 변화에 대해 자기지도 보조 손실로 테스트 시점 적응 | 비전 도메인 중심, RL 미사용, LLM 추론 미적용 |
| **Self-Consistency (Wang et al.)** | 2022 | 다수결 투표로 추론 일관성 향상 | 추론 시점에만 사용, 모델 파라미터 업데이트 없음 |
| **LMSI (Huang et al.)** | 2022 | 다수결 답을 SFT로 학습(self-training) | SFT 사용, RL 미사용 → 상한이 초기 maj@n |
| **Self-Rewarding LM (Yuan et al.)** | 2024 | LLM이 LLM-as-a-Judge로 보상을 생성 | DPO 기반, 선호 비교 사용; TTRL은 규칙 기반 보상 |
| **TTT for ARC (Akyürek et al.)** | 2024 | LoRA로 ARC 추상 추론 과제 적응 | 변환된 데이터로 학습, RL 미사용 |
| **DeepSeek-R1 (Guo et al.)** | 2025 | 대규모 RLVR로 long-CoT 능력 학습 | 정답 레이블 필요, 대규모 라벨링 데이터 의존 |
| **EMPO (Zhang et al.)** | 2025 | 잠재 의미 공간에서 예측 엔트로피 최소화 | 엔트로피 기반, 다수결 미사용 |
| **Genius (Xu et al.)** | 2025 | 일반 쿼리만으로 자기 훈련 | 검증·정답 모두 불필요, 더 일반화된 시도 |
| **Absolute Zero (Zhao et al.)** | 2025 | 자기 플레이로 데이터·라벨 없이 추론 학습 | TTRL은 기존 테스트 데이터에 적응; AZR은 데이터 자체를 생성 |
| **TTRL (본 논문)** | 2025 | **다수결 보상 + 온라인 RL** | 라벨 불필요, 모델 파라미터 업데이트, 자기-부트스트래핑 |
| **DARE (Du et al.)** | 2026 | 분포 인지형 보상 추정 | TTRL의 다수결 한계를 보완(편향 감소) |
| **$T^3RL$ (Liao et al.)** | 2026 | 도구 검증으로 가짜 다수 모드 붕괴 방지 | TTRL + 외부 검증 도구 결합 |
| **TEMPO (2026)** | 2026 | Critic-Policy 교대로 다양성 붕괴 방지 | 보상 보정으로 TTRL의 pass@K 문제 해결 |

(출처: 본 논문 References 및 위 검색 결과)

---

## 참고 자료 (출처)

1. **본 논문**: Zuo et al., "TTRL: Test-Time Reinforcement Learning," arXiv:2504.16084v3 (2025), NeurIPS 2025 Poster — https://arxiv.org/abs/2504.16084
2. **GitHub 공식 저장소**: https://github.com/PRIME-RL/TTRL
3. **OpenReview**: https://openreview.net/forum?id=VuVhgEiu20
4. **NeurIPS 2025 Poster**: https://neurips.cc/virtual/2025/poster/117645
5. **AQA-TTRL**: Zhang et al., "AQA-TTRL: Self-Adaptation in Audio Question Answering with Test-Time Reinforcement Learning," arXiv:2510.05478
6. **DARE**: Du et al., "Distribution-Aware Reward Estimation for Test-Time Reinforcement Learning," arXiv:2601.21804
7. **$T^3RL$**: Liao et al., "Tool Verification for Test-Time Reinforcement Learning," arXiv:2603.02203
8. **TEMPO**: "TEMPO: Scaling Test-time Training for Large Reasoning Models," arXiv:2604.19295
9. **Co-rewarding**: arXiv:2508.00410 (Stable Self-supervised RL for Eliciting Reasoning)
10. **Self-Rewarding RL (CoVo)**: arXiv:2506.08745
11. **EMPO**: Zhang et al., "Right Question is Already Half the Answer," arXiv:2504.05812
12. **Genius**: Xu et al., arXiv:2504.08672
13. **Absolute Zero**: Zhao et al., arXiv:2505.03335
14. **DeepSeek-R1**: Guo et al., arXiv:2501.12948
15. **TTT for Abstract Reasoning**: Akyürek et al., arXiv:2411.07279

> ⚠️ **주의사항**: 본 분석에서 제시한 수식은 논문의 식 (1)~(3)을 기반으로 LaTeX로 재구성한 것이며, 일부 비교 연구(DARE, $T^3RL$, TEMPO)의 정확한 수치·발표 시점·게재 여부는 arXiv preprint 단계의 정보로, 최종 게재본과 다를 수 있습니다. 향후 인용 시 원 논문의 최신 버전을 직접 확인해 주시기 바랍니다.
