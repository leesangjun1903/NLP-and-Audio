# Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **LLM이 자신의 수행 능력(task performance)을 정확히 판단할 수 있도록 훈련하면, 그 메타인지적 신호를 강화학습의 피드백으로 활용하여 더 나은 faithful calibration(FC)을 달성할 수 있다**는 것입니다.

즉, 모델이 자신이 얼마나 잘 답했는지를 스스로 정확하게 평가할 수 있다면, 그 능력이 표현하는 불확실성의 신뢰성(faithfulness)을 향상시키는 데에도 활용될 수 있다는 아이디어입니다.

### 주요 기여 4가지

| 기여 | 내용 |
|------|------|
| **①** RLMF 도입 | 메타인지적 피드백을 활용하여 completion ranking을 정제하는 새로운 RL 패러다임 제안 |
| **②** End-to-End FC 파이프라인 | 수치적·언어적 불확실성을 동시에 faithfully calibrate하는 최초의 종합적 파이프라인 구축 |
| **③** Metacognitive Data Selection | 모델의 자기평가 점수를 이용한 효율적 학습 데이터 선별 방법 제안 (active learning 능가) |
| **④** cMFG* 메트릭 | 기존 cMFG의 한계를 보완한 개선된 faithful calibration 평가 지표 제안 |

---

## 2. 상세 분석: 해결 문제, 제안 방법(수식), 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

**Faithful Calibration (FC)의 미해결 문제:**

LLM은 다음 세 가지 메타인지적 결함을 보입니다:
- 높은 확신을 가지고 환각(hallucination) 발생
- 지식 경계(knowledge boundary) 인식 실패
- 내재적 불확실성과 표현된 불확실성 간의 불일치

기존의 **factual calibration** (신뢰도 ↔ 정확도 정렬)과 달리, **faithful calibration**은 모델이 표현하는 불확실성이 내재적 불확실성(intrinsic uncertainty)과 일치하는지를 다룹니다. 모델은 factually calibrated되어 보이면서도 내부 신념과 불일치할 수 있습니다.

기존 접근법의 한계:
- **MetaFaith**: 프롬프트 기반으로 task accuracy 저하 초래
- **FUT (Faithful Uncertainty Tuning)**: SFT 기반으로 QA 유사 태스크에만 효과적, 일반화 부족
- **Steering**: open-weight 모델에만 적용 가능

---

### 2.2 제안 방법 및 수식

#### 2.2.1 강화학습 프레임워크 기반

GRPO(Group Relative Policy Optimization)를 기반으로 하며, 목적 함수는:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{g=1}^{G}\min\left(\frac{\pi_\theta(r_g|q)}{\pi_{\text{old}}(r_g|q)}A_g,\ \text{clip}\left(\frac{\pi_\theta(r_g|q)}{\pi_{\text{old}}(r_g|q)}, 1-\epsilon, 1+\epsilon\right)A_g\right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta|\pi_{\text{ref}})\right]$$

여기서 각 completion $r_g$는 다음과 같이 구성됩니다:

$$r_g = \{(s_1, c_1), \ldots, (s_{N_g}, c_{N_g})\} \quad \text{for } g = 1, \ldots, G \tag{1}$$

$s_i$는 문장, $c_i$는 해당 문장에 대한 모델이 출력한 confidence score입니다.

---

#### 2.2.2 Gold FC Level 추정 수식

모델 $M$이 completion $r_g$에서 보이는 **실제(gold) faithful calibration level**:

$$F_{\text{gold}}^{(g)} := \frac{\sum_i \mathbf{1}(|c_i - g_i| < \tau)}{N_g} \in [0, 1] \tag{2}$$

- $c_i$: 모델이 표현한 confidence
- $g_i$: sampling consistency 기반 추정 intrinsic confidence
- $\tau$: 허용 오차 임계값 (실험에서 $\tau = 0.10$ 사용)

---

#### 2.2.3 Metacognitive Signal $Z_g$ 수식

모델의 자기예측 정확도를 나타내는 **metacognitive awareness signal**:

$$Z_g := 1 - \left(F_{\text{pred}}^{(g)} - F_{\text{gold}}^{(g)}\right)^2 \in [0, 1] \tag{3}$$

- $F_{\text{pred}}^{(g)}$: 모델이 스스로 예측한 FC 수준 (온라인 추론으로 획득)
- $Z_g = 1$: 완벽한 메타인지적 인식
- 이차식(quadratic) 사용 이유: Brier Score와 유사한 형태로 경험적으로 가장 우수

---

#### 2.2.4 RLMF의 핵심: Metacognitive Advantage Scaling

Advantage를 $A_g = (o_g - \bar{o}) + (f_g - \bar{f})$로 분해한 후, RLMF는 다음과 같이 advantage를 조정합니다:

$$A_g^{\text{RLMF}} = (o_g - \bar{o}) + \begin{cases} (f_g - \bar{f}) \cdot (k + Z_g) & \text{if } f_g > \bar{f} \\ f_g - \bar{f} & \text{otherwise} \end{cases} \tag{4}$$

여기서:
- $f_g = w_{\text{faith}} \cdot r_{\text{faith}}$: 주 학습 목표인 faithfulness 컴포넌트
- $o_g = w_{\text{factual calib}} \cdot r_{\text{factual calib}} + w_{\text{acc}} \cdot r_{\text{acc}} + w_{\text{strict}} \cdot r_{\text{strict}} + w_{\text{soft}} \cdot r_{\text{soft}}$: 보조 품질 제약
- $k = 1$: 상수 (위 평균 faithfulness이지만 낮은 metacognition을 가진 completion이 부당하게 낮게 평가되지 않도록 보장)

**직관**: 평균 이상의 faithfulness를 보이는 completion 중에서, 자기평가를 더 정확히 한 것에 더 높은 가중치를 부여합니다.

---

#### 2.2.5 Faithfulness Reward 수식

$$r_{\text{faith}} = \frac{1}{N_g}\sum_{i=1}^{N_g} 1 - (c_i - g_i)^2 \tag{10}$$

이는 inverted Brier Score 형태로, $c_i$와 $g_i$가 일치할수록 최대화됩니다.

---

#### 2.2.6 Response-level Faithful Calibration Score

$$F_{Q,R}^M := 1 - \frac{1}{L}\sum_{l=1}^{L}|\text{conf}_M^{\text{expressed}}(s_l) - \text{conf}_M^{\text{intrinsic}}(s_l)| \tag{5}$$

---

#### 2.2.7 cMFG* 메트릭

기존 cMFG의 문제(빈 bin, 제한된 support 패널티)를 해결하는 개선 메트릭:

$$\text{cMFG}^* = \frac{\sum_{j=1}^{N_b} w_j \cdot \hat{f}_j}{\sum_{j=1}^{N_b} w_j} \tag{8}$$

이는 다음의 quadrature 근사:

$$\text{cMFG}^* = \frac{1}{|S|}\int_S \mathbb{E}\left[F_{Q,R}^M \mid \text{conf}_M^{\text{intrinsic}}(R) = v\right] dv \tag{9}$$

- Equal-mass bins 사용으로 빈 bin 제거
- Width-proportional weighting으로 실제 모델의 intrinsic confidence 분포 구간만 평가

---

### 2.3 모델 구조 및 파이프라인

**2단계 분리 파이프라인 (Decoupled Two-Stage Framework):**

```
[Stage 1] RLMF + Metacognitive Data Selection
       ↓
  수치적 confidence score의 faithful calibration 달성
       ↓
[Stage 2] Targeted Rewriting (Gemini-2.5-Flash-Lite)
       ↓
  자연어 hedge 표현으로 변환 (컨텍스트 적응형)
```

**Stage 1 상세:**
1. **Pre-SFT**: 출력 포맷 학습 (`<sentence>...<confidence>X</confidence>` 형식)
2. **Metacognitive Data Selection**: 2000개 샘플에서 상위/하위 각 1000개 선별
3. **RLMF 훈련**: GRPO + metacognitive advantage scaling (G=32 completions)

**Stage 2 상세:**
- confidence score → hedge phrase 매핑 (Fagen-Ulmschneider & Tao et al.의 인간 주석 활용)
- bin size 0.05, 각 bin에서 20개 hedge 무작위 샘플링
- 단일 패스 포괄적 편집으로 자연스러운 언어적 불확실성 표현

**Reward 가중치:**
$$\rho_g = w_{\text{strict}} \cdot r_{\text{strict}} + w_{\text{soft}} \cdot r_{\text{soft}} + w_{\text{factual calib}} \cdot r_{\text{factual calib}} + w_{\text{acc}} \cdot r_{\text{acc}} + w_{\text{faith}} \cdot r_{\text{faith}}$$

최종 가중치: $w_{\text{strict}}=3, w_{\text{soft}}=3, w_{\text{factual calib}}=1, w_{\text{acc}}=1, w_{\text{faith}}=12$

---

### 2.4 성능 향상

| 비교 기준 | 성능 향상 |
|-----------|-----------|
| MetaFaith 대비 | +29% cMFG* |
| FUT 대비 | +25% cMFG* |
| 표준 RL 대비 | 최대 +63% |
| GPT-5 대비 | +37% cMFG* |
| Gemini-3.1-Pro 대비 | +17% cMFG* |
| Gemini-3-Flash 대비 | +25% cMFG* |

**주요 성능 결과 (Table 1 기준):**
- Llama3.1-8B + RLMF: cMFG* = **0.84** (baseline 0.60)
- Qwen3-8B + RLMF: cMFG* = **0.83** (baseline 0.54)
- 모든 설정에서 $\text{cMFG}^* \geq 0.80$ 달성
- Task accuracy 및 Brier Score 보존 (MetaFaith, FUT와 달리 degradation 없음)
- 인간 평가: FUT 대비 다양성 98%, 자연성 98%, 도움성 95%, 맥락 적합성 96% win rate

---

### 2.5 한계

**명시적 한계:**
1. **메타인지 자기평가의 한계**: RLMF로 개선되는 것은 "특정 태스크에서의 자기 수행 평가 능력"이며, 이는 광범위한 메타인지 능력과 동일하지 않음
2. **Reward hacking 위험**: Zg를 추가 reward로 사용 시 모델이 낮은 Fpred를 일관적으로 출력하면서 FC가 나쁜 상태로 보상을 극대화할 수 있음
3. **계산 비용**: 각 completion에 대해 온라인으로 $F_{\text{pred}}$ 추론 필요 → 6 GPU 필요 (4 훈련 + 2 추론)
4. **단일 데이터셋 학습의 한계**: PopQA 단일 훈련으로도 강력하나, 다른 training set에서 약간의 성능 차이 존재 (0.80~0.84 범위)
5. **Smarter data selection의 모델 의존성**: Qwen3-8B는 smarter selection에서 더 유의미한 향상 → 모델마다 최적 데이터 선별 전략이 다를 수 있음
6. **검증 불충분성**: 향상된 faithful calibration이 factual correctness의 대체물이 될 수 없으며, 추가적인 사실 검증 필요

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 Out-of-Distribution 일반화의 핵심 증거

**단일 데이터셋 훈련 → 10개 다양한 태스크 평가:**

모델은 **PopQA 하나의 데이터셋만**으로 훈련되었으나, Table 2에서 보듯이 다양한 훈련 데이터셋에서도 일관적으로 강력한 성능을 보입니다:

| 훈련 데이터셋 | Llama3.1-8B cMFG* | Qwen3-8B cMFG* |
|---------------|-------------------|-----------------|
| None (baseline) | 0.60 | 0.54 |
| PopQA | **0.84** | **0.83** |
| SelfAware | 0.81 | 0.81 |
| HaluEval | 0.80 | 0.81 |
| UMWP | 0.80 | 0.81 |

수학 추론(UMWP), 환각 탐지(HaluEval), 답변가능성(SelfAware) 등 완전히 다른 태스크로 훈련해도 10개 평가 태스크에서 일관된 성능을 보입니다.

### 3.2 일반화를 가능하게 하는 메커니즘

**① Metacognitive Advantage Scaling의 태스크 독립적 특성:**

$Z_g$는 "모델이 자신의 수행을 얼마나 정확히 예측하는가"를 측정하는 보편적 메타인지 능력입니다. 이는 특정 도메인 지식이 아닌 **자기 인식 능력** 자체를 강화하므로, 훈련 태스크에 무관하게 전이 가능합니다.

**② RLMF vs 표준 RL의 일반화 차이:**

FUT(SFT 기반)는 QA 유사 태스크에만 효과적인 반면, RLMF는 복잡한 장문 추론(MATH)이나 어려운 OOD 설정(SimpleQA)에서도 유사한 성능을 보입니다. 이는 RLMF가 태스크 특화적 패턴이 아닌 **메타인지적 자기 모니터링 능력** 자체를 향상시키기 때문입니다.

**③ Reliability Diagram 분석 (Fig. 3):**

RLMF는 모든 intrinsic confidence 수준(낮은 신뢰도 포함)에서 균등하게 효과적입니다. 반면 FUT와 원본 모델은 낮은 confidence 구간에서 체계적으로 실패합니다. 이것이 OOD 일반화의 핵심 이유입니다.

**④ Cross-task bin-level 분석 (Table 20):**

```
Llama3.1-8B (PopQA로만 훈련) → 다른 태스크 평가:
- PopQA: 0.71~0.90 (전 구간 고른 성능)
- SimpleQA: 0.76~0.89
- SciQ: 0.65~0.97
- MMLU: 0.59~0.98
```

낮은 intrinsic confidence 구간(0~0.3)에서도 0.59~0.77을 유지합니다.

**⑤ Metacognitive Data Selection의 역할:**

단순 active learning(어려운 샘플만 선택)이나 random selection보다 metacognitive selection이 우수한 이유: 상위 + 하위 scoring 샘플의 조합이 전체 confidence 스펙트럼에 걸친 균형 잡힌 학습 신호를 제공합니다 (Table 19 참조).

### 3.3 일반화의 한계

- 낮은 intrinsic confidence 구간에서의 일반화는 SciQ, MMLU에서 여전히 PopQA보다 약함
- 더 극단적인 OOD 태스크(예: 완전히 다른 언어, 멀티모달)에서의 검증은 미실시
- 소규모 모델(1.7B)에서도 유사한 일반화가 관찰되나, 성능 절대값이 더 낮을 수 있음

---

## 4. 연구에 미치는 영향 및 향후 연구 시 고려사항

### 4.1 앞으로의 연구에 미치는 영향

**① 새로운 RL 피드백 패러다임 제시:**

RLMF는 기존 RLHF(외부 피드백)나 RLIF(내부 confidence 신호)를 넘어, **메타인지적 수행 평가**라는 상위 수준의 내부 신호를 RL에 활용합니다. 이는 다음 방향으로 확장 가능합니다:
- 다양한 태스크에서의 메타인지 신호 정의 및 활용
- 더 복잡한 메타인지 계층 구조 (메타-메타인지) 연구
- 자기주도적 학습(self-directed learning) 알고리즘 개발

**② LLM 정렬(Alignment) 연구의 새로운 방향:**

메타인지 능력의 향상이 단순히 calibration 개선을 넘어, 모델의 자기 인식과 AI 안전성에 기여할 수 있음을 시사합니다. 특히 고위험 환경(의료, 법률)에서 신뢰할 수 있는 불확실성 소통의 중요성이 부각됩니다.

**③ Scalable Self-Improvement 연구에 기여:**

모델이 외부 주석 없이 자신의 훈련 데이터를 선별할 수 있다는 발견은, LLM의 자율적 자기 개선 가능성의 이론적 근거를 제공합니다. 그러나 이는 감독 없는 자율 행동의 위험성과 함께 연구되어야 합니다.

**④ Faithful vs Factual Calibration 연구의 분리:**

기존 연구가 factual calibration에 집중한 것과 달리, 이 논문은 faithful calibration이라는 독립적이고 중요한 문제를 체계적으로 다루며 향후 연구의 새로운 벤치마크를 제시합니다.

---

### 4.2 향후 연구 시 고려해야 할 점

**기술적 고려사항:**

1. **$\tau$ 및 $k$ 하이퍼파라미터 민감성**: 현재 $\tau = 0.10$, $k = 1$로 설정되어 있으나, 다른 태스크나 모델에서의 최적값이 다를 수 있음. 자동 조정 메커니즘 개발 필요.

2. **Reward hacking 방지**: metacognitive reward를 단독 사용 시 hacking 위험이 있음. 향후 연구에서 더 견고한 anti-hacking 메커니즘 설계 필요.

3. **$F_{\text{pred}}$ 추정 방법의 다양화**: 현재 전체 response에 대한 단일 점수이나, 문장별 메타인지 점수 평균 방식도 탐구 가치 있음.

4. **Intrinsic confidence 추정의 개선**: 현재 sampling consistency 기반($K=20$ 응답)으로 추정하나, 더 효율적이고 정확한 추정 방법 연구 필요.

5. **더 적은 GPU로의 확장**: 현재 6 GPU 필요. 계산 효율적인 RLMF 구현 방법 연구.

**방법론적 고려사항:**

6. **메타인지 신호의 범위 확장**: 현재 FC 수준 예측에 국한된 메타인지 신호를 다른 태스크(추론 품질, 사실 정확도)로 확장.

7. **더 광범위한 메타인지 능력 측정**: 논문도 인정하듯, 수행 자기평가 개선이 전체 메타인지 능력의 개선을 의미하지 않음. 메타인지의 다양한 측면(모니터링, 제어, 전략 선택 등)을 포괄하는 연구 필요.

8. **문화적·언어적 다양성**: 언어적 불확실성 표현은 문화마다 다름 (Lauwereyns, 2002; Mur-Dueñas, 2021). RLMF의 다국어/다문화 확장성 검증 필요.

**윤리적 고려사항:**

9. **자율적 메타인지의 위험**: 향상된 자기평가 능력이 더 자율적 행동으로 이어질 경우, 적절한 감독 메커니즘 필요.

10. **오용 방지**: faithful calibration의 향상이 악의적 목적으로 사용될 경우(예: 확신 있게 허위 정보 표현), 안전장치 설계 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 LLM 불확실성 표현 관련 연구 흐름

| 연구 | 방법 | 한계 | RLMF와의 차이 |
|------|------|------|----------------|
| **Lin et al. (2022)** "Teaching models to express their uncertainty in words" | SFT로 언어적 불확실성 학습 | 도메인 특화 훈련, zero-shot 불가 | RLMF는 RL 기반으로 직접 최적화 |
| **Mielke et al. (2022)** "Reducing conversational agents' overconfidence through linguistic calibration" | 제한된 점수 척도 사용 | 표현의 단순화 | RLMF는 풍부한 hedge 표현 허용 |
| **Kuhn et al. (2023)** "Semantic uncertainty" | 의미적 불확실성 측정 | factual calibration 중심 | RLMF는 faithful calibration 타겟 |
| **Tian et al. (2023)** "Just ask for calibration" | 프롬프팅으로 calibrated 점수 유도 | 불안정한 프롬프트 의존성 | RLMF는 훈련으로 능력 내재화 |
| **Yona et al. (2024)** "Can LLMs faithfully express their intrinsic uncertainty?" | faithful calibration 벤치마크 제시 | 해결책 제시 없음 | RLMF는 실질적 해결책 제공 |
| **Liu et al. (2025, MetaFaith)** EMNLP | 메타인지 프롬프팅으로 FC 개선 | accuracy 저하, 일반화 부족 | RLMF는 훈련 기반으로 안정적 |
| **Eikema et al. (2025, FUT)** | SFT로 faithful 불확실성 학습 | QA 유사 태스크에만 효과적 | RLMF는 광범위한 태스크에 일반화 |
| **Zhang et al. (2025)** "RL for better verbalized confidence" | RL로 factual calibration 개선 | faithful calibration 미타겟 | RLMF는 faithful 측면 직접 최적화 |
| **Damani et al. (2026)** "Beyond binary rewards" | 불확실성 추론 학습 | faithful vs factual 구분 모호 | RLMF는 두 측면 명확히 분리 |
| **Ji et al. (2025)** "Calibrating verbal uncertainty as a linear feature" | steering으로 언어적 불확실성 조정 | open-weight 모델만 가능 | RLMF는 어떤 모델에도 적용 |
| **Stangel et al. (2025)** "Rewarding doubt" | Brier Score 기반 RL reward | metacognitive 신호 미활용 | RLMF는 메타인지 피드백 추가 |

### 5.2 RL with Internal Feedback 관련 연구 흐름

| 연구 | 특징 | RLMF와의 차이 |
|------|------|----------------|
| **DeepSeek-R1 (Guo et al., 2025)** | GRPO 기반 reasoning 강화 | calibration 미고려 |
| **Chen et al. (2025, Seed-GRPO)** | semantic entropy로 GRPO advantage scaling | factual calibration 중심 | 메타인지 신호 미활용 |
| **Xie et al. (2026)** "Unlocking exploration in RLVR" | uncertainty-aware advantage shaping | exploration 개선 중심 | FC 미타겟 |
| **Zhang et al. (2025, No free lunch)** | 내부 피드백의 한계 분석 | 진행될수록 성능 저하 관찰 | RLMF는 이 한계를 극복함 |
| **Zhao et al. (2026)** "Learning to reason without external rewards" | self-certainty를 RL reward로 | output confidence 수준에서 작동 | RLMF는 더 상위 메타인지 수준 활용 |

### 5.3 핵심 차별점 요약

```
기존 RLIF: output confidence (1차 신호) → reward
RLMF:      metacognitive accuracy (2차 신호, 수행 자기평가) → advantage scaling
```

RLMF의 가장 중요한 차별점은 **피드백의 추상화 수준**입니다. 기존 방법들이 모델의 출력 신뢰도를 직접 사용하는 반면, RLMF는 "모델이 자신의 수행을 얼마나 정확히 평가하는가"라는 한 단계 높은 메타 수준의 신호를 활용합니다.

---

## 참고 자료 (출처)

**주요 참고 논문 (논문 내 인용):**

- **Liu et al. (2606.32032v1, 2026)**: "Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs" *(본 논문)*
- **Liu et al. (2025, MetaFaith)**: "MetaFaith: Faithful natural language uncertainty expression in LLMs." EMNLP 2025
- **Eikema et al. (2025, FUT)**: "Teaching language models to faithfully express their uncertainty." arXiv:2510.12587
- **Yona et al. (2024)**: "Can large language models faithfully express their intrinsic uncertainty in words?" EMNLP 2024
- **Shao et al. (2024, GRPO/DeepSeekMath)**: "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." arXiv:2402.03300
- **Guo et al. (2025, DeepSeek-R1)**: "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning." arXiv:2501.12948
- **Ouyang et al. (2022, RLHF)**: "Training language models to follow instructions with human feedback." arXiv:2203.02155
- **Lin et al. (2022)**: "Teaching models to express their uncertainty in words." TMLR 2022
- **Kuhn et al. (2023)**: "Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation." ICLR 2023
- **Zhang et al. (2025, No free lunch)**: "No free lunch: Rethinking internal feedback for LLM reasoning." arXiv:2506.17219
- **Chen et al. (2025, Seed-GRPO)**: "Seed-GRPO: Semantic entropy enhanced GRPO for uncertainty-aware policy optimization." arXiv:2505.12346
- **Damani et al. (2026)**: "Beyond binary rewards: Training LMs to reason about their uncertainty." ICLR 2026
- **Fleming & Lau (2014)**: "How to measure metacognition." Frontiers in Human Neuroscience
- **Steyvers & Peters (2025)**: "Metacognition and uncertainty communication in humans and large language models." Current Directions in Psychological Science
- **Manakul et al. (2023, SelfCheckGPT)**: "SelfCheckGPT: Zero-resource black-box hallucination detection for generative LLMs." EMNLP 2023
- **Guo et al. (2017)**: "On calibration of modern neural networks." ICML 2017
