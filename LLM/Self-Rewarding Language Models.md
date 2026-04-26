# Self-Rewarding Language Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다:

> **"초인간적(superhuman) 에이전트를 달성하기 위해서는, 미래의 모델이 적절한 훈련 신호를 제공할 수 있는 초인간적 피드백을 필요로 한다."**

기존 RLHF(Reinforcement Learning from Human Feedback) 패러다임은 두 가지 근본적 병목(bottleneck)을 가집니다:

1. **인간 성능 수준의 상한선**: 인간 선호도로 훈련된 보상 모델은 인간 수준 이상으로 개선되기 어렵습니다.
2. **고정된 보상 모델**: 별도로 훈련된 보상 모델은 LLM 훈련 중 업데이트되지 않습니다.

이를 해결하기 위해 **Self-Rewarding Language Models(자기 보상 언어 모델)** 을 제안하며, 하나의 모델이 지시 따르기(instruction following)와 자기 평가(self-evaluation)를 동시에 수행합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 훈련 패러다임** | LLM-as-a-Judge 기반 자기 보상 생성 |
| **반복적 자기 개선** | Iterative DPO를 통한 지속적 성능 향상 |
| **이중 능력 동시 향상** | 지시 따르기 + 보상 모델링 능력 동시 개선 |
| **경쟁력 있는 성능** | Llama 2 70B 기반으로 Claude 2, Gemini Pro, GPT-4 0613 능가 |
| **데이터 효율성** | 소규모 시드 데이터만으로 자기 생성 훈련 데이터 구성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 접근법의 한계:**

$$\text{RLHF: } \underbrace{\theta_{\text{LLM}}}_{\text{업데이트}} \leftarrow \underbrace{r_{\phi}(\cdot)}_{\text{고정된 보상 모델}} \leftarrow \underbrace{\mathcal{D}_{\text{human}}}_{\text{인간 선호 데이터}}$$

- 보상 모델 $r_{\phi}$가 훈련 중 고정(frozen)되어 있어 개선 불가
- 인간 어노테이션 데이터의 규모와 질에 의해 상한선이 결정됨
- 별도의 보상 모델 관리에 따른 계산 비용 증가

**DPO의 한계:**

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y^w, y^l) \sim \mathcal{D}} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \beta \log \frac{\pi_\theta(y^l|x)}{\pi_{\text{ref}}(y^l|x)}\right)\right]$$

여기서 $y^w$는 선호 응답(winning response), $y^l$은 비선호 응답(losing response), $\beta$는 KL 페널티 계수입니다. DPO도 여전히 외부 인간 선호 데이터 $\mathcal{D}$에 의존합니다.

---

### 2.2 제안하는 방법

#### 전체 자기 정렬 알고리즘

**모델 시퀀스 정의:**

$$M_0 \rightarrow M_1 \rightarrow M_2 \rightarrow M_3$$

$$M_0: \text{Base pretrained LLM (no fine-tuning)}$$

$$M_1: M_0 \xrightarrow{\text{SFT}} \text{IFT} + \text{EFT seed data}$$

$$M_2: M_1 \xrightarrow{\text{DPO}} \text{AIFT}(M_1)$$

$$M_3: M_2 \xrightarrow{\text{DPO}} \text{AIFT}(M_2)$$

여기서:
- **IFT(Instruction Fine-Tuning)**: 인간 작성 지시-응답 시드 데이터
- **EFT(Evaluation Fine-Tuning)**: LLM-as-a-Judge 평가 능력 훈련 데이터
- **AIFT(AI Feedback Training)**: 모델 자신이 생성한 선호 쌍 데이터

#### Self-Instruction Creation 과정

각 반복(iteration) $t$에서:

**Step 1: 새 프롬프트 생성**

$$x_i \sim p_{\text{few-shot}}(\cdot \mid \mathcal{D}_{\text{seed}}), \quad i = 1, \ldots, |\mathcal{X}|$$

**Step 2: 후보 응답 생성**

$$\{y_i^1, y_i^2, \ldots, y_i^N\} \sim M_t(\cdot \mid x_i), \quad N=4$$

**Step 3: LLM-as-a-Judge 자기 평가**

$$r_i^n = M_t^{\text{judge}}(x_i, y_i^n) \in \{0, 1, 2, 3, 4, 5\}$$

평균 점수 사용 (분산 감소를 위해 3회 샘플링):

$$\bar{r}_i^n = \frac{1}{K}\sum_{k=1}^{K} r_{i,k}^n, \quad K=3$$

**Step 4: 선호 쌍 구성**

$$y_i^w = \arg\max_{n} \bar{r}_i^n, \quad y_i^l = \arg\min_{n} \bar{r}_i^n$$

$$\mathcal{D}_{\text{AIFT}}(M_t) = \{(x_i, y_i^w, y_i^l) \mid \bar{r}_i^w \neq \bar{r}_i^l\}$$

**Step 5: DPO 훈련**

$$M_{t+1} = \arg\min_{\theta} \mathcal{L}_{\text{DPO}}(\pi_\theta; M_t \mid \mathcal{D}_{\text{AIFT}}(M_t))$$

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y^w, y^l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y^w|x)}{\pi_{M_t}(y^w|x)} - \beta \log \frac{\pi_\theta(y^l|x)}{\pi_{M_t}(y^l|x)}\right)\right]$$

#### LLM-as-a-Judge 평가 기준 (가산 점수 방식)

$$r = \sum_{k=1}^{5} c_k, \quad c_k \in \{0, 1\}$$

| 점수 | 기준 |
|---|---|
| +1 | 응답이 관련성 있고 정보 제공 |
| +1 | 사용자 질문의 상당 부분 다룸 |
| +1 | 기본 요소에 유용하게 답변 |
| +1 | AI 어시스턴트 관점에서 명확하고 종합적으로 작성 |
| +1 | 전문 지식을 반영한 완벽한 답변 |

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────┐
│              Self-Rewarding LLM (M_t)               │
│                                                     │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │  Instruction    │    │    LLM-as-a-Judge       │ │
│  │  Following      │    │    (Self-Reward)        │ │
│  │  Module         │    │    Module               │ │
│  │                 │    │                         │ │
│  │  x → y (생성)   │    │  (x, y) → r ∈ [0,5]    │ │
│  └─────────────────┘    └─────────────────────────┘ │
│           ↑                         ↑               │
│           └──────── 동일 모델 ────────┘               │
└─────────────────────────────────────────────────────┘
         ↓ AIFT preference pairs
┌─────────────────────────────────────────────────────┐
│              Iterative DPO Training                 │
│    (x_i, y_i^w, y_i^l) → M_{t+1}                  │
└─────────────────────────────────────────────────────┘
```

**베이스 모델**: Llama 2 70B (Touvron et al., 2023)

**시드 데이터**:
- IFT: Open Assistant 3,200개 예시 (최고 품질 rank 0만 선택)
- EFT: Open Assistant에서 구성된 1,630개 훈련 / 541개 평가 예시

**DPO 하이퍼파라미터**:
- Learning rate: $1 \times 10^{-6}$ (cosine decay to $1 \times 10^{-7}$)
- Batch size: 16
- Dropout: 0.1
- $\beta = 0.1$
- 후보 응답 수: $N = 4$ (temperature $T = 0.7$, $p = 0.9$)

---

### 2.4 성능 향상

#### AlpacaEval 2.0 결과 (GPT-4 Turbo 대비 win rate)

| 모델 | Win Rate | Distilled | Proprietary |
|---|---|---|---|
| $M_1$ (Iter 1) | 9.94% | | |
| $M_2$ (Iter 2) | 15.38% | | |
| $M_3$ (Iter 3) | **20.44%** | | |
| Claude 2 | 17.19% | | ✓ |
| Gemini Pro | 16.85% | | ✓ |
| GPT-4 0613 | 15.76% | | ✓ |
| LLaMA2 Chat 70B | 13.87% | | ✓ |

#### Head-to-head 비교 (vs. SFT Baseline)

$$\underbrace{M_1}_{\text{Iter1}} \xrightarrow{+18.7\%} \underbrace{M_2}_{\text{Iter2}} \xrightarrow{+13.3\%} \underbrace{M_3}_{\text{Iter3}}$$

- $M_1$ vs SFT: 30.5% wins / 30.9% wins (거의 동등)
- $M_2$ vs SFT: **49.2% wins** / 14.5% wins
- $M_3$ vs SFT: **62.5% wins** / 9.8% wins

#### 보상 모델링 능력 향상

| 지표 | SFT Baseline | $M_1$ | $M_2$ | $M_3$ |
|---|---|---|---|---|
| Pairwise Accuracy | 65.1% | 78.7% | 80.4% | **81.7%** |
| Spearman Corr. | 0.253 | 0.279 | 0.331 | **0.349** |
| Kendall $\tau$ | 0.233 | 0.253 | 0.315 | **0.324** |

#### MT-Bench 결과

| 모델 | Overall | Math & Code | Humanities & Writing |
|---|---|---|---|
| SFT | 6.85 | 3.93 | 8.60 |
| $M_1$ | 6.78 | 3.83 | 8.55 |
| $M_2$ | 7.01 | 4.05 | 8.79 |
| $M_3$ | **7.25** | **4.17** | **9.10** |

---

### 2.5 한계점

논문에서 명시된 주요 한계는 다음과 같습니다:

1. **수학·논리 추론 태스크 개선 제한**: AlpacaEval 분석에서 수학/논리 추론 분야는 상대적으로 개선 폭이 작음. Open Assistant 시드 데이터의 구성이 이 분야를 상대적으로 적게 포함하기 때문으로 추정됨.

2. **응답 길이 증가 편향**: 반복이 진행될수록 생성 응답이 길어지는 경향 존재:
   $$\bar{l}\_{M_1} = 1092 \rightarrow \bar{l}\_{M_2} = 1552 \rightarrow \bar{l}_{M_3} = 2552$$
   이는 평가 지표에서의 성능 향상이 실제 질적 향상인지 길이 편향인지 구분이 어려움.

3. **Reward Hacking 위험**: 자기 평가하는 모델이 reward hacking을 일으킬 가능성 — 즉 실제 품질 향상 없이 높은 점수를 받는 응답을 학습할 위험.

4. **반복 횟수 제한**: 논문에서는 3회 반복만 실험하여 포화(saturation) 지점 및 스케일링 법칙 미검증.

5. **안전성 평가 부재**: 안전(safety) 평가가 수행되지 않았으며, 적대적 입력이나 유해 콘텐츠에 대한 강건성 미확인.

6. **단일 모델 기반 평가**: 훈련 보상 생성자와 최종 평가자(GPT-4) 모두 LLM이므로, 순환적 평가 편향 가능성.

7. **NLP 벤치마크 성능 유지 수준**: ARC-Challenge, HellaSwag 등 전통적 NLP 벤치마크에서 일부 지표 소폭 하락 — "alignment tax" 현상과 유사.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 지지하는 실험적 근거

#### (1) 다양한 카테고리에 걸친 성능 향상

AlpacaEval 테스트 세트를 20개 카테고리로 분류한 결과, 대부분의 카테고리에서 반복에 따른 win rate 향상 관찰:

$$\Delta\text{WR}_{M_1 \to M_3} > 0 \quad \text{for most categories}$$

특히 건강, 언어학, 엔터테인먼트, 기술, 문학 등 다양한 도메인에서 개선.

#### (2) 다양한 복잡도 수준에서의 개선

복잡도 1~8에 걸쳐 win rate 향상이 관찰되며, 특히 **중간 난이도(complexity 5~7)** 에서 가장 두드러진 개선:

$$\forall c \in \{1, 2, \ldots, 8\}: \text{WR}_{M_3}(c) \geq \text{WR}_{M_1}(c)$$

이는 단순 암기가 아닌 일반적 추론 능력이 향상됨을 시사합니다.

#### (3) 기대 응답 길이와 무관한 개선

단문(1 sentence)부터 장문(2 paragraphs 이상)까지 모든 길이 범주에서 성능 향상:

$$\text{WR}_{M_3}(\text{length}) > \text{WR}_{M_1}(\text{length}), \quad \forall \text{ length categories}$$

#### (4) 인간 평가와의 일치

인간 평가(50개 지시, 3명의 맹목 평가자)에서도 자동 평가와 동일한 경향 확인:

| 비교 | Self-Rewarding Wins | Tie | SFT Wins |
|---|---|---|---|
| $M_1$ vs SFT | 28.0% | 26.0% | 46.0% |
| $M_2$ vs SFT | 56.0% | 24.0% | 20.0% |
| $M_3$ vs SFT | **66.0%** | 16.0% | 18.0% |

#### (5) MT-Bench 멀티턴 평가에서의 일반화

훈련 시 **단일 턴(single-turn)** 데이터만 사용했음에도 불구하고, **멀티턴 평가(MT-Bench)** 에서 성능 향상:

$$\text{MT-Bench}: M_1(6.78) \rightarrow M_2(7.01) \rightarrow M_3(7.25)$$

이는 훈련 분포를 넘어서는 **진정한 일반화** 능력 향상의 증거입니다.

#### (6) EFT 데이터의 전이 학습 효과

EFT(보상 모델링) 훈련이 IFT(지시 따르기) 성능을 저해하지 않음:

$$\text{IFT 성능}: M_1(\text{IFT+EFT}) \approx M_1(\text{IFT only})$$

이는 **멀티태스크 학습을 통한 긍정적 전이(positive transfer)** 가 발생함을 의미합니다. Collobert & Weston (2008)의 멀티태스크 학습 원리와 일치합니다.

### 3.2 일반화의 핵심 메커니즘

**보상 모델과 생성 모델의 공동 진화(co-evolution):**

$$\text{Generation Quality}_{t+1} \propto f\left(\text{Reward Quality}_t\right)$$

$$\text{Reward Quality}_{t+1} \propto g\left(\text{General IF Ability}_{t+1}\right)$$

이 두 관계의 양의 피드백 루프(virtuous circle)가 일반화 성능의 지속적 개선을 가능하게 합니다:

```
더 나은 지시 따르기 능력
        ↓
더 정확한 자기 보상 생성
        ↓
더 고품질의 선호 데이터
        ↓
더 나은 DPO 훈련
        ↓
더 나은 지시 따르기 능력 (반복)
```

### 3.3 일반화 한계 및 주의사항

그러나 다음과 같은 한계도 존재합니다:

- **수학·논리 추론**: 시드 데이터의 편향으로 인해 이 영역의 일반화는 제한적
- **분포 이탈(OOD) 성능**: Open Assistant 시드 기반이라 다른 도메인으로의 일반화 보장 불명확
- **포화 가능성**: 논문 저자들도 "실제 환경에서 이 효과는 포화될 가능성이 높다"고 인정

$$\lim_{t \to \infty} \text{Performance}(M_t) = \text{saturation point}$$

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 연구 계보 및 비교 분석

```
InstructGPT/RLHF          Constitutional AI         Self-Instruct
(Ouyang et al., 2022)  →  (Bai et al., 2022b)   →  (Wang et al., 2023)
        ↓                         ↓                       ↓
   고정 보상 모델             별도 보상 모델              데이터 증강
        ↓                         ↓                       ↓
DPO (Rafailov et al., 2023)   RLAIF (Lee et al., 2023)    ↓
        ↓                         ↓                       ↓
        └──────────────┬───────────┘                       ↓
                       ↓                                   ↓
              Iterative DPO (Xu et al., 2023)              ↓
                       ↓                                   ↓
                       └──────────────┬────────────────────┘
                                      ↓
                    Self-Rewarding LMs (Yuan et al., 2024)
```

### 4.2 주요 연구 비교표

| 논문 | 방법 | 보상 소스 | 보상 모델 업데이트 | 외부 데이터 의존성 |
|---|---|---|---|---|
| **InstructGPT** (Ouyang et al., 2022) | PPO + 고정 RM | 인간 | ✗ (고정) | 대규모 인간 어노테이션 |
| **Constitutional AI** (Bai et al., 2022b) | RLAIF | AI + 원칙 | ✗ (고정) | 헌법 원칙 |
| **DPO** (Rafailov et al., 2023) | 직접 선호 최적화 | 인간 | ✗ (해당없음) | 인간 선호 데이터 |
| **RLAIF** (Lee et al., 2023) | PPO + AI-RM | AI LLM | ✗ (고정) | Off-the-shelf LLM |
| **Iterative DPO/PCO** (Xu et al., 2023) | 반복 DPO | 외부 고정 RM | ✗ (고정) | 외부 RM |
| **SPIN** (Chen et al., 2024b) | Iterative DPO | 인간 레이블 | ✗ (해당없음) | 인간 정답 응답 |
| **Instruction Backtranslation** (Li et al., 2024) | SFT + 자기 선별 | 자기 평가 | 부분 | 웹 문서 |
| **Self-Rewarding LMs** (Yuan et al., 2024) | Iterative DPO + 자기 보상 | **자기 자신** | **✓ (지속 업데이트)** | **최소 (시드만)** |

### 4.3 핵심 차별점 분석

#### RLAIF (Lee et al., 2023)과의 차이

$$\text{RLAIF}: \underbrace{M_{\text{external}}}_{\text{별도 고정 LLM}} \rightarrow r \rightarrow \text{PPO 훈련}$$

$$\text{Self-Rewarding}: \underbrace{M_t}_{\text{동일 모델}} \rightarrow r \rightarrow \text{DPO 훈련} \rightarrow M_{t+1}$$

RLAIF는 별도의 큰 모델을 PPO 루프 내에서 직접 사용하여 **계산 비용이 매우 높음**. Self-Rewarding은 오프라인 반복 방식으로 **상대적으로 저비용**.

#### SPIN (Chen et al., 2024b)과의 차이

$$\text{SPIN}: y^w = \text{human label}, \quad y^l = M_{t-1}(x)$$

$$\text{Self-Rewarding}: y^w = \arg\max_n M_t^{\text{judge}}(x, y^n), \quad y^l = \arg\min_n M_t^{\text{judge}}(x, y^n)$$

SPIN은 인간 정답 레이블이 반드시 필요하며, 모델이 인간 수준에 도달하면 병목이 발생합니다. Self-Rewarding은 인간 정답 없이도 계속 개선 가능합니다.

#### ReST (Gulcehre et al., 2023)과의 차이

ReST는 외부 고정 보상으로 긍정적 예시만 추가하는 방식입니다. Self-Rewarding 논문(Appendix A.4)에서 이 방식을 실험한 결과:

$$\text{SFT only (positive examples)}: 29\% \text{ wins vs } 30\% \text{ wins (개선 없음)}$$

**선호 쌍(preference pairs)을 이용한 DPO 방식이 긍정적 예시만 추가하는 방식보다 우월**함을 실험적으로 확인.

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 향후 연구에 미치는 영향

#### (1) 자기 개선 패러다임의 새로운 기준점

Self-Rewarding LMs는 **"모델이 자신의 훈련 신호를 스스로 생성"** 하는 패러다임을 실증적으로 검증함으로써, 향후 LLM 정렬 연구의 새로운 기준점(baseline)을 제시했습니다.

#### (2) 초인간적 보상 모델 가능성 시사

$$\text{If } \text{RM}_{t+1} > \text{RM}_t > \text{RM}_{\text{human-trained}}, \text{ then } \exists t^* : \text{RM}_{t^*} > \text{Human-level RM}$$

이 가설이 참이라면, 인간 어노테이션의 근본적 한계를 극복할 수 있는 경로가 열립니다.

#### (3) 단일 모델 다중 역할 통합 트렌드 강화

별도의 보상 모델, 생성 모델, 평가 모델을 하나의 모델로 통합하는 **통합 에이전트(unified agent)** 연구 방향을 촉진합니다.

#### (4) LLM-as-a-Judge 연구 활성화

평가 프롬프트 설계(가산 점수 방식 vs. 다중선택 방식)가 성능에 미치는 영향을 실증적으로 보여줌으로써, **평가 프롬프트 엔지니어링** 연구의 중요성을 부각시켰습니다.

#### (5) 데이터 효율적 훈련 패러다임

소규모 시드 데이터(3,200개 IFT + 1,630개 EFT)만으로 대규모 독점 데이터를 사용한 모델들을 능가하는 결과는, **데이터 효율적 자기 개선 방법론** 연구를 촉진합니다.

---

### 5.2 향후 연구 시 고려사항

#### (1) 보상 해킹(Reward Hacking) 방지 메커니즘 연구

자기 평가 모델이 생성하는 보상이 **편향되거나 부정확**할 경우, 잘못된 방향으로 훈련이 진행될 위험:

$$\text{Risk: } M_t^{\text{judge}} \text{ 가 편향된 } r \text{ 생성} \rightarrow \text{잘못된 } (y^w, y^l) \rightarrow \text{성능 저하}$$

**고려 방안**:
- 외부 검증 신호와의 주기적 교차 검증
- 앙상블 기반 보상 추정
- 보상 분포의 통계적 모니터링

#### (2) 스케일링 법칙(Scaling Laws) 탐구

논문은 3회 반복만 실험했습니다. 향후 연구에서 반드시 탐구해야 할 질문:

$$\text{Performance}(M_t) = f(t, |\theta|, |\mathcal{D}_{\text{seed}}|)$$

- 반복 횟수 $t$에 따른 성능 포화 지점은 어디인가?
- 모델 크기 $|\theta|$가 자기 개선 효율에 미치는 영향은?
- 시드 데이터 크기 $|\mathcal{D}_{\text{seed}}|$의 최소 요구사항은?

#### (3) 수학·추론 영역으로의 확장

현재 방법은 수학, 코딩, 논리 추론 분야에서 제한적 개선을 보입니다. 이를 극복하기 위한 고려사항:

- **도메인 특화 시드 데이터** 구성 (예: GSM8K, MATH 데이터셋 활용)
- **검증 가능한 보상(verifiable rewards)** 활용: 수학 문제의 경우 정답 여부로 명확한 보상 정의 가능
- **Process Reward Models(PRM)** 통합: 최종 답변이 아닌 풀이 과정 단계별 평가

#### (4) 안전성(Safety) 통합

$$\text{Safety-aware Self-Rewarding}: r_{\text{total}} = \alpha \cdot r_{\text{quality}} + (1-\alpha) \cdot r_{\text{safety}}$$

- LLM-as-a-Judge를 안전성 평가에도 적용하는 방안 탐구
- 안전성 기준을 명시적으로 포함한 EFT 데이터 구성
- Constitutional AI의 원칙을 자기 보상 프레임워크에 통합

#### (5) 길이 편향(Length Bias) 완화

반복이 진행될수록 응답 길이가 급격히 증가하는 현상:

$$\bar{l}_{M_1}=1092 \rightarrow \bar{l}_{M_2}=1552 \rightarrow \bar{l}_{M_3}=2552$$

**고려 방안**:
- 길이에 독립적인 보상 함수 설계
- 길이 페널티 항 추가: $r_{\text{adjusted}} = r_{\text{raw}} - \lambda \cdot \text{length}(y)$
- 정보 밀도(information density) 기반 평가 메트릭 도입

#### (6) 다중 모달리티(Multi-modality) 확장

현재는 텍스트 전용 모델에 한정됩니다. 시각, 음성 등 다중 모달리티 환경에서의 자기 보상 메커니즘 탐구가 필요합니다.

#### (7) 분산 훈련 및 실용화 고려

- 70B 파라미터 모델의 반복 훈련에 따른 **계산 비용 최적화** 방안
- 더 작은 모델에서의 효과 검증 (모델 크기 의존성)
- **온라인(online) 자기 보상** 방식으로의 전환 가능성

#### (8) 이론적 수렴성 분석

현재 논문은 경험적 결과만 제시하며 이론적 보장이 없습니다:

$$\text{Question: } \exists \text{ convergence guarantee for } M_t \text{ as } t \to \infty?$$

향후 연구에서 자기 보상 반복 훈련의 수렴 조건 및 최적해 특성에 대한 이론적 분석이 필요합니다.

---

## 참고자료

**본 논문 (직접 분석)**
- Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., & Weston, J. (2024). **Self-Rewarding Language Models**. arXiv:2401.10020v3.

**논문 내 주요 인용 문헌**
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 35*.
- Rafailov, R. et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*.
- Bai, Y. et al. (2022a). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv:2204.05862*.
- Bai, Y. et al. (2022b). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*.
- Lee, H. et al. (2023). RLAIF: Scaling reinforcement learning from human feedback with AI feedback. *arXiv:2309.00267*.
- Xu, J. et al. (2023). Some things are more cringe than others: Preference optimization with the pairwise cringe loss. *arXiv:2312.16682*.
- Chen, Z. et al. (2024b). Self-play fine-tuning converts weak language models to strong language models. *arXiv:2401.01335*.
- Li, X. et al. (2024). Self-alignment with instruction backtranslation. *ICLR 2024*.
- Wang, Y. et al. (2023). Self-instruct: Aligning language models with self-generated instructions. *ACL 2023*.
- Gulcehre, C. et al. (2023). Reinforced self-training (ReST) for language modeling. *arXiv:2308.08998*.
- Zheng, L. et al. (2023b). Judging LLM-as-a-judge with MT-bench and chatbot arena. *NeurIPS 2023 Datasets and Benchmarks Track*.
- Touvron, H. et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*.
- Köpf, A. et al. (2023). OpenAssistant conversations–democratizing large language model alignment. *arXiv:2304.07327*.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Collobert, R. & Weston, J. (2008). A unified architecture for natural language processing. *ICML 2008*.
