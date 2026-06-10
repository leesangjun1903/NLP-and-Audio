# Self-Distillation Enables Continual Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **"전문가 시연(expert demonstrations)만 존재하는 환경에서도, 모델 자신의 In-Context Learning(ICL) 능력을 활용한 온-정책(on-policy) 학습이 가능하며, 이를 통해 지속적 학습(continual learning)에서의 파국적 망각(catastrophic forgetting)을 획기적으로 줄일 수 있다."**

기존의 Supervised Fine-Tuning(SFT)은 오프-정책(off-policy) 학습 방식이어서 파국적 망각이 발생하고, 반면 온-정책 강화학습(RL)은 명시적 보상 함수(reward function)가 필요합니다. 본 논문은 이 두 패러다임 사이의 간극을 메우는 **Self-Distillation Fine-Tuning(SDFT)** 를 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **방법론적 기여** | 보상 함수 없이 시연 데이터만으로 온-정책 학습 실현 |
| **이론적 기여** | SDFT가 암묵적 IRL(Inverse RL)과 수학적으로 동치임을 증명 |
| **실험적 기여** | Skill Learning & Knowledge Acquisition 두 설정 모두에서 SFT 대비 우월한 성능 입증 |
| **순차적 학습** | 단일 모델이 여러 기술을 순서대로 습득하면서 이전 기술을 유지하는 데 성공 |
| **추론 모델 적용** | Chain-of-Thought 어노테이션 없이도 추론 모델의 성능 향상 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제의 구조:**

```
파운데이션 모델의 지속적 학습 딜레마
├── SFT (오프-정책): 새 기술 습득 O, 파국적 망각 O
├── On-policy RL: 망각 최소화 O, 보상 함수 필요 (실제 환경에서 부재)
└── IRL: 이론적 우아함 O, 확장성 부족, 강한 사전 가정 필요
```

핵심 연구 질문: **"보상 함수 없이 시연 데이터만으로 온-정책 학습의 이점을 얻을 수 있는가?"**

### 2.2 제안하는 방법: SDFT (Self-Distillation Fine-Tuning)

#### 핵심 아이디어

동일한 모델을 **교사(Teacher)**와 **학생(Student)** 두 역할로 활용:

- **Student**: 쿼리 $x$만 조건으로 하는 기본 정책 $\pi_\theta(\cdot|x)$
- **Teacher**: 쿼리 $x$와 전문가 시연 $c$를 조건으로 하는 정책 $\pi(\cdot|x, c)$

#### 수식 (1): 핵심 학습 목적함수

학습 목적은 학생과 교사 분포 간의 **역방향 KL 발산(Reverse KL Divergence)** 최소화입니다:

$$\mathcal{L}(\theta) = D_{KL}\left(\pi_\theta(\cdot|x) \,\|\, \pi(\cdot|x, c)\right) = \mathbb{E}_{y \sim \pi_\theta(y|x)}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x, c)}\right] \tag{1}$$

여기서:
- $\theta$: 학생(= 훈련 대상) 모델 파라미터
- $x$: 태스크 입력 프롬프트
- $c$: 전문가 시연(demonstration)
- $y$: 학생 정책에서 샘플링된 응답

#### 수식 (2): 토큰 수준 그래디언트 추정량

자기회귀(autoregressive) 구조를 활용하여 목적함수를 토큰 수준으로 분해하면:

$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\left[\sum_t \sum_{y_t \in \mathcal{V}} \log \frac{\pi_\theta(y_t|y_{<t}, x)}{\pi(y_t|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t|y_{<t}, x)\right] \tag{2}$$

여기서 $\mathcal{V}$는 토큰 어휘집합(vocabulary)입니다.

**중요:** 교사 모델의 파라미터는 학생 파라미터의 **지수이동평균(EMA, Exponential Moving Average)**으로 유지됩니다:

$$\phi \leftarrow \alpha\theta + (1 - \alpha)\phi$$

### 2.3 SDFT와 Inverse RL의 동치성

#### 신뢰 영역 정규화 RL 출발점

$$\pi_{k+1} = \max_\pi \mathbb{E}_{y \sim \pi}[r(y,x)] - \beta D_{KL}(\pi(\cdot|x) \| \pi_k(\cdot|x)) \tag{3}$$

이 목적함수의 최적 정책은 다음과 같은 닫힌 형태(closed-form)를 가집니다:

$$\pi^*_{k+1}(y|x) \propto \pi_k(y|x) \exp\!\left(\frac{1}{\beta} r(y,x)\right)$$

#### In-Context 가정 (핵심 가설)

$$\pi^*_{k+1}(y|x) \approx \pi(y|x, c) \tag{4}$$

즉, **시연에 조건화된 모델이 unknown 최적 정책을 근사**한다고 가정합니다.

#### 암묵적 보상 함수 유도

이 가정을 대입하면 암묵적 보상 함수가 도출됩니다:

$$r(y, x, c) = \log \pi(y|x, c) - \log \pi_k(y|x) \tag{5}$$

#### 토큰 수준 즉각 보상

$$r_t(y_t | y_{ < t}, x, c) = \log \frac{\pi(y_t | y_{ < t}, x, c)}{\pi_k(y_t | y_{ < t}, x)}$$

이를 이용한 정책 그래디언트:

$$\nabla_\theta J(\pi_k) = \mathbb{E}_{y \sim \pi_k}\left[\log \frac{\pi(y|x, c)}{\pi_k(y|x)} \nabla_\theta \log \pi_k(y|x)\right] \tag{6}$$

**식 (6)은 식 (2)의 역방향 KL 발산 그래디언트와 기댓값 의미에서 동치**입니다. 따라서 SDFT는 명시적 보상 함수 추론 없이 IRL을 수행하는 것과 동등합니다.

### 2.4 모델 구조 및 알고리즘

```
SDFT 알고리즘 흐름:
┌─────────────────────────────────────────────────────────┐
│  1. 시연 데이터셋 D = {(x_i, c_i)} 준비                 │
│  2. 매 훈련 스텝:                                        │
│     a. 미니배치 샘플링                                   │
│     b. Student rollout: y_i ~ π_θ(·|x_i) [온-정책]     │
│     c. Teacher logprob 계산: π_φ(y_i,t|y_i,<t, x_i, c_i)│
│     d. Student logprob 계산: π_θ(y_i,t|y_i,<t, x_i)   │
│     e. Analytic per-token KL 그래디언트 계산            │
│     f. 파라미터 업데이트: θ ← θ - η·g                  │
│     g. EMA 교사 업데이트: φ ← αθ + (1-α)φ             │
└─────────────────────────────────────────────────────────┘
```

**교사 프롬프트 템플릿:**
```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

**KL 그래디언트 추정량 (Analytic per-token):**

논문이 최종 채택한 추정량:

$$\hat{g}_{\text{analytic}} = \sum_{t=1}^{T} \sum_{v \in \mathcal{V}} \log \frac{\pi_\theta(v|y_{<t}, x)}{\pi(v|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(v|y_{<t}, x)$$

토큰 수준(partial) 추정량:

$$\hat{g}_{\text{token}} = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t|y_{<t}, x)}{\pi(y_t|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t|y_{<t}, x)$$

Rao-Blackwellized 추정량 (분산 감소, 계산 비용 높음):

$$\hat{g}_{\text{rb}} = \sum_{t=1}^{T}\left[\sum_{v \in \mathcal{V}} \log \frac{\pi_\theta(v|y_{<t}, x)}{\pi(v|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(v|y_{<t}, x) + k_\theta(y_{<t}) \sum_{i=1}^{t-1} \nabla_\theta \log \pi_\theta(y_i|y_{<i}, x)\right]$$

### 2.5 성능 향상

#### Skill Learning 결과 (Qwen2.5-7B-Instruct)

| 방법 | Science Q&A (New) | Prior Tasks Avg. |
|---|---|---|
| Base | 32.1 | 65.5 |
| SFT | 66.2 | 53.4 (-12.1) |
| SFT+Re-invoke | 66.0 | 60.2 (-5.3) |
| DFT | 54.8 | 60.2 (-5.3) |
| **SDFT (Ours)** | **70.2** | **64.5 (-1.0)** |

| 방법 | Tool Use (New) | Prior Tasks Avg. |
|---|---|---|
| SFT | 63.2 | 56.0 (-9.5) |
| **SDFT** | **70.6** | **65.4 (-0.1)** |

#### Knowledge Acquisition 결과

| 방법 | Strict Acc. | Lenient Acc. | OOD Acc. |
|---|---|---|---|
| Base | 0 | 0 | 0 |
| Oracle RAG | 91 | 100 | 100 |
| CPT | 9 | 37 | 7 |
| SFT | 80 | 95 | 80 |
| **SDFT** | **89** | **100** | **98** |

#### 추론 모델 실험 (Olmo-3-7B-Think)

| 방법 | Accuracy | Avg. Tokens |
|---|---|---|
| Base | 31.2 | 4612 |
| +SFT | 23.5 (-7.7) | 3273 (추론 붕괴) |
| **+SDFT** | **43.7 (+12.5)** | **4180 (추론 유지)** |

#### 모델 크기에 따른 성능 (Science Q&A)

| 모델 크기 | SFT | SDFT | 차이 |
|---|---|---|---|
| 3B | ~67 | ~63 | -3.3 (SDFT 불리) |
| 7B | ~66 | ~70 | +4.0 |
| 14B | ~71 | ~78 | +6.9 |

### 2.6 한계점

1. **계산 비용**: SFT 대비 약 2.5× FLOPs, 4× 벽시계 훈련 시간
2. **소형 모델 의존성 문제**: ICL 능력이 약한 3B 이하 모델에서는 SFT보다 성능이 낮음
3. **학습된 아티팩트(Learned Artifacts)**: 교사가 "Based on the text..." 같은 구절을 생성하면 학생이 이를 학습하는 문제 → 처음 몇 토큰 마스킹으로 임시 해결(원칙적 해결책 부재)
4. **행동 패턴의 근본적 전환 어려움**: 비추론 모델을 추론 모델로 변환하는 등 근본적 생성 패턴 변화에는 어려움
5. **ICL 가정의 이론적 보장 부재**: In-Context 가정(식 4)은 이론적으로 검증되지 않고 경험적으로만 확인됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 In-Distribution 일반화

논문은 SFT가 오프-정책 학습이기 때문에 **분포 이탈(distributional shift)** 문제를 초래한다고 분석합니다.

> "오프-정책 학습은 전문가 유도 궤적에서만 훈련하므로, 테스트 시 오류가 정책을 미지의 상태로 밀어내면 오류가 복리로 증가한다(compounding errors). 온-정책 학습은 학습된 정책 자체가 유도하는 상태 분포에서 훈련함으로써 이 불일치를 방지한다." (Ross et al., 2011)

**실험적 증거:** 모든 Skill Learning 태스크에서 SDFT는 SFT 대비 높은 new-task 정확도를 달성 (Science Q&A: 66.2% → 70.2%, Tool Use: 63.2% → 70.6%)

### 3.2 Out-of-Distribution 일반화

Knowledge Acquisition 설정에서 SDFT의 OOD 일반화 능력이 특히 두드러집니다:

- **SFT의 OOD 정확도**: 80% (in-distribution과 동일 수준 → 단순 암기)
- **SDFT의 OOD 정확도**: 98% (지식이 모델의 더 넓은 지식 체계에 통합됨을 시사)

이는 SDFT가 특정 질문-답변 쌍을 암기하는 것이 아니라 **지식의 의미적 구조 자체를 내면화**함을 보여줍니다.

### 3.3 Pass@k 분석 (다양성과 품질)

SFT 후 성능 향상이 단순히 출력 분포의 엔트로피 감소(분산 수렴)에 의한 것인지 확인하기 위해 pass@k를 측정:

- $k = 1, 2, 4, ..., 128$ 범위 전체에서 SDFT가 SFT 대비 우월
- 이는 **진정한 기술 획득**이지 엔트로피 붕괴(entropy collapse)가 아님을 입증

### 3.4 스케일링과 일반화의 관계

모델 크기가 클수록 ICL 능력이 강해지고, 교사 신호의 품질이 향상되어 일반화 성능이 더욱 개선됩니다:

$$\text{일반화 향상폭} \propto \text{모델 크기} \propto \text{ICL 능력}$$

3B → 7B → 14B로 갈수록 SDFT와 SFT의 성능 차이가 -3.3 → +4.0 → +6.9로 단조 증가합니다.

### 3.5 SDFT가 일반화를 향상시키는 메커니즘 요약

```
SDFT의 일반화 향상 메커니즘:

1. 온-정책 학습 → 분포 이탈 방지
   └─ 학생의 실제 생성 분포에서 훈련 → 테스트 시 오류 복리 방지

2. 역방향 KL 최적화 → mode-seeking
   └─ 교사의 고품질 모드에 집중 학습

3. 토큰 밀도 신호 → 세밀한 크레딧 할당
   └─ 단순 정답/오답보다 정밀한 학습 신호

4. EMA 교사 → 안정적 타겟 분포
   └─ 학습 과정에서 교사가 점진적 진화

5. ICL 기반 교사 → 사전 분포 근접 유지
   └─ 기존 능력 보존하면서 새 지식 추가
```

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 지속적 학습 패러다임의 전환

SDFT는 지속적 학습 연구에서 "보상 함수 없이도 온-정책 학습이 가능하다"는 패러다임 전환을 제시합니다. 이는 실제 배포 환경에서 파운데이션 모델의 동적 업데이트를 가능하게 하는 핵심 블록이 될 수 있습니다.

#### (2) RLHF/RLAIF 파이프라인과의 통합 가능성

SDFT가 pass@k 전체에서 성능을 향상시킨다는 결과는 **SDFT → RL 파인튜닝**의 순차적 파이프라인 가능성을 시사합니다. SDFT로 초기화된 강한 시작 정책이 이후 RL 학습을 가속할 수 있습니다.

#### (3) Context Distillation 연구의 새 방향

기존 Context Distillation 연구(Bai et al., 2022; Snell et al., 2022)가 오프라인 방식에 머물렀다면, SDFT는 **온-정책 + 인스턴스별 동적 컨텍스트** 조합이라는 새 방향을 제시합니다.

#### (4) 추론 모델의 데이터 효율적 훈련

체인-오브-쏫(Chain-of-Thought) 어노테이션 없이도 추론 모델을 개선할 수 있다는 결과는, **어노테이션 비용이 높은 도메인**(의료, 법률, 과학)에서의 모델 훈련에 실질적 영향을 미칩니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) ICL 가정의 이론적 정당화 필요

현재 핵심 가정인 $\pi^*_{k+1}(y|x) \approx \pi(y|x, c)$ (식 4)는 경험적으로만 검증되었습니다. 향후 연구에서는:
- 이 근사가 성립하는 조건의 이론적 특성화
- 근사 오류가 최종 성능에 미치는 영향 분석
- 더 정교한 교사 정책 구성 방법 탐색이 필요합니다.

#### (2) 계산 효율성 개선

현재 SFT 대비 4× 벽시계 시간 오버헤드는 대규모 실용 배포에 장벽이 됩니다. 고려할 점:
- **LoRA/PEFT와의 결합**: 풀 파인튜닝 대신 효율적 어댑터 방식 적용 가능성
- **투기적 디코딩(speculative decoding)** 활용으로 온-정책 롤아웃 비용 감소
- **배치 롤아웃 최적화** 전략 개발

#### (3) 비전문가/노이즈 시연 처리

현재 SDFT는 **고품질 전문가 시연**에 의존합니다. 실제 환경에서는:
- 사용자 인터랙션 로그 같은 **노이즈 있는 시연**에서 학습하는 방법
- **시연 품질 자동 평가** 및 필터링 메커니즘 개발이 필요합니다.

#### (4) 망각의 완전한 제거

SDFT가 SFT보다 파국적 망각을 크게 줄이지만, 여전히 소량의 성능 저하가 관찰됩니다(예: Science Q&A 훈련 후 Prior Tasks 65.5 → 64.5). 이를 위한 보완 기법:
- **Elastic Weight Consolidation(EWC)** 와의 결합 (Kirkpatrick et al., 2017)
- **Progressive Neural Networks** 아이디어 통합
- **메모리 리플레이(experience replay)** 메커니즘 통합

#### (5) 멀티모달 및 다양한 도메인으로의 확장

현재 실험은 텍스트 기반 LLM에 집중되어 있습니다. 확장 고려:
- **비전-언어 모델(VLM)**: 시각적 시연 활용 가능성
- **로보틱스**: 물리적 시연에서의 온-정책 학습
- **코드 생성**: 실행 피드백과 SDFT의 결합

#### (6) 아티팩트 문제의 원칙적 해결

"Based on the text..." 같은 교사 아티팩트 전파 문제에 대해 현재 토큰 마스킹은 휴리스틱입니다. 더 원칙적인 접근:
- **역할 분리 학습(role-disentangled training)** 목적함수 설계
- **조건부 정보 흐름 제어** 기법 개발

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 온-정책? | 보상 필요? | 주요 차별점 vs SDFT |
|---|---|---|---|---|
| **DAgger** (Ross et al., 2011) | 반복적 온-정책 모방 학습 | ✅ | ❌ | 전문가 개입 필요, LLM 규모 미검증 |
| **Constitutional AI** (Bai et al., 2022) | 오프라인 Context Distillation | ❌ | ❌ | 오프라인, 고정 컨텍스트 |
| **Learning by Distilling Context** (Snell et al., 2022) | 오프라인 Context Distillation | ❌ | ❌ | 오프라인, 학생이 교사 분포에서 학습 |
| **RLHF** (Ouyang et al., 2022) | PPO + 보상 모델 | ✅ | ✅ | 선호 데이터 쌍 및 보상 모델 필요 |
| **DPO** (Rafailov et al., 2023) | 선호 최적화 | ❌ | ❌ (암묵적) | 선호 쌍 필요, 오프-정책 |
| **GKD/On-policy KD** (Agarwal et al., 2024) | 온-정책 증류 | ✅ | ❌ | 별도 교사 모델 필요 |
| **SFT Memorizes, RL Generalizes** (Chu et al., 2025) | 비교 분석 | - | - | SDFT의 일반화 주장 지지 |
| **Re-invoke** (Lu & Lab, 2025) | SFT 후 온-정책 복원 | 부분적 | ❌ | 2단계 훈련, SDFT보다 열등 |
| **DFT** (Wu et al., 2025b) | 중요도 샘플링으로 오프라인→온라인 | 근사 | ❌ | 근사적 온-정책, SDFT보다 열등 |
| **Efficient KI via Self-Distillation** (Kujanpää et al., 2025) | 지식 주입을 위한 자기증류 | ❌ | ❌ | 오프라인, 텍스트만 조건화 |
| **RL's Razor** (Shenfeld et al., 2025) | 온-정책 RL vs 오프라인 비교 | ✅ | ✅ | RL 기반, 보상 함수 필요 |
| **SDFT (본 논문)** | 온-정책 자기증류 | ✅ | ❌ | **시연만으로 온-정책, 자기 교사** |

### 핵심 포지셔닝

```
               보상 함수 필요 여부
                    필요        불필요
               ┌───────────┬───────────┐
온-정책   Yes  │   RLHF    │  SDFT ★   │
               │   PPO     │  DAgger   │
               ├───────────┼───────────┤
오프-정책 No   │  (희귀)   │   SFT     │
               │           │   DPO     │
               │           │   Const.AI│
               └───────────┴───────────┘
```

SDFT는 **보상 불필요 + 온-정책**이라는 독특한 위치를 차지합니다.

---

## 참고 자료 (출처)

본 답변은 다음 자료를 직접 참조하였습니다:

1. **Shenfeld, I., Damani, M., Hübotter, J., & Agrawal, P. (2026). "Self-Distillation Enables Continual Learning." arXiv:2601.19897v1 [cs.LG]** (제공된 PDF 원문, 2026년 1월 27일 업로드)

본 답변에서 인용된 논문 내 참고문헌:

2. Ross, S., Gordon, G., & Bagnell, D. (2011). "A reduction of imitation learning and structured prediction to no-regret online learning." *AISTATS 2011.*
3. Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." *PNAS, 114(13).*
4. Brown, T., et al. (2020). "Language models are few-shot learners." *NeurIPS 2020.*
5. Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI feedback." *arXiv:2212.08073.*
6. Snell, C., Klein, D., & Zhong, R. (2022). "Learning by distilling context." *arXiv:2209.15189.*
7. Rafailov, R., et al. (2023). "Direct preference optimization." *NeurIPS 2023.*
8. Agarwal, R., et al. (2024). "On-policy distillation of language models." *ICLR 2024.*
9. Chu, T., et al. (2025). "SFT Memorizes, RL Generalizes." *arXiv:2501.17161.*
10. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." *NeurIPS 2022.*
11. Tang, Y., & Munos, R. (2025). "On a few pitfalls in KL divergence gradient estimation for RL." *arXiv:2506.09477.*
12. Amini, A., Vieira, T., & Cotterell, R. (2025). "Better estimation of the Kullback–Leibler divergence between language models." *NeurIPS 2025.*
13. Kujanpää, K., et al. (2025). "Efficient knowledge injection in LLMs via self-distillation." *TMLR 2025.*
14. Shenfeld, I., Pari, J., & Agrawal, P. (2025). "RL's Razor: Why online reinforcement learning forgets less." *arXiv:2509.04259.*
15. Schulman, J., et al. (2015). "Trust region policy optimization." *ICML 2015.*
16. Mecklenburg, N., et al. (2024). "Injecting new knowledge into large language models via supervised fine-tuning." *arXiv:2404.00213.*
