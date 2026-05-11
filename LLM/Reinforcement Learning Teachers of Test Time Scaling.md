# Reinforcement Learning Teachers of Test Time Scaling

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

기존 RL 기반 추론 언어모델(LM) 훈련은 **탐색(exploration) 문제**와 **목적 불일치(objective mismatch)** 라는 두 가지 근본적 한계를 지닌다. 본 논문은 이를 해결하기 위해 **Reinforcement-Learned Teachers (RLTs)** 라는 새로운 모델 클래스를 제안한다.

RLT의 핵심 아이디어: "실제 교사의 능력은 스스로 어려운 문제를 풀어내는 것이 아니라, 주어진 해답을 학생이 이해할 수 있도록 효과적으로 설명하는 것이다."

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① RLT 프레임워크 도입** | 탐색 문제를 회피하는 dense reward 기반 RL 훈련법 |
| **② 효율적 증류 데이터 생성** | 7B RLT의 원본 출력이 수백억 규모 LM의 후처리된 출력을 능가 |
| **③ 일반화 및 재사용성** | 더 큰 학생 모델, RL cold-start, zero-shot OOD 전이에서도 우수 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: RL의 탐색(Exploration) 도전**

기존 RL reasoning 방식은 정답/오답에 대한 **one-hot 정확도 보상(sparse reward)** 을 사용한다. 모델이 초기화 시점에 정답을 전혀 생성하지 못하면 gradient가 0이 되어 학습이 불가능하다.

$$r_i \in \{-1, -0.5, +1\} \quad \text{(기존 correctness-based reward)}$$

**문제 2: 목적 불일치(Objective Mismatch)**

실제 RL 훈련된 LM의 주요 사용 목적은 직접 배포가 아니라 **학생 증류(distillation)의 교사 역할**이다. 그러나 정확도 기반 보상으로 훈련된 모델의 추론 흔적(reasoning trace)은 학생 증류에 최적화되어 있지 않다.

**문제 3: 고비용 후처리 의존성**

기존 파이프라인(DeepSeek-R1, QwQ 등)은:
- 수백억 파라미터 모델 사용 ($>688{,}000$ GPU-hours for R1 training)
- 다중 생성, 필터링, GPT를 이용한 후처리 필요

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 RLT 태스크 재정의

기존 RL: $(q_i) \rightarrow \text{think} \rightarrow \text{solution}$ (처음부터 풀기)

RLT: $(q_i, s_i) \rightarrow \text{explanation}$ (해답 주어진 상태에서 설명 생성)

RLT는 문제 $q_i$와 정답 $s_i$를 모두 입력받아 학생이 이해할 수 있는 단계별 설명 $t_{o_i}$를 생성한다.

#### 2.2.2 RLT 보상 함수 (Dense Reward)

보상은 두 항으로 구성된다:

**① $r^{SS}$: 학생의 솔루션 이해도 측정**

학생 $\pi_s$가 교사의 설명 $t_{o_i}$와 질문 $q_i$가 주어졌을 때 정답 $s_i$를 생성할 log-probability를 측정한다.

$$r^{SS}(o_i, s_i, q_i) = \text{avg}\{\log \pi_s^{s_i}\} + \alpha \min\{\log \pi_s^{s_i}\}, \quad \text{where } \pi_s^{s_i} = \pi_s(s_i \mid t_{o_i}, q_i) $$

- $\text{avg}$: 솔루션 토큰 전체의 평균 log-prob
- $\min$: 가장 낮은 확률의 토큰 (모든 토큰을 균등히 고려)
- $\alpha = 0.01$: min 항의 스케일 계수

**② $r^{KL}$: 설명의 논리적 자연스러움 측정**

학생 관점에서 교사의 설명 토큰 $t_{o_i}$이 논리적 연속성을 갖는지를 KL divergence로 측정한다.

```math
r^{KL}(o_i, s_i, q_i) = \text{avg}\left\{\mathbb{D}_{KL}\left(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}}\right)\right\} + \alpha \max\left\{\mathbb{D}_{KL}\left(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}}\right)\right\}
```

$$\text{where} \quad \pi_s^{t_{o_i}} = \pi_s(t_{o_i} \mid q_i), \quad \pi_\theta^{t_{o_i}} = \pi_\theta(t_{o_i} \mid s_i, q_i)$$

- $\pi_\theta^{t_{o_i}}$: 정답 $s_i$가 주어진 교사 분포
- $\pi_s^{t_{o_i}}$: 질문 $q_i$만 주어진 학생 분포
- KL이 크면 → 학생 입장에서 이해 불가한 비약 존재 → 패널티

> **직관**: $r^{KL}$이 없으면, 교사는 단순히 정답 토큰을 반복하여 $r^{SS}$를 높이는 shortcut을 찾는다. $r^{KL}$은 각 설명 단계가 학생의 이전 이해를 바탕으로 자연스럽게 이어지도록 강제한다.

**③ 최종 RLT 보상**

$$r_i^{RLT} = r^{SS}(o_i, s_i, q_i) - \lambda r^{KL}(o_i, s_i, q_i) $$

- $\lambda = 3$: $r^{KL}$ 스케일 계수 (초기 completions에서 두 항의 크기를 맞추도록 설정)

#### 2.2.3 RLT 학습 목적함수

GRPO 알고리즘 기반:

$$J^{RLT}(\theta) = \mathbb{E}_{q,s \sim D,\, \{o\}_1^G \sim \pi_\theta(\cdot|s,q)} \left[\frac{1}{G} \sum_{i=1}^{G} \left(A_i^{RLT} - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right)\right] $$

Advantage 계산 (그룹 내 정규화):

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})} $$

- $G = 64$: 그룹 크기
- $\beta = 0.04$: KL 페널티 계수
- **핵심**: correctness-based reward와 달리 $r^{RLT}$는 **dense**하여 초기부터 의미 있는 학습 신호 제공

---

### 2.3 모델 구조

#### 전체 파이프라인

```
[훈련 데이터]
질문-정답 쌍 (q_i, s_i) × 17K개
         ↓
[Phase 1: 짧은 SFT]
RLT 입력 포맷 적응 (Labs/Bespoke 예시 traces 활용)
         ↓
[Phase 2: RLT RL 훈련]
Qwen2.5-7B-Instruct 기반
GRPO + r^RLT (dense reward)
125 steps, batch=1024, lr=1×10^{-6}
         ↓
[RLT 추론]
(q_i, s_i) → 설명 생성 (explanation tokens t_{o_i})
         ↓
[포맷 변환]
explanation → think 태그로 변환 + 정답 s_i 추가
         ↓
[학생 증류 (SFT)]
Qwen2.5-7B/32B-Instruct 파인튜닝
```

#### 주요 구성요소

| 구성요소 | 세부 사항 |
|---------|----------|
| **교사 모델 (RLT)** | Qwen2.5-7B-Instruct (7B) |
| **학생 모델 (보상 계산용)** | Qwen2.5-7B-Instruct (동일 사이즈) |
| **증류 학생** | Qwen2.5-7B/32B-Instruct |
| **RL 알고리즘** | GRPO (RLOO도 지원) |
| **훈련 데이터** | Li et al. [12] 의 17K math/coding 문제 |
| **컴퓨팅** | 단일 노드 8×H100 GPU |

#### 입출력 포맷 비교

| | 기존 RL 형식 | RLT 형식 |
|---|---|---|
| **입력** | 질문 $q_i$만 | 질문 $q_i$ + 정답 $s_i$ |
| **생성** | `<think>` → `<solution>` | `<solution>` → `<explanation>` |
| **목표** | 정답 도출 | 설명 생성 |
| **보상** | Sparse (정오답) | Dense (학생 이해도) |

---

### 2.4 성능 향상

#### 메인 실험 결과 (Table 1)

| 모델 | 데이터 | AIME 2024 | MATH 500 | GPQA Diamond | Overall |
|------|--------|-----------|----------|--------------|---------|
| Qwen2.5-7B-Instruct | - | 10.00 | 74.20 | 33.30 | 39.17 |
| Bespoke-7B-1K | 1K | 13.30 | 80.00 | 33.80 | 42.37 |
| **RLT-7B-1K (Ours)** | 1K | **20.00** | **80.40** | **41.90** | **47.43** |
| Bespoke-7B | 17K | 20.00 | 82.00 | 37.80 | 46.60 |
| **RLT-7B (Ours)** | 17K | **23.30** | **82.80** | **42.40** | **49.50** |
| s1-32B | 1K | 50.00 | 92.60 | 56.60 | 66.40 |
| Bespoke-32B-1K | 1K | 46.70 | 92.60 | 57.50 | 65.60 |
| **RLT-32B-1K (Ours)** | 1K | **60.00** | **94.00** | **60.10** | **71.37** |
| Bespoke-32B | 17K | 63.30 | 93.00 | 58.10 | 71.47 |
| **RLT-32B (Ours)** | 17K | **66.70** | **93.40** | **59.60** | **73.23** |

> **핵심**: 7B RLT의 raw 출력이 수십~수백 배 큰 모델(DeepSeek-R1, QwQ 등)의 후처리된 traces를 사용한 파이프라인을 능가

#### 비용 비교 (Table 4)

| 모델 | 훈련 (GPU-hours) | 데이터 생성 (GPU-hours) |
|------|-----------------|----------------------|
| DeepSeek R1 | >688,000 (H800) | >1,067 (H100) |
| **7B RLT** | **280.4 (H100)** | **6.7 (H100)** |

→ 훈련 비용 **~2,450배**, 데이터 생성 비용 **~159배** 절감

#### Cold-start RL 성능 (Table 2)

| 모델 | Overall |
|------|---------|
| RL no cold-start | 40.77 |
| RL cold-start (raw) + RL | 38.60 |
| RL cold-start (GPT) + RL | 43.93 |
| Bespoke-7B + RL | 48.30 |
| **RLT-7B + RL (Ours)** | **50.53** |

---

### 2.5 한계

| 한계 | 내용 |
|------|------|
| **정답 의존성** | 정답 $s_i$가 필요 → 정답 불명확한 도메인에서 적용 어려움 |
| **초기 SFT 의존** | RL만으로 훈련 불가; 초기 format 적응을 위한 SFT 단계 필요 |
| **추가 학생 모델 필요** | 보상 계산을 위해 별도 학생 LM을 메모리에 유지해야 함 |
| **컨텍스트 길이 제한** | 훈련 중 최대 16,384 토큰으로 제한 |
| **스케일링 미검증** | 7B RLT만 주로 실험; 더 큰 RLT에 대한 탐구 부족 |
| **데이터 폭 제한** | 시작 question-solution 쌍의 다양성/규모 확장 미실험 |

---

## 3. 일반화 성능 향상 가능성 (심층 분석)

### 3.1 도메인 내 일반화: 데이터 효율성

RLT의 보상 함수 $r^{RLT}$는 학생의 이해도를 직접 최적화하므로, **적은 데이터로도 높은 성능**을 달성한다.

- 1K 데이터로 Bespoke-7B(17K)보다 높은 성능 달성 (47.43 vs 46.60)
- R1 traces는 subsampling 시 성능이 크게 저하되는 반면, RLT traces는 일관성 유지

이는 $r^{SS}$가 학생이 각 예시에서 최대한 학습할 수 있도록 설명을 최적화하기 때문이다:

$$\max_\theta \, \mathbb{E}\left[\text{avg}\{\log \pi_s(s_i \mid t_{o_i}, q_i)\}\right]$$

### 3.2 OOD 제로샷 전이 (핵심 결과)

**카운트다운 태스크 실험** (섹션 4.4):
- RLT는 수학/코딩으로 훈련, 카운트다운(수식 계산 퍼즐)에 **제로샷** 적용
- 결과: RLT 제로샷 증류(56.6) > 카운트다운 직접 RL(50.8)

$$\text{RLT transfer OOD} \approx 56.6 > \text{Direct RL on CD} \approx 50.8$$

**왜 가능한가?**

$r^{KL}$ 항이 핵심 역할을 한다:

```math
r^{KL} = \text{avg}\left\{\mathbb{D}_{KL}(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}})\right\} + \alpha \max\left\{\mathbb{D}_{KL}(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}})\right\}
```

이 항은 교사가 **도메인 독립적인 "설명하는 방법"** 자체를 학습하도록 유도한다. 특정 도메인 지식이 아니라, 학생이 이해할 수 있는 단계적 추론 경로를 구성하는 일반적 능력을 RLT가 획득한다는 것을 시사한다.

> 직접 RL은 98.5% 이상의 풀린 문제가 기존 베이스라인과 겹침 → 탐색 challenge로 인해 새 도메인 지식 학습이 실질적으로 불가

### 3.3 학생 규모 일반화

7B RLT → 32B 학생 증류 성공:
- RLT-32B: 73.23 Overall (교사보다 4.5배 큰 학생)
- 모든 기존 방법 대비 우수

**왜 teacher < student 크기 불일치에서도 작동하는가?**

$r^{KL}$ 최적화 과정에서, 3B 교사가 특정 문제를 논리적으로 설명 불가능할 때에도:

$$\min_\theta r^{KL} \Rightarrow \pi_\theta(t_{o_i}|s_i, q_i) \rightarrow \pi_s(t_{o_i}|q_i)$$

교사의 출력 분포가 학생 분포에 수렴하도록 유도되어, 학생이 소화 가능한 설명을 생성한다. 이는 **capacity gap 문제**를 자동으로 완화하는 메커니즘이다.

### 3.4 코딩 및 다국어 일반화 (Appendix C.1)

수학으로만 훈련된 RLT-7B/32B가 코딩(LiveCodeBench), 다국어(OlympiadBench)에서도:

| 모델 | LCB-Hard | OlympiadBench | Overall |
|------|----------|---------------|---------|
| Bespoke-7B | 1.60 | 43.30 | 39.70 |
| **RLT-7B** | **3.30** | **46.10** | **40.37** |
| Bespoke-32B | 26.20 | 60.30 | 65.68 |
| **RLT-32B** | **32.50** | **64.00** | **67.62** |

### 3.5 보상-성능 상관관계 (Section 4.5)

RLT 보상이 실제 일반화 성능과 높은 상관성을 보임:

- Pearson 상관계수 **0.89** (보상 순위 vs 학생 성능)
- RL 훈련 전 7B 모델의 상위 ranked traces만으로 이미 베이스라인 R1 성능의 90% 달성

이는 $r^{RLT}$가 domain-specific 암기가 아닌 **진정한 이해도 향상**을 유도함을 시사한다.

### 3.6 일반화 한계

- 논문이 주로 수학/코딩에서 카운트다운으로의 전이만 검증; 완전히 다른 도메인(의학, 법률 등) 검증 부족
- Chu et al. (2025) [44]의 "SFT memorizes, RL generalizes" 주장과의 관계: RLT가 진정한 일반화를 유도하는지 vs 학생의 in-distribution 성능 향상인지 추가 검증 필요

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 계보

```
[2020-2022] 기초 연구
├── Hinton et al. (2015): Knowledge Distillation 원론
├── STaR (Zelikman et al., 2022): 자체 추론 bootstrapping
└── Chain-of-Thought (Kojima et al., NeurIPS 2022)

[2023-2024] Test-time Scaling 부상
├── Math-Shepherd (Wang et al., 2023): 단계별 검증
├── Snell et al. (2024): Test-time compute scaling
└── OpenAI o1 (2024): RL reasoning 새 패러다임

[2025] 현재 연구 경쟁
├── DeepSeek-R1 (Guo et al., 2025): GRPO 기반 대규모 RL
├── s1 (Muennighoff et al., 2025): 1K 데이터 Test-time scaling
├── Sky-T1 (NovaSky, 2025): QwQ 증류
├── Bespoke-Stratos (Labs, 2025): R1 증류 최적화
├── Li et al. (2025): 구조가 내용보다 중요
├── LIMO (Ye et al., 2025): 소량 데이터의 힘
└── **RLT (Cetin et al., 2025): 교사 특화 RL**
```

### 4.2 방법론 비교

| 항목 | 기존 RL (DeepSeek-R1) | 기존 증류 (s1, Bespoke) | **RLT** |
|------|----------------------|------------------------|---------|
| **보상 타입** | Sparse (one-hot) | N/A (SFT) | Dense |
| **입력** | 질문만 | 질문+추론흔적 | 질문+정답 |
| **탐색 의존성** | 높음 | 없음 | **없음** |
| **목적 정렬** | 문제풀기 | 증류 (간접) | **증류 (직접)** |
| **후처리 필요** | 필요 | 필요 | **불필요** |
| **모델 크기** | 671B | 7B-70B | **7B** |
| **OOD 전이** | 어려움 | 제한적 | **가능** |
| **비용** | 매우 높음 | 중간 | **낮음** |

### 4.3 Li et al. (2025) "구조가 내용보다 중요"와의 관계

Li et al.은 증류 데이터의 **구조와 포맷**이 내용보다 더 중요하다고 주장한다. RLT의 $r^{KL}$은 이를 직접 최적화하는 메커니즘으로 해석 가능하다:

$$r^{KL} \propto -\text{KL}(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}})$$

이는 학생 관점에서 자연스러운 추론 구조를 갖도록 강제하며, Li et al.의 발견을 실험적으로 뒷받침한다.

### 4.4 "SFT memorizes, RL generalizes" (Chu et al., 2025)와의 관계

Chu et al.은 RL이 SFT보다 더 나은 일반화를 유도한다고 주장한다. RLT는 SFT(학생 증류)를 수행하면서도 RL(교사 훈련)의 장점을 활용하는 **하이브리드 접근**으로, 두 방법의 장점을 결합한다.

### 4.5 Yue et al. (2025) "RL이 실제로 새 능력 유도하는가?"와의 관계

Yue et al.은 RL이 base model의 능력 이상을 유도하는지 의문을 제기한다. 이와 관련하여 RLT 논문도 직접 RL의 98.5% 이상 문제 중복을 관찰하며 같은 회의적 시각을 공유하고, 이를 우회하는 방법을 제시한다.

---

## 5. 미래 연구에 대한 영향과 고려사항

### 5.1 미래 연구에 미치는 영향

#### (1) 패러다임 전환: "교사 역할" 특화 모델

RLT는 LM을 문제 풀이자(solver)와 교사(teacher)로 **역할 분리**하는 새 패러다임을 제시한다. 향후 AI 훈련 파이프라인이 이 두 역할을 명시적으로 분리하고 각각 최적화하는 방향으로 발전할 가능성이 있다.

#### (2) Dense Reward Engineering의 중요성

$r^{RLT} = r^{SS} - \lambda r^{KL}$의 설계는 LM 훈련에서 **domain-specific dense reward**의 중요성을 재확인한다. 향후 다른 NLP 태스크에서도 유사한 dense reward 설계가 활발히 연구될 것이다.

#### (3) 소형 특화 모델의 가능성

7B RLT > 수천억 파라미터 범용 모델이라는 결과는 **"작지만 특화된 모델"** 패러다임을 강화한다. 이는 compute 민주화 측면에서 중요하며, 독립 연구자와 소규모 기관도 경쟁력 있는 AI 훈련 파이프라인을 구축할 수 있음을 시사한다.

#### (4) 교사-학생 공동 훈련 (Co-training)

논문이 미탐구 방향으로 언급한 교사-학생 동시 최적화는 유망한 연구 방향이다:

```math
\theta^*_{teacher}, \theta^*_{student} = \arg\max_{\theta_T, \theta_S} \mathbb{E}[r^{RLT}(\pi_{\theta_T}, \pi_{\theta_S})]
```

이는 curriculum learning, adaptive teaching 등과 결합 가능하다.

### 5.2 향후 연구 시 고려할 점

#### (1) 정답 없는 도메인 확장

현재 RLT는 verifiable solutions가 필요하다. 다음 연구가 필요하다:
- 열린 질문(open-ended QA), 창작, 의료 진단 등에서의 적용 방법
- 정답 대신 부분 정보나 다중 기준(multi-criteria)을 활용하는 보상 설계

#### (2) 교사 규모 스케일링 법칙

현재 3B와 7B 교사만 실험되었다. 다음이 중요하다:
- RLT 교사 크기와 학생 성능 사이의 **스케일링 법칙** 수립
- 특히 교사 < 학생 상황에서의 최적 교사 크기 탐구

#### (3) 보상 함수 계수 자동 조정

현재 $\lambda = 3, \alpha = 0.01$은 수동 설정이다:
- 자동 보상 가중치 조정(e.g., Lagrangian 방법)
- 학습 진행에 따른 동적 $\lambda$ 스케줄링

$$\lambda(t) = f(t, \mathbb{E}[r^{SS}], \mathbb{E}[r^{KL}])$$

#### (4) 다양한 도메인 전이의 체계적 평가

현재 수학 → 카운트다운만 검증됨:
- 과학, 법률, 의학 등 전문 도메인으로의 전이 평가 필요
- 도메인 유사성(domain similarity)과 전이 성능 관계 분석

#### (5) 학생 모델 선택의 영향

보상 계산에 사용되는 학생 모델의 선택이 RLT 품질에 미치는 영향 체계적 분석 필요:
- 강한 학생으로 훈련된 RLT가 약한 학생에게도 효과적인가?
- Section C.4의 "stronger students make stronger teachers" 현상의 메커니즘 규명

#### (6) 긴 컨텍스트 및 더 큰 RLT 스케일링

현재 16,384 토큰 한계 극복:
- 32,768+ 토큰 컨텍스트에서의 설명 품질 변화
- 더 큰 RLT(70B+)에서의 성능 탐구

#### (7) 동적 교사-학생 공동 훈련

$$\theta_T^{t+1}, \theta_S^{t+1} = \text{update}(\theta_T^t, \theta_S^t, r^{RLT}(\pi_{\theta_T^t}, \pi_{\theta_S^t}))$$

- 학생의 실시간 학습 상태에 교사가 적응하는 온라인 학습
- 커리큘럼 학습과의 결합 가능성

#### (8) "SFT memorizes, RL generalizes" 맥락에서 RLT 재평가

RLT로 증류된 학생이 진정한 일반화를 달성하는지 vs in-distribution 성능만 향상하는지 체계적 평가:
- Distribution shift 실험 (완전히 다른 유형의 문제)
- Few-shot 성능 평가
- 적대적 예시(adversarial examples)에 대한 강건성

---

## 참고 자료

본 답변은 다음 문헌을 직접 참조하였습니다:

**주요 논문 (PDF 원문 기반)**
- **Cetin, E., Zhao, T., & Tang, Y. (2025). "Reinforcement Learning Teachers of Test Time Scaling." NeurIPS 2025. arXiv:2506.08388v3**
  - GitHub: https://github.com/SakanaAI/RLT

**논문 내 인용 문헌 (직접 참조됨)**
- Guo et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948
- Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300 [GRPO]
- Muennighoff et al. (2025). "s1: Simple Test-Time Scaling." arXiv:2501.19393
- Li et al. (2025). "LLMs Can Easily Learn to Reason from Demonstrations: Structure, Not Content, Is What Matters!" arXiv:2502.07374
- Bespoke Labs. (2025). "Bespoke-Stratos: The Unreasonable Effectiveness of Reasoning Distillation." www.bespokelabs.ai
- Yue et al. (2025). "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" arXiv (참조 [9])
- Chu et al. (2025). "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-Training." arXiv:2501.17161
- Ye et al. (2025). "LIMO: Less is More for Reasoning." arXiv:2502.03387
- Li et al. (2025). "LIMR: Less is More for RL Scaling." arXiv:2502.11886
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
- Ahmadian et al. (2024). "Back to Basics: Revisiting REINFORCE Style Optimization for LLMs." arXiv:2402.14740 [RLOO]
- Zelikman et al. (2022). "STaR: Bootstrapping Reasoning with Reasoning." arXiv:2203.14465
- Hinton, Vinyals & Dean. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531
- Kojima et al. (NeurIPS 2022). "Large Language Models are Zero-Shot Reasoners."
- Snell et al. (2024). "Scaling LLM Test-Time Compute Optimally..." arXiv:2408.03314
- Gandhi et al. (2024). "Stream of Search (SoS): Learning to Search in Language." arXiv:2404.03683 [Countdown task]
- Rein et al. (2024). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." COLM 2024
- Hendrycks et al. (NeurIPS 2021). "Measuring Mathematical Problem Solving with the MATH Dataset."
- Jain et al. (2024). "LiveCodeBench." arXiv:2403.07974
- He et al. (2024). "OlympiadBench." arXiv:2402.14008
