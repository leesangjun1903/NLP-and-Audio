# Training Large Language Models to Reason in a Continuous Latent Space

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

LLM의 추론 과정을 자연어(언어 공간, language space)에 한정할 필요가 없으며, **연속적 잠재 공간(continuous latent space)** 에서 추론하는 것이 특정 과제에서 더 효과적일 수 있다는 주장입니다.

> 신경과학적 근거: 인간의 언어 네트워크는 수학적·논리적 추론 시 대체로 비활성 상태이며(Fedorenko et al., 2024; Monti et al., 2012), 언어는 추론보다 **소통**을 위해 최적화되어 있다는 점을 논문은 동기로 제시합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 패러다임 Coconut** | Chain of Continuous Thought: 언어 토큰 없이 히든 스테이트로 추론 |
| **다단계 커리큘럼 학습** | 언어 CoT를 점진적으로 연속 사고로 대체하는 훈련 전략 |
| **BFS 유사 추론 창발** | 명시적 지시 없이도 너비 우선 탐색 패턴이 자동으로 출현 |
| **ProsQA 데이터셋 제안** | 광범위한 계획 능력이 필요한 새로운 논리 추론 벤치마크 |
| **효율성-정확도 트레이드오프 개선** | CoT 대비 적은 토큰으로 유사하거나 더 높은 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 1: 언어 공간의 비효율성**

Chain-of-Thought(CoT)에서 대부분의 토큰은 텍스트 유창성을 위한 것이며, 실제 추론에 기여하지 않습니다. 반면 일부 핵심 토큰은 복잡한 계획이 필요한데, 현재 LLM 아키텍처는 모든 토큰에 동일한 컴퓨팅 예산을 할당합니다.

**문제 2: 탐색 불가능성 (Planning Limitation)**

CoT는 자기회귀적(autoregressive) 특성 때문에 하나의 결정론적 경로에 조기 커밋(premature commitment)하게 됩니다. 초기에 잘못된 경로를 선택하면 되돌아갈 수 없습니다.

**문제 3: 언어-추론 불일치 (Unfaithfulness)**

LLM이 CoT를 생성하더라도 실제 내부 추론 과정이 생성된 언어와 다를 수 있습니다 (Turpin et al., 2024).

---

### 2.2 제안하는 방법 (수식 포함)

#### 표준 언어 모델 표기

입력 시퀀스 $x = (x_1, \ldots, x_T)$에 대해 표준 LLM $\mathcal{M}$은:

$$H_t = \text{Transformer}(E_t)$$

$$\mathcal{M}(x_{t+1} \mid x_{\leq t}) = \text{softmax}(W h_t)$$

여기서:
- $E_t = [e(x_1), e(x_2), \ldots, e(x_t)]$: 위치 $t$까지의 토큰 임베딩 시퀀스
- $H_t \in \mathbb{R}^{t \times d}$: 마지막 레이어의 히든 스테이트 행렬
- $h_t = H_t[t, :]$: 위치 $t$의 마지막 히든 스테이트
- $e(\cdot)$: 토큰 임베딩 함수
- $W$: 언어 모델 헤드 파라미터

#### Coconut의 핵심 수정

잠재 모드(latent mode)에서, 위치 $i$(= `<bot>`)와 $j$(= `<eot>`) 사이의 연속 사고 구간에서:

$$E_t = [e(x_1), \ldots, e(x_i), \underbrace{h_i, h_{i+1}, \ldots, h_{t-1}}_{\text{연속 사고 (continuous thoughts)}}]$$

즉, 토큰 임베딩 $e(x_{t})$ 대신 **이전 스텝의 마지막 히든 스테이트 $h_{t-1}$** 을 다음 입력 임베딩으로 직접 사용합니다.

잠재 모드 종료 후($t \geq j$):

$$E_t = [e(x_1), \ldots, e(x_i), h_i, \ldots, h_{j-1}, e(x_j), \ldots, e(x_t)]$$

이 설계의 핵심은 **연속 사고가 완전히 미분가능(differentiable)** 하여 역전파(backpropagation)를 통한 end-to-end 최적화가 가능하다는 점입니다.

#### 훈련 목적 함수

표준 음의 로그 우도(Negative Log-Likelihood, NLL)를 사용하되, 질문 토큰과 연속 사고에 대한 손실은 마스킹합니다:

$$\mathcal{L} = -\sum_{t \in \mathcal{T}_{\text{answer}}} \log \mathcal{M}(x_t \mid x_{<t})$$

여기서 $\mathcal{T}_{\text{answer}}$는 연속 사고 이후의 정답 토큰 위치 집합입니다.

---

### 2.3 모델 구조

#### 다단계 커리큘럼 훈련 (Multi-Stage Curriculum Training)

```
Stage 0: [Q] <bot><eot> [Step1] [Step2] ... [StepN] [Answer]   (초기: 일반 CoT)
Stage 1: [Q] <bot> [c개 연속사고] <eot> [Step2] ... [StepN] [Answer]
Stage 2: [Q] <bot> [2c개 연속사고] <eot> [Step3] ... [StepN] [Answer]
  ...
Stage N: [Q] <bot> [N×c개 연속사고] <eot> [Answer]            (최종: 완전 잠재 추론)
```

- $c$: 하나의 언어 추론 단계를 대체하는 연속 사고의 수 (하이퍼파라미터)
- 각 스테이지 전환 시 **옵티마이저 상태 초기화** (Deng et al., 2024 방식 채택)
- $n$개의 연속 사고가 있을 때 $n+1$번의 순전파(forward pass) 수행

#### 추론(Inference) 과정

1. 질문 토큰 뒤에 `<bot>` 삽입
2. 잠재 모드: 마지막 히든 스테이트를 다음 입력으로 직접 피드백
3. `<eot>` 후 언어 모드로 전환하여 최종 답변 생성
4. `<eot>` 종료 시점: 고정 길이 패딩 방식 사용 (이진 분류기 방식도 유사한 성능)

---

### 2.4 성능 향상

#### 메인 실험 결과 (Table 1 기반)

| 방법 | GSM8k 정확도 (%) | GSM8k 토큰 수 | ProntoQA 정확도 (%) | ProntoQA 토큰 수 | ProsQA 정확도 (%) | ProsQA 토큰 수 |
|---|---|---|---|---|---|---|
| CoT | 42.9 | 25.0 | 98.8 | 92.5 | 77.5 | 49.4 |
| No-CoT | 16.5 | 2.2 | 93.8 | 3.0 | 76.7 | 8.2 |
| iCoT | 30.0 | 2.2 | 99.8 | 3.0 | 98.2 | 8.2 |
| Pause Token | 16.4 | 2.2 | 77.7 | 3.0 | 75.9 | 8.2 |
| **Coconut (Ours)** | **34.1** | **8.2** | **99.8** | **9.0** | **97.0** | **14.2** |

**핵심 관찰:**
- **ProntoQA/ProsQA**: Coconut이 CoT 대비 훨씬 적은 토큰으로 동등하거나 더 높은 정확도 달성
- **GSM8k**: CoT 절대 성능에는 못 미치나, 정확도-효율성 트레이드오프에서 우위
- **ProsQA (k=6)**: 약 97.0% vs CoT 77.5% — 계획 능력이 핵심인 과제에서 두드러진 우위

#### 추론 시간 비교 (Table 4)

| 방법 | GSM8k (초) | ProntoQA (초) | ProsQA (초) |
|---|---|---|---|
| No-CoT | 0.03 | 0.03 | 0.08 |
| CoT | 0.26 | 0.85 | 0.47 |
| Coconut | 0.09 | 0.11 | 0.15 |

---

### 2.5 한계

| 한계 | 설명 |
|---|---|
| **복잡한 수학 추론에서 CoT 미달** | GSM8k에서 CoT(42.9%) 대비 Coconut(34.1%)으로 절대 성능 낮음 |
| **커리큘럼 의존성** | 다단계 훈련 없이는 no-CoT와 유사한 성능. 언어 CoT 데이터 필요 |
| **병렬화 어려움** | $n+1$번의 순차적 순전파 필요로 훈련 효율 저하 |
| **대형 모델 전이 어려움** | Llama 3B/8B에서 개선폭이 GPT-2 대비 작음 (언어 편향 프리트레이닝 영향) |
| **연속 사고 수 확장 불안정** | $c=3$ 이상 시 훈련 손실 스파이크 발생 |
| **해석 가능성 부재** | 연속 사고는 언어로 직접 해석 불가 |
| **사전학습 미적용** | 파인튜닝 단계에서만 실험; 사전학습 스케일에서의 검증 부재 |

---

## 3. 일반화 성능 향상 가능성

### 3.1 BFS 유사 추론의 일반화 잠재력

Coconut의 가장 중요한 일반화 관련 특성은 **연속 사고가 여러 후보 추론 경로를 동시에 인코딩**할 수 있다는 점입니다.

$$p(\text{concept}_j \mid \text{continuous thought}_k) = \prod_{\text{tokens} \in \text{concept}_j} p(\text{token} \mid \text{previous context})$$

이를 암묵적 가치 함수(implicit value function)로 해석할 수 있습니다:

$$V(s_k) \approx p(\text{correct concept} \mid h_k)$$

ProsQA 실험에서 높이(height, 리프 노드까지의 최단 거리)가 낮은 노드일수록 정확한 확률 평가를 받는다는 관찰은 **지연된 결정(delayed commitment)의 일반화 이점**을 보여줍니다:

- **높이가 높은 노드 (초기 단계)**: 평가 불확실성 높음 → 여러 경로를 동시 유지
- **높이가 낮은 노드 (최종 단계에 가까울수록)**: 평가 확실성 높음 → 최적 경로로 수렴

### 3.2 일반화 가능성의 구체적 근거

**① 태스크 불가지론적(task-agnostic) 패턴 창발**

BFS 유사 추론은 명시적으로 학습되지 않았음에도 자연스럽게 창발되었습니다. 이는 연속 사고 메커니즘이 특정 태스크 구조에 과적합되지 않고 일반적인 탐색 전략을 내재화할 수 있음을 시사합니다.

**② 스케일링 가능성 (Test-Time Scaling)**

$$c \uparrow \Rightarrow \text{Accuracy} \uparrow \quad (\text{GSM8k에서 } c \in \{0,1,2\} \text{에서 단조 증가 확인})$$

연속 사고를 더 많이 사용할수록 성능이 향상되는 패턴은 **추론 시간 컴퓨팅 확장(test-time compute scaling)** 의 유효성을 보여줍니다.

**③ 이론적 뒷받침**

후속 연구인 Zhu et al. (2025b)은 연속 CoT가 여러 추론 경로를 중첩(superposition) 상태로 인코딩할 수 있어, 특정 과제에서 이산 CoT보다 효율적임을 이론적으로 증명했습니다.

**④ 할루시네이션 감소와 일반화**

ProsQA에서 CoT는 존재하지 않는 엣지를 생성(hallucination)하는 반면, Coconut은 이를 현저히 줄였습니다. 이는 잠재 추론이 **분포 외(out-of-distribution) 추론 패턴에도 더 견고할 가능성**을 시사합니다.

### 3.3 일반화의 현재 한계

- **훈련 데이터 의존성**: 언어 CoT 지도 데이터가 필요하여 CoT 데이터가 없는 새로운 도메인에서의 일반화 어려움
- **사전학습 미적용**: 현재는 파인튜닝 수준에서만 검증되었으며, 사전학습 수준으로 확장 시 일반화 성능 향상 기대
- **대형 모델 이식성**: Llama 3B/8B에서 개선폭이 작아, 언어 편향이 강한 모델에서의 일반화 한계 존재

---

## 4. 연구 영향 및 향후 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### 패러다임 전환 촉발

Coconut은 "LLM은 반드시 언어로 추론해야 한다"는 암묵적 가정에 도전하며, **잠재 공간 추론 연구의 새로운 방향**을 제시합니다. 이는 다음 연구 흐름을 촉발시킬 것으로 예상됩니다:

1. **잠재 공간 사전학습(Latent Space Pre-training)**: 언어 편향 없는 추론 특화 사전학습 방식 개발 필요 (논문 자체에서도 미래 과제로 언급)
2. **하이브리드 추론 시스템**: 언어 추론과 잠재 추론을 상황에 따라 결합하는 적응형 시스템 (DualFormer, Su et al., 2024 방향)
3. **암묵적 가치 함수 학습**: 연속 사고 내의 암묵적 가치 함수를 명시적으로 학습하는 방향 (강화학습과의 연계)

#### 관련 최신 연구와의 연계 (2020년 이후 비교 분석)

| 연구 | 발표 | 방향 | Coconut과의 관계 |
|---|---|---|---|
| **Wei et al. (2022)** Chain-of-Thought | 2022 | 언어 공간 추론 강화 | Coconut이 극복하고자 하는 기준선 |
| **Goyal et al. (2023)** Pause Tokens | 2023 | 학습가능 `<pause>` 토큰 삽입 | Coconut 대비 expressivity 확장 한계 |
| **Deng et al. (2024)** iCoT | 2024 | CoT를 점진적으로 내재화 | Coconut 커리큘럼 설계에 영감 제공; 순수 잠재 표현 없음 |
| **Pfau et al. (2024)** Filler Tokens | 2024 | "..." 필러 토큰으로 잠재 계산 | 병렬화 가능 문제에 한정, 일반 추론 미확장 |
| **Yao et al. (2023)** Tree of Thoughts | 2023 | 명시적 트리 탐색 | 언어 공간 내 탐색; Coconut은 암묵적 BFS 달성 |
| **Zelikman et al. (2024)** Quiet-STaR | 2024 | 발화 전 내부 사고 학습 | 여전히 언어 토큰 기반; Coconut은 진정한 연속 공간 |
| **Geiping et al. (2025)** Recurrent Depth | 2025 | 반복적 깊이로 잠재 추론 스케일링 | Coconut의 확장 방향으로 논문에서 직접 언급 |
| **Zhu et al. (2025a,b)** | 2025 | Coconut 이론적 분석 | 중첩(superposition) 창발 메커니즘 이론화 |
| **DeepSeek-R1 (Guo et al., 2025)** | 2025 | RL로 추론 능력 강화 | 언어 공간 내 강화학습; 잠재 공간 통합 가능성 |

---

### 4.2 향후 연구 시 고려할 점

#### ① 잠재 공간 사전학습 설계

현재 Coconut은 파인튜닝 수준에서만 적용됩니다. 더 강한 일반화를 위해서는:
- **사전학습 단계에서의 잠재 추론 통합** (논문 결론에서 명시적으로 제안)
- 다양한 도메인과 태스크에 걸친 잠재 표현의 **보편성(universality) 확보**

#### ② 훈련 효율 개선

$$n+1 \text{ forward passes} \Rightarrow \text{순차적 계산 병목}$$

- KV 캐시를 활용하더라도 순차적 특성은 불가피
- **병렬 연속 사고 생성** 또는 더 효율적인 역전파 방식 개발 필요

#### ③ 커리큘럼 설계 일반화

현재 커리큘럼은 CoT 데이터에 의존합니다:
- CoT 데이터 없이도 잠재 추론을 학습할 수 있는 **자기지도(self-supervised) 방식** 개발
- iCoT(Deng et al., 2024)와 Coconut의 결합: 더 세밀한 제거 스케줄 적용

#### ④ 대형 모델에서의 적용성

| 모델 | no-CoT | Coconut |
|---|---|---|
| Llama 3.2-3B | 26.0% | 31.7% |
| Llama 3-8B | 42.2% | 43.6% |

개선폭이 작은 이유는 언어 편향 프리트레이닝 때문으로 분석됩니다. 이를 해결하기 위해:
- **언어-잠재 이중 프리트레이닝** 전략 설계
- 잠재 공간을 명시적으로 추론에 최적화하는 방식 (Geiping et al., 2025; Barrault et al., 2024 방향과 통합)

#### ⑤ 해석 가능성 (Interpretability)

연속 사고는 언어로 직접 디코딩되지 않아 블랙박스 문제가 있습니다:
- Figure 9에서 첫 번째 연속 사고를 디코딩하면 중간 변수에 해당하는 토큰이 나타남 — 이 현상의 **체계적 분석** 필요
- 프로빙(probing) 기법을 통한 연속 사고의 내용 해석 연구 필요

#### ⑥ 강화학습과의 통합

DeepSeek-R1(Guo et al., 2025)과 같은 RL 기반 추론 강화 방식과 결합:
- 연속 사고를 **보상 신호(reward signal)** 로 학습하는 방식
- 암묵적 가치 함수를 명시적으로 최적화하는 방향

#### ⑦ 연속-언어 하이브리드 추론

```
[Question] → [언어 스케치] → <bot> [연속 사고] <eot> → [언어 답변]
```

언어로 추론의 뼈대(skeleton)를 생성하고, 세부 계산은 잠재 공간에서 수행하는 하이브리드 방식이 안정성과 성능을 동시에 개선할 수 있습니다.

---

## 참고 자료

**주 논문:**
- Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024). *Training Large Language Models to Reason in a Continuous Latent Space*. arXiv:2412.06769v3.

**논문 내 인용 핵심 참고문헌:**
- Wei, J. et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. NeurIPS 35.
- Deng, Y., Choi, Y., & Shieber, S. (2024). *From explicit CoT to implicit CoT: Learning to internalize CoT step by step*. arXiv:2405.14838.
- Goyal, S. et al. (2023). *Think before you speak: Training language models with pause tokens*. arXiv:2310.02226.
- Pfau, J., Merrill, W., & Bowman, S. R. (2024). *Let's think dot by dot: Hidden computation in transformer language models*. arXiv:2404.15758.
- Yao, S. et al. (2023). *Tree of thoughts: Deliberate problem solving with large language models*. NeurIPS 36.
- Zelikman, E. et al. (2024). *Quiet-STaR: Language models can teach themselves to think before speaking*. arXiv:2403.09629.
- Zhu, H. et al. (2025a). *Emergence of superposition: Unveiling the training dynamics of chain of continuous thought*. arXiv:2509.23365.
- Zhu, H. et al. (2025b). *Reasoning by superposition: A theoretical perspective on chain of continuous thought*. arXiv:2505.12514.
- Geiping, J. et al. (2025). *Scaling up test-time compute with latent reasoning: A recurrent depth approach*. arXiv:2502.05171.
- Guo, D. et al. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning*. arXiv:2501.12948.
- Fedorenko, E., Piantadosi, S. T., & Gibson, E. A. F. (2024). *Language is primarily a tool for communication rather than thought*. Nature, 630(8017):575–586.
- Saparov, A. & He, H. (2022). *Language models are greedy reasoners*. arXiv:2210.01240.
- Cobbe, K. et al. (2021). *Training verifiers to solve math word problems*. arXiv:2110.14168.
- Su, D. et al. (2024). *DualFormer: Controllable fast and slow thinking by learning with randomized reasoning traces*. arXiv:2410.09918.
- Barrault, L. et al. (2024). *Large concept models: Language modeling in a sentence representation space*. arXiv:2412.08821.

**코드 저장소:**
- https://github.com/facebookresearch/coconut
