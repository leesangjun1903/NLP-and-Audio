# Titans: Learning to Memorize at Test Time

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Titans 논문의 핵심 주장은 다음 세 가지로 요약됩니다:

1. **메모리 이분법**: Attention은 정확한 의존성 모델링을 제공하지만 고정된 컨텍스트 창에 제한된 **단기 기억(short-term memory)**으로, Neural Memory는 과거 데이터를 파라미터에 압축하는 **장기 기억(long-term memory)**으로 기능할 수 있다.

2. **테스트 시간 학습**: 신경망 메모리 모듈이 훈련 데이터에 과적합하지 않고, **테스트 시간에 온라인 방식으로 데이터를 기억하는 법을 학습**할 수 있다.

3. **하이브리드 아키텍처 우위**: 장기 기억(Neural Memory) + 단기 기억(Attention) + 영구 기억(Persistent Memory)의 조합이 Transformer 및 최신 선형 순환 모델보다 우수하다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| Neural Long-term Memory | 그래디언트 기반 Surprise Metric으로 학습하는 신경 장기 기억 모듈 |
| Forgetting Mechanism | 적응형 가중치 감쇠를 통한 메모리 관리 |
| 병렬화 알고리즘 | matmul 기반 빠른 병렬 훈련 |
| Titans 아키텍처 3종 | MAC, MAG, MAL 세 가지 변형 |
| 이론적 표현력 | TC⁰를 초월하는 표현 능력 증명 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 1: Transformer의 이차 복잡도**

표준 Attention의 복잡도는 시퀀스 길이 $N$에 대해 $O(N^2)$으로, 매우 긴 컨텍스트에서 비실용적이다.

$$\mathbf{y}_i = \sum_{j=1}^{i} \frac{\exp\!\left(\mathbf{Q}_i^\top \mathbf{K}_j / \sqrt{d_{\text{in}}}\right) \mathbf{V}_j}{\sum_{\ell=1}^{i} \exp\!\left(\mathbf{Q}_i^\top \mathbf{K}_\ell / \sqrt{d_{\text{in}}}\right)} $$

**문제 2: 선형 순환 모델의 고정 크기 메모리 한계**

선형 Transformer는 키-값 쌍을 행렬 메모리에 누적 압축하지만:

$$\mathcal{M}_t = \mathcal{M}_{t-1} + K_t^\top V_t $$

이 단순 누적 방식은 긴 시퀀스에서 **메모리 오버플로우(memory overflow)**를 유발한다.

**문제 3: 일반화, 길이 외삽, 추론 능력 부족**

기존 모델들(Hopfield Network, LSTM, Transformer)은 모두 인간 뇌의 복합적 메모리 시스템(단기/장기/메타 기억)을 제대로 모사하지 못한다.

---

### 2.2 제안 방법 (수식 포함)

#### (A) Surprise Metric 기반 장기 기억 학습

**직관**: 인간은 놀라운(surprising) 사건을 더 잘 기억한다. 신경망에서 놀라움은 입력에 대한 그래디언트 크기로 측정된다.

**Step 1 - 연상 기억 손실 함수 정의:**

입력 $x_t$를 키($\mathbf{k}_t$)와 값($\mathbf{v}_t$)으로 투영:

$$\mathbf{k}_t = x_t W_K, \quad \mathbf{v}_t = x_t W_V $$

메모리 모듈 $\mathcal{M}$이 키-값 연상을 학습하도록 손실 정의:

$$\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2 $$

**Step 2 - 기본 Surprise 기반 업데이트:**

$$\mathcal{M}_t = \mathcal{M}_{t-1} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{Surprise}} $$

**Step 3 - 모멘텀 기반 개선 (과거 놀라움 + 현재 놀라움):**

단순 그래디언트 방식은 큰 놀라움 이후 그래디언트가 소멸하는 문제가 있다. 이를 해결하기 위해 모멘텀을 도입:

$$\mathcal{M}_t = \mathcal{M}_{t-1} + S_t $$

$$S_t = \underbrace{\eta_t S_{t-1}}_{\text{Past Surprise}} - \underbrace{\theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{Momentary Surprise}} $$

여기서:
- $\eta_t \in [0,1]$: 데이터 의존적 놀라움 감쇠율 (컨텍스트 변화 감지)
- $\theta_t$: 현재 놀라움의 반영 정도

**Step 4 - 적응형 망각 메커니즘 추가:**

$$\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t $$

$$S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t) $$

$\alpha_t \in [0,1]$: 게이팅 메커니즘
- $\alpha_t \to 0$: 과거 추상 정보 유지
- $\alpha_t \to 1$: 메모리 전체 초기화

**Step 5 - 메모리 검색(Retrieval):**

쿼리 $\mathbf{q}_t = x_t W_Q$로 메모리에서 정보 검색 (가중치 업데이트 없는 순전파):

$$y_t = \mathcal{M}^*({\mathbf{q}_t}) $$

#### (B) 병렬화 훈련 알고리즘

시퀀스를 청크(chunk) 크기 $b$로 분할하여 미니배치 그래디언트 하강으로 재공식화:

$$\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t) = \beta_t \mathcal{M}_0 - \sum_{i=1}^{t} \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(\mathcal{M}_{t'}; x_i) $$

여기서 $\beta_i = \prod_{j=1}^{i}(1 - \alpha_j)$.

선형 메모리의 경우:

$$\nabla \ell(W_0; x_t) = (W_0 x_t - x_t) x_t^\top \Rightarrow \sum_{i=1}^{b} \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(W_0; x_i) = \Theta_b \mathbf{B}_b (W_0 X - X) X^\top $$

모멘텀 항에 대해 선형 점화식을 병렬 연관 스캔(parallel associative scan)으로 계산:

$$S_t = \eta_t S_{t-1} - \theta_t u_t $$

여기서 $u_t = \nabla \ell(\mathcal{M}_{t'}; x_t)$.

#### (C) 영구 기억 (Persistent Memory)

데이터 독립적 학습 가능 파라미터 $P = [p_1, p_2, \ldots, p_{N_p}]$를 시퀀스 앞에 추가:

$$x_{\text{new}} = [p_1 \; p_2 \; \cdots \; p_{N_p}] \; \| \; x $$

이는 FFN의 역할을 수행하며 (Sukhbaatar et al., 2019):

$$FFN(x) = W_V \, \text{Softmax}(W_K x) $$

---

### 2.3 모델 구조: Titans 3가지 변형

#### 변형 1: Memory as a Context (MAC)

시퀀스를 고정 크기 세그먼트 $S^{(t)}$로 분할. 현재 세그먼트가 장기 기억 쿼리:

$$h_t = \mathcal{M}^*_{t-1}(\mathbf{q}_t), \quad \mathbf{q}_t = S^{(t)} W_Q $$

$$\tilde{S}^{(t)} = [p_1 \; p_2 \; \cdots \; p_{N_p}] \; \| \; h_t \; \| \; S^{(t)} $$

$$y_t = \text{Attn}(\tilde{S}^{(t)}) $$

$$\mathcal{M}_t = \mathcal{M}_{t-1}(y_t), \quad o_t = y_t \otimes \mathcal{M}^*_t(y_t) $$

**장점**: Attention이 장기 기억과 현재 정보를 모두 보며 필요한 정보를 선택할 수 있음.

#### 변형 2: Memory as a Gate (MAG)

슬라이딩 윈도우 Attention(SWA)과 신경 메모리를 병렬로 결합:

$$\tilde{x} = [p_1 \; p_2 \; \cdots \; p_{N_p}] \; \| \; x $$

$$y = \text{SW-Attn}^*(\tilde{x}) $$

$$o = y \otimes \mathcal{M}(\tilde{x}) $$

#### 변형 3: Memory as a Layer (MAL)

신경 메모리를 레이어로 사용한 순차적 결합 (기존 하이브리드 모델과 유사):

$$\tilde{x} = [p_1 \; p_2 \; \cdots \; p_{N_p}] \; \| \; x $$

$$y = \mathcal{M}(\tilde{x}) $$

$$o = \text{SW-Attn}(y) $$

---

### 2.4 성능 향상

#### 언어 모델링 (Perplexity ↓, 760M 파라미터)

| 모델 | Wiki PPL↓ | LMB PPL↓ | Avg Acc↑ |
|------|-----------|----------|----------|
| Transformer++ | 25.21 | 27.64 | 48.69 |
| Mamba2 | 22.94 | 28.37 | 48.34 |
| Gated DeltaNet-H2* | 19.88 | 20.83 | 51.49 |
| **Titans (MAC)** | **19.93** | **20.12** | **52.51** |
| **Titans (MAG)** | **18.61** | **19.86** | **52.50** |

#### Needle-in-a-Haystack (S-NIAH, 16K 컨텍스트)

| 모델 | S-NIAH-PK | S-NIAH-N | S-NIAH-W |
|------|-----------|----------|----------|
| Mamba2 | 5.4 | 0.0 | 0.0 |
| DeltaNet | 71.4 | 5.4 | 0.0 |
| **Titans (MAC)** | **98.4** | **97.4** | **95.2** |

**BABILong 벤치마크**: Titans (MAC)은 파라미터 수가 약 70배 많은 Llama3.1-8B+RAG, GPT-4를 능가.

#### 시계열 예측 (ETTm1 MSE ↓)

| 모델 | MSE |
|------|-----|
| iTransformer | 0.407 |
| **Neural Memory** | **0.358** |

---

### 2.5 한계

1. **연산 비용**: 심층 메모리($L_M \geq 2$)는 Mamba2, Gated DeltaNet 대비 훈련 처리량(throughput)이 낮음.
2. **구현 최적화 부족**: Mamba2와 같은 고도로 최적화된 커널이 없음.
3. **MAL의 열세**: MAL 변형이 MAC, MAG 대비 성능이 낮아, 층별 순차 결합 방식의 한계를 보임.
4. **대규모 모델 결과 미완성**: 논문 자체에서 더 큰 모델 결과를 후속 버전에 발표할 것이라 명시.
5. **메모리 깊이-효율 트레이드오프**: 깊은 메모리는 성능은 좋지만 훈련이 더 느림.

---

## 3. 일반화 성능 향상 가능성

### 3.1 테스트 시간 학습(Test-Time Learning)을 통한 일반화

Titans의 핵심 혁신 중 하나는 **분포 외(out-of-distribution) 데이터에 대한 적응 능력**이다.

논문은 다음과 같이 명시한다:

> *"In this setup, the model is learning a function that is capable of memorization, but it is not overfitting to the training data, resulting in a better generalization at test time."*

이는 기존 LLM의 암기(memorization)가 일반화를 해치는 현상(Bayat et al., 2024)과 대비된다. Titans의 신경 메모리는:
- 훈련 데이터를 파라미터에 고정시키는 것이 아닌
- **테스트 시간에 주어진 컨텍스트를 온라인으로 학습**하는 메타 모델

따라서 테스트 데이터가 훈련 분포를 벗어나더라도, 해당 시퀀스의 패턴을 즉석에서 학습하여 적응한다.

### 3.2 망각 메커니즘을 통한 일반화

적응형 망각 메커니즘 $\alpha_t$는 일반화에 결정적 기여를 한다:

$$\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t$$

- **관련 없는 이전 정보를 선택적으로 삭제**함으로써 새로운 패턴 학습에 방해받지 않음.
- 컨텍스트가 바뀔 때 $\alpha_t \to 1$로 설정하여 메모리를 재초기화 가능.
- 이는 Continual Learning의 catastrophic forgetting 문제를 완화함.

**Ablation Study 결과** (망각 메커니즘 제거 시):

$$\text{PPL}: 27.01 \to 29.04, \quad \text{Long Context Acc}: 92.68 \to 85.60$$

망각 메커니즘 제거가 가장 큰 성능 저하를 유발, 일반화에 핵심 역할임을 확인.

### 3.3 깊은 비선형 메모리를 통한 일반화

선형 메모리($L_M=1$)는 데이터 간의 선형 의존성만 포착한다:

$$\ell(W_{t-1}; x_t) = \|W_{t-1} \mathbf{k}_t - \mathbf{v}_t\|_2^2$$

이는 온라인 선형 회귀와 동치. 반면, 심층 메모리($L_M \geq 2$)는 유니버설 근사 정리(Hornik et al., 1989)에 의해 비선형 의존성도 포착:

$$\text{PPL (Linear Memory)}: 28.49 \quad \text{vs} \quad \text{PPL (Deep Memory, } L_M=4): \approx 26$$

실험 결과 깊은 메모리는 특히 더 긴 시퀀스에서 강건성(robustness)이 높아, 분포 외 길이를 가진 시퀀스에도 잘 일반화한다.

### 3.4 다양한 도메인에서의 일반화

Titans는 단일 도메인에 특화되지 않고 다음 다양한 도메인에서 일관된 성능 향상을 보임:

| 도메인 | 결과 |
|--------|------|
| 언어 모델링 | Transformer++ 대비 PPL 감소 |
| 상식 추론 | 모든 벤치마크에서 우수 |
| DNA 서열 모델링 | GenomicsBenchmarks에서 SOTA 수준 |
| 시계열 예측 | ETT, ECL, Weather 등 전 데이터셋 우수 |
| 긴 문서 추론 | BABILong에서 GPT-4 능가 |

이는 신경 메모리 구조가 **특정 아키텍처 가정에 의존하지 않는 범용 귀납 편향(inductive bias)**을 제공함을 시사한다.

### 3.5 이론적 표현력과 일반화

**Theorem 4.1** (논문 원문):
> *Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to TC⁰, Titans are capable of solving problems beyond TC⁰, meaning that Titans are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.*

TC⁰를 초월하는 표현 능력은 Transformer가 해결할 수 없는 상태 추적(state tracking) 문제를 처리할 수 있음을 의미하며, 이는 이론적으로 더 넓은 함수 클래스를 학습할 수 있어 일반화 잠재력이 높다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 선형 순환 모델 계보와 비교

$$\text{1세대} \to \text{2세대} \to \text{3세대} \to \text{Titans}$$

| 세대 | 대표 모델 | 핵심 특징 | Titans 대비 한계 |
|------|-----------|-----------|-----------------|
| 1세대 | RetNet (2023), S4 (2022), RWKV (2023) | 데이터 독립 전이 행렬 | 망각 메커니즘 없음, 선형 메모리 |
| 2세대 | Mamba (2024), Griffin (2024), xLSTM (2024) | 게이팅 메커니즘 추가 | 여전히 선형 메모리, 모멘텀 없음 |
| 3세대 | TTT (2024), DeltaNet (2024), Gated DeltaNet (2024) | 메타러닝/온라인 학습 기반 업데이트 | 모멘텀 없거나 망각 메커니즘 미흡 |
| **Titans** | LMM, MAC, MAG, MAL | 모멘텀+망각+깊은 비선형 메모리 | — |

### 4.2 TTT (Test-Time Training) 계열과 비교

**TTT Layer (Yu Sun et al., 2024)**: 그래디언트 기반 업데이트를 사용하는 가장 유사한 모델

| 비교 항목 | TTT | Titans (LMM) |
|-----------|-----|--------------|
| 망각 메커니즘 | ❌ 없음 | ✅ 있음 ($\alpha_t$) |
| 모멘텀 | ❌ 순간 놀라움만 | ✅ 과거+현재 놀라움 |
| 메모리 깊이 | 실험적 미검증 | ✅ $L_M=1,2,3,4$ 체계적 분석 |
| 이론적 표현력 | TC⁰ | TC⁰ 초월 |

**S-NIAH 16K 성능**: TTT 4.4% vs Titans (MAC) **97.4%** — 망각 메커니즘의 결정적 차이.

### 4.3 Mamba/Mamba2와 비교

**Mamba (Gu & Dao, 2024)**: SSM 기반 선택적 상태 공간 모델

- 게이팅 메커니즘 보유 → 망각 능력 있음
- 그러나 델타 룰(Delta Rule) 미사용 → 과거 메모리 직접 삭제 불가
- 선형(벡터값) 메모리 → 비선형 의존성 포착 한계
- S-NIAH-W 16K: Mamba2 **0.0%** vs Titans **95.2%**

### 4.4 DeltaNet / Gated DeltaNet과 비교

**DeltaNet (Yang et al., 2024)**: 델타 룰 기반 메모리 업데이트

$$S_{t+1} = S_t \left(I - \theta_t \mathbf{k}_t \mathbf{k}_t^\top\right) + \theta_t \mathbf{v}_t \mathbf{k}_t^\top $$

**Gated DeltaNet (Yang et al., 2024)**: 망각 게이트 추가

LMM은 Gated DeltaNet을 다음 세 측면에서 일반화:
1. **모멘텀 기반 룰**: $\eta_t = 0$ 설정 시 Gated DeltaNet과 동치
2. **깊은 메모리**: 선형 메모리 한계 극복
3. **비선형 점화식**: 인트라-청크 선형 + 인터-청크 비선형

### 4.5 RMT (Recurrent Memory Transformer)와 비교

**RMT (Bulatov et al., 2022)**: 벡터값 메모리를 청크 간 전달

- 메모리 크기: 16 토큰 크기의 벡터 → 표현력 매우 제한
- BABILong Fine-tuning: RMT << Titans (MAC)

Titans (MAC)는 RMT의 **일반화 버전**: 소규모 벡터 메모리 대신 신경망 파라미터 전체를 메모리로 사용.

### 4.6 Infini-Attention과 비교

**Infini-Attention (Munkhdalai et al., 2024)**: 선형 어텐션 기반 압축 메모리

- 단순 선형 메모리 (행렬값) → 비선형 패턴 포착 불가
- 모멘텀 없음, 망각 메커니즘 미흡
- Titans의 MAC 변형과 개념적으로 유사하나 메모리 표현력에서 열세

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (A) 새로운 연구 패러다임 제시

Titans는 **"시퀀스 모델 = 메모리 시스템"** 관점을 체계화하여, 이후 연구가 메모리 구조를 명시적으로 설계하도록 유도한다. 특히:

- **메모리 구조 설계**: 어떤 신경망 아키텍처가 메모리로 최적인가? (현재 MLP 사용, 향후 다른 구조 탐색 여지)
- **메모리 통합 방식**: MAC/MAG/MAL 이외의 새로운 통합 방식 탐색
- **다중 시간 규모 기억**: 인간 기억처럼 초단기/단기/장기 메모리를 계층적으로 구성

#### (B) 연속 학습(Continual Learning) 연구에 영향

테스트 시간에 지속적으로 학습하는 망각 메커니즘은 Continual Learning 분야의 **catastrophic forgetting 해결책**으로 활용될 수 있다. $\alpha_t$를 통한 선택적 기억 삭제는 새로운 태스크 학습 시 이전 태스크 정보를 보존하는 방법론에 영감을 줄 수 있다.

#### (C) 긴 컨텍스트 처리 패러다임 전환

2M 이상의 컨텍스트 창 지원은 다음 분야에서 실질적 영향을 미칠 것으로 예상:
- **유전체 분석**: 수백만 염기쌍을 단일 모델로 처리
- **초장편 문서 이해**: 책 전체, 법률 문서, 코드베이스 전체
- **비디오 이해**: 장시간 비디오의 시간적 의존성 포착

#### (D) 메타러닝과 언어 모델의 통합

Titans의 내부 루프(inner-loop) 최적화 방식은 **MAML(Model-Agnostic Meta-Learning)** 계열 연구와 언어 모델 연구의 융합을 촉진할 것이다. 특히 "어떻게 빠르게 새로운 분포에 적응하는가"라는 질문에 새로운 답을 제시한다.

### 5.2 앞으로 연구 시 고려할 점

#### (A) 이론적 측면

1. **수렴성 분석**: 테스트 시간 내부 루프 최적화의 수렴 조건과 수렴 속도에 대한 엄밀한 이론적 분석 필요.

2. **일반화 경계**: VC 차원 또는 PAC 학습 프레임워크에서 Titans의 일반화 오차 경계 분석 필요.

3. **최적 메모리 깊이**: $L_M$ 값에 따른 표현력-효율 트레이드오프의 이론적 특성화.

#### (B) 아키텍처 측면

4. **메모리 모듈 다양화**: 논문 자체에서 언급하듯, MLP 대신 더 표현력 있는 아키텍처(Mamba, Attention 등)를 메모리 모듈로 사용하는 것이 흥미로운 미래 연구가 될 수 있음.

5. **청크 크기 최적화**: 현재 고정 청크 크기를 사용하지만, 입력 데이터의 특성에 따라 동적으로 조정하는 적응형 청크 방식 연구.

6. **멀티모달 확장**: 텍스트-이미지-오디오 등 다양한 모달리티에서 장기 메모리가 어떻게 작동하는지 탐구.

#### (C) 훈련 효율성 측면

7. **커널 최적화**: Mamba2의 FlashAttention처럼, LMM을 위한 특화된 하드웨어 커널 개발 필요. 현재 LMM은 Mamba2 대비 처리량이 낮음.

8. **청크 함수로서의 파라미터**: $\alpha_t, \theta_t, \eta_t$를 토큰 의존 대신 청크 의존으로 설정하여 훈련 속도를 높이는 방향 탐구.

#### (D) 일반화 및 안전성 측면

9. **프라이버시 리스크**: 신경망이 테스트 시간에 데이터를 파라미터에 인코딩하므로, 민감한 정보가 메모리에 영구 저장될 수 있는 프라이버시 문제 고려 필요.

10. **분포 외 데이터 탐지**: 테스트 시간 학습이 진정으로 OOD 적응을 제공하는지, 아니면 단순히 트레이닝 데이터에 인접한 분포에만 효과적인지 체계적으로 검증 필요.

11. **하이퍼파라미터 민감도**: $\eta_t$, $\theta_t$, $\alpha_t$가 모두 입력 의존적이므로, 이들의 초기화 및 정규화 방법이 성능에 미치는 영향 분석.

#### (E) 실증적 검증 측면

12. **더 큰 모델 스케일**: 논문은 760M까지만 결과를 보고하고 있어, 7B, 70B 규모에서의 검증이 필요.

13. **다양한 데이터셋**: FineWeb-Edu 외 다양한 훈련 데이터에서의 일반화 성능 검증.

14. **추론 능력(Reasoning)**: 단순 회상을 넘어 복잡한 다단계 추론 태스크에서의 성능 검증.

---

## 참고 자료

**주요 참고 논문 (본 논문 PDF 내 인용 기준)**:

1. **Behrouz, A., Zhong, P., & Mirrokni, V. (2024).** *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663v1.

2. **Vaswani, A. et al. (2017).** *Attention is All You Need.* NeurIPS 2017.

3. **Katharopoulos, A. et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.

4. **Gu, A. & Dao, T. (2024).** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024.

5. **Dao, T. & Gu, A. (2024).** *Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality.* arXiv:2405.21060.

6. **Yang, S., Wang, B., Shen, Y. et al. (2024).** *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024.

7. **Yang, S., Wang, B., Zhang, Y. et al. (2024).** *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.

8. **Yang, S., Kautz, J., & Hatamizadeh, A. (2024).** *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464.

9. **Sun, Y. et al. (2024).** *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv:2407.04620.

10. **Hornik, K., Stinchcombe, M., & White, H. (1989).** *Multilayer Feedforward Networks are Universal Approximators.* Neural Networks 2(5).

11. **Merrill, W., Petty, J., & Sabharwal, A. (2024).** *The Illusion of State in State-Space Models.* ICML 2024.

12. **Hsieh, C. P. et al. (2024).** *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024.

13. **Kuratov, Y. et al. (2024).** *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS 2024 Datasets Track.

14. **Bulatov, A., Kuratov, Y., & Burtsev, M. (2022).** *Recurrent Memory Transformer.* NeurIPS 2022.

15. **Munkhdalai, T., Faruqui, M., & Gopal, S. (2024).** *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-Attention.* arXiv:2404.07143.

16. **Bayat, R. et al. (2024).** *The Pitfalls of Memorization: When Memorization Hurts Generalization.* arXiv:2412.07684.

17. **Smith, J. T., Warrington, A., & Linderman, S. (2023).** *Simplified State Space Layers for Sequence Modeling.* ICLR 2023.

18. **Sukhbaatar, S. et al. (2019).** *Augmenting Self-Attention with Persistent Memory.* arXiv:1907.01470.
