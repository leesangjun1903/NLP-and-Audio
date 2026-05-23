# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

> **논문 정보**
> - **저자**: Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin
> - **출처**: arXiv:2407.04620v4 (2024), ICML 2025 채택
> - **코드**: [JAX](https://github.com/test-time-training/ttt-lm-jax) / [PyTorch](https://github.com/test-time-training/ttt-lm-pytorch)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **시퀀스 모델링에서 선형 복잡도(linear complexity)를 유지하면서도 표현력 있는 은닉 상태(expressive hidden state)를 갖는 새로운 프레임워크**를 제안합니다. 핵심 아이디어는 다음과 같습니다:

1. **RNN의 은닉 상태를 머신러닝 모델 자체로 정의**하고,
2. **업데이트 규칙을 자기지도 학습(self-supervised learning)의 한 스텝**으로 만듭니다.

테스트 시퀀스에서도 은닉 상태가 학습(training)을 통해 업데이트되므로, 이 레이어를 **Test-Time Training (TTT) 레이어**라고 명명합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **개념적 프레임워크** | 은닉 상태를 임의의 ML 모델로 인스턴스화할 수 있는 실용적 프레임워크 제시 |
| **두 가지 인스턴스화** | TTT-Linear (선형 모델)과 TTT-MLP (2층 MLP)를 은닉 상태로 사용 |
| **이론적 동등성 증명** | TTT + 선형 모델 + 배치 GD = Linear Attention과 동등함을 증명 (Theorem 1) |
| **효율성 개선** | 미니배치 TTT와 이중 형식(dual form)을 통한 GPU/TPU 활용 최적화 |
| **장문맥 성능** | 컨텍스트 길이가 길어질수록 Mamba 대비 TTT의 우위가 확대됨을 실증 |

> [!IMPORTANT]
> TTT 레이어의 가장 큰 차별점은 **테스트 시에도 은닉 상태가 자기지도 학습을 통해 지속적으로 업데이트**된다는 점입니다. 이로 인해 Transformer처럼 더 많은 토큰을 조건으로 할수록 perplexity가 계속 감소하는 특성을 보입니다. 반면, Mamba는 16k 컨텍스트 이후 perplexity 감소가 정체됩니다.

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

시퀀스 모델링에서 두 가지 핵심 트레이드오프가 존재합니다:

- **Self-attention (Transformer)**: 장문맥에서 성능이 우수하나, 시간 복잡도가 $O(T^2)$으로 이차적(quadratic)
- **기존 RNN (LSTM, Mamba 등)**: 시간 복잡도가 $O(T)$으로 선형적이지만, 고정 크기의 은닉 상태로 문맥을 압축해야 하므로 **장문맥에서 표현력이 제한**됨

Kaplan et al. (2020)의 스케일링 법칙 논문에서 이미 LSTM이 Transformer처럼 스케일링되지 못한다는 것을 보였고, 저자들은 최신 RNN인 Mamba에서도 **16k 컨텍스트 이후 perplexity 감소가 정체**되는 동일한 문제를 관찰합니다.

> **문제의 본질**: RNN은 문맥을 고정 크기 은닉 상태로 압축해야 하는데, 이 압축 휴리스틱이 수천~수백만 토큰 간의 구조와 관계를 발견하기에 충분히 표현력 있지 않습니다.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기본 구조: 은닉 상태 = ML 모델

은닉 상태 $s_t$를 모델 $f$의 가중치 $W_t$로 정의합니다.

**출력 규칙 (Output Rule)**:

$$z_t = f(x_t; W_t)$$

직관적으로, 출력 토큰 $z_t$는 업데이트된 가중치 $W_t$로 $x_t$에 대한 예측을 수행한 결과입니다.

**업데이트 규칙 (Update Rule)** — 자기지도 손실 $\ell$에 대한 경사 하강법:

$$W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$$

여기서 $\eta$는 학습률입니다.

#### 2.2.2 자기지도 학습 과제의 설계

단순한 재구성(reconstruction) 대신, **다중 뷰(multi-view) 재구성 과제**를 학습합니다:

$$\ell(W; x_t) = \| f(\theta_K x_t; W) - \theta_V x_t \|^2$$

여기서:
- $\theta_K x_t$: **훈련 뷰 (training view)** — 입력의 저차원 투영 (손상된 입력)
- $\theta_V x_t$: **레이블 뷰 (label view)** — 재구성 목표
- $\theta_K, \theta_V$: 외부 루프(outer loop)에서 학습되는 파라미터

출력 규칙도 수정되어 **테스트 뷰 (test view)**를 사용합니다:

$$z_t = f(\theta_Q x_t; W_t)$$

$\theta_K, \theta_V, \theta_Q$는 self-attention의 Key, Value, Query 행렬과의 연결을 암시합니다.

#### 2.2.3 경사 하강법의 일반적 형태

$$G_t = \nabla \ell(W_{t'}; x_t), \quad W_t = W_{t-1} - \eta G_t$$

세 가지 변형:
- **Online GD**: $G_t = \nabla \ell(W_{t-1}; x_t)$ — 순차적, 병렬화 불가
- **Batch GD**: $G_t = \nabla \ell(W_0; x_t)$ — 병렬화 가능하나 탐색 공간 제한
- **Mini-batch GD** (제안): $G_t = \nabla \ell(W_{t'}; x_t)$, $t' = t - \text{mod}(t, b)$ — 속도와 품질 간 균형

실험에서 TTT 배치 크기 $b = 16$을 사용합니다.

#### 2.2.4 이중 형식 (Dual Form)

TTT-Linear의 단순화된 경우, $X = [x_1, \ldots, x_b]$로 정의하면:

**미니배치 끝의 가중치 계산**:

$$W_b = W_0 + 2\eta (W_0 X - X) X^T$$

**출력 토큰 계산**:

$$\Delta = (W_0 X - X) \cdot \text{mask}(X^T X)$$

$$Z = W_0 X - 2\eta \Delta$$

여기서 $\text{mask}$는 상삼각 마스크(zeros 기반, attention mask와 유사)입니다. 이를 통해 개별 $G_t$를 명시적으로 계산하지 않고도 matmul 연산만으로 효율적 계산이 가능합니다.

- **Primal form 복잡도**: $O(b \times d^2)$ (미니배치 내)
- **Dual form 복잡도**: $O(b \times d^2) + O(b^2 \times d)$ — 이론적으로는 더 복잡하나 하드웨어 활용이 5배 이상 빠름

#### 2.2.5 이론적 동등성

**Theorem 1** (Linear Attention과의 동등성):

> TTT 레이어에서 $f(x) = Wx$ (선형 모델), 배치 GD ($\eta = 1/2$), $W_0 = 0$을 사용하면, 출력 시퀀스는 **Linear Attention**과 동일합니다.

증명의 핵심:

$$z_t = \sum_{s=1}^{t} (\theta_V x_s)(\theta_K x_s)^T (\theta_Q x_t)$$

이것은 정확히 Linear Attention의 정의입니다.

**Theorem 2** (Self-Attention과의 동등성):

> TTT 레이어에서 **Nadaraya-Watson 추정기**(비모수적 학습기)를 사용하면, 출력은 Self-Attention과 동일합니다.

Nadaraya-Watson 추정기:

$$f(x; x_1, \ldots, x_t) = \frac{\sum_{s=1}^{t} \kappa(x_s, x) y_s}{\sum_{s=1}^{t} \kappa(x_s, x)}$$

커널 $\kappa(x_s, x) = \exp((\theta_K x_s)^T (\theta_Q x))$일 때 Self-Attention과 동등합니다.

### 2.3 모델 구조

#### TTT-Linear
- 은닉 상태: 선형 모델 $f_{\text{lin}}(x) = Wx$ (정방 행렬)
- 안정성을 위해 Layer Normalization + 잔차 연결 추가: $f(x) = x + \text{LN}(f_{\text{res}}(x))$

#### TTT-MLP
- 은닉 상태: 2층 MLP (hidden dim = 4× input dim, GELU 활성화)
- 동일하게 LN + 잔차 연결 포함

#### 공통 설계 요소
| 구성 요소 | 설명 |
|-----------|------|
| **학습 가능한 $W_0$** | 초기 가중치를 외부 루프에서 학습 (모든 시퀀스 공유, 훈련 안정성 향상) |
| **학습 가능한 $\eta$** | $\eta(x) = \eta_{\text{base}} \cdot \sigma(\theta_{\text{lr}} \cdot x)$ — 입력 의존적 학습률 |
| **백본 아키텍처** | Mamba 백본 사용 (시간적 합성곱 포함) |
| **미니배치 크기** | $b = 16$ 고정 |

### 2.4 성능 향상

#### Short context (Pile, 2k)
- TTT-Linear ≈ Mamba ≈ Transformer (비슷한 성능)
- TTT-MLP는 FLOPs 기준 약간 불리

#### Short-to-Medium context (Pile, 8k)
- TTT-Linear, TTT-MLP 모두 **Mamba보다 유의미하게 우수**
- Transformer는 perplexity는 좋으나 FLOPs 대비 경쟁력 부족

#### Long context (Books3, 32k)
- **컨텍스트가 길어질수록 TTT의 Mamba 대비 우위가 확대**
- TTT-MLP (Transformer 백본)도 Mamba보다 약간 우수
- Transformer (finetune)은 2.8배의 추론 FLOPs를 사용하는 유리한 조건에서도 TTT에 비해 큰 격차 없음

#### Wall-clock Time
- TTT-Linear: Transformer 대비 이미 10% 빠른 학습 (v5e-256 TPU, 2k 컨텍스트)
- 추론 시 forward(prefill): Transformer는 컨텍스트 길이에 비례하여 증가, TTT/Mamba는 대략 일정
- TTT-MLP: FLOPs 효율적이나 메모리 I/O 비용이 높음

### 2.5 한계

> [!WARNING]
> 저자들이 명시적으로 인정한 한계점들입니다.

1. **TTT-MLP의 메모리 I/O 문제**: MLP 구조의 추가적 복잡성이 FLOPs 대비 wall-clock time을 크게 증가시킴
2. **시스템 최적화 부족**: 현 구현은 예비적(preliminary) 수준이며, 대규모 최적화 여지가 있음
3. **제한된 실험 규모**: 학술적 자원 한계로 125M~1.3B 파라미터, 최대 32k 컨텍스트까지만 실험
4. **하이브리드 아키텍처 미탐색**: Self-attention + TTT 혼합 구조를 사용하면 더 나을 수 있으나 학술적 평가의 명확성을 위해 배제
5. **선형 모델 은닉 상태의 한계**: TTT-Linear는 DeltaNet과 유사한 구조로, 이미 기존 연구에서 다루어진 영역

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 TTT의 일반화 메커니즘

TTT의 일반화 성능 향상 가능성은 다음과 같은 핵심 메커니즘에 기반합니다:

#### (1) 테스트 시 적응 (Test-Time Adaptation)

기존 모델이 고정된 파라미터로 추론하는 것과 달리, TTT는 **각 테스트 시퀀스에 대해 은닉 상태(모델)를 자기지도 학습으로 동적 적응**시킵니다:

$$W_t = W_{t-1} - \eta(x_t) \nabla \ell(W_{t'}; x_t)$$

이는 각 시퀀스가 **자체적인 학습 문제를 정의**한다는 TTT의 핵심 철학에서 비롯됩니다. 즉, 각 테스트 인스턴스 $x$에 대해 일반적 예측기 $f(x)$가 아닌 **인스턴스-특화 예측기** $f_x(x)$를 사용합니다.

#### (2) 압축 휴리스틱으로서의 자기지도 학습

자기지도 학습은 대규모 훈련 데이터를 모델 가중치로 압축하면서 데이터의 **구조와 관계를 발견**하는 능력이 있습니다. TTT는 이 능력을 활용하여:

- 문맥 내 토큰들의 의미적 연결을 포착
- 큰 경사(gradient)를 생성하는 입력을 기억 (중요한 정보 선택적 보존)
- 재구성 손실을 통해 차원 간 상관관계 학습

#### (3) 학습된 자기지도 과제 (Learned Self-Supervised Task)

기존 TTT 연구가 인간의 사전지식(prior)으로 과제를 수동 설계한 반면, 이 논문은 **자기지도 과제 자체를 외부 루프에서 학습**합니다:

```math
\theta_K^*, \theta_V^*, \theta_Q^* = \arg\min_{\theta_K, \theta_V, \theta_Q} \mathcal{L}_{\text{outer}}(\theta_K, \theta_V, \theta_Q, \theta_{\text{rest}})
```

이는 다중 뷰 재구성 과제의 패밀리에서 최적의 과제를 선택하는 것으로, **내부 루프의 특징 학습이 외부 루프의 최종 목표(next-token prediction)에 직접 최적화**됩니다.

### 3.2 장문맥에서의 일반화 확장

논문에서 가장 두드러진 일반화 관련 관찰은:

> **Transformer처럼, TTT-Linear과 TTT-MLP는 더 많은 토큰을 조건으로 할수록 perplexity를 계속 감소시킬 수 있는 반면, Mamba는 16k 컨텍스트 이후 그렇지 못합니다.**

이는 TTT의 은닉 상태가 장문맥에서도 **새로운 정보를 효과적으로 통합**할 수 있음을 의미합니다.

특히 TTT-MLP는:
- 짧은 컨텍스트에서는 용량 과잉으로 FLOPs 매칭 시 불리
- **긴 컨텍스트에서는 더 표현력 있는 은닉 상태의 큰 용량이 효과적으로 활용**

### 3.3 분포 이동(Distribution Shift) 대응

TTT의 원형 연구 (Sun et al., 2020)는 **분포 이동 하에서의 일반화 향상**을 목표로 하였습니다. 본 논문에서는 이를 시퀀스 모델링으로 확장하여:

- 각 시퀀스 내에서 발생하는 **국소적 분포 변화에 실시간 적응**
- 비디오 프레임 스트림에서의 autoregressive TTT (Wang et al., 2023)와 유사한 원리

### 3.4 향후 일반화 성능 향상 방향

| 방향 | 설명 | 기대 효과 |
|------|------|-----------|
| 더 표현력 있는 $f$ | CNN, 더 깊은 MLP 등 | 수백만~수십억 토큰 컨텍스트 처리 |
| 학습된 옵티마이저 | Adam 등 상태 기반 옵티마이저 | 더 정교한 내부 루프 학습 |
| 다중 수준 학습 | $f$ 자체가 attention → 중첩된 내부 루프 | 더 복잡한 계층적 표현 학습 |
| 외부 루프 파라미터화 | 더 일반적인 자기지도 과제 패밀리 | 최적 특징 학습 과제 발견 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 분야에 미치는 영향

#### (1) 시퀀스 모델링의 새로운 패러다임

TTT는 시퀀스 모델링 레이어를 **학습기(learner)**의 관점에서 통합합니다:
- **모수적(parametric) 학습기**: TTT-Linear (Linear Attention과 동등), TTT-MLP 등
- **비모수적(nonparametric) 학습기**: Nadaraya-Watson 추정기 (Self-Attention과 동등)

이를 통해 **Self-Attention과 RNN이 동일한 프레임워크 내의 서로 다른 인스턴스화**임을 보입니다.

#### (2) RNN 연구의 재활성화

- Linear Attention, DeltaNet, Mamba, RWKV 등 현대 RNN의 한계점(장문맥 정체)을 극복할 수 있는 이론적 기반 제공
- **은닉 상태의 표현력**이라는 핵심 차원에서의 진전 가능성 제시

#### (3) 메타러닝과의 새로운 연결

기존 메타러닝이 "데이터셋 수준"의 외부 루프를 필요로 한 반면, TTT의 외부 루프는 **일반적인 지도 학습과 동일한 수준**에서 작동합니다. 이는 메타러닝의 스케일링 문제를 우회합니다.

#### (4) 비디오 및 멀티모달 AI

수백만~수십억 토큰의 컨텍스트가 필요한 비디오 이해, 로봇 제어 등의 분야에서 TTT의 잠재적 가치가 매우 큽니다.

### 4.2 향후 연구 시 고려사항

> [!TIP]
> 이 논문을 기반으로 후속 연구를 진행할 때 고려해야 할 핵심 사항들입니다.

#### 실용성 관련
1. **Wall-clock Time 최적화**: TTT-MLP의 실질적 속도 개선이 핵심 과제 (custom CUDA kernel, FlashAttention 스타일 최적화)
2. **메모리 효율성**: gradient checkpointing through time을 더 정교하게 적용
3. **하이브리드 아키텍처**: Self-Attention + TTT의 혼합이 실용적 최적해일 가능성

#### 이론적 관련
4. **은닉 상태 모델의 최적 선택**: 어떤 $f$가 어떤 과제에 최적인지 체계적 탐색 필요
5. **자기지도 과제 설계**: 다중 뷰 재구성을 넘어서는 더 일반적인 과제 패밀리 탐색
6. **수렴 보장**: 내부 루프의 미니배치 GD에 대한 이론적 수렴 분석

#### 스케일링 관련
7. **더 긴 컨텍스트**: 수백만 토큰 이상에서의 검증
8. **더 큰 모델**: 1.3B 이상 (10B+)에서의 스케일링 법칙
9. **파이프라인 병렬화**: 시간 축을 따른 병렬화로 장문맥 처리의 다중 디바이스 분산

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 대상 개요

| 연구 | 연도 | 핵심 아이디어 | 은닉 상태 유형 |
|------|------|---------------|---------------|
| **Linear Attention** (Katharopoulos et al.) | 2020 | Softmax 제거한 Attention | 행렬 $\sum v_s k_s^T$ |
| **Mamba** (Gu & Dao) | 2023 | 선택적 상태 공간 모델 | 벡터 (SSM state) |
| **RWKV** (Peng et al.) | 2023–2024 | RNN-Transformer 결합 | 행렬 (matrix-valued) |
| **xLSTM** (Beck et al.) | 2024 | 지수 게이팅 + 행렬 메모리 | 행렬 (mLSTM) |
| **DeltaNet** (Schlag et al.) | 2021–2024 | Delta rule 기반 업데이트 | 행렬 |
| **Mamba-2** (Dao & Gu) | 2024 | 구조적 상태공간 이중성 | 행렬 (SSD) |
| **Gated DeltaNet** (Yang et al.) | 2023–2024 | 게이팅 + Delta rule | 행렬 |
| **Griffin** (De et al.) | 2024 | 게이트 선형 순환 + 로컬 어텐션 | 벡터/행렬 하이브리드 |
| **TTT (본 논문)** | 2024 | 은닉 상태 = ML 모델 | 모델 가중치 (임의 NN) |

### 5.2 핵심 차별점 비교

#### 은닉 상태의 표현력

```
표현력 (낮음 → 높음):
벡터 (Mamba) < 행렬 (Linear Attn, DeltaNet) < 임의 NN (TTT-MLP) ≤ 리스트 (Self-Attn)
```

- **Mamba/SSM**: 고정 크기 벡터 상태 → 16k 이후 정체
- **Linear Attention/DeltaNet**: 행렬 상태 ($d \times d$) → TTT-Linear의 특수 경우
- **TTT-MLP**: 2층 MLP의 전체 가중치가 상태 → 장문맥에서도 perplexity 지속 감소
- **Self-Attention**: KV 캐시 (리스트, $t$에 비례 증가) → 가장 표현력 있으나 $O(T^2)$

#### 계산 복잡도

| 모델 | 훈련 복잡도 | 추론 (토큰당) | 메모리 |
|------|------------|--------------|--------|
| Transformer | $O(T^2 d)$ | $O(Td)$ | $O(Td)$ |
| Mamba | $O(Td^2)$ | $O(d^2)$ | $O(d)$ |
| TTT-Linear | $O(Td^2)$ | $O(d^2)$ | $O(d^2)$ |
| TTT-MLP | $O(Td^2)$ | $O(d^2)$ | $O(d^2)$ |

#### DeltaNet과의 관계

> DeltaNet = TTT-Linear + 미니배치 크기 1 + LN/잔차 연결 없음

TTT는 DeltaNet을 **일반화**한 것으로, 미니배치 크기를 늘리고 LN/잔차 연결을 추가하는 것이 핵심적 성능 향상 요소입니다.

### 5.3 Mamba-2 vs TTT

**Mamba-2** (Dao & Gu, 2024):
- **Structured State Space Duality (SSD)** 개념 도입
- SSM과 attention 메커니즘 사이의 이중성 활용
- 하드웨어 효율적 matmul 활용에 최적화
- 선형 시간 스케일링 유지

**TTT와의 비교**:
- Mamba-2는 **하드웨어 효율성**에 중점, TTT는 **표현력**에 중점
- Mamba-2의 행렬 상태는 여전히 고정 크기로, 장문맥 정체 문제 잠재
- TTT는 은닉 상태의 **동적 진화**를 통해 이 한계를 극복 가능

### 5.4 xLSTM vs TTT

**xLSTM** (Beck et al., 2024):
- **지수 게이팅** (exponential gating)으로 안정성 개선
- **mLSTM**: 행렬 메모리로 병렬화 가능
- **sLSTM**: 스칼라 메모리, 고전적 순환 구조 유지

**TTT와의 비교**:
- xLSTM의 mLSTM은 행렬 상태를 사용하여 DeltaNet/Linear Attention과 유사한 범주
- TTT는 행렬을 넘어 **임의의 신경망**을 은닉 상태로 사용할 수 있는 더 일반적인 프레임워크
- xLSTM은 게이팅 메커니즘에 의존하는 반면, TTT는 **자기지도 학습**에 의존

### 5.5 현대적 트렌드: 하이브리드 아키텍처

2024–2025년의 주요 트렌드는 **하이브리드 아키텍처**입니다:
- **Jamba** (AI21): Mamba + Transformer 혼합
- **RWKV-X**: RWKV + 희소 어텐션
- **Griffin**: 게이트 선형 순환 + 로컬 어텐션

TTT 역시 하이브리드화의 유력한 후보입니다. 논문에서는 학술적 명확성을 위해 하이브리드를 배제했지만, **실용적 배포에서는 TTT + Self-Attention 혼합이 최적일 가능성**이 높습니다.

### 5.6 요약 비교표

| 특성 | TTT | Mamba/Mamba-2 | xLSTM | RWKV | Gated DeltaNet |
|------|-----|---------------|-------|------|----------------|
| 은닉 상태 유형 | 임의 NN | 벡터/행렬 | 벡터/행렬 | 행렬 | 행렬 |
| 장문맥 perplexity 감소 | ✅ 지속 | ❌ 정체 | △ | △ | △ |
| 추론 복잡도 | $O(d^2)$ | $O(d^2)$ | $O(d^2)$ | $O(d^2)$ | $O(d^2)$ |
| 하드웨어 효율성 | △ (개선 필요) | ✅ | ✅ | ✅ | ✅ |
| 이론적 프레임워크 | ✅ (일반적) | △ | △ | △ | △ |
| 테스트 시 적응 | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 참고자료

### 논문 직접 인용

1. Yu Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States," arXiv:2407.04620v4 (2024), ICML 2025
2. Albert Gu & Tri Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752 (2023)
3. Tri Dao & Albert Gu, "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality," arXiv:2405.21060 (2024) [Mamba-2]
4. Angelos Katharopoulos et al., "Transformers are RNNs: Fast autoregressive transformers with linear attention," ICML 2020
5. Imanol Schlag et al., "Linear transformers are secretly fast weight programmers," ICML 2021 [DeltaNet]
6. Songlin Yang et al., "Gated linear attention transformers with hardware-efficient training," arXiv:2312.06635 (2023) [Gated DeltaNet]
7. Songlin Yang et al., "Parallelizing linear transformers with the delta rule over sequence length," arXiv:2406.06484 (2024)
8. Maximilian Beck et al., "xLSTM: Extended long short-term memory," arXiv:2405.04517 (2024)
9. Bo Peng et al., "RWKV: Reinventing RNNs for the transformer era," arXiv:2305.13048 (2023)
10. Bo Peng et al., "Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence," arXiv:2404.05892 (2024)
11. Soham De et al., "Griffin: Mixing gated linear recurrences with local attention for efficient language models," arXiv:2402.19427 (2024)
12. Jared Kaplan et al., "Scaling laws for neural language models," arXiv:2001.08361 (2020)
13. Yu Sun et al., "Test-time training with self-supervision for generalization under distribution shifts," ICML 2020
14. Chelsea Finn et al., "Model-agnostic meta-learning for fast adaptation of deep networks," ICML 2017
15. Renhao Wang et al., "Test-time training on video streams," arXiv:2307.05014 (2023)

### 웹 검색 참조

16. arXiv HTML 전문: https://arxiv.org/html/2407.04620v4
17. arXiv 초록 페이지: https://arxiv.org/abs/2407.04620

> [!NOTE]
> 본 분석은 논문 원문(arXiv:2407.04620v4)의 HTML 버전을 직접 읽고, 관련 최신 연구에 대한 웹 검색 결과를 종합하여 작성되었습니다. 수식은 논문에서 직접 인용하였으며, 확인되지 않은 내용은 포함하지 않았습니다.
