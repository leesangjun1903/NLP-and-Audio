# Mamba-3: Improved Sequence Modeling using State Space Principles
---

## 1. 핵심 주장과 주요 기여 요약

Mamba-3는 **추론 효율성(inference efficiency)**을 최우선으로 설계된 차세대 State Space Model(SSM)이다. 저자들은 기존 Mamba-2 및 Gated DeltaNet(GDN) 등 서브-쿼드래틱(sub-quadratic) 모델들이 (a) 모델 품질과 효율성 사이의 트레이드오프, (b) state tracking 능력 부재, (c) 이론적 선형 추론이 하드웨어 비효율적이라는 문제를 안고 있음을 지적하고, SSM 관점에서 세 가지 핵심 방법론적 개선을 제안한다:

1. **Exponential-Trapezoidal Discretization** — SSM 이산화(discretization)의 2차 정확도 향상을 통한 더 표현력 있는 재귀식
2. **Complex-valued State Transitions** — 복소수 상태 전이를 통해 state tracking 능력 확보 (data-dependent RoPE로 구현)
3. **Multi-Input, Multi-Output (MIMO)** — 디코딩 시 하드웨어 활용도를 높이면서 모델 성능 향상

1.5B 규모에서 Mamba-3 SISO는 차선 모델(GDN) 대비 평균 downstream accuracy **+0.6pp**, MIMO 변형은 추가로 **+1.2pp** (총 +1.8pp) 향상을 달성하며, 절반의 state size로 Mamba-2와 동등한 perplexity를 기록한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

| 문제 | 설명 |
|------|------|
| **Transformer의 추론 비효율** | Self-attention의 $O(T^2)$ 연산 및 KV 캐시의 $O(T)$ 메모리 증가 |
| **기존 선형 모델의 품질 저하** | Mamba-2 등은 효율을 위해 표현력을 희생, parity 등 state tracking 실패 |
| **하드웨어 비효율** | SSM 디코딩의 arithmetic intensity가 ~2.5 ops/byte로 GPU 텐서 코어(295 ops/byte) 대비 극히 낮음 |
| **이산화 이론적 정당성 부재** | Mamba-1/2의 이산화가 이론적 근거 없는 heuristic이었음 |

### 2.2 제안 방법 및 수식

#### (A) Exponential-Trapezoidal Discretization

연속 시간 SSM은 다음과 같이 정의된다:

$$\dot{\mathbf{h}}(t) = A(t)\,\mathbf{h}(t) + \mathbf{B}(t)\,x(t), \qquad y(t) = \mathbf{C}(t)^\top \mathbf{h}(t)$$

**Mamba-2 (Exponential-Euler)**: 기존 Mamba-1/2의 이산화는 Euler 근사를 사용하여 $O(\Delta_t^2)$ 오차를 가진다:

$$\mathbf{h}_t = e^{\Delta_t A_t}\,\mathbf{h}_{t-1} + \Delta_t\,\mathbf{B}_t\,x_t $$

**Mamba-3 (Exponential-Trapezoidal)**: 저자들은 일반화된 사다리꼴 규칙을 적용하여 $O(\Delta_t^3)$ 오차의 2차 정확도를 달성한다:

$$\mathbf{h}_t = e^{\Delta_t A_t}\,\mathbf{h}_{t-1} + (1 - \lambda_t)\Delta_t\,e^{\Delta_t A_t}\,\mathbf{B}_{t-1}\,x_{t-1} + \lambda_t\,\Delta_t\,\mathbf{B}_t\,x_t $$

$$=: \alpha_t\,\mathbf{h}_{t-1} + \beta_t\,\mathbf{B}_{t-1}\,x_{t-1} + \gamma_t\,\mathbf{B}_t\,x_t $$

여기서 $\alpha_t := e^{\Delta_t A_t}$, $\beta_t := (1-\lambda_t)\Delta_t\,e^{\Delta_t A_t}$, $\gamma_t := \lambda_t\,\Delta_t$이며, $\lambda_t \in [0,1]$은 data-dependent 스칼라이다.

- $\lambda_t = 1$이면 Mamba-2의 Euler 이산화로 환원
- $\lambda_t = \frac{1}{2}$이면 고전적 사다리꼴 규칙으로 환원

이 3-항 재귀식은 state-input $\mathbf{B}_t x_t$에 대한 **폭 2의 data-dependent 합성곱(convolution)**으로 해석되며, $B$, $C$ bias와 결합하면 기존 모델에서 필수적이던 **외부 short causal convolution을 제거**할 수 있다.

**병렬 표현 (SSD 프레임워크)**: Mamba-3의 mask 행렬 $\mathbf{L}$은 1-semiseparable 행렬과 2-band 행렬의 곱으로 표현된다:

```math
\mathbf{L} = \underbrace{\begin{bmatrix} 1 \\ \alpha_1 & 1 \\ \alpha_2\alpha_1 & \alpha_2 & 1 \\ \vdots & & \ddots \\ \alpha_{T\cdots 1} & \cdots & & 1 \end{bmatrix}}_{\text{decay}} \underbrace{\begin{bmatrix} \gamma_0 \\ \beta_1 & \gamma_1 \\ 0 & \beta_2 & \gamma_2 \\ \vdots & & \ddots \\ 0 & \cdots & & \gamma_T \end{bmatrix}}_{\text{2-band conv}}
```

#### (B) Complex-valued State Space Model

실수 고유값만 사용하는 Mamba-2는 parity 등 "회전(rotation)" 동역학이 필요한 state tracking 작업을 해결할 수 없다. Mamba-3는 복소수 SSM을 도입한다:

$$\dot{\mathbf{h}}(t) = \text{Diag}\big(A(t) + i\boldsymbol{\theta}(t)\big)\,\mathbf{h}(t) + \big(\mathbf{B}(t) + i\hat{\mathbf{B}}(t)\big)\,x(t) $$

이를 exponential-Euler로 이산화하면 등가적인 실수 SSM으로 변환된다 (Proposition 2):

$$\mathbf{h}\_t = e^{\Delta_t A_t}\,\mathbf{R}\_t\,\mathbf{h}_{t-1} + \Delta_t\,\mathbf{B}_t\,x_t, \qquad y_t = \mathbf{C}_t^\top\,\mathbf{h}_t $$

여기서 $\mathbf{R}\_t := \text{Block} \left(\{R(\Delta\_t\boldsymbol{\theta}\_t[i])\}_{i=1}^{N/2}\right)$이고 

```math
R(\theta) := \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}
```

이다.

핵심 통찰은 **"RoPE trick"** (Proposition 3)으로, 복소수 SSM의 출력이 $\mathbf{B}$, $\mathbf{C}$에 **data-dependent rotary embedding**을 적용한 스칼라 전이 SSM과 동등하다는 것이다:

$$\mathbf{h}_t = e^{\Delta_t A_t}\,\mathbf{h}_{t-1} + \left(\prod_{i=0}^{t}\mathbf{R}_i^\top\right)\Delta_t\,\mathbf{B}_t\,x_t, \qquad y_t = \left[\left(\prod_{i=0}^{t}\mathbf{R}_i^\top\right)\mathbf{C}_t\right]^\top\mathbf{h}_t $$

**Exponential-Trapezoidal과 결합한 최종 재귀식** (Proposition 4):

$$\mathbf{h}_t = \alpha_t\,\mathbf{h}_{t-1} + \beta_t\left(\prod_{i=0}^{t-1}\mathbf{R}_i^\top\right)\mathbf{B}_{t-1}\,x_{t-1} + \gamma_t\left(\prod_{i=0}^{t}\mathbf{R}_i^\top\right)\mathbf{B}_t\,x_t$$

$$y_t = \left[\left(\prod_{i=0}^{t}\mathbf{R}_i^\top\right)\mathbf{C}_t\right]^\top\mathbf{h}_t $$

#### (C) Multi-Input, Multi-Output (MIMO)

SISO에서 $\mathbf{B}_t \in \mathbb{R}^N$, $\mathbf{x}_t \in \mathbb{R}^P$였던 것을 $\mathbf{B}_t \in \mathbb{R}^{N \times R}$, $\mathbf{x}_t \in \mathbb{R}^{P \times R}$으로 확장한다. Arithmetic intensity가 $\Theta(1)$에서 $\Theta(R)$로 증가하며 (Table 2), 메모리 I/O는 state $\mathbf{H}_t \in \mathbb{R}^{N \times P}$에 의해 지배되므로 **디코딩 wall-clock은 거의 동일**하다.

MIMO 재귀식:

$$\mathbf{h}_t^{(j)} \leftarrow \alpha_t\,\mathbf{h}_{t-1}^{(j)} + \Delta_t\,\mathbf{B}_t^{(j)}\,\mathbf{x}_t^{(j)} $$

$$\mathbf{h}_t = \sum_{j=0}^{R-1}\mathbf{h}_t^{(j)} $$

$$\mathbf{y}_t^{(i)} \leftarrow \left(\mathbf{C}_t^{(i)}\right)^\top\mathbf{h}_t $$

Chunked 알고리즘에서 chunk size를 $C_{\text{MIMO}} = \frac{1}{R}C_{\text{SISO}}$로 설정하면 전체 FLOP은 $R$배만 증가하여 $R^2$배 증가를 방지한다.

### 2.3 모델 구조

Mamba-3는 **Llama 스타일 아키텍처**(Mamba-3 블록 + SwiGLU MLP 블록 교대, pre-norm)를 따르며, Mamba-2 대비 주요 변경사항은:

- SSD 레이어를 복소수 exponential-trapezoidal SSM으로 교체
- **BC/QK Normalization**: $\mathbf{B}$, $\mathbf{C}$ 프로젝션 후 RMS 정규화 추가 (학습 안정화, post-gate RMSNorm 제거 가능)
- **B, C Biases**: 학습 가능한 head-specific, channel-wise bias 추가 (universal approximation 능력 부여)
- 외부 short causal convolution 및 활성화 함수 **제거**
- 선택적 MIMO 프로젝션 추가

### 2.4 성능 향상

**언어 모델링 (Table 3, 100B FineWeb-Edu 토큰):**

| 모델 (1.5B) | FW-Edu ppl↓ | 평균 downstream acc↑ |
|---|---|---|
| Transformer | 10.51 | 55.4 |
| Mamba-2 | 10.47 | 55.7 |
| GDN | 10.45 | 55.8 |
| **Mamba-3 SISO** | **10.35** | **56.4** |
| **Mamba-3 MIMO** | **10.24** | **57.6** |

**State tracking (Table 5b, scaled accuracy %):**

| 모델 | Parity | Arith. w/o brackets | Arith. w/ brackets |
|---|---|---|---|
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| Mamba-3 (w/o RoPE) | 2.27 | 1.49 | 0.72 |
| **Mamba-3** | **100.00** | **98.51** | **87.75** |
| GDN [-1,1] | 100.00 | 99.25 | 93.50 |

**추론 효율 (Figure 3):** Mamba-3 MIMO는 state size 64에서 Mamba-2 state size 128과 동일한 perplexity → **절반의 latency로 동일 성능**

**커널 latency (Table 6, BF16, $d_\text{state}=128$):**
- Mamba-3 SISO: 0.156ms (Mamba-2: 0.203ms, GDN: 0.257ms)
- Mamba-3 MIMO ($R=4$): 0.179ms

### 2.5 한계

1. **고정 크기 상태의 retrieval 한계**: 반구조화/비구조화 데이터에서의 정보 추출(SWDE, FDA) 성능이 Transformer 대비 낮음 (Table 4)
2. **Hybrid 모델 설계의 불확실성**: RMSNorm 유형(grouped vs default)과 배치(pre- vs post-gate)에 따른 경쟁적 트레이드오프가 존재하며, 최적 구성이 불명확
3. **MIMO의 학습 비용**: MIMO는 디코딩 속도를 유지하지만 학습 시 $R$배 FLOP 증가 (실제로는 $R=4$일 때 ~2배 slowdown)
4. **Modular Arithmetic with brackets에서의 갭**: GDN (93.50%) 대비 Mamba-3 (87.75%)로 완전한 해결에는 미달
5. **대규모 검증 부족**: 실험이 1.5B 규모까지이며, 수십~수백 B 규모에서의 검증은 이루어지지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

Mamba-3의 일반화 성능 향상은 세 가지 차원에서 분석할 수 있다:

### 3.1 이산화 정확도를 통한 일반화

Exponential-trapezoidal 이산화는 state-input 적분의 $O(\Delta_t^3)$ 오차(vs Euler의 $O(\Delta_t^2)$ )를 달성하며, 이는 연속 동역학을 더 정확히 근사한다. 이 향상은 **모든 시퀀스 길이와 도메인에 걸쳐** 보다 충실한 시스템 표현을 가능케 하며, Table 5a에서 bias+trapezoidal의 시너지 효과 (ppl 16.68 → 15.72)로 확인된다. 또한 Figure 4에서 Mamba-3는 학습 길이(2K) 이상에서도 perplexity가 안정적으로 감소하여 **긴 문맥 외삽(length extrapolation)** 능력이 뛰어남을 보인다.

### 3.2 복소수 상태 전이를 통한 표현력 확장

Grazzi et al. (2025)의 이론적 결과에 따르면, 실수 비음수 고유값으로 제한된 SSM은 TC⁰ 복잡도 클래스에 갇혀 parity와 같은 기본적인 state tracking도 불가능하다. Mamba-3의 복소수 전이 행렬은 이 제약을 돌파하여:
- **Parity**: 0.90% → 100.00% (Mamba-2 → Mamba-3)
- **Modular Arithmetic**: 47.81% → 98.51%

이는 형식 언어(formal language) 인식이라는 근본적 능력에서의 일반화를 의미하며, Yu and Erichson (2025)이 증명한 B bias의 universal approximation 속성과 결합하여 **이론적으로 더 넓은 함수 클래스를 표현**할 수 있게 된다.

### 3.3 MIMO를 통한 효율적 용량 확장

MIMO는 state size를 늘리지 않으면서(따라서 디코딩 latency 증가 없이) 모델의 FLOPs를 $R$배 증가시킨다. Figure 3의 Pareto frontier에서 **MIMO는 SISO 대비 일관된 하향 이동**을 보이며, 이는 고정된 추론 예산 하에서 더 나은 일반화를 달성함을 의미한다. 특히 state size 64의 Mamba-3 MIMO가 state size 128의 Mamba-2와 동등한 perplexity를 달성한다는 점에서, **같은 추론 비용에서 2배 더 큰 "유효 용량"**을 얻는 것과 같다.

### 3.4 Hybrid 아키텍처에서의 일반화

Table 4에서 Mamba-3를 NoPE self-attention과 5:1 비율로 교차 배치한 hybrid 모델은:
- 실세계 retrieval에서 Transformer 수준에 근접
- 합성 NIAH에서 길이 일반화 대폭 개선 (pre-gate grouped RMSNorm 사용 시 4096 토큰에서 100% → 이전 36.2%)

이는 Mamba-3가 attention 레이어의 보완재로서도 강한 일반화 능력을 가짐을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **SSM 관점의 부활**: Linear attention, TTT 등 대안적 프레임워크와 달리, 이산화·복소수·MIMO라는 고전적 SSM 개념이 현대 시퀀스 모델링에 유효함을 입증. 향후 제어 이론/신호 처리의 다른 도구들(예: 고차 이산화, 적응형 step size)을 탐색하는 연구를 촉발할 것이다.

2. **추론 효율 중심 설계 패러다임**: Arithmetic intensity를 설계 원칙의 핵심 지표로 제시하여, 향후 아키텍처 설계가 학습 효율뿐 아니라 **디코딩 시 하드웨어 활용도**를 동시에 최적화하는 방향으로 전환할 것이다.

3. **Hybrid 모델의 표준화**: Mamba-3가 Nemotron-H (NVIDIA), Kimi Linear, Hunyuan-TurboS 등과 같이 hybrid 아키텍처의 선형 레이어 후보로 자리매김. 향후 연구는 최적의 선형/attention 레이어 비율과 배치 전략을 규명해야 한다.

4. **State tracking의 실용적 활용**: Parity, modular arithmetic 해결은 이론적으로 의미 있지만, 실제 언어 태스크에서 state tracking이 어떻게 성능으로 전환되는지에 대한 추가 연구가 필요하다.

5. **MIMO의 범용 적용 가능성**: 저자들이 지적하듯 MIMO 기법은 SSM에 국한되지 않고 다른 선형 모델(DeltaNet, Linear Attention)에도 적용 가능하여, 광범위한 서브-쿼드래틱 모델의 효율을 개선하는 범용 도구가 될 수 있다.

### 4.2 향후 연구 시 고려할 점

1. **대규모 검증**: 7B, 70B+ 규모에서의 scaling 행동 및 instruction tuning/RLHF 후 성능 확인 필요
2. **긴 문맥 학습**: 2K 토큰에서만 학습 — 128K+ 문맥에서의 학습 및 검증이 중요
3. **Retrieval 한계 극복**: 고정 크기 상태의 근본적 한계를 해결하기 위한 새로운 메커니즘 (예: 외부 메모리, adaptive state size) 탐구
4. **Hybrid 모델 이론**: Cabannes et al. (2025)이 지적한 바와 같이 hybrid 모델의 동작이 직관적이지 않으며, 이론적 이해가 부족
5. **다중 모달리티**: 비전, 오디오 등에서 복소수 SSM과 MIMO의 효과 검증
6. **양자화/경량화 호환성**: MIMO의 추가 연산이 INT8/FP4 등 양자화 환경에서 어떻게 동작하는지 검증
7. **Data-dependent RoPE의 이론적 분석**: 학습된 $\theta_t$의 동역학과 주파수 특성에 대한 심층 분석

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | Mamba-3와의 관계 |
|------|------|----------|----------------|
| **S4** (Gu, Goel, Ré) | 2022 | 구조화된 SSM (NPLR 행렬), LTI 기반 장기 의존성 | Mamba-3의 선조격; LTI → LTV(selective) 전환의 출발점 |
| **Mamba-1 (S6)** (Gu & Dao) | 2024 | Selective SSM, data-dependent 파라미터, 하드웨어 인식 병렬 스캔 | 직접적 선행 연구; Mamba-3가 이론적으로 정당화하는 heuristic 이산화 사용 |
| **Mamba-2** (Dao & Gu) | 2024 | SSD 프레임워크, 스칼라 전이, matmul 기반 효율적 학습 | 직접적 선행 연구; Mamba-3가 세 가지 방법론으로 개선 |
| **Gated DeltaNet** (Yang et al.) | 2025 | Delta rule 기반 메모리 업데이트, Householder 전이 | 주요 경쟁 모델; state tracking 가능(음수 고유값)하지만 Mamba-3가 downstream에서 우세 |
| **Linear Attention** (Katharopoulos et al., 2020) | 2020 | 커널 특징 맵을 통한 선형 시간 attention | SSD의 가장 단순한 경우(인과 삼각 mask); Mamba-3는 더 표현력 있는 구조화된 mask 사용 |
| **RetNet** (Sun et al.) | 2023 | 상수 스칼라 decay + 복소수 위상(RoPE 구현) | Mamba-3의 RoPE trick과 유사하지만 LTI이며 data-independent; Mamba-3는 data-dependent RoPE로 확장 |
| **TTT/TTR** (Sun et al., 2025; Zhang et al., 2025) | 2025 | 추론 시 온라인 학습으로 상태 업데이트 | 대안적 관점; 복소수 값이 회귀 목적함수의 계수로 해석 불가하여 Mamba-3의 복소수 SSM은 이 프레임워크에서 자연스럽지 않음 |
| **S5** (Smith et al.) | 2023 | MIMO LTI SSM, parallel scan | Mamba-3의 MIMO와 역방향 동기: S5는 상태 용량 축소로 단순화, Mamba-3는 상태 용량 유지하며 표현력 증대 |
| **Kimi Linear** (Kimi Team) | 2025 | Delta rule 기반 대규모 hybrid 모델 | GDN 계열의 산업적 적용; Mamba-3의 MIMO 기법이 이러한 모델에도 적용 가능 |
| **Nemotron-H** (NVIDIA et al.) | 2025 | Mamba-2 기반 hybrid 모델의 대규모 학습 | Mamba-3가 Mamba-2를 대체할 경우 추가 성능 향상 기대 |
| **Block-Biased Mamba** (Yu & Erichson) | 2025 | B bias 추가로 universal approximation 달성 증명 | Mamba-3의 B,C bias 설계에 직접 영향; 단, Mamba-3는 B만이 아닌 B+C bias의 시너지를 발견 |
| **Grazzi et al.** | 2025 | 음수 고유값이 state tracking(parity 등)을 가능케 함을 증명 | Mamba-3의 복소수 전이가 이 이론적 요구를 충족; RoPE trick으로 효율적 구현 |
| **Comba** (Hu et al.) | 2025 | Closed-loop control로 bilinear RNN 개선 | 제어 이론 기반 개선의 또 다른 사례; Mamba-3의 SSM 관점과 상보적 |

---

## 참고자료

1. **Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T., & Gu, A.** (2026). *Mamba-3: Improved Sequence Modeling using State Space Principles*. arXiv:2603.15569 [cs.LG]. — 본 분석의 주 논문
2. **Dao, T. & Gu, A.** (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. arXiv:2405.21060
3. **Gu, A. & Dao, T.** (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752
4. **Yang, S., Kautz, J., & Hatamizadeh, A.** (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv:2412.06464
5. **Gu, A., Goel, K., & Ré, C.** (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. arXiv:2111.00396
6. **Smith, J. T. H., Warrington, A., & Linderman, S. W.** (2023). *Simplified State Space Layers for Sequence Modeling*. arXiv:2208.04933
7. **Grazzi, R., Siems, J., Zela, A., et al.** (2025). *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*. arXiv:2411.12537
8. **Su, J., Lu, Y., Pan, S., et al.** (2023). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864
9. **Yu, A. & Erichson, N. B.** (2025). *Block-Biased Mamba for Long-Range Sequence Processing*. arXiv:2505.09022
10. **Sun, Y., Dong, L., et al.** (2023). *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv:2307.08621
11. **Kimi Team et al.** (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture*. arXiv:2510.26692
12. **NVIDIA et al.** (2025). *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*. arXiv:2504.03624
13. **Katharopoulos, A., et al.** (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. arXiv:2006.16236
14. **Sun, Y., et al.** (2025). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620
15. **Merrill, W., Petty, J., & Sabharwal, A.** (2025). *The Illusion of State in State-Space Models*. arXiv:2404.08819
16. **Tenenbaum, M. & Pollard, H.** (1985). *Ordinary Differential Equations*. Dover Publications. — Variation of Constants 유도에 사용
17. **Süli, E. & Mayers, D. F.** (2003). *An Introduction to Numerical Analysis*. Cambridge University Press. — 이산화 오차 분석 참조
