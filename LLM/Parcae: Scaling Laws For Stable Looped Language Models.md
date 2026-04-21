# Parcae: Scaling Laws For Stable Looped Language Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**Parcae**는 루프(looped) 언어 모델의 훈련 불안정성 문제를 **동역학계(dynamical systems) 이론**으로 분석·해결하고, 루핑(looping)을 **예측 가능하고 독립적인 컴퓨팅 스케일링 축**으로 확립한다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| **이론적 분석** | 루프 모델을 비선형 시변 동역학계로 재구성 → 불안정성의 근본 원인 규명 |
| **안정적 아키텍처** | 스펙트럼 놈(spectral norm) 제약을 통한 Parcae 설계 |
| **훈련 스케일링 법칙** | 최적 루핑과 데이터를 동시에 증가시키는 파워 법칙 도출 |
| **테스트-타임 스케일링 법칙** | 포화 지수 감쇠 + 통합 스케일링 법칙 수립 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 루프 트랜스포머([Geiping et al., 2025] RDM 등)는 다음 문제를 가진다:

1. **잔차 폭발(Residual Explosion)**: 루프 반복 시 은닉 상태 $\|h_T\|_2$가 기하급수적으로 증가
2. **손실 스파이크(Loss Spikes)**: 가변 깊이 학습 중 불규칙적인 손실 급등
3. **하이퍼파라미터 민감성**: 학습률 $2 \times 10^{-4}$에서만 수렴, 고학습률에서 발산

이는 파라미터 수를 늘리지 않고 FLOPs를 증가시키는 루핑의 장점을 활용하지 못하게 한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 핵심 프레임워크: 비선형 시변 동역학계

루프 모델의 순전파를 다음과 같이 재구성한다:

$$h_{t+1} = \overline{A}h_t + \overline{B}e + \overline{\mathcal{R}}(h_t, e) \tag{1}$$

여기서:
- $h_t \in \mathbb{R}^{n \times d_h}$: 루프 반복 $t$에서의 은닉 상태
- $e \in \mathbb{R}^{n \times d_h}$: 프리루드(prelude) $\mathcal{P}$의 출력 (입력 임베딩)
- $\overline{A} \in \mathbb{R}^{d_h \times d_h}$: 이전 상태와 현재 상태의 균형을 제어하는 상태 전이 행렬
- $\overline{B} \in \mathbb{R}^{d_h \times d_e}$: 입력 $e$가 잔차에 미치는 영향 제어
- $\overline{\mathcal{R}}$: Attention, FFN 등 비선형 연산자

$\overline{\mathcal{R}}$을 제거하는 **선형화(linearization)**를 적용하면:

$$h_{t+1} = \overline{A}h_t + \overline{B}e \tag{2}$$

이는 **이산 선형 시불변(LTI) 시스템**으로, 안정성이 $\overline{A}$의 스펙트럴 반경(spectral radius)에 의해 결정된다:

$$\rho(\overline{A}) < 1 \implies \text{stable}, \quad \rho(\overline{A}) \geq 1 \implies \text{unstable}$$

#### 기존 방법의 불안정성

| 방법 | $\overline{A}$ | $\rho(\overline{A})$ | LTI 안정성 |
|------|----------------|----------------------|-----------|
| Addition [Yang et al., 2024] | $I$ | $= 1$ | 한계 안정 |
| Concatenation [Geiping et al., 2025] | $\mathbb{R}^{d_h \times d_h}$ | $\in \mathbb{R}$ (비제어) | 불안정 |
| **Parcae (제안)** | $\text{ZOH}(\text{Diag}(-\exp(\mathbb{R}^{d_h})))$ | $< 1$ | **안정** |

#### Parcae의 안정적 파라미터화

$\overline{A}$를 **음의 대각 행렬(negative diagonal matrix)**로 파라미터화한다:

$$A := \text{Diag}(-\exp(\log A))$$

여기서 $\text{Diag}(-\exp(\cdot))$는 음수성을 강제하고, $\log A \in \mathbb{R}^{d_h}$는 학습 가능한 벡터이다.

연속 표현을 **ZOH(Zero-Order Hold) 이산화**로 변환:

$$\overline{A} = \exp(\Delta A), \quad \overline{B} = \Delta B$$

여기서 $\Delta \in \mathbb{R}^{d_h}$는 학습 가능한 이산화 파라미터이다.

#### Parcae의 업데이트 규칙

$$e = \text{LN}(\mathcal{P}(s)), \quad h_{t+1} = \overline{A}h_t + \overline{B}e + \overline{\mathcal{R}}(h_t, e), \quad p = \mathcal{C}(\mathcal{C}h_T) \tag{3}$$

초기 상태: $h_0 \sim \mathcal{N}(0, \sigma I_{d_h \times d_h})$

세 가지 안정화 기법:
1. $A$를 음의 대각 행렬로 제약 → $\rho(\overline{A}) < 1$ 보장
2. **프리루드 정규화(Prelude Norm)**: $e \leftarrow \text{LN}(\mathcal{P}(s))$ → 후기 훈련 손실 스파이크 방지
3. **시퀀스별 깊이 샘플링(Per-Sequence Depth Sampling)**: 마이크로배치 내 각 시퀀스에 독립적으로 재귀 깊이 샘플링

#### 훈련 스케일링 법칙

파라메트릭 손실 함수:

$$\hat{\mathcal{L}}_{\text{train}}(\mu_{\text{rec}}, D) = E + X \cdot \mathbf{N}(\mu_{\text{rec}})^{-x} + Y \cdot D^{-y} \tag{7}$$

여기서 $\mathbf{N}(\mu_{\text{rec}})$은 루프를 펼쳤을 때의 유효 파라미터 수, $D$는 훈련 토큰 수이다.

최적 재귀 횟수와 토큰 수의 파워 법칙:

$$\mu_{\text{rec}}^* \propto C^{0.40}, \quad D^* \propto C^{0.78}$$

#### 테스트-타임 스케일링 법칙

테스트-타임 손실은 포화 지수 감쇠로 기술된다:

$$\mathcal{L}(T) = \mathcal{L}_\infty + Z e^{-z \cdot T}$$

#### 통합 스케일링 법칙

$$\hat{\mathcal{L}}_{\text{unified}}(T \mid \mu_{\text{rec}}, D) = \underbrace{E + X \cdot \mathbf{N}(\mu_{\text{rec}})^{-x} + Y \cdot D^{-y}}_{\text{Training Law Floor } \hat{\mathcal{L}}_{\text{train}}(\mu_{\text{rec}}, D)} + \underbrace{Z \cdot \exp\!\left(-z \cdot T \cdot \mu_{\text{rec}}^{-1}\right)}_{\text{Test-Time Decay}} \tag{4}$$

---

### 2.3 모델 구조

```
Input s
    ↓
[Prelude P] → e = LN(P(s))    ← 프리루드 정규화
    ↓
    ┌─────────────────────────────┐
    │   [Recurrent Unit R]        │ ← T번 반복
    │   h_{t+1} = Ah_t + Be      │
    │           + R(h_t, e)       │
    │                             │
    │   A: 음의 대각 행렬          │
    │   (ZOH 이산화)               │
    │   B: Euler 이산화            │
    └─────────────────────────────┘
    ↓
[Coda C] → p = C(Ch_T)
```

- **Middle-Looped 구조**: Prelude $\mathcal{P}$ (1/3 레이어) + Recurrent $\mathcal{R}$ (1/3, 루프) + Coda $\mathcal{C}$ (1/3)
- **내부 블록**: Causal Self-Attention (QK-Norm 포함) + $\text{ReLU}^2$ MLP + RoPE 위치 임베딩
- **파라미터 오버헤드**: Transformer 대비 0.35~0.83% 추가 파라미터만 필요

---

### 2.4 성능 향상

#### vs. RDM 비교

| 모델 크기 | 지표 | RDM | Parcae | 향상 |
|-----------|------|-----|--------|------|
| 100M | Val. PPL | 14.23 | 13.59 | **↓4.5%** |
| 100M | WikiText PPL | 63.27 | 60.33 | **↓4.6%** |
| 350M | Val. PPL | 10.76 | 10.09 | **↓6.2%** |

#### vs. Transformer 비교 (파라미터 매칭)

| 크기 | 모델 | Val. PPL↓ | Core↑ | Core-Ext↑ |
|------|------|----------|-------|-----------|
| 140M | Transformer | 21.48 | 13.00 | 8.80 |
| 140M | Parcae (T=8) | **19.06** | **14.04** | **9.67** |
| 1.3B | Transformer | 11.95 | 25.45 | 15.90 |
| 1.3B | Parcae (T=8) | **11.42** | **28.44** | **17.08** |

- **770M Parcae ≈ 1.3B Transformer** (Core 기준): 절반의 파라미터로 동등한 품질
- 파라미터 효율성: Core에서 23.3~87.5%, Core-Extended에서 29.9~58.2% 향상

#### 하이퍼파라미터 강인성

| 학습률 | 기본 RDM | Res. Norm RDM | Parcae |
|--------|----------|---------------|--------|
| 2e-4 | ✓ | ✓ | ✓ |
| 4e-4 | ✗ | ✓ | ✓ |
| 6e-4~1e-3 | ✗ | ✗ | ✓ |

---

### 2.5 한계

1. **관찰 규모 제한**: 140M, 370M, 770M, 1.3B 파라미터까지만 검증 → 더 큰 모델 (10B+)에서의 스케일링 법칙 일반화 미확인
2. **테스트-타임 포화**: $T > \mu_{\text{rec}}$ 이상 루핑 시 성능 향상이 포화됨 (훈련 깊이가 상한선 결정)
3. **추론 비용**: $\mu_{\text{rec}}$ 증가 시 동등 품질 달성에 필요한 추론 스텝도 증가
4. **대각 행렬 제약**: $A$를 대각으로 제한하여 full-rank 파라미터화의 잠재력 미탐색
5. **루프 단위 구성**: 루프 블록의 배치, 크기, 다양한 아키텍처 조합에 대한 심층 탐구 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 안정적 훈련이 일반화에 미치는 영향

Parcae의 핵심 기여인 $\rho(\overline{A}) < 1$ 제약은 **훈련 안정성을 통한 일반화 향상**을 가능하게 한다. 기존 RDM이 높은 학습률에서 발산하는 반면, Parcae는 다양한 학습률에서 안정적으로 수렴하므로 더 넓은 하이퍼파라미터 공간에서 최적화가 가능하다.

### 3.2 가변 깊이 훈련과 일반화

**시퀀스별 깊이 샘플링**은 일반화 성능에 직접적인 영향을 준다:

$$\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D},\, T \sim \Lambda}\left[\ell\left(f_\theta(x; T), y\right)\right]$$

기존 [Geiping et al., 2025]의 분포 불일치 문제:
- 실제 분포: $\tau \sim \mathcal{N}(\log(\mu_{\text{rec}} - \mu_{\text{bwd}}) - \frac{1}{2}\sigma^2, \sigma)$, $n \sim \mathcal{P}(e^\tau) + 1$
- 이는 목표 분포 $\Lambda(\mu_{\text{rec}})$가 $\mu_{\text{bwd}}$에 의해 **절단·압축**되는 문제 발생

Parcae는 $T \sim \Lambda(\mu_{\text{rec}})$로 직접 샘플링하여 이 불일치를 해소, **훈련 중 다양한 깊이 노출**을 보장하여 테스트-타임 일반화를 크게 향상시킨다 (Table 8: 100M 모델 T=1에서 Per-Batch PPL 300.32 → Per-Sequence 70.47).

### 3.3 역동적 시스템 관점에서의 고정점(Fixed-Point) 일반화

$\rho(\overline{A}) < 1$은 선형화된 시스템이 **지수적으로 고정점(fixed-point)으로 수렴**함을 보장한다. 이는:

```math
\|h_t - h^*\| \leq C \cdot \rho(\overline{A})^t \|h_0 - h^*\|
```

반복이 증가할수록 은닉 상태가 안정적인 표현으로 수렴함을 의미하며, 이는 더 일관된 예측(일반화)으로 이어진다.

### 3.4 테스트-타임 스케일링을 통한 일반화

추론 시 루핑 횟수 $T$를 증가시켜 추가 컴퓨팅 없이 성능 향상이 가능하다:

$$\mathcal{L}(T) = \mathcal{L}_\infty + Ze^{-zT}$$

- 훈련 시 사용하지 않은 더 높은 $T$에서도 성능이 **예측 가능하게** 향상
- 1.3B Parcae가 1~15 루핑 범위에서 일관된 포화 패턴을 보이며, 모든 모델 크기에서 일관적

### 3.5 제로샷 및 다운스트림 벤치마크 일반화

Parcae는 훈련 데이터(FineWeb-Edu)와 다른 분포인 Lambada, HellaSwag, ARC, PIQA, BoolQ, SciQ 등에서도 일관되게 Transformer를 능가하며, 이는 단순 훈련 손실 최소화가 아닌 **표현의 질적 향상**을 시사한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 컴퓨팅 효율적 AI의 새로운 패러다임

Parcae는 **파라미터 수 고정 + 루핑 증가**라는 새로운 스케일링 축을 제시한다. 기존 Chinchilla [Hoffmann et al., 2022]가 파라미터-토큰의 최적 비율을 제시했다면, Parcae는 다음의 3차원 스케일링:

$$\text{FLOPs} = f(\underbrace{N}_{\text{파라미터}}, \underbrace{D}_{\text{데이터}}, \underbrace{\mu_{\text{rec}}}_{\text{루핑}})$$

을 확립하여 엣지 디바이스 및 추론 집약적 환경에서의 AI 배포 방향을 제시한다.

#### (2) 제어 이론과 딥러닝의 융합

LTI 시스템을 통한 불안정성 분석 방법론은 다른 반복적 아키텍처(RNN, State Space Model, DEQ 등)에도 적용 가능하며, **제어 이론 기반 신경망 설계**라는 연구 방향을 강화한다.

#### (3) 테스트-타임 컴퓨팅 확장

[DeepSeek-R1, OpenAI o1 등]이 추론 시 컴퓨팅을 확장하는 흐름과 Parcae의 테스트-타임 스케일링 법칙이 연결되어, **예측 가능한 테스트-타임 컴퓨팅 할당**에 대한 연구를 촉진할 것이다.

#### (4) 잠재적 추론(Latent Reasoning) 연구

고정점 수렴 특성이 보장되는 Parcae는 수학적 추론, 알고리즘 문제 등 **반복적 사고가 필요한 작업**에서의 latent reasoning 연구에 이론적 기반을 제공한다.

---

### 4.2 앞으로의 연구 시 고려할 점

#### (1) 대규모 스케일링 검증

현재 실험은 140M~1.3B로 제한되어 있다. 10B+ 규모에서:
- 스케일링 법칙 지수($\gamma_\mu \approx 0.40$, $\gamma_D \approx 0.78$)의 일관성 검증 필요
- 초대형 모델에서의 최적 $\mu_{\text{rec}}/\mu_{\text{bwd}}$ 비율 탐색

#### (2) Full-Rank $A$ 파라미터화 탐구

현재 $A$를 대각 행렬로 제약하여 계산 효율성을 확보했으나, **full-rank 파라미터화**가 더 풍부한 상태 전이를 학습할 수 있을지 탐구할 필요가 있다:

$$A \in \mathbb{R}^{d_h \times d_h}, \quad \text{s.t. } \rho(A) < 1$$

단, 이 경우 계산 비용( $O(d_h^2)$ )이 증가하므로 비용-성능 트레이드오프 분석이 필요하다.

#### (3) 파라미터-데이터-루핑의 3차원 최적화

현재 연구는 파라미터를 고정하고 루핑과 데이터만 최적화했다. 세 요소를 동시에 최적화하는 통합 스케일링 법칙:

$$\hat{\mathcal{L}}(N, D, \mu_{\text{rec}}) = E + A \cdot N^{-a} + B \cdot D^{-b} + C \cdot \mathbf{N}(\mu_{\text{rec}})^{-c}$$

형태의 도출이 필요하며, 이는 더 복잡한 최적화 지형(optimization landscape)을 수반한다.

#### (4) 추론 효율화

$\mu_{\text{rec}}$가 증가할수록 동등 품질 달성에 필요한 테스트-타임 스텝도 증가한다. 이를 해결하기 위해:
- **조기 종료(Early Stopping)**: KL-divergence 기반 수렴 판단
- **적응적 컴퓨팅(Adaptive Compute)**: 토큰별 필요 루핑 횟수 차별화
- **병렬 루프 실행**: [Wu et al., 2025, Parallel Loop Transformer] 방향 탐구

#### (5) 다양한 아키텍처와의 결합

Parcae의 안정화 기법(음의 대각 $A$, ZOH 이산화)은 Mamba, RWKV 등 **비-Transformer 아키텍처**와의 결합 가능성이 있으며, 이는 SSM 계열 모델의 훈련 안정성 개선에 기여할 수 있다.

#### (6) 멀티모달 및 도메인 특화 적용

루핑이 언어 이외에 [Jacobs et al., 2026]의 Vision Transformer, [Alabdulmohsin & Zhai, 2025]의 멀티모달 시스템에도 적용 가능함이 시사되었으나, 각 도메인에서의 최적 설계는 별도 탐구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | Parcae와의 차별점 |
|------|------|-----------|-----------------|
| **Universal Transformers** [Dehghani et al.] | 2019 | 레이어 전체를 루프, 명시적 halting | 안정성 미보장, 스케일링 법칙 미제시 |
| **Looped Transformers** [Yang et al.] | 2024 | Addition 주입($A=I$), 알고리즘 학습 | $\rho(A)=1$ → 한계 안정, 소규모 태스크 |
| **RDM** [Geiping et al.] | 2025 | Concatenation 주입, Post+Pre Norm | $\rho(A)$ 비제어 → 불안정, 스케일링 법칙 미제시 |
| **Retrofitted Recurrence** [McLeish et al.] | 2025 | 사전학습 모델에 루프 사후 적용 | 훈련 불안정, 분포 불일치 문제 |
| **Loopformer** [Jeddi et al.] | 2026 | 일관성 손실 + 정적 깊이 | 적응적 스케일링 법칙 미제시 |
| **Mixture-of-Recursions** [Bae et al.] | 2025 | 토큰별 동적 깊이 할당 라우터 | 명시적 halting, 이론적 안정성 분석 부재 |
| **Deep Equilibrium Models** [Bai et al.] | 2019~2022 | 암묵적 미분, 무한 깊이 근사 | 명시적 backprop 미사용, 다른 패러다임 |
| **Mamba/SSM** [Gu & Dao] | 2024 | 선택적 상태 공간 모델 | 루프 아닌 시퀀스 방향 재귀, 안정성 분석 공유 |
| **Parallel Loop Transformer** [Wu et al.] | 2025 | 루프를 병렬 실행으로 효율화 | 훈련 안정성 이론 미제시 |
| **Scaling Latent Reasoning** [Zhu et al.] | 2025 | 루프 언어 모델의 잠재 추론 | 안정성 분석 부족, 통합 스케일링 법칙 미제시 |
| **Parcae (본 논문)** | 2026 | LTI 분석 + $\rho(A)<1$ 제약 + 통합 스케일링 법칙 | **안정성 이론 + 스케일링 법칙 + 대규모 검증 통합** |

---

## 참고 자료

**주요 참고 논문 (제공된 PDF 내 인용 기준)**

1. **Parcae 논문 (본 문서)**: Prairie, H., Novack, Z., Berg-Kirkpatrick, T., & Fu, D. Y. (2026). *Parcae: Scaling Laws For Stable Looped Language Models*. arXiv:2604.12946v1.
2. Geiping, J., et al. (2025). *Scaling up test-time compute with latent reasoning: A recurrent depth approach*. NeurIPS 2025.
3. Hoffmann, J., et al. (2022). *Training compute-optimal large language models* (Chinchilla). arXiv:2203.15556.
4. Kaplan, J., et al. (2020). *Scaling laws for neural language models*. arXiv:2001.08361.
5. Yang, L., et al. (2024). *Looped transformers are better at learning learning algorithms*. ICLR 2024.
6. Gu, A., & Dao, T. (2024). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv:2312.00752.
7. Bai, S., Kolter, J. Z., & Koltun, V. (2019). *Deep equilibrium models*. arXiv:1909.01377.
8. Desoer, C., & Wu, M. (1968). *Stability of linear time-invariant systems*. IEEE Transactions on Circuit Theory.
9. Dehghani, M., et al. (2019). *Universal transformers*. ICLR 2019.
10. Saunshi, N., et al. (2025). *Reasoning with latent thoughts: On the power of looped transformers*. arXiv:2502.17416.
11. McLeish, S., et al. (2025). *Teaching pretrained language models to think deeper with retrofitted recurrence*. arXiv:2511.07384.
12. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality*. arXiv:2405.21060.
13. Zhu, R., et al. (2025). *Scaling latent reasoning via looped language models*. arXiv:2510.25741.
14. Wu, B., et al. (2025). *Parallel loop transformer for efficient test-time computation scaling*. arXiv:2510.24824.
15. Bae, S., et al. (2025). *Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation*. arXiv:2507.10524.
