# Continuous Audio Language Models (CALM) 

---

## 📌 참고 자료

- **주 논문**: Rouard, S., Orsini, M., Roebel, A., Zeghidour, N., & Défossez, A. (2026). *Continuous Audio Language Models*. arXiv:2509.06926v3 [cs.SD]. (ICLR 제출 preprint)
- Li, T. et al. (2024). *Autoregressive Image Generation without Vector Quantization (MAR)*. NeurIPS.
- Lu, C. & Song, Y. (2025). *Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (TrigFlow)*. ICLR.
- Boffi, N. M. et al. (2025). *How to Build a Consistency Model: Learning Flow Maps via Self-Distillation (LSD)*. arXiv:2505.18825.
- Défossez, A. et al. (2024b). *Moshi: A Speech-Text Foundation Model for Real-Time Dialogue*. arXiv:2410.00037.
- Copet, J. et al. (2023). *Simple and Controllable Music Generation (MusicGen)*. NeurIPS.
- Jia, D. et al. (2025). *DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation*. arXiv:2502.03930.
- Turetzky, A. et al. (2024). *SALAD: Continuous Speech Synthesis using Per-Token Latent Diffusion*. arXiv:2410.16048.
- Pasini, M. et al. (2024b). *Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation*. NeurIPS Audio Imagination Workshop.
- Tschannen, M. et al. (2024). *GIVT: Generative Infinite-Vocabulary Transformers*. arXiv:2312.02116.
- Song, Y. et al. (2023). *Consistency Models*. ICML.
- Ho, J. et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Zeghidour, N. et al. (2021). *SoundStream: An End-to-End Neural Audio Codec*. arXiv:2107.03312.
- Kyutai. (2025). *Pocket TTS Technical Report*. kyutai.org/pocket-tts-technical-report.

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

기존 오디오 언어 모델(Audio Language Model, ALM)은 오디오를 **이산 토큰(discrete token)** 으로 변환하는 손실 압축(RVQ-based codec)에 의존하여, **품질(fidelity)과 계산 비용(computational cost) 사이의 근본적인 트레이드오프**를 갖는다. CALM은 이 문제를 VAE의 **연속 잠재 공간(continuous latent space)** 에서 직접 오토레그레시브 생성을 수행함으로써 해결한다.

### 🔑 주요 기여 (6가지 혁신)

| # | 혁신 | 설명 |
|---|---|---|
| 1 | **노이즈 주입 + 단기 컨텍스트 Transformer** | 훈련 시 장기 컨텍스트에 노이즈를 주입하여 오류 누적 방지; 단기 clean 잠재를 별도 transformer로 보완 |
| 2 | **Diffusion → Consistency 교체** | Diffusion 헤드를 Consistency 모델로 대체하여 추론 속도를 최대 ×20 가속 |
| 3 | **가우시안 온도 샘플링** | Consistency 모델에 온도 제어 메커니즘을 근사하는 휴리스틱 도입 |
| 4 | **Head Batch Multiplier** | 동일 잠재 변수 $\mathbf{z}^s_{\text{long}}$을 여러 노이즈 레벨에 재사용하여 훈련 효율 향상 |
| 5 | **Latent Classifier Free Guidance (CFG)** | 1-step consistency에 CFG를 잠재 변수 수준에서 적용 |
| 6 | **Latent Distillation** | CFG-guided 교사 모델을 소규모 학생 모델로 증류하여 추론 배치 크기 절반으로 감소 |

**실용적 성과**: 100M 파라미터 **Pocket TTS** — 노트북 CPU에서 실시간보다 빠르게 동작하는 오픈소스 TTS 모델.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제의 핵심**: RVQ(Residual Vector Quantization) 기반 이산 오디오 토큰의 구조적 한계

$$\underbrace{\text{음질 향상}}_{\text{RVQ 깊이 증가} \Rightarrow K \uparrow} \implies \underbrace{\text{시퀀스 길이 증가}}_{S \times K} \implies \underbrace{\text{계산 비용 폭증}}_{\mathcal{O}((S \cdot K)^2)}$$

구체적으로:
1. **손실 압축 품질 저하**: 이산화 과정에서 필연적 지각 품질 손실 발생
2. **계산 비용 vs 품질 트레이드오프**: 고품질 생성을 위해 더 깊은 RVQ 계층이 필요하지만, 이는 Transformer의 이차 복잡도를 악화시킴
3. **코드북 붕괴(codebook collapse)**: VQ 훈련의 불안정성
4. **오류 누적(error accumulation)**: 연속 도메인에서 자기회귀 생성 시 초기 오류가 누적되어 품질 저하

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 VAE-GAN 손실 함수

$$\mathcal{L}_{\text{VAE}} = \lambda_t \mathcal{L}_t(x, \hat{x}) + \lambda_f \mathcal{L}_f(x, \hat{x}) + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}(\hat{x}) + \lambda_{\text{feat}} \mathcal{L}_{\text{feat}}(x, \hat{x}) + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}} \tag{2}$$

- $\mathcal{L}_t$, $\mathcal{L}_f$: 시간/주파수 재구성 손실
- $\mathcal{L}_{\text{adv}}$: GAN 적대적 손실
- $\mathcal{L}_{\text{KL}}$: VAE KL 정규화
- $\mathcal{L}_{\text{distill}}$: WavLM 의미론적 증류 손실 (음성 VAE에만 적용)

#### 2.2.2 연속 잠재 모델링 — MAR 기반 배경

MAR(Li et al., 2024)의 Backbone Transformer $T_\theta$는 컨텍스트 임베딩을 생성:

$$\mathbf{z}^s = T_\theta(\mathbf{x}^1, \ldots, \mathbf{x}^{s-1})$$

Diffusion 손실:

$$\mathcal{L}_{\text{diff}}(\theta, \phi) = \sum_{s=1}^{S} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), t \sim [0,1]} \left[ \| \epsilon - \epsilon_\phi(\mathbf{x}^s_t, \mathbf{z}^s, t) \|^2 \right]$$

여기서 $\mathbf{x}^s_t = \alpha_t \mathbf{x}^s + \sigma_t \epsilon$

#### 2.2.3 Flow Matching 기반

$$\mathcal{L}_{\text{FM}}(\phi) = \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I), t \sim \mathcal{U}(0,1)} \left[ w(t) \| F_\phi(\mathbf{x}_t, t) - (\alpha'_t \mathbf{x}_0 + \sigma'_t \epsilon) \|^2_2 \right]$$

Probability Flow ODE:

$$\frac{d\mathbf{x}_t}{dt} = F_\phi(\mathbf{x}_t, t), \quad \mathbf{x}_T \sim \mathcal{N}(0, I)$$

#### 2.2.4 연속 시간 Consistency 모델 (핵심)

Consistency 모델 $f_\phi(\mathbf{x}_t, t)$의 파라미터화:

$$f_\phi(\mathbf{x}_t, t) = c_{\text{skip}}(t)\mathbf{x}_t + c_{\text{out}}(t)F_\phi(\mathbf{x}_t, t)$$

경계 조건: $c_{\text{skip}}(0) = 1$, $c_{\text{out}}(0) = 0$

$T = \frac{\pi}{2}$, $\alpha_t = \cos(t)$, $\sigma_t = \sin(t)$ (TrigFlow 설정)로부터 Lu & Song (2025)이 유도한 **연속 시간 Consistency 손실**:

$$\mathcal{L}_{\text{CM}}(\phi, \psi) = \mathbb{E}_{\mathbf{x}_t, t} \left[ \frac{e^{w_\psi(t)}}{D} \left\| F_\phi(\mathbf{x}_t, t) - F_{\phi^-}(\mathbf{x}_t, t) - \cos(t) \frac{d f_{\phi^-}(\mathbf{x}_t, t)}{dt} \right\|^2_2 - w_\psi(t) \right] \tag{1}$$

여기서 $w_\psi(t)$는 적응적 가중치 함수, $\phi^-$는 EMA(Exponential Moving Average) 파라미터.

#### 2.2.5 CALM 전체 훈련 목적 함수

$$\mathcal{L}_{\text{CALM}}(\theta, \phi, \psi) = \sum_{s=1}^{S} \mathbb{E}_{t, \epsilon} \left[ \frac{e^{w_\psi(t)}}{D} \left\| F_\phi(\mathbf{x}^s_t, t, \mathbf{Z}^s) - F_{\bar{\phi}}(\mathbf{x}^s_t, t, \mathbf{Z}^s) - \cos(t) \frac{d f_{\bar{\phi}}(\mathbf{x}^s_t, t, \mathbf{Z}^s)}{dt} \right\|^2_2 - w_\psi(t) \right] \tag{3}$$

조건 변수:

$$\mathbf{Z}^s = \mathbf{z}^s_{\text{long}} + \mathbf{z}^s_{\text{short}} = T_{\text{long}, \theta_1}(\tilde{\mathbf{x}}^1, \ldots, \tilde{\mathbf{x}}^{s-1}) + T_{\text{short}, \theta_2}(\mathbf{x}^{s-K}, \ldots, \mathbf{x}^{s-1})$$

노이즈 주입 (분산 보존):

$$\tilde{\mathbf{x}}^s = \sqrt{k_s} \epsilon_s + \sqrt{1 - k_s} \mathbf{x}^s, \quad k_s \sim \mathcal{U}(0,1), \quad \epsilon_s \sim \mathcal{N}(0, I)$$

1-step 생성:

$$\hat{\mathbf{x}}^s = f_\phi(\mathbf{x}^s_1 = \epsilon, t=1, \mathbf{Z}^s), \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 2.2.6 Latent CFG

$$\mathbf{Z}^s_{\text{CFG}} = \mathbf{Z}^s_{\emptyset} + \alpha (\mathbf{Z}^s_C - \mathbf{Z}^s_{\emptyset})$$

#### 2.2.7 Lagrangian Self-Distillation (LSD)

LSD 모델 $f_\phi(\mathbf{x}, t, s)$의 정의:

$$f_\phi(\mathbf{x}, t, s) = \mathbf{x} + (s - t) F_\phi(\mathbf{x}, t, s) \tag{4}$$

LSD 손실:

$$\mathcal{L}_{\text{LSD}}(\phi, \psi) = \mathbb{E}_{\mathbf{x}_0, \epsilon, t, s} \left[ e^{-w_\psi(t,s)} \left\| \partial_s f_\phi(\mathbf{x}_t, t, s) - F_{\phi^-}(f_\phi(\mathbf{x}_t, t, s), s, s) \right\|^2_2 + w_\psi(t, s) \right] \tag{6}$$

---

### 2.3 모델 구조

```
오디오 파형 W
    ↓ [VAE Encoder]
연속 잠재 시퀀스 (x¹, x², ..., xˢ)
    ↓
┌────────────────────────────────────────────┐
│           CALM 아키텍처                     │
│                                            │
│  [노이즈 주입] x̃ˢ = √kε + √(1-k)xˢ       │
│       ↓                                   │
│  [Causal Backbone Transformer Tlong]       │
│  → zˢlong (장기 맥락, 거친 구조)            │
│                                            │
│  [Short-Context Transformer Tshort]        │
│  → zˢshort (K=10 최근 clean 잠재, 세밀)    │
│       ↓                                   │
│  Zˢ = zˢlong + zˢshort                    │
│       ↓                                   │
│  [Consistency MLP Head fφ]                │
│  → x̂ˢ (1-step 또는 4-step 생성)           │
└────────────────────────────────────────────┘
    ↓ [VAE Decoder]
재구성된 오디오 파형
```

**주요 구성 요소 파라미터** (음악 모델 기준):
- Backbone Transformer: 1.35B 파라미터 (dim=1536, 48 layers, 24 heads)
- Short-Context Transformer: 113M 파라미터 (4 layers, K=10)
- Consistency MLP Head: 601M 파라미터 (12 layers)
- Music VAE: 128-dim 잠재, 25Hz frame rate, 32kHz

---

### 2.4 성능 향상

#### 음성 연속(Speech Continuation) — Table 2

| 모델 | 온도 | 전체 속도향상 | 샘플러 속도향상 | PPX↓ | VERT↓ | 음향품질↑ | 의미론적 Elo↑ |
|---|---|---|---|---|---|---|---|
| RQ-Transformer (8 RVQ) | 1.0 | ×1.0 | ×1.0 | 52.4 | 36.3 | 2.42 | 1841 |
| RQ-Transformer (8 RVQ) | 0.8 | ×1.0 | ×1.0 | 26.8 | 33.1 | 2.75 | 1870 |
| **CALM-Consistency (1-step)** | **0.8** | **×1.3** | **×12.3** | **23.8** | **31.2** | **3.45** | **2023** |

#### 음악 연속(Music Continuation) — Table 4

| 모델 | 전체 속도향상 | 샘플러 속도향상 | FAD↓ | 음향품질↑ |
|---|---|---|---|---|
| RQ-Transformer (32 RVQ) | ×1.0 | ×1.0 | 1.06 | 2.85 |
| **CALM-Consistency (1-step)** | **×2.2** | **×19.3** | **0.83** | **2.90** |
| CALM-Consistency (4-step) | ×1.9 | ×5.4 | 0.71 | 3.07 |

#### TTS — Table 3

| 모델 | WER↓ | CER↓ | 음향품질↑ |
|---|---|---|---|
| F5-TTS (NFE=32) | 2.42 | - | 54.7 |
| DSM (16 RVQ) | 1.95 | - | 60.2 |
| **CALM w/ LSD (NFE=1)** | **1.81** | **0.57** | **61.1** |

#### Pocket TTS — Table 12

| 모델 | 파라미터 | WER↓ | CPU 실시간 이하 |
|---|---|---|---|
| F5-TTS | 336M | 2.21 | ✗ |
| DSM | 750M | 1.84 | ✗ |
| **Pocket TTS** | **100M** | **1.84** | **✓** |

---

### 2.5 한계점

1. **화자 유사도(Speaker Similarity)**: CALM의 자동 화자 유사도 점수(SIM=0.52)가 낮게 측정됨. 단, 이는 VAE 자체의 임베딩 공간 변환으로 인한 계측 오류로 추정되며, 인간 평가에서는 오히려 ground truth를 상회함 (모든 방법이 ground truth보다 높은 점수 획득).

2. **음악 의미 증류 부재**: 음성 VAE는 WavLM 의미 증류를 적용하지만, 음악 VAE에는 의미 정의의 어려움으로 적용하지 못함 → 향후 과제.

3. **단기 컨텍스트 트랜스포머의 음성 미적용**: 음성 continuation에서는 단기 컨텍스트 트랜스포머와 노이즈 주입이 성능 향상에 기여하지 않아 생략됨 → 과제별 설계 차이 존재.

4. **확장성 연구 미완**: 3B 파라미터까지 성능이 향상됨을 확인했으나, 완전한 확장성 연구는 향후 과제로 남김.

5. **1-step consistency의 TrigFlow 대비 품질 열세**: 충분한 추론 단계(>25 steps)에서는 TrigFlow가 더 나은 품질을 보임. Consistency는 빠른 추론에서만 우위.

6. **스테레오/다채널 오디오 미지원**: 현재 32kHz 모노 포맷으로 제한.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 설계 요소

#### (1) 노이즈 주입을 통한 오류 누적 방지

자기회귀 생성의 핵심 일반화 문제는 **훈련-추론 분포 불일치(train-inference distribution mismatch)** 이다. 훈련 시에는 ground truth 잠재를 입력으로 사용하지만, 추론 시에는 모델 자신이 생성한 잠재를 입력으로 사용하여 오류가 누적된다.

CALM은 이를 분산 보존 노이즈 주입으로 해결한다:

$$\tilde{\mathbf{x}}^s = \sqrt{k_s} \epsilon_s + \sqrt{1 - k_s} \mathbf{x}^s$$

이 설계는 모델이 훈련 시에도 **잡음 섞인 맥락**에서 학습하도록 강제함으로써, 추론 시 발생하는 부정확한 잠재 입력에 대한 로버스트성(robustness)을 획득하게 한다. 기존 Pasini et al. (2024b)의 단순 노이즈 주입 $\tilde{\mathbf{x}}^s = k_s \epsilon_s + (1-k_s)\mathbf{x}^s$은 분산이 불보존되어 품질 저하를 초래했음을 실험적으로 확인.

#### (2) 이중 컨텍스트 구조 (Dual Context)

$$\mathbf{Z}^s = \underbrace{\mathbf{z}^s_{\text{long}}}_{\text{거친 장기 구조}} + \underbrace{\mathbf{z}^s_{\text{short}}}_{\text{세밀한 단기 정보}}$$

- **장기 컨텍스트**: 노이즈가 주입된 잠재를 처리하여 **전역적(global) 구조** 학습
- **단기 컨텍스트**: 최근 K=10개의 clean 잠재를 처리하여 **지역적(local) 세부 정보** 보완

이 이중 구조는 DiTAR(Jia et al., 2025)의 "패치 기반 로컬 컨텍스트"와 유사한 관찰에서 출발한다: 로컬 컨텍스트 제공이 생성 품질에 결정적임. Ablation study (Table 5)에서 단기 컨텍스트 트랜스포머 제거 시 FAD가 $0.93 \to 4.03$으로 급등 (약 4.3배 악화)하여 일반화에 필수적임을 확인.

#### (3) VAE의 연속 잠재 공간

VAE는 가우시안 사전분포(prior)를 강제하여 **구조화되고 부드러운(smooth) 잠재 공간**을 학습한다:

$$\mathcal{L}_{\text{KL}} = \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| \mathcal{N}(0, I))$$

이 공간은 이산 코드북과 달리:
- 양자화 오류(quantization error)가 없어 보간(interpolation) 및 외삽(extrapolation)이 자연스러움
- 새로운 도메인의 오디오에도 부드럽게 대응 가능 (코드북 OOV 문제 없음)
- 잠재 벡터 간의 의미론적 관계가 연속적으로 인코딩되어 제로샷(zero-shot) 일반화 가능성 향상

#### (4) 도메인 독립적 아키텍처

CALM은 동일한 프레임워크를 4개 과제에 적용:
- 음성 연속(Speech Continuation)
- 텍스트-음성 변환(TTS)
- 음악 연속(Music Continuation)  
- 텍스트-음악 생성(Text-to-Music)

과제별 조건 신호만 변경하고 핵심 아키텍처는 공유 → **도메인 간 일반화 잠재력** 입증.

#### (5) 확장성 (Scalability)

Table 10에서 1.3B → 3B 파라미터 확장 시 CALM과 RQ-Transformer 모두 유사한 비율로 FAD 개선:

| 모델 | 1.3B FAD | 3B FAD | 개선율 |
|---|---|---|---|
| CALM | 0.71 | 0.62 | -12.7% |
| RQ-Transformer | 1.06 | 0.98 | -7.5% |

CALM이 스케일 확장에서 더 큰 이득을 보이며, 대규모 모델로의 일반화 가능성이 더 높음을 시사.

#### (6) 의미론적 증류를 통한 표현 구조화

음성 VAE에서 WavLM(Chen et al., 2021b)을 교사 모델로 사용한 의미론적 증류:

```math
\mathcal{L}_{\text{distill}} = 1 - \text{cosine\_sim}(\mathbf{z}_{\text{VAE}}, \mathbf{z}_{\text{WavLM}})
```

이를 통해 잠재 공간이 음성의 **음성학적(phonetic) 구조**를 인코딩하게 되어, 훈련 도메인 외 발화에서도 의미론적으로 일관된 표현을 생성할 수 있다. Table 1에서 VAE가 VQ-VAE보다 ABX 음성 판별 점수가 9.4% → 8.1%로 향상됨을 확인.

#### (7) 가우시안 온도 샘플링의 일반화 효과

$$\epsilon_{\text{temp}} \sim \mathcal{N}(0, \tau \cdot I), \quad 0 < \tau \leq 1$$

온도 $\tau = 0.8$ 적용 시 다양성(diversity)과 품질(fidelity) 사이의 균형을 조절할 수 있으며, 이는 이산 모델의 temperature sampling과 유사한 효과를 보임(Figure 2). 이 메커니즘은 **분포 외(out-of-distribution) 생성 시 품질 저하를 억제**하는 정규화 역할.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 연속 오디오 생성 패러다임 비교

| 연구 | 연도 | 방법 | 표현 | 속도 | 음질 | 다중 과제 |
|---|---|---|---|---|---|---|
| **WaveNet** (van den Oord et al.) | 2016 | 자기회귀 (raw) | 파형 | 매우 느림 | 높음 | ✗ |
| **Jukebox** (Dhariwal et al.) | 2020 | VQ-VAE + Transformer | 이산 | 느림 | 중간 | ✗ |
| **AudioLM** (Borsos et al.) | 2023 | RVQ + AR | 이산 | 중간 | 높음 | △ |
| **MusicGen** (Copet et al.) | 2023 | RVQ + delay pattern | 이산 | 중간 | 높음 | ✗ |
| **GIVT** (Tschannen et al.) | 2024 | GMM + Transformer | 연속 | 중간 | 중간 | ✗ (이미지) |
| **MAR** (Li et al.) | 2024 | Diffusion head + Transformer | 연속 | 느림 | 높음 | ✗ (이미지) |
| **SALAD** (Turetzky et al.) | 2024 | Per-token latent diffusion | 연속 | 느림 | 높음 | ✗ (TTS) |
| **MELLE** (Meng et al.) | 2024 | Mel-spectrogram AR | 스펙트럼 | 빠름 | 높음 | ✗ (TTS) |
| **DiTAR** (Jia et al.) | 2025 | Patch + Diffusion Transformer | 연속 | 느림 | 높음 | ✗ (TTS) |
| **IMPACT** (Huang et al.) | 2025 | MAR + 반복 마스킹 | 연속 | 중간 | 높음 | ✗ (T2A) |
| **MingUni-Audio** (Yan et al.) | 2025 | MoE 연속 LM (20B) | 연속 | 느림 | 매우 높음 | ✓ |
| **CALM (본 논문)** | 2026 | Consistency + 이중 Transformer | 연속 | **매우 빠름** | **높음** | **✓** |

### 4.2 기술적 세부 비교

#### CALM vs. MAR (Li et al., 2024)

| 비교 항목 | MAR | CALM |
|---|---|---|
| 생성 헤드 | Diffusion (수백 스텝) | Consistency (1~4 스텝) |
| 오류 누적 대응 | 없음 | 노이즈 주입 + 분산 보존 |
| 단기 컨텍스트 | 없음 | Short-Context Transformer |
| 적용 도메인 | 이미지 | 음성 + 음악 |
| 추론 속도 | 느림 | 최대 ×20 빠름 |

#### CALM vs. SALAD (Turetzky et al., 2024)

| 비교 항목 | SALAD | CALM |
|---|---|---|
| 생성 헤드 | Per-token latent diffusion | Consistency/LSD (1-step) |
| 음악 생성 | ✗ | ✓ |
| 데이터셋 규모 | 소규모 | 88K시간 (TTS), 20K시간 (음악) |
| 추론 속도 | NFE=20 필요 | NFE=1 가능 |

#### CALM vs. DiTAR (Jia et al., 2025)

| 비교 항목 | DiTAR | CALM |
|---|---|---|
| 로컬 컨텍스트 | 패치(patch) 기반 | Short-Context Transformer |
| 전역-지역 통합 | LM + Bidirectional DiT | Causal Transformer + 이중 구조 |
| 추론 단계 | NFE=10 (권장) | NFE=1 가능 |
| WER (LibriSpeech) | 2.39 | **1.81** |

#### MusicCaps 벤치마크 비교 (Table 11)

| 모델 | FAD↓ | KLD↓ | CLAP↑ |
|---|---|---|---|
| MusicLM | 4.00 | - | - |
| MusicGen | 3.40 | 1.23 | 0.37 |
| AudioLDM2 | 3.13 | 1.20 | 0.43 |
| Noise2Music | 2.10 | - | - |
| Jen-1 | 2.00 | 1.29 | - |
| MusicFlow | 2.69 | 1.23 | 0.52 |
| **CALM-Consistency (4-step)** | **2.14** | **1.30** | **0.44** |

CALM이 전용 텍스트-음악 모델들과 경쟁적 성능을 보임.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### (1) 패러다임 전환의 촉매

CALM은 오디오 생성의 **이산화 의존성 탈피**를 실용적으로 입증한 첫 번째 포괄적 연구이다. 음성과 음악 모두에서 이산 모델 대비 우위를 보임으로써, 향후 오디오 생성 연구의 기본 방향이 연속 잠재 공간 모델링으로 이동할 가능성을 시사한다.

#### (2) 엣지(Edge) 배포 가능성 개척

Pocket TTS (100M 파라미터, 노트북 CPU에서 실시간 이하 동작)는 **모바일/임베디드 음성 AI**의 새로운 기준을 제시한다. 이는 다음 연구들을 촉진할 것으로 예상된다:
- 모델 압축과 지식 증류(knowledge distillation)를 결합한 경량 오디오 LM 연구
- On-device 실시간 음성 합성 및 대화 시스템 연구

#### (3) 연속 표현의 의미론적 구조화 방향

WavLM 증류를 통한 의미론적 잠재 공간 구축은 **다중 모달(multimodal) 통합**에 직접적으로 활용 가능하다. 텍스트-음성-이미지를 동일한 연속 잠재 공간에서 처리하는 통합 모델 연구가 활성화될 것으로 예상.

#### (4) Consistency 모델의 오디오 분야 확산

1-step 고품질 생성이 가능한 consistency 모델의 오디오 적용 가능성을 입증함으로써:
- 실시간 음성 변환(voice conversion)
- 온라인 스트리밍 오디오 생성
- 저지연 음악 반주 시스템

등의 응용 연구가 활발해질 것으로 예상.

#### (5) MingUni-Audio (Yan et al., 2025)의 방향성

CALM 이후 MingUni-Audio (20B MoE, 3B active)가 연속 음성 LM의 대규모 확장 가능성을 보여주었다. CALM의 기술적 혁신이 대규모 모델에서도 유효함을 시사.

---

### 5.2 향후 연구 시 고려할 점

#### (1) 음악 의미론적 표현 학습

**현재 한계**: 음악 VAE에는 의미론적 증류가 적용되지 않음.  
**고려할 방향**: 
- CLAP(Elizalde et al., 2023) 등 음악-언어 정렬 모델을 교사로 활용한 의미 증류
- 음악 구조 분석(조성, 리듬, 멜로디)을 잠재 공간에 명시적으로 인코딩하는 방법론

$$\mathcal{L}_{\text{music-distill}} = 1 - \text{cosine sim}(\mathbf{z}_{\text{VAE}}, \mathbf{z}_{\text{CLAP-audio}})$$

#### (2) 오류 누적 메커니즘의 이론적 분석

**현재 한계**: 노이즈 주입의 효과가 경험적으로 입증되었으나 이론적 근거 미흡.  
**고려할 방향**:
- 자기회귀 생성의 compounding error를 수식적으로 분석
- 최적 노이즈 스케줄 $k_s$를 학습 가능한 파라미터로 설계

$$k_s^* = \arg\min_{k} \mathbb{E}\left[\|\mathbf{x}^s_{\text{gen}} - \mathbf{x}^s_{\text{GT}}\|^2_2\right]$$

#### (3) 장기 의존성과 단기 세부정보의 균형

**현재 한계**: K=10 (약 0.4초)의 단기 컨텍스트 창 크기가 경험적으로 설정됨.  
**고려할 방향**:
- 적응적 컨텍스트 창 크기 (Adaptive K)
- Attention 기반 동적 단기-장기 가중치 할당

#### (4) 다중 화자/음악가 제어

**현재 한계**: 화자 유사도 측정에서 VAE 공간 내 화자 정체성 보존이 불완전.  
**고려할 방향**:
- 화자 임베딩을 잠재 공간에 명시적으로 분리하는 disentanglement 학습
- 조건부 VAE (CVAE)로 확장하여 화자 ID를 구조적으로 인코딩

#### (5) 스테레오 및 고해상도 오디오

**현재 한계**: 32kHz 모노로 제한.  
**고려할 방향**:
- 다채널(stereo/surround) VAE 설계
- 48kHz 이상 고해상도 오디오 지원을 위한 계층적 VAE

#### (6) 완전한 확장성 연구

**현재 한계**: 3B 파라미터까지만 실험, 체계적 scaling law 분석 미완.  
**고려할 방향**:
- Chinchilla 법칙과 유사한 오디오 생성의 최적 파라미터-데이터 비율 도출
- 연속 잠재 공간의 차원(dimension)과 모델 크기의 공동 최적화

#### (7) 스트리밍 생성의 지연 최소화

**고려할 방향**:
- Consistency 모델의 1-step 특성과 완전 인과적(fully causal) 설계를 결합하여 청크(chunk) 단위 스트리밍 최적화
- 실시간 대화 시스템(spoken dialogue)에서의 지연(latency) 임계값 분석

$$\text{목표}: \text{RTF} = \frac{T_{\text{inference}}}{T_{\text{audio}}} < 0.5 \quad \text{(CPU 기준)}$$

#### (8) 데이터 효율성 및 소규모 데이터 학습

**현재 한계**: 대규모 데이터(88K시간 TTS, 20K시간 음악)에 의존.  
**고려할 방향**:
- Few-shot 또는 zero-shot 적응을 위한 메타러닝(meta-learning)과의 결합
- 연속 잠재 공간의 표현력을 활용한 데이터 증강 전략

#### (9) 평가 지표의 한계 극복

**현재 한계**: WavLM 기반 화자 유사도가 VAE 통과 후 실제 품질을 반영하지 못함.  
**고려할 방향**:
- VAE-aware 화자 유사도 측정 방법론 개발
- 연속 잠재 공간에 특화된 새로운 품질 평가 지표 설계

---

## 결론 요약

CALM은 오디오 생성 분야에서 **이산 토큰 패러다임을 연속 잠재 공간으로 전환**하는 핵심적인 이정표를 제시한다. 노이즈 주입, 이중 컨텍스트 트랜스포머, Consistency 모델, Latent CFG, Latent Distillation의 조합은 품질-속도-크기의 삼중 트레이드오프를 동시에 개선하며, Pocket TTS라는 실용적 성과물로 구체화되었다. 향후 연구는 음악 의미론, 장기 확장성, 다채널 지원, 그리고 엣지 배포 최적화 방향으로 진행될 것으로 전망된다.
