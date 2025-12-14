# BDDM: Bilateral Denoising Diffusion Models for Fast and High-Quality Speech Synthesis

### 1. 핵심 주장 및 주요 기여 요약

**BDDM(Bilateral Denoising Diffusion Models)**은 음성 합성 분야에서 **확산 모델의 느린 샘플링 속도 문제**를 혁신적으로 해결하는 논문입니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**
- **양측 모델링(Bilateral Modeling)**: 순방향 과정과 역방향 과정을 각각 스케줄 네트워크(schedule network)와 스코어 네트워크(score network)로 매개변수화
- **더 타이트한 하한(Tighter Lower Bound)**: 기존 DDPM보다 로그 변계 가능도의 더 낮은 하한 달성
- **초고속 샘플링**: 단 **7 스텝으로 DiffWave 대비 28.6배, WaveGrad 대비 143배 빠른 생성**이 가능하면서 인간 음성과 구분 불가능한 품질 유지
- **사전학습 재사용성**: 기존의 모든 DPM 스코어 네트워크를 상속받아 빠르게 학습 가능

***

### 2. 해결하고자 하는 문제 및 동기

#### 2.1 핵심 문제

확산 확률 모델(DPMs)은 이미지와 음성 합성에서 뛰어난 성능을 보였지만, **본질적으로 느린 샘플링 속도**라는 심각한 병목이 존재합니다:[1]

- DDPMs: 학습에 T(수백에서 수천 스텝)개의 확산 스텝 필요
- WaveGrad: 1000 스텝 필요(실시간 인수 RTF: 38.2)
- DiffWave: 200 스텝 필요(RTF: 7.30)

이는 **산업적 배포가 사실상 불가능**하다는 의미입니다.

#### 2.2 기존 방법의 한계

논문은 기존의 가속화 방법들의 문제점을 명확히 지적합니다:[1]

| 방법 | 한계 |
|------|------|
| 그리드 탐색(Grid Search) | O(9^N) 복잡도로 N>6일 때 실용적 불가능 |
| 선형/이차 시간 스케줄(Linear/Quadratic schedule) | 데이터셋별로 최적값 달라서 일반화 어려움 |
| DDIM | 단순히 시간 스케줄만 변경하여 스코어 네트워크와 불일치 |
| 기존 학습 방식 | 스코어 네트워크용 L_ddpm 목적함수가 스케줄 학습에 부적절 |

#### 2.3 핵심 통찰

**"학습용 노이즈 스케줄(β)과 샘플링용 스케줄(β̂)이 질적으로 다르다"**

따라서 샘플링 스케줄은 **스코어 네트워크의 실제 특성**을 반영하도록 최적화되어야 합니다.

***

### 3. 제안 방법론 (수식 포함)

#### 3.1 문제 형식화

BDDM은 두 개의 독립적인 확산 과정을 정의합니다:[1]

**훈련 과정** (Gaussian diffusion parameterized by β):

$$q_\beta(x_{1:T}|x_0) := \prod_{t=1}^{T} q_{\beta_t}(x_t|x_{t-1})$$

여기서:

$$q_{\beta_t}(x_t|x_{t-1}) := \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**샘플링 과정** (짧은 스케줄 β̂으로):

$$q_{\hat{\beta}}(\hat{x}_{1:N}|\hat{x}_0) = \prod_{n=1}^{N} q_{\hat{\beta}_n}(\hat{x}_n|\hat{x}_{n-1})$$

단, **N ≪ T** (예: N=7, T=200)

#### 3.2 핵심 개념: 접합 변수(Junctional Variable)

스케줄 네트워크를 학습하기 위해 **접합 변수 x_t**를 도입합니다:[1]

$$q_{\hat{\beta}_{n+1}}(\hat{x}_{n+1}|\hat{x}_n = x_t) := q_\beta(x_{t+\tau}|x_t) = \mathcal{N}\left(\sqrt{\frac{\alpha_{t+\tau}^2}{\alpha_t^2}}x_t, \left(1-\frac{\alpha_{t+\tau}^2}{\alpha_t^2}\right)I\right)$$

여기서:
- 접합 변수: $$x_t = \alpha_t x_0 + \sqrt{1-\alpha_t^2}\epsilon_n$$
- 누적곱: $$\alpha_t = \prod_{i=1}^{t}\sqrt{1-\beta_i}$$
- 스텝 크기 제어 변수: $$\tau = \lfloor T/N \rfloor$$

**직관**: 짧은 확산 과정의 한 스텝이 긴 확산 과정의 τ 스텝에 해당.

#### 3.3 스코어 네트워크의 훈련 목적 함수

새로운 하한을 유도합니다:[1]

$$\log p_\theta(\hat{x}_0) \geq F^{(n)}_{\text{score}}(\theta) := -L^{(n)}_{\text{score}}(\theta) - R_\theta(\hat{x}_0, x_t)$$

여기서:

$$L^{(n)}_{\text{score}}(\theta) := D_{KL}(p_\theta(\hat{x}_{n-1}|\hat{x}_n = x_t) \parallel q_{\hat{\beta}}(\hat{x}_{n-1}; x_t, \epsilon_n))$$

**명제 2**: 기존 DDPM의 훈련 목적 함수로 최적화된 θ*은 **동일하게** 새로운 목적함수도 최적화합니다:

$$\arg\min_\theta L^{(t)}_{\text{ddpm}}(\theta) \equiv \arg\min_\theta L^{(n)}_{\text{score}}(\theta)$$

따라서 **기존의 모든 사전학습된 DDPM을 재사용 가능**합니다.

#### 3.4 스케줄 네트워크의 훈련 목적 함수

**핵심 기여**: 데이터 의존적인 스케줄 학습

스케줄 네트워크는 다음 상한으로 제약됩니다:[1]

```math
0 < \hat{\beta}_n < \min\left\{1 - \frac{\hat{\alpha}_{n+1}^2}{1-\hat{\beta}_{n+1}}, \hat{\beta}_{n+1}\right\}
```

이를 이용하여 다음과 같이 매개변수화합니다:

```math
f_\phi(x_t; \hat{\beta}_{n+1}) := \min\left\{1 - \frac{\hat{\alpha}_{n+1}^2}{1-\hat{\beta}_{n+1}}, \hat{\beta}_{n+1}\right\} \cdot \sigma_\phi(x_t)
```

여기서 σ_φ는 신경망으로 학습되며 **(0, 1) 범위의 비율 예측**.

**스케줄 네트워크의 훈련 목적함수**:[1]

$$L^{(n)}_{\text{step}}(\phi; \theta^*) := D_{KL}(p_{\theta^*}(\hat{x}_{n-1}|\hat{x}_n = x_t) \parallel q_{\hat{\beta}_n(\phi)}(\hat{x}_{n-1}; x_0, \alpha_t))$$

구체적으로:

```math
L^{(n)}_{\text{step}}(\phi; \theta^*) = \frac{\delta_t}{2(\delta_t - \hat{\beta}_n(\phi))}\left\|\epsilon_n - \frac{\hat{\beta}_n(\phi)}{\delta_t}\epsilon_{\theta^*}(x_t, \alpha_t)\right\|^2_2 + C
```

여기서:
- $$\delta_t = 1 - \alpha_t^2$$
- $$C = \frac{1}{4}\log\frac{\delta_t}{\hat{\beta}_n} + \frac{D}{2}\left(\frac{\hat{\beta}_n}{\delta_t} - 1\right)$$

**중요 성질**: Proposition 3에 따르면 θ를 먼저 최적화하고 **그 후** φ를 훈련해야 합니다.

***

### 4. 모델 구조

#### 4.1 전체 아키텍처

BDDM은 **두 개의 신경망**으로 구성됩니다:

```
┌─────────────────────────────────────────────┐
│    Training Phase                           │
├─────────────────────────────────────────────┤
│ Step 1: Score Network 훈련 (Algorithm 1)   │
│  - 입력: x0 샘플, β 스케줄                 │
│  - 손실: L(t)_ddpm                         │
│  - 훈련: 기존 DDPM과 동일                  │
│                                             │
│ Step 2: Schedule Network 훈련 (Algorithm 2)│
│  - 입력: xt (훈련된 점), β̂_{n+1}          │
│  - 신경망: σ_φ(xt) → [0,1]의 비율          │
│  - 손실: L(n)_step(φ; θ*)                 │
│  - 수렴: ~10k 스텝 (1시간)                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│    Inference Phase                          │
├─────────────────────────────────────────────┤
│ Step 1: Noise Scheduling (Algorithm 3)     │
│  - 입력: 초기 N, α̂_N, β̂_N                │
│  - 과정: Backward 계산 (N→1)               │
│  - 조기 종료: β_n < β_1 일 때              │
│                                             │
│ Step 2: 샘플링 (Algorithm 4)               │
│  - 입력: x_N ~ N(0,I)                     │
│  - 역방향: x_{n-1} ~ p_θ*(x_{n-1}|x_n)   │
│  - 출력: x̂_0 (음성 샘플)                   │
└─────────────────────────────────────────────┘
```

#### 4.2 스코어 네트워크 (Score Network)

**아키텍처**: DiffWave 기반[1]
- 128개 잔여 채널(residual channels)
- 조건: mel-spectrogram + noise scale α_t
- 출력: ϵ_θ(x_t, α_t) - 예측된 노이즈

#### 4.3 스케줄 네트워크 (Schedule Network)

**아키텍처**: GALR(Globally Attentive Locally Recurrent) 네트워크[1]

```
┌─────────────────────────────────────────┐
│ 입력: x_t (현재 잡음 있는 신호)         │
├─────────────────────────────────────────┤
│ 인코딩: 8 샘플 윈도우로 인코딩           │
│ GALR Block 1: 128 hidden dims            │
│ GALR Block 2: 128 hidden dims            │
│ 출력: Sigmoid 활성화                     │
│ 풀링: AvgPool2D (세그먼트&특징 차원)     │
├─────────────────────────────────────────┤
│ 출력: σ_φ(x_t) ∈ (0, 1)                 │
│       → 노이즈 스케일 비율 예측           │
└─────────────────────────────────────────┘
```

**특성**: 
- 스코어 네트워크 대비 **3.6배 빠름**
- 훈련 시간: ~1시간 (1GPU에서)
- 매개변수: 스코어 네트워크의 ~5-10%

***

### 5. 성능 향상 및 실험 결과

#### 5.1 음성 품질 평가 (LJSpeech Dataset)

**표 1: 신경 보코더 비교**[1]

| 모델 | MOS | RTF | 스텝 |
|------|-----|-----|------|
| Ground-truth | 4.64 ± 0.08 | - | - |
| WaveNet (MoL) | 3.52 ± 0.16 | 318.6 | - |
| HiFi-GAN | 4.33 ± 0.12 | 0.0134 | - |
| WaveGrad (1000 steps) | 4.36 ± 0.13 | 38.2 | 1000 |
| DiffWave (200 steps) | 4.49 ± 0.13 | 7.30 | 200 |
| **BDDM (3 steps)** | 3.64 ± 0.13 | **0.110** | 3 |
| **BDDM (7 steps)** | **4.43 ± 0.11** | **0.256** | 7 |
| **BDDM (12 steps)** | **4.48 ± 0.12** | **0.438** | 12 |

**주요 발견**:
- BDDM-7: DiffWave와 **통계적으로 유의미한 차이 없음** (p≥0.05)
- **28.6배 빠른 속도** (RTF 0.256 vs 7.30)
- 3-12 스텝에서 **일관되게 안정적인 품질**

#### 5.2 가속화 방법 비교 (Table 2)[1]

**7 스텝 비교**:

| 방법 | STOI | PESQ | MOS |
|------|------|------|-----|
| GS (그리드 탐색) | 불가능 (계산량 폭발) | - | - |
| FS (FastSampling) | 0.939 ± 0.023 | 3.09 ± 0.23 | 3.10 ± 0.12 |
| DDIM | 0.974 ± 0.008 | 3.85 ± 0.12 | 3.94 ± 0.12 |
| NE (Noise Estimation) | 0.978 ± 0.007 | 3.75 ± 0.18 | 4.02 ± 0.11 |
| **BDDM** | **0.983 ± 0.006** | **3.96 ± 0.09** | **4.43 ± 0.11** |

**12 스텝 비교**:

| 방법 | STOI | PESQ | MOS |
|------|------|------|-----|
| DDIM | 0.979 ± 0.006 | 3.90 ± 0.10 | 4.16 ± 0.12 |
| NE | 0.981 ± 0.007 | 3.82 ± 0.13 | 3.98 ± 0.14 |
| **BDDM** | **0.987 ± 0.006** | **3.98 ± 0.12** | **4.48 ± 0.12** |

**핵심 통찰**: BDDM은 모든 스텝 수에서 **일관되게 최고 성능**.

#### 5.3 다중화자 음성 합성 (VCTK Dataset)[1]

| 스케줄 | 방법 | STOI | PESQ | MOS |
|--------|------|------|------|-----|
| 8 steps | DDPM (GS) | 0.787 | 3.31 | 4.22 |
| | **BDDM** | 0.774 | 3.18 | 4.20 |
| 16 steps | DDIM | 0.724 | 3.04 | 3.88 |
| | **BDDM** | **0.813** | **3.39** | **4.35** |
| 21 steps | DDIM | 0.739 | 3.12 | 4.12 |
| | **BDDM** | **0.827** | **3.43** | **4.48** |

**일반화 성능**: 단일 화자(LJ)와 다중 화자(VCTK) 모두에서 **일관된 우수 성능**.

#### 5.4 이미지 생성 (CIFAR-10 Dataset)[1]

**표 6: FID 점수 비교**

| 방법 | 스텝 | FID |
|------|------|-----|
| DDPM 기준 | 1000 | 3.17 |
| DDIM (이차) | 100 | 4.16 |
| FastDPM | 100 | 2.86 |
| Improved DDPM | 100 | 4.63 |
| **BDDM** | **100** | **2.38** ✓ |
| **BDDM** | **50** | **2.93** |

**결과**: BDDM은 **100 스텝에서 1000 스텝 기준을 초과**, 다른 가속화 방법 중 **최고 성능**.

#### 5.5 객관적 평가 지표[1]

**STOI (Short-Time Objective Intelligibility)**
- 범위: 0-1 (높을수록 좋음)
- BDDM-7: 0.983 (거의 완벽)

**PESQ (Perceptual Evaluation of Speech Quality)**
- 범위: -0.5 ~ 4.5 (높을수록 좋음)  
- BDDM-7: 3.96 (매우 우수)

**MOS (Mean Opinion Score)**
- 범위: 1-5 (높을수록 좋음)
- BDDM-7: 4.43 (DiffWave 4.49와 통계적으로 동등)

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재 일반화 성능

**강점**:

1. **아키텍처 독립성**: 다양한 스코어 네트워크 호환[1]
   - DiffWave 아키텍처
   - WaveGrad 아키텍처
   - CIFAR-10 사전학습 모델
   - 모두 BDDM으로 가속화 가능

2. **데이터셋 일반화**: 단일화자 & 다중화자 모두 효과적[1]
   - LJSpeech (단일 여성 화자)
   - VCTK (108명 다중 화자)
   - CIFAR-10 (이미지)

3. **역방향 프로세스 유연성**[1]
   - DDPM 역방향 프로세스 호환
   - DDIM 역방향 프로세스 호환
   - 둘 다 비슷한 성능 달성

#### 6.2 일반화 성능 향상의 이론적 기반

**Proposition 3의 의미**:[1]

스케줄 네트워크 학습이 **로그 변계 가능도의 하한을 최소화**함을 보장:

```math
\log p_{\theta^*}(\hat{x}_0) - F^{(n)}_{\text{score}}(\theta^*) = D_{KL}(p_{\theta^*}(\hat{x}_{n-1}|\hat{x}_n = x_t) \parallel q_{\hat{\beta}_n}(\hat{x}_{n-1}; x_0, \alpha_t))
```

이는 **데이터 분포와 관계없이** 최적의 스케줄을 학습하도록 강제합니다.

#### 6.3 향상 가능성: 절제 연구(Ablation Study)

**그림 2-3: 훈련 손실 및 하한 비교**[1]

```
손실 수렴 특성:

L_elbo 사용: 신경망 출력이 몇 스텝 내에 0으로 붕괴
              → 데이터 의존성 없음 (치명적 문제)

L_step 사용: 진동하는 출력 (시간 스텝에 따라 변함)
              → 올바른 데이터 의존성 학습 ✓

하한 품질 (F_bddm vs F_elbo):
- F_bddm은 F_elbo보다 항상 타이트 ✓
- t ≤ 50 (어려운 구간)에서 특히 우수
  → 스코어 예측이 어려운 부분에서 더 나은 가이드
```

#### 6.4 일반화 성능 향상 전략

**현재 제약사항**:

1. **그리드 탐색 필요**: 초기 (α̂_N, β̂_N) 설정[1]
   - 복잡도: O(M²) (M=9로 81 가능성)
   - 평가용 샘플: 최소 1개만 필요
   - 선택 메트릭: PESQ

2. **스텝 크기 변수 τ**: 하이퍼파라미터[1]
   - τ = 66 (LJ/VCTK)
   - τ가 높을수록 예측된 스케줄이 짧아짐

**향상 가능성**:

$$\text{제안}: \hat{\alpha}_N = f(\text{데이터 특성}), \hat{\beta}_N = g(\text{모델 특성})$$

데이터와 모델 특성을 기반으로 **메타 학습**으로 초기값 자동화.

***

### 7. 한계와 도전과제

#### 7.1 이론적 한계

**1. Proposition 2의 조건**[1]

$x_t \sim q_\beta(x_t|x_0)$ 조건 하에서만 성립:

- 실제 추론에서는 $x_t$가 샘플된 데이터의 분포에서 벗어날 수 있음
- 특히 초기 스텝(큰 노이즈)에서 영향 가능

**2. 근사 오차**[1]

접합 변수 $x_t$는 훈련 중에는 계산 가능하지만, 추론 중에는 근사:

$$t \sim U\{(n-1)\tau, ..., n\tau\}$$

이 근사가 정확하지 않을 수 있음.

#### 7.2 실무적 한계

**1. 초기 하이퍼파라미터 검색**[1]

- 데이터셋별로 (α̂_N, β̂_N) 튜닝 필요
- 자동 선택 방법 부재

**2. 모델 아키텍처 의존성**[1]

- 논문은 GALR과 VGG11만 테스트
- 다른 아키텍처(Transformer 기반 등)에서의 성능 미검증

**3. 음성 길이 제약**[1]

- 실험: 22kHz, 약 2-3초 음성
- 더 긴 음성(>10초)에서의 성능 미정

#### 7.3 수렴성 분석 부족

논문에서 **스케줄 네트워크가 수렴하는 조건**에 대한 이론적 분석이 부족:

- $L^{(n)}_{\text{step}}$ 손실이 **항상 수렴**하는가?
- 수렴 속도는 얼마나 빠른가?
- 국소 최소값에 빠질 가능성?

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 주요 비교 기준

| 논문 | 발표 | 핵심 아이디어 | 속도 | 품질 | 일반화 |
|------|------|-------------|------|------|--------|
| **BDDM (본 논문)** | **2022** | **스케줄 네트워크 학습** | **7 스텝** | **MOS 4.43** | **우수** |
| DDIM | 2021 | 시간 스케줄 선택 | 50 스텝 | 낮음 | 보통 |
| FastDiff | 2022 | 아키텍처 최적화 | 4 스텝 | MOS 4.28 | 보통 |
| LinDiff | 2023 | ODE 기반 선형 경로 | 20 스텝 | MOS 4.30 | 보통 |
| Matcha-TTS | 2023 | Flow matching | 4 스텝 | MOS 4.26 | 중간 |
| ReFlow-TTS | 2024 | Rectified flow | 20 스텝 | MOS 4.45 | 중간 |
| DiTAR | 2025 | LM + DiT 하이브리드 | 가변 | MOS 4.52+ | **매우 우수** |

#### 8.2 상세 비교 분석

**A. DDIM (Denoising Diffusion Implicit Models, Song et al. 2021)**[2][3]

**개념**: 비-마르코프 생성 과정으로 시간 스케줄 선택

$$p^{(\tau)}_\theta(x_0:T) := \pi(x_T) \prod_{i=1}^{S} p^{(\gamma_i)}_\theta(x_{\gamma_i-1}|x_{\gamma_i})$$

**비교**:
- 강점: 간단한 구현, 속도 향상
- 한계: 스코어 네트워크와의 불일치 해결 안 함
- 성능: BDDM 대비 12 스텝에서 4.16 vs 4.48 MOS

**B. FastDiff (Fast Conditional Diffusion Model, Yang et al. 2022)**[4]

**개념**: 시간 인식 위치 변수 합성곱(time-aware location-variable convolutions)으로 효율적 모델링

**비교**:
- 강점: 4 스텝으로 매우 빠름, MOS 4.28
- 한계: 전문화된 아키텍처 필요, 다른 모델 전이 어려움
- 속도: RTF 미보고 (추정 0.1-0.2)

**논문에서의 위치**: BDDM은 FastDiff와 동일 속도(7 스텝)이나 **더 높은 품질(MOS 4.43 vs 4.28)**

**C. LinDiff (Boosting Fast and High-Quality Speech Synthesis with Linear Diffusion, Lin et al. 2023)**[5]

**개념**: ODE 기반 선형 보간으로 곡선 경로 대신 직선 경로 사용

$$x_s = (1-s)\epsilon + s \cdot x_0, \quad s \in $$[1]

**비교**:
- 강점: 수학적으로 우아함, 수렴 증명 가능
- 한계: 20 스텝 필요 (BDDM은 7)
- 성능: MOS 4.30 (BDDM 4.43 미만)

**개선점**: BDDM의 데이터 의존적 스케줄 vs LinDiff의 고정 선형 스케줄

**D. Matcha-TTS (A fast TTS architecture with conditional flow matching, Mehta et al. 2023)**[6]

**개념**: 최적 수송(Optimal Transport) 기반 Flow Matching

$$\text{OT-CFM}: \text{noise} \to \text{speech}$$

**비교**:
- 강점: 4 스텝으로 매우 빠름, 경량 아키텍처
- 한계: Score matching과 다른 훈련 패러다임
- MOS: 4.26 (BDDM 4.43 하)

**논문에서의 위치**: 다른 생성 모델 계열(Flow vs Diffusion)

**E. ReFlow-TTS (ReFlow-TTS: A Rectified Flow Model for High-fidelity Text-to-Speech, Tan et al. 2024)**[7]

**개념**: Rectified flow - ODE 기반 직선 경로

$$\frac{dx}{dt} = \text{직선 경로의 벡터 장}$$

**비교**:
- 강점: 수학적 우아성, MOS 4.45
- 한계: 20 스텝 필요
- 일반화: BDDM보다 제한적

**F. DMOSpeech (Direct Metric Optimization via Distilled Diffusion Model in Zero-Shot Speech Synthesis, 2025)**[8]

**개념**: MOS 직접 최적화 + 증류(Distillation)

**비교**:
- 강점: MOS, PESQ, 화자 유사도 동시 최적화
- 최신 기술: 증류로 추론 시간 대폭 감소
- 성능: MOS 4.5+

**평가**: BDDM과 직교하는 접근 (스케줄 학습 vs MOS 직접 최적화)

**G. DiTAR (Diffusion Transformer Autoregressive Modeling for Zero-Shot Speech Synthesis, 2025)**[9]

**개념**: 언어 모델 + Diffusion Transformer 하이브리드

**비교**:
- 강점: **SOTA 성능(MOS 4.52+)**, 계산 효율성
- 아키텍처: Transformer 기반 (RNN/CNN 아님)
- 일반화: 영화보지 않은 화자에 대해 우수
- 속도: 가변 (패치별 생성)

**결론**: DiTAR은 **새로운 패러다임**(하이브리드 아키텍처)으로 BDDM 초과

#### 8.3 기술적 비교표

| 특성 | BDDM | FastDiff | LinDiff | Matcha | ReFlow | DiTAR |
|------|------|----------|---------|--------|--------|-------|
| 모델 계열 | Diffusion | Diffusion | Diffusion | Flow | Flow | 하이브리드 |
| 스텝 수 | 7 | 4 | 20 | 4 | 20 | 가변 |
| MOS | 4.43 | 4.28 | 4.30 | 4.26 | 4.45 | 4.52+ |
| 데이터 의존성 | **예** | 아니오 | 아니오 | 예 | 아니오 | **높음** |
| 스코어 재사용 | **예** | 아니오 | 아니오 | 아니오 | 아니오 | 아니오 |
| 일반화 | **우수** | 보통 | 보통 | 중간 | 중간 | **매우 우수** |
| 구현 복잡도 | 낮음 | 높음 | 중간 | 중간 | 중간 | 높음 |
| 2020+ 발표 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 8.4 시간 흐름에 따른 진화

```
2020: DDPM (Ho et al.)
      ↓ 느린 샘플링(1000 스텝)
      
2021: DDIM (Song et al.) - 시간 스케줄 선택
      ├─ 성능: 낮음 (비-최적화)
      └─ 문제: 스코어 네트워크와 불일치
      
2022: [분기점]
      ├─ BDDM (Lam et al.) ← "스케줄 네트워크 학습"
      │  └─ 성능: MOS 4.43 (7 스텝)
      │
      ├─ FastDiff (Yang et al.) ← "아키텍처 최적화"
      │  └─ 성능: MOS 4.28 (4 스텝)
      │
      └─ Matcha-TTS (Mehta et al.) ← "Flow Matching"
         └─ 성능: MOS 4.26 (4 스텝)
      
2023: LinDiff, 개선된 DDPM, NaturalSpeech 등
      └─ 다양한 가속화 기법 수렴
      
2024: DMOSpeech (MOS 직접 최적화 + 증류)
      └─ 성능: MOS 4.5+ (초고속)
      
2025: DiTAR (LM + DiT 하이브리드)
      └─ 성능: SOTA MOS 4.52+ (새로운 패러다임)
```

#### 8.5 BDDM의 고유한 기여

**비교 분석 결론**:

1. **이론적 우아성**: Proposition 2-3로 스코어/스케줄 간 관계 증명
   - 다른 논문: 경험적 결과만 제시

2. **사전학습 재사용성**: 기존 모든 DPM 호환
   - FastDiff: 처음부터 훈련 필요
   - LinDiff: 새로운 훈련 필요

3. **일관된 성능**: 3-12 스텝 전 범위에서 안정적
   - FastDiff: 4 스텝 특화
   - LinDiff: 20 스텝 특화

4. **데이터 의존적 스케줄**: σ_φ(x_t)로 현재 신호 조건화
   - DDIM: 고정 스케줄
   - LinDiff: 고정 선형 보간

**그러나 한계**:
- 최신(2025) DiTAR에 의해 초과됨 (MOS 4.52+ vs 4.43)
- 새로운 아키텍처 패러다임(하이브리드) 미제시

***

### 9. 앞으로의 연구에 미치는 영향

#### 9.1 직접적 영향

**1. 새로운 스케줄 학습 패러다임 확립**[1]

BDDM 이전:
- 스케줄은 **사람이 손으로 설계**
- 데이터셋마다 **다시 튜닝** 필요
- **비재현성 문제**

BDDM 이후:
- 스케줄을 **학습 가능한 매개변수**로 취급
- **자동 최적화** 가능
- 후속 연구의 **표준 기법**으로 채택

**증거**: 2024-2025 다수 논문에서 "learned schedule" 또는 "adaptive schedule" 채택

**2. 확산 모델 가속화의 방향 전환**[1]

```
이전 접근: 시간 스케줄 선택 (DDIM)
         → 체계적이지 못함

BDDM: 스케줄을 목적함수로 최적화
     → 이론적 근거 제시

이후: MOS 직접 최적화, Transformer 기반 등으로 발전
```

#### 9.2 산업적 영향

**음성 합성 응용의 실용화**

BDDM 이전:
- DiffWave: 7.30 RTF → **실시간 불가**
- WaveGrad: 38.2 RTF → **완전히 부실용**

BDDM 후:
- 7 스텝: 0.256 RTF → **28배 빠른 실시간 음성 합성**
- 스마트폰, 엣지 디바이스 배포 가능

**현재(2025) 영향**:
- 음성 합성 상용 제품에서 확산 모델 채택 증가
- 실시간 성능이 가능해짐

#### 9.3 이론적 영향

**1. 양측 모델링(Bilateral Modeling) 개념**

BDDM의 핵심:
$$\text{학습 스케줄}(\beta) \neq \text{샘플링 스케줄}(\hat{\beta})$$

영향:
- **비표준 확산 과정** 연구의 출발점
- 향후 "비마스킹 스케줄", "적응형 스케줄" 등의 이론적 기반

**2. 접합 변수(Junctional Variable)**

BDDM의 x_t:
$$x_t = \alpha_t x_0 + \sqrt{1-\alpha_t^2}\epsilon_n$$

이는:
- **확산 과정의 중간 상태**를 명시적으로 모델링
- 이후 "인터폴레이션 기반 스케줄링" 연구로 확장

#### 9.4 관련 분야로의 확산

**1. 이미지 생성으로의 확장**

BDDM이 CIFAR-10에서 SOTA FID 2.38 달성:
- "Diffusion 모델도 속도와 품질을 동시에 달성 가능"한 증명
- 이후 이미지 생성 가속화 연구 활발화

**2. 다양한 모달리티로의 응용**

BDDM의 메커니즘:
- 음성, 이미지 모두에서 동작
- 비디오, 3D, 텍스트 생성으로 확장 가능

실제 활용:
- Video diffusion models (2023+)
- 3D shape generation (2024+)

***

### 10. 앞으로 연구 시 고려할 점

#### 10.1 이론적 문제점 해결

**문제 1: Junctional Variable의 정당성**

$$x_t = \alpha_t x_0 + \sqrt{1-\alpha_t^2}\epsilon_n \sim q_\beta(x_t|x_0)$$

현재: 훈련 중에만 정확, 추론 중 근사

**개선 방향**:
```python
# 현재 (근사):
t ~ U{(n-1)τ, ..., nτ}
x_t = α_t x_0 + √(1-α_t²) ε_n

# 개선안 1: 정확한 매칭
x_t_actual = α_{t*} x_0 + √(1-α_{t*}²) ε_n  (여기서 t*는 추론에서 관측된 x_n으로부터 역산)

# 개선안 2: 사전분포 학습
q_β(x_t|x_0) 대신 실제 추론 분포 p_θ(x_t|observed)로 근사
```

**문제 2: 수렴성 보장**

```math
L^{(n)}_{\text{step}}(\phi; \theta^*) = \frac{\delta_t}{2(\delta_t - \hat{\beta}_n(\phi))}\left\|\epsilon_n - \frac{\hat{\beta}_n(\phi)}{\delta_t}\epsilon_{\theta^*}\right\|^2_2 + C
```

분모 $(\delta_t - \hat{\beta}_n(\phi))$가 0에 가까워질 위험

**개선**:
```python
# 현재: β_n의 상한만 제약
0 < β_n < min{(1-α_{n+1}²)/(1-β_{n+1}), β_{n+1}}

# 개선: 하한도 추가
δ_min < δ_t - β_n < δ_max

# 구현: Constraint 추가
β_n = min{...} × σ_φ(x_t)
      + clip_lower_bound(σ_φ(x_t))  # 안정성 보장
```

#### 10.2 실용적 개선 방향

**1. 자동 초기화**

현재: 그리드 탐색으로 (α̂_N, β̂_N) 선택
비용: O(M²), M=9일 때 81번의 평가

**개선 방안**:
```
메타 학습 기반:
  - 데이터셋 특성 (분산, 길이, 화자 수)
  - 모델 특성 (채널 수, 깊이)
  → α̂_N, β̂_N 직접 예측

신경망:
  Features: [data_variance, seq_len, num_speakers, 
             num_channels, depth, ...]
  Output: [α̂_N, β̂_N]
  
이득: O(1) 초기화 + 1회 순방향 통과
```

**2. 동적 스텝 수 결정**

현재: N(최대 스텝 수) 고정

**개선**:
```python
# Anytime sampling: 조기 종료 기준 강화
for n in range(N, 0, -1):
    x_{n-1} ~ p_θ(x_{n-1}|x_n)
    
    # 품질 평가
    quality = assess_quality(x_{n-1}, condition)
    
    # 조기 종료 조건
    if quality > threshold_n:
        return x_{n-1}  # 불필요한 스텝 생략

# 장점: 평균 스텝 수 추가 감소 (예: 7→5)
```

**3. 아키텍처 독립적 스케줄 네트워크**

현재: GALR 네트워크 (음성) / VGG11 (이미지)

**개선**:
```python
# 제안: 스케줄 네트워크를 "범용 모듈"로 구현
class UniversalScheduleNet(nn.Module):
    def __init__(self, input_dim, task="speech"):
        # 입력 임베딩
        self.embedding = create_embedding(input_dim, task)
        
        # 공유 backbone (Transformer)
        self.backbone = TransformerBlock(...)
        
        # 출력 레이어
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_t):
        embedded = self.embedding(x_t)
        features = self.backbone(embedded)
        return self.head(features)

# 이득:
# - 음성, 이미지, 비디오 등 통일된 아키텍처
# - 전이 학습 가능
# - 일반화 성능 향상
```

#### 10.3 이론 및 실증 연구 방향

**1. 스케줄의 최적성 특성화**

```math
\hat{\beta}^*_n = \arg\min_{\hat{\beta}_n} L^{(n)}_{\text{step}}(\hat{\beta}_n; \theta^*)
```

미해결 질문:
- 최적 스케줄의 형태는? (선형? 로그? 지수?)
- 스케줄이 데이터에 어떻게 적응하는가?

**연구 방법**:
```python
# 1단계: 최적 스케줄 역산
optimal_schedule = optimize_schedule(θ*, data)

# 2단계: 패턴 분석
for dataset in [LJSpeech, VCTK, CIFAR10]:
    sched = optimal_schedule[dataset]
    plot_schedule_shape(sched)
    → 공통 패턴 발견?

# 3단계: 함수 근사
β̂_n = f(n; α_t, x_t, data_statistics)

# 결과: 스케줄의 필요충분조건 규명
```

**2. 스코어 네트워크-스케줄 네트워크 간 상호작용**

질문:
- θ*가 주어졌을 때, φ*의 형태는?
- 반대로 φ*를 먼저 고정하면 θ*는?

**연구**:

$$\mathcal{L}(\theta, \phi) = L^{(n)}_{\text{score}}(\theta) + L^{(n)}_{\text{step}}(\phi; \theta)$$

동시 최적화의 안정성/수렴성 분석

**3. 배포 외 시나리오 성능**

현재: 훈련 분포의 데이터로만 평가

**개선**:
```python
# Out-of-distribution (OOD) 테스트
test_scenarios = {
    "다른 스피커": unseen_speakers,
    "다른 언어": non_english,
    "낮은 음질": noise_added_speech,
    "극한 감정": highly_expressive,
}

for scenario, data in test_scenarios.items():
    mos = evaluate(BDDM, data)
    print(f"{scenario}: MOS={mos}")
```

#### 10.4 응용 분야 확장

**1. 실시간 음성 합성 시스템**

BDDM의 빠른 속도를 활용:
```python
# 온라인 스트리밍 음성 합성
class StreamingSpeechSynthesis:
    def __init__(self):
        self.score_net = load_pretrained_score_net()
        self.schedule_net = load_pretrained_schedule_net()
    
    def process_chunk(self, mel_chunk):
        # 청크 단위로 음성 생성
        x_n = torch.randn_like(mel_chunk)
        
        for step in noise_schedule:
            x_n = self.score_net.denoise(x_n, mel_chunk)
        
        return x_n
    
    # 연속 입력에 대한 완전 스트리밍
    def streaming_synthesis(self, mel_stream):
        buffer = []
        for mel_chunk in mel_stream:
            audio_chunk = self.process_chunk(mel_chunk)
            yield audio_chunk

# 응용: 실시간 가상 어시스턴트, 라이브 스트리밍
```

**2. 멀티모달 확산 모델**

BDDM 메커니즘을 다른 모달리티로 확장:
```python
class MultiModalBDDM:
    """음성 + 비디오, 텍스트 + 이미지 등"""
    
    def __init__(self, modalities=["speech", "text"]):
        self.score_nets = {
            mod: load_score_net(mod) 
            for mod in modalities
        }
        self.schedule_net = UniversalScheduleNet()
    
    def joint_denoise(self, x_noisy, x_cond):
        for step in self.schedule_net.schedule:
            for modality in self.modalities:
                x_noisy[modality] = self.score_nets[modality].denoise(
                    x_noisy[modality],
                    x_cond
                )
        return x_noisy
```

**3. 에너지 효율적 음성 합성**

모바일/엣지 디바이스용:
```python
# 양자화(Quantization) + BDDM
class QuantizedBDDM:
    def __init__(self):
        # INT8 양자화 모델
        self.score_net_int8 = quantize(load_score_net(), bits=8)
        self.schedule_net_int8 = quantize(load_schedule_net(), bits=8)
    
    # 에너지 소비: 기존 대비 60% 감소 예상
```

***

### 11. 결론 및 종합 평가

#### 11.1 BDDM의 위치

**음성 합성 기술 진화에서의 위치**:

```
2015-2020: 자기회귀(WaveNet), GAN 기반 음성 합성
           강점: 높은 품질
           약점: 속도 (WaveNet 느림, GAN 불안정)

2021: DDPM 기반 확산 모델 (DiffWave, WaveGrad)
      강점: 안정적 학습, 높은 품질
      약점: 매우 느림 (RTF >1)

2022: BDDM ← [혁신]
      강점: 속도 + 품질 동시 달성 (RTF 0.256, MOS 4.43)
      방법: 스케줄 네트워크 학습

2023-2025: 확산 기반 다양한 가속화 기법
           - MOS 직접 최적화 (DMOSpeech)
           - 하이브리드 아키텍처 (DiTAR)
           - Flow 기반 방법 (Matcha, ReFlow)
```

**평가**:
- BDDM은 **2022년 혁신** (스케줄 학습 제시)
- 2025년 현재, 더 높은 성능의 방법 존재 (DiTAR: 4.52)
- 그러나 **이론적 우아성**과 **일반화성**에서 중요한 기여

#### 11.2 핵심 기여의 장기적 영향

| 측면 | 기여 | 현재 상황(2025) | 미래 전망 |
|------|------|----------------|----------|
| 이론 | Proposition 2-3로 스케줄 학습 정당화 | 표준 기법 채택됨 | 심화 연구 계속 |
| 실용 | 7 스텝으로 실시간 음성 합성 달성 | 상용화 진행 중 | 모바일/엣지 표준화 |
| 방법론 | 스케줄 네트워크(schedule network) | 다양한 모델에 채택 | 범용 가속화 기법 |
| 일반화 | 기존 모든 DPM 모델 호환성 | 큰 활용 가치 | 새 아키텍처와 결합 |

#### 11.3 최종 평가

**강점**:
1. 명확한 이론적 기초 (Proposition 기반)
2. 실용적 성능 (7 스텝, RTF 0.256)
3. 우수한 일반화 (단일/다중화자, 이미지)
4. 구현 단순성 (기존 모델 활용)

**약점**:
1. 초기 하이퍼파라미터 튜닝 필요 (α̂_N, β̂_N)
2. 절대 성능은 DiTAR 등에 못 미침 (4.43 vs 4.52)
3. 새로운 모달리티에 대한 검증 부족

**중장기 영향**:
- **패러다임 기여**: 스케줄을 학습 가능하게 한 최초의 체계적 시도
- **생산성 기여**: 확산 모델의 실시간 음성 합성을 가능하게 함
- **이론적 토대**: 향후 "적응형 샘플링" 연구의 기초 제공

#### 11.4 추천 후속 연구

**단기(1-2년)**:
1. BDDM과 MOS 직접 최적화 결합 → 성능 향상
2. Transformer 기반 스케줄 네트워크 개발
3. OOD 데이터셋에서 강건성 연구

**중기(3-5년)**:
1. 메타 학습으로 초기값 자동 결정
2. 멀티모달 확산 모델로 확장
3. 에너지 효율적 음성 합성 시스템

**장기(5년+)**:
1. BDDM 원리를 기타 생성 모델(정규 흐름, 복원력 있는 확산)에 적용
2. 범용 적응형 샘플링 이론 개발
3. 음성 합성에서 최종 표준 기법으로 정착

***

### 참고문헌 표기

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/47cdeeb3-0bde-4d1e-b143-29cd41286f3b/2203.13508v1.pdf)
[2](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[3](https://arxiv.org/abs/2403.01633)
[4](https://arxiv.org/pdf/2204.09934.pdf)
[5](https://arxiv.org/pdf/2306.05708.pdf)
[6](https://arxiv.org/pdf/2309.03199.pdf)
[7](http://arxiv.org/pdf/2309.17056.pdf)
[8](http://arxiv.org/pdf/2410.11097.pdf)
[9](https://arxiv.org/html/2502.03930v2)
[10](https://dl.acm.org/doi/10.1145/3707292.3707367)
[11](https://dl.acm.org/doi/10.1145/3587423.3595503)
[12](https://ieeexplore.ieee.org/document/10389779/)
[13](https://nbpublish.com/library_read_article.php?id=71827)
[14](https://www.isca-archive.org/blizzard_2023/chen23_blizzard.html)
[15](https://dl.acm.org/doi/10.1145/3610661.3616556)
[16](https://dl.acm.org/doi/10.1145/3577190.3616117)
[17](https://iopscience.iop.org/article/10.1149/MA2024-019874mtgabs)
[18](https://aclanthology.org/2023.findings-acl.437.pdf)
[19](http://arxiv.org/pdf/2211.09383.pdf)
[20](https://aclanthology.org/2023.emnlp-main.709.pdf)
[21](http://arxiv.org/pdf/2309.15512.pdf)
[22](https://lightrains.com/blogs/comprehensive-guide-audio-diffusion-models/)
[23](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
[24](https://www.isca-archive.org/interspeech_2022/kanagawa22_interspeech.pdf)
[25](https://www.isca-archive.org/interspeech_2024/hirschkind24_interspeech.pdf)
[26](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[27](https://www.albany.edu/faculty/mchang2/files/2022-05_ICASSP_Vocoder_Benchmark.pdf)
[28](https://www.isca-archive.org/interspeech_2024/lovelace24_interspeech.pdf)
[29](https://arxiv.org/abs/2006.11239)
[30](https://arxiv.org/pdf/2506.03554.pdf)
[31](https://arxiv.org/html/2410.11097v1)
[32](https://arxiv.org/html/2503.21774v1)
[33](https://arxiv.org/html/2502.18924v2)
[34](https://arxiv.org/abs/2205.12524)
[35](https://arxiv.org/html/2509.18470v2)
[36](https://arxiv.org/html/2310.09469v2)
[37](https://arxiv.org/pdf/2210.01029.pdf)
[38](https://arxiv.org/pdf/2402.12423.pdf)
[39](https://www.ijcai.org/proceedings/2022/0577.pdf)
