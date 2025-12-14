# DiffWave: A Versatile Diffusion Model for Audio Synthesis

### 1. 핵심 주장과 주요 기여도 요약

**DiffWave**는 확산 확률 모델(Diffusion Probabilistic Model)을 활용하여 조건부 및 무조건부 음성 파형 생성을 수행하는 다목적 생성 모델입니다. 이 논문의 핵심 주장은 백색 잡음에서 시작하여 마르코프 체인을 통해 점진적으로 노이즈를 제거함으로써 고충실도 오디오를 합성할 수 있다는 것입니다.

**주요 기여도는 다음과 같습니다:**

1. **WaveNet 대비 동등한 성능**: 신경 보코더 작업에서 WaveNet과 동일한 음질(MOS: 4.44 vs 4.43)을 달성하면서도 병렬 처리로 인해 합성 속도가 수백 배 빠름[1]
2. **효율적인 모델 구조**: 2.64M의 소형 모델이 V100 GPU에서 실시간 음성 합성(5배 이상) 가능[1]
3. **무조건부 생성 획기적 개선**: 자동회귀 및 GAN 기반 모델 대비 훨씬 우수한 무조건부 음성 생성(MOS: 3.39 vs 1.43/2.03)[1]
4. **훈련 안정성**: VAE의 포스터리어 붕괴나 GAN의 모드 붕괴 문제를 자연스럽게 회피하는 구조

***

### 2. 해결하고자 하는 문제

#### 2.1 기존 음성 합성 모델의 한계

**자동회귀 모델(WaveNet)**은 순차적 생성으로 인한 극도로 느린 합성 속도(500배 이상 느림)가 주요 문제입니다. 또한 무조건부 생성 환경에서는 매우 낮은 품질의 "만들어진 단어 같은 소리"를 생성하는 경향이 있습니다.[1]

**GAN 기반 모델**은 훈련 불안정성, 모드 붕괴, 조건부 정보 처리의 어려움이 있으며, 고충실도 합성을 위해 스펙트로그램 손실 등 보조 손실함수가 필요합니다.[1]

**VAE 기반 모델**은 포스터리어 붕괴(posterior collapse) 문제로 인해 정보를 제대로 인코딩하지 못합니다.[1]

**흐름 기반 모델(Flow-based)**은 가역성 유지라는 아키텍처 제약으로 인해 설계 자유도가 제한되고, 더 큰 모델 크기가 필요합니다.[1]

#### 2.2 특히 도전적인 문제: 무조건부 생성

16,000개 타임스텝(16kHz, 1초)의 음성을 조건 정보 없이 생성해야 하는 무조건부 생성은 다음의 어려움이 있습니다:

- 화자, 음성 속도, 녹음 환경 등 데이터의 모든 변이를 제한된 모델 용량으로 학습해야 함
- 장거리 음성 구조(문장 수준 패턴)를 모델링할 충분한 수용 영역 필요
- 긴 시퀀스에서의 모드 붕괴 위험

***

### 3. 제안하는 방법론과 수식

#### 3.1 확산 프로세스의 수학적 정의

**순전파(Forward) 프로세스**: 데이터 $x_0 \sim q_{data}$에서 시작하여 점진적으로 노이즈 추가

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

여기서 각 단계는:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

$\beta_t$는 작은 양수 상수이며, 충분히 큰 $T$에서 $q(x_T|x_0) \approx \mathcal{N}(0, I)$로 수렴합니다[1].

**역전파(Reverse) 프로세스**: 학습된 신경망 매개변수로 노이즈에서 데이터로 점진적 변환

$$p_\theta(x_{T:0}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

여기서:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \sigma_t^2 I)$$

#### 3.2 변분 하한(ELBO) 기반 훈련

전체 우도는 계산 불가능하므로 변분 하한을 최적화합니다:[1]

$$\mathbb{E}_{q_{data}(x_0)} [\log p(x_0)] \geq \mathbb{E}_{q_{data}(x_0)} [\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]$$

이를 전개하면:

$$\text{ELBO} = -\mathbb{E}_{q} [KL(q(x_T|x_0) \| p(x_T))] - \sum_{t=2}^{T} \mathbb{E}_{q} [KL(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))] - \mathbb{E}_{q} [\log p_\theta(x_0|x_1)]$$

#### 3.3 정규화된 파라미터화

Ho et al. (2020)의 닫힌형 표현을 따르면, 신경망은 노이즈를 직접 예측합니다:[1]

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))$$

$$\sigma_\theta(x_t, t) = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$, $\alpha_t = 1 - \beta_t$입니다.[1]

**명제 1**: 고정된 분산 일정 $\{\beta_t\}\_{t=1}^{T}$에 대해, $\epsilon \sim \mathcal{N}(0, I)$, $x_0 \sim q_{data}$일 때, 위의 파라미터화 하에서:

$$\text{ELBO} = c - \sum_{t=1}^{T} \lambda_t \mathbb{E}_{x_0, \epsilon} [\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2]$$

여기서 $\lambda_t = \frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)\sigma_t^2}$입니다.[1]

#### 3.4 훈련 목표함수

실제로는 가중되지 않은 변형을 사용하여 더 나은 생성 품질 달성:[1]

$$\mathcal{L}_{unweighted} = \mathbb{E}_{x_0, \epsilon, t} [\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2]$$

여기서 $t \sim \text{Uniform}(1, T)$이고, 이는 모든 시간 단계에 동등한 가중치를 부여합니다.[1]

#### 3.5 빠른 샘플링 알고리즘

훈련 시 $T = 200$ 단계가 필요하지만, 추론 시에는 $T_{infer} = 6$ 같은 훨씬 적은 단계로도 고품질 생성 가능합니다. 핵심은 노이즈 레벨 정렬을 통한 분산 스케줄 재설계입니다:[1]

$$\sigma_{align}(s) = \text{interpolate between training noise levels}$$

***

### 4. 모델 구조

#### 4.1 전체 아키텍처

DiffWave의 신경망 $\epsilon_\theta(x_t, t)$는 다음과 같이 구성됩니다:

- **입력층**: 잠재 변수 $x_T \sim \mathcal{N}(0, I)$
- **처리 모듈**: 양방향 확장된 합성곱(Bidirectional Dilated Convolution) 기반 잔차 층 30개
- **출력층**: 예측 노이즈 $\epsilon_\theta(x_t, t)$

#### 4.2 핵심 컴포넌트

**확산 단계 임베딩**: 정현파 위치 인코딩(sinusoidal positional encoding)을 사용하여 확산 시간 단계 $t$를 인코딩합니다:[1]

$$\text{embedding}_t = [\sin(10^{-4 \cdot 0/63}t), ..., \sin(10^{-4 \cdot 63/63}t), \cos(10^{-4 \cdot 0/63}t), ..., \cos(10^{-4 \cdot 63/63}t)]$$

이 128차원 벡터는 3개의 완전연결층을 통과한 후 각 잔차 층에 브로드캐스트됩니다.[1]

**양방향 확장된 합성곱(Bi-DilConv)**: WaveNet과 달리 인과적(causal) 제약이 없어 양방향 처리가 가능합니다. 각 블록 내에서 확장도(dilation)는 1, 2, 4, ..., 512로 기하급수적으로 증가합니다.[1]

**조건부 입력 통합**:
- **로컬 조건자(Mel-spectrogram)**: 256배 업샘플링 후 각 잔차 층의 확장된 합성곱 직전에 편향으로 추가
- **글로벌 조건자(클래스 라벨)**: 128차원 임베딩으로 인코딩 후 2C 채널로 매핑하여 편향 추가[1]

**무조건부 생성을 위한 수용 영역 확대**: 

30층 확장된 합성곱의 단일 순전파로는 수용 영역이 약 6,139 샘플(0.38초)에 불과합니다. 이를 해결하기 위해 DiffWave는 역전파 프로세스를 T 단계로 반복하여 **유효 수용 영역을 $T \times r$로 확대**합니다. 이는 긴 구조를 학습할 수 있게 해줍니다.[1]

***

### 5. 성능 향상 및 한계

#### 5.1 신경 보코더 성능

표 1: 신경 보코더 성능 비교(LJ Speech 데이터셋)[1]

| 모델 | 파라미터 | MOS | 합성 속도(V100 기준) |
|------|---------|-----|------------------|
| WaveNet | 4.57M | 4.43±0.10 | 0.002× (500배 느림) |
| DiffWave LARGE | 6.91M | 4.44±0.07 | 3.5× 빠름 |
| DiffWave BASE | 2.64M | 4.35±0.10 | 1.1× 빠름 |
| DiffWave LARGE Fast | 6.91M | 4.42±0.09 | 3.5× 빠름 |
| WaveGlow | 87.88M | 4.33±0.12 | 40× 빠름 |
| WaveFlow | 5.91M | 4.30±0.11 | - |

**주요 결과**: DiffWave LARGE는 WaveNet과 동등한 음질(4.44)을 달성하면서도 비자동회귀 구조로 훨씬 빠른 합성이 가능합니다. 작은 모델도 여전히 4.35의 높은 MOS를 유지합니다.[1]

#### 5.2 무조건부 생성 성능

표 2: 무조건부 생성 자동 평가 지표(SC09 데이터셋)[1]

| 지표 | DiffWave | WaveNet-256 | WaveGAN |
|------|----------|------------|---------|
| FID (낮을수록 좋음) | 1.287 | 2.947 | 1.349 |
| IS (높을수록 좋음) | 5.30 | 2.84 | 4.53 |
| mIS (높을수록 좋음) | 59.4 | 10.0 | 36.6 |
| MOS (높을수록 좋음) | 3.39±0.32 | 1.43±0.30 | 2.03±0.33 |

**획기적 개선**: 무조건부 음성 생성에서 WaveNet은 1.43의 매우 낮은 MOS를 보이지만, DiffWave는 3.39로 **2.4배 향상**되었습니다. 이는 자동회귀 모델의 근본적 한계를 극복했음을 의미합니다.[1]

#### 5.3 클래스 조건부 생성 성능

표 3: 클래스 조건부 생성(숫자 0-9 음성)[1]

| 모델 | 정확도 | FID-class | IS | mIS | MOS |
|------|--------|-----------|-----|-----|-----|
| DiffWave | 91.20% | 1.113 | 30.569 | 6.63 | 3.50±0.31 |
| DiffWave(심층 얇음) | 94.00% | 0.932 | 20.450 | 6.92 | 3.44±0.36 |
| WaveNet-256 | 60.70% | 6.954 | 2.114 | 3.46 | 1.58±0.36 |

**결과**: DiffWave는 WaveNet 대비 분류 정확도를 **50% 향상**시켰으며, 품질도 월등합니다.[1]

#### 5.4 한계점

**추론 속도 제약**: DiffWave는 V100에서 3.5배 실시간(6 단계)에 불과하지만, WaveFlow는 40배 빠릅니다. 이는 확산 모델의 반복적 샘플링 특성 때문입니다.[1]

**훈련 데이터셋 제한**: LJ Speech(단일 여성 화자, 24시간)와 SC09(영어 숫자만)로만 평가되어, 다국어, 다중 화자 환경에서의 일반화 능력이 미지수입니다.[1]

**긴 시간 길이 모델링**: 무조건부 생성은 1초 음성으로 제한되어, 더 긴 음성 구조 학습 능력이 평가되지 않았습니다.[1]

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재 일반화 특성

**제로샷 음성 디노이징(Zero-shot Speech Denoising)**: 훈련되지 않은 6가지 노이즈 유형(흰 노이즈, 핑크 노이즈, 기타 환경음)에 대해 추론 시 25단계에서 시작하여 성공적 디노이징을 수행합니다. 이는 모델이 음성의 우수한 사전지식을 학습했음을 시사합니다.[1]

**무조건부 생성의 다양성**: 다양한 화자, 음성 속도, 녹음 조건 등 광범위한 변이를 성공적으로 모델링합니다.[1]

#### 6.2 일반화 개선 전략

**다중 데이터셋 사전훈련**: 현재 LJ Speech 단일 데이터셋 대신 다국어, 다중 화자, 다양한 녹음 환경을 포함한 대규모 말뭉치 활용 시 음향 특성의 더 강력한 일반화 가능.[1]

**전이 학습(Transfer Learning)**: 대규모 데이터에서 사전훈련 후 새 화자/언어로 효율적 미세조정. 파라미터 효율적 미세조정(예: LoRA)으로 소량 데이터에서의 빠른 적응 가능.

**메타러닝**: 다양한 작업에 빠르게 적응할 수 있는 학습 알고리즘으로 적은 데이터로도 새로운 음성 특성 학습.

**계층적 조건화**: Mel-spectrogram 외에 화자 임베딩, 감정 정보, 스타일 벡터를 동시에 활용하여 세밀한 제어 가능.

***

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 7.1 패러다임 전환

**확산 모델의 음성 합성 도입**: DiffWave는 확산 모델이 음성 합성 분야에 성공적으로 적용될 수 있음을 최초 입증했습니다. 이후 음성 합성 연구의 주요 방향을 설정했습니다.[1]

#### 7.2 무조건부 생성의 새로운 기준

이전까지는 음성 무조건부 생성이 GAN이나 저품질 자동회귀 모델 중심이었으나, DiffWave의 획기적 성과(MOS 3.39)는 새로운 벤치마크를 설정했습니다.[1]

#### 7.3 후속 연구의 발전

- **WaveGrad (Chen et al., 2020)**: 점수 매칭 기반 대안으로 6단계에서 고품질 달성
- **WaveGrad 2 (Chen et al., 2021)**: 엔드-투-엔드 텍스트-음성 합성으로 확장
- **FastDiff (Huang et al., 2022)**: 위치별 가변 합성곱으로 4단계 추론 달성
- **LinDiff (Liu et al., 2023)**: 선형 ODE 경로로 1-50단계 감소
- **BridgeVoC (Li et al., 2025)**: Schrodinger Bridge 프레임워크로 최고 성능(MOS 4.47) 달성

#### 7.4 다양한 응용분야로의 확산

- **음악 생성**: MusicLDM(2023), Multi-Track MusicLDM(2024)
- **오디오-비디오 생성**: PAVAS(2025) - 물리 기반 비디오-오디오 합성
- **조정 가능 오디오 생성**: Audio Palette(2025) - Foley 음향 효과 합성

***

### 8. 앞으로 연구 시 고려할 점

#### 8.1 모델 설계

**아키텍처 최적화**: 양방향 확장된 합성곱의 필수성 재검증, Transformer 메커니즘 통합으로 장거리 의존성 개선.[2]

**노이즈 일정 설계**: 고정 선형 일정의 한계를 넘어 입력-적응적 노이즈 스케줄 설계, 작업별 최적 일정 탐색.[3]

**풍부한 조건화**: 화자 임베딩, 감정 정보, 환경 맥락 등 다차원적 조건 통합으로 세밀한 제어 실현.

#### 8.2 효율성 극대화

**추론 단계 감소**: DiffWave의 6단계에서 LinDiff의 1-50단계(평균 10), BridgeVoC의 단일 단계로 진화하고 있습니다.[4]

**모델 압축**: 양자화, 증류, 프루닝을 통해 메모리 사용과 계산량 감소.

**하드웨어 최적화**: CUDA 커널 설계로 실제 실시간 성능 달성.

#### 8.3 일반화 능력

**다양한 데이터 학습**: 다국어, 다중 화자, 다양한 녹음 환경 포함 대규모 데이터셋으로 훈련.[1]

**도메인 간 전이**: 신경 보코더 → 음악 생성 → 오디오-비디오 생성 등으로의 구조적 확장.

**적응형 미세조정**: 새로운 음성 특성에 대한 빠른 적응을 위한 메타러닝 접근.

#### 8.4 평가 방법론

**다차원 평가**: MOS, FAD뿐 아니라 PESQ, STOI, VISQOL 등 다양한 음성 품질 지표 병행.[5]

**도메인별 지표**: 신경 보코더(조건 정렬도), 무조건부 생성(다양성), 다중 화자(화자 특성 보존도) 등 작업 특화 메트릭.

**통계적 신뢰성**: 더 많은 리스너와 A/B 테스트를 통해 MOS의 통계적 유의성 강화.

***

### 9. 2020년 이후 관련 최신 연구 비교 분석

#### 9.1 주요 후속 연구 개요

| 논문/모델 | 출판 | 핵심 기여 | DiffWave 대비 개선 |
|----------|------|---------|------------------|
| **WaveGrad** | 2020 | 점수 매칭, 연속 노이즈 스케줄 | 6단계로 고품질 달성[6] |
| **WaveGrad 2** | 2021 | 엔드-투-엔드 TTS, 텍스트 조건화 | 조건부 mel-spectrogram 제거[7] |
| **FastDiff** | 2022 | 위치별 가변 합성곱 | 200→4 단계 감소[3] |
| **LinDiff** | 2023 | 선형 ODE 경로 | 1-50 단계로 극적 감소[4] |
| **ReFlow-TTS** | 2023 | Rectified Flow | 수학적 간단성과 효율성[8] |
| **BridgeVoC** | 2025 | Schrodinger Bridge | 4단계, MOS 4.47[9] |

**기술 진화 패턴**:
- **2020-2021**: 비자동회귀 확산 모델 기본 설정 (DiffWave, WaveGrad)
- **2021-2022**: 조건화 확대 및 아키텍처 개선
- **2023-2024**: 추론 단계 극적 감소, 멀티모달 조건화
- **2024-2025**: 하이브리드 접근(확산+GAN), 초고효율 모델

#### 9.2 성능 지표 진화

**신경 보코더 MOS 트렌드**:[9][3][4][1]
```
WaveNet (2016):      4.43 ± 0.10
DiffWave (2020):     4.44 ± 0.07 ← 최초 동등 성능
WaveGrad (2020):     4.41 ± 0.08
FastDiff (2022):     4.43 ± 0.09
LinDiff (2023):      4.42 ± 0.10
BridgeVoC (2025):    4.47 ± 0.08 ← 최고 성능 갱신
```

**무조건부 생성 MOS 트렌드**:[1]
```
WaveNet:     1.43 ± 0.30 (극도로 낮음)
DiffWave:    3.39 ± 0.32 (획기적 개선) ← 2.4배 향상
```

**추론 속도 진화**:[9][3][4][1]
```
2020: DiffWave           3.5× (6 단계, 기준선)
2022: FastDiff           6-8× (4 단계)
2023: LinDiff            10-15× (1-50 단계)
2025: BridgeVoC          20-30× (단일 단계 가능)
2025: Flow2GAN           매우 빠름 (1-3 단계, GAN 최적화)
```

#### 9.3 조건화 메커니즘의 진화

**2020-2021**: Mel-spectrogram 직접 조건화 (DiffWave, WaveGrad)[6][1]

**2021-2022**: 텍스트 조건화로 확장, 엔드-투-엔드 학습 (WaveGrad 2, Grad-TTS)[7]

**2023-2024**: 멀티모달 조건화 출현
- PAVAS(2025): 비디오 입력 + 물리 매개변수[10]
- Audio Palette(2025): 텍스트 + 음향 특성(음량, 음정, 음색)[11]

#### 9.4 아키텍처 진화 방향

**양방향 확장된 합성곱** (DiffWave의 설계)은 이후 음성 처리의 표준 설계로 인정되었습니다.[3][4][1]

**Transformer 통합** (2021-2022): WaveGrad 2부터 Transformer 기반 인코더 도입[7]

**Diffusion Transformer (DiT)** (2023-2025): Vision Transformer 개념을 오디오에 적용[12]

**하이브리드 접근** (2024-2025): Flow2GAN은 확산 모델의 안정성과 GAN의 속도를 결합하여 1-3 단계로 고품질 달성[13]

***

### 10. 결론

DiffWave는 **2020년 확산 모델을 음성 합성에 처음 성공적으로 적용한 이정표적 논문**입니다.[1]

**이론적 기여**: 무조건부 음성 생성의 구조적 해결책(수용 영역 반복 확대)을 제시하여, 이전의 낮은 품질(자동회귀 MOS 1.43)을 획기적으로 개선(3.39)했습니다.[1]

**실제적 성취**: WaveNet 수준의 음질(MOS 4.44)을 병렬 처리로 빠르게 달성하면서도 2.64M의 작은 모델 크기 유지.[1]

**방법론적 영향**: 양방향 확장된 합성곱, 정현파 임베딩, 빠른 샘플링 알고리즘 등 설계가 이후 음성 생성 연구의 표준으로 채택.[4][9][3]

**응용 확대**: 신경 보코더에서 음악, 오디오-비디오 생성까지 다양한 응용 분야의 기반 제공.[14][11][10]

**현재 상황(2025년)**: 확산 기반 음성 생성은 DiffWave의 기본 구조를 바탕으로 계속 진화하고 있으며, 추론 단계(200→1), 성능(MOS 4.44→4.47), 조건화 능력 모든 면에서 지속적 개선이 이루어지고 있습니다.[9][3][4]

***

### 참고 문헌 인덱스

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/450ef389-5216-41f5-bb27-780765a90d26/2009.09761v3.pdf)
[2](https://www.emergentmind.com/topics/diffusion-based-sound-synthesis-model)
[3](https://arxiv.org/pdf/2204.09934.pdf)
[4](https://arxiv.org/pdf/2306.05708.pdf)
[5](https://archives.ismir.net/ismir2022/paper/000103.pdf)
[6](https://openreview.net/pdf?id=NsMLjcFaO8O)
[7](https://www.academia.edu/87864710/WaveGrad_2_Iterative_Refinement_for_Text_to_Speech_Synthesis)
[8](http://arxiv.org/pdf/2309.17056.pdf)
[9](https://arxiv.org/abs/2511.07116)
[10](https://www.semanticscholar.org/paper/8fb130175704840fc56d6f2ed1448d014f43364e)
[11](https://arxiv.org/abs/2510.12175)
[12](https://arxiv.org/html/2503.10522v2)
[13](https://openreview.net/pdf/9454bef5222b2f852e41f22717f7a68b5e798e6b.pdf)
[14](https://arxiv.org/html/2409.02845v2)
[15](https://www.semanticscholar.org/paper/34bf13e58c7226d615afead0c0f679432502940e)
[16](https://www.semanticscholar.org/paper/0102eca5d1cba0b65a6a8b68d64278d5980e61bb)
[17](https://itiis.org/digital-library/103081)
[18](https://arxiv.org/abs/2503.12008)
[19](https://arxiv.org/abs/2506.10005)
[20](https://link.springer.com/10.1007/s10462-025-11110-3)
[21](https://royalsocietypublishing.org/doi/10.1098/rsta.2024.0322)
[22](https://arxiv.org/abs/2502.17119)
[23](https://aclanthology.org/2023.findings-acl.437.pdf)
[24](http://arxiv.org/pdf/2407.10471.pdf)
[25](https://arxiv.org/pdf/2311.08667.pdf)
[26](https://arxiv.org/pdf/2409.13894.pdf)
[27](https://arxiv.org/pdf/2211.09707.pdf)
[28](https://arxiv.org/ftp/arxiv/papers/2301/2301.13267.pdf)
[29](https://arxiv.org/html/2410.06544v1)
[30](https://liner.com/review/transfer-learning-for-diffusion-models)
[31](https://www.isca-archive.org/interspeech_2022/koizumi22_interspeech.pdf)
[32](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffwave/)
[33](https://arxiv.org/html/2406.00773v1)
[34](https://arxiv.org/abs/2210.07508)
[35](https://arxiv.org/html/2405.16876v1)
[36](https://randomsampling.tistory.com/47)
[37](https://openreview.net/forum?id=a-xFK8Ymz5J)
[38](https://arxiv.org/html/2302.00646v3)
[39](https://arxiv.org/html/2510.11335v1)
[40](https://arxiv.org/html/2511.13936v1)
[41](https://ar5iv.labs.arxiv.org/html/2206.04658)
[42](https://arxiv.org/pdf/2508.10949.pdf)
[43](https://arxiv.org/html/2412.13933v1)
[44](https://arxiv.org/html/2509.13049v1)
[45](https://arxiv.org/html/2509.11898v1)
[46](https://proceedings.neurips.cc/paper_files/paper/2023/file/f115f619b62833aadc5acb058975b0e6-Paper-Conference.pdf)
[47](https://www.sonyresearchindia.com/hierarchical-diffusion-models-for-singing-voice-neural-vocoder/)
[48](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/keynote_slides.pdf)
[49](https://lans-tts.uantwerpen.be/index.php/LANS-TTS/article/view/763)
[50](http://arxiv.org/pdf/2310.01381.pdf)
[51](http://arxiv.org/pdf/2104.11347.pdf)
[52](https://arxiv.org/pdf/2106.09660.pdf)
[53](http://arxiv.org/pdf/2211.09383.pdf)
[54](http://arxiv.org/pdf/2309.08030.pdf)
[55](https://openreview.net/pdf?id=105yqGIpVW)
[56](https://arxiv.org/html/2401.13249v2)
[57](https://music-audio-ai.tistory.com/9)
[58](https://arxiv.org/html/2506.08457v1)
[59](https://arxiv.org/abs/2106.09660)
[60](https://arxiv.org/html/2509.18470)
[61](https://arxiv.org/html/2509.00051v1)
[62](https://arxiv.org/list/cs/new)
[63](https://arxiv.org/html/2311.01616v2)
[64](https://www.semanticscholar.org/paper/WaveGrad-2:-Iterative-Refinement-for-Text-to-Speech-Chen-Zhang/10ae9a3d1e0874a50820766bd414f98e095cdd8a)
[65](https://arxiv.org/html/2510.09586v1)
[66](https://arxiv.org/html/2508.06842)
[67](https://research.google/pubs/wavegrad-2-iterative-refinement-for-text-to-speech-synthesis/)
[68](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)
[69](https://tik-old.ee.ethz.ch/file/e2394c9aba4e90f07e4f1bd8eedc823e/Music_Generation_Audio_Metrics___ICASSP_2025.pdf)
