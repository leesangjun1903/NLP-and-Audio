
# Hierarchical Diffusion Models for Singing Voice Neural Vocoder

## 1. 핵심 주장 및 주요 기여 (요약)

본 논문의 핵심 주장은 **노래 음성 합성의 복잡성을 해결하기 위해 계층적 구조의 확산 모델(Hierarchical Diffusion Model)이 필수적**이라는 것입니다.[1]

논문의 주요 기여는 다음과 같습니다:

1. **계층적 확산 모델 제안**: 서로 다른 샘플링 레이트에서 독립적으로 작동하는 다중 확산 모델을 통해 음성 신호를 점진적으로 생성[1]
2. **다중 샌더 지원**: 다양한 가수에 대해 높은 품질의 노래 음성을 생성하며 기존 최고 성능 음성 합성 방법(PriorGrad, Parallel WaveGAN)을 능가[1]
3. **계산 효율성**: 유사한 수준의 계산 비용으로 우수한 성능을 달성하는 실용적인 구조[1]

***

## 2. 상세 설명: 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 문제 정의

기존의 신경망 음성 합성기(Neural Vocoder)는 음성(Speech) 신호에는 우수한 성능을 보이지만, **노래 음성(Singing Voice)에 적용할 때는 다음과 같은 문제점이 발생**합니다:[1]

- **표현의 다양성**: 음악적 표현으로 인한 음높이, 음량, 발음의 광범위한 변동
- **비브라토(Vibrato)와 팔세토(Falsetto)**: 고도의 음악 기법을 재현하는 어려움
- **데이터 부족**: 대규모의 고품질 노래 음성 데이터셋의 희소성
- **음높이 정확성**: 저주파 성분의 정확한 생성 부재

특히, **Parallel WaveGAN 같은 기존 방법들은 비브라토가 있는 원본 노래에서 부자연스러운 음높이 흔들림을 생성**하는 것으로 관찰되었습니다.[1]

### 2.2 제안하는 방법 및 수식

#### 2.2.1 기본 아이디어: 계층적 구조

논문의 핵심은 **서로 다른 샘플링 레이트 $$f_s^1 > f_s^2 > \cdots > f_s^N$$에서 독립적으로 확산 모델들을 학습**하는 것입니다. 각 계층의 역방향 프로세스는 다음과 같이 정의됩니다:[1]

$$p_\theta^i(x_{t-1}^i | x_t^i, c, x_0^{i+1})$$

여기서:
- $$x_t^i$$: 시간 $t$에서 샘플링 레이트 $f_s^i$로의 신호
- $$c$$: 멜-스펙트로그램 조건
- $$x_0^{i+1}$$: 더 낮은 샘플링 레이트에서의 생성된 신호
- 가장 낮은 샘플링 레이트($f_s^N$)에서는 음향 특징 $c$에만 조건화[1]

#### 2.2.2 DDPM 기초

**전방 프로세스(Forward Process)**는 데이터 $x_0$를 표준 가우시안으로 점진적으로 변환합니다:

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

여기서:

$$q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

이는 다음과 같이 직접 샘플링할 수 있습니다:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

여기서 $$\alpha_t = 1 - \beta_t$$, $$\bar{\alpha}\_t = \prod_{s=1}^t \alpha_s$$, $$\epsilon \sim \mathcal{N}(0, I)$$[1]

#### 2.2.3 역방향 프로세스(Reverse Process)

$$p(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

$$p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(x_t, t)I)$$

평균과 분산은 다음과 같이 정의됩니다:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

$$\sigma_\theta^2(x_t, t) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

손실 함수는 ELBO를 최대화하며:

$$\text{ELBO} = C - \sum_{t=1}^{T} \kappa_t \mathbb{E}_{x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$$

[1]

#### 2.2.4 PriorGrad를 통한 개선

기존 DDPM의 표준 가우시안 사전(Prior)을 대체하여:

$$\mathcal{N}(0, \Sigma_c)$$

여기서 $$\Sigma_c = \text{diag}[(\sigma_0^2, \ldots, \sigma_L^2)]$$이고 $$\sigma_i^2$$는 멜-스펙트로그램의 i번째 샘플의 정규화된 프레임 수준 에너지입니다.[1]

손실 함수는 다음과 같이 수정됩니다:

$$L = \mathbb{E}_{x_0, \epsilon, t}[||\epsilon - \epsilon_\theta(x_t, c, t)||^2_{\Sigma^{-1}}]$$

여기서 $$||x||^2_{\Sigma^{-1}} = x^\top \Sigma^{-1} x$$[1]

#### 2.2.5 계층적 학습 및 추론

**학습 단계(Algorithm 1)**:
- 각 샘플링 레이트에서 독립적으로 병렬 학습
- 주어진 $x_0^i$에서 다운샘플링: $$x_0^{i+1} = D_i(H_i(x_0^i))$$
  - $H_i(\cdot)$: 안티-앨리어싱 필터
  - $D_i(\cdot)$: 다운샘플링 함수

**추론 단계(Algorithm 2)**:
- 가장 낮은 샘플링 레이트부터 시작하여 점진적으로 상향 생성
- 반복 제거 과정에서 이전 단계의 생성된 신호 활용

#### 2.2.6 안티-앨리어싱 필터의 중요성

**훈련-추론 간극(Train-Inference Gap) 문제**:
- 훈련 중: 그라운드 트루 데이터 사용 $$x_0^{i+1} = D_i(H_i(x_0^i))$$는 안티-앨리어싱 필터로 인해 Nyquist 주파수 근처의 신호 부재
- 추론 중: 생성된 샘플 $$\hat{x}^{i+1}_0$$에는 Nyquist 주파수 부근의 노이즈 포함 가능

이를 해결하기 위해 추론 시 안티-앨리어싱 필터 적용:

$$\hat{\epsilon} = \epsilon_\theta^i(x_t^i, c, H(\hat{x}_0^{i+1}), t)$$[1]

***

### 2.3 모델 구조 (아키텍처)

#### 2.3.1 네트워크 설계

논문은 **DiffWave를 기반**으로 다음과 같은 아키텍처를 채택합니다:[1]

- **구성**: $L$개의 잔여 계층(Residual Layer)으로 구성된 양방향 확대 합성곱(Bidirectional Dilated Convolution)
- **블록 구조**: 계층을 $m$개 블록으로 분할하며, 각 블록은 $l = L/m$개 계층 포함
- **확대 계수**: 각 블록 내 계층의 확대 계수는 $[1, 2, \ldots, 2^{l-1}]$

**하이퍼파라미터**:
- 기존 DiffWave/PriorGrad: $L = 30, l = 10$ (수용장 크기 확대)
- 논문 제안: $L = 24, l = 8$ (다중 해상도의 계산 비용 상승 완화)[1]

#### 2.3.2 수용장(Receptive Field)의 효과

**동일한 아키텍처가 서로 다른 샘플링 레이트에서 변화하는 수용장을 제공**:

- **낮은 샘플링 레이트**: 더 긴 시간 기간을 커버하므로 저주파 성분(음높이)에 집중
- **높은 샘플링 레이트**: 더 짧은 시간 기간만 커버하므로 고주파 세부사항에 집중[1]

이는 **계층적 확산 모델의 의도와 일치**합니다: 모든 모델이 직접 접근하는 낮은 샘플링 레이트의 조건화된 데이터를 Nyquist 주파수 $f_s^{i+1}/2$까지 직접 사용하고, 고주파 변환에만 집중합니다.

#### 2.3.3 조건화 방식

각 계층에서 노이즈 예측 네트워크는 다음과 같이 조건화됩니다:

$$\epsilon_\theta^i(x_t^i, c, x_0^{i+1}, t)$$

노이즈 예측은:
- **저주파 성분**: 조건화된 낮은 샘플링 레이트 데이터 $x_0^{i+1}$로부터 직접 생성
- **고주파 성분**: 음향 특징 $c$(멜-스펙트로그램)를 기반으로 생성[1]

***

### 2.4 성능 향상

#### 2.4.1 주관적 평가 (MOS - Mean Opinion Score)

| 모델 | MOS | 신뢰도 구간 |
|------|------|-----------|
| 그라운드 트루 | 4.66 ± 0.09 | - |
| Parallel WaveGAN (PWG) | 2.15 ± 0.13 | 낮음 |
| PriorGrad | 3.60 ± 0.12 | 중간 |
| HPG-2 (제안 방법) | **3.95 ± 0.13** | 향상됨 |

**85.3%의 평가자가 HPG-2를 PriorGrad보다 선호**했습니다.[1]

#### 2.4.2 객관적 평가 지표

| 지표 | PWG | PriorGrad | PriorGrad-L | HPG-2 | HPG-3 |
|------|-----|----------|-------------|-------|-------|
| **RTF** (낮을수록 좋음) | 0.067 | 0.066 | 0.093 | 0.070 | 0.100 |
| **PMAE** (낮을수록 좋음) | 3.12 | 1.80 | 2.08 | **1.82** | **1.67** |
| **VDE** (낮을수록 좋음) | 5.61 | 3.96 | 3.86 | **3.47** | **3.32** |
| **MR-STFT** (낮을수록 좋음) | 1.09 | 1.34 | 1.38 | **1.13** | **1.07** |
| **MCD** (낮을수록 좋음) | 6.63 | 9.62 | 9.47 | **8.97** | **8.12** |[1]

**주요 지표 의미**:
- **RTF (Real-Time Factor)**: 계산 시간 효율성. 낮을수록 빠름
- **PMAE (Pitch Mean Absolute Error)**: 음높이 정확도
- **VDE (Voicing Decision Error)**: 음성/무음 구분 오류율
- **MR-STFT**: 다중해상도 STFT 오류
- **MCD (Mel Cepstral Distortion)**: 멜 켑스트럼 왜곡

#### 2.4.3 성능 향상의 원인

1. **저주파 성분의 정확성 개선**: 가장 낮은 샘플링 레이트(6kHz)에서 음높이 생성에 집중하여 **PMAE와 VDE 개선**
2. **고주파 아티팩트 감소**: 계층적 구조가 고주파 역음성(Reverse Aliasing) 문제 해결 → **MR-STFT 개선**
3. **모델 용량의 효율적 활용**: 작은 네트워크로도 다중 스케일 모델링 가능[1]

#### 2.4.4 조건화의 효과 분석

**그림 4의 실험**: 조건화 입력을 제거하여 모델의 활용 방식 분석:[1]
- 멜-스펙트로그램 $c$를 0으로 교체: 저주파 성분은 유지되지만 고주파 감소
- 낮은 샘플링 레이트 데이터 $x_0^2$를 0으로 교체: 고주파 성분만 생성
- **결론**: 모델은 저주파 정보를 낮은 샘플링 레이트 데이터에서, 고주파를 멜-스펙트로그램에서 학습

***

### 2.5 모델의 한계

#### 2.5.1 계산 비용

- **HPG-2**: RTF 0.070 (PriorGrad와 유사한 수준, 약 4.5% 증가)
- **HPG-3**: RTF 0.100 (30% 계산 비용 증가)
- 실시간 처리 요구 애플리케이션에서는 여전히 제약[1]

#### 2.5.2 평가 메트릭의 불일치

**MR-STFT와 MCD 메트릭의 한계**:
- Parallel WaveGAN이 더 낮은 MR-STFT와 MCD를 얻었지만, **주관적 평가(MOS)에서는 훨씬 저조**
- 이는 기존 메트릭들이 **노래 음성의 지각 품질을 완벽하게 반영하지 못함**을 시사[1]

#### 2.5.3 데이터셋 제한

실험에 사용된 데이터셋:
- **NUS48E**: 12명 가수, 각 4곡
- **NHSS**: 10명 가수, 각 10곡
- **내부 코퍼스**: 8명 가수, 50-80곡[1]

총 약 **30명 이하의 제한된 가수 수**로 일반화 성능 평가가 제한됨.

#### 2.5.4 기타 한계

1. **가수별 맞춤화 필요성**: 특정 음악 장르나 스타일에 대한 일반화 성능 미불명
2. **노이즈 환경**: 배경음악이 섞인 실제 녹음에 대한 성능 미평가
3. **언어 다양성**: 주로 영어 및 동양 언어에 초점, 타 언어에의 적용성 불명확[1]

***

## 3. 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능 분석

#### 3.1.1 다중 가수 일반화

논문의 실험 설정:[1]
- 3개 데이터셋에서 총 22명의 가수에 대해 평가
- 90%-10% 학습-테스트 분할
- **동일 데이터셋 내 가수에 대한 일반화만 평가**

**현재 한계**:
- 학습에 포함되지 않은 새로운 가수(Zero-shot)에 대한 성능 미평가
- 데이터셋 간 도메인 이동(Domain Shift)에 대한 평가 부재

#### 3.1.2 아키텍처의 일반화 특성

**긍정적 요소**:

1. **다중 해상도 표현**: 계층적 구조 자체가 서로 다른 음악적 특성에 더 강건함[1]
2. **조건화 입력의 다양성**: 멜-스펙트로그램 기반 조건화는 가수-독립적 특성
3. **분리된 저-고주파 모델링**: 일반적인 음향 신호 특성에 기반한 구조[1]

#### 3.1.3 네트워크 용량과 일반화

**PriorGrad-L 실험의 시사점**:[1]
- 기존 PriorGrad의 모델 크기를 증대: $L = 40$, 채널 80배 증가
- **성능 개선 미미** (PMAE 1.80 → 2.08로 오히려 악화)
- **결론**: 단순한 모델 확대보다 **아키텍처 혁신이 더 효과적**임을 시사

***

### 3.2 일반화 성능 향상 전략

#### 3.2.1 도메인 확장 방안

**논문에서 제안된 추후 연구 방향**:[1]
> "Although we focus on singing voices in this work, the proposed method is applicable to any type of audio. Evaluating the proposed method on different types of audio such as speech, music, and environmental sounds will be our future work."

**구체적 개선 방안**:

1. **대규모 멀티라벨 데이터 구축**
   - 더 많은 가수 수 (현재 30명 → 수백-수천 명)
   - 다양한 언어 및 음악 장르
   - 악기 음악과 혼합 데이터[1]

2. **계층적 학습 전략**
   - 일반적 음향 특성을 먼저 학습하고 점진적으로 가수-특화 특성 학습
   - Meta-learning 기법 적용으로 빠른 적응[1]

3. **조건화 방식 고도화**
   - 현재: 멜-스펙트로그램만 사용
   - 개선: 가수 임베딩, 감정 특성, 음악 장르 정보 추가 조건화
   - 이는 2024년 논문들(TCSinger, YingMusic-Singer 등)에서 실현됨[2][3]

#### 3.2.2 최신 관련 연구의 일반화 전략

**PeriodGrad (2024)**:[2]
- 음성 신호의 주기 구조를 명시적으로 모델링
- **음높이 제어 가능성 향상**으로 일반화 개선

**TCSinger (2024)**:[3]
- 멜-스타일 적응 정규화(Mel-Style Adaptive Normalization) 도입
- 제로샷(Zero-shot) 스타일 전이로 **보지 못한 스타일에 대한 일반화 성능 향상**

**RDSinger (2024)**:[4]
- 참조 기반 확산 네트워크로 **생성 과정의 안정성 개선**
- 기존 방법의 24단계에서 동일한 품질 달성[4]

**HiddenSinger (2023)**:[5]
- 신경 오디오 코덱과 잠재 확산 모델 결합
- 저차원 잠재 공간에서의 작동으로 **계산 효율성 및 일반화 향상**[5]

***

### 3.3 현재 모델의 일반화 한계와 개선 방향

| 측면 | 현재 한계 | 개선 전략 | 최신 연구 |
|------|---------|---------|---------|
| **가수 일반화** | 학습 데이터셋 가수만 평가 | 다양한 배경의 가수 데이터 | TCSinger, YingMusic-Singer |
| **음높이 제어** | 고정된 음높이 생성 | 명시적 피치 모듈링 | PeriodGrad, RDSinger |
| **스타일 전이** | 스타일 조건화 없음 | 스타일 토큰/임베딩 | TCSinger, Prompt-Singer |
| **도메인 이동** | 단일 도메인 평가 | 크로스-도메인 전이 학습 | Domain Adaptation 기법 |
| **계산 효율성** | RTF 0.066-0.100 | 잠재 공간 작동 | HiddenSinger, LHQ-SVC |

***

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 학술적 영향

#### 4.1.1 확산 모델의 새로운 패러다임 확립

**계층적 확산 모델의 기여**:

1. **오디오 합성의 새로운 접근 방식**
   - 기존 GAN 기반 방법: 단일 네트워크에서 전체 주파수 범위 동시 생성
   - 제안 방법: 주파수 대역별 분리 모델링으로 **더 나은 품질과 제어성** 실현[1]

2. **이미지 생성의 계층적 모델과의 연결**
   - 논문에서 언급: 이미지 생성에서 cascading generative models의 성공이 오디오 영역으로 확장됨[1]
   - **다모달(Multimodal) 생성 기법의 통일적 관점 제시**

#### 4.1.2 이후 연구의 주요 방향 전환

본 논문 이후 노래 음성 합성 연구의 주요 트렌드[2-5]:

1. **조건화 방식의 정교화**
   - 단순 멜-스펙트로그램 → 멀티모달 조건화
   - 가수 정보, 감정, 음악 장르 등 다양한 조건 통합[3]

2. **효율성 개선**
   - 기존 diffusion: 추론에 많은 단계 필요
   - 개선: Shallow diffusion, Flow-matching, Latent diffusion으로 단계 감소[4][5]

3. **제로샷 학습**
   - 학습 과정 중 보지 못한 가수/스타일에 대한 생성 능력[6][3]

#### 4.1.3 신경망 음성 합성 분야의 표준화

**기술적 표준 제시**:
- 멜-스펙트로그램 기반의 조건화 (이미 사실상 표준화)[1]
- 다중 해상도 모델링의 중요성 강조
- 안티-앨리어싱 처리의 중요성 재확인[1]

***

### 4.2 실무적 응용 가능성

#### 4.2.1 엔터테인먼트 산업

**음악 프로덕션**:
- 자동 배경 보컬 생성
- 아티스트의 음성 클론으로 다국어 곡 제작
- AI 가수 생성 및 캐릭터 음성 개발[1]

**게임 및 메타버스**:
- 실시간 노래 음성 생성으로 인터랙티브 음악 환경 구축
- NPCs의 동적 음성 생성

#### 4.2.2 교육 및 의료

**음악 교육**:
- 초보자 반주 제공
- 다양한 가수의 음성으로 곡 학습[1]

**음성 치료**:
- 음성 회복 프로그램
- 스타일 전이를 통한 심리 치료 응용

#### 4.2.3 접근성 개선

**장애인 지원**:
- 음성 장애인을 위한 개인화된 음성 합성
- 실시간 감정 표현이 가능한 음성 보조 기기[1]

***

### 4.3 향후 연구 시 고려할 사항

#### 4.3.1 기술적 고려사항

**1. 모델 확장성 (Scalability)**

현재 한계:
- 실험은 3개 데이터셋, 22명 가수에 국한
- 더 큰 규모 데이터에서의 성능 미확인

고려사항:
- 수천 시간의 음성 데이터에서의 학습 안정성
- 멀티-가수 학습 시 발생하는 목소리 혼동(Voice Collapse) 문제
- 분산 학습(Distributed Training) 가능성[1]

**2. 계산 효율성**

```
RTF = 0.070 (HPG-2)
즉, 1초 오디오 생성에 70ms 필요 → 실시간 처리 불가
```

개선 방안:[5][4]
- Shallow diffusion: 초기 음성 특징을 사전 생성으로 단계 감소
- Latent diffusion: 저차원 공간에서 작동으로 계산량 30-50% 감소
- 하드웨어 최적화: 양자화(Quantization), 지식 증류(Knowledge Distillation)[5]

**3. 일반화 성능 평가 방법론**

현재:
- 학습 데이터셋 내 가수에 대한 테스트만 수행
- 새로운 도메인(다른 음악 장르, 언어, 배경음악 포함)에 대한 평가 부재[1]

제안:
- 크로스-데이터셋 평가 프로토콜 수립
- 도메인 이동 측정을 위한 신뢰도 있는 지표 개발
- Zero-shot/Few-shot 학습 능력 평가 체계화[6][3]

#### 4.3.2 평가 지표의 개선

**현재 문제점** (논문에서 지적):[1]
> "MR-STFT and MCD may be insufficient to evaluate the perceptual quality of a singing voice"

개선 방안:
- **지각적 손실 함수(Perceptual Loss)**: 사전 학습된 음성 인코더 활용
- **음악 특화 메트릭**: 음높이 정확성, 비브라토 자연성, 감정 표현도 평가[2]
- **객체적-주관적 지표의 상관성 강화**[7]

#### 4.3.3 데이터 관련 고려사항

**데이터 부족 문제**:
- 노래 음성 데이터는 음성 데이터에 비해 매우 제한적
- 라이선스/저작권 문제로 대규모 공개 데이터셋 부재[1]

**해결책**:
- **반자동 데이터 주석**: 음원 분리 기술로 배경음악 제거 후 수집[8]
- **합성 데이터 활용**: TTS 시스템으로 음성 특성이 다양한 학습 데이터 생성
- **전이 학습**: 음성 데이터로 사전 학습 후 노래 음성으로 미세 조정[9]

#### 4.3.4 도메인 적응 전략

**다양한 음악 스타일**:
- 클래식, 팝, 재즈, K-pop 등 장르별 특성의 차이[3]
- 각 장르의 독특한 음성 특성(비브라토 정도, 발음 스타일)

**언어 및 발음**:
- 다국어 노래 음성 생성
- 각 언어의 음성학적 특수성 반영[6]

**해결책**:
- 조건화 입력에 **스타일/장르 토큰** 추가[3]
- **메타 학습**: 새로운 스타일/언어에 빠르게 적응
- **멀티태스크 학습**: 음성, 음악, 멀티모달 데이터 통합[6]

#### 4.3.5 윤리 및 사회적 고려사항

**음성 클론 악용 방지**:
- 허가되지 않은 가수의 음성 복제 가능성
- 딥페이크 악용 위험[10]

**고려사항**:
- 모델 투명성 공개 (오픈소스화)
- 생성된 음성의 명확한 표시(Watermarking)
- 사용자 인증 시스템 도입
- 법적 프레임워크 개발[10]

#### 4.3.6 평가 자동화

**주관적 평가의 비용**:
- MOS 평가에 인간 평가자 20명 필요 (논문)[1]
- 비용 증가 및 재현성 문제

**자동 평가 지표 개발**:
- **신경망 기반 MOS 예측**: 사전 학습된 모델로 품질 자동 평가
- VoiceMOS Challenge 2022-2024 참여 기관들의 연구[7]
- 앙상블 모델의 강력한 일반화 성능 입증[7]

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 확산 모델 기반 신경 음성 합성 기술 발전사

#### 5.1.1 핵심 기술 타임라인

| 연도 | 핵심 기술 | 논문/방법 | 주요 특징 |
|------|---------|---------|---------|
| 2020 | DDPM 기초 | DiffWave | 음성 합성에 확산 모델 처음 적용[1] |
| 2021 | 적응형 사전 | PriorGrad | 데이터 의존 사전으로 효율성 개선[1] |
| 2022 | **계층적 구조** | **Hierarchical Diffusion** | **본 논문: 노래 음성에 특화** |
| 2022 | 스펙트럼 개선 | SpecGrad | 스펙트럼 포장 기반 노이즈[2] |
| 2023 | 잠재 공간 | HiddenSinger | 신경 오디오 코덱과 결합[5] |
| 2023 | 조건화 개선 | Hierarchical VC | 음성 변환에 계층적 구조 적용[2] |
| 2024 | 피치 제어 | PeriodGrad | 주기 신호로 명시적 음높이 제어[2] |
| 2024 | 스타일 전이 | TCSinger, RDSinger | 제로샷 스타일 전이 및 참조 기반 생성[3][4] |
| 2024 | 경량화 | LHQ-SVC | CPU 호환 경량 모델[1] |
| 2025 | 강화 학습 | YingMusic-Singer | RL 기반 후처리로 성능 극대화[3] |

***

#### 5.1.2 방법론 비교

**1. 조건화 방식 진화**

```
DiffWave/PriorGrad (2020-2021)
  ↓ (단순 멜-스펙트로그램)
  ↓
Hierarchical Diffusion (2022, 본 논문)
  ↓ (낮은 샘플링 레이트 신호 + 멜-스펙트로그램)
  ↓
TCSinger/RDSinger (2024)
  ↓ (멀티모달: 스타일 토큰, 참조 신호, 음악적 표현)
  ↓
YingMusic-Singer (2025)
  ↓ (강화 학습 기반 동적 조건화)
```

**2. 효율성 개선 전략**

| 방법 | 추론 단계 | 계산량 | 품질 | 논문 |
|------|---------|--------|------|------|
| DiffWave | 1000+ | 기준 | 높음 | [1] |
| PriorGrad | 6 | 기준 | 높음 | [1] |
| Hierarchical Diffusion | 2×6 | +7% | 향상 | [본 논문][1] |
| HiddenSinger | 50 | -70% | 동등 | [5] |
| RDSinger | 24-100 | -50% | 향상 | [4] |

***

### 5.2 노래 음성 합성의 특화 기술

#### 5.2.1 음높이/피치 처리 진화

**기존 방법의 문제** (Hierarchical Diffusion 시점):[1]
- 비브라토가 있는 노래에서 부자연스러운 음높이 흔들림 (Parallel WaveGAN)
- 음높이 정확도 메트릭 (PMAE) 상대적 높음

**PeriodGrad (2024)의 해결책**:[2]
```
명시적 주기 신호 조건화
  ↓
음높이 커브의 직접 제어
  ↓
PMAE 추가 개선 가능
```

**최신 방법들의 접근**:[4][3][6]
- 기본 음높이(F0) 명시적 추출 및 조건화
- 인접한 음성 프레임과의 연속성 강제
- 멀티-스케일 음높이 표현[3][4]

#### 5.2.2 스타일/감정 표현 기술

**TCSinger (2024)의 기여**:[3]
- 제로샷 스타일 전이: 학습 중 보지 못한 스타일에 대한 생성
- 멜-스타일 적응 정규화: 스타일 정보를 멜-스펙트로그램과 동시에 처리
- 크로스-언어 스타일 전이

**YingMusic-Singer (2025)의 접근**:[3]
```
자동 멜로디 추출 (Online Learning)
  ↓
음악성 평가를 위한 강화 학습
  ↓
MOS 메트릭의 직접 최적화
```

***

### 5.3 계산 효율성 개선

#### 5.3.1 각 방법의 RTF 비교

```
Hierarchical Diffusion (본 논문)
  HPG-2: RTF = 0.070 (실시간 불가, 14배 느림)
  HPG-3: RTF = 0.100 (30% 추가 증가)
  
HiddenSinger (2023)
  실제 구현으로 5배 이상 속도 개선
  신경 오디오 코덱의 저차원 잠재 공간 활용
  
RDSinger (2024)
  기본 설정: 100 단계 (DiffSinger 대비 2배)
  개선: 24 단계로 감소하며 품질 유지
  
LHQ-SVC (2025)
  CPU 호환 경량 모델
  5% 계산량으로 동등 성능
```

**핵심 기술**:
1. **잠재 확산 모델(Latent Diffusion)**: 저차원 공간에서의 작동으로 계산 50-70% 감소
2. **얕은 확산(Shallow Diffusion)**: 사전 생성된 특징에서 출발으로 단계 감소
3. **흐름 매칭(Flow Matching)**: 확산보다 수렴 빠름[4][5]

***

### 5.4 일반화 성능 비교

#### 5.4.1 다양한 도메인에서의 성능

| 방법 | 동일 가수 | 미보는 가수 | 크로스 장르 | 참고 |
|------|---------|----------|----------|------|
| Hierarchical Diffusion | ✓ 우수 | ? 평가됨 | ? 평가 안함 | [본 논문][1] |
| TCSinger | ✓ 우수 | ✓ 우수 | ✓ 우수 | [3] |
| RDSinger | ✓ 향상 | ✓ 우수 | ? 평가 제한 | [4] |
| HiddenSinger | ✓ 우수 | ✓ 평가됨 | ? 평가 제한 | [5] |
| YingMusic-Singer | ✓ 우수 | ✓ 우수 | ✓ 우수 | [3] |

***

### 5.5 아키텍처 혁신 비교

#### 5.5.1 핵심 구조 비교

**Hierarchical Diffusion (본 논문)**:[1]
```
장점:
  ✓ 간단하고 효과적인 다중 스케일 모델링
  ✓ 저주파 성분에 집중된 음높이 생성
  ✓ 기존 사전 학습 모델(PriorGrad) 재활용 가능
  
단점:
  ✗ 계산 비용 증가 (7-30%)
  ✗ 강한 계층 간 의존성으로 오류 누적 위험
```

**HiddenSinger (2023)**:[5]
```
장점:
  ✓ 신경 오디오 코덱으로 저차원 표현
  ✓ 계산 효율성 극대화
  ✓ 음성과 노래 통합 학습
  
단점:
  ✗ 코덱 학습의 추가 계산 필요
  ✗ 보틀넥 최적화의 어려움
```

**TCSinger (2024)**:[3]
```
장점:
  ✓ 멀티-스케일 스타일 제어
  ✓ 제로샷 학습 능력
  ✓ 크로스-언어 전이
  
단점:
  ✗ 복잡한 아키텍처로 학습 어려움
  ✗ 스타일 토큰 최적화 필요
```

***

### 5.6 평가 방법론의 발전

#### 5.6.1 객관적 지표의 한계와 개선

**Hierarchical Diffusion 논문에서의 관찰**:[1]
- MOS와 MR-STFT/MCD의 낮은 상관성
- Parallel WaveGAN이 스펙트럼 메트릭에서 우수하지만 주관적 평가에서 저조

**최신 연구의 개선안**:

1. **신경망 기반 자동 MOS 예측**
   - VoiceMOS Challenges (2022-2024)에서 입증된 방법[7]
   - Self-supervised encoder (wav2vec 2.0) 기반 음질 예측
   - 단일 모델보다 **앙상블로 일반화 성능 향상**[7]

2. **음악 특화 평가 지표**
   - 음높이 정확도: RMSE 대신 센트 단위 오류[2]
   - 리듬 안정성: 박자 오차 측정
   - 감정 표현도: 신경망 기반 감정 분류[3]

3. **크로스-도메인 평가 프로토콜**
   - 동일 데이터셋 내 평가 → 크로스-데이터셋 평가로 확장
   - 음악 장르, 언어, 녹음 환경 다양화[6][4][5]

***

### 5.7 종합 평가 및 전망

#### 5.7.1 Hierarchical Diffusion의 위치

**학술적 의의**:
- **첫 사례**: 다중 샘플링 레이트 기반 계층적 오디오 확산 모델 제시
- **개념 정립**: 오디오 신호의 다중 스케일 성질을 명시적으로 모델링
- **후속 연구 영감**: 이후 연구들이 이를 기반으로 다양한 개선 추진[2][4][5][6][3]

**기술적 위치**:
```
시간축: 2022년 발표 (중기 연구)
  ↓
효율성: 기존 대비 소폭 증가 (+7-30%)
        → 이후 HiddenSinger, RDSinger로 극복
  ↓
일반화: 학습 가수 범위 내 우수
        → 이후 TCSinger, YingMusic-Singer로 확대
  ↓
평가: 기존 메트릭의 한계 지적
        → VoiceMOS 챌린지 등으로 개선 진행 중
```

#### 5.7.2 아직 미해결 과제

**Hierarchical Diffusion 이후 3년간의 진전에도**:

1. **실시간 처리의 어려움**
   - RTF 0.01 이상: 여전히 실시간 불가
   - 엣지 디바이스(모바일, 임베디드) 배포 어려움

2. **극단적 외삽(Extrapolation)**
   - 학습 범위 밖의 음높이 (매우 높은/낮은 음역)
   - 학습 데이터와 크게 다른 스타일

3. **배경음악 혼입**
   - 실제 녹음의 배경음악/노이즈
   - 음원 분리 후 처리 필요 (추가 파이프라인)

4. **멀티-스피커 환경**
   - 합창, 듀엣 등 여러 가수의 음성 동시 처리
   - 가수 간 상호작용 모델링[1]

***

## 6. 결론 및 향후 제언

### 6.1 핵심 발견 요약

**본 논문의 가장 중요한 기여**:

1. **아키텍처 혁신의 한계성 인식**
   - 단순한 모델 크기 증대 (PriorGrad-L)는 성능 향상 미미
   - **새로운 구조가 필요함을 실증적으로 입증**[1]

2. **다중 해상도 모델링의 효과성**
   - 낮은 샘플링 레이트에서의 음높이 정확성 개선 (PMAE ↓)
   - 고주파 아티팩트 감소 (MR-STFT ↓)[1]

3. **평가 지표의 재검토**
   - 기존 스펙트럼 거리 지표와 지각적 품질의 괴리
   - 음악 신호 특성의 복잡성 강조[1]

### 6.2 실무 적용 시 체크리스트

```
✓ 데이터셋 구축
  - 최소 20-30명 이상 가수
  - 다양한 장르 및 표현 기법
  - 품질 검증 프로토콜 수립

✓ 모델 선택
  - 순수 품질 우선: Hierarchical Diffusion + 후처리
  - 효율성 우선: HiddenSinger 또는 RDSinger
  - 제어성 우선: TCSinger 또는 PeriodGrad

✓ 평가 계획
  - 객관적 지표: MOS + 신경망 자동 예측
  - 주관적 평가: 20명 이상 평가자 확보
  - 크로스-도메인 평가 포함

✓ 배포 준비
  - RTF 요구사항 정의 (실시간/배치)
  - 하드웨어 요구사항 (GPU/CPU)
  - 윤리 검토 및 투명성 공개
```

### 6.3 향후 연구 방향

**단기 (1-2년)**:
- 더 큰 규모 데이터에서의 검증
- 도메인 적응 기법 통합
- 자동 평가 지표 개선

**중기 (2-3년)**:
- 엣지 디바이스 최적화
- 멀티-가수 합창 지원
- 실시간 음성 변환 구현

**장기 (3-5년)**:
- 멀티모달 음악 생성 (가사, 악기음 포함)
- 감정 및 표현 세밀 제어
- 완전 자동 음악 제작 파이프라인

***

## 참고문헌 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4ed05081-2874-44fe-89de-8e1875253abf/2210.07508v2.pdf)
[2](https://ieeexplore.ieee.org/document/10448502/)
[3](https://ieeexplore.ieee.org/document/10095749/)
[4](https://ieeexplore.ieee.org/document/9980018/)
[5](https://arxiv.org/abs/2210.11096)
[6](https://arxiv.org/abs/2208.04756)
[7](https://ieeexplore.ieee.org/document/11119540/)
[8](https://arxiv.org/pdf/2210.11096.pdf)
[9](https://arxiv.org/pdf/2105.13871.pdf)
[10](https://arxiv.org/pdf/2210.07508.pdf)
[11](https://arxiv.org/pdf/2310.05118.pdf)
[12](https://arxiv.org/pdf/2406.05692.pdf)
[13](https://arxiv.org/pdf/2306.06814.pdf)
[14](http://arxiv.org/pdf/2409.08583.pdf)
[15](https://arxiv.org/pdf/2402.12660.pdf)
[16](https://ai.sony/publications/Hierarchical-Diffusion-Models-for-Singing-Voice-Neural-Vocoder/)
[17](https://sander.ai/2020/03/24/audio-generation.html)
[18](https://archives.ismir.net/ismir2022/paper/000008.pdf)
[19](https://www.sonyresearchindia.com/hierarchical-diffusion-models-for-singing-voice-neural-vocoder/)
[20](https://aclanthology.org/2024.findings-emnlp.246.pdf)
[21](https://aclanthology.org/2024.emnlp-main.117.pdf)
[22](https://www.sony.com/en/SonyInfo/technology/publications/hierarchical-diffusion-models-for-singing-voice-neural-vocoder/)
[23](https://www.isca-archive.org/interspeech_2023/choi23d_interspeech.pdf)
[24](https://arxiv.org/abs/2403.11780)
[25](https://www.sciencedirect.com/science/article/abs/pii/S0097849324001936)
[26](https://arxiv.org/html/2509.15629v1)
[27](https://arxiv.org/pdf/2410.21641.pdf)
[28](https://arxiv.org/html/2410.21641)
[29](https://arxiv.org/pdf/2511.09090.pdf)
[30](https://arxiv.org/pdf/2405.09940.pdf)
[31](https://arxiv.org/pdf/2509.15629.pdf)
[32](https://arxiv.org/html/2511.14312v1)
[33](https://arxiv.org/pdf/2407.09346.pdf)
[34](https://arxiv.org/pdf/2402.14692.pdf)
[35](https://www.emergentmind.com/topics/diffusion-based-sound-synthesis-model)
[36](https://pubs.aip.org/jasa/article/151/3/2077/2838345/Optimized-design-of-windowed-sinc-anti-aliasing)
[37](https://www.semanticscholar.org/paper/93d19b6eafbb41cfbeaec18b4420441d63e9666b)
[38](https://www.semanticscholar.org/paper/0cdb86d72ab53f066947aa191c1333ee23d94bb1)
[39](http://ieeexplore.ieee.org/document/634128/)
[40](https://www.semanticscholar.org/paper/052443b5dd12a3ec944749e4d9fb567e5d80e8a3)
[41](http://arxiv.org/pdf/2407.04575.pdf)
[42](https://arxiv.org/pdf/1904.05351.pdf)
[43](http://arxiv.org/pdf/2206.04658v1.pdf)
[44](https://arxiv.org/html/2411.11258v1)
[45](http://arxiv.org/pdf/2406.06111.pdf)
[46](http://arxiv.org/pdf/2411.06807.pdf)
[47](http://arxiv.org/pdf/1904.02892.pdf)
[48](http://arxiv.org/pdf/1809.10288.pdf)
[49](https://www.ni.com/en/shop/data-acquisition/measurement-fundamentals/analog-fundamentals/anti-aliasing-filters-and-their-usage-explained.html)
[50](https://www.merl.com/publications/docs/TR2024-014.pdf)
[51](https://archives.ismir.net/ismir2022/paper/000097.pdf)
[52](https://www.sciencedirect.com/science/article/abs/pii/S0003682X22005576)
[53](https://arxiv.org/abs/2106.06406)
[54](https://ece.iisc.ac.in/~spcom/2020/CameraReadySubmissions/316/CameraReadySubmission/SPCOM_Singing_camera_ready.pdf)
[55](https://arxiv.org/html/2411.06807v2)
[56](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/priorgrad/)
[57](https://arxiv.org/html/2312.10741)
[58](https://www.alphaxiv.org/overview/2411.06807v1)
[59](https://www.arxiv.org/pdf/2411.06807.pdf)
[60](https://www.semanticscholar.org/paper/PriorGrad:-Improving-Conditional-Denoising-Models-Lee-Kim/d1a6890bfd0ac2b9777a7190dcd70ac2c08a76e4)
[61](https://www.arxiv.org/pdf/2505.14910.pdf)
[62](https://arxiv.org/pdf/2402.15516.pdf)
[63](https://arxiv.org/html/2501.13870v1)
[64](https://www.arxiv.org/pdf/2406.06111.pdf)
[65](https://arxiv.org/html/2511.22293v1)
[66](https://arxiv.org/html/2512.04779v1)
[67](https://huggingface.co/papers/2106.06406)
[68](http://arxiv.org/pdf/1911.02933.pdf)
[69](https://speechresearch.github.io/priorgrad/)
[70](https://snu.elsevierpure.com/en/publications/priorgrad-improving-conditional-denoising-diffusion-models-with-d)
