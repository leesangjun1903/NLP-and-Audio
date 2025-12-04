# Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech

### 1. 논문의 핵심 주장과 주요 기여

**Grad-TTS**는 Vadim Popov와 그의 팀이 2021년 제시한 혁신적인 텍스트-음성 합성(TTS) 모델로, 확산 확률 모델(Diffusion Probabilistic Models, DPMs)을 음성 특성 생성에 처음으로 적용한 연구입니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**
1. **확산 기반 음향 특성 생성기의 제시** - DPM을 TTS의 피처 생성기로 활용한 첫 사례
2. **일반화된 확산 프레임워크** - 표준 정규분포 $N(0, I)$ 대신 $N(\mu, I)$로부터 데이터를 복원하는 일반화된 순방향/역방향 확산 프로세스 제안
3. **품질-속도 트레이드오프의 명시적 제어** - 추론 단계의 수를 조절하여 음성 합성 품질과 추론 속도의 균형을 유연하게 조절 가능
4. **경쟁력 있는 성능** - 최첨단 모델들과 비교 가능한 Mean Opinion Score(MOS) 달성

***

### 2. 상세 설명: 해결 문제, 제안 방법, 모델 구조, 성능

#### **2.1 해결하고자 하는 문제**

종래의 TTS 시스템은 크게 두 가지 한계를 지니고 있었습니다.[1]

1. **자기회귀 모델의 문제점**: Tacotron2와 WaveNet은 높은 품질을 제공하지만 계산 비효율적이고 어텐션 메커니즘의 실패로 인한 발음 오류가 발생
2. **비자기회귀 모델의 한계**: Glow-TTS는 정규화 흐름(Normalizing Flow)을 기반으로 병렬 생성을 가능하게 했으나, 로봇음성 인상과 단조로운 억양 문제 지속

이전의 확산 모델 기반 음성 합성은 주로 **보코더(vocoder)** 영역에만 적용되어 있었으며(WaveGrad, DiffWave), TTS의 음향 특성 생성 부분에 확산 모델을 적용한 연구는 부재했습니다.[1]

#### **2.2 제안하는 방법 (수식 포함)**

**기본 개념: 확산 확률 모델의 일반화**

Grad-TTS는 확률 미분 방정식(SDE) 프레임워크를 사용하여 기존 확산 모델을 일반화합니다.[1]

**순방향 확산(Forward Diffusion):**

확산 과정이 다음 SDE를 만족한다고 가정합니다:

$$dX_t = -\frac{1}{2}\beta_t(\mu - X_t)\beta_t dt + \beta_t dW_t, \quad t \in [0, T]$$

여기서 $\beta_t$는 노이즈 스케줄, $\mu$는 평균, $\Sigma$는 대각 공분산 행렬입니다.

이 SDE의 해는:

$$X_t = \alpha_{\mu,\Sigma,t} \mu + \alpha^{\mu,\Sigma,t} X_0 + \int_0^t \beta_s \alpha^{\mu,\Sigma,t-s} dW_s$$

여기서:
$$\alpha_{\mu,\Sigma,t} = (I - e^{-\frac{1}{2}\int_0^t \beta_s ds})$$
$$\alpha^{\mu,\Sigma,t} = e^{-\frac{1}{2}\int_0^t \beta_s ds}$$

**조건부 분포:**

$X_0$ 주어졌을 때 $X_t$의 조건부 분포는 가우시안:

$$\mathcal{L}(X_t|X_0) = \mathcal{N}(\mu_{\mu,\Sigma,t}(X_0), \sigma^2_{\mu,\Sigma,t}(X_0) I)$$

무한 시간 지평에서 $X_t \xrightarrow{d} \mathcal{N}(\mu, \Sigma)$로 수렴합니다.[1]

**역방향 확산(Reverse Diffusion):**

Anderson(1982)의 결과에 따라 역방향 동역학은:

$$dX_t = \left[-\frac{1}{2}\beta_t(\mu - X_t) + \beta_t^2 \nabla_{X_t} \log p_t(X_t)\right]dt + \beta_t d\bar{W}_t$$

또는 동치적으로 ODE 형태로:

$$dX_t = \left[-\frac{1}{2}\beta_t(\mu - X_t) + \beta_t^2 \nabla_{X_t} \log p_t(X_t)\right]dt$$

**손실 함수(Loss Function):**

스코어 매칭(score matching)을 통해 신경망 $s_\theta(X_t, \mu, t)$가 로그 밀도의 기울기를 학습합니다. 

$X_0$가 주어졌을 때:

$$\nabla_{X_t} \log p_t(X_t|X_0) = \frac{\epsilon_t}{\sigma_{\mu,\Sigma,t}}$$

여기서 $\epsilon_t \sim \mathcal{N}(0, \sigma_{\mu,\Sigma,t} I)$이고 $X_t = \mu_{\mu,\Sigma,t}(X_0) + \sigma_{\mu,\Sigma,t}\epsilon_t$입니다.[1]

확산 손실(Diffusion Loss):

$$\mathcal{L}_{diff} = \mathbb{E}_{X_0, t} \left[\lambda_t \mathbb{E}_{\epsilon_t}\left\|s_\theta(X_t, \mu, t) - \frac{\epsilon_t}{\sigma_{\mu,\Sigma,t}}\right\|_2^2\right]$$

**인코더 손실(Encoder Loss):**

정렬된 인코더 출력이 노이즈 분포 $\mathcal{N}(\mu, I)$를 매개변수화하도록:

$$\mathcal{L}_{enc} = -\sum_{j=1}^F \log \mathcal{N}(y_{A(j)}|\mu_j, I)$$

**지속 시간 예측 손실(Duration Prediction Loss):**

$$\mathcal{L}_{dp} = \text{MSE}(\log(\text{DP}(\text{sg}(\mu))), \log(d))$$

#### **2.3 모델 구조**

Grad-TTS는 세 가지 주요 모듈로 구성됩니다:[1]

**1. 인코더(Encoder)**
- Transformer-TTS에서 차용한 구조
- 구성: 사전 네트워크(pre-net) → 6개의 Transformer 블록 → 선형 투영 층
- 입력: 텍스트 시퀀스 $x_{1:L}$ (문자 또는 음소)
- 출력: 특성 시퀀스 $\mu_{1:L}$

**2. 지속 시간 예측기(Duration Predictor)**
- 2개의 합성곱 층 + 투영 층
- 각 텍스트 요소의 지속 시간을 예측
- Monotonic Alignment Search(MAS)와 함께 작동

**3. 디코더(Decoder) - 확산 기반**
- **U-Net 아키텍처** (Ho et al., 2020 기반)
- 채널: 표준 이미지 생성의 절반
- 해상도: 3가지 (80×F, 40×F/2, 20×F/4)
  - 80차원 멜-스펙트로그램 대응
- 정렬된 인코더 출력이 U-Net 입력에 추가 채널로 연결됨[1]

**Monotonic Alignment Search (MAS):**

MAS는 동적 계획법을 사용하여 텍스트 시퀀스와 음향 프레임 사이의 최적 일대일 대응을 찾습니다. 이 정렬 $A$는 단조성(monotonicity)과 전사성(surjectivity)을 보장하여 텍스트의 올바른 발음 순서를 유지합니다.[1]

#### **2.4 추론 프로세스**

종료 조건에서 시작: $X_T \sim \mathcal{N}(\mu, \tau^2 I)$ (온도 $\tau=1.5$ 사용)

ODE를 Euler 스킴으로 역시간 풀이:

$$\frac{dX_t}{dt} = -\frac{1}{2}\beta_t(\mu - X_t) + \frac{1}{2}\beta_t^2 s_\theta(X_t, \mu, t)$$

단계 크기 $h$를 조절하여 품질-속도 트레이드오프 제어[1]

#### **2.5 성능 비교**

**주관적 평가 (Mean Opinion Score):**[1]

| 모델 | MOS | RTF (Real-Time Factor) |
|------|-----|------------------------|
| Grad-TTS-1000 | 4.44 ± 0.05 | 3.663 |
| Grad-TTS-100 | 4.38 ± 0.06 | 0.363 |
| Grad-TTS-10 | 4.38 ± 0.06 | 0.033 |
| Grad-TTS-4 | 3.96 ± 0.07 | 0.012 |
| Tacotron2 | 4.32 ± 0.07 | 0.075 |
| Glow-TTS | 4.11 ± 0.07 | 0.008 |
| Ground Truth | 4.53 ± 0.06 | - |

**주요 성능 특징:**
1. Grad-TTS-10 (10 반복)은 Tacotron2와 경쟁 가능한 품질 유지
2. 단계별 성능 향상이 명확: 4 → 10 반복 사이에 큰 개선, 이후 점진적 개선
3. GPU에서 100 스텝 이하로 실시간 합성 가능 (RTF < 1.0)
4. Tacotron2 대비 약 2배 빠른 추론 속도[1]

**객관적 평가:**

Grad-TTS는 더 큰 용량의 Glow-TTS(3배 큰 디코더)보다 높은 로그 우도(log-likelihood)를 달성했습니다. 이는 확산 모델의 효율성을 시사합니다.[1]

**오류 분석:**

Grad-TTS와 Tacotron2는 유사한 오류 패턴을 보이지만, Glow-TTS는 다음 문제가 두드러집니다:[1]
- 잘못된 단어 강조 (~40%)
- 로봇음성 인상 (~25%)

***

### 3. 모델의 일반화 성능 향상 가능성

#### **3.1 현재 한계점**

논문에서 논의된 일반화 관련 한계:[1]

1. **LJSpeech 데이터셋의 제약**: 단일 화자(여성 영어), 24시간 데이터로 훈련
2. **단순화된 구조**: $\Sigma = I$ 선택으로 모델 단순화
3. **노이즈 스케줄 문제**: 선형 스케줄 사용, 최적화되지 않은 선택
4. **손실 가중치**: 시간 $t$에서의 손실 가중치 $\lambda_t$ 선택이 임의적

#### **3.2 일반화 성능 향상 전략**

**가. 다중 스피커/다중 언어 확장**

최근 연구들이 이 방향을 탐색했습니다:[2][3]

- **Multi-GradSpeech** (2023): 다중 스피커 시나리오에서의 표본 드리프트(sampling drift) 문제 해결 시도[2]
  - 일관성 있는 확산 모델 도입
  - 더 복잡한 데이터 분포 모델링

- **MParrotTTS** (2023): 다국어 다중 스피커 합성[3]
  - 자체 지도 학습 표현 활용
  - 저자원 언어로의 적응 가능성

**나. 잠재 확산 모델(Latent Diffusion Models)**

최신 연구 (2024-2025)에서 잠재 공간 기반 접근이 부상:[4][5]

- **DiTTo-TTS** (2024): Diffusion Transformer 기반
  - 82,000시간 대규모 훈련 데이터
  - 음소와 지속 시간 같은 영역 특화 요소 제거
  - 개선된 음향 피처 복구

- **Schrodinger Bridges** (2023): 확산 모델 개선[4]
  - 사전 정의된 데이터-노이즈 프로세스의 제약 극복
  - Grad-TTS 대비 50단계/1000단계 합성에서 성능 향상

**다. 구조적 개선**

1. **노이즈 스케줄 최적화**
   - 다양한 스케줄(코사인, 지수, 다항식) 탐색
   - 시간 $t$에 따른 적응적 가중치

2. **신경망 아키텍처 개선**
   - Transformer 디코더로의 전환 (최근 추세)
   - Lipschitz 제약 도입으로 SDE 안정성 향상[1]

3. **조건부 모델링 강화**
   - 스타일/음향 특성의 명시적 모델링
   - **DEX-TTS**: 스타일 모델링을 통한 음성 변위성 향상[6]

#### **3.3 일반화 성능 지표**

현재 측정되는 일반화 지표:

1. **음성 품질**: MOS, UTMOS
2. **발음/자연성**: 문자 오류율(CER), 단어 오류율(WER)
3. **스피커 유사성**: 스피커 임베딩 유사도
4. **효율성**: 실시간 인수(RTF), 매개변수 수

***

### 4. 논문이 향후 연구에 미치는 영향 및 고려 사항

#### **4.1 학술적 영향**

**Grad-TTS의 선구적 역할:**

1. **확산 기반 TTS의 개척**
   - 2021년 발표 이후 확산 모델 기반 음성 합성이 주류 기술으로 발전
   - 인용 수 775회 이상 (ICML 2021 발표)

2. **이론적 기여**
   - SDE 프레임워크를 TTS에 적용한 첫 사례
   - 조건부 분포를 갖는 일반화된 확산 과정의 수학적 체계화[1]

3. **산업 적용**
   - Huawei Noah's Ark Lab 주도의 오픈소스 공개
   - 다양한 후속 TTS 시스템의 기초 모델

#### **4.2 후속 연구 동향 (2021-2025)**

**주요 연구 방향:**

1. **효율성 개선**[7][8]
   - **ResGrad** (2022): 잔차 구조로 10배 이상 속도 향상
   - **DCTTS** (2023): 이산 확산 + 대조 학습으로 자원 소비 감소
   - **FastDiff 2** (2023): GAN과 확산 모델 결합

2. **표현력 확장**[9][10]
   - **StyleTTS 2** (2023): 대형 음성 언어 모델과 통합하여 인간 수준의 합성 달성
   - **Guided-TTS** (2022): 분류기 가이던스로 전사 없는 다화자 합성
   - **EdiTTS** (2022): 스코어 기반 세밀한 음성 편집

3. **멀티모달/멀티태스크**[11]
   - 연속 토큰 기반 확산으로 대형 언어 모델 통합
   - 화자-참조 TTS의 이중 헤드 아키텍처

4. **음성 강화/품질 개선**[12]
   - 조건부 잠재 확산 모델(cLDM)
   - 이중 맥락 학습으로 미학습 노이즈 환경 대응

#### **4.3 향후 연구 시 고려 사항**

**이론적 고려사항:**

1. **SDE 기초 강화**
   - Lipschitz 제약이 신경망 학습에 미치는 영향 분석
   - 다양한 SDE 형태(VP, VE, sub-VP)의 TTS 적용성 검토

2. **손실 함수 설계**
   - 최적 가중치 함수 $\lambda_t$ 결정 방법론 개발
   - 시간별 그래디언트 예측 어려움의 근본 원인 분석

3. **샘플 효율성**
   - 필요한 훈련 데이터량 최소화
   - 전이 학습(transfer learning) 가능성 탐구

**실무적 고려사항:**

1. **배포 최적화**
   - 모바일/임베디드 환경에서의 추론 속도 개선
   - 양자화(quantization), 지식 증류(knowledge distillation) 적용

2. **다국어/다문화 적응**
   - 저자원 언어에서의 일반화 성능 향상
   - 음운 체계의 차이 처리 방안

3. **표현 제어**
   - 감정, 운율, 음성 특성의 세밀한 제어
   - 참조 음성 기반의 영점 학습(zero-shot) 능력

4. **실시간성 극복**
   - 추론 단계 감소 기법 (ODE 솔버 개선, 지식 증류 등)
   - 병렬화 가능한 아키텍처 설계

5. **평가 표준화**
   - 객관적 메트릭 개선 (현재 주관적 평가에 의존)
   - 다양한 언어/화자에 대한 벤치마크 구축

***

### 5. 2020년 이후 관련 최신 연구 탐색

#### **5.1 확산 모델 기반 TTS 주요 발전**

| 시기 | 주요 연구 | 핵심 혁신 |
|------|---------|---------|
| 2021 | Grad-TTS | 확산 기반 멜-스펙트로그램 생성 첫 제시 |
| 2021 | DiffWave | 파형 레벨 확산 보코더 |
| 2022 | EdiTTS | 스코어 기반 음성 편집 |
| 2022 | ResGrad | 잔차 구조로 속도 10배 향상 |
| 2022 | Guided-TTS | 분류기 가이던스 기반 다화자 합성 |
| 2023 | StyleTTS 2 | SLM 통합으로 인간 수준 품질 달성 |
| 2023 | Multi-GradSpeech | 다중 스피커 확장 |
| 2023 | DCTTS | 이산 확산으로 효율성 극대화 |
| 2024 | DiTTo-TTS | Diffusion Transformer로 대규모 학습 |
| 2025 | DLPO | 강화 학습으로 TTS 확산 모델 최적화 |

#### **5.2 기술적 트렌드**

**1. 모델 구조 진화:**
- U-Net → Transformer/DiT로 전환
- 잠재 확산 모델의 확대 도입

**2. 데이터 스케일링:**
- 단일 화자 24시간 → 다중 화자 82,000시간+
- 대형 음성 언어 모델 통합

**3. 효율성 개선:**
- 역방향 단계 수 감소 (1000 → 4-10 단계)
- 실시간 처리 가능 (GPU RTF < 1.0)

**4. 표현력 확장:**
- 스타일/감정/음향 제어
- 다언어 능력 강화

#### **5.3 핵심 논문들의 성능 비교**

| 모델 | 기술 | MOS | RTF | 특징 |
|------|------|-----|-----|------|
| Grad-TTS (10단계) | SDE 확산 | 4.38 | 0.033 | 품질-속도 균형 |
| Schrodinger Bridges | 최적 전송 | 4.62 | 0.05 | 확산 과정 개선 |
| StyleTTS 2 | SLM + 확산 | 4.57 | 0.10 | 인간 수준 품질 |
| DiTTo-TTS | Diffusion Transformer | 4.54 | 0.12 | 대규모 확장성 |
| DEX-TTS | 스타일 모델링 | 4.45 | 0.045 | 표현력 강화 |

***

### 결론

**Grad-TTS**는 확산 확률 모델을 음성 합성에 적용한 선구적 연구로, 다음과 같은 혁신을 제시했습니다:

1. **이론적 기여**: 일반화된 SDE 프레임워크로 조건부 노이즈 분포 처리
2. **실무적 성과**: 품질-속도 명시적 제어로 실용성 확보
3. **연구 방향 제시**: 확산 기반 TTS의 학술적 및 산업적 발전 촉발

향후 연구는 **다중 화자/언어 확장**, **대규모 데이터 활용**, **실시간 추론**, **표현 제어 강화**라는 네 가지 축을 중심으로 진행되고 있으며, 이는 인간 수준의 자연스러운 음성 합성 실현을 향한 지속적 진화를 보여줍니다.[10][3][9][2][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e7e4afdd-f728-4f17-8d53-cddcbca630e2/2105.06337v2.pdf)
[2](http://arxiv.org/pdf/2406.11427.pdf)
[3](http://arxiv.org/pdf/2309.06787.pdf)
[4](https://arxiv.org/pdf/2111.11755.pdf)
[5](http://arxiv.org/pdf/2312.03491.pdf)
[6](https://arxiv.org/pdf/2212.14518.pdf)
[7](https://arxiv.org/abs/2406.19135)
[8](http://arxiv.org/pdf/2211.09383.pdf)
[9](https://arxiv.org/pdf/2306.07691.pdf)
[10](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.pdf)
[11](https://www.isca-archive.org/interspeech_2022/tae22_interspeech.pdf)
[12](https://proceedings.mlr.press/v139/popov21a/popov21a.pdf)
[13](https://arxiv.org/abs/2105.06337)
[14](https://arxiv.org/abs/2506.08457)
[15](https://arxiv.org/html/2311.01797v4)
[16](https://aclanthology.org/2025.coling-main.352.pdf)
[17](https://scorebasedgenerativemodeling.github.io)
[18](https://www.isca-archive.org/syndata4genai_2024/rossenbach24_syndata4genai.pdf)
[19](https://proceedings.mlr.press/v139/popov21a.html)
[20](https://arxiv.org/abs/2306.03509)
[21](https://arxiv.org/abs/2308.10428)
[22](https://arxiv.org/pdf/2305.11926.pdf)
[23](http://arxiv.org/pdf/2406.17257.pdf)
[24](https://arxiv.org/pdf/1907.04448.pdf)
[25](https://arxiv.org/pdf/2108.07737.pdf)
[26](https://arxiv.org/html/2510.12995v1)
[27](https://arxiv.org/pdf/2501.10052.pdf)
[28](https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2021_okamoto.pdf)
[29](https://www.emergentmind.com/topics/cross-lingual-tts-model)
[30](https://arxiv.org/pdf/2406.11427.pdf)
[31](https://arxiv.org/pdf/2206.09920.pdf)
[32](https://arxiv.org/html/2510.08373v1)
[33](https://openreview.net/forum?id=m4mwbPjOwb)
[34](https://aclanthology.org/2023.findings-acl.437.pdf)
[35](https://www.sciencedirect.com/science/article/abs/pii/S0885230822000584)
