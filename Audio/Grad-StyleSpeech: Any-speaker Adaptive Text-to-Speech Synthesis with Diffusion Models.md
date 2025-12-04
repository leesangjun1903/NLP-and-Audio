# Grad-StyleSpeech: Any-speaker Adaptive Text-to-Speech Synthesis with Diffusion Models

### 1. 핵심 주장과 주요 기여

**Grad-StyleSpeech**는 Score-based Diffusion Model과 계층적 트랜스포머 인코더를 결합한 Zero-shot 임의 화자 적응형 텍스트-음성(TTS) 합성 모델입니다. 이 논문의 핵심 주장은 **기존 임의 화자 적응형 TTS 방법들이 목표 화자의 음성 스타일 모방에서 부족하다**는 문제를 지적하고, 이를 해결하기 위해 확산 모델의 강력한 생성 능력과 스타일 기반 생성 모델을 결합하는 것입니다.[1]

**주요 기여:**

1. **Zero-shot 적응형 TTS 모델 제안**: 단 몇 초의 참조 음성만으로 임의의 새로운 화자 음성을 고품질로 합성[1]

2. **계층적 트랜스포머 인코더**: Score-based Diffusion 모델을 위해 대표적인 사전 노이즈 분포를 생성하는 새로운 구조 제안[1]

3. **실증적 성능 개선**: LibriTTS 및 VCTK 데이터셋에서 최근 기준선 모델들을 능가하는 성능 달성[1]

### 2. 해결하고자 하는 문제

기존 임의 화자 적응형 TTS 방법들은 두 가지 주요 한계를 가지고 있습니다:[1]

- **지도학습 기반 방법의 한계**: AdaSpeech, Meta-StyleSpeech 등 이전 연구들은 목표 화자로부터 전사된(supervised) 샘플을 필요로 하며, 이는 모델 파라미터 업데이트에 큰 계산 비용을 초래합니다[1]

- **Zero-shot 방법의 성능 부족**: 신경 인코더 기반의 최신 Zero-shot 방법들(YourTTS, Meta-StyleSpeech)은 목표 화자 음성과의 유사도가 낮고, 감정 음성 같은 독특한 스타일에 취약합니다[1]

### 3. 제안하는 방법 및 수식

**전체 프레임워크 구성:**

Grad-StyleSpeech는 세 가지 핵심 컴포넌트로 구성됩니다:[1]

#### 3.1 Mel-Style Encoder

참조 음성을 스타일 벡터로 임베딩하는 컴포넌트:

$$s = h_\psi(Y)$$

여기서 $s \in \mathbb{R}^{d'}$는 스타일 벡터이고, $h_\psi$는 스펙트럼-시간 처리기, 멀티헤드 자기 주의(Multi-head Self-Attention), 그리고 시간 평균 풀링으로 구성됩니다.[1]

#### 3.2 Score-based Diffusion Model

**Forward Diffusion Process:**

$$dY_t = -\frac{1}{2}\beta(t)(Y_t - \mu)dt + \sqrt{\beta(t)}dW_t$$

여기서 $t \in [0, T]$는 연속 시간 단계, $\beta(t)$는 노이즈 스케줄링 함수, $W_t$는 위너 프로세스(Wiener Process)입니다. $\mu$는 텍스트 및 스타일 조건화된 표현입니다.[1]

전이 커널은 가우시안 분포로 계산됩니다:[1]

$$p_{0t}(Y_t|Y_0) = \mathcal{N}(Y_t; \gamma_t, \sigma_t^2 I)$$

$$\sigma_t^2 = I - e^{-\int_0^t \beta(s)ds}$$

$$\gamma_t = (I - e^{-\frac{1}{2}\int_0^t \beta(s)ds})\mu + e^{-\frac{1}{2}\int_0^t \beta(s)ds}Y_0$$

**Reverse Diffusion Process:**

$$dY_t = \left[-\frac{1}{2}\beta(t)(Y_t - \mu) - \beta(t)\nabla_{Y_t}\log p_t(Y_t)\right]dt + \sqrt{\beta(t)}d\tilde{W}_t$$

여기서 $\nabla_{Y_t}\log p_t(Y_t)$는 스코어 함수이며, 신경망 $\epsilon_\theta(Y_t, t, \mu, s)$로 근사됩니다.[1]

#### 3.3 Score Network Training

스코어 추정 네트워크는 다음 손실 함수로 훈련됩니다:[1]

$$L_{diff} = \mathbb{E}_{t \sim \mathcal{U}(0,T)} \mathbb{E}_{Y_0 \sim p_0(Y_0)} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left\|\epsilon_\theta(Y_t, t, \mu, s) + \sigma_t^{-1}\epsilon\right\|_2^2$$

여기서 $\sigma_t = \sqrt{1 - e^{-\int_0^t \beta(s)ds}}$[1]

#### 3.4 계층적 트랜스포머 인코더

세 단계 계층 구조로 구성되어 있습니다:[1]

1. **텍스트 인코딩**: $H = f_\lambda(x) \in \mathbb{R}^{n \times d}$ - 입력 텍스트를 숨겨진 표현으로 변환

2. **길이 정규화**: $\tilde{H} = \text{Align}(H, x, Y) \in \mathbb{R}^{m \times d}$ - 비지도 정렬 학습 프레임워크 사용

3. **스타일 적응 인코딩**: $\mu = g_\varphi(\tilde{H}, s)$ - Style-Adaptive Layer Normalization(SALN)을 통한 스타일 조건화[1]

**Prior Loss:**

$$L_{prior} = \|\mu - Y\|_2^2$$

#### 3.5 전체 훈련 목적 함수

$$L = L_{diff} + L_{prior} + L_{align}$$

여기서 $L_{align}$은 정렬기 및 지속 시간 예측기 훈련을 위한 손실입니다.[1]

### 4. 모델 구조

**아키텍처 구성:**

1. **Mel-Style Encoder**: 스펙트럼 및 시간 처리기, 멀티헤드 자기 주의, 시간 평균 풀링으로 구성[1]

2. **Hierarchical Transformer Encoder**: 
   - 텍스트 인코더: 4개의 트랜스포머 블록
   - 스타일 적응 인코더: 4개의 트랜스포머 블록 + SALN
   - 정렬기: 비지도 정렬 학습[1]

3. **Diffusion Model**: 
   - 음성 길이 예측기(Duration Predictor)
   - U-Net 기반 노이즈 추정 네트워크 $\epsilon_\theta$[1]

4. **Vocoder**: HiFi-GAN으로 멜-스펙트로그램을 파형으로 변환[1]

### 5. 성능 향상

**객관적 평가 결과:**

논문에서는 Speaker Embedding Cosine Similarity(SECS)와 Character Error Rate(CER)를 측정했습니다:[1]

| 평가 지표 | LibriTTS (clean-360) | VCTK |
|---------|-------------------|------|
| SECS ↑ | 87.51 ± 0.86 | 83.64 ± 0.78 |
| CER ↓ | 4.12 ± 1.64 | 3.65 ± 1.46 |

Grad-StyleSpeech(full)는 기존 기준선들과 비교하여:[1]
- YourTTS보다 VCTK에서 SECS 0.85 포인트 향상
- Grad-TTS(any-speaker) 대비 LibriTTS에서 SECS 6.98 포인트 향상

**주관적 평가:**

Mean Opinion Score(MOS)와 Similarity MOS(SMOS) 측정 결과:[1]

| 모델 | LibriTTS MOS | LibriTTS SMOS | VCTK MOS | VCTK SMOS |
|------|-------------|--------------|---------|-----------|
| Ground Truth | 4.57 ± 0.13 | 3.50 ± 0.24 | 4.32 ± 0.12 | 4.06 ± 0.16 |
| Grad-StyleSpeech (clean) | 4.18 ± 0.18 | 3.83 ± 0.25 | 4.13 ± 0.14 | 3.95 ± 0.16 |

**Fine-tuning 성능:**

제한된 목표 화자 데이터(20개 음성, VCTK)로 100단계의 Fine-tuning만으로도 AdaSpeech보다 우수한 성능을 달성했습니다.[1]

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 계층적 구조의 일반화 이점

계층적 트랜스포머 인코더 설계는 다음과 같이 일반화 성능을 향상시킵니다:[1]

1. **텍스트 표현 분리**: 텍스트 인코더에서 언어적 내용만 추출하고, 스타일 정보는 스타일-적응 인코더에서 처리함으로써 각 컴포넌트의 역할이 명확해집니다

2. **스타일 조건화의 효율성**: SALN을 통한 스타일 조건화로 보지 못한 화자의 특성을 더 잘 포착할 수 있습니다[1]

3. **비지도 정렬 학습**: 전사 데이터 없이도 텍스트와 음성의 정렬을 학습하여 데이터 효율성 향상[1]

#### 6.2 Score-based Diffusion 모델의 일반화 강점

Grad-TTS 기반의 Diffusion 모델은:[1]

1. **유연한 조건화**: 선택적 조건화를 통해 보지 못한 화자의 음성 특성을 더 잘 적응시킬 수 있습니다

2. **고주파 성분 모델링**: 과도한 평활화 문제를 극복하고 세부적인 음성 특징을 생성함으로써 음성 유사도 향상[1]

3. **점진적 생성**: 사전 노이즈 분포 $N(\mu, I)$에서 시작하여 데이터 분포로 점진적으로 변환되므로, 보지 못한 화자에 대해 더 안정적인 생성이 가능합니다[1]

#### 6.3 Zero-shot 일반화 성능

실험 결과에서 보이는 일반화 성능 개선:[1]

- **보지 못한 화자 적응(VCTK 평가)**: LibriTTS로만 훈련했음에도 불구하고 VCTK 데이터셋의 보지 못한 화자에 대해 우수한 성능 달성
- SECS: 83.64 ± 0.78 (매우 높은 화자 유사도)
- SMOS: 3.95 ± 0.16 (Ground Truth 4.06과 거의 유사)

### 7. 한계

**논문에서 인정하는 한계점:**

1. **데이터 품질 의존성**: Full 모델이 clean-360 부분에서 낮은 CER을 보이는 것은 데이터 품질이 중요함을 시사합니다. 더 많은 깨끗한 샘플을 포함한 데이터셋으로 훈련하면 성능이 향상될 수 있습니다[1]

2. **계산 비용**: Diffusion 모델의 본질적인 한계로 100단계의 Denoising 프로세스가 필요하여 추론 속도가 느릴 수 있습니다[1]

3. **정렬 정확도**: Duration Predictor에 의존하는 구조로 인해 복잡한 음절 구조의 언어에서 정렬 오류 발생 가능성[1]

### 8. 논문의 앞으로의 연구에 미치는 영향

#### 8.1 이론적 영향

1. **Diffusion 기반 TTS의 가능성 증명**: Score-based Diffusion 모델이 단순히 고품질 음성 생성뿐 아니라 스타일 기반 적응형 합성에도 효과적임을 보여줍니다[1]

2. **계층적 조건화의 중요성**: 텍스트, 스타일, 음성 생성의 분리된 처리가 모델 일반화를 향상시킨다는 것을 입증합니다[1]

3. **Zero-shot 음성 합성의 경계 확장**: 참조 음성만으로 새로운 화자의 음성을 높은 품질로 합성할 수 있음을 시연했습니다[1]

#### 8.2 기술적 영향

**후속 연구들이 참고할 수 있는 핵심 기법:**

1. **스타일 추출과 적응**: Mel-style 인코더와 SALN의 조합은 이후 음성 스타일 변환, 감정 음성 합성 등 다양한 분야에서 적용되었습니다[2][3]

2. **Hierarchical 구조의 효율성**: 멀티레벨 스타일 적응기(Multi-level Style Adaptor)를 통한 계층적 처리는 GenerSpeech, DiffStyleTTS 등에서 채용되었습니다[4][5]

3. **Diffusion 모델과 스타일 결합**: 후속 연구인 StyleTTS 2는 스타일 확산(Style Diffusion)을 더욱 발전시켜 참조 음성 없이도 다양한 스타일의 음성을 생성할 수 있게 했습니다[3]

#### 8.3 실무 적용

1. **음성 클로닝 기술의 민주화**: 소수의 참조 음성만으로 고품질 음성 합성이 가능해져 접근성 향상[1]

2. **개인화된 음성 생성**: 장애인 의사소통 보조(텍스트-음성 변환), 다국어 콘텐츠 제작 등 다양한 응용 분야 확대

3. **실시간 처리로의 진화**: F5-TTS와 같이 확산 모델의 효율성을 개선한 후속 연구로 실시간 음성 합성 가능[6][7]

### 9. 앞으로 연구 시 고려할 점

#### 9.1 기술적 개선 방향

1. **추론 속도 개선**: 
   - 현재 100단계 Denoising은 실시간 처리에 부적합합니다
   - **개선 방안**: F5-TTS처럼 Flow Matching 또는 Latent Diffusion을 활용하여 단계 수 축소[7]
   - Maximum Likelihood SDE 소유자 대신 더 효율적인 샘플링 전략(예: DDIM, Exponential Integrator) 탐색[1]

2. **다언어 확장**:
   - 현재 English만 평가됨
   - **개선 방안**: XTTS처럼 16개 언어 이상의 다국어 모델로 확장[8]
   - 언어 간 스타일 전이(Cross-lingual Style Transfer) 연구

3. **감정 표현 향상**:
   - 현재 감정 음성에 취약하다는 한계 인정[1]
   - **개선 방안**: 감정 레이블이나 감정 기술자(Emotion Descriptor)를 스타일 벡터에 통합
   - 참고: StyleFusion-TTS는 멀티모달 스타일 제어 제안[9]

#### 9.2 일반화 성능 개선

1. **분포 외(Out-of-Distribution) 일반화**:
   - 논문에서는 LibriTTS와 VCTK만 평가됨
   - **개선 방안**: GenerSpeech처럼 분포 외 텍스트에 대한 강건성 평가 추가[5]
   - Mix-Style Layer Normalization(MSLN) 도입으로 스타일 불변 특성 추출

2. **소수 화자 적응(Few-shot Fine-tuning)**:
   - 현재 100단계 Fine-tuning으로 우수한 성능 달성[1]
   - **개선 방안**: Stable-TTS처럼 사전 보존 손실(Prior-preservation Loss) 도입[10]
   - 고품질 사전 샘플 활용으로 적응형 오버피팅 방지

3. **노이즈 견고성**:
   - **개선 방안**: 노이즈가 있는 참조 음성에 대한 견고성 평가 필요
   - 데이터 증강을 통한 노이즈 강건성 향상

#### 9.3 방법론적 고려사항

1. **대안 아키텍처 탐색**:
   - **Transformer 대안**: Conformer 또는 State Space Model(Mamba) 같은 효율적 아키텍처 시도[4]
   - **Attention 메커니즘**: Linear Attention으로 계산 효율성 향상[1]

2. **다양한 조건화 방식**:
   - **Classifier-free Guidance**: StyleTTS 2처럼 스타일 없이도 다양한 스타일 생성 가능[3]
   - **Prompt-based 제어**: ControlSpeech처럼 텍스트 스타일 기술자로 세밀한 제어[11]

3. **손실 함수 개선**:
   - 현재 L2 기반 Prior Loss 외에 Perceptual Loss나 Adversarial Loss 고려[3]
   - 화자 일관성 손실(Speaker Consistency Loss) 추가로 음성 유사도 향상[12]

#### 9.4 평가 지표 확장

1. **현재 평가의 한계**:
   - SECS(화자 유사도)만으로 부족할 수 있음
   - **개선 방안**: 
     - 음질 평가(Mel-cepstral Distortion, Vocoder Quality)
     - 프로소디 충실성(Prosody Fidelity) 평가
     - 음성 자연스러움(Speech Intelligibility) 측정

2. **인간 평가 확대**:
   - 현재 16명의 평가자로 제한됨[1]
   - 더 많은 평가자와 다양한 화자로 평가 확대

#### 9.5 윤리 및 안전 고려사항

1. **음성 클로닝의 악용 방지**:
   - 생성된 음성의 출처 표시(Watermarking) 기술 개발[6]
   - 음성 위변조 탐지(Deepfake Detection) 방법 연구

2. **개인정보 보호**:
   - 참조 음성 데이터의 동의 과정 명확화
   - 학습된 스타일 정보의 개인 식별 가능성 평가

### 10. 2020년 이후 관련 최신 연구 동향

#### 10.1 Diffusion 기반 TTS의 발전

**Grad-TTS (2021)**: Grad-StyleSpeech의 기초가 된 Score-based Diffusion 기반 비자동회귀 TTS 모델. 기존 Diffusion-TTS와 달리 SDE 기반 프레임워크 도입[13][14]

**StyleTTS 2 (2023)**: Style Diffusion과 대규모 음성 언어 모델(SLM)의 적대적 훈련을 결합하여 인간 수준의 TTS 달성. 참조 음성 없이도 다양한 스타일의 음성 생성[3]

**DiffStyleTTS (2025)**: 조건부 확산 모듈과 개선된 Classifier-free Guidance를 활용한 계층적 프로소디 모델링. FastSpeech2보다 향상된 자연스러움 달성[4]

#### 10.2 Zero-shot 음성 합성의 혁신

**VALL-E (2023)**: 신경 코덱 언어 모델 기반 Zero-shot TTS. 3초의 참조 음성만으로 음성 합성, 감정 및 배경음 보존[15]

**VALL-E 2 (2024)**: Repetition Aware Sampling과 Grouped Code Modeling으로 인간 동등 수준의 성능 달성[16]

**F5-TTS (2024)**: Flow Matching과 Diffusion Transformer(DiT) 기반 비자동회귀 모델. RTF 0.15로 매우 빠른 추론 속도 달성. 100K 시간의 다국어 데이터로 훈련[7]

#### 10.3 스타일 기반 적응형 TTS

**Meta-StyleSpeech (2021)**: 메타 학습을 활용한 Zero-shot 음성 합성. 스타일 기반 생성 모델 도입[17][1]

**GenerSpeech (2022)**: 스타일 불변 및 스타일 특정 표현 분리를 통한 분포 외 일반화. Mix-Style Layer Normalization(MSLN) 제안[5]

**YourTTS (2022)**: Flow 기반 다화자 TTS에 화자 일관성 손실 도입. 다국어 지원[12][1]

#### 10.4 음성 클로닝의 고급 기법

**XTTS-v2 (2024)**: 16개 언어 지원의 대규모 다국어 Zero-shot TTS. 6초의 참조 음성으로 음성 클로닝[8]

**Stable-TTS (2024)**: 제한된 노이즈가 있는 대상 샘플로도 안정적인 음성 합성. Prior-preservation Loss로 오버피팅 방지[10]

**DS-TTS (2024)**: Dual-Style Encoding Network와 Dynamic Generator Network로 보지 못한 화자 적응 향상[18]

#### 10.5 효율성 개선 연구

**MegaTTS 3 (2025)**: Sparse Alignment Enhanced Latent Diffusion Transformer. 명시적 음성-텍스트 정렬로 견고성 향상[13]

**FlashSpeech (2024)**: 1-2개 샘플링 단계로 Zero-shot 합성. 기존 모델보다 약 20배 빠른 추론 속도[19]

**TacoLM (2024)**: Gated Attention 메커니즘으로 파라미터 90% 감소, 5.2배 속도 향상[20]

#### 10.6 다국어 및 교차언어 연구

**VALL-E X (2023)**: Cross-lingual 음성 합성 및 음성-음성 번역[21]

**Hola-TTS (2024)**: 중국어, 영어, 일본어, 한국어 4개 언어 교차언어 Zero-shot TTS[22]

**ControlSpeech (2024)**: 텍스트 스타일 기술자를 통한 스타일 제어와 함께 Zero-shot 화자 클로닝[11]

### 결론

Grad-StyleSpeech는 **Score-based Diffusion 모델과 계층적 스타일 기반 생성 모델의 결합**을 통해 Zero-shot 임의 화자 적응형 TTS에서 획기적인 성능 향상을 달성했습니다. 특히 **계층적 구조의 설계**와 **스타일 조건화의 효율성**은 이후 StyleTTS 2, GenerSpeech, DiffStyleTTS 등 수많은 후속 연구의 기초가 되었습니다.

앞으로의 연구에서는 **추론 속도 개선**(F5-TTS, FlashSpeech), **다언어 확장**(XTTS, Hola-TTS), **분포 외 일반화 강화**(GenerSpeech, Stable-TTS), 그리고 **윤리적 고려사항**에 중점을 두어야 합니다. 특히 VALL-E 2의 인간 동등 수준 성능 달성은 음성 합성 분야가 실용화 단계에 진입했음을 의미하며, 이는 접근성 향상, 장애인 지원, 다국어 콘텐츠 제작 등 광범위한 응용 분야를 열어주고 있습니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c2f761b7-70ad-46f0-ae6e-9f8ecf416ba2/2211.09383v2.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11759097/)
[3](https://arxiv.org/pdf/2306.07691.pdf)
[4](https://aclanthology.org/2025.coling-main.352.pdf)
[5](https://openreview.net/pdf?id=dmCyoqxEwHf)
[6](https://www.marktechpost.com/2024/10/13/f5-tts-a-fully-non-autoregressive-text-to-speech-system-based-on-flow-matching-with-diffusion-transformer-dit/)
[7](https://arxiv.org/abs/2410.06885)
[8](https://arxiv.org/abs/2406.04904)
[9](https://arxiv.org/html/2409.15741v1)
[10](https://arxiv.org/html/2412.20155v1)
[11](http://arxiv.org/pdf/2406.01205.pdf)
[12](https://arxiv.org/html/2509.18470v2)
[13](https://arxiv.org/html/2502.18924v4)
[14](http://arxiv.org/pdf/2309.15512.pdf)
[15](http://arxiv.org/pdf/2301.02111v1.pdf)
[16](https://arxiv.org/abs/2406.05370)
[17](https://arxiv.org/html/2505.00579v1)
[18](https://arxiv.org/html/2506.01020v1)
[19](http://arxiv.org/pdf/2404.14700.pdf)
[20](https://arxiv.org/abs/2406.15752)
[21](http://arxiv.org/pdf/2303.03926v1.pdf)
[22](https://ieeexplore.ieee.org/document/10800183/)
[23](https://www.ssrn.com/abstract=5331595)
[24](https://ieeexplore.ieee.org/document/11244635/)
[25](http://arxiv.org/pdf/2211.09383.pdf)
[26](https://www.mdpi.com/1424-8220/25/3/833)
[27](https://arxiv.org/abs/2303.13336)
[28](https://arxiv.org/pdf/2412.06602.pdf)
[29](https://arxiv.org/pdf/2106.15561.pdf)
[30](https://www.ijraset.com/research-paper/human-level-text-to-speech-synthesis-using-style-diffusion-and-deep-learning-techniques)
[31](https://lyricwinter.com/blog/voice-cloning-evolution)
[32](https://neurips.cc/virtual/2022/poster/54425)
[33](https://www.themoonlight.io/ko/review/ditto-tts-diffusion-transformers-for-scalable-text-to-speech-without-domain-specific-factors)
[34](https://arxiv.org/abs/2406.17801)
[35](https://ieeexplore.ieee.org/document/10626897/)
[36](https://aclanthology.org/2024.iwslt-1.2)
[37](https://ieeexplore.ieee.org/document/10800694/)
[38](https://ieeexplore.ieee.org/document/10800681/)
[39](https://ieeexplore.ieee.org/document/10800495/)
[40](https://www.sciencepublishinggroup.com/article/10.11648/j.ajsea.20241202.11)
[41](https://ieeexplore.ieee.org/document/10800403/)
[42](https://ieeexplore.ieee.org/document/10626687/)
[43](https://ieeexplore.ieee.org/document/10800708/)
[44](https://arxiv.org/pdf/2212.14227.pdf)
[45](https://arxiv.org/pdf/2104.01818.pdf)
[46](http://arxiv.org/pdf/2408.11528.pdf)
[47](http://arxiv.org/pdf/2404.16619.pdf)
[48](https://blog.gopenai.com/code-examples-and-insights-from-tortoise-tts-and-styletts-2-133814647f87)
[49](https://www.nature.com/articles/s41598-025-90507-0)
[50](https://www.isca-archive.org/interspeech_2024/kim24o_interspeech.pdf)
[51](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
[52](https://www.isca-archive.org/interspeech_2024/scheibler24_interspeech.pdf)
[53](https://arxiv.org/abs/2308.13007)
[54](https://arxiv.org/html/2506.08457v1)
[55](https://github.com/sidharthrajaram/StyleTTS2)
[56](https://arxiv.org/abs/2406.07855)
[57](https://arxiv.org/abs/2403.05989)
[58](https://ieeexplore.ieee.org/document/10890943/)
[59](https://arxiv.org/abs/2410.04380)
[60](https://arxiv.org/abs/2401.07333)
[61](http://arxiv.org/pdf/2406.05370.pdf)
[62](https://arxiv.org/pdf/2401.07333.pdf)
[63](https://arxiv.org/ftp/arxiv/papers/2307/2307.10550.pdf)
[64](http://arxiv.org/pdf/2406.15752.pdf)
[65](http://arxiv.org/pdf/2406.04904.pdf)
[66](https://syncedreview.com/2024/06/11/microsofts-vall-e-2-first-time-human-parity-in-zero-shot-text-to-speech-achieved/)
[67](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/articles/vall-e-2-enhancing-the-robustness-and-naturalness-of-text-to-speech-models/)
[68](https://www.isca-archive.org/interspeech_2023/kim23k_interspeech.pdf)
[69](https://randomsampling.tistory.com/106)
[70](https://openreview.net/forum?id=JiX2DuTkeU)
[71](https://www.themoonlight.io/ko/review/vall-e-2-neural-codec-language-models-are-human-parity-zero-shot-text-to-speech-synthesizers)
