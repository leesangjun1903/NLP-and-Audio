# Moûsai: Efficient Text-to-Music Diffusion Models

---

## 1. 핵심 주장 및 주요 기여 요약

**Moûsai**는 텍스트 설명으로부터 고품질 스테레오 음악(48kHz)을 수 분 단위로 생성할 수 있는 **2단계 캐스케이딩 잠재 확산 모델(cascading latent diffusion model)**이다. 이 논문의 핵심 주장은 다음과 같다:

1. **텍스트-음악 확산 모델의 최초 제안**: 2단계 캐스케이딩 잠재 확산 모델링을 통해 텍스트에서 음악을 생성하는 최초의 접근법을 제시.
2. **높은 효율성**: 64배 오디오 압축률과 특수화된 1D U-Net 설계로, 단일 A100 GPU에서 약 1주일 훈련, 소비자용 GPU에서 실시간(real-time) 추론 달성.
3. **TEXT2MUSIC 데이터셋 수집**: 50K 텍스트-음악 쌍(2,500시간)으로 다양한 장르를 포괄.
4. **기존 모델 대비 우위**: 11개 평가 기준에서 효율성, 텍스트-음악 관련성, 음악 품질, 장기 구조(long-term structure) 등에서 뚜렷한 성능 향상 입증.

---

## 2. 상세 분석: 문제, 방법, 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 음악 생성 모델들은 다음 네 가지 핵심 한계를 가지고 있었다:

1. **긴 음악 생성의 어려움**: 대부분의 text-to-audio 시스템(Riffusion, AudioGen 등)은 수 초 단위 오디오만 생성 가능.
2. **모델 효율성 부족**: Jukebox 등은 1분의 오디오 생성에 GPU 수 시간이 필요.
3. **다양성 부족**: 많은 모델(RAVE, Musika)이 단일 장르에 제한.
4. **텍스트 기반 제어 부재**: 대부분 잠재 상태(latent state), 음악 시작 부분, 또는 가사 기반 제어에 의존.

### 2.2 제안하는 방법

Moûsai는 **2단계 파이프라인**으로 구성된다:

#### Stage 1: Diffusion Magnitude-Autoencoding (DMAE) — 음악 인코딩

**v-objective diffusion**을 사용한다. 데이터 분포 $p(\mathbf{x}\_0)$에서 샘플 $\mathbf{x}\_0$, 노이즈 스케줄 $\sigma_t \in [0, 1]$, 노이즈가 추가된 데이터 $\mathbf{x}\_{\sigma_t} = \alpha_{\sigma_t}\mathbf{x}\_0 + \beta_{\sigma_t}\boldsymbol{\epsilon}$에 대해, 모델 $\hat{\mathbf{v}}\_{\sigma_t} = f(\mathbf{x}_{\sigma_t}, \sigma_t)$를 다음 목적 함수를 최소화하여 학습한다:

$$\mathbb{E}\_{t \sim [0,1], \sigma_t, \mathbf{x}_{\sigma_t}} \left[ \| f_\theta(\mathbf{x}_{\sigma_t}, \sigma_t) - \mathbf{v}_{\sigma_t} \|_2^2 \right] $$

여기서:

$$\mathbf{v}_{\sigma_t} = \frac{\partial \mathbf{x}_{\sigma_t}}{\sigma_t} = \alpha_{\sigma_t}\boldsymbol{\epsilon} - \beta_{\sigma_t}\mathbf{x}_0$$

$$\phi_t := \frac{\pi}{2}\sigma_t, \quad \alpha_{\sigma_t} := \cos(\phi_t), \quad \beta_{\sigma_t} := \sin(\phi_t)$$

**DDIM 샘플러**를 사용한 디노이징 과정:

$$\hat{\mathbf{v}}_{\sigma_t} = f_\theta(\mathbf{x}_{\sigma_t}, \sigma_t) $$

$$\hat{\mathbf{x}}_0 = \alpha_{\sigma_t}\mathbf{x}_{\sigma_t} - \beta_{\sigma_t}\hat{\mathbf{v}}_{\sigma_t} $$

$$\hat{\boldsymbol{\epsilon}}_{\sigma_t} = \beta_{\sigma_t}\mathbf{x}_{\sigma_t} + \alpha_{\sigma_t}\hat{\mathbf{v}}_{\sigma_t} $$

$$\hat{\mathbf{x}}_{\sigma_{t-1}} = \alpha_{\sigma_{t-1}}\hat{\mathbf{x}}_0 + \beta_{\sigma_{t-1}}\hat{\boldsymbol{\epsilon}}_t $$

**전체 DMAE 과정**: 파형 $\mathbf{w}$ (shape $[c, t]$)에서 STFT를 적용하여 크기 스펙트로그램 $\mathbf{m}_\mathbf{w}$를 추출하고, 1D 합성곱 인코더로 잠재 표현을 생성한다:

$$\mathbf{z} = \mathcal{E}_{\theta_e}(\mathbf{m}_\mathbf{w})$$

위상(phase)은 버리고 크기(magnitude)만 사용하며, 확산 디코더로 파형을 복원한다:

$$\hat{\mathbf{w}} = \mathcal{D}_{\theta_d}(\mathbf{z}, \boldsymbol{\epsilon}, s)$$

이 과정에서 **64배 압축률**을 달성하며, tanh 함수를 병목(bottleneck)에 적용하여 값의 범위를 $[-1, 1]$로 제한한다.

#### Stage 2: Text-Conditioned Latent Diffusion (TCLD) — 텍스트 조건부 생성

사전 학습된 동결(frozen) T5 언어 모델로 텍스트 임베딩 $\mathbf{e}$를 생성하고, **classifier-free guidance (CFG)**를 확률 0.1의 학습된 마스크로 적용한다. U-Net 구성:

$$f_{\theta_g}(\mathbf{z}_{\sigma_t}; \sigma_t, \mathbf{e})$$

생성기 $\mathcal{G}_{\theta_g}(\mathbf{e}, \boldsymbol{\epsilon}, s)$가 DDIM 샘플링으로 근사 잠재 $\hat{\mathbf{z}}$를 생성하고, 최종 추론 스택은:

$$\hat{\mathbf{w}} = \mathcal{D}_{\theta_d}\left(\mathcal{G}_{\theta_g}(\mathbf{e}, \boldsymbol{\epsilon}_g, s_g), \boldsymbol{\epsilon}_d, s_d\right) $$

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---------|---------|
| **DMAE (Stage 1)** | 185M 파라미터, 7개 중첩 U-Net 블록, 채널 수 [256, 512, 512, 512, 1024, 1024, 1024], 다운샘플링 [1, 2, 2, 2, 2, 2, 2] → 64배 압축 |
| **TCLD (Stage 2)** | 857M 파라미터 (T5-base 포함), 6개 중첩 U-Net 블록, 채널 수 [128, 256, 512, 512, 1024, 1024] |
| **1D U-Net** | ResNet 잔차 블록, 모듈레이션(노이즈 레벨 조건부), 주입(inject) 항목 (Stage 1), 셀프 어텐션 + 크로스 어텐션 (Stage 2) |
| **텍스트 인코더** | 동결된 T5-base 모델 |

### 2.4 성능 향상

| 메트릭 | Riffusion | Musika | **Moûsai** |
|--------|-----------|--------|-----------|
| FAD (↓) | 0.0018 | 0.0020 | **0.00015** |
| Fidelity (↑) | 2.8 | 3.04 | **3.17** |
| Melody (↑) | 2.66 | **3.21** | 3.15 |
| Harmony (↑) | 2.48 | 3.04 | **3.08** |
| Clarity (↑) | 2.37 | 2.88 | **2.92** |
| CLAP Score (↑) | 0.06 | — | **0.13** |

**효율성 비교** (43초 음악 생성 기준):

| 메트릭 | Riffusion | **Moûsai** |
|--------|-----------|-----------|
| 추론 시간 (s) | 218.0 | **49.2** |
| 메모리 (GB) | 8.85 | **5.04** |
| 실시간 계수 (RTF) | 5.07 | **1.14** |

- FAD에서 기존 모델 대비 **약 10배 이상 개선**
- CLAP 점수에서 Riffusion 대비 **2배 이상** 향상
- 추론 속도에서 Riffusion 대비 **약 4.4배** 빠름

### 2.5 한계

1. **데이터 규모**: 2,500시간으로, Jukebox(70K시간), AudioLM(40K시간) 대비 소규모. 50K–100K 시간 규모 학습 필요.
2. **고주파 음역 처리 한계**: 저주파(드럼, 베이스 등)에서 우수하지만, 고주파(클래식 음악 등)에서 화성(harmony)이 불분명하게 생성됨 (클래식 음악 전용 학습 시 fidelity 9.5% 하락).
3. **L2 손실 기반 학습**: 인지적(perceptual) 손실 함수를 사용하지 않아 비인지적 소리도 동일 가중치로 처리.
4. **텍스트 임베딩 한계**: T5-base 수준의 언어 모델로, 더 큰 사전학습 언어 모델 사용 시 품질 향상 가능성 있음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 근거

- **다중 장르 학습**: Pop(27.29%), Electronic(19.38%), Rock(17.79%), Metal(8.92%), Hip Hop(4.06%) 등 다양한 장르로 학습되어, 단일 장르 모델(RAVE, Musika)보다 **장르 일반화** 능력이 높음.
- **4AFC 실험**: 4개 장르에 대한 혼동 행렬(confusion matrix)에서 Moûsai가 대각선에 가장 많은 질량을 보여, 장르별 구분 가능한 음악을 생성함을 입증. Riffusion은 대부분 Pop으로 분류됨.
- **장기 구조 일반화**: 세그먼트 태그("1 of 4", "2 of 4" 등)로 구간별 음악 생성 시, sonata form 구조(점진적 도입 → 본론 → 점진적 마무리)를 학습하여 보여줌.

### 3.2 일반화 성능 향상을 위한 방안

1. **데이터 규모 확장**: 논문에서 직접 제안하듯, 50K–100K 시간 규모 학습이 품질을 크게 향상시킬 가능성이 높음. 이는 Dhariwal et al. (2020)과 Borsos et al. (2022)의 관찰과 일치.

2. **더 큰 사전학습 언어 모델 활용**: Saharia et al. (2022)의 Imagen 연구에서, 더 큰 사전학습 언어 모델이 텍스트-이미지 정합성을 크게 개선함을 보였음. T5-base 대신 T5-XL, T5-XXL 등으로 교체 시 텍스트-음악 바인딩 일반화가 향상될 것으로 기대.

3. **어텐션 블록 증가**: 잠재 확산 모델의 어텐션 블록 수를 4–8개에서 32개 이상으로 증가시키면 **장기 구조 일반화**가 개선됨을 실험적으로 확인. 어텐션 블록 없이는 의미 있는 장기 구조를 학습하지 못함.

4. **지각적 손실 함수(Perceptual Loss)**: L2 대신 지각적 손실을 waveform에 적용하면 비인지적 소리의 불필요한 처리를 줄이고, **고주파 일반화**를 개선할 수 있음.

5. **Mel-spectrogram 사용**: magnitude spectrogram 대신 mel-spectrogram을 입력으로 사용하면 인간 청각 특성에 더 부합하는 표현 학습이 가능하여 일반화 향상 가능.

6. **다양한 조건부 방식**: 텍스트 이외에 DreamBooth 유사 접근법(Ruiz et al., 2022)으로 오디오 잠재 공간을 탐색하면, 텍스트로 표현하기 어려운 음악적 특성에 대한 일반화가 가능.

7. **압축률 조정**: 64배에서 32배로 압축률을 낮추면 저주파 품질이 향상되지만 속도가 감소하는 트레이드오프 존재. 지각적 손실과 결합 시 높은 압축률에서도 일반화 유지 가능.

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 연구에 미치는 영향

1. **확산 모델의 오디오 영역 확장**: 컴퓨터 비전에서 성공한 캐스케이딩 잠재 확산 기법을 오디오 생성에 최초 적용하여, 후속 연구(MusicGen, AudioLDM, MusicLM 등)의 기반을 마련.

2. **효율적 음악 생성의 민주화**: 단일 소비자 GPU에서 실시간 추론, 단일 A100에서 1주 훈련이라는 낮은 진입 장벽을 제시하여, 대학 연구실 수준에서도 접근 가능한 음악 생성 연구의 문을 열었음.

3. **오픈소스 생태계 기여**: 코드, 데이터 수집 파이프라인, 음악 샘플을 오픈소스로 공개하여 재현성과 후속 연구를 촉진.

4. **다중 모달리티 연구 촉진**: NLP와 음악의 교차점에서 연구를 수행하여, 텍스트-음악 외에도 텍스트-오디오, 텍스트-사운드스케이프 등 다중 모달리티 생성 연구에 영향.

### 4.2 향후 연구 시 고려할 점

1. **데이터 품질 및 규모**: 메타데이터 기반 텍스트 프롬프트(제목, 장르, 아티스트 등)는 자연어 설명보다 정보량이 제한적. 자연어 기반의 상세한 음악 설명 데이터셋 구축이 필요.

2. **평가 체계 표준화**: 음악 생성 분야에서 통일된 벤치마크가 부재하며, FAD, CLAP, 인간 평가 등이 혼용됨. 표준화된 평가 프레임워크 개발이 시급.

3. **저작권 및 윤리적 문제**: 학습 데이터의 저작권 문제, 생성된 음악의 기존 작품 유사성 탐지 메커니즘, 음악가와 작곡가에 대한 경제적 영향 등을 고려해야 함.

4. **고급 확산 샘플러 및 증류 기법**: 더 적은 샘플링 단계에서 높은 품질을 달성하기 위한 고급 ODE/SDE 솔버, progressive distillation (Salimans and Ho, 2022) 등의 적용.

5. **멀티모달 조건부 생성**: 텍스트뿐 아니라 이미지, 비디오, MIDI 등 다양한 모달리티를 조건으로 음악을 생성하는 방향으로 확장 가능.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 접근 방식 | 샘플레이트 | 텍스트 조건부 | 생성 길이 | 주요 특징 |
|------|------|---------|----------|-----------|---------|---------|
| **Jukebox** (Dhariwal et al., 2020) | 2020 | VQ-VAE + Autoregressive Transformer | 44.1kHz@1ch | 가사, 아티스트 | 분 단위 | 긴 음악 생성 가능하나 추론 수 시간 소요 |
| **RAVE** (Caillon & Esling, 2021) | 2021 | VAE 기반 실시간 합성 | 48kHz@2ch | 잠재 벡터 | 초 단위 | 단일 장르, 실시간 추론 가능 |
| **Musika** (Pasini & Schlüter, 2022) | 2022 | GAN 기반 | 22.5kHz@2ch | 컨텍스트 벡터 | 초 단위 | 단일 장르, 빠른 추론 |
| **Riffusion** (Forsgren & Martiros, 2022) | 2022 | Stable Diffusion (스펙트로그램 이미지) | 44.1kHz@1ch | 텍스트 | 5초 고정 | 텍스트-음악, 매우 짧은 생성 |
| **AudioLM** (Borsos et al., 2022) | 2022 | Language Modeling (SoundStream + w2v-BERT) | 16kHz@1ch | 음악 시작 부분 | 초 단위 | 텍스트 조건 없음, 높은 일관성 |
| **AudioGen** (Kreuk et al., 2022) | 2022 | Autoregressive Transformer + EnCodec | 16kHz@1ch | 텍스트 (일상 소리) | 초 단위 | 일반 소리 생성, 음악 미특화 |
| **Moûsai** (Schneider et al., 2023) | 2023 | **2단계 캐스케이딩 잠재 확산** | **48kHz@2ch** | **텍스트 (장르, 메타데이터)** | **분 단위** | **고효율, 다장르, 장기 구조** |
| **MusicLM**¹ (Agostinelli et al., 2023) | 2023 | AudioLM + MuLan 임베딩 | 24kHz | 텍스트 | 분 단위 | MuLan으로 텍스트-음악 정합, 고품질 |
| **MusicGen**² (Copet et al., 2023) | 2023 | Single-stage Transformer + EnCodec | 32kHz | 텍스트 또는 멜로디 | ~30초 | 단일 LM으로 효율적, 멜로디 조건부 가능 |
| **AudioLDM**³ (Liu et al., 2023) | 2023 | Latent Diffusion + CLAP | 16kHz | 텍스트 | 초 단위 | CLAP 기반 텍스트-오디오 정합, 일반 오디오 |
| **Stable Audio**⁴ (Evans et al., 2024) | 2024 | Latent Diffusion + Timing Conditioning | 44.1kHz@2ch | 텍스트 + 길이 | 최대 ~95초 | 타이밍 조건부, 높은 음질 |

### 주요 비교 관점

1. **효율성**: Moûsai는 Jukebox(수 시간)과 비교해 극적으로 빠른 추론 시간(RTF 1.14)을 달성. MusicGen도 효율적이나 단일 transformer 기반으로 접근 방식이 상이.

2. **품질 vs. 데이터 규모**: MusicLM(280K시간)과 MusicGen(20K시간)은 훨씬 큰 데이터셋으로 학습하여 음질이 높음. Moûsai(2.5K시간)는 상대적으로 소규모이지만, 구조적 혁신으로 경쟁력 있는 결과 달성.

3. **텍스트-음악 바인딩**: MusicLM은 MuLan 임베딩으로 강력한 텍스트-음악 정합을 달성. Moûsai는 T5 임베딩과 CFG를 사용하지만, CLAP 스코어(0.13)는 상대적으로 낮은 수준.

4. **장기 구조**: Moûsai의 세그먼트 태그 기반 접근법은 독창적이나, MusicGen이나 MusicLM 등은 자체적으로 장기 일관성을 학습.

5. **오픈소스 접근성**: Moûsai는 코드와 데이터 파이프라인을 공개한 초기 모델 중 하나로, 후속 연구에 실질적 기여.

---

## 참고자료

1. Schneider, F., Kamal, O., Jin, Z., & Schölkopf, B. (2023). "Moûsai: Efficient Text-to-Music Diffusion Models." *arXiv:2301.11757v3* [cs.CL].
2. Salimans, T. & Ho, J. (2022). "Progressive Distillation for Fast Sampling of Diffusion Models." *ICLR 2022*.
3. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." *ICLR 2021*.
4. Rombach, R. et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
5. Preechakul, K. et al. (2022). "Diffusion Autoencoders: Toward a Meaningful and Decodable Representation." *CVPR 2022*.
6. Saharia, C. et al. (2022). "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *arXiv:2205.11487*.
7. Dhariwal, P. et al. (2020). "Jukebox: A Generative Model for Music." *arXiv:2005.00341*.
8. Borsos, Z. et al. (2022). "AudioLM: A Language Modeling Approach to Audio Generation." *arXiv:2209.03143*.
9. Kreuk, F. et al. (2022). "AudioGen: Textually Guided Audio Generation." *arXiv:2209.15352*.
10. Forsgren, S. & Martiros, H. (2022). "Riffusion - Stable Diffusion for Real-time Music Generation."
11. Ho, J. & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." *arXiv:2207.12598*.
12. Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 21:140.
13. Wu, Y. et al. (2023). "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation."

**추가 참고 (2020년 이후 비교 분석용, 논문 본문에 직접 포함되지 않은 후속 연구):**

- ¹ Agostinelli, A. et al. (2023). "MusicLM: Generating Music From Text." *arXiv:2301.11325*.
- ² Copet, J. et al. (2023). "Simple and Controllable Music Generation." *NeurIPS 2023*. (*MusicGen*)
- ³ Liu, H. et al. (2023). "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models." *ICML 2023*.
- ⁴ Evans, Z. et al. (2024). "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion." *arXiv:2402.04825*.

> **주의**: 비교 분석 표에서 MusicLM, MusicGen, AudioLDM, Stable Audio에 대한 정보는 해당 논문들의 공개 자료에 기반한 것이며, Moûsai 논문 자체에는 포함되지 않은 후속 연구입니다. 이들의 정확한 수치 비교는 각 논문의 원문을 직접 확인하시기 바랍니다.
