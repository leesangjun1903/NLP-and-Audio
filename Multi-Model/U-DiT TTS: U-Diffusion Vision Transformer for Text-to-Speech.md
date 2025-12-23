# U-DiT TTS: U-Diffusion Vision Transformer for Text-to-Speech

### I. 논문의 핵심 주장과 주요 기여

**U-DiT TTS**는 텍스트-음성 합성(TTS) 분야에서 **Vision Transformer(ViT) 아키텍처를 확산 모델의 핵심 백본으로 적용**하는 혁신적인 접근을 제시한다. 기존의 확산 기반 TTS 시스템에서 U-Net 아키텍처가 지배적이었던 상황에서, 본 논문의 핵심 기여는 다음과 같다:[1]

1. **U-DiT 아키텍처 제안**: U-Net과 ViT의 장점을 결합한 모듈식 설계로, 다양한 데이터 규모에 걸쳐 확장성과 다재다능성을 제공[1]
2. **LJSpeech에서 SOTA 성능 달성**: Frechet Distance(FD) 메트릭과 Mean Opinion Score(MOS) 평가에서 Grad-TTS를 상회[1]
3. **일반화 성능 개선**: Latent space 활용과 적응형 계층 정규화(adaLN) 기법으로 훈련 안정성 향상

***

### II. 문제 정의 및 제안 방법

#### A. 해결하고자 하는 문제

**기존 U-Net 기반 확산 모델의 한계:**
- 고해상도 픽셀 공간에서 직접 diffusion을 수행하면 생성된 음성에 노이즈와 일관성 부족[1]
- U-Net의 down-sampling/up-sampling 과정에서 정보 손실 가능성
- 멜 스펙트로그램의 공간적 관계를 충분히 포착하지 못함

#### B. 제안하는 방법 및 수식

**1) Score-Based Generative Models(SGM) 기반 Forward/Reverse 과정:**

Score-based generative models는 Itô SDE(Stochastic Differential Equation)를 통해 확산 과정을 정의한다:[1]

$$d\mathbf{X}_t = f(\mathbf{X}_t, t)dt + g(t)d\mathbf{W}_t \quad (1)$$

여기서 $f$는 드리프트 계수, $g$는 확산 계수, $\mathbf{W}_t$는 Wiener 과정이다.[1]

대응하는 역방향 SDE는:[1]

$$d\mathbf{X}_t = \left[f(\mathbf{X}_t, t) - g(t)^2 \nabla_{\mathbf{X}_t} \log p_t(\mathbf{X}_t)\right]dt + g(t)d\overline{\mathbf{W}}_t \quad (2)$$

**2) TTS 시스템을 위한 Forward 과정:**

논문은 Grad-TTS의 forward diffusion 과정을 따르며, Ornstein-Uhlenbeck SDE로 정의된다:[1]

$$d\mathbf{X}_t = \frac{1}{2}\Lambda^{-1}(\boldsymbol{\mu} - \mathbf{X}_t)\beta_t dt + \sqrt{\beta_t}d\mathbf{W}_t, \quad t \in [0, T] \quad (3)$$

여기서 $\beta_t$는 노이즈 스케줄, $\boldsymbol{\mu}$는 벡터, $\Lambda$는 양수 원소를 가진 대각 행렬이다.

사전 분포는 닫힌 형태로 표현된다:[1]

$$p_{0t}\{\mathbf{X}_t|\mathbf{X}_0, \boldsymbol{\mu}\} = \mathcal{N}(\mathbf{X}_t; \rho(\mathbf{X}_0, \Lambda, \boldsymbol{\mu}, t), \delta(\Lambda, t)^2\mathbf{I}) \quad (4)$$

평균과 분산의 닫힌 형태 표현:[1]

$$\rho(\mathbf{X}_0, \Lambda, \boldsymbol{\mu}, t) = (I - e^{-\frac{1}{2}\Lambda^{-1}\int_0^t \beta_s ds})\boldsymbol{\mu} + e^{-\frac{1}{2}\Lambda^{-1}\int_0^t \beta_s ds}\mathbf{X}_0 \quad (5)$$

$$\delta(\Lambda, t)\mathbf{I} = \Lambda\left(I - e^{-\Lambda^{-1}\int_0^t \beta_s ds}\right) \quad (6)$$

**3) Reverse 과정 (ODE 형태):**

계산 속도를 개선하기 위해 Song et al.의 확률 흐름 ODE를 사용:[1]

$$d\mathbf{X}_t = \left[\frac{1}{2}\Lambda^{-1}(\boldsymbol{\mu} - \mathbf{X}_t) - \nabla \log p_t(\mathbf{X}_t)\right]\beta_t dt \quad (7)$$

#### C. 훈련 목적함수

**1) Encoder 손실:**[1]

$$L_{enc} = -\sum_{i=1}^{N} \log \phi(y_i; \tilde{\mu}_A^{(i)}, I) \quad (8)$$

로그 도메인에서 MSE 손실로 변환된다.

**2) Duration Predictor 손실:**[1]

$$d_i = \log \sum_{j=1}^{N_{freq}} \mathbb{1}_{A^*(j)=i}, \quad i = 1, ..., N_{freq}$$

$$L_{DP} = \text{MSE}(DP(sg[\tilde{\mu}], d)) \quad (9)$$

여기서 $sg[\cdot]$는 stop gradient 연산자이다.

**3) Diffusion 손실:**[1]

$$L_t(\mathbf{X}_0) = \mathbb{E}_{\epsilon_t}\left\|\mathbf{s}_\theta(\mathbf{X}_t, t) + \lambda(\Lambda, t)^{-1}\epsilon_t\right\|_2^2 \quad (10)$$

노이즈 분포를 단순화하여:

$$\mathbf{X}_t = \rho(\mathbf{X}_0, I, \boldsymbol{\mu}, t) + \sqrt{\lambda_t}\xi_t \quad (11)$$

$$\lambda_t = 1 - e^{\int_0^t \beta_s ds} \quad (12)$$

최종 diffusion 손실:[1]

$$L_{diff} = \mathbb{E}_{\mathbf{X}_0, t}\left[\lambda_t \mathbb{E}_{\xi_t}\left\|\mathbf{s}_\theta(\mathbf{X}_t, \boldsymbol{\mu}, t) + \frac{\xi_t}{\sqrt{\lambda_t}}\right\|_2^2\right] \quad (13)$$

***

### III. 모델 구조 (U-DiT 아키텍처)

#### A. 시스템 개요

U-DiT TTS 시스템은 다음 세 가지 핵심 구성요소로 이루어진다:[1]

1. **Text Encoder**: 6개의 Transformer 블록과 다중 헤드 자기 주의(MHSA) 및 최종 선형 투영 계층
2. **Duration Predictor**: Monotonic Alignment Search(MAS)를 통한 최적 정렬
3. **U-DiT Decoder**: DiT 블록을 통한 멜 스펙트로그램 생성

#### B. DiT 블록의 세부 설계[1]

**1) Patchify 및 위치 임베딩:**
- 입력을 패치 시퀀스로 변환 (패치 크기: )[2][3]
- 표준 주파수 기반 위치 임베딩 적용

**2) Adaptive Layer Normalization (adaLN-Zero):**

adaLN은 조건 벡터 $c$로부터 정규화의 스케일과 시프트 파라미터를 동적으로 생성한다:[1]

$$\mathbf{x}' = \gamma \cdot \frac{\mathbf{x} - \mu(\mathbf{x})}{\sigma(\mathbf{x})} + \beta$$

여기서 $\gamma$와 $\beta$는 4계층 MLP를 통해 시간 임베딩 $t_e$와 라벨 임베딩 $l_e$의 합으로부터 회귀된다.

**3) Latent Space 설계:**

고해상도 픽셀 공간에서의 직접 diffusion은 노이즈가 많고 일관성이 부족했으므로, U-Net의 downsampling/upsampling 구성요소를 활용하여 입력 스펙트로그램을 latent space로 변환:[1]

- Downsampling: 다중 residual 블록 + 그룹 정규화 + 자기 주의 계층
- Latent 특징 → 작은 패치로 분할 → sinusoidal 위치 임베딩
- Upsampling: Downsampling과 대칭적 구조

#### C. 모델 아키텍처 구성[1]

- **Text Encoder**: Pre-net(3개 conv층 + FC) + 6 transformer 블록 + 선형 투영
- **Duration Predictor**: FastSpeech 2 기반 MSE 손실
- **Decoder**: 2계층 U-Net down/upsampling + 2~8개 DiT 블록 (실험 결과 2개가 최적)
- **Vocoder**: 사전훈련된 HiFi-GAN

***

### IV. 성능 향상 및 한계

#### A. 객관적 평가 결과 (LJSpeech 테스트 셋)[1]

| 메트릭 | FD ↓ | LSD ↓ | KLD ↓ | MOS |
|--------|------|-------|-------|-----|
| Ground Truth (GT) | - | - | - | 4.10 |
| GT mel (mel→HiFi-GAN→mel) | 0.6985 | 1.8202 | 0.0109 | 3.90 |
| Grad-TTS (baseline) | 3.0046 | 2.1765 | 0.0647 | 3.62 |
| **U-DiT TTS** | **0.8960** | 2.1745 | 0.0316 | **3.91** |

U-DiT TTS는 Grad-TTS 대비 **FD에서 70.2% 개선**, MOS에서 **0.29 포인트 향상**을 달성했다. KLD도 51.2% 감소하여 생성된 음성이 실제 음성과 더 유사한 분포를 갖추었다.[1]

#### B. Ablation Study - 역 스텝과 온도 파라미터[1]

**역 스텝에 따른 성능:**

역 스텝이 80~150 사이일 때 **고품질과 효율성의 최적 절충**을 달성:[1]

- 30 스텝: 허용할 수 있는 음성 품질, 빠른 합성
- 80 스텝: 최적 FD (0.0275) 및 KLD 달성
- 150 스텝: 200 스텝 이상에서 수렴

**온도 파라미터 τ의 영향:**

τ = 1.1~2.0 범위에서 **최적 성능**:
- FD 지속적 감소
- KLD 미미한 증가
- τ = 1.5에서 최종 선택[1]

#### C. 모델의 일반화 성능 향상[1]

**1) Latent Space 활용의 효과:**

고해상도 픽셀 공간에서의 훈련 실패 → latent space 도입으로:
- 공간적 관계 캡처 능력 향상
- 배경 및 시스템 노이즈 감소
- 생성된 멜 스펙트로그램의 명확성 개선

**2) Adaptive Layer Normalization의 역할:**

- 더 부드러운 그래디언트와 개선된 일반화 정확도[1]
- Zero-initialization 전략과 결합하여 대규모 훈련 가속

**3) 다양한 데이터 규모에서의 확장성:**

U-DiT의 모듈식 설계는:
- 다양한 데이터 스케일에 걸친 **유연한 적응** 가능
- 서로 다른 모델 크기(2, 4, 8 DiT 블록)에서의 성능 검증
- 최종적으로 **2개 블록이 최적** (데이터 제약으로 더 큰 모델은 성능 악화)[1]

#### D. 한계점

1. **고정 입력 크기 제약:** Transformer 아키텍처의 기본 제한으로 인한 세그멘테이션 필요[1]
2. **훈련 데이터 품질의 높은 요구:** LJSpeech와 같은 고품질 단일 화자 데이터셋에서의 최적화
3. **제한된 다중 화자 성능:** 현재 시스템은 단일 화자 데이터셋에서 주로 평가
4. **계산 효율:** HiFi-GAN vocoder의 부분적 오디오 복원 한계로 인한 미미한 MOS 차이

***

### V. 모델의 일반화 성능 향상 가능성

#### A. 현재 강점

**1) Spatial Relationship Modeling:**

U-DiT는 ViT의 글로벌 주의 메커니즘을 통해 멜 스펙트로그램의 공간적 관계를 더 잘 포착한다. 이는 Grad-TTS의 U-Net 기반 로컬 특징 추출과 대조된다.[1]

**2) Latent Space 일반화:**

Latent space에서의 학습은:
- 의미론적 정보에 집중 가능
- 고주파 세부사항의 노이즈 감소
- 서로 다른 데이터 도메인 간 **더 나은 전이 학습** 잠재력

**3) Modular Design Benefits:**

- 다양한 데이터 스케일에 대한 적응성
- 하이퍼파라미터 조정을 통한 성능 미세조정 가능
- 복합 음성 특성(피치, 에너지, 듀레이션) 동시 모델링

#### B. 향후 개선 가능성

**1) Multi-Speaker/Multilingual 확장:**

현재 제한사항:
- LJSpeech(단일 화자, 24시간)에서만 SOTA 달성
- 다중 화자 데이터셋에서의 성능 평가 부재

개선 방향:
- VCTK 등 다중 화자 데이터셋에서의 벤치마킹
- Speaker embedding 통합을 통한 **화자 다양성 향상**
- Cross-lingual 일반화 능력 강화

**2) Fixed Input Size 해결:**

현재 제약:
- Transformer의 고정된 시퀀스 길이 요구
- 음성 세그멘테이션의 필요성

해결 방안:
- **Dynamic patching 또는 sparse attention** 도입
- Relative position bias를 통한 가변 길이 지원
- Rotary position embeddings (RoPE) 활용[4]

**3) Domain Adaptation:**

- 사전훈련된 모델의 **few-shot fine-tuning** 가능성
- 다양한 음성 도메인(감정, 방언, 배경음)에 대한 **건강한 일반화**

***

### VI. 최신 연구(2020년 이후) 비교 분석

#### A. U-DiT vs. 관련 최신 연구

| 논문 | 발표 | 백본 | 핵심 개선 | 성능(WER/FD) |
|------|------|------|----------|-------------|
| **U-DiT TTS** | 2023 | ViT + U-Net | adaLN-Zero, Latent space | FD: 0.896, MOS: 3.91[1] |
| **DiTTo-TTS**[4] | 2025 | Pure DiT | Variable-length predictor, No phonemes | WER: 1.78(cont), 2.56(cross)[4] |
| **DPI-TTS**[5] | 2024 | DiT | Directional patch interaction, Acoustic properties | 2x 훈련 속도, WER: 6.57(LJ)[5] |
| **Grad-TTS** | 2021 | U-Net | Score-based generative model | FD: 3.00[1] |
| **ViT-TTS** | 2023 | ViT | Visual features, scalable | SOTA image-gen 영감[6] |

#### B. 주요 차이점

**1) Architecture Evolution:**
- **Grad-TTS (2021)**: U-Net 확산 모델, 기준선
- **U-DiT TTS (2023)**: U-Net + ViT 결합, adaLN 도입
- **DiTTo-TTS (2025)**: Pure DiT, phoneme/duration 제거, 대규모 데이터(82K hrs)

**2) Generalization Focus:**
- U-DiT: 단일 화자 고품질 음성 합성에 최적화
- DiTTo: **Domain-specific factors 없이 zero-shot 성능** (더 나은 일반화)[4]
- DPI-TTS: **음향 특성 활용**으로 자연스러운 음성 생성[5]

**3) 성능 메트릭:**

| 메트릭 | U-DiT | DiTTo-B | DPI-TTS |
|--------|-------|---------|---------|
| FD/WER | 0.896 | - | - |
| MOS | 3.91 | - | 4.38(VCTK) |
| 훈련 속도 | baseline | 빠름[4] | 2배[5] |

#### C. DiTTo-TTS (2025)의 일반화 우월성[4]

DiTTo-TTS는 U-DiT보다 **더 강력한 일반화 능력**을 시연:

- **Zero-shot 성능**: Phoneme/duration 없이도 SOTA 달성
- **Scale up**: 790M 파라미터, 82K 시간 데이터로 훈련
- **Multilingual**: 9개 언어에서 SOTA 또는 비교 가능한 성능[4]

반면 U-DiT는:
- LJSpeech 단일 데이터셋에서 최적화
- 다중 화자/언어 평가 부재
- Phoneme 및 duration 요구

***

### VII. 향후 연구에 미치는 영향 및 고려사항

#### A. 학술적 영향

**1) Vision Transformer의 음성 처리 적용:**

U-DiT는 이미지 도메인의 성공적인 ViT 기법을 음성 합성에 **최초로 체계적으로 적용**했다. 이는 다음 연구의 토대 마련:[1]
- DiTTo-TTS의 Pure DiT 아키텍처 개발[4]
- DPI-TTS의 음향 특성 통합[5]

**2) Diffusion Model 설계 원칙:**

- **Latent space 학습의 중요성**: 픽셀 공간 대비 압축된 표현이 생성 품질 향상[1]
- **Adaptive Conditioning의 효과**: adaLN-Zero가 대규모 모델 훈련 안정화[1]
- **모듈식 아키텍처의 확장성**: 다양한 데이터 규모에서의 유연한 적응[1]

#### B. 실무적 고려사항

**1) 배포 최적화:**

| 측면 | U-DiT의 한계 | 개선 방안 |
|------|------------|---------|
| 실시간 성능 | 30 역 스텝으로도 지연 | 12-25 스텝 가속 확산 연구[5] |
| 메모리 효율 | 2 DiT 블록 필수 | Flash Attention, 양자화 적용 |
| 다중 화자 | 미지원 | Speaker embedding 추가[4] |

**2) 데이터 요구사항:**

U-DiT:
- 고품질 단일 화자 데이터 필수 (LJSpeech 24시간)
- 낮은 품질 데이터에서 성능 저하[1]

개선:
- **합성 데이터 활용** (TacotronX 등)
- **약한 지도학습**(phoneme-less training)[4]

#### C. 미래 연구 방향

**1) Hybrid Architecture 탐색:**

- U-Net의 multi-scale 장점 + ViT의 글로벌 컨텍스트[1]
- DPI-TTS처럼 **음향 특성 기반 attention**[5]

**2) Cross-Modal Generalization:**

- 음성 → 텍스트 정렬 개선을 통한 제로샷 성능[4]
- Semantic alignment via pretrained speech/text encoders[4]

**3) Efficient Diffusion:**

- Flow Matching 기반 단계 감소[4]
- Consistency Distillation으로 단계 압축

**4) Expressiveness Enhancement:**

- Style temporal modeling (DPI-TTS)[5]
- Emotion, prosody 세밀 제어

***

### VIII. 결론

U-DiT TTS는 **Vision Transformer를 음성 합성에 적용한 선구적 연구**로서, U-Net 기반 확산 모델의 한계를 극복하는 효과적인 방법을 제시했다. Latent space 학습, adaLN 기반 조건화, 모듈식 설계를 통해 **LJSpeech에서 state-of-the-art 성능**을 달성했다.[1]

그러나 **일반화 성능**의 관점에서는 후속 연구인 **DiTTo-TTS(2025)**가 phoneme/duration 제거, 대규모 데이터 활용, zero-shot 멀티링구얼 지원으로 더 우수한 일반화 능력을 시연했다. **DPI-TTS(2024)**는 음향 특성 기반 설계로 훈련 효율과 자연스러운 음성 생성을 동시에 달성했다.[5][4]

향후 연구는:
1. **Fixed input size 제약 해결** (동적 시퀀스 길이 지원)
2. **Multi-speaker/multilingual 확장** (대규모 데이터셋 활용)
3. **Domain-specific factors 제거** (DiTTo 방향)
4. **음향 특성 통합** (DPI 방향)

을 중점적으로 추진해야 하며, U-DiT의 **핵심 아이디어인 ViT 아키텍처 적용**은 음성 처리의 future direction으로 자리매김했다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f68407f-fb3f-44f8-bf54-0b80777cb7ef/2305.13195v1.pdf)
[2](https://arxiv.org/pdf/2508.10949.pdf)
[3](https://arxiv.org/html/2503.22732v1)
[4](http://arxiv.org/pdf/2406.11427.pdf)
[5](https://arxiv.org/pdf/2308.16569.pdf)
[6](https://arxiv.org/abs/2406.19135)
[7](http://arxiv.org/pdf/2309.06787.pdf)
[8](http://arxiv.org/pdf/2312.03491.pdf)
[9](http://arxiv.org/pdf/2211.09383.pdf)
[10](https://arxiv.org/html/2409.11835)
[11](https://aclanthology.org/2023.emnlp-main.990.pdf)
[12](https://www.isca-archive.org/interspeech_2025/choi25c_interspeech.pdf)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC11034915/)
[14](https://www.isca-archive.org/interspeech_2019/hao19_interspeech.pdf)
[15](https://reverieinc.com/blog/role-of-diffusion-models-in-advancing-text-to-speech-technology/)
[16](https://yiyibooks.cn/__src__/arxiv/2305.13195v1/index.html)
[17](https://arxiv.org/pdf/2012.03594.pdf)
[18](https://aclanthology.org/2025.coling-main.352.pdf)
[19](https://pubmed.ncbi.nlm.nih.gov/40039596/)
[20](https://s.makino.w.waseda.jp/reprint/Makino/YuehaiZhang25apsipa819-824.pdf)
[21](https://www.isca-archive.org/interspeech_2025/chen25b_interspeech.pdf)
[22](https://arxiv.org/html/2502.03930v4)
[23](https://arxiv.org/html/2508.07558v1)
[24](https://arxiv.org/html/2508.10949)
[25](https://arxiv.org/abs/2312.06613)
[26](https://arxiv.org/html/2511.13936v1)
[27](https://arxiv.org/abs/2305.13195)
[28](https://arxiv.org/pdf/2307.14464.pdf)
[29](https://arxiv.org/pdf/2511.13936.pdf)
[30](https://wikidocs.net/237410)
[31](http://arxiv.org/abs/2305.13195)
[32](https://www.hse.ru/data/2024/10/04/1888260947/%D0%90%D0%BD%D0%B4%D1%80%D0%B5%D0%B5%D0%B2_summary.pdf)
[33](https://ieeexplore.ieee.org/document/8461368/)
[34](https://ieeexplore.ieee.org/document/11002503/)
[35](https://ieeexplore.ieee.org/document/9743540/)
[36](https://ieeexplore.ieee.org/document/9746582/)
[37](https://arxiv.org/abs/2207.01454)
[38](https://www.isca-archive.org/interspeech_2021/kongthaworn21_interspeech.html)
[39](https://arxiv.org/abs/2203.01080)
[40](https://www.semanticscholar.org/paper/dee0e9495e401b1d4f22c66d4d8a001950bba61d)
[41](https://www.frontiersin.org/articles/10.3389/frai.2024.1499913/full)
[42](http://arxiv.org/pdf/2206.15276.pdf)
[43](https://arxiv.org/pdf/2212.14518.pdf)
[44](http://arxiv.org/pdf/2406.05298.pdf)
[45](https://arxiv.org/pdf/2206.02512.pdf)
[46](http://arxiv.org/pdf/2204.00768.pdf)
[47](http://arxiv.org/pdf/2407.08551.pdf)
[48](http://arxiv.org/pdf/2312.14569.pdf)
[49](https://arxiv.org/pdf/1905.09263.pdf)
[50](https://pypi.org/project/tts-scores/)
[51](https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/practice-building-dit-block)
[52](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse)
[53](https://github.com/neonbjb/tts-scores/blob/main/README.md)
[54](https://leeyngdo.github.io/blog/generative-model/2024-07-01-diffusion-transformer/)
[55](https://www.isca-archive.org/interspeech_2025/sun25f_interspeech.pdf)
[56](https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf)
[57](https://blog.csdn.net/suiyueruge1314/article/details/148015643)
[58](https://arxiv.org/html/2506.08457v1)
[59](https://xai.kaist.ac.kr/static/files/2025_hcai_workshop/paper_14.pdf)
[60](https://arxiv.org/pdf/2112.03099.pdf)
[61](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
[62](https://www.arxiv.org/pdf/2510.26193.pdf)
[63](https://arxiv.org/pdf/2505.07701.pdf)
[64](https://arxiv.org/html/2509.24579)
[65](https://arxiv.org/pdf/2207.09983.pdf)
[66](https://arxiv.org/html/2312.04557v1)
[67](https://arxiv.org/pdf/2502.00336.pdf)
[68](https://arxiv.org/pdf/2305.07243.pdf)
