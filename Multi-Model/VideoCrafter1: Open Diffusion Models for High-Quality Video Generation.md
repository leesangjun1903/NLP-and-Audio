# VideoCrafter1: Open Diffusion Models for High-Quality Video Generation

---

## 1. 핵심 주장 및 주요 기여 요약

VideoCrafter1은 고품질 비디오 생성을 위한 두 가지 확산 모델을 제안한다: **텍스트-투-비디오(T2V)**와 **이미지-투-비디오(I2V)** 모델이다. T2V 모델은 텍스트 입력으로부터 비디오를 합성하고, I2V 모델은 추가적으로 이미지 입력을 받아 비디오를 생성한다. 제안된 T2V 모델은 $1024 \times 576$ 해상도의 사실적이고 영화 수준 품질의 비디오를 생성하며, 다른 오픈소스 T2V 모델 대비 품질에서 우수한 성능을 보인다.

I2V 모델은 제공된 참조 이미지의 내용, 구조, 스타일을 충실히 유지하면서 비디오를 생성하도록 설계되었으며, 이는 콘텐츠 보존 제약 하에서 이미지를 비디오 클립으로 변환할 수 있는 **최초의 오픈소스 I2V 파운데이션 모델**이다.

**주요 기여:**
1. 상업적 도구는 그럴듯한 비디오를 생성할 수 있지만, 연구자와 엔지니어가 사용 가능한 오픈소스 모델은 제한적이었다. VideoCrafter1은 이 격차를 해소하는 오픈소스 비디오 생성 프레임워크를 제시함.
2. T2V 모델이 $1024 \times 576$ 해상도에서 사실적이고 시네마틱 품질의 비디오를 생성하며, 다른 오픈소스 T2V 모델을 품질 면에서 능가함.
3. I2V 모델은 참조 이미지의 콘텐츠, 구조, 스타일을 보존하면서 비디오를 생성하는 최초의 오픈소스 I2V 파운데이션 모델.
4. 이러한 오픈소스 비디오 생성 모델이 커뮤니티의 기술적 발전에 크게 기여할 것이라고 기대함.

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

텍스트-투-이미지 생성 모델(특히 Stable Diffusion)의 성공과 그 활발한 생태계는 오픈소스 모델과 활발한 커뮤니티에 기인하지만, 고품질 비디오 생성은 아직 오픈소스 커뮤니티에서 충분히 발전하지 못했으며 주로 스타트업 기업에 국한되어 있었다.

핵심 도전 과제는 다음과 같다:
- 비디오 생성은 이미지 생성의 상위 집합으로, 시간에 걸친 프레임 간 시간적 일관성(temporal consistency)에 대한 추가 요구사항이 있으며, 이는 자연스럽게 더 많은 세계 지식(world knowledge)이 모델에 인코딩되어야 함을 의미한다.
- 텍스트나 이미지에 비해 대량의 고품질, 고차원 비디오 데이터 수집이 훨씬 어렵다.
- 상업적 비디오 모델은 대규모의 잘 필터링된 고품질 비디오 데이터에 의존하며, 이는 커뮤니티에서 접근할 수 없다. 많은 기존 연구들은 저품질 WebVid-10M 데이터셋으로 모델을 학습하여 고품질 비디오 생성에 어려움을 겪는다.

### 2.2 제안하는 방법 (수식 포함)

VideoCrafter1은 **Latent Diffusion Model (LDM)** 프레임워크를 기반으로 비디오 생성을 수행한다. 핵심 수식 체계는 아래와 같다.

#### (1) Forward Diffusion Process (순방향 확산 과정)

원본 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가하는 과정:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I})$$

전체 $T$ 스텝 후의 분포는:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\mathbf{I})$$

여기서 $\alpha_t = 1-\beta_t$, $\bar{\alpha}\_t = \prod_{s=1}^{t}\alpha_s$이다.

#### (2) Reverse Denoising Process (역방향 디노이징)

노이즈 예측 네트워크 $\epsilon_\theta$를 학습하여 디노이징:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2\mathbf{I}\right)$$

#### (3) Latent Space에서의 학습 목표 (Training Objective)

VideoCrafter1은 오토인코더의 잠재 공간(latent space)에서 비디오 UNet을 학습하며, FPS를 조건으로 사용하여 생성 비디오의 모션 속도를 제어한다.

학습 목표 함수(Loss Function)는 표준 DDPM의 noise-prediction loss를 따른다:

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,\mathbf{I}), t, c}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

여기서:
- $z_0 = \mathcal{E}(x_0)$: 오토인코더 $\mathcal{E}$를 통해 얻은 잠재 표현
- $z_t$: 시간 $t$에서의 노이즈가 추가된 잠재 표현
- $c$: 텍스트/이미지 조건 (CLIP 임베딩 등)
- $\epsilon_\theta$: UNet 기반 노이즈 예측 네트워크

#### (4) Classifier-Free Guidance (CFG)

추론(inference) 시 텍스트 조건의 영향력을 조절:

$$\hat{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \varnothing) + s \cdot \left(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing)\right)$$

여기서 $s$는 가이던스 스케일, $\varnothing$는 빈 조건을 의미한다.

### 2.3 모델 구조

VideoCrafter1의 프레임워크는 오토인코더의 잠재 공간에서 비디오 UNet을 학습시키는 구조이다. FPS를 조건으로 사용하여 모션 속도를 제어하고, T2V 모델에서는 텍스트 프롬프트만 spatial transformer에 cross-attention으로 입력되며, I2V 모델에서는 텍스트와 이미지 프롬프트 모두 입력으로 사용된다.

#### 구조적 세부사항:

| 구성 요소 | 설명 |
|---|---|
| **Backbone** | 시공간(space-time)으로 분해된(factorized) 특정 유형의 3D-UNet을 사용 |
| **Spatial Module** | Stable Diffusion 사전학습 가중치를 활용한 2D 공간 처리 블록 |
| **Temporal Module** | 시간적 어텐션 레이어(temporal attention layer)를 통해 생성 비디오의 시간적 일관성을 보장하여 부자연스러운 점프나 정지를 방지 |
| **Encoder (VAE)** | 비디오 프레임을 잠재 공간으로 인코딩 |
| **Text Encoder** | CLIP 텍스트 인코더를 사용한 텍스트 조건화 |
| **Image Encoder (I2V)** | CLIP 알고리즘으로 이미지 특징(image embedding)을 추출하고, 이 이미지 임베딩을 cross-attention 메커니즘을 통해 SD UNet에 주입 |
| **FPS Conditioning** | 모션 속도를 명시적으로 제어 |

#### 학습 전략:

두 가지 학습 전략이 존재한다: (1) 공간 모듈을 고정하고 비디오로 시간 모듈만 학습하는 **부분 학습(partial training)**, (2) SD 가중치를 초기화로 사용하여 공간과 시간 모듈을 모두 학습하는 **전체 학습(full training)**.

VideoCrafter2 논문에서는 오픈소스 VideoCrafter1의 아키텍처에 FPS 조건을 적용하고, ModelScopeT2V의 시간적 합성곱(temporal convolution)도 통합하여 시간적 일관성을 향상시켰다.

### 2.4 성능 향상

제안된 T2V 모델은 $1024 \times 576$ 해상도에서 사실적이고 시네마틱 품질의 비디오를 생성하며, 다른 오픈소스 T2V 모델을 품질 면에서 능가한다.

VideoCrafter1은 Tencent AI Lab에서 개발한 고품질 비디오 생성 모델이다. 주요 성능 포인트:

- **해상도**: $1024 \times 576$ (당시 오픈소스 최고 수준)
- **시간적 일관성**: Temporal attention을 통한 프레임 간 일관성 확보
- **콘텐츠 보존 (I2V)**: 참조 이미지의 내용/구조/스타일을 충실히 유지
- **유연성**: FPS 조건화를 통한 모션 속도 제어

### 2.5 한계

1. **학습 데이터 품질**: WebVid-10M 같은 저품질 데이터셋으로 학습된 모델은 고품질 비디오 생성에 어려움을 겪는다. VideoCrafter1도 이 한계에서 완전히 자유롭지 못함.

2. **비디오 길이 제한**: 원칙적으로 비디오를 임의로 확장할 수 있지만, 시간이 지남에 따라 반복과 품질 저하 문제가 발생한다.

3. **계산 비용**: 완전한 3D 아키텍처는 매우 높은 계산 비용과 연관된다. Factorized attention으로 이를 완화하지만 근본적 한계가 있음.

4. **공간-시간 모듈 간 간섭**: 전체 학습된 모델의 시공간 모듈 분포가 더 강하게 결합되어 있어, 분포 이동(distribution shift)이 더 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Stable Diffusion 사전학습의 활용

VideoCrafter1의 일반화 성능의 핵심은 **Stable Diffusion의 사전학습 가중치를 활용한 transfer learning**에 있다.

사전학습된 이미지-텍스트 확산 모델에 시간적 레이어를 삽입하여 "inflate"하는 접근법은, 텍스트-이미지 쌍의 사전 지식을 새 모델이 상속하므로 텍스트-비디오 쌍 데이터에 대한 요구를 완화할 수 있다.

이는 일반화에 다음과 같은 이점을 제공한다:
- 이미지 도메인에서 학습된 풍부한 시각적 의미론(semantics)이 비디오로 전이
- 다양한 스타일과 도메인에 대한 기본적 이해력 확보
- 학습 데이터가 부족한 비디오 도메인에서도 안정적 성능

### 3.2 공간-시간 분해(Factorized Space-Time) 아키텍처

분해된 시공간 어텐션(factorized space-time attention)은 비디오 트랜스포머에서 계산 효율성이 뛰어나며, 시간 어텐션 블록 내의 어텐션 연산을 마스킹하여 비디오 대신 독립적인 이미지에서도 실행할 수 있어, 비디오와 이미지 생성을 공동으로 학습할 수 있다.

이러한 공동 학습(joint training)은 샘플 품질에 중요한 역할을 한다.

이 구조가 일반화에 미치는 영향:

$$\text{UNet}_{3D}(z_t, t, c) = \text{SpatialBlock}(z_t) \circ \text{TemporalBlock}(z_t) \circ \text{CrossAttn}(z_t, c)$$

- 공간 모듈과 시간 모듈의 분리로 각각 독립적으로 최적화 가능
- 이미지-비디오 공동 학습으로 더 넓은 시각적 분포를 커버

### 3.3 데이터 수준의 일반화 전략

VideoCrafter2에서는 Stable Diffusion 확장 비디오 모델의 학습 체계를 탐구하여, 저품질 비디오와 합성된 고품질 이미지를 활용해 고품질 비디오 모델을 얻는 가능성을 조사하여, 범용 고품질 비디오 모델을 도출했다.

이는 VideoCrafter1의 일반화 한계를 극복하기 위한 방향으로:
- 1단계에서 대량의 저품질 비디오로 VDM을 완전 학습하여 모션을 학습하고, 2단계에서 T2I 모델이 생성한 고품질 이미지로 공간 파라미터만 학습하여 품질을 향상.
- 데이터 수준에서 외관(appearance)과 모션(motion)을 분리하여 일반화 성능 향상

### 3.4 FPS 조건화를 통한 일반화

FPS를 조건으로 사용하여 생성 비디오의 모션 속도를 제어할 수 있으며, 이는 다양한 속도의 비디오 생성에 대한 일반화를 가능하게 한다. 조건화 수식:

$$\epsilon_\theta(z_t, t, c_{\text{text}}, c_{\text{fps}})$$

### 3.5 LoRA 기반 커스터마이제이션

사전학습된 T2V 모델과 VideoLora 모델이 공개되어, LoRA(Low-Rank Adaptation)를 통한 효율적 미세 조정으로 특정 도메인에 대한 일반화 성능을 확장할 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **오픈소스 비디오 생성 생태계 촉진**: Stable Diffusion이 이미지 생성 분야에서 촉발한 것과 유사하게, VideoCrafter1은 비디오 생성 분야의 오픈소스 생태계를 위한 기초를 마련했으며, 연구와 실제 제작 모두에 기여하는 고급 사용자 친화적 비디오 생성 모델을 목표로 한다.

2. **후속 연구의 기반**: VideoCrafter2는 오픈소스 VideoCrafter1의 아키텍처를 따르며, DynamiCrafter, EvalCrafter 등 다수의 후속 연구에 기반을 제공했다.

3. **I2V 연구 활성화**: 최초의 오픈소스 I2V 파운데이션 모델로서, 이미지에서 비디오 클립으로의 변환이 가능하여 이 분야의 연구를 크게 촉진시켰다.

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **데이터 품질 vs. 규모** | 저품질 대규모 데이터와 고품질 소규모 데이터 간의 균형 전략 필요 |
| **아키텍처 진화** | UNet 기반에서 DiT(Diffusion Transformer) 기반으로의 전환 검토 |
| **시간적 일관성** | 다수 계층의 시간적 어텐션 맵 간 상당한 불일치가 구조적으로 불합리하거나 시간적으로 비일관적인 비디오 출력을 초래할 수 있음 |
| **장기 비디오 생성** | 장기적 시간 의존성 모델링이 아직 신뢰할 수 있는 수준에 도달하지 못했으며, 자기회귀 및 계층적 업샘플링 기법이 아티팩트와 시간에 따른 품질 저하를 야기할 수 있음 |
| **평가 지표** | FVD, FID 외에 인간 선호도 기반 평가 체계의 정교화 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 아키텍처 | 핵심 특징 | 해상도 |
|---|---|---|---|---|
| **VDM** (Ho et al.) | 2022 | 3D U-Net (factorized over space and time) | 최초의 비디오 확산 모델, 공동 비디오-이미지 학습 | 64×64 → 128×128 |
| **Make-A-Video** | 2022 | Inflated T2I 모델 | 사전학습된 확산 이미지 모델을 시간 차원으로 확장 | ~768p |
| **Imagen Video** | 2022 | Cascaded Diffusion | 픽셀 레벨에서 동작하는 캐스케이드 확산 모델 | 1280×768 |
| **LAVIE** | 2023 | Cascaded Latent Diffusion | 캐스케이드 접근법으로 기본 생성과 초해상도 단계를 결합 | 512×320+ |
| **VideoCrafter1** | 2023 | 3D-UNet (Factorized ST) | 오픈소스 T2V+I2V, FPS 조건화 | $1024 \times 576$ |
| **AnimateDiff** | 2023 | Temporal Attn + T2I | 고정된 텍스트-이미지 모델에 학습 가능한 시간적 어텐션 레이어를 도입하여 프레임 간 상관관계를 포착 | SD 해상도 |
| **Show-1** | 2023 | Hybrid Pixel+Latent | 픽셀 기반과 잠재 기반 VDM을 결합한 최초의 하이브리드 모델 | 다단계 |
| **VideoCrafter2** | 2024 | Enhanced 3D-UNet | 저품질 비디오와 합성된 고품질 이미지를 활용한 학습 체계 탐구 | 512×320 |
| **Lumiere** | 2024 | Space-Time UNet (STUNet) | UNet에 시간적 다운샘플링/업샘플링을 추가하여 시공간 정보를 압축, 계산 비용 절감과 시간적 일관성 향상을 동시에 달성 | 고해상도 |
| **Sora** (OpenAI) | 2024 | DiT (Diffusion Transformer) | 시공간 패치(spacetime patches) 위에서 동작하는 DiT 아키텍처를 활용하며, 시각적 입력을 트랜스포머 토큰으로 표현 | 최대 1080p, 최대 1분 |
| **Open-Sora 2.0** | 2025 | DiT + Video DC-AE | 110억 파라미터 규모로 확장, HunyuanVideo 및 Step-Video에 필적하는 벤치마크 성능을 달성하며, 상업급 모델 학습 비용을 약 $200k로 추정 | ~768p |

### 아키텍처 패러다임 변화 추이

```
2022: VDM (3D-UNet)
  ↓
2023: VideoCrafter1 / LAVIE / AnimateDiff (Factorized 3D-UNet + SD 기반)
  ↓
2024: Sora / Lumiere (DiT / STUNet — 패러다임 전환)
  ↓
2025: Open-Sora 2.0 / HunyuanVideo (대규모 DiT + 효율적 VAE)
```

Sora는 다양한 기간, 종횡비, 해상도의 비디오와 이미지를 생성할 수 있는 시각 데이터의 범용 모델로, LLM의 패러다임에서 영감을 받아 visual patches라는 효과적인 표현을 사용하여 다양한 유형의 비디오와 이미지에 대한 생성 모델 학습에서 뛰어난 확장성을 보인다.

### VideoCrafter1의 위치와 의의

VideoCrafter1은 **UNet 기반 비디오 확산 모델의 오픈소스화**를 이끈 중요한 전환점이었다. 이후 DiT 기반 아키텍처로의 패러다임 전환이 이루어졌지만, VideoCrafter1이 확립한 다음 원칙들은 여전히 유효하다:

1. **사전학습된 이미지 모델의 활용**: T2I → T2V 전이 학습
2. **시공간 분해**: 공간/시간 모듈의 효율적 분리
3. **조건화 전략**: FPS, 이미지, 텍스트 등 다양한 조건 통합
4. **오픈소스 접근**: 커뮤니티 기여를 통한 빠른 기술 발전

---

## 참고자료 및 출처

1. **Chen, H., et al.** "VideoCrafter1: Open Diffusion Models for High-Quality Video Generation." arXiv preprint arXiv:2310.19512 (2023). — [arxiv.org/abs/2310.19512](https://arxiv.org/abs/2310.19512)
2. **Chen, H., et al.** "VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models." CVPR 2024. — [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_VideoCrafter2_Overcoming_Data_Limitations_for_High-Quality_Video_Diffusion_Models_CVPR_2024_paper.pdf)
3. **VideoCrafter1 공식 프로젝트 페이지** — [ailab-cvc.github.io/videocrafter1](https://ailab-cvc.github.io/videocrafter1/)
4. **GitHub: AILab-CVC/VideoCrafter** — [github.com/AILab-CVC/VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)
5. **Ho, J., Salimans, T., et al.** "Video Diffusion Models." NeurIPS 2022. — [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2022/file/39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf)
6. **Brooks, T., et al.** "Video generation models as world simulators." OpenAI (2024). — [openai.com](https://openai.com/index/video-generation-models-as-world-simulators/)
7. **Lilian Weng.** "Diffusion Models for Video Generation." Lil'Log, April 2024. — [lilianweng.github.io](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
8. **Open-Sora 2.0** — [arxiv.org/abs/2503.09642](https://arxiv.org/html/2503.09642v1)
9. **Zheng, Z., et al.** "Open-Sora: Democratizing Efficient Video Production for All." arXiv:2412.20404 (2024). — [arxiv.org](https://arxiv.org/html/2412.20404v1)
10. **Hugging Face Papers: VideoCrafter1** — [huggingface.co/papers/2310.19512](https://huggingface.co/papers/2310.19512)
11. **Broadway: Boost Your Text-to-Video Generation Model in a Training-free way.** arXiv:2410.06241 (2024). — [arxiv.org](https://arxiv.org/html/2410.06241v1)
12. **Marvik Blog.** "Diffusion models for video generation." — [marvik.ai](https://www.marvik.ai/blog/diffusion-models-for-video-generation)
13. **Video Diffusion Models — A Survey.** OpenReview / TMLR. — [openreview.net](https://openreview.net/pdf?id=sgDFqNTdaN)
14. **kinson chow.** "VideoCrafter1: an open diffusion model." Medium, Nov 2023. — [medium.com](https://medium.com/@tofujoy77/videocrafter1-an-open-diffusion-model-1f24e33977ca)
15. **AI Free Forever.** "31 Open-Source AI Video Models." 2025. — [aifreeforever.com](https://aifreeforever.com/blog/open-source-ai-video-models-free-tools-to-make-videos)

> **참고**: 본 분석에서 VideoCrafter1 논문의 세부 수식은 Latent Diffusion Model(LDM)과 DDPM의 표준 프레임워크를 기반으로 작성하였습니다. 논문 원문의 특정 수식 표기와 약간의 차이가 있을 수 있으므로, 정확한 수식은 원문(arXiv:2310.19512)을 직접 참조하시기 바랍니다.
