
# REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents

> **논문 정보:**
> - **제목:** REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents
> - **저자:** Rui Tian, Qi Dai, Jianmin Bao, Kai Qiu, Yifan Yang, Chong Luo, Zuxuan Wu, Yu-Gang Jiang
> - **소속:** Fudan University (Institute of Trustworthy Embodied AI), Microsoft Research
> - **arXiv:** [2411.13552](https://arxiv.org/abs/2411.13552) (2024년 11월 20일)
> - **학회:** ICCV 2025 (OpenAccess 게재)

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

상업용 비디오 생성 모델들은 현실적이고 고품질의 결과물을 보여주지만, 제한된 접근성과 높은 학습·추론 비용이 여전히 큰 장벽으로 남아 있다.

이 논문은 **비디오가 이미지보다 훨씬 많은 중복(redundant) 정보를 포함하고 있으며**, 따라서 매우 적은 수의 **motion latent**로 인코딩될 수 있다고 주장한다.

특히, 이미지 prior가 비디오 콘텐츠의 풍부한 공간 정보를 전달하기 때문에, **비디오는 motion 변수를 나타내는 아주 적은 수의 latent + content image**로 인코딩될 수 있다는 것이 핵심 인사이트이다.

### 🏆 주요 기여 3가지

| 기여 | 설명 |
|---|---|
| **Reducio-VAE** | image-conditioned VAE로 비디오를 극도로 압축된 motion latent 공간에 인코딩 |
| **Reducio-DiT** | 압축된 latent 기반의 고해상도 비디오 생성 확산 트랜스포머 |
| **2단계 생성 패러다임** | Text→Image → Image+Text→Video의 효율적 파이프라인 |

이 방법은 **비디오 LDM의 학습·추론 효율성을 획기적으로 향상**시켜, Reducio-DiT는 총 **3,200 A100 GPU 시간**만으로 학습되었으며, 단일 A100 GPU에서 **16프레임 1024×1024 비디오 클립을 15.5초 내에 생성**할 수 있다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Sora, Runway Gen-3, Movie Gen과 같은 상업용 모델들은 학습에 수천 GPU와 수백만 GPU 시간이 필요하며, 비디오 1초 생성에 수 분이 소요된다. 이러한 높은 요구사항은 해당 솔루션들을 비용적으로 부담스럽고 대부분의 응용에 비실용적으로 만든다.

**핵심 문제 정의:**

$$\text{Latent Size} \propto \frac{T}{f_t} \times \frac{H}{f_s} \times \frac{W}{f_s} \times |z|$$

여기서 $T$는 프레임 수, $f_t$는 시간 다운샘플링 팩터, $f_s$는 공간 다운샘플링 팩터, $|z|$는 latent 채널 수이다. 기존 2D VAE 방식은 이 크기를 충분히 줄이지 못해 높은 연산 비용이 발생한다.

### 2.2 제안 방법 (Reducio-VAE & Reducio-DiT)

#### 📌 핵심 인사이트: Content–Motion 분리

Image-to-video 생성에서, content image는 풍부한 시각적 디테일과 상당한 중복 정보를 포함하기 때문에, **video latent는 motion 정보를 표현하도록 더욱 압축될 수 있다.**

이 아이디어는 **비디오 코덱(video codec)**에서 참조 프레임을 활용하여 압축된 코드로부터 고충실도 복원을 수행하는 방식과 유사한 정신을 갖는다.

---

#### 🔧 Reducio-VAE 설계

Reducio-VAE는 입력 비디오를 **4096배 다운샘플된 컴팩트 공간**으로 공격적으로 압축하는 3D 인코더와, 중간 프레임의 **feature pyramid를 content condition**으로 융합하는 3D 디코더로 구성된다.

구체적으로, 중간 프레임 $V_{T/2}$를 content guidance로 선택하고:

$$z_{\mathcal{V}} = \text{Enc}_{3D}(V_{1:T}) \quad \in \mathbb{R}^{|z| \times \frac{T}{f_t} \times \frac{H}{f_s} \times \frac{W}{f_s}}$$

디코딩 시 content frame의 feature pyramid $\mathcal{F}(V_{T/2})$를 활용:

$$\hat{V}_{1:T} = \text{Dec}_{3D}\left(z_{\mathcal{V}},\ \mathcal{F}(V_{T/2})\right)$$

Reducio-VAE는 16프레임 비디오 클립을 $\frac{T}{4} \times \frac{H}{32} \times \frac{W}{32}$의 latent 공간으로 인코딩하며, 이는 content image prior를 기반으로 비디오에 대한 **4096배 압축률**을 가능하게 한다.

이를 기존 2D VAE와 비교하면:

$$\text{Compression Ratio} = \frac{\text{latent size}_{2D\text{-VAE}}}{\text{latent size}_{\text{Reducio-VAE}}} = 64\times \ (\text{spatial}) \times \frac{f_t^{\text{Reducio}}}{f_t^{2D}} \ (\text{temporal})$$

전체 공간-시간적 압축은 **4096배** (= $64 \times$ 공간 $\times$ 시간 압축)로 공통 2D VAE 대비 **64배 축소**.

---

#### 🔧 Reducio-DiT 설계

Reducio-DiT 학습 프레임워크는 vanilla DiT와 달리, **cross-attention을 통해 키프레임의 의미론적 정보와 콘텐츠를 주입**하는 추가적인 image-condition 모듈을 채택한다.

이후 Diffusion Transformer(DiT) 기반의 LDM을 구축하며, 텍스트 조건으로 **T5 features**를 사용하고, **이미지 semantic encoder**와 **context encoder**를 추가로 활용하여 공간적 콘텐츠 정보를 모델에 전달한다.

Reducio-DiT의 확산 과정은 아래와 같이 정의된다:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{z_0, \epsilon, t}\left[\left\|\epsilon_\theta\left(z_t, t, c_{\text{text}}, c_{\text{img}}\right) - \epsilon\right\|^2\right]$$

여기서:
- $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ : noised latent
- $c_{\text{text}}$ : T5 기반 텍스트 조건
- $c_{\text{img}}$ : image semantic encoder + context encoder 기반 이미지 조건
- $\epsilon_\theta$ : Reducio-DiT 파라미터

전체 생성 파이프라인은 **2단계 패러다임**을 채택하며, 먼저 text-to-image 생성으로 조건 이미지를 생성하고, 이후 text-image-to-video 생성을 통해 Reducio-DiT로 비디오를 생성한다.

---

### 2.3 모델 구조 전체 다이어그램 (개념도)

```
[텍스트 프롬프트]
       │
       ▼
 ┌─────────────────┐
 │  T2I Generator  │   (Stage 1: Text → Keyframe Image)
 └────────┬────────┘
          │ Content Image V_{T/2}
          ▼
 ┌─────────────────────────────────┐
 │         Reducio-DiT             │   (Stage 2: Text+Image → Video)
 │  ┌──────────────────────────┐   │
 │  │  Noise + Latent Tokens   │   │
 │  │  z_V ∈ R^(C×T/4×H/32×W/32) │
 │  └──────────┬───────────────┘   │
 │             │ Cross-Attn         │
 │  ┌──────────┴───────────────┐   │
 │  │ Image Semantic Encoder   │   │
 │  │ + Context Encoder        │   │
 │  └──────────────────────────┘   │
 │  Text: T5 Features (cross-attn) │
 └────────────────┬────────────────┘
                  │ Denoised z_V
                  ▼
 ┌─────────────────────────────────┐
 │        Reducio-VAE Decoder      │
 │  Dec_3D(z_V, F(V_{T/2}))        │
 └─────────────────────────────────┘
                  │
                  ▼
         [1024×1024 Video]
```

---

### 2.4 성능 향상

극도로 압축된 비디오 latent 덕분에 Reducio-DiT는 빠른 학습·추론 속도와 높은 생성 품질을 동시에 달성하며, UCF-101에서 **FVD 318.5**를 기록하여 다수의 이전 연구를 능가한다.

정량적으로, Reducio-VAE는 PSNR 기준으로 최신 2D VAE(SD2.1-VAE, SDXL-VAE)보다 **5dB 이상** 우수하며, OmniTokenizer, OpenSora-1.2 등 비디오 전용 VAE보다도 더 나은 성능을 보인다.

또한 동시 연구인 **Cosmos-VAE** 대비 SSIM 0.2, PSNR 5dB 우위를 달성한다.

전체적으로 Reducio-DiT는 LaVie 등 기존 방법 대비 **16.6배의 속도 향상**을 보이면서도 UCF-101에서 FVD 318.5를 달성한다.

**성능 요약표:**

| 지표 | Reducio-VAE (제안) | SD2.1-VAE / SDXL-VAE | OmniTokenizer | Cosmos-VAE |
|---|---|---|---|---|
| PSNR | **최고** (+5dB↑) | 기준 | 열위 | −5dB |
| SSIM | **최고** | 열위 | 열위 | −0.2 |
| FVD (UCF-101) | **318.5** | - | - | - |
| 추론 시간 (1024²) | **15.5초/클립** | - | - | - |
| 학습 비용 | **3,200 A100 GPU-hrs** | 수십만 GPU-hrs | - | - |

---

### 2.5 한계점

논문 자체에서 언급하는 미래 방향으로, 이 방법은 **efficient attention, rectified flow, diffusion distillation** 등 다른 가속 기법과도 호환 가능하나, 이에 대한 탐구는 향후 과제로 남겨두었다.

논문에서 암묵적으로 드러나는 한계:
1. **2단계 파이프라인 의존성:** Stage 1의 이미지 품질이 최종 비디오 품질에 큰 영향을 미침
2. **Content image 필요:** 완전한 text-only 비디오 생성보다는 image-conditioned 구조에 특화됨
3. **단기 클립 제한:** 16프레임 수준의 짧은 클립에 집중되어 있어 장기 비디오로의 확장 미검증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 구조적 강점

논문은 **비디오 latent 공간에서 공간적 중복성(spatial redundancy)의 중요성을 인식하는 것이 핵심**임을 제안하며, 이는 다양한 도메인에서 동일하게 적용될 수 있는 범용적인 인사이트이다.

Reducio-DiT는 고해상도($1024^2$)로 쉽게 확장 가능하며, 감당 가능한 비용으로 **다양한 해상도로의 업스케일**이 가능함을 실험으로 보인다.

### 3.2 일반화 성능 향상 메커니즘

**(a) 비디오 코덱에서 영감받은 Content-Motion 분리:**

비디오 코덱의 참조 프레임 방식에서 영감을 얻은 content–motion 분리는, **모션 패턴이 콘텐츠 도메인과 독립적**이라는 전제를 바탕으로 하므로, 학습 도메인 외의 영상 스타일에도 이론적으로 일반화 가능하다.

**(b) 공개된 사전학습 모델 활용:**

Reducio-DiT는 T5 언어 모델과 이미지 semantic encoder를 활용하여 텍스트–이미지–비디오 간의 풍부한 cross-modal 조건을 제공하며, 이러한 대규모 사전학습 모델의 활용은 도메인 일반화에 긍정적으로 기여한다.

**(c) 한정된 GPU 자원 하의 강건성:**

광범위한 실험 결과, Reducio-DiT는 **제한된 GPU 자원으로 학습되었음에도 불구하고** 평가에서 강한 성능을 달성함을 보여, 컴퓨팅 효율성과 품질 간의 균형 측면에서 일반화 가능성을 시사한다.

**(d) 가속 기법과의 호환성:**

효율적인 어텐션, rectified flow, diffusion distillation 등 다양한 가속 기법과 호환 가능하다는 점은, 이 프레임워크가 미래의 다양한 비디오 생성 파이프라인에 **모듈식으로 통합**되어 일반화 성능을 높이는 방향으로 발전할 가능성을 보여준다.

### 3.3 일반화의 한계와 개선 방향

일반적으로 비디오 생성 모델들은 "사례 기반(case-based)" 일반화 거동을 보이며, 스케일링만으로는 근본적인 물리 법칙 습득이 불충분하다는 연구 결과가 있으므로, Reducio도 이 한계에서 자유롭지 않을 수 있다. 개선 방향으로는:

- **더 다양하고 대규모의 학습 데이터** 확보
- **물리 기반 손실(physics-informed loss)** 도입
- **도메인 적응(domain adaptation)** 기법 결합

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비디오 VAE / 토크나이저 비교

| 모델 | 공간 압축 | 시간 압축 | 특징 |
|---|---|---|---|
| **SD-VAE (2022)** | $8\times$ | 없음 | 이미지 전용 2D VAE |
| **OpenSora-1.2 (2024)** | $8\times$ | $4\times$ | 공개 소스 3D VAE |
| **OmniTokenizer (2024)** | $8\times$ | $4\times$ | 통합 이미지/비디오 토크나이저 |
| **Cosmos-VAE (2024)** | $8\times$ | $4\times$ | NVIDIA 압축 VAE |
| **LTX-Video-VAE (2024)** | $32\times$ | $8\times$ | 1:192 압축비 |
| **Reducio-VAE (2024)** | **$32\times$** | **$4\times$ (이상)** | **Content image 조건, 4096배 압축** |

LTX-Video는 Video-VAE가 **1:192의 높은 압축비**와 $32 \times 32 \times 8$ 픽셀/토큰의 시공간 다운스케일링을 달성하여 전체 시공간 셀프어텐션을 효율적으로 수행할 수 있게 한다. 이는 Reducio-VAE와 방향성은 같지만, Reducio는 **content image를 prior로 활용**한다는 점에서 차별화된다.

### 4.2 비디오 생성 모델 전반 비교

Sora, Runway Gen-3, Movie Gen 등 상업용 모델들은 학습에 수천 GPU와 수백만 GPU 시간이 필요하지만, Reducio-DiT는 이를 극적으로 단축한다.

| 모델 | 학습 비용 | 추론 속도 | 최대 해상도 | FVD (UCF-101) |
|---|---|---|---|---|
| **LaVie** | 높음 | 느림 | 512p | >5000 |
| **OpenSora** | 중간 | 중간 | 720p | ~500 |
| **CogVideoX** | 높음 | 느림 | 1080p | ~300 |
| **Reducio-DiT** | **3,200 GPU-hrs** | **15.5초/클립** | **1024p** | **318.5** |

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

**① 효율적 비디오 생성의 새로운 패러다임:**
비디오 생성은 아직도 감당하기 어려운 연산 비용에 묶여 있는데, 이 논문은 content image prior를 활용하여 **latent 코드가 motion 변수만을 표현하도록 극단적 압축**을 달성하는 방향으로 오버헤드를 줄이는 방법을 탐구한다. 이는 향후 비디오 생성 연구의 설계 원칙으로 자리잡을 가능성이 크다.

**② 비디오 코덱 아이디어의 딥러닝 접목:**
비디오 코덱의 참조 프레임 방식을 딥러닝 잠재 공간에 접목한 접근은 비디오 압축, 스트리밍, 생성 등 여러 분야에 걸쳐 새로운 응용 가능성을 열어준다.

**③ 민주화(Democratization) 가능성:**
Reducio-DiT는 연산 도전 과제로 부담이 컸던 산업에 비용 효율적 솔루션을 제공하여, **고해상도 비디오 생성을 더 많은 사람들이 접근할 수 있도록**한다.

### 5.2 향후 연구 시 고려할 점

**① 가속 기법과의 통합 탐구:**
논문이 명시적으로 남겨둔 미래 방향으로, efficient attention, rectified flow, diffusion distillation 등과의 결합으로 **추가 속도 향상**이 가능하다.

**② 장기 비디오 생성으로의 확장:**
현재 16프레임 클립에 집중되어 있으므로, temporal autoregressive 방식이나 sliding window 기법과 결합하여 장기 비디오 생성으로 확장하는 연구가 필요하다.

**③ 물리적 일관성 및 일반화:**
비디오 생성 모델에서 스케일링만으로는 근본적인 물리 법칙 습득에 불충분하다는 연구 결과를 고려할 때, Reducio 프레임워크에서도 물리 기반 제약 또는 시뮬레이션 데이터 활용이 일반화 성능 향상에 기여할 수 있다.

**④ 멀티모달 확장:**
텍스트·이미지 외 오디오, 깊이 맵, 광학 흐름 등 다양한 조건 신호를 활용한 제어 가능한 비디오 생성 연구가 자연스러운 확장 방향이다.

**⑤ Content Image 품질 의존성 극복:**
2단계 파이프라인의 첫 번째 단계(T2I 생성) 품질이 전체 결과를 좌우하므로, **end-to-end 학습** 또는 **강건한 image prior 설계**가 중요한 연구 주제가 될 것이다.

---

## 📚 참고 자료 (출처)

1. **arXiv 논문 원문:** Rui Tian et al., "REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents," arXiv:2411.13552, 2024. [https://arxiv.org/abs/2411.13552](https://arxiv.org/abs/2411.13552)
2. **arXiv HTML (v1):** [https://arxiv.org/html/2411.13552v1](https://arxiv.org/html/2411.13552v1)
3. **arXiv HTML (v3):** [https://arxiv.org/html/2411.13552v3](https://arxiv.org/html/2411.13552v3)
4. **ICCV 2025 OpenAccess:** [https://openaccess.thecvf.com/content/ICCV2025/papers/Tian_REDUCIO...](https://openaccess.thecvf.com/content/ICCV2025/papers/Tian_REDUCIO_Generating_1K_Video_within_16_Seconds_using_Extremely_Compressed_ICCV_2025_paper.pdf)
5. **Microsoft Research 공식 페이지:** [https://www.microsoft.com/en-us/research/publication/reducio-generating-1024-1024-video-within-16-seconds-using-extremely-compressed-motion-latents/](https://www.microsoft.com/en-us/research/publication/reducio-generating-1024%E2%A8%891024-video-within-16-seconds-using-extremely-compressed-motion-latents/)
6. **Papers With Code:** [https://paperswithcode.com/paper/reducio-generating-1024-times-1024-video](https://paperswithcode.com/paper/reducio-generating-1024-times-1024-video)
7. **MarkTechPost 분석:** [https://www.marktechpost.com/2024/11/21/microsoft-research-introduces-reducio-dit-enhancing-video-generation-efficiency-with-advanced-compression/](https://www.marktechpost.com/2024/11/21/microsoft-research-introduces-reducio-dit-enhancing-video-generation-efficiency-with-advanced-compression/)
8. **OpenReview:** [https://openreview.net/forum?id=USgMeRFiWt](https://openreview.net/forum?id=USgMeRFiWt)
9. **ADS Abstract:** [https://ui.adsabs.harvard.edu/abs/2024arXiv241113552T/abstract](https://ui.adsabs.harvard.edu/abs/2024arXiv241113552T/abstract)
10. **관련 연구 - "How Far Is Video Generation from World Model: A Physical Law Perspective," arXiv:2411.02385** (일반화 분석 참조)
11. **관련 연구 - LTX-Video: Realtime Video Latent Diffusion**, HuggingFace Papers, arXiv:2501.00103
