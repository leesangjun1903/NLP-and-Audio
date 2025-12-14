# StyO: Stylize Your Face in Only One-shot

## 1. 핵심 주장 및 주요 기여 요약

**StyO**는 단 한 장의 타겟 예술 작품(One-shot)만으로 사용자의 얼굴 사진을 해당 화풍으로 변환하는 **One-shot Face Stylization** 모델입니다.

*   **핵심 주장:** 기존 GAN 기반 방법론은 '스타일(기하학적 변형)'과 '콘텐츠(사용자 신원)'를 분리하는 데 한계가 있어, 과도한 변형 시 원본 얼굴을 잃거나 스타일이 제대로 입혀지지 않는 문제가 있습니다. StyO는 이를 **Latent Diffusion Model(LDM)**의 강력한 생성 능력과 새로운 **Disentanglement(분리) 및 Recombination(재결합)** 전략으로 해결합니다.
*   **주요 기여:**
    1.  **Contrastive Prompt Learning:** 긍정/부정 텍스트 프롬프트를 대조적으로 학습시켜 스타일과 콘텐츠를 명확히 분리하는 **IDL(Identifier Disentanglement Learner)** 제안.
    2.  **Fine-grained Content Control:** Cross-attention 맵을 조작하여 생성 과정에서 원본 얼굴의 세밀한 특징(포즈, 표정 등)을 유지하는 **FCC(Fine-grained Content Controller)** 제안.
    3.  **SOTA 성능 달성:** 기하학적 변형이 큰 예술 작품(캐리커처, 유화 등)에서도 원본 아이덴티티를 유지하며 기존 GAN 모델(JoJoGAN 등)을 능가하는 성능 입증.

***

## 2. 상세 분석: 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제 (Problem Statement)
기존의 One-shot Face Stylization 연구(예: JoJoGAN, BlendGAN)는 주로 **StyleGAN**의 잠재 공간(Latent Space)을 이용했습니다. 그러나 이 방식은 두 가지 치명적인 한계가 있습니다.
1.  **GAN의 Prior 한계:** 실제 사람 얼굴 데이터셋(FFHQ)으로 학습된 StyleGAN은 '실사 얼굴' 분포에 묶여 있어, 만화나 캐리커처처럼 눈이 커지거나 얼굴형이 찌그러지는 **큰 기하학적 변형(Geometry Variation)**을 표현하기 어렵습니다.
2.  **Entanglement 문제:** 잠재 공간 내에서 스타일(화풍)과 콘텐츠(신원)가 얽혀 있어, 스타일을 강하게 적용하면 얼굴이 뭉개지고, 얼굴을 살리면 스타일이 약해지는 트레이드오프가 발생합니다.

### 2.2 제안하는 방법 (Methodology)
StyO는 **LDM(Latent Diffusion Models)**을 기반으로 **"분리 후 재결합(Disentanglement and Recombination)"** 전략을 사용합니다.

#### **Step 1: Identifier Disentanglement Learner (IDL)**
소스 이미지(사용자)와 타겟 이미지(예술 작품)의 스타일과 콘텐츠를 각각 독립적인 **토큰(Identifier)**으로 학습합니다. 이를 위해 대조적 프롬프트(Contrastive Prompt)를 설계합니다.

*   **프롬프트 템플릿:**
    *   소스 이미지 학습용: `“a drawing with [Ssrc][not Stgt] style of [Csrc][not Ctgt] portrait”`
    *   타겟 이미지 학습용: `“a drawing with [Stgt][not Ssrc] style of [Ctgt][not Csrc] portrait”`
    *   여기서 `[S]`는 스타일 토큰, `[C]`는 콘텐츠 토큰, `[not ...]`은 부정(Negative) 토큰입니다.

*   **Triple Reconstruction Loss:**
    모델이 각 토큰에 정확한 정보를 인코딩하도록 세 가지 재구성 손실을 최소화합니다.

$$
    \min_{\theta} \mathbb{E} \left[ \|\epsilon - \epsilon_\theta(z_{src}^t, t, \gamma_{src})\|^2 + \|\epsilon - \epsilon_\theta(z_{tgt}^t, t, \gamma_{tgt})\|^2 + \|\epsilon - \epsilon_\theta(z_{aux}^t, t, \gamma_{aux})\|^2 \right]
    $$
    
*   $z$: Latent vector, $\gamma$: Text embedding
*   보조 데이터셋($x_{aux}$)을 추가하여 스타일 토큰이 특정 이미지에 과적합되지 않도록 돕습니다.

#### **Step 2: Fine-grained Content Controller (FCC)**
학습된 토큰을 재결합하여 스타일라이징을 수행합니다. 이때 원본의 디테일을 살리기 위해 **Cross Attention Control**을 적용합니다.

*   **재결합 프롬프트:** 타겟의 스타일(`[Stgt]`)과 소스의 콘텐츠(`[Csrc]`)를 결합합니다.
    `“a drawing with [Stgt][not Ssrc] style of [Csrc][not Ctgt] portrait”`

*   **Cross Attention Control (수식):**
    생성 중 스타일 이미지의 Attention Map($M_{sty}$)을 사용하는 대신, 콘텐츠와 관련된 부분은 소스 이미지의 Attention Map($M_{src}$)으로 교체합니다.

$$
    \text{Attn}(z_{sty}, \gamma_{sty}; z_{src}, \gamma_{src}) = M_{ctr} V_{sty}
    $$
    
```math
    \text{where } M_{ctr}^i = \begin{cases} M_{src}^i & \text{if } i \in \text{Content Index} \\ M_{sty}^i & \text{otherwise} \end{cases}
```

이 과정을 통해 생성된 이미지가 타겟의 화풍을 따르면서도, 눈/코/입의 위치와 형태는 소스 이미지를 정확히 따르게 됩니다.

### 2.3 모델 구조 (Model Architecture)
*   **Base Model:** Pre-trained **Stable Diffusion (LDM)**.
*   **Encoder/Decoder:** 이미지를 Latent space로 압축/복원하는 VQ-GAN 혹은 KL-Autoencoder.
*   **Prompt Learning:** 기존의 Text Encoder(CLIP)를 미세 조정(Fine-tuning)하여 새로운 토큰 임베딩을 학습.

### 2.4 성능 향상 및 한계
*   **성능 향상:**
    *   **Geometry:** GAN이 실패하던 큰 눈, 과장된 턱선 등의 기하학적 변형을 자연스럽게 반영.
    *   **Content Preservation:** 머리카락 색, 시선 처리 등 미세한 콘텐츠 정보를 유지하는 데 탁월함 (사용자 평가에서 Identity 보존 점수가 타 모델 대비 높음).
*   **한계점:**
    *   **Inference Speed:** Diffusion 모델 특성상 GAN 기반 모델(JoJoGAN 등)에 비해 추론 속도가 느림 (수 초 vs 수십 초).
    *   **Hyperparameter Sensitivity:** 프롬프트 내 토큰 반복 횟수($n_s, n_c$) 등 하이퍼파라미터 조절에 따라 결과물의 스타일 강도가 달라져 튜닝이 필요할 수 있음.

***

## 3. 모델의 일반화 성능 향상 가능성 (Generalization)

StyO의 가장 큰 강점은 **일반화 성능(Generalization Capability)**에 있습니다.

1.  **데이터 분포의 확장:** 기존 StyleGAN 기반 모델은 사람 얼굴 데이터(FFHQ)에 편향된 Prior를 가지고 있어, '사람 얼굴이 아닌 듯한' 예술적 표현(큐비즘, 추상화 등)으로의 일반화에 실패했습니다. 반면, **LDM은 수십억 장의 이미지(LAION 등)로 학습**되어 있어 훨씬 광범위한 분포를 커버하므로, 학습 본 적 없는 극단적인 예술 스타일로도 일반화가 가능합니다.
2.  **Disentanglement의 효과:** 스타일과 콘텐츠를 명확히 분리하는 Contrastive Learning 전략 덕분에, 새로운 타겟 스타일이 들어와도 기존 콘텐츠 토큰과 충돌하지 않고 유연하게 결합됩니다. 이는 **Unseen Style**에 대한 일반화 성능을 크게 높입니다.
3.  **Cross-Attention의 강건함:** 텍스트 기반 제어뿐만 아니라 Spatial Attention Map을 직접 주입하므로, 입력 이미지의 포즈나 구도가 학습 데이터와 달라도(Out-of-distribution) 구조적 일관성을 유지하는 강력한 일반화 성능을 보입니다.

***

## 4. 향후 연구 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향
*   **Generative Paradigm Shift:** Face Stylization 분야의 주류가 GAN Inversion에서 **Diffusion Fine-tuning & Control**로 넘어가는 흐름을 가속화할 것입니다.
*   **Controllable Diffusion:** 텍스트 프롬프트에만 의존하던 기존 방식에서 벗어나, Attention Map 조작과 같은 **내부 레이어 제어(Internal Representation Control)**가 스타일 전이의 핵심 기술로 자리 잡을 것입니다.

### 4.2 연구 시 고려할 점
*   **속도 최적화:** 실시간 애플리케이션 적용을 위해 Consistency Distillation(LCM 등)이나 Flow Matching 기반의 고속 샘플링 기법과 결합하는 연구가 필요합니다.
*   **3D 일관성:** 현재 2D 이미지 기반이므로, 비디오나 3D 아바타 생성 시 시점 변화에 따른 스타일 일관성을 보장하는 연구로 확장되어야 합니다.

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후 One-shot Face Stylization 분야는 GAN에서 Diffusion으로 기술 트렌드가 급격히 변화했습니다.

| 구분 | **StyO (Ours, 2023/2025)** | **JoJoGAN (CVPR 2022)** | **StyleID / Custom Diffusion (2023-2024)** | **InstantID / InstantStyle (2024)** |
| :--- | :--- | :--- | :--- | :--- |
| **기반 모델** | **Latent Diffusion (LDM)** | StyleGAN2 | Diffusion (Fine-tuning) | Diffusion (ControlNet/IP-Adapter) |
| **핵심 기술** | Contrastive Prompt + Attn Control | GAN Inversion + Style Mixing | Textual Inversion / LoRA | Image Prompt Adapter |
| **장점** | **기하학적 변형 우수**, 콘텐츠 보존 탁월 | 매우 빠른 추론 속도, 고해상도 텍스처 | 일반적인 객체/스타일 학습 가능 | **Zero-shot** 가능 (학습 불필요), 빠름 |
| **단점** | 추론 속도가 상대적으로 느림 | 큰 모양 변화(눈 크기 등) 반영 불가 | 얼굴 디테일(ID) 보존력이 떨어짐 | 타겟 이미지의 '추상적 스타일' 추출은 약함 |
| **비고** | 기하학+텍스처 모두 잡은 SOTA | 실사 기반 스타일 변환의 Baseline | 텍스트 기반 편집에 더 초점 | 최신 트렌드는 Zero-shot으로 이동 중 |

*   **비교 요약:**
    *   **vs JoJoGAN:** StyO는 JoJoGAN이 절대 흉내 낼 수 없는 **'구조적 변형(Deformation)'**에서 압도적입니다. JoJoGAN은 색감(Texture)만 바꿀 뿐 얼굴 형태를 못 바꾸지만, StyO는 타겟 그림처럼 눈을 키우거나 얼굴을 둥글게 만들 수 있습니다.
    *   **vs 2024년 최신 연구 (InstantID 등):** 최신 연구들은 별도 학습 없이(Zero-shot) 스타일을 입히는 방향으로 가고 있습니다. 하지만 StyO와 같은 One-shot Fine-tuning 방식은 특정 타겟의 화풍을 **가장 정교하게 모방**해야 하는 고품질 작업(게임 캐릭터 생성, 전문 예술 작업)에서는 여전히 우위를 점합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e42a73e5-41de-465c-9efe-84c412d5b7be/2303.03231v3.pdf)
[2](https://arxiv.org/html/2303.03231v3)
[3](http://arxiv.org/pdf/2403.00459.pdf)
[4](https://arxiv.org/abs/2210.04120)
[5](http://arxiv.org/pdf/2403.15227.pdf)
[6](https://arxiv.org/pdf/2110.09425.pdf)
[7](https://arxiv.org/pdf/2305.03043.pdf)
[8](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/cgf.14890)
[9](http://arxiv.org/pdf/2306.17123.pdf)
[10](http://arxiv.org/pdf/2303.03231.pdf)
[11](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000621)
[12](https://www.paperdigest.org/2025/02/aaai-2025-papers-highlights/)
[13](https://www.catalyzex.com/s/One%20Shot%20Face%20Stylization)
[14](https://blog.paperspace.com/one-shot-face-stylization-with-jojogan/)
[15](https://www.nature.com/articles/s41598-025-17899-x)
[16](https://www.reddit.com/r/MachineLearning/comments/1h8kkjv/d_aaai_2025_phase_2_decision/)
[17](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Deformable_One-shot_Face_Stylization_via_DINO_Semantic_Guidance_CVPR_2024_paper.pdf)
[18](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760124.pdf)
[19](https://openaccess.thecvf.com/content/ICCV2023/papers/Everaert_Diffusion_in_Style_ICCV_2023_paper.pdf)
[20](https://arxiv.org/html/2405.15287v2)
[21](https://arxiv.org/pdf/2509.16925.pdf)
[22](https://arxiv.org/html/2512.01895v1)
[23](https://www.semanticscholar.org/paper/StyO:-Stylize-Your-Face-in-Only-One-Shot-Li-Zhang/984a9d3217e8e53de5b57097b693b4116766dea0)
[24](https://arxiv.org/html/2410.20084v5)
[25](https://arxiv.org/html/2408.12315v1)
[26](https://arxiv.org/html/2502.18417v4)
[27](https://arxiv.org/html/2509.25172v1)
[28](https://arxiv.org/html/2505.11550v1)
[29](https://www.youtube.com/watch?v=lMl7Yd7jC6A)
