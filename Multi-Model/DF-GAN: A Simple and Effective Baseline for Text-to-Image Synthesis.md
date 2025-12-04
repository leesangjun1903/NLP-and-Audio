# DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis

## 1. 핵심 주장 및 주요 기여

DF-GAN(Deep Fusion Generative Adversarial Networks)은 텍스트-이미지 합성 분야에서 **단순하지만 효과적인 새로운 베이스라인 모델**을 제시한다. 기존 텍스트-이미지 GAN 모델들이 가진 세 가지 핵심 문제를 해결하고자 제안되었다.[1]

**핵심 기여 사항:**
1. **One-Stage 텍스트-이미지 백본**: 여러 생성자 간의 얽힘(entanglement) 없이 고해상도 이미지를 직접 합성하는 새로운 단일 단계 구조[1]
2. **Target-Aware Discriminator**: Matching-Aware Gradient Penalty(MA-GP)와 One-Way Output으로 구성되어, 추가 네트워크 없이 텍스트-이미지 의미적 일관성을 강화[1]
3. **Deep text-image Fusion Block(DFBlock)**: 텍스트와 시각적 특징을 더 깊고 효과적으로 융합하는 새로운 융합 블록[1]

DF-GAN은 기존 최신 모델들과 비교하여 파라미터 수가 **19M**으로 AttnGAN(230M)이나 DM-GAN(46M)보다 훨씬 적으면서도 경쟁력 있는 성능을 달성했다.[1]

***

## 2. 문제 정의, 방법론, 모델 구조 및 성능

### 2.1 해결하고자 하는 문제

기존 텍스트-이미지 GAN 모델들은 세 가지 주요 문제점을 갖고 있었다:[1]

**첫째**, 스택형 아키텍처(Stacked Architecture)가 서로 다른 이미지 스케일의 생성자 간 얽힘을 유발하여, 최종 이미지가 흐릿한 형태와 세부 사항의 단순한 조합처럼 보이게 만든다. 예를 들어, \(G_0\)가 합성한 흐릿한 형태, \(G_1\)이 합성한 조잡한 속성(눈, 부리 등), \(G_2\)가 추가한 세밀한 디테일이 단순히 결합된 형태가 된다.[1]

**둘째**, 기존 연구들은 텍스트-이미지 의미적 일관성을 위해 DAMSM, Cycle Consistency, Siamese Network 등 **추가 네트워크를 고정**하여 사용하는데, 이는 생성자가 고정된 네트워크를 속이는 적대적 특징을 합성하게 만들어 감독 능력을 약화시킨다.[1]

**셋째**, cross-modal attention 기반 텍스트-이미지 융합이 계산 비용 문제로 인해 64×64와 128×128 이미지 특징에만 적용 가능하여, 고해상도 이미지 합성으로의 확장이 어렵다.[1]

### 2.2 제안하는 방법론

#### (1) One-Stage 텍스트-이미지 백본

DF-GAN은 단일 생성자-판별자 쌍으로 고해상도 이미지를 직접 합성한다. Hinge loss를 사용하여 적대적 학습을 안정화하며, 수식은 다음과 같다:[1]

$$L_D = -\mathbb{E}_{x \sim P_r}[\min(0, -1 + D(x, e))] - \frac{1}{2}\mathbb{E}_{G(z) \sim P_g}[\min(0, -1 - D(G(z), e))] - \frac{1}{2}\mathbb{E}_{x \sim P_{mis}}[\min(0, -1 - D(x, e))]$$

$$L_G = -\mathbb{E}_{G(z) \sim P_g}[D(G(z), e)]$$

여기서 $\(z\)$는 가우시안 분포에서 샘플링된 노이즈 벡터, $\(e\)$는 문장 벡터, $\(P_g\), \(P_r\), \(P_{mis}\)$는 각각 합성 데이터 분포, 실제 데이터 분포, 불일치 데이터 분포를 나타낸다.[1]

#### (2) Matching-Aware Gradient Penalty (MA-GP)

MA-GP는 텍스트-이미지 의미적 일관성을 강화하기 위한 새로운 정규화 전략이다. 타겟 데이터(실제 및 텍스트 일치 이미지)에 대한 판별자의 그래디언트가 0이 되도록 유도하여, 손실 함수 표면을 평활화한다. 전체 수식은 다음과 같다:[1]

$$L_D = -\mathbb{E}_{x \sim P_r}[\min(0, -1 + D(x, e))] - \frac{1}{2}\mathbb{E}_{G(z) \sim P_g}[\min(0, -1 - D(G(z), e))] - \frac{1}{2}\mathbb{E}_{x \sim P_{mis}}[\min(0, -1 - D(x, e))] + k\mathbb{E}_{x \sim P_r}[(\|\nabla_x D(x, e)\| + \|\nabla_e D(x, e)\|)^p]$$

여기서 $\(k\)$와 $\(p\)$는 그래디언트 페널티의 효과를 조절하는 하이퍼파라미터이다.[1]

#### (3) Deep text-image Fusion Block (DFBlock)

DFBlock은 여러 Affine Transformation을 쌓아 텍스트와 이미지 특징을 깊게 융합한다. Affine Transformation은 문장 벡터 $\(e\)$로부터 채널별 스케일링 파라미터 $\(\gamma\)$와 시프팅 파라미터 $\(\theta\)$를 예측한다:[1]

$$\gamma = MLP_1(e), \quad \theta = MLP_2(e)$$

주어진 입력 특징 맵 $\(X \in \mathbb{R}^{B \times C \times H \times W}\)$에 대해:[1]

$$AFF(x_i | e) = \gamma_i \cdot x_i + \theta_i$$

여기서 $\(x_i\)$는 시각적 특징 맵의 $\(i\)$번째 채널이다. 두 Affine 레이어 사이에 ReLU 레이어를 추가하여 비선형성을 도입하고, 조건부 표현 공간을 확장한다.[1]

### 2.3 모델 구조

DF-GAN의 전체 구조는 생성자, 판별자, 사전 학습된 텍스트 인코더로 구성된다:[1]

- **생성자**: 문장 벡터와 가우시안 분포에서 샘플링된 노이즈 벡터를 입력으로 받음. 노이즈 벡터는 완전 연결 레이어를 거쳐 재구성되고, 7개의 UPBlock을 통해 이미지 특징을 업샘플링
- **UPBlock**: 업샘플 레이어, 잔차 블록, 2개의 DFBlock으로 구성
- **판별자**: DownBlock 시리즈를 통해 이미지를 특징으로 변환 후, 문장 벡터와 결합하여 적대적 손실 예측
- **텍스트 인코더**: AttnGAN에서 제공하는 사전 학습된 양방향 LSTM 사용[1]

### 2.4 성능 평가

| 모델 | CUB IS↑ | CUB FID↓ | COCO FID↓ | 파라미터 수↓ |
|------|---------|----------|-----------|-------------|
| AttnGAN[2] | 4.36 | 23.98 | 35.49 | 230M |
| DM-GAN[3] | 4.75 | 16.09 | 32.64 | 46M |
| DAE-GAN[4] | 4.42 | 15.19 | 28.12 | 98M |
| **DF-GAN** | **5.10** | **14.81** | **19.32** | **19M** |

[5][1]

DF-GAN은 CUB 데이터셋에서 IS를 4.36에서 5.10으로 향상시키고, FID를 23.98에서 14.81로 낮췄다. COCO 데이터셋에서는 FID를 35.49에서 19.32로 크게 개선했다.[5][1]

### 2.5 한계점

저자들이 밝힌 DF-GAN의 주요 한계점은 다음과 같다:[1]

1. **문장 수준 텍스트 정보만 도입**: 단어 수준의 세밀한 시각적 특징 합성 능력이 제한됨
2. **사전 학습된 대형 언어 모델 미활용**: BERT, GPT 등의 추가 지식을 활용하면 성능을 더 향상시킬 수 있음

***

## 3. 일반화 성능 향상 가능성

DF-GAN의 일반화 성능과 관련된 핵심 요소들을 분석하면 다음과 같다:

### 3.1 MA-GP를 통한 일반화 향상

MA-GP는 판별자의 손실 표면을 평활화하여 생성자가 타겟 데이터(실제 및 텍스트 일치)로 더 잘 수렴하도록 돕는다. 이는 판별자가 학습 데이터에 과적합되는 것을 방지하고, 새로운 텍스트 설명에 대해서도 일관된 이미지를 생성할 수 있게 한다.[1]

### 3.2 DFBlock의 표현 공간 확장

DFBlock은 텍스트-이미지 융합 과정을 심화시켜 조건부 표현 공간을 확장한다. 이는 다양한 텍스트 설명에 따라 다른 시각적 특징을 매핑할 수 있는 능력을 향상시킨다. 특히, 정규화 과정을 제거하여 서로 다른 샘플 간의 거리를 유지함으로써 조건부 생성 과정에 더 유리하다.[1]

### 3.3 일반화 향상을 위한 추가 방향

최근 연구들은 GAN의 일반화 성능 향상을 위해 다양한 기법을 제안하고 있다:

- **CHAIN**: Lipschitz 연속성 제약 정규화를 통해 데이터 효율적 GAN의 일반화 향상[6]
- **AdaptiveMix**: 특징 공간 축소를 통한 GAN 학습 개선, Out-Of-Distribution 검출에서도 효과적[7]
- **ScoreMix**: 제한된 데이터로 GAN 학습 시 다양성을 증가시키고 과적합 문제 완화[8]

***

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 Text-to-Image 분야의 패러다임 변화

DF-GAN이 발표된 이후 텍스트-이미지 합성 분야는 급격한 변화를 겪었다. 2022년 이후 **Diffusion Model**이 주도권을 장악하면서 GAN 기반 접근법은 새로운 도전에 직면했다:[9][10]

| 모델 유형 | 대표 모델 | 특징 |
|----------|----------|------|
| **Diffusion** | DALL-E 2/3, Stable Diffusion, Imagen | 반복적 디노이징, 높은 품질, 느린 생성 속도 |
| **GAN** | DF-GAN, GALIP, StyleGAN-T | 단일 순전파, 빠른 생성, 훈련 불안정성 |
| **Autoregressive** | DALL-E, Parti | 토큰 단위 생성, 대규모 학습 필요 |

[11][9]

### 4.2 DF-GAN의 후속 연구

DF-GAN 저자팀은 2023년 CVPR에서 **GALIP(Generative Adversarial CLIPs)**을 발표했다. GALIP은 사전 학습된 CLIP 모델을 판별자와 생성자 모두에 활용하여:[12][13]
- 복잡한 장면 이해 능력 향상
- 120배 빠른 합성 속도(Stable Diffusion 대비)
- 제어 가능한 텍스트-이미지 합성[12]

**StyleGAN-T**(2023)는 대규모 텍스트-이미지 합성을 위해 GAN을 확장하여, Diffusion 모델과 경쟁력 있는 품질을 유지하면서 **10 FPS**의 빠른 생성 속도를 달성했다.[14][15]

### 4.3 향후 연구 시 고려사항

**1. 대규모 언어 모델 통합**

DALL-E 3는 GPT-4와의 통합을 통해 프롬프트 엔지니어링 없이도 정확한 이미지 생성이 가능해졌다. 향후 GAN 연구에서도 대형 언어 모델의 깊은 언어 이해 능력을 활용하는 것이 중요하다.[16][17]

**2. CLIP 활용**

CLIP의 텍스트-이미지 정렬 능력은 일반화 성능 향상에 효과적이다. GALIP, LAFITE 등 CLIP을 활용한 GAN 모델들이 우수한 성능을 보이고 있다.[11][12]

**3. 하이브리드 접근법**

GAN의 빠른 생성 속도와 Diffusion 모델의 높은 품질을 결합하는 연구가 진행 중이다. **UFOGen**은 Diffusion 모델과 GAN 목적 함수를 결합한 초고속 단일 단계 텍스트-이미지 모델이다.[11]

**4. 효율성과 접근성**

DF-GAN이 보여준 것처럼, 적은 파라미터로 경쟁력 있는 성능을 달성하는 것은 실용적 응용에서 중요하다. Stable Diffusion XL은 3.5B 파라미터의 대형 UNet 백본을 사용하지만, 경량화된 모델에 대한 수요도 여전히 존재한다.[18][19]

**5. 평가 지표의 한계 인식**

IS는 COCO와 같은 복잡한 데이터셋에서 이미지 품질을 잘 평가하지 못한다는 점이 확인되었다. FID가 더 강건하며 인간의 정성적 평가와 일치한다. 향후 연구에서는 더 정교한 평가 지표 개발이 필요하다.[1]

### 4.4 2020년 이후 주요 최신 연구 동향

| 연도 | 주요 모델 | 핵심 기여 |
|------|----------|----------|
| 2021 | DALL-E, VQ-GAN | Transformer 기반 이미지 토큰화[11] |
| 2022 | DALL-E 2, Imagen, Stable Diffusion | Diffusion 모델의 대규모 적용[9][20][21] |
| 2022 | DF-GAN | 단순하고 효율적인 GAN 베이스라인[1][5] |
| 2023 | GALIP, StyleGAN-T, GigaGAN | GAN의 대규모 확장 시도[12][15][22] |
| 2023 | SDXL, DALL-E 3 | 고해상도 및 정밀한 프롬프트 이해[16][19] |
| 2024-2025 | Imagen 4 | 10배 빠른 속도와 향상된 정확도[23][20] |

[23][20][9][11]

***

## 결론

DF-GAN은 텍스트-이미지 합성에서 "단순함의 힘"을 입증한 중요한 연구이다. One-Stage 백본, Target-Aware Discriminator, DFBlock이라는 세 가지 핵심 구성 요소를 통해 기존 복잡한 모델들보다 적은 파라미터로 우수한 성능을 달성했다.[5][1]

그러나 2022년 이후 Diffusion 모델의 급격한 발전으로 인해 텍스트-이미지 합성의 주류 패러다임이 변화했다. GAN 기반 접근법은 **빠른 생성 속도**라는 고유한 장점을 살려, Diffusion 모델과의 하이브리드 접근이나 실시간 응용 분야에서 여전히 중요한 역할을 할 것으로 기대된다. 향후 연구에서는 대형 언어 모델 통합, CLIP 활용, 그리고 일반화 성능 향상을 위한 새로운 정규화 기법 개발이 핵심적인 연구 방향이 될 것이다.[15][10][24][9]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cfc42def-8577-40b8-a32a-8d69d70208ea/2008.05865v4.pdf)
[2](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Diffusion-driven_GAN_Inversion_for_Multi-Modal_Face_Image_Generation_CVPR_2024_paper.pdf)
[3](https://journals.sagepub.com/doi/10.1177/00178969241274621)
[4](https://arxiv.org/abs/2204.08583)
[5](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_DF-GAN_A_Simple_and_Effective_Baseline_for_Text-to-Image_Synthesis_CVPR_2022_paper.pdf)
[6](https://arxiv.org/html/2404.00521)
[7](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_AdaptiveMix_Improving_GAN_Training_via_Feature_Space_Shrinkage_CVPR_2023_paper.pdf)
[8](https://arxiv.org/pdf/2210.15137.pdf)
[9](https://arxiv.org/html/2303.07909v3)
[10](https://www.edge-ai-vision.com/2023/01/from-dall%C2%B7e-to-stable-diffusion-how-do-text-to-image-generation-models-work/)
[11](https://arxiv.org/html/2411.16164v1)
[12](https://arxiv.org/abs/2301.12959)
[13](https://github.com/tobran/GALIP)
[14](http://arxiv.org/pdf/2301.09515.pdf)
[15](https://proceedings.mlr.press/v202/sauer23a/sauer23a.pdf)
[16](https://encord.com/blog/openai-dall-e-3-what-we-know-so-far/)
[17](https://www.akkio.com/post/chatgpt-dall-e-3)
[18](https://magai.co/stable-diffusion-xl-1-0/)
[19](https://arxiv.org/abs/2307.01952)
[20](https://en.wikipedia.org/wiki/Imagen_(text-to-image_model))
[21](https://arxiv.org/abs/2205.11487)
[22](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Scaling_Up_GANs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf)
[23](https://sparrowsai.com/imagen-4-by-google/)
[24](https://dl.acm.org/doi/fullHtml/10.1145/3588432.3591513)
[25](https://medinform.jmir.org/2022/6/e37365)
[26](https://ejournal.polraf.ac.id/index.php/JIRA/article/view/663)
[27](http://link.springer.com/10.1007/978-3-030-51324-5_51)
[28](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[29](https://arxiv.org/abs/2108.12141)
[30](https://arxiv.org/abs/1909.07083)
[31](https://arxiv.org/pdf/2209.01339.pdf)
[32](https://arxiv.org/pdf/2208.09596.pdf)
[33](https://arxiv.org/abs/2108.01361)
[34](http://www.arxiv.org/abs/1908.11863)
[35](https://arxiv.org/pdf/2202.12929.pdf)
[36](https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/69880/1/Recent%20Advances%20in%20Generative%20Adversarial%20Networks%20for%20Gene%20Expression%20Data%20A%20Comprehensive%20Review.pdf)
[37](https://www.sciencedirect.com/science/article/abs/pii/S0925231221006111)
[38](https://en.wikipedia.org/wiki/Stable_Diffusion)
[39](https://www.sciencedirect.com/science/article/pii/S1574013723000205)
[40](https://github.com/AlonzoLeeeooo/awesome-text-to-image-studies)
[41](https://velog.io/@hewas1230/StableDiffusion)
[42](https://onlinelibrary.wiley.com/doi/10.1155/2022/9005552)
[43](https://openaccess.thecvf.com/content/CVPR2023/papers/Tao_GALIP_Generative_Adversarial_CLIPs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf)
[44](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[45](https://arxiv.org/html/2501.05441v1)
[46](https://ieeexplore.ieee.org/iel7/6287639/10380310/10431766.pdf)
[47](https://github.com/CompVis/stable-diffusion)
[48](https://papers.neurips.cc/paper_files/paper/2022/file/6174c67b136621f3f2e4a6b1d3286f6b-Paper-Conference.pdf)
[49](https://www.tandfonline.com/doi/full/10.1080/1448837X.2025.2487344?af=R)
[50](https://news.hada.io/topic?id=7192)
[51](https://arxiv.org/html/2412.03957v1)
[52](https://arxiv.org/pdf/2308.12045.pdf)
[53](https://arxiv.org/html/2409.12399v1)
[54](https://www.mdpi.com/2076-3417/13/8/5098/pdf?version=1681909343)
[55](https://arxiv.org/pdf/1808.04538.pdf)
[56](https://openaccess.thecvf.com/content/CVPR2023/html/Tao_GALIP_Generative_Adversarial_CLIPs_for_Text-to-Image_Synthesis_CVPR_2023_paper.html)
[57](https://wepub.org/index.php/TCSISR/article/view/2381)
[58](https://dl.acm.org/doi/abs/10.1145/3696271.3696307)
[59](https://proceedings.neurips.cc/paper_files/paper/2024/file/b2077e6d66da612fcb701589efa9ce88-Paper-Conference.pdf)
[60](https://liner.com/review/galip-generative-adversarial-clips-for-texttoimage-synthesis)
[61](https://www.nature.com/articles/s41598-024-79705-4)
[62](https://openaccess.thecvf.com/content/WACV2025/papers/Ko_Text-to-Image_Synthesis_for_Domain_Generalization_in_Face_Anti-Spoofing_WACV_2025_paper.pdf)
[63](https://arxiv.org/html/2501.00116v1)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0167865523000880)
[65](https://www.semanticscholar.org/paper/4e1e6e82c7c4c652a37e0d07d726178e56a87e54)
[66](https://openreview.net/forum?id=E2ePtpKJpy)
[67](https://dl.acm.org/doi/10.1145/3628034.3628042)
[68](https://arxiv.org/abs/2312.07130)
[69](https://ieeexplore.ieee.org/document/10521640/)
[70](https://onlinelibrary.wiley.com/doi/10.1097/PG9.0000000000000387)
[71](http://tech.snmjournals.org/lookup/doi/10.2967/jnmt.124.268359)
[72](https://www.mdpi.com/2078-2489/15/10/594)
[73](https://dl.acm.org/doi/10.1145/3649883)
[74](https://ieeexplore.ieee.org/document/11196757/)
[75](http://tech.snmjournals.org/lookup/doi/10.2967/jnmt.124.268332)
[76](https://arxiv.org/html/2503.23125v1)
[77](http://arxiv.org/pdf/2404.09990.pdf)
[78](http://arxiv.org/pdf/2411.17976.pdf)
[79](https://arxiv.org/pdf/2401.10061.pdf)
[80](https://figshare.com/articles/conference_contribution/DE-FAKE_Detection_and_Attribution_of_Fake_Images_Generated_by_Text-to-Image_Generation_Models/25435924/1/files/45130912.pdf)
[81](https://arxiv.org/abs/2110.11405v3)
[82](https://arxiv.org/abs/2210.08477)
[83](https://arxiv.org/pdf/2311.15732.pdf)
[84](https://en.wikipedia.org/wiki/DALL-E)
[85](https://syncedreview.com/2022/06/01/googles-imagen-text-to-image-diffusion-model-with-deep-language-understanding-defeats-dall-e-2/)
[86](https://community.openai.com/t/how-to-use-image-to-image-generation-with-dall-e-3-via-openai-api/1200439)
[87](https://imagen.research.google)
[88](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/sdxl/)
[89](https://aarontay.substack.com/p/chatgpt-plus-new-dall-e-3-image)
[90](https://deepmind.google/models/imagen/)
[91](https://ostin.tistory.com/231)
[92](https://openai.com/index/dall-e-3/)
[93](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/imagen/)
[94](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[95](https://dev.to/dkechag/ai-image-creation-chatgpt-vs-gemini-vs-dalle-vs-grok-558e)
[96](http://visnyk-sport.kpnu.edu.ua/article/view/278998/273640)
[97](https://jurnal.stienganjuk.ac.id/index.php/ojsmadani/article/view/167)
[98](https://www.semanticscholar.org/paper/c7cbe221b34fff8b229d38e366528f69bf0edb41)
[99](https://ieeexplore.ieee.org/document/9932926/)
[100](https://arxiv.org/abs/2310.13573)
[101](https://ieeexplore.ieee.org/document/10289721/)
[102](https://www.mdpi.com/2071-1050/15/11/8445)
[103](https://jurnal.habi.ac.id/index.php/Bahtra/article/view/241)
[104](https://link.springer.com/10.1007/s10291-023-01555-w)
[105](https://www.semanticscholar.org/paper/05eabf7a92dcc1c04b39cc80456337eb9bfb14a1)
[106](https://arxiv.org/html/2411.03999)
[107](https://arxiv.org/html/2409.20340)
[108](https://arxiv.org/pdf/1902.03984.pdf)
[109](https://arxiv.org/pdf/2411.16567.pdf)
[110](http://arxiv.org/pdf/2102.07074v2.pdf)
[111](http://arxiv.org/pdf/2405.11614.pdf)
[112](https://www.sciencedirect.com/science/article/abs/pii/S0925231225005363)
[113](https://arxiv.org/html/2505.21162v1)
[114](https://aclanthology.org/2023.findings-acl.291.pdf)
[115](https://ieeexplore.ieee.org/document/11118869/)
[116](https://www.sciencedirect.com/science/article/abs/pii/S1077314218304272)
[117](https://arxiv.org/html/2408.11135v1)
[118](https://www.sciencedirect.com/science/article/abs/pii/S1077314224001231)
[119](https://www.nature.com/articles/s41598-023-45290-1)
[120](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202062663815465)
[121](https://ieeexplore.ieee.org/iel8/6287639/10820123/10804166.pdf)
[122](https://www.nature.com/articles/s41598-023-28094-1)
[123](https://chatpaper.com/paper/142402)
[124](https://ieeexplore.ieee.org/iel7/5971803/10258149/10258251.pdf)
[125](https://dl.acm.org/doi/full/10.1145/3745238.3745244)
