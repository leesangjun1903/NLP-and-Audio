# InjectFusion: Training-free Content Injection using h-space in Diffusion Models

### I. 논문의 핵심 주장과 주요 기여

**InjectFusion**은 확산 모델(Diffusion Models, DMs)의 U-Net 병목 부분인 h-space(semantic latent space)를 직접 조작하여, 한 이미지의 콘텐츠를 다른 이미지에 주입하는 훈련-무료 방법을 제시합니다.[1]

핵심 주장은 다음과 같습니다:
1. h-space는 생성 이미지의 의미론적 정보(내용)를 담고 있으며, 이를 조작함으로써 세밀한 콘텐츠 제어가 가능함
2. 기존 DreamBooth나 LoRA 기반 방법과 달리, 최적화나 미세조정 없이 사전학습된 확산 모델만으로 콘텐츠 주입이 가능함
3. Skip connection과의 상관관계를 유지하면서 h-space를 조작하는 것이 고품질 이미지 생성의 핵심

### II. 해결하는 문제와 제안 방법

#### 2.1 기본 문제
확산 모델의 중간 변수들이 충분히 연구되지 않아, 생성 과정의 직접적인 제어가 어렵습니다. 특히 GAN의 잠재 공간처럼 명시적인 조작이 불가능했습니다.

#### 2.2 제안 방법: InjectFusion의 두 가지 핵심 기술

**1) Slerp를 통한 점진적 블렌딩**

두 이미지의 h-space 특징을 단순히 더하거나 교체하면 심각한 왜곡이 발생합니다. 이를 해결하기 위해 정규화된 구면 선형 보간(Slerp)을 사용합니다:

$$h_t^{mixed} = \text{Slerp}(h_t^{original}, h_t^{content}; \gamma)$$

여기서 $\gamma \in $은 콘텐츠 주입 비율을 제어합니다. Slerp는 특징 맵의 규범(norm)을 보존하여 h-space와 skip connection 간의 상관관계를 유지합니다.[1]

$$\rho_{homo} = \frac{\text{Cov}(\|h\|, \|g\|)}{\sigma(\|h\|)\sigma(\|g\|)}$$

여기서 $\rho_{homo}$는 원본 이미지의 h-space와 skip connection 간의 Pearson 상관계수로, 약 0.3 이상의 높은 양의 상관을 보입니다.[1]

**2) Latent Calibration**

Skip connection과 h-space 간의 상관관계를 더욱 정밀하게 유지하기 위해 latent 변수 $x_t$를 직접 보정합니다:

$$\Delta x_t = \alpha \frac{\partial P_t}{\partial x_t} w(1-\alpha) \rho_t + \epsilon$$

보정의 강도는 하이퍼파라미터 $w$로 제어되며, DDIM 방정식을 통해 다음 스텝으로 진행합니다:

$$x_{t-1} = \alpha_{t-1} P_t(x_t + \Delta x_t) + \sqrt{1-\alpha_{t-1}^2} D_t(x_t + \Delta x_t)$$

이 방법은 InjectFusion뿐 아니라 Plug-and-Play, MasaCtrl 등 다른 특징 조작 방법에도 적용 가능합니다.[1]

#### 2.3 모델 구조

InjectFusion의 전체 파이프라인:
1. 두 이미지를 DDIM 순방향 과정을 통해 노이즈로 역변환
2. 원본 이미지의 생성 과정을 $t_{edit}$부터 $t=1$까지 진행하면서 콘텐츠 이미지의 h-space를 주입
3. 각 타임스텝에서 Slerp를 통해 h-space 블렌딩
4. Latent calibration으로 skip connection 상관관계 유지
5. 최종 이미지 생성

모든 조작은 사전학습된 모델의 가중치를 동결한 상태에서 이루어지므로 훈련이 필요하지 않습니다.

### III. 성능 향상 및 정량적 결과

#### 3.1 핵심 성능 지표

표 1: InjectFusion의 구성 요소별 성능 비교[1]

| 방법 | FID | ID 유사도 | Gram Loss |
|------|-----|----------|-----------|
| h-space 단순 교체 | 49.94 | 0.3581 | 0.0415 |
| 선형 보간(Lerp) | 36.89 | 0.4040 | 0.0318 |
| Slerp | 32.09 | 0.4390 | 0.0310 |

Slerp를 사용하면 FID가 49.94에서 32.09로 개선되며, ID 유사도는 0.3581에서 0.4390으로 향상됩니다.[1]

#### 3.2 콘텐츠 주입 비율($\gamma$) 선택

실험 결과, $\gamma \leq 0.6$일 때 최적의 성능을 보입니다. $\gamma > 0.6$이면 콘텐츠 변화가 포화되면서 FID 저하와 인공적 왜곡이 발생합니다.[1]

#### 3.3 편집 구간($t_{edit}$) 결정

최적의 편집 개시 타임스텝은 $t_{edit} = 400$입니다. 이 설정에서:
- 질 좋은 콘텐츠 주입 달성
- FID와 ID 유사도의 균형 유지
- Gram loss 최소화[1]

#### 3.4 내용 정의 (User Study)

사용자 연구(50명)를 통해 콘텐츠와 원본 요소의 정의를 명확히 했습니다:[1]

| 속성 | 콘텐츠 비율 | 원본 비율 |
|------|-----------|----------|
| 코, 눈, 턱선 | 71.94% | 28.06% |
| 표정 | 63.27% | 36.73% |
| 안경 | 94.37% | 5.63% |
| 머리 색상 | 4.26% | 95.74% |
| 메이크업 | 9.30% | 90.60% |

이는 InjectFusion이 **의미론적 특징(콘텐츠 생성)과 색상 분포(원본 요소)를 효과적으로 분리**함을 보여줍니다.

#### 3.5 다양한 데이터셋에서의 성능

InjectFusion은 다양한 아키텍처와 데이터셋에서 일관된 성능을 보입니다:[1]
- **픽셀 공간 DM**: DDPM, iDDPM, ADM
- **데이터셋**: CelebA-HQ, AFHQ-Dog, METFACES, LSUN-Church/Bedroom, ImageNet

기술적으로 중요한 발견은 **pixel-level DM에서 h-space의 특성이 명확**하지만, **Latent Diffusion Model(Stable Diffusion)에서는 h-space가 VAE의 잠재 공간에서 작동하기 때문에 의미론적 특성이 다르다**는 것입니다.[1]

### IV. 모델의 일반화 성능 분석

#### 4.1 도메인 내(In-Domain) 성능

사전학습된 데이터 분포 내에서는 우수한 일반화를 보입니다. CelebA-HQ와 AFHQ 데이터셋에서의 실험 결과는 얼굴, 동물 이미지 모두에서 일관된 콘텐츠 주입을 달성합니다.[1]

#### 4.2 도메인 외(Out-of-Domain) 성능

흥미로운 점은 **예술적 스타일의 이미지도 원본 이미지로 사용 가능**하다는 것입니다. 이는 skip connection이 원본 이미지의 색상 분포와 구조 정보를 보존하기 때문입니다. 다만:[1]
- Out-of-domain 이미지를 **콘텐츠**로 사용하면 의미 있는 결과가 나오지 않음
- 이는 h-space가 "보편적 콘텐츠 표현"이 아닌 "학습 데이터 분포 내 의미론적 표현"임을 의미[1]

#### 4.3 Latent Diffusion Model에서의 제한

Stable Diffusion 실험 결과, h-space의 특성이 달라집니다:[1]
- 더 많은 의미론적 요소가 주입됨
- 픽셀 공간 DM만큼 명확하지 않음
- 이는 VAE 인코더가 이미 의미론적 특징을 압축하기 때문

이는 **향후 연구가 latent diffusion model의 메커니즘을 더 깊이 있게 이해**해야 함을 시사합니다.

### V. 한계와 개선 가능성

#### 5.1 현재 한계

1. **공간 해상도 제한**: h-space의 작은 공간 해상도(8×8×256)로 인해 세밀한 지역적 제어가 어렵습니다. 마스크 기반 지역 혼합을 제공하지만, 경계선에서 인공적 현상이 발생할 수 있습니다.[1]

2. **Out-of-domain 콘텐츠 취약성**: 학습 분포 밖의 이미지를 콘텐츠로 사용하면 의미 없는 결과가 나옵니다. 이는 h-space의 일반성 한계를 드러냅니다.[1]

3. **Latent 모델 호환성**: Stable Diffusion과 같은 VAE 기반 latent diffusion model에서는 h-space의 의미론적 특성이 불명확합니다.[1]

#### 5.2 향후 연구 방향

1. **고해상도 h-space 활용**: 다층 특징 맵을 결합하거나 U-Net의 다른 병목을 탐색하여 공간 해상도를 개선할 수 있습니다.

2. **Latent diffusion의 메커니즘 규명**: Stable Diffusion과 같은 모델에서 h-space의 역할을 더 깊이 있게 이해해야 합니다.

3. **적응형 γ 스케줄**: 현재 고정된 $\gamma$ 값 대신, 타임스텝에 따라 동적으로 변하는 스케줄 탐색.[1]

4. **Cross-modal 확장**: 텍스트, 음성 등 다양한 모달리티를 통한 콘텐츠 제어.

### VI. 최신 관련 연구 비교 분석 (2020년 이후)

#### 6.1 선행 연구

**Asyrp (2022)**[2]
- h-space를 **처음 발견**한 논문
- 속성 편집(예: 미소 추가)에만 초점
- 비대칭 역방향 과정(Asymmetric Reverse Process) 도입
- 특징: 단순한 벡터 이동으로 의미론적 변화 달성

**InjectFusion과의 차이**: Asyrp는 h-space의 **방향성 편집**에 주력했다면, InjectFusion은 **특징 맵의 완전한 교체**를 통한 콘텐츠 주입을 새롭게 제시합니다.[1]

#### 6.2 경쟁 기술

**DiffuseIT (2023)**[3]
- 자주의(Self-Attention) 특징을 조작하여 스타일 전달
- DINO ViT를 사용한 콘텐츠 보존
- 비교 실험 결과, InjectFusion은 색상 분포 차이가 큰 경우 더 우수한 성능[1]

**특징 비교**:

| 측면 | InjectFusion | DiffuseIT |
|------|-------------|----------|
| 병목 | h-space | 자주의 |
| 훈련 | 불필요 | 불필요 |
| 외부 네트워크 | 없음 | DINO ViT 필요 |
| 색상 보존 | 우수 | 약함 |
| 의존성 | 폐쇄형 | 외부 모델 의존 |

**ControlNet (2023)**[4]
- 공간적 조건부 제어(엣지, 깊이, 포즈 등)
- 대규모 사전학습 모델 잠금 및 재사용
- 제어 범위가 **구조/형태로 제한**되어 세밀한 콘텐츠 주입 불가

**Style Injection (2024)**[5]
- 자주의 계층의 key/value 치환으로 스타일 전달
- Query 보존 및 주의 온도 스케일링 기법 도입
- Stable Diffusion과 같은 대규모 모델 적용 가능

#### 6.3 관련 개념적 진전

**h-space의 의미론적 특성 규명**

Park et al. (2023)은 x-space 트래버설을 통해 이미지 편집이 가능함을 보였고, 이는 인코더 특징 맵의 pullback metric으로 유도되는 지역 잠재 기저를 활용합니다.[6]

**Latent space disentanglement (2024)**[7]

Diffusion Transformer(DiT)의 잠재 공간이 본질적으로 의미론적으로 분해되어 있으며, text와 image 잠재 공간이 함께 작동함을 발견했습니다. 이는 InjectFusion의 h-space 조작이 특정 아키텍처에 한정될 수 있음을 시사합니다.

#### 6.4 종합적 비교 표

| 방법 | 연도 | 핵심 기술 | 훈련 필요 | 외부 네트워크 | 적용 범위 |
|------|------|---------|---------|-----------|---------|
| Asyrp | 2022 | 방향성 h-space 편집 | 불필요 | 없음 | 속성 변화 |
| DiffuseIT | 2023 | 자주의 조작 + DINO | 불필요 | DINO ViT | 스타일 전달 |
| **InjectFusion** | **2023** | **Slerp + Latent Calib** | **불필요** | **없음** | **콘텐츠 주입** |
| ControlNet | 2023 | 조건부 인코더 | 필요 | 자체 아키텍처 | 구조 제어 |
| Style Injection | 2024 | Key/Value 치환 | 불필요 | 없음 | 스타일 전달 |

### VII. 연구의 영향과 의의

#### 7.1 기술적 기여

1. **h-space 활용의 확장**: Asyrp의 속성 편집을 넘어 **완전한 콘텐츠 주입**이 가능함을 보임
2. **skip connection 메커니즘 규명**: U-Net의 skip connection이 원본 이미지의 색상/구조 정보를 담고 있음을 명확히 함
3. **정규화 기반 보간 기법**: Slerp를 통한 통계 보존은 다른 특징 조작 방법에도 적용 가능

#### 7.2 실용적 영향

- **훈련-무료**: 기존 LoRA, DreamBooth와 달리 시간 소비 없이 사용 가능
- **외부 의존성 제거**: DINO, CLIP 같은 외부 네트워크 불필요
- **광범위한 호환성**: 다양한 아키텍처 지원

#### 7.3 향후 연구 방향

1. **Multimodal 확산 모델 확장**: 음성, 텍스트 등을 포함한 h-space 활용
2. **고해상도 생성**: 현재 256×256 해상도를 넘어 4K 생성 가능성 탐색
3. **이론적 이해 심화**: h-space가 왜 특정 의미론적 특성을 갖는지에 대한 수학적 분석
4. **적대적 안정성**: h-space 조작의 안정성과 adversarial robustness 연구

### VIII. 결론

InjectFusion은 확산 모델의 중간 특징 공간에 대한 이해를 한 단계 진전시킨 중요한 연구입니다. **Slerp와 Latent Calibration의 결합**으로 skip connection과의 상관관계를 유지하면서도 정교한 콘텐츠 주입이 가능함을 보였습니다. 

특히 **훈련 없이 사전학습 모델만으로** 고품질 이미지 조작이 가능하다는 점은 실용적 가치가 높습니다. 다만 h-space의 도메인 한계와 latent diffusion 모델과의 호환성 문제는 향후 해결해야 할 과제입니다.

2023-2024년의 후속 연구들(Style Injection, Latent Disentanglement 등)은 InjectFusion이 개척한 **특징 공간 조작의 길**을 따르며, 더욱 정교한 제어와 이론적 이해를 추구하고 있습니다.

***

**참고문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3da5dd8d-383d-4349-8c0a-f745895eabb6/2303.15403v2.pdf)
[2](https://ieeexplore.ieee.org/document/10484256/)
[3](https://ieeexplore.ieee.org/document/10656174/)
[4](https://ieeexplore.ieee.org/document/10657216/)
[5](https://arxiv.org/abs/2403.01633)
[6](https://arxiv.org/abs/2403.02332)
[7](https://arxiv.org/abs/2410.19429)
[8](https://arxiv.org/abs/2406.14555)
[9](https://arxiv.org/abs/2410.14429)
[10](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[11](https://dl.acm.org/doi/10.1145/3707292.3707367)
[12](http://arxiv.org/pdf/2303.15403.pdf)
[13](https://arxiv.org/html/2403.09055)
[14](https://techrxiv.figshare.com/articles/preprint/Diffusion-based_Reinforcement_Learning_for_Edge-enabled_AI-Generated_Content_Services/24648723/1/files/43317999.pdf)
[15](https://arxiv.org/abs/2312.09008)
[16](https://arxiv.org/html/2502.06805v1)
[17](https://arxiv.org/html/2312.08873)
[18](https://arxiv.org/html/2409.07451)
[19](http://arxiv.org/pdf/2306.06874.pdf)
[20](https://arxiv.org/abs/2303.15403)
[21](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Wu_LatentPS_Image_Editing_Using_Latent_Representations_in_Diffusion_Models_WACVW_2025_paper.pdf)
[22](https://aclanthology.org/2024.acl-short.24.pdf)
[23](https://openaccess.thecvf.com/content/WACV2024/papers/Jeong_Training-Free_Content_Injection_Using_H-Space_in_Diffusion_Models_WACV_2024_paper.pdf)
[24](https://www.ewadirect.com/proceedings/tns/article/view/2768)
[25](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06072.pdf)
[26](https://curryjung.github.io/InjectFusion/)
[27](https://papers.neurips.cc/paper_files/paper/2023/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffstyle/)
[29](https://openaccess.thecvf.com/content/WACV2024/html/Jeong_Training-Free_Content_Injection_Using_H-Space_in_Diffusion_Models_WACV_2024_paper.html)
[30](https://openaccess.thecvf.com/content/CVPR2024/papers/Chung_Style_Injection_in_Diffusion_A_Training-free_Approach_for_Adapting_Large-scale_CVPR_2024_paper.pdf)
[31](https://arxiv.org/abs/2307.12868)
[32](https://arxiv.org/html/2509.20091v1)
[33](https://arxiv.org/html/2502.02225v1)
[34](https://arxiv.org/html/2509.15257v1)
[35](https://arxiv.org/html/2303.15403v2)
[36](https://arxiv.org/abs/2210.10960)
[37](https://arxiv.org/abs/2212.06013)
[38](https://ieeexplore.ieee.org/document/10204406/)
[39](https://link.springer.com/10.1007/s10462-023-10504-5)
[40](https://arxiv.org/abs/2411.08196)
[41](https://arxiv.org/abs/2408.13335)
[42](https://ieeexplore.ieee.org/document/10581912/)
[43](https://arxiv.org/abs/2402.12423)
[44](https://arxiv.org/abs/2305.14742)
[45](https://arxiv.org/abs/2302.12469)
[46](https://arxiv.org/pdf/2403.12585.pdf)
[47](https://arxiv.org/pdf/2210.11427.pdf)
[48](https://arxiv.org/html/2411.13982v2)
[49](http://arxiv.org/pdf/2503.08116.pdf)
[50](https://arxiv.org/abs/2408.16845)
[51](https://arxiv.org/html/2411.08196)
[52](https://arxiv.org/html/2404.01050v1)
[53](https://deepai.org/publication/diffusion-models-already-have-a-semantic-latent-space)
[54](https://pure.kaist.ac.kr/en/publications/diffusion-based-image-translation-using-disentangled-style-and-co)
[55](https://huggingface.co/papers/2302.05543)
[56](https://jeonghwarr.github.io/posts/diffusion_models_already_have_a_semantic_latent_space/)
[57](https://openreview.net/pdf/b3174c74984a2e90538982e09e40225a474d34e0.pdf)
[58](https://stable-diffusion-art.com/controlnet/)
[59](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffuseit/)
[60](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)
[61](https://github.com/zilaeric/asyrp-extension/blob/main/blogpost.md)
[62](https://arxiv.org/html/2512.12800v1)
[63](https://openaccess.thecvf.com/content/CVPR2023/papers/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.pdf)
[64](https://arxiv.org/abs/2302.05543)
[65](https://arxiv.org/abs/2209.15264)
[66](https://www.semanticscholar.org/paper/Adding-Conditional-Control-to-Text-to-Image-Models-Zhang-Rao/efbe97d20c4ffe356e8826c01dc550bacc405add)
[67](https://www.semanticscholar.org/paper/Diffusion-Models-already-have-a-Semantic-Latent-Kwon-Jeong/a02313d56a6f71be9aafe43628e0f3a1d0cb858e)
[68](https://arxiv.org/pdf/2211.12572.pdf)
[69](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
[70](https://yonsei.elsevierpure.com/en/publications/diffusion-models-already-have-a-semantic-latent-space)
