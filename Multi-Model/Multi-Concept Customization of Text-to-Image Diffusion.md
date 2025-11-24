
# Multi-Concept Customization of Text-to-Image Diffusion

## 1. 핵심 주장과 주요 기여

Custom Diffusion은 대규모 사전학습된 text-to-image diffusion model을 소수의 이미지(4~20장)로 개인화할 수 있는 효율적인 fine-tuning 기법이다. 본 논문의 핵심 주장은 **cross-attention layer의 key와 value projection matrix만을 업데이트**하는 것만으로도 새로운 개념을 효과적으로 학습할 수 있다는 점이다.[1]

주요 기여는 다음과 같다:

**효율성**: 전체 모델 파라미터의 3%만 업데이트하여 학습 시간을 6분(A100 GPU 2대 기준)으로 단축하고, 각 개념당 75MB의 저장공간만 필요하다. 이는 DreamBooth 대비 2~4배 빠르며, 저장공간은 3GB 대비 40배 이상 효율적이다.[2][1]

**다중 개념 합성**: 여러 개념을 동시에 학습하거나(joint training), 개별 학습된 모델을 closed-form constrained optimization을 통해 병합할 수 있다. 최적화 기반 방법의 수식은 다음과 같다:[1]

$$\hat{W} = \arg\min_{W} ||WC_{reg}^{\top} - W_0C_{reg}^{\top}||_F$$

subject to $$WC^{\top} = V$$, where $$C = [c_1 \cdots c_N]^{\top}$$ and $$V = [W_1c_1^{\top} \cdots W_Nc_N^{\top}]^{\top}$$

이는 Lagrange multiplier를 사용하여 closed-form으로 해결된다:[1]

$$\hat{W} = W_0 + v^{\top}d$$

where $$d = C(C_{reg}^{\top}C_{reg})^{-1}$$ and $$v^{\top} = (V - W_0C^{\top})(dC^{\top})^{-1}$$

**망각 방지**: LAION-400M 데이터셋에서 타겟 이미지와 유사한 캡션을 가진 200개의 정규화 이미지를 검색하여 사용함으로써 모델이 기존 개념을 잊는 것을 방지한다.[1]

## 2. 문제 정의 및 제안 방법

### 해결하고자 하는 문제

기존 text-to-image 모델은 대규모 데이터로 학습되어 일반적인 개념은 잘 생성하지만, 사용자의 **개인화된 개념**(예: 특정 애완동물, 개인 소장품)을 생성하는 데 한계가 있다. Fine-tuning 시 발생하는 주요 문제는:[1]

1. **Language drift**: 새로운 개념 학습 시 기존 개념의 의미가 손실됨 (예: "moongate" 학습 시 "moon"과 "gate"의 의미 상실)[1]
2. **Overfitting**: 소수의 샘플에 과적합되어 다양성 감소[1]
3. **Compositional challenges**: 여러 새로운 개념을 동시에 합성하는 것의 어려움[1]

### 제안 방법 및 모델 구조

**Diffusion Model 학습 목표**:

기본적으로 Latent Diffusion Model(Stable Diffusion)을 backbone으로 사용하며, 학습 목표는 다음과 같다:[1]

$$\mathbb{E}_{\epsilon, x, c, t}[w_t||\epsilon - \epsilon_{\theta}(x_t, c, t)||]$$

여기서 $$x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$$이며, $$\epsilon_{\theta}$$는 모델 예측, $$w_t$$는 시간 의존적 가중치이다.[1]

**Cross-Attention Mechanism**:

본 논문의 핵심은 cross-attention layer에서 key와 value projection만 업데이트하는 것이다. Single-head cross-attention 연산은 다음과 같다:[1]

$$Q = W^q f, \quad K = W^k c, \quad V = W^v c$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^{\top}}{\sqrt{d'}}\right)V$$

여기서 $$f \in \mathbb{R}^{(h \times w) \times l}$$는 latent image feature, $$c \in \mathbb{R}^{s \times d}$$는 text feature이다. **$$W^k$$와 $$W^v$$만 학습**한다.[1]

**가중치 변화 분석**:

저자들은 모든 레이어의 가중치 변화율을 분석했다:[1]

$$\Delta_l = \frac{||\theta'_l - \theta_l||}{||\theta_l||}$$

실험 결과, cross-attention layer의 변화율이 다른 레이어보다 현저히 높았으며(전체 파라미터의 5%만 차지함에도), 이는 이 레이어가 fine-tuning에 핵심적임을 시사한다.[1]

**Text Encoding**:

- 일반 개념: 기존 텍스트 설명 사용 (예: "moongate")
- 개인화 개념: 새로운 modifier token $$V^\*$$ 도입 (예: " $$V^\*$$ dog ")[1]
- $$V^*$$는 rare token으로 초기화되고 cross-attention 파라미터와 함께 최적화됨[1]

**Data Augmentation**:

학습 중 타겟 이미지를 0.4~1.4배 무작위로 resize하고, resize 비율에 따라 텍스트 프롬프트에 "very small", "far away", "zoomed in", "close up" 등을 추가한다. 이는 수렴 속도를 높이고 결과를 개선한다.[1]

**학습 설정**:

- Batch size: 8
- Learning rate: $$8 \times 10^{-5}$$
- Single-concept: 250 steps
- Multi-concept (joint): 500 steps[1]

### 다중 개념 학습 방법

**Joint Training**: 각 개념의 학습 데이터를 결합하여 동시에 학습하며, 서로 다른 modifier token ( $$V_1^\*, V_2^*$$ )을 사용한다.[1]

**Optimization-based Merging**: 개별 학습된 모델을 위의 closed-form solution으로 병합한다. 이 방법은 약 2초만 소요되어 joint training보다 빠르다.[1]

## 3. 성능 향상 및 한계

### 정량적 성능

**Single-concept Fine-tuning**:[1]
- Text-alignment: 0.795 (Textual Inversion: 0.670, DreamBooth: 0.781)
- Image-alignment: 0.775 (Textual Inversion: 0.827, DreamBooth: 0.776)
- KID (validation): 20.96×10⁻³ (DreamBooth: 32.53×10⁻³)

**Multi-concept Fine-tuning**:[1]
- Joint training - Text-alignment: 0.801, Image-alignment: 0.706
- Optimization - Text-alignment: 0.800, Image-alignment: 0.695
- DreamBooth - Text-alignment: 0.783, Image-alignment: 0.695

**Human Preference Study**:[1]
- Single-concept: 72.62% text-alignment 선호도, 51.62% image-alignment 선호도
- Multi-concept (Joint): 86.65% text-alignment 선호도, 81.89% image-alignment 선호도

### 일반화 성능 향상

**정규화 데이터셋의 효과**: CLIP text encoder feature space에서 0.85 이상의 유사도를 가진 200개의 실제 이미지를 정규화에 사용함으로써, 기존 개념의 망각을 효과적으로 방지한다. 정규화 없이 학습한 경우 KID가 32.64×10⁻³로 증가하여 overfitting이 심화되었다.[1]

**생성 이미지를 정규화로 사용한 경우**: DreamBooth처럼 생성된 이미지를 정규화로 사용하면 타겟 개념에서는 유사한 성능을 보이지만, KID가 34.70×10⁻³로 높아져 일반화 성능이 저하된다.[1]

**모델 압축 가능성**: Singular Value Decomposition(SVD)을 통해 업데이트된 가중치 차이를 low-rank로 근사하면, 상위 60% rank만 사용해도(5배 압축) 성능이 유사하게 유지된다. 이는 일반화 능력을 유지하면서도 저장공간을 추가로 절감할 수 있음을 시사한다.[1]

**MS-COCO FID**: 대부분의 데이터셋에서 16.25~17.21로 사전학습 모델(16.35)과 유사한 수준을 유지하여, 관련 없는 개념에 대한 생성 분포가 거의 변하지 않음을 보여준다.[1]

### 한계점

**어려운 합성**: 개와 고양이처럼 자주 함께 등장하지 않는 개념의 합성은 여전히 어렵다. Attention map 분석 결과, "dog"와 "cat" token의 attention이 크게 겹쳐 분리된 생성이 어려운 것으로 나타났다.[1]

**다중 개념 확장의 한계**: 3개 이상의 개념을 합성하는 것은 여전히 도전적이다. 일부 조합(예: 2개 객체 + 1개 스타일)에서는 성공했으나, 일반화는 제한적이다.[1]

**사전학습 모델의 한계 상속**: 사전학습 모델이 생성하지 못하는 조합은 fine-tuning 후에도 여전히 어렵다.[1]

**초기화 민감성**: $$V^*$$ token의 초기화 방법에 따라 성능이 달라진다. Random normal distribution으로 초기화하면 image-alignment가 0.765로 감소하고, 원래 카테고리 단어의 매핑이 타겟 이미지로 더 많이 shift된다.[1]

## 4. 후속 연구에 미치는 영향 및 고려사항

### 영향력

Custom Diffusion은 personalized image generation 분야의 기초 연구로, 1195회 이상 인용되었다. 주요 영향은 다음과 같다:[3]

**Parameter-Efficient Fine-Tuning의 방향 제시**: Cross-attention layer의 일부만 업데이트하는 접근이 후속 연구의 표준이 되었다. Personalized Residuals (CVPR 2024)는 이를 발전시켜 cross-attention output projection의 LoRA만 fine-tuning하여 파라미터를 모델의 0.1%로 더 줄였다.[4][5]

**다중 개념 합성 연구 촉진**: TweedieMix (ICLR 2025)는 Tweedie's formula를 사용하여 여러 customized model을 inference 단계에서 합성하는 방법을 제안했다. MC² (CVPR 2025)는 multi-concept guidance 기법을 통해 합성 성능을 더욱 향상시켰다.[6][7]

**Dataset 기여**: CustomConcept101 데이터셋(101개 개념과 평가 프롬프트)은 personalization 연구의 표준 벤치마크가 되었다.[2][1]

### 최신 연구 동향 (2024-2025)

**Survey 논문의 등장**: 2024년 이후 personalized content synthesis에 대한 포괄적 survey가 등장했다. 이들은 방법론을 Test-Time Fine-Tuning (TTF)과 Pre-Trained Adaptation (PTA)로 분류하며, Custom Diffusion은 TTF의 대표적 방법으로 분류된다.[5][8][9]

**Tuning-Free 방향**: IDAdapter (2024), Imagine yourself (2024) 등은 fine-tuning 없이 personalization을 수행하는 방법을 제안하여, inference 시간을 더욱 단축시켰다.[10][11][5]

**Diffusion Transformer (DiT) 적응**: "Personalize Anything for Free with Diffusion Transformer" (2025)는 DiT 구조에서 denoising token을 reference subject의 token으로 교체하는 것만으로 zero-shot reconstruction이 가능함을 보였다.[12]

**Continual Learning**: "Mining Your Own Secrets" (2025)와 "Continual Personalization for Diffusion Models" (ICCV 2025)는 여러 개념을 순차적으로 학습하면서 이전 개념을 잊지 않는 continual learning 접근을 제안했다.[13][14]

**Video로의 확장**: ConceptMaster (2025)는 video customization으로 확장하여, multi-concept video generation을 가능하게 했다.[15][5]

### 향후 연구 시 고려사항

**일반화 성능 개선**:

1. **정규화 전략의 발전**: 단순히 유사 이미지를 검색하는 것을 넘어, diffusion classifier scores를 활용한 parameter-space와 function-space 정규화가 제안되고 있다. 향후 연구는 정규화 데이터 선택의 자동화와 최적화를 고려해야 한다.[13]

2. **Low-rank 구조 활용**: Custom Diffusion은 사후 압축에 SVD를 사용했지만, PaRa (ICLR 2024)는 학습 중 명시적으로 rank를 제어하여 생성 공간을 제한하고 일반화를 향상시켰다. 향후 연구는 학습 단계부터 low-rank constraint를 통합하는 방향을 탐구해야 한다.[16][5][1]

3. **Attribute Disentanglement**: ProSpect (2023)는 material, style, layout 등 특정 visual attribute를 분리하여 편집하는 방법을 제안했다. 일반화를 위해서는 개념의 identity와 attribute를 분리하는 연구가 필요하다.[17][5]

**모델 병합의 고도화**:

Custom Diffusion의 closed-form optimization은 단순하지만, 복잡한 합성에서 한계가 있다. TokenVerse (2025)는 token modulation space에서 여러 개념을 disentangle하는 방법을 제안했다. 향후 연구는 semantic-level에서 개념을 분리하고 조합하는 방법을 탐구해야 한다.[18][1]

**Compositional Generation 강화**:

Attention mechanism 분석 결과, 유사한 개념들의 attention이 겹치는 문제가 있다. 향후 연구는:[1]
- Layout control을 통한 spatial separation[5]
- Multi-object aware sampling[6]
- Attention re-weighting 기법[5]
등을 통해 compositional generation을 개선해야 한다.

**윤리적 고려사항**:

Survey 논문들은 fake image detection의 중요성을 강조한다. 특히 얼굴 personalization은 deepfake 악용 가능성이 있어, 다음을 고려해야 한다:[19][5]
- Watermarking 기술 통합[5]
- Generated image detection 방법 개발[5]
- 사용자 consent mechanism 설계[5]

**계산 효율성**:

모바일 디바이스에서의 personalization을 위해 "Hollowed Net" (2025)은 on-device fine-tuning을 가능하게 하는 메모리 효율적 방법을 제안했다. 향후 연구는 edge computing 환경을 고려한 경량화를 탐구해야 한다.[20]

**평가 메트릭 개선**:

현재 CLIP 기반 메트릭은 subject fidelity와 text alignment의 trade-off를 잘 포착하지 못한다. 향후 연구는:[9][5]
- Human preference에 더 align된 메트릭 개발
- Identity preservation의 정량적 측정 방법
- Compositional correctness 평가 지표
등을 개발해야 한다.

**Auto-regressive Models와의 통합**:

"Personalized Text-to-Image Generation with Auto-Regressive Models" (2025)는 transformer 기반 auto-regressive model에서도 personalization이 가능함을 보였다. Diffusion과 auto-regressive의 장점을 결합하는 hybrid 접근이 유망한 방향이다.[21]

Custom Diffusion은 효율적인 parameter selection, 정규화 기법, 그리고 다중 개념 합성 방법을 통해 personalized generation의 기초를 마련했으며, 이후 연구들은 일반화 성능, 계산 효율성, compositional ability를 개선하는 방향으로 발전하고 있다.[8][9][5][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/648082b9-2e7e-4352-99c9-fd1774adb896/2212.04488v2.pdf)
[2](https://www.cs.cmu.edu/~custom-diffusion/)
[3](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf)
[4](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/personalized-residuals/)
[5](https://link.springer.com/10.1007/s11633-025-1563-3)
[6](https://openreview.net/forum?id=ee2c4MEx9l)
[7](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Jiang_MC2_Multi-concept_Guidance_for_Customized__Multi-concept_Generation_CVPR_2025_paper.pdf)
[8](http://arxiv.org/pdf/2405.05538.pdf)
[9](https://arxiv.org/abs/2405.05538)
[10](http://arxiv.org/pdf/2403.13535.pdf)
[11](http://arxiv.org/pdf/2409.13346.pdf)
[12](https://arxiv.org/html/2503.12590v1)
[13](https://arxiv.org/html/2410.00700v3)
[14](https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_Continual_Personalization_for_Diffusion_Models_ICCV_2025_paper.pdf)
[15](https://arxiv.org/abs/2501.04698)
[16](https://liner.com/ko/review/para-personalizing-texttoimage-diffusion-via-parameter-rank-reduction)
[17](https://dl.acm.org/doi/pdf/10.1145/3618342)
[18](https://arxiv.org/html/2501.12224v1)
[19](https://www.sciencedirect.com/science/article/abs/pii/S1568494625007811)
[20](https://www.themoonlight.io/ko/review/hollowed-net-for-on-device-personalization-of-text-to-image-diffusion-models)
[21](https://arxiv.org/abs/2504.13162)
[22](https://arxiv.org/pdf/2211.01324.pdf)
[23](https://arxiv.org/abs/2302.13861)
[24](https://arxiv.org/pdf/2307.01097.pdf)
[25](http://arxiv.org/pdf/2406.18944.pdf)
[26](https://hhhhhsk.tistory.com/35)
[27](https://seunkorea.tistory.com/44)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/custom-diffusion/)
[29](https://tech.kakao.com/posts/709)
[30](https://scienceon.kisti.re.kr/srch/selectPORSrchArticleOrgnl.do?cn=JAKO202014151761480)
[31](https://juniboy97.tistory.com/119)
[32](https://github.com/adobe-research/custom-diffusion)
[33](https://www.codil.or.kr/filebank/original/RK/OTKCRK240336/OTKCRK240336.pdf)
[34](https://www.kipo.go.kr/ko/kpoBultnFileDown.do?ntatcSeq=16993&ntatcAtflSeq=1&sysCd=&aprchId=BUT0000048)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/prompt-free-diffusion/)
[36](https://www.sciencedirect.com/science/article/pii/S0957417425017890)
[37](https://smartcity.cau.ac.kr/%EC%9E%AC%EC%84%A0%EC%A0%95%ED%8F%89%EA%B0%80_%EC%8B%A0%EC%B2%AD%EC%84%9C.pdf)
[38](https://dmqa.korea.ac.kr/activity/seminar/452)
[39](https://posthumanism.co.uk/jp/article/view/3283)
[40](https://jurnal.stikes-ibnusina.ac.id/index.php/SABER/article/view/3184)
[41](https://jurnalp4i.com/index.php/academia/article/view/4981)
[42](https://jurnal.uinsyahada.ac.id/index.php/JP/article/view/14309)
[43](http://medrxiv.org/lookup/doi/10.1101/2025.08.09.25333364)
[44](https://arxiv.org/abs/2504.16081)
[45](https://arxiv.org/abs/2409.05033)
[46](https://journals.sagepub.com/doi/10.1177/21582440251363668)
[47](https://arxiv.org/abs/2503.13576)
[48](https://arxiv.org/html/2305.16225)
[49](https://arxiv.org/pdf/2210.09292.pdf)
[50](https://arxiv.org/pdf/2312.06354.pdf)
[51](https://arxiv.org/html/2501.06655v1)
[52](https://jsycsjh.github.io/assets/publications/2023_diffusion/kdd_adversarial_training.pdf)
[53](https://arxiv.org/html/2411.16164v1)
[54](https://arxiv.org/abs/2306.02618)
[55](https://aclanthology.org/2025.acl-long.1201.pdf)
[56](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Customization_Assistant_for_Text-to-Image_Generation_CVPR_2024_paper.pdf)
[57](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09769.pdf)
[58](https://www.mi-research.net/article/doi/10.1007/s11633-025-1563-3)
[59](https://dl.designresearchsociety.org/cgi/viewcontent.cgi?article=1179&context=iasdr)
[60](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models)
[61](https://dl.acm.org/doi/10.1145/3580305.3599333)
[62](https://dl.acm.org/doi/10.1145/3711896.3736554)
[63](https://dl.acm.org/doi/10.1145/3680528.3687642)
