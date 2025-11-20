# VPD : Unleashing Text-to-Image Diffusion Models for Visual Perception | Semantic segmentation, Depth estimation

## 1. 핵심 주장과 주요 기여

VPD(Visual Perception with a pre-trained Diffusion model)는 **대규모 텍스트-이미지 디퓨전 모델의 학습된 지식을 시각적 인식 작업에 활용하는 새로운 프레임워크**를 제시합니다. 논문의 핵심 아이디어는 생성 모델로 설계된 Stable Diffusion의 사전 학습된 지식이 인식 작업에도 효과적으로 전이될 수 있다는 것입니다.[1][2]

**주요 기여점:**
- 텍스트-이미지 디퓨전 모델을 백본으로 사용하여 시각적 인식 작업을 수행하는 새로운 패러다임 제시
- 텍스트 프롬프트와 크로스 어텐션 맵을 활용한 암시적 및 명시적 의미적 가이던스 방법론 개발
- 의미적 분할(semantic segmentation), 참조 이미지 분할(referring image segmentation), 깊이 추정(depth estimation)에서 새로운 최고 성능 달성
- NYUv2 깊이 추정에서 0.254 RMSE, RefCOCO 참조 이미지 분할에서 73.3% oIoU 기록[2][1]

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의

기존 시각적 사전 학습 방법들(supervised pre-training, self-supervised learning)과 달리, 텍스트-이미지 디퓨전 모델은 생성 작업을 위해 설계되어 인식 작업으로의 전이가 어려웠습니다. 특히 두 가지 주요 과제가 존재했습니다:

1. **디퓨전 파이프라인과 시각적 인식 작업 간의 비호환성**
2. **UNet 기반 디퓨전 모델과 기존 시각적 백본 간의 구조적 차이**

### 제안 방법

VPD는 일반적인 시각적 인식 작업을 다음과 같이 분해하여 해결합니다:

$$p_\phi(y|x, S) = p_{\phi_3}(y|F)p_{\phi_2}(F|x, C)p_{\phi_1}(C|S)$$

여기서:
- $$y$$: 작업별 라벨
- $$x$$: 입력 이미지  
- $$S$$: 카테고리 이름 집합
- $$F$$: 특성 맵 집합
- $$C$$: 텍스트 특성

**세 가지 핵심 구성 요소:**

1. **텍스트 프롬프팅** $$p_{\phi_1}(C|S)$$: 
   - "a photo of a [CLS]" 템플릿 사용
   - 텍스트 어댑터를 통한 도메인 갭 완화: $$C \leftarrow C + \gamma \text{MLP}(C)$$

2. **특성 추출** $$p_{\phi_2}(F|x, C)$$:
   - VQGAN 인코더로 이미지를 잠재 공간으로 변환: $$z_0 = E(x)$$
   - 노이즈 없이 단일 디노이징 단계 수행 (t=0)
   - 계층적 특성 맵 추출

3. **예측 헤드** $$p_{\phi_3}(y|F)$$:
   - Semantic FPN을 활용한 경량 디코더

### 모델 구조

VPD의 전체 아키텍처는 다음과 같이 구성됩니다:

1. **사전 학습된 이미지 인코더** (VQGAN 인코더)
2. **사전 학습된 텍스트 인코더** (CLIP 텍스트 인코더)
3. **텍스트 어댑터** (2층 MLP)
4. **디노이징 UNet** (Stable Diffusion에서 차용)
5. **작업별 디코더** (Semantic FPN 기반)

**크로스 어텐션 메커니즘:**
디퓨전 모델의 크로스 어텐션 맵을 명시적 의미적 가이던스로 활용하여, 각 해상도에서 평균화된 어텐션 맵 $$A_i \in \mathbb{R}^{|S| \times H_i \times W_i}$$를 특성 맵과 연결합니다.

## 3. 성능 향상과 일반화 가능성

### 성능 향상

**의미적 분할 (ADE20K):**
- 80K 반복 훈련으로 54.6% mIoU 달성
- ConvNeXt-XL 대비 우수한 성능
- 8K 반복만으로도 기존 방법들 능가

**참조 이미지 분할:**
- RefCOCO: 73.25% oIoU
- RefCOCO+: 62.69% oIoU  
- G-Ref: 61.96% oIoU

**깊이 추정 (NYUv2):**
- 0.254 RMSE로 새로운 최고 기록
- 1 epoch 훈련으로도 SwinV2-L보다 빠른 수렴

### 일반화 성능 향상 가능성

1. **스케일링 특성**: 더 오랜 시간 사전 학습된 디퓨전 모델일수록 다운스트림 작업에서 더 좋은 성능을 보임[1]

2. **빠른 적응성**: 
   - 4K 반복에서 44.7% mIoU 달성 (기존 방법 대비 우수)
   - 1 epoch 훈련으로도 경쟁력 있는 성능

3. **도메인 간 일반화**: 후속 연구인 TADP에서 크로스 도메인 성능 향상 확인[3]

4. **멀티태스크 학습 잠재력**: EVP와 같은 후속 연구에서 다양한 작업에 대한 일반화 가능성 입증[4]

## 4. 한계점

논문에서 명시한 주요 한계점은 **높은 계산 비용**입니다. 생성 모델 특성상 합성 품질에 중점을 둔 설계로 인해 효율성이 부족하며, 인식 작업에 최적화되지 않은 구조적 한계가 존재합니다.[1]

추가적으로 다음과 같은 한계점들이 관찰됩니다:
- 낮은 해상도 크로스 어텐션 맵의 부정확성
- 텍스트-이미지 정렬 불일치 문제 (후속 연구에서 해결)
- 단일 디노이징 단계의 제한적 활용

## 5. 향후 연구에 미치는 영향과 고려사항

### 연구 영향

1. **새로운 연구 방향 개척**: 
   - EVP, TADP, IEDP 등 다수의 후속 연구 파생[5][4][3]
   - 디퓨전 기반 인식 모델의 새로운 패러다임 확립

2. **생성-인식 통합 모델의 발전**:
   - 생성과 인식을 통합하는 연구 방향 제시
   - Foundation model로서의 디퓨전 모델 활용 가능성 확인

3. **Few-shot Learning 응용**: DiffewS 등에서 few-shot semantic segmentation에 성공적 적용[6]

### 향후 연구 고려사항

1. **효율성 개선**:
   - 경량화된 디퓨전 아키텍처 개발 필요
   - 생성과 인식을 동시에 고려한 효율적 설계 요구

2. **텍스트-이미지 정렬 향상**:
   - 자동 캡션 생성을 통한 정렬 개선 (TADP 방식)
   - 암시적 언어 가이던스 활용 (IEDP 방식)

3. **멀티모달 통합**:
   - 다양한 모달리티 통합을 위한 확장 가능한 프레임워크 개발
   - Cross-domain 일반화 성능 향상 방안

4. **실용적 응용 확대**:
   - 실시간 추론이 가능한 경량 모델 개발
   - 산업 응용을 위한 최적화 연구

VPD는 생성 모델의 강력한 표현 학습 능력을 인식 작업에 성공적으로 전이시킨 선구적 연구로, AI 분야에서 생성과 인식의 경계를 허무는 중요한 이정표를 제시했습니다.

[1] https://ieeexplore.ieee.org/document/10377753/
[2] https://arxiv.org/abs/2303.02153
[3] https://ieeexplore.ieee.org/document/10656119/
[4] https://arxiv.org/abs/2312.08548
[5] https://ieeexplore.ieee.org/document/10814050/
[6] https://proceedings.neurips.cc/paper_files/paper/2024/file/4b2a917e30e1bb1aff055b4d8c6c081c-Paper-Conference.pdf
[7] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2fb83c40-37cb-4c66-a618-37bde27dd7b1/2303.02153v1.pdf
[8] https://ieeexplore.ieee.org/document/10658348/
[9] https://arxiv.org/abs/2308.07428
[10] https://ieeexplore.ieee.org/document/10633628/
[11] https://arxiv.org/abs/2312.14733
[12] https://arxiv.org/abs/2506.02605
[13] https://arxiv.org/abs/2411.13842
[14] https://arxiv.org/pdf/2303.02153.pdf
[15] https://arxiv.org/pdf/2312.14733.pdf
[16] https://arxiv.org/html/2501.03495v2
[17] https://arxiv.org/html/2502.17157v1
[18] https://arxiv.org/pdf/2309.01141.pdf
[19] https://arxiv.org/abs/2211.08332
[20] https://arxiv.org/html/2402.16627v3
[21] http://arxiv.org/pdf/2403.14526.pdf
[22] https://arxiv.org/pdf/2501.00917.pdf
[23] http://arxiv.org/pdf/2404.07600.pdf
[24] https://openaccess.thecvf.com/content/CVPR2025/papers/Ravishankar_Scaling_Properties_of_Diffusion_Models_For_Perceptual_Tasks_CVPR_2025_paper.pdf
[25] https://arxiv.org/pdf/2103.00020.pdf
[26] https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Unleashing_Text-to-Image_Diffusion_Models_for_Visual_Perception_ICCV_2023_paper.pdf
[27] https://lunaleee.github.io/posts/blip/
[28] https://arxiv.org/html/2404.07600v1
[29] https://huggingface.co/papers/2502.17157
[30] https://arxiv.org/html/2410.10879v2
[31] https://openaccess.thecvf.com/content/CVPR2024/papers/Kondapaneni_Text-Image_Alignment_for_Diffusion-Based_Perception_CVPR_2024_paper.pdf
[32] https://www.sciencedirect.com/science/article/abs/pii/S0950705125010317
[33] https://github.com/wl-zhao/VPD
[34] https://openreview.net/forum?id=BgYbk6ZmeX
[35] https://openai.com/index/clip/
[36] https://www.ijcai.org/proceedings/2024/0082.pdf
[37] https://velog.io/@pabiya/BLIP-Bootstrapping-Language-Image-Pre-training-forUnified-Vision-Language-Understanding-and-Generation
[38] https://www.youtube.com/watch?v=4lJ58sMbnDw
[39] https://pmc.ncbi.nlm.nih.gov/articles/PMC7614424/
[40] https://github.com/melvinsevi/MVA-Project-Unleashing-Text-to-Image-Diffusion-Models-for-Visual-Perception
