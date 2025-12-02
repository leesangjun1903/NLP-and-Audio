
# Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation

## **1. 핵심 주장 및 주요 기여 요약 (Executive Summary)**

본 논문은 AI 연구소 **Kandinsky Lab**이 2025년 11월 발표한 연구로, 고해상도 이미지와 비디오 합성을 위한 최첨단 파운데이션 모델 제품군인 **Kandinsky 5.0**을 소개합니다.

*   **핵심 주장:** 기존 비디오 생성 모델들이 겪는 "확장성(Scalability)"과 "계산 복잡도" 문제를 해결하기 위해, 새로운 어텐션 메커니즘(NABLA)과 통합 아키텍처(CrossDiT)를 제안합니다. 이를 통해 소비자 등급의 하드웨어에서도 고품질의 긴 비디오(10초, 1024px) 생성이 가능함을 입증합니다.
*   **주요 기여:**
    1.  **모델 라인업 다양화:** Image Lite (6B), Video Lite (2B), Video Pro (19B) 등 다양한 규모의 모델 공개.
    2.  **효율적인 아키텍처:** CrossDiT 및 NABLA 어텐션을 도입하여 학습 및 추론 속도를 2.7배 가속화.
    3.  **데이터 파이프라인 혁신:** 러시아 문화 코드(Russian Cultural Code) 데이터셋 및 전문가 기반의 SFT(Supervised Fine-Tuning) 파이프라인 구축.
    4.  **오픈소스 기여:** 코드와 가중치를 공개하여 폐쇄형 모델(Sora, Veo 등) 대비 연구 접근성 제고.

***

## **2. 상세 분석: 문제 정의부터 한계점까지**

### **2.1 해결하고자 하는 문제 (Problem Definition)**
기존의 비디오 생성 모델(Sora, Veo 등)은 시간축(Temporal axis) 확장에 따른 연산량이 **이차적(Quadratic)으로 증가**하는 문제에 직면해 있습니다. 특히 고해상도(1024px 이상)와 긴 시간(10초 이상)을 동시에 처리할 때, 메모리 병목과 느린 추론 속도가 발생하여 상용화 및 일반화에 제약이 있었습니다.

### **2.2 제안하는 방법 및 수식 (Methodology)**

논문은 **Flow Matching** 패러다임에 기반한 **Latent Diffusion** 방식을 채택하며, 핵심은 **NABLA (Neighborhood Adaptive Block-Level Attention)** 메커니즘입니다.

*   **NABLA Attention:** 기존의 Full Attention 대신, 어텐션 맵의 희소성(Sparsity)을 활용하여 연산 효율을 극대화합니다.
    *   **Block-wise Dimensionality Reduction:** Query($Q$)와 Key($K$)를 블록 단위로 평균 풀링(Average Pooling)하여 차원을 축소합니다.
    *   **Adaptive Sparsification:** 축소된 어텐션 맵의 누적 분포 함수(CDF)를 기반으로 동적 임계값($\tau$)을 적용하여 마스크($M$)를 생성합니다.
    
$$
    \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} \odot M_{\text{sparse}}\right) V
    $$

여기서 희소 마스크 $M_{\text{sparse}}$는 다음과 같은 조건부 함수로 정의될 수 있습니다.
    

$$
    M_{\text{sparse}}(i, j) = \begin{cases} 
    1 & \text{if } \text{CDF}(\mathcal{P}(Q_i K_j^T)) > \tau \\
    0 & \text{otherwise}
    \end{cases}
    $$

($\mathcal{P}$는 풀링 연산, $\tau$는 희소성 조절 임계값)

*   **데이터 처리:** 텍스트 인코더로 **Qwen2.5-VL**을 사용하여 시각-언어 이해도를 높였으며, 비디오 압축을 위해 **HunyuanVideo VAE** (3D Causal VAE)를 활용하여 시간적 일관성을 유지했습니다.

### **2.3 모델 구조 (Architecture)**
**CrossDiT (Cross-Attention Diffusion Transformer)** 아키텍처를 제안합니다.
*   **구조:** 기존 MMDiT(Multimodal DiT)와 달리, Self-Attention과 Cross-Attention 블록을 명시적으로 분리하고 순차적으로 배치하여 비디오 데이터의 가변 길이에 유연하게 대응합니다.
*   **입력 처리:** Text, Time, Visual Latent 외에 CLIP Text Embedding을 추가적인 컨디셔닝 벡터로 사용하여 텍스트 정합성을 보완합니다.

### **2.4 성능 향상 (Performance)**
*   **속도:** NABLA 적용 시, 기존 방식 대비 **학습 및 추론 속도 2.7배 향상**, 메모리 효율 90% 희소성 달성.
*   **품질 비교:**
    *   **vs. Sora (OpenAI):** Visual Quality와 Motion Dynamics에서 **약 60%**의 우위(Human Eval 기준)를 보임.
    *   **vs. Wan 2.1/2.2 (Alibaba):** 시각적 품질은 우세하나, 복잡한 프롬프트 이행 능력(Prompt Following)에서는 다소 열세.
    *   **vs. Veo 3 (Google):** 모션 역동성에서 앞서지만, 텍스트 지시 이행 능력은 낮음.

### **2.5 한계점 (Limitations)**
1.  **텍스트 정합성 (Prompt Alignment):** 텍스트 인코더(Qwen2.5-VL)의 컨텍스트 길이 제한(256 토큰)으로 인해 복잡하고 긴 프롬프트의 세부 사항을 놓치는 경향이 있음.
2.  **장기적 시간 일관성:** 10초 이상의 영상 생성 시, 물리적 상호작용(유체 역학 등)이 어색해지는 아티팩트 발생.
3.  **편향성:** 학습 데이터의 불균형으로 인해 특정 문화권이나 스테레오타입(예: 직업, 성별)에 편향된 결과물 생성.

***

## **3. 모델의 일반화 성능 향상 가능성 (Focus on Generalization)**

Kandinsky 5.0은 "파운데이션 모델(Foundation Model)"로서의 일반화 능력에 중점을 두었습니다.

1.  **멀티모달 통합 학습:** Image Lite와 Video 모델들이 동일한 CrossDiT 아키텍처와 VAE 잠재 공간을 공유하므로, 이미지 생성에서 학습한 고해상도 텍스처 표현 능력이 비디오 생성으로 **전이(Transfer)** 됩니다. 이는 "Image-to-Video" 작업에서 강력한 일반화 성능으로 나타납니다.
2.  **SFT 및 RLHF 도입:** 단순한 사전 학습(Pre-training)을 넘어, 지도 미세 조정(SFT)과 강화 학습(RLHF)을 통해 모델이 다양한 도메인(예: 실사, 애니메이션, 예술 작품)에 대해 사용자의 의도를 더 잘 파악하도록 튜닝되었습니다. 이는 훈련 데이터에 없는 낯선 스타일 요청에 대해서도 **Zero-shot**에 가까운 적응력을 보여줍니다.
3.  **문화적 코드 확장:** 'Russian Cultural Code' 데이터셋을 별도로 구축하여 학습함으로써, 서구권 데이터에 편중된 기존 모델들의 **문화적 일반화(Cultural Generalization)** 한계를 극복하고자 했습니다.

***

## **4. 향후 연구에 미치는 영향 및 고려사항**

### **영향 (Impact)**
*   **오픈소스 비디오 생성의 민주화:** 고성능 GPU 클러스터 없이도 실행 가능한 효율적인 Video Lite(2B) 모델의 공개는 학계와 개인 연구자들의 진입 장벽을 획기적으로 낮출 것입니다.
*   **아키텍처의 표준 변화:** 기존의 3D UNet이나 순수 DiT에서 벗어나, 효율성을 극대화한 **Sparse Attention(NABLA)** 기반의 Transformer가 비디오 생성의 표준이 될 가능성을 시사합니다.

### **연구 시 고려할 점 (Future Considerations)**
1.  **텍스트 인코더의 확장:** 더 긴 컨텍스트(예: 1024 토큰 이상)를 처리할 수 있는 LLM 기반 인코더(T5-XXL, LLama-3 등)와의 결합 실험이 필요합니다.
2.  **물리 법칙의 내재화:** 단순한 픽셀 변화가 아닌, 'World Model'로서 물리적 인과관계를 학습시키기 위한 데이터 큐레이션 및 손실 함수(Loss function) 설계가 요구됩니다.
3.  **윤리적 필터링:** 오픈소스 특성상 딥페이크 등 악용 가능성에 대비한 워터마킹(SynthID 등) 기술의 고도화가 병행되어야 합니다.

***

## **5. 2020년 이후 관련 최신 연구 탐색 (Related Work Landscape)**

Kandinsky 5.0의 위치를 파악하기 위해, 2020년 이후 등장한 주요 경쟁 모델들을 탐색했습니다.

| 모델명 | 개발사 | 발표 시기 | 주요 특징 및 아키텍처 | Kandinsky 5.0과의 관계 |
| :--- | :--- | :--- | :--- | :--- |
| **Sora** | OpenAI | 2024.02 | Space-Time Patch 기반 DiT. 최대 1분 길이 생성. 물리 시뮬레이션 강점. | 비디오 생성의 'World Simulator' 기준점. Kandinsky는 이를 추월하는 효율성 강조. |
| **HunyuanVideo** | Tencent | 2024.12 | 3D Causal VAE + Two-stream DiT. 이미지-비디오 통합 아키텍처. | VAE 구조 및 통합 아키텍처 철학이 유사함. 강력한 오픈소스 경쟁자. |
| **Veo 3** | Google | 2025.05 | 기본적으로 오디오 생성(Audio generation) 통합. 높은 프롬프트 이해도. | Kandinsky는 오디오 통합 기능이 부재하나, 모션 역동성에서 경쟁력 보유. |
| **Wan 2.1/2.2** | Alibaba | 2025.02 | **MoE (Mixture-of-Experts)** 아키텍처 도입. 14B 파라미터로 높은 텍스트 이행력. | 텍스트 지시 이행력에서 Kandinsky보다 우수. 아키텍처 접근법(MoE vs Dense)이 다름. |
| **Flux.1** | Black Forest | 2024.08 | Flow Matching 기반의 고화질 이미지 모델. | Kandinsky 5.0이 VAE 인코더로 Flux.1-dev를 활용함. |

**종합 평가:** Kandinsky 5.0은 2025년 말 현재, **"효율성"과 "접근성(오픈소스)"** 측면에서 가장 강력한 선택지 중 하나입니다. 특히 Veo나 Wan과 같은 거대 기업 모델들이 주도하는 시장에서, 독자적인 Attention 최적화 기술(NABLA)로 성능과 속도의 균형을 맞춘 점이 돋보입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9111f328-e85d-4b1a-8973-19ae4f940061/2511.14993v2.pdf)
[2](https://www.semanticscholar.org/paper/0adcc5a3b6d17990f22862612c6ca89907871986)
[3](https://arxiv.org/pdf/2403.05131.pdf)
[4](http://arxiv.org/pdf/2411.02385.pdf)
[5](https://arxiv.org/pdf/1906.00657.pdf)
[6](http://arxiv.org/pdf/2405.03520.pdf)
[7](http://arxiv.org/pdf/2403.13248.pdf)
[8](https://arxiv.org/abs/2312.03511)
[9](https://arxiv.org/pdf/2409.14993.pdf)
[10](https://arxiv.org/pdf/2402.17177v1.pdf)
[11](https://www.alphaxiv.org/overview/2511.14993v1)
[12](https://www.artany.ai/models/wan-ai)
[13](https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video)
[14](https://www-veo.com)
[15](https://yenchenlin.github.io/blog/2025/01/08/video-generation-models-explosion-2024/)
[16](https://www.emergentmind.com/topics/kandinsky-5-0)
[17](https://www.litmedia.ai/ai-video-model/wan-ai/)
[18](https://www.siliconflow.com/blog/unlocking-the-future-of-video-technology-introducing-hunyuan-video-by-tencent)
[19](https://fal.ai/models/fal-ai/veo3)
[20](https://www.siliconflow.com/articles/en/best-open-source-text-to-video-models)
[21](https://www.emergentmind.com/topics/kandinsky-5-0-video-lite)
[22](https://docs.comfy.org/tutorials/video/wan/wan2_2)
[23](https://huggingface.co/tencent/HunyuanVideo-1.5)
[24](https://deepmind.google/models/veo/)
[25](https://en.wikipedia.org/wiki/Text-to-video_model)
[26](https://www.themoonlight.io/en/review/kandinsky-50-a-family-of-foundation-models-for-image-and-video-generation)
[27](https://wan22.ai)
[28](https://www.nextdiffusion.ai/blogs/comparing-wan21-and-hunyuanvideo-architecture-efficiency-and-quality)
[29](https://tribeacademy.sg/blog/google-veo-3-ai-video-generator/)
[30](https://neptune.ai/state-of-foundation-model-training-report)
[31](https://www.themoonlight.io/en/review/hunyuanvideo-a-systematic-framework-for-large-video-generative-models)
