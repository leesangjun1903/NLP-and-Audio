# InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

### **1. 핵심 주장 및 주요 기여 요약 (Executive Summary)**

**InternVL3**는 기존의 "텍스트 전용 LLM + 시각 어댑터"라는 사후 결합(post-hoc) 방식을 탈피하고, **'네이티브 멀티모달 사전학습(Native Multimodal Pre-training)'** 패러다임을 제시한 최신 연구입니다. 이 논문의 핵심은 언어와 시각 능력을 별도 단계가 아닌 단일 사전학습 단계에서 공동으로 최적화함으로써, 모델의 정렬(alignment) 효율성과 성능을 극대화했다는 점입니다.

**주요 기여:**
*   **네이티브 사전학습 도입:** 텍스트와 멀티모달 데이터를 처음부터 함께 학습시켜 별도의 연결 모듈 없이도 깊이 있는 모달리티 통합을 달성했습니다.
*   **V2PE (Variable Visual Position Encoding):** 시각적 토큰의 위치 인코딩을 가변적으로 설계하여 긴 문맥(Long Context) 처리 능력을 향상시켰습니다.
*   **고도화된 훈련 레시피:** SFT(지도 미세조정)와 MPO(Mixed Preference Optimization)를 결합하여 추론 및 응답 품질을 개선했습니다.
*   **SOTA 달성:** MMMU 벤치마크에서 **72.2점**을 기록하며 오픈소스 모델 중 최고 성능을 달성했고, GPT-4o나 Gemini 1.5 Pro와 같은 독점(Proprietary) 모델과 대등한 성능을 입증했습니다.

***

### **2. 논문 상세 분석: 문제 정의부터 해결책까지**

#### **2.1 해결하고자 하는 문제 (Problem Statement)**
대다수의 기존 MLLM(Multimodal Large Language Model)은 텍스트 전용으로 학습된 LLM에 시각 인코더(Vision Encoder)를 덧붙이는 **'Post-hoc' 방식**을 사용했습니다.
*   **한계:** 이 방식은 언어 능력과 시각 능력 간의 정렬(Alignment)을 위해 복잡한 다단계 미세조정이 필요하며, 두 모달리티가 유기적으로 통합되지 않아 정보 손실이나 환각(Hallucination) 문제가 발생하기 쉽습니다.

#### **2.2 제안하는 방법 (Proposed Method)**

**1) 네이티브 멀티모달 사전학습 (Native Multimodal Pre-training)**
InternVL3는 텍스트 코퍼스와 멀티모달 데이터(이미지-텍스트, 비디오-텍스트)를 혼합하여 단일 단계에서 학습합니다.
*   **손실 함수 (Objective Function):**
    모델 파라미터 $\theta$에 대해, 전체 토큰 시퀀스 $x = \{x_1, ..., x_L\}$ 중 텍스트 토큰($x_i \in \text{Text}$)에 대해서만 다음 토큰 예측(Next Token Prediction)을 수행합니다. 시각 토큰은 예측 대상이 아니지만, 텍스트 생성의 조건부 컨텍스트(Context)로 기능하며 그래디언트가 전파됩니다.

$$ \mathcal{L}_{\text{text-only}}(\theta) = -\sum_{i=2, x_i \in \text{Text}}^{L} w_i \cdot \log p_\theta(x_i | x_1, ..., x_{i-1}) $$

*   여기서 $w_i$는 데이터 불균형을 해소하기 위한 가중치(Square Averaging 등)입니다.

**2) 가변 시각 위치 인코딩 (V2PE: Variable Visual Position Encoding)**
긴 시각적 입력(고해상도 이미지, 비디오 등)을 처리할 때 컨텍스트 윈도우가 급격히 소진되는 것을 막기 위해, 시각 토큰의 위치 인덱스를 1보다 작은 값($\delta$)만큼만 증가시킵니다.

$$ p_i = p_{i-1} + \begin{cases} 1, & \text{if } x_i \text{ is a textual token} \\ \delta, & \text{if } x_i \text{ is a visual token} \end{cases} $$

*   $\delta < 1$ (예: 1/2, 1/4 등)로 설정하여 시각 정보가 차지하는 논리적 위치 범위를 압축합니다.

**3) 고급 사후 학습 (Advanced Post-Training)**
*   **SFT (Supervised Fine-Tuning):** 다양한 고품질 멀티모달 데이터로 미세조정.
*   **MPO (Mixed Preference Optimization):** 모델의 추론 능력을 강화하기 위해 선호도 최적화를 수행합니다. DPO(Direct Preference Optimization) 손실과 품질(Quality) 손실 등을 결합하여 학습합니다.

$$ \mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g $$

#### **2.3 모델 구조 (Model Architecture)**
InternVL3는 **ViT-MLP-LLM**의 전형적인 구조를 따르되, 각 구성 요소를 최신화했습니다.
*   **Vision Encoder:** InternViT-300M 또는 InternViT-6B (최대 448x448 해상도 타일 처리).
*   **LLM Backbone:** Qwen2.5 시리즈 및 InternLM3를 기반으로 초기화하되, 네이티브 학습을 통해 파라미터를 전면 재조정합니다.
*   **Connector:** 2-layer MLP를 사용하여 시각 특징을 언어 임베딩 공간으로 투영합니다.

#### **2.4 성능 향상 및 한계**
*   **성능:** MMMU(72.2), MathVista(79.6) 등 주요 벤치마크에서 오픈소스 SOTA를 갱신했습니다. 특히 OCR(광학 문자 인식)과 차트 이해 능력에서 GPT-4o를 능가하거나 대등한 결과를 보였습니다.
*   **한계:** 여전히 **환각(Hallucination)** 문제가 완전히 해결되지 않았으며(HallusionBench 등에서 일부 경쟁 모델 대비 열세), 극도로 복잡한 3D 공간 추론이나 뉘앙스가 깊은 비디오 이해에서는 개선의 여지가 있습니다.

***

### **3. 모델의 일반화 성능 향상 가능성 (Generalization Capabilities)**

InternVL3는 단순한 벤치마크 점수 향상을 넘어, 실세계 문제에 대한 일반화 능력을 크게 강화했습니다.

1.  **데이터 다양성 확보:** 학습 데이터에 GUI(그래픽 유저 인터페이스), 도구 사용(Tool Use), 3D 장면 이해, 비디오 데이터 등을 대거 포함하여, 모델이 본 적 없는 **"Out-of-Distribution"** 태스크(예: 새로운 앱의 UI 조작)에서도 강건하게 작동하도록 했습니다.
2.  **Test-Time Scaling (추론 시 확장):**
    *   추론 단계에서 **VisualPRM(Process Reward Model)**을 활용한 **Best-of-N** 전략을 도입했습니다.
    *   이는 모델이 하나의 답만 내놓는 것이 아니라 여러 경로를 탐색하고, 비평가(Critic) 모델이 가장 논리적인 답을 선택하게 함으로써 복잡한 수학/추론 문제에서의 일반화 성능을 비약적으로 높였습니다.
3.  **V2PE를 통한 문맥 일반화:** 위치 인코딩을 유연하게 처리함으로써, 학습 때 보지 못한 긴 비디오나 다량의 문서 이미지 입력이 들어와도 성능 저하 없이 처리할 수 있는 기반을 마련했습니다.

***

### **4. 향후 연구에 미치는 영향 및 고려할 점**

#### **영향 (Impact)**
*   **네이티브 학습의 표준화:** InternVL3는 텍스트 모델을 사후 개조하는 방식보다, 처음부터 멀티모달로 학습하는 것이 효율적임을 증명했습니다. 이는 향후 LMM 연구가 "LLM + Vision Adapter" 형태에서 **"Unified Foundation Model"** 형태로 이동하는 기폭제가 될 것입니다.
*   **오픈소스 생태계 기여:** 고성능 모델 가중치와 훈련 데이터를 공개함으로써, 학계와 산업계가 GPT-4 급의 멀티모달 모델을 기반으로 응용 연구(로보틱스, 에이전트 등)를 수행할 수 있게 되었습니다.

#### **연구 시 고려할 점**
*   **컴퓨팅 자원 효율성:** 네이티브 사전학습은 막대한 컴퓨팅 자원(수천 개의 GPU)을 요구합니다. 대학이나 소규모 연구소에서는 이를 효율적으로 재현하거나 경량화할 수 있는 **Efficient Training** 기법(예: LoRA의 멀티모달 적용 등) 연구가 필요합니다.
*   **환각 억제:** MPO를 통해 개선되었으나 여전히 시각적 사실과 다르게 텍스트를 생성하는 환각 현상은 치명적입니다. RAG(검색 증강 생성)나 사실 검증(Fact-checking) 모듈과의 결합 연구가 중요합니다.

***

### **5. 2020년 이후 관련 최신 연구 탐색 (Landscape since 2020)**

InternVL3의 위치를 파악하기 위해 2020년부터 현재(2025년 시점)까지의 LMM 발전 흐름을 요약합니다.

| 시기 | 주요 트렌드 | 대표 모델 및 연구 | 비고 |
| :--- | :--- | :--- | :--- |
| **2020-2021** | **태동기 (Foundation)** | **CLIP (OpenAI):** 이미지-텍스트 대조 학습으로 멀티모달의 기초 마련.<br>**DALL-E:** 텍스트 $\to$ 이미지 생성 가능성 증명. | 시각과 언어의 임베딩 공간 정렬(Alignment)에 집중. |
| **2022-2023** | **확장기 (Explosion)** | **Flamingo (DeepMind):** 시각적 컨텍스트 학습(In-context Learning) 제시.<br>**LLaVA (2023):** "Visual Instruction Tuning" 개념 도입, 오픈소스 붐 주도.<br>**GPT-4V (OpenAI):** 상용 수준의 고성능 LMM 등장. | LLM에 Vision Encoder를 붙이는 **Post-hoc** 방식이 주류를 이룸. |
| **2024** | **성숙기 (Refinement)** | **Gemini 1.5 Pro (Google):** 1M+ 토큰의 긴 문맥(Long Context)과 비디오 이해 강조.<br>**GPT-4o (OpenAI):** 텍스트/오디오/비전의 실시간 통합 (Omni).<br>**InternVL 1.5/2.0:** 오픈소스 진영의 성능이 상용 모델에 근접하기 시작. | 성능 경쟁 심화, 모달리티 간 경계가 허물어짐. |
| **2025 (현재)** | **통합기 (Unification)** | **InternVL3:** **Native Multimodal Pre-training**으로 회귀 및 통합.<br>**DeepSeek-VL / Qwen-VL Next:** 추론(Reasoning) 능력 강화 및 에이전트 활용성 증대. | 단순 인식을 넘어 **추론(Reasoning), 행동(Action), 3D 공간 이해**로 확장. |

**요약:** 2020년의 연구가 "이미지와 텍스트를 연결할 수 있는가?"에 집중했다면, 2023년은 "LLM의 지능을 시각에 빌려주자(Instruction Tuning)"는 흐름이었습니다. 2025년 InternVL3가 제시하는 흐름은 **"처음부터 하나의 지능으로 가르치자(Native Pre-training)"**는 것으로, 이는 AGI(일반 인공지능)로 가는 중요한 교두보가 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/92bd34b5-8efb-4cc1-b4c1-c540da2c3bb0/2504.10479v3.pdf)
[2](https://www.nap.edu/catalog/25761)
[3](https://link.springer.com/10.1007/s10546-020-00560-7)
[4](https://www.semanticscholar.org/paper/8b2fb8135d815f5bb911012959172b0bbb934a18)
[5](https://www.mdpi.com/2075-4450/13/7/618)
[6](https://www.semanticscholar.org/paper/5d92f901f736fd048d60d16541397bf9966e81e3)
[7](https://www.semanticscholar.org/paper/9ac5f0c54b7ff19a761c69062ee08ac1af185bb6)
[8](https://www.semanticscholar.org/paper/2d35c3ac3600d9de32b165519c5dda2523dd9786)
[9](https://www.semanticscholar.org/paper/2d7dd4e01fd4818d73455cefaa39faff72bf04bd)
[10](https://www.semanticscholar.org/paper/e8af2421dcf265e41eef0b0e582f14b67efcab0c)
[11](https://arxiv.org/pdf/2410.05608.pdf)
[12](http://arxiv.org/pdf/2409.18991.pdf)
[13](http://arxiv.org/pdf/2502.15336.pdf)
[14](https://arxiv.org/pdf/2407.00118.pdf)
[15](https://arxiv.org/pdf/2412.02104.pdf)
[16](https://arxiv.org/html/2405.19334)
[17](http://arxiv.org/pdf/2401.08092.pdf)
[18](https://arxiv.org/pdf/2105.11087.pdf)
[19](https://aclanthology.org/2024.findings-acl.807.pdf)
[20](https://cathcarttechnology.co.th/insights/the-rise-of-multimodal-ai-how-its-transforming-industries-in-2025/)
[21](https://www.aitimelines.club/timeline)
[22](https://arxiv.org/html/2405.17927v1)
[23](https://www.congruencemarketinsights.com/report/multimodal-ai-market)
[24](https://www.youtube.com/watch?v=-l-vIGscWsA)
[25](https://dev.to/aniruddhaadak/the-evolution-of-ai-models-since-2020-a-beginners-guide-49h7)
[26](https://www.coherentmarketinsights.com/industry-reports/multimodal-ai-market)
[27](https://lifearchitect.ai/timeline/)
[28](https://www.nature.com/articles/s41539-025-00301-w)
[29](https://futureagi.com/blogs/multimodal-ai-2025)
[30](https://www.sciencedirect.com/science/article/abs/pii/S092523122401909X)
