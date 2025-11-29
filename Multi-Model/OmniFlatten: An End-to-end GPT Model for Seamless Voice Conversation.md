# OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation

### **1. 핵심 주장 및 주요 기여 요약 (Executive Summary)**

**OmniFlatten**은 기존의 턴제(Turn-based) 대화 시스템의 한계를 넘어, 인간과 유사한 실시간 **전이중(Full-duplex)** 음성 대화를 구현하기 위해 제안된 **End-to-End GPT 기반 모델**입니다.

*   **핵심 주장:** 복잡한 아키텍처 수정 없이, 음성과 텍스트 토큰을 하나의 시퀀스로 **'평탄화(Flattening)'** 하여 학습함으로써, LLM(Large Language Model)이 본래 가진 능력을 유지하면서도 낮은 지연 시간(Low Latency)으로 끼어들기(Interruption)와 동시 발화(Overlapping speech)를 처리할 수 있다는 것입니다.
*   **주요 기여:**
    1.  **Flattening 기법 도입:** 텍스트와 음성 스트림을 미세한 청크(Chunk) 단위로 쪼개어 하나의 선형 시퀀스로 변환, 표준 GPT 아키텍처에서 텍스트와 음성을 통합 처리하는 단순하고 효율적인 방법론 제시.
    2.  **다단계 사후 학습(Multi-stage Post-training):** 모달리티 정렬(Alignment) → 반이중(Half-duplex) 학습 → 전이중(Full-duplex) 학습으로 이어지는 단계적 학습 파이프라인 구축.
    3.  **데이터 합성 파이프라인:** 고비용의 전이중 대화 데이터를 수집하는 대신, 오픈소스 텍스트 데이터를 활용해 정교한 전이중 음성 대화 데이터를 생성하는 시뮬레이션 방법론 제안.

***

### **2. 상세 분석: 문제 정의, 제안 방법, 모델 구조 및 성능**

#### **2.1 해결하고자 하는 문제 (Problem Statement)**
기존의 음성 대화 시스템은 주로 **반이중(Half-duplex)** 방식으로, 사용자가 말을 끝낼 때까지 기다려야만 시스템이 응답할 수 있었습니다. 이는 인간의 자연스러운 대화 특징인 **끼어들기(Barge-in)**, **맞장구(Backchannel)**, **동시 발화**를 구현하지 못하며, 응답 지연 시간이 길어지는 한계가 있습니다. 반면, 기존의 전이중 시스템(예: Moshi)은 복잡한 병렬 스트림 처리를 위해 아키텍처를 수정해야 하거나 구현 난이도가 높았습니다.

#### **2.2 제안하는 방법 및 수식 (Proposed Method)**

OmniFlatten은 **'Flattening(평탄화)'** 연산을 통해 다중 모달리티(텍스트, 음성)와 다중 화자(사용자, 시스템)의 데이터를 단일 시퀀스로 변환합니다.

**[수식적 표현]**
시간 $t$에서의 대화 상태를 나타내기 위해, 사용자의 음성 토큰 $S^u$, 시스템의 텍스트 토큰 $T^a$, 시스템의 음성 토큰 $S^a$를 정의합니다. 기존 방식이 이들을 별도의 채널로 처리했다면, OmniFlatten은 이를 시간 순서에 따라 인터리빙(interleaving)하여 하나의 시퀀스 $X$로 구성합니다.

특정 시간 청크 $k$에서의 입력 시퀀스 $X_k$는 다음과 같이 표현될 수 있습니다:

$$
X_k = [S^u_{k,1}, ..., S^u_{k,N}, T^a_{k,1}, ..., T^a_{k,M}, S^a_{k,1}, ..., S^a_{k,N}]
$$

여기서 $S^u_{k}$는 $k$번째 청크의 사용자 음성 입력, $T^a_{k}$와 $S^a_{k}$는 해당 입력에 대해 모델이 예측해야 할 시스템의 텍스트 생각(Thought)과 음성 반응입니다.
모델은 자기회귀(Autoregressive) 방식으로 다음 토큰을 예측합니다:

$$
P(X) = \prod_{i=1}^{L} P(x_i | x_{ < i})
$$

이때 손실 함수 $\mathcal{L}$은 텍스트와 음성 토큰 예측에 대한 Cross-Entropy Loss로 정의됩니다.

**학습 파이프라인 (3단계):**
1.  **Modality Alignment:** ASR(음성인식) 및 TTS(음성합성) 태스크를 통해 텍스트 LLM이 음성 토큰을 이해하고 생성하도록 튜닝.
2.  **Half-duplex Learning:** 턴제 대화 데이터로 기본적인 대화 흐름 학습.
3.  **Full-duplex Learning:**
    *   **3-Stream Training:** User 음성 $\rightarrow$ System 텍스트 $\rightarrow$ System 음성 순으로 학습 (System의 텍스트적 사고 과정 포함).
    *   **2-Stream Training:** 지연 시간을 극도로 줄이기 위해 중간 Text 생성을 생략하고 음성-to-음성(User 음성 $\rightarrow$ System 음성)으로 직접 매핑하는 단계 추가.

#### **2.3 모델 구조 (Model Architecture)**
*   **Backbone:** **Qwen2-0.5B** (경량화된 LLM 사용). 구조적 수정 없이 표준 Transformer Decoder 아키텍처를 그대로 사용합니다.
*   **Tokenizer:** **CosyVoice**의 음성 토크나이저와 Flow Matching 기반 De-tokenizer를 사용하여 고품질 음성 생성 및 의미적(Semantic) 정보 보존을 동시에 달성했습니다.

#### **2.4 성능 향상 및 한계 (Performance & Limitations)**
*   **성능 향상:**
    *   **지연 시간(Latency):** Moshi 등 경쟁 모델 대비 빠른 응답 속도(Assistant Turn-taking)와 사용자 끼어들기 반응 속도를 기록했습니다.
    *   **대화 품질:** 텍스트 LLM의 지식을 효과적으로 전이하여, 단순 음성 모델보다 문맥 이해도가 높습니다.
*   **한계:**
    *   **모델 크기:** 0.5B의 작은 모델을 사용하여 복잡한 지식 추론에는 한계가 있을 수 있습니다.
    *   **User Turn-taking:** 사용자가 끼어들 때 발화를 멈추는 기능은 향상되었으나, 여전히 100% 완벽하지 않으며(약 50~70% 정확도), 응답 속도와 정확도 간의 트레이드오프가 존재합니다.
    *   **Backchannel 부재:** "음", "아하" 같은 자연스러운 맞장구 반응 생성은 아직 미흡합니다.

***

### **3. 모델의 일반화 성능 향상 가능성 (Generalization Capabilities)**

이 논문에서 가장 주목해야 할 점은 **'단순화(Simplification)'를 통한 일반화 가능성**입니다.

1.  **Backbone Agnostic (백본 독립성):** Flattening 기법은 특정 LLM 구조에 의존하지 않습니다. 즉, Qwen뿐만 아니라 Llama 3, GPT-4 등 더 크고 강력한 모델에도 동일한 방법론을 즉시 적용할 수 있습니다. 이는 모델의 일반화 성능(Generalization Performance)을 비약적으로 확장할 수 있는 잠재력을 가집니다.
2.  **Modality Unification (모달리티 통합):** 데이터를 단일 시퀀스로 평탄화함으로써, 텍스트, 음성뿐만 아니라 향후 **이미지, 비디오** 등 다른 모달리티가 추가되더라도 동일한 학습 프레임워크를 유지할 수 있습니다. ("Omni" 모델로의 확장성)
3.  **Data Synthesis Pipeline:** 논문에서 제안한 데이터 합성 파이프라인은 특정 도메인에 국한되지 않고 다양한 시나리오(감정 표현, 다화자 대화 등)로 데이터를 무한히 확장할 수 있게 해 주어, 데이터 부족으로 인한 일반화 저하 문제를 해결할 핵심 열쇠가 됩니다.

***

### **4. 향후 연구에 미치는 영향 및 고려할 점 (Impact & Future Directions)**

최신 연구(2024년 말~2025년 초) 흐름을 바탕으로 본 논문의 영향과 향후 연구 방향을 제안합니다.

#### **영향 (Impact)**
*   **Omni-model의 표준화:** 복잡한 병렬 아키텍처 대신, **Flattening** 방식이 전이중 음성 모델(Duplex Model)의 표준 베이스라인(Baseline)으로 자리 잡을 가능성이 큽니다. 최근 발표된 **FlexDuo (2025)**, **SOVA-Bench** 등의 연구에서도 OmniFlatten은 시간 청킹(Time Chunking) 기반 모델의 대표적인 사례로 인용되며 비교되고 있습니다.
*   **실시간 대화의 민주화:** 거대한 컴퓨팅 자원 없이 0.5B 수준의 경량 모델로도 전이중 대화가 가능함을 입증하여, 온디바이스(On-device) AI 에이전트 연구를 가속화할 것입니다.

#### **향후 연구 시 고려할 점 (Based on Recent Research)**
1.  **제어 모듈의 분리 (Decoupling Control):** 최신 연구인 **FlexDuo** (Liao et al., 2025)는 OmniFlatten과 같은 통합 모델이 '상태 제어(언제 듣고 언제 말할지)'와 '대화 생성'을 동시에 수행할 때 발생하는 혼선을 지적합니다. 향후 연구에서는 OmniFlatten의 구조에 **명시적인 턴 제어(Turn-taking Control) 토큰**이나 별도의 경량 제어 모듈을 결합하여 안정성을 높이는 방향이 고려되어야 합니다.
2.  **Non-verbal Communication 강화:** 단순히 말하고 듣는 것을 넘어, **비언어적 신호(웃음, 숨소리, 억양 변화)** 를 명시적으로 모델링하는 연구가 필요합니다. 이는 모델이 인간의 감정 상태를 더 잘 파악하고 공감하는 데 필수적입니다.
3.  **Streaming 구조 최적화:** 현재의 Chunk 기반 처리는 미세한 지연을 발생시킵니다. 이를 **토큰 단위 스트리밍(Token-level Streaming)** 으로 더욱 세분화하거나, 문맥 정보를 잃지 않으면서 청크 사이즈를 동적으로 조절하는 적응형(Adaptive) 알고리즘 연구가 유망합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9689dee5-59f7-4a5b-b1b2-6c69e0e86a85/2410.17799v2.pdf)
[2](https://link.springer.com/10.1007/s12351-023-00790-w)
[3](https://link.springer.com/10.1007/s41324-023-00551-z)
[4](https://ijebmr.com/uploads/pdf/archivepdf/2024/IJEBMR_1391.pdf)
[5](https://jrenewables.springeropen.com/articles/10.1186/s40807-024-00119-x)
[6](http://www.emerald.com/ijebr/article/29/4/816-837/117413)
[7](https://link.springer.com/10.1007/978-3-319-97385-2_14)
[8](https://link.springer.com/10.1007/s10479-020-03912-1)
[9](https://journals.sagepub.com/doi/10.1177/15330338221132927)
[10](https://www.mdpi.com/0718-1876/18/4/109)
[11](https://journals.sagepub.com/doi/10.1177/13621688241229534)
[12](http://arxiv.org/pdf/2410.17799.pdf)
[13](http://arxiv.org/pdf/2105.04448.pdf)
[14](http://link.aps.org/pdf/10.1103/PhysRevD.104.076027)
[15](http://arxiv.org/pdf/1911.09107.pdf)
[16](https://arxiv.org/pdf/2105.09923.pdf)
[17](https://arxiv.org/pdf/2402.16014.pdf)
[18](https://arxiv.org/html/2504.06857v1)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC9489871/)
[20](https://aclanthology.org/2025.acl-long.709.pdf)
[21](https://arxiv.org/html/2502.13472v1)
[22](https://www.isca-archive.org/interspeech_2025/hou25b_interspeech.pdf)
[23](https://chatpaper.com/paper/70632)
[24](https://www.emergentmind.com/topics/full-duplex-spoken-dialogue-model)
[25](https://chatpaper.com/paper/176105)
[26](https://arxiv.org/html/2410.17799v2)
[27](https://www.isca-archive.org/interspeech_2025/hu25f_interspeech.pdf)
[28](https://www.isca-archive.org/interspeech_2025/peng25b_interspeech.pdf)
[29](https://arxiv.org/html/2410.17799v1)
[30](https://www.themoonlight.io/en/review/omniflatten-an-end-to-end-gpt-model-for-seamless-voice-conversation)
[31](https://powerdrill.ai/discover/discover-OmniFlatten-An-End-to-end-cm2nsfxirwip801dfnjb23vf1)
