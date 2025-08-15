# A Survey of Resource-Efficient LLM and Multimodal Foundation Models

## 1. 핵심 주장 및 주요 기여  
이 설문조사는 대규모 언어·비전·멀티모달 파운데이션 모델이 갖는 **과도한 자원 소모 문제**를 다루고, 이를 해결하기 위한  
- 모델 아키텍처 최적화  
- 학습·추론 알고리즘 개선  
- 시스템 설계 혁신  
등 세 가지 차원에서 **종합적·체계적 분류**와 **최신 연구 동향 정리**를 제공한다.  
주요 기여:  
- 자원 소모 분석(연산량·메모리·전력)  
- 효율적 어텐션·MoE·조기 종료 등 구조별 기법 망라  
- PEFT·양자화·지식증류·저순위 분해 등 알고리즘별 기법 정리  
- 분산·연합·엣지·클라우드 서버 시스템 최적화 기법 비교

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계  
### 2.1 해결하고자 하는 문제  
- **연산·메모리·전력 비용 폭증**: LLaMA-70B 사전학습 1.7M GPU시간, 2.5×10¹²J 소비  
- **긴 문맥 처리 비효율**: 어텐션의 O(T²D) 복잡도  
- **추론 지연**: 자주 반복되는 KV 캐시 재계산  
- **분산·엣지 환경 부적합**: 높은 대역폭·메모리 요구

### 2.2 제안 기법  
1) **효율적 어텐션**  
   - 희소 어텐션: Longformer, BigBird  
   - 근사 어텐션: Linformer (K, V 차원에 투영; O(Td²))  
   - 무어텐션 대체: SSM, Hyena (선형·서브쿼드 메모리)  
2) **동적 신경망**  
   - Mixture-of-Experts (MoE): Switch, GLaM (희소 활성화)  
   - Early-Exit: DeeBERT, PABEE (신뢰도 기반 중간층 분기)  
3) **모델 압축**  
   - PEFT: LoRA, Adapter, Prompt–Prefix 튜닝  
   - 양자화: GPTQ (3–4비트, 2차원 재구성), SmoothQuant (채널별 스케일)  
   - 지식증류: Black-box CoT-KD, White-box MiniLLM  
   - 저순위 분해: LoRD (W≈UV)  
4) **시스템 설계**  
   - 분산 학습: ZeRO-Offload, PipeFisher, HetPipe  
   - 연합 학습: FedLLM, FedPrompt, FwdLLM (0차 최적화)  
   - 서버 추론: FlexGen, vLLM (PagedAttention), FlashDecoding  
   - 엣지 추론: LLMCad (Speculative Decoding), PowerInfer (핫 뉴런 캐싱)

### 2.3 수식 예시  
Transformer 어텐션 연산:  

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\bigl(\tfrac{QK^T}{\sqrt{d_k}}\bigr)V
$$  

Linformer 근사:  

$$
K' = EK,\quad V' = FV,\quad \mathrm{Attention}\approx \mathrm{softmax}\bigl(\tfrac{QK'^T}{\sqrt{d_k}}\bigr)V'
$$  

LoRA 저순위 업데이트:  

$$
W_{\text{new}} = W_0 + A\,B,\quad A\in\mathbb{R}^{d\times r},\,B\in\mathbb{R}^{r\times d},\,r\ll d
$$

### 2.4 성능 향상 및 한계  
- O(10–100×) 연산·메모리 절감, 2–5% 이내 성능 저하  
- MoE 학습 비용 ↓, 모델 용량 ↑  
- 양자화·지식증류 후 추론 속도·메모리 요구 ↓  
- 한계:  
  - PEFT/양자화는 초대형 모델(100B+) 일반화 효과 한계  
  - 구조·알고리즘 결합 시 상호작용 복잡성  
  - 시스템 최적화, 이기종 환경 및 네트워크 변동성 제약

## 3. 일반화 성능 향상 관점  
- **희소 전문가(MoE)**: 다양한 경로로 학습된 전문가 결합 시 추가 데이터에도 강건  
- **조기 종료(Early-Exit)**: 예측 신뢰도 기반 분기 정책이 불확실한 장문장 일반화 개선  
- **프롬프트 소량 튜닝(PEFT)**: Frozen 백본에 Task별 소량 파라미터 추가로 과적합 억제  
- **양자화+지식증류 결합**: 고정밀 teacher 지식 전수, 저정밀 student 일반화 유지  
- **유동 어텐션(SSM/Hyena)**: 위치 민감도↓, 다양한 길이 문맥에 동시 적응

## 4. 향후 연구 영향 및 고려 사항  
- **클라우드↔엣지 협업**: 엣지 기기 가속·프라이버시 보호를 위한 경량·적응형 분산 추론  
- **스파스 활성화 활용**: Runtime activation sparsity 탐색해 추가 일반화·효율성 확보  
- **에이전트 시스템 관점**: 여러 FM 협업 워크플로우 최적 스케줄링·자원 배분 알고리즘  
- **프라이버시·안전성**: DP·HE·FL급 기법 호환 PEFT 기법으로 대규모 개인정보 학습  
- **스케일링 법칙 이론화**: 왜 대규모 모델이 작은 모델보다 일반화 잘하는지 이론 분석, 최적 구조 탐색

—  
이 설문조사는 자원 절감과 성능 간 균형을 체계적·종합적으로 제시하여, 향후 대규모 파운데이션 모델의 지속 가능하고 폭넓은 적용 연구에 핵심 지침을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cda0acb0-5ce5-4423-8db3-d99dd5f709fa/2401.08092v2.pdf

# 2 FOUNDATION MODEL OVERVIEW 

## 2.1 Language Foundation Models  
언어 분야의 파운데이션 모델은 주로 텍스트(및 음성)를 처리하며, 모델 아키텍처와 주요 사례를 나눠 살펴봅니다.  

  1. 모델 아키텍처 (§2.1.1)  
     - **토크나이저 + 임베딩**: 입력 문장을 토큰화한 뒤, 각 토큰을 벡터로 변환.  
     - **어텐션(Self-Attention)**: 토큰 간 상호작용을 계산. $$식 (1)$$  
     - **멀티헤드 어텐션**: 여러 시각(서브스페이스)에서 어텐션 수행  
     - **인코더-디코더 구조**  
       -  인코더: 입력 문장의 관계 파악  
       -  디코더: 생성(아웃풋) 과정에서 이전 토큰을 활용한 자기회귀(autoregressive) 디코딩  
     - **KV 캐시**: 생성 단계에서 이전 어텐션 키·값을 저장해 반복계산 방지  

  2. 대표 모델 및 응용 (§2.1.2)  
     - **인코더 전용**: BERT(마스킹), RoBERTa, DistilBERT, Sentence-BERT 등 → 문장 분류·유사도, 질의응답  
     - **인코더-디코더**: T5, BART → 요약·번역·질의응답을 통일된 텍스트-투-텍스트로  
     - **디코더 전용**: GPT 계열(GPT-1/2/3/4), LLaMA, PaLM 등 → 텍스트 생성·영어 쓰기·대화형 AI  
     - **음성 모델**: wav2vec 2.0, HuBERT, Whisper → 자동음성인식(ASR)  

  3. 비용 분석 (§2.1.3)  
     - **저장용량 비중**: 임베딩(약 25%) vs. 어텐션·FFN  
     - **계산량(Flops)**: FFN(두 개의 완전연결층)이 가장 무겁고, 어텐션은 시퀀스 길이 제곱에 비례  
     - **시퀀스 길이 증가 시**: 어텐션 O(T²D), FFN O(TD²) → 토큰 수 늘면 급격히 느려짐  
     - **KV 캐시 메모리**: B×S×D×L×2×4 Bytes (배치, 토큰 길이, 은닉 차원, 레이어 수)  

## 2.2 Vision Foundation Models  
컴퓨터 비전 분야의 파운데이션 모델은 이미지 분류·객체검출·세그먼테이션·생성 등에 활용됩니다.  

  1. 모델 아키텍처 (§2.2.1)  
     - **Vision Transformer(ViT)** 흐름  
       1) 이미지를 고정 크기 패치(예: 16×16)로 분할 → 패치 임베딩(벡터)  
       2) [CLS] 토큰 추가, 위치 임베딩 삽입  
       3) Transformer 인코더 처리 → 분류·검출·세그먼테이션 헤드로 연결  

  2. 대표 모델 및 응용 (§2.2.2)  
     - **인코더 전용**: ViT, DeiT(지식증류), BEiT, MAE(패치 복원), DINOv2(1B 파라미터), Swin(Shifted Window)  
     - **인코더-디코더**: DETR(엔드투엔드 검출·세그멘테이션), SegFormer(경량 MLP 디코더), LVM  
     - **검출·세그먼테이션**: 객체검출(COCO), 분할(ADE20K)에서 SOTA 성능  

  3. 비용 분석 (§2.2.3)  
     - ViT는 BERT와 유사한 구조·비용  
     - 고정 크기의 토큰 수 → 어텐션·FFN 대다수 계산량 차지 (자세한 수식은 §2.1 참조)  

## 2.3 Multimodal Foundation Models  
멀티모달 모델은 텍스트·이미지·오디오·센서 데이터를 융합·생성하며, 두 가지 큰 축으로 구분됩니다.  

  1. 주요 아키텍처 (§2.3.1)  
     - **Multi-Encoder**: 각 모달리티별(텍스트·이미지·오디오·IMU) 인코더 → 공통 잠재 공간 정렬(대조학습)  
     - **Fusion Decoder**: 정렬된 표현을 LLM(디코더-only) 혹은 디퓨전 모듈에 입력  
     - **Diffusion**: VAE(인코더/디코더) + U-Net(노이즈 제거)으로 텍스트→이미지 생성  

  2. 대표 모델 및 응용 (§2.3.2)  
     - **정렬 모델**: CLIP, ALBEF, ALIGN → 이미지·텍스트 인식·검색  
     - **텍스트 생성**: Flamingo, LLaVA, GPT-4V → 시각 질문응답·비쥬얼 챗  
     - **이미지 생성**: DALL-E, Stable Diffusion, ERNIE-ViLG, Consistency Models → 텍스트→이미지, 인페인팅, 에디팅  
     - **Any-to-Any**: CoDi, NExT-GPT, MiniGPT-4 → 다양한 입력·출력 모달리티 혼합  

  3. 비용 분석 (§2.3.3)  
     - **Multi-Encoder**: 평균 0.27B 파라미터, 1.1 GB 메모리, 65.9 GFLOPs (이미지 인코더가 최고)  
     - **LLM 모듈**: 예) Vicuna-7B → 14 GB, 312 GFLOPs  
     - **Diffusion 모듈**(Stable Diffusion 2.1):  
       -  U-Net: 865 M 파라미터, 759 GFLOPs  
       -  VAE: 83 M 파라미터, 4 TFlops  
       -  CLIP 텍스트 인코더: 289 M 파라미터, 토큰당 289 MFLOPs  

---  
위와 같이, 2장에서는 파운데이션 모델의 구조(언어·비전·멀티모달), 대표 사례, 자원(메모리·연산) 특성을 망라하여 **파운데이션 모델의 개념과 현황**을 폭넓게 소개하고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cda0acb0-5ce5-4423-8db3-d99dd5f709fa/2401.08092v2.pdf

# 3 Resource-Efficient Architectures

Section 3에서는 대형 파운데이션 모델의 연산·메모리 비용을 획기적으로 줄이기 위한 **모델 구조(architecture) 관점**의 최적화 기법들을 4가지 범주로 구분해 다룹니다. 각 범주는 핵심 블록(Attention, FFN, Diffusion, ViT)과 모델의 동적 실행 패턴을 토대로 분류되었습니다.

***

## 3.1 Efficient Attention (효율적 어텐션)

**문제 의식**: Transformer의 self‐attention은 길이 T 시퀀스에 대해 O(T²·D) 연산을 수행하므로, 긴 입력의 학습·추론 비용이 급격히 증가합니다.

세부 기법  
1) **Sparse Attention (희소 어텐션, §3.1.1)**  
   - 입력 토큰 간 완전 연결 대신, 국소 창(windowed) + 소수의 전역 토큰(global)을 엮어 어텐션 행렬을 선형 복잡도로 근사  
   - Longformer, BigBird: sliding window + “CLS” 토큰 글로벌  
   - ETC: 토큰 위치에 따른 희소성, 헤드별 stride  
   - ALBERT: 레이어 간 파라미터 공유  
2) **Approximate Attention (근사 어텐션, §3.1.2)**  
   - Keys/Values 차원 축소(Linear projection)로 어텐션 행렬을 저순위(low‐rank) 근사 (Linformer)  
   - 해싱(Reformer), 커널 기법(Performer), 다항 스케치(PolySketchFormer) 등을 이용해 softmax 연산 재구성  
   - Deformable Attention: 정규화된 어텐션 위치를 데이터에 따라 학습  
3) **Attention-Free Approaches (어텐션 대체, §3.1.3)**  
   - State-Space Models (SSM), Hyena(장기 의존성용 긴 컨볼루션), AFT(학습 가능한 게이팅), RWKV(RNN+병렬화)  
   - RetNet: 유지(retention) 메커니즘으로, key-value 캐시 없이 선형 복잡도 달성  

> 표 1: 주요 어텐션 기법 시간·공간 복잡도 비교  
> Transformer: O(T²D) 시간 / O(T²+TD) 메모리  
> Reformer: O(T log T · D) / O(T log T+TD)  
> RetNet, RWKV: O(TD) / O(D)  

***

## 3.2 Dynamic Neural Network (동적 신경망)

모델 실행 시점에 **계산량을 동적으로 조절**하여, 불필요한 연산을 줄입니다.

1) **Mixture-of-Experts (MoE, §3.2.1)**  
   - FFN 내에서 여러 “Expert” 중 일부(소수)만 활성화  
   - Switch Transformer, GLaM, V-MoE, Mistral 등: 1 조 파라미터 이상에서도 연산량·통신량 절감  
   - “Sparse Upcycling”: 기존 dense 모델에서 희소 MoE 초기화 후 업사이클링  
2) **Early-Exiting (조기 종료, §3.2.2)**  
   - 각 레이어마다 **중간 분기(exit) 헤드** 추가, 예측 신뢰도에 따라 중간에 출력을 결정  
   - DeeBERT, PABEE, SkipDecode: 배치 내 토큰별·인스턴스별 exit 달리해 계산량 감소  

> 그림 10: (a) 기존 Transformer  
> (b) MoE 버전: 라우터+여러 Expert(FFN)  
> (c) Early-Exiting: 레이어별 분기  

***

## 3.3 Diffusion-Specific Optimization (확산 모델 최적화)

이미지 생성용 Diffusion Models의 **반복적 노이즈 제거** 비용을 줄이는 방법들.

1) **Efficient Sampling (효율적 샘플링, §3.3.1)**  
   - DDIM, PNDM, DPM-Solver: ODE 기반 고차 솔버로 단계 수 10~20으로 감소  
   - Speculative Caching, ReDi: 이전 계산 궤적 재사용  
2) **Diffusion in Latent Space (잠재 공간 확산, §3.3.2)**  
   - VAE로 이미지→잠재(z) 변환 후 Diffusion 수행 (Stable Diffusion)  
   - LDM, SALAD: 고해상도 생성 시 픽셀 공간 대비 메모리·연산량 대폭 절감  
3) **Architecture Variants (아키텍처 변형, §3.3.3)**  
   - SnapFusion: 모바일용 경량 U-Net + step‐distillation으로 512×512 이미지 2s 내 생성  
   - ERNIE-ViLG: MoDE(denoising experts) 통합, 24B 파라미터로 zero-shot FID 6.75  

***

## 3.4 ViT-Specific Optimizations (ViT 특화 최적화)

Vision Transformer의 입력·연산 패턴에 최적화된 경량화 기법.

- **LeViT**: 초기 임베딩용 컨볼루션 강화, 피라미드 구조, per‐head attention bias  
- **PoolFormer**: attention→pooling으로 대체해 82.1% ImageNet Top-1 달성  
- **MobileViT**: Conv+Transformer 블록 조합, 모바일 하드웨어 친화적  
- **EfficientFormer**: CNN+Transformer 하이브리드, iPhone12에서 1.6 ms 추론  
- **EfficientViT**: Linear attention으로 softmax 비용 절감, super‐resolution서 6.4× 가속  

> 그림 11: ViT 변형 모델 구조 요약  

***

이처럼 Section 3에서는 “어텐션 비용 절감→동적 실행→Diffusion→Vision”이라는 네 개 레이어별·모델별 범주로 나누어, 각각의 **핵심 알고리즘과 구조 혁신**을 구체적으로 제시합니다. 이를 통해 대형 파운데이션 모델의 확장성은 유지하면서도 실전 배포 가능한 수준으로 연산·메모리 효율을 비약적으로 개선하는 연구 동향을 한눈에 파악할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cda0acb0-5ce5-4423-8db3-d99dd5f709fa/2401.08092v2.pdf

# 4. Resource-Efficient Algorithms

대형 파운데이션 모델의 학습 및 추론 과정에서 계산량과 메모리 비용을 줄이기 위한 알고리즘들은 크게 네 가지 카테고리에 속합니다. 이 장에서는 파운데이션 모델 수명 주기를 따라  
1) 사전 학습(Pre-training),  
2) 파인튜닝(Fine-tuning),  
3) 추론(Inference),  
4) 모델 압축(Model Compression)  
단계별로, 각각의 기술을 자세하고 이해하기 쉽게 설명합니다.

***

## 4.1 사전 학습 알고리즘(Pre-training Algorithms)

파운데이션 모델의 사전 학습은 막대한 연산량과 데이터 요구량을 수반하므로, 효율적 자원 활용을 위한 네 가지 핵심 전략이 있습니다.

### 4.1.1 학습 데이터 절감(Training Data Reduction)  
- **중복 제거(De-duplication)**  
  - 거대 말뭉치 내의 유사 또는 중복 문장, 반복 서브스트링을 제거하여 토큰 수를 줄임.  
- **이미지 패치 선택**  
  - 비전·멀티모달 모델에서 불필요한 이미지 패치를 동적으로 건너뛰거나 마스킹 비율을 조정해 연산 절감.  
- **패치 조합·압축**  
  - MixMAE, COPA, PatchDropout 기법처럼 [MASK] 대신 실제 패치를 섞거나 일부 패치를 드롭아웃하여 입력 길이 단축.

### 4.1.2 신경망 구조 탐색(Neural Architecture Search)  
- **제로샷 NAS(Zeroshot NAS)**  
  - 모델 훈련 없이 간단한 지표(예: 기울기 분산, 구조적 특징)만으로 아키텍처 우수도를 평가.  
- **동적 자원 할당(PASHA)**  
  - 검색 초기엔 자원 적게, 유망 후보에 점진적 자원 투입.  
- **순위 기반 탐색(RankNAS)**  
  - 아키텍처 성능 예측을 순위 매김 문제로 변환하여 효율성 개선.

### 4.1.3 점진적 학습(Progressive Learning)  
- **모델 성장(Progressive Stacking)**  
  - 작은 모델을 먼저 학습한 뒤, 레이어를 늘려가며 큰 모델로 확장.  
- **단계별 훈련(Staged Training)**  
  - 폭(width), 깊이(depth), 입력 길이를 단계별로 확장하면서 학습 상태(모델·옵티마이저) 전이.

### 4.1.4 혼합 정밀도 훈련(Mixed Precision Training)  
- **FP16/FP32 혼합**  
  - 가중치·활성화를 절반 정밀도로 처리해 메모리 절반, 대역폭 절감.  
- **활성화 압축(Mesa)**  
  - 중요도에 따라 활성화 텐서를 동적으로 압축해 추가 메모리 절감.

***

## 4.2 파인튜닝 알고리즘(Fine-tuning Algorithms)

사전 학습된 대형 모델을 특정 업무에 맞추어 조정할 때, 전체 파라미터를 갱신하지 않고도 높은 성능을 내도록 하기 위한 세 가지 기법을 다룹니다.

### 4.2.1 추가적 파라미터 삽입(Additive Tuning)  
- **어댑터(Adapter) 튜닝**  
  - 기존 레이어 사이에 작은 MLP 블록(어댑터)을 삽입하고, 이 모듈만 학습. LoRA, Residual Adapters, AdaMix 등.  
- **프롬프트(Prompt) 튜닝**  
  - 입력 토큰 앞에 학습 가능한 “소프트 프롬프트”를 결합해 모델 입력만 수정. PromptTuning, ATTEMPT 등.  
- **프리픽스(Prefix) 튜닝**  
  - 각 레이어 키·쿼리·밸류에 학습 가능한 프리픽스 벡터를 추가. Prefix-tuning, DOPA, UAPT 등.

### 4.2.2 선택적 파라미터 갱신(Selective Tuning)  
- **계층 동결(Layer Freezing)**  
  - 민감도가 높은 일부 계층만 동결 해제하고 학습. SAM, SmartFRZ, GreenTrainer 등.  
- **중요 파라미터만 갱신**  
  - 그래디언트 중요도나 샘플 반응을 기준으로 일부 파라미터만 선택적 업데이트.

### 4.2.3 재매개변수화 튜닝(Re-parameter Tuning)  
- **저순위 업데이트(LoRA 계열)**  
  - 가중치를 두 개의 저순위 행렬 곱(A·B)으로 근사, A·B만 학습. LoRA, QLoRA, Delta-LoRA, PiSSA, DoRA 등.  
- **학습률·초기화 최적화**  
  - A·B의 학습률·초기화 방식 조정으로 동시 수렴 가속.

***

## 4.3 추론 알고리즘(Inference Algorithms)

오토리그레시브(autoregressive) 구조의 속도·메모리 병목을 완화하기 위한 비(非)순차적·토큰·케시 최적화 기법들입니다.

### 4.3.1 기회주의적 디코딩(Opportunistic Decoding)  
- **추측 디코딩(Speculative Decoding)**  
  - 경량 예측 모델로 토큰 초안 생성 후, 본 모델로 동시 검증. Yaniv Leviathan et al., SpecTr.  
- **프롬프트 조회(Look-up Decoding)**  
  - 빈번한 문장 패턴을 사전 캐시하여 재사용. Prompt Cache, LLMA.

### 4.3.2 입력 필터링·압축(Input Filtering & Compression)  
- **프롬프트 압축(Prompt Compression)**  
  - 키워드 추출·요약 벡터화(LLMLingua, LLMZip)로 입력 길이를 줄임.  
- **토큰 제거(Token Pruning)**  
  - 중요도 기반으로 덜 중요한 토큰 실시간 삭제. PoWER-BERT, DynamicViT, AdaViT, Evo-ViT.

### 4.3.3 키·값 캐시 최적화(Key-Value Cache)  
- **양자화(Quantization)**  
  - KV 캐시를 저정밀 축소. LLM.int8(), SmoothQuant+, ATOM.  
- **희소화(Sparse Eviction)**  
  - 중요도 낮은 캐시 토큰 제거. H2O, Dynamic Context Pruning, Scissorhands.  
- **페이지 관리(Paged Attention)**  
  - vLLM의 블록 단위 온디맨드 메모리 할당.

### 4.3.4 긴 맥락 지원(Long Context)  
- **재귀 구조(Recurrent)**  
  - Transformer-XL, Block-Recurrent Transformer: 고정 메모리 크기로 무한 맥락 처리.  
- **주의 메커니즘 최적화**  
  - 희소·확산·스트리밍 어텐션(PCW, LM-Infinite, StreamingLLM, LongNet, SLED).

***

## 4.4 모델 압축(Model Compression)

모델 크기·연산량 축소를 목표로 네 가지 기법을 소개합니다.

### 4.4.1 가지치기(Pruning)  
- **구조적(Structured)**  
  - 레이어, 채널, 헤드 단위 제거. LLM-Pruner, LoRAPrune, AdaPrune.  
- **비구조적(Unstructured)**  
  - 개별 가중치 제거. SparseGPT, Wanda.  
- **문맥 기반(Contextual)**  
  - 동적 활성화 예측 후 필요한 부분만 연산. Deja Vu, PowerInfer.

### 4.4.2 지식 증류(Knowledge Distillation)  
- **블랙박스(BB-KD)**  
  - API 호출로 만든 대량 프롬프트-응답을 교사로 활용. Meta-ICL, SOCRATIC CoT, LaMini-LM.  
- **화이트박스(WB-KD)**  
  - 내부 히든스테이트 활용. MiniLLM, GKD, TED, MixKD.

### 4.4.3 양자화(Quantization)  
- **사후 양자화(PTQ)**  
  - 학습 없이 정밀도 축소. LLM.int8(), GPTQ, SpQR, AWQ, ZeroQuant, LLM-FP4.  
- **훈련 인식 양자화(QAT)**  
  - 저정밀 훈련으로 정밀도 손실 보정. LLM-QAT, QuantGPT, EfficientQAT.

### 4.4.4 저순위 분해(Low-Rank Decomposition)  
- **가중치 근사**  
  - W ≈ U·V 형태로 분해해 파라미터·연산량 절감. LoRD, TensorGPT, LoSparse, ViTALiTy.

***

이상으로, 파운데이션 모델의 사전 학습부터 추론, 파인튜닝, 압축에 이르는 네 단계별 자원 효율화 알고리즘을 정리했습니다. 각 기법은 모델 규모와 목적에 따라 선택·조합하여 실제 시스템에 적용할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cda0acb0-5ce5-4423-8db3-d99dd5f709fa/2401.08092v2.pdf
